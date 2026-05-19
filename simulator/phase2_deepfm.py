"""
phase2_deepfm.py  (개선판 v7 — cross/context feature 보강)
─────────────────────────────────────────
변경 요지 (v7):
  ✅ Cross feature 추가: user_pid_prev (user-product 과거 카운트 binned)
                       cat3_match, tier_match (user-item 정렬 신호)
  ✅ Context feature 추가: hour_bin, dow_weekend
  ✅ User feature 추가: user_activity (low/mid/high)
  → candidate 300개 사이에서 ranking 신호 비약적 강화

기존:
  ✅ 입력 피처: User(persona) + Item(category, price_tier) + Cross + Context
  ✅ AUC ≥ 0.70 을 위한 hard negative mining
       - positive: cart/purchase
       - negative: 같은 유저의 bounced view
       - 추가 random negative 1:4 (project.md 명세)
  ✅ valid_logs.csv 로 검증 (시간 기반 누수 방지)
"""

import os
import json
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 1. 데이터 로드 및 라벨링 전략
# ──────────────────────────────────────────────
print("데이터 로딩 중...")
df_users  = pd.read_csv("data/users.csv")
df_prods  = pd.read_csv("data/products.csv")
df_train  = pd.read_csv("data/train_logs.csv")
df_valid  = pd.read_csv("data/valid_logs.csv") if os.path.exists("data/valid_logs.csv") else None

df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
if df_valid is not None:
    df_valid["timestamp"] = pd.to_datetime(df_valid["timestamp"])


# ──────────────────────────────────────────────
# 1-B. Cross/Context feature lookup 사전 계산
#      ★ 누수 방지: train_logs 만 사용
# ──────────────────────────────────────────────
print("Cross/Context feature lookup 계산 중...")

# user-product 과거 카운트 (전체 이벤트 기준)
pair_counts = (
    df_train.groupby(["user_id", "product_id"]).size().to_dict()
)
print(f"  user-product 페어 카운트: {len(pair_counts):,}")

# v7a: product 전체 등장 횟수 (인기도)
pid_global_count = df_train.groupby("product_id").size().to_dict()

# v7a: user-cat3 view 카운트 (user 의 cat3 관심 강도)
df_train_with_cat3 = df_train.merge(df_prods[["product_id", "category_L3"]], on="product_id")
user_cat3_count = df_train_with_cat3.groupby(["user_id", "category_L3"]).size().to_dict()
print(f"  pid_global: {len(pid_global_count):,}, user_cat3: {len(user_cat3_count):,}")

# user 최다 cat3 / 최다 tier (cart+purchase 기준)
df_train_pos = df_train[df_train["event_type"].isin(["cart", "purchase"])]
df_train_pos = df_train_pos.merge(df_prods[["product_id", "category_L3", "price_tier"]], on="product_id")

user_top_cat3 = (
    df_train_pos.groupby("user_id")["category_L3"]
    .agg(lambda s: s.value_counts().index[0] if len(s) > 0 else "unknown")
    .to_dict()
)
user_top_tier = (
    df_train_pos.groupby("user_id")["price_tier"]
    .agg(lambda s: s.value_counts().index[0] if len(s) > 0 else "medium")
    .to_dict()
)

# user activity level (low/mid/high)
user_event_counts = df_train.groupby("user_id").size()
q33, q67 = np.quantile(user_event_counts.values, [0.33, 0.67])
def activity_bin(c):
    if c < q33: return 0
    if c < q67: return 1
    return 2
user_activity_map = {uid: activity_bin(c) for uid, c in user_event_counts.items()}
print(f"  user_activity quantile: q33={q33:.0f}, q67={q67:.0f}")

# product → cat3/tier 빠른 lookup
prod_cat3_map = df_prods.set_index("product_id")["category_L3"].to_dict()
prod_tier_map = df_prods.set_index("product_id")["price_tier"].to_dict()


def pid_prev_bin(c: int) -> int:
    """user-product 카운트 binning: 0 / 1 / 2 / 3+"""
    if c == 0: return 0
    if c == 1: return 1
    if c == 2: return 2
    return 3


def hour_bin(h: int) -> int:
    """0~5 night / 6~11 morning / 12~17 afternoon / 18~23 evening"""
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3


def pop_bin(c: int) -> int:
    """product 인기도 binning: <5 / <20 / <100 / <500 / 500+"""
    if c < 5:   return 0
    if c < 20:  return 1
    if c < 100: return 2
    if c < 500: return 3
    return 4


def cat3_view_bin(c: int) -> int:
    """user-cat3 view 카운트 binning: 0 / 1~4 / 5~19 / 20~99 / 100+"""
    if c == 0:  return 0
    if c < 5:   return 1
    if c < 20:  return 2
    if c < 100: return 3
    return 4


def add_cross_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """user_pid_prev, user_activity, cat3_match, tier_match, hour_bin, dow_weekend,
       + v7a: pid_popularity, user_cat3_view 추가."""
    df = df.copy()
    # user_pid_prev
    pairs = list(zip(df["user_id"].values, df["product_id"].values))
    df["user_pid_prev"] = [pid_prev_bin(pair_counts.get(k, 0)) for k in pairs]
    # user_activity
    df["user_activity"] = df["user_id"].map(user_activity_map).fillna(0).astype(int)
    # cat3_match (현재 row product cat3 == user 의 top cat3)
    if "category_L3" not in df.columns:
        df["category_L3"] = df["product_id"].map(prod_cat3_map).fillna("unknown")
    else:
        df["category_L3"] = df["category_L3"].fillna("unknown")
    df["cat3_match"] = (df["category_L3"].astype(str) == df["user_id"].map(user_top_cat3).fillna("unknown").astype(str)).astype(int)
    # tier_match
    if "price_tier" not in df.columns:
        df["price_tier"] = df["product_id"].map(prod_tier_map).fillna("medium")
    else:
        df["price_tier"] = df["price_tier"].fillna("medium")
    df["tier_match"] = (df["price_tier"].astype(str) == df["user_id"].map(user_top_tier).fillna("medium").astype(str)).astype(int)
    # 시간 feature
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour_bin"]    = ts.dt.hour.map(hour_bin)
        df["dow_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    else:
        df["hour_bin"]    = 1
        df["dow_weekend"] = 0
    # v7a: pid_popularity (product 인기도 binned)
    df["pid_popularity"] = df["product_id"].map(pid_global_count).fillna(0).astype(int).map(pop_bin)
    # v7a: user_cat3_view (user 의 그 cat3 안에서 본 횟수 binned)
    uc_pairs = list(zip(df["user_id"].values, df["category_L3"].astype(str).values))
    df["user_cat3_view"] = [cat3_view_bin(user_cat3_count.get(k, 0)) for k in uc_pairs]
    return df


# ──────────────────────────────────────────────
# 2. 라벨링 + random negative
# ──────────────────────────────────────────────

def build_supervised(df_logs: pd.DataFrame) -> pd.DataFrame:
    df = df_logs.copy()
    df = df[df["event_type"].isin(["view", "cart", "purchase"])]

    if "is_bounced" in df.columns:
        pos = df[df["event_type"].isin(["cart", "purchase"])].copy()
        pos["label"] = 1.0

        neg = df[(df["event_type"] == "view") & (df["is_bounced"] == 1)].copy()
        neg["label"] = 0.0

        out = pd.concat([pos, neg], ignore_index=True)
    else:
        df["event_label"] = df["event_type"].isin(["cart", "purchase"]).astype(float)
        out = (
            df.groupby(["user_id", "product_id"], as_index=False)
              .agg({"event_label": "max", "timestamp": "max"})
              .rename(columns={"event_label": "label"})
        )

    # 같은 (user, product) 쌍은 라벨 최대 + 최신 timestamp 로 dedupe
    out = (
        out.groupby(["user_id", "product_id"], as_index=False)
           .agg({"label": "max", "timestamp": "max"})
    )
    return out


def add_random_negatives(df_lbl: pd.DataFrame, all_product_ids: np.ndarray,
                         k: int = 4, seed: int = 42) -> pd.DataFrame:
    """positive 1개당 random product k개를 negative(label=0)로 추가 (out-of-pool 신호)."""
    pos = df_lbl[df_lbl["label"] == 1.0]
    if len(pos) == 0:
        return df_lbl
    rng = np.random.default_rng(seed)
    n_pos = len(pos)
    user_ids   = np.repeat(pos["user_id"].values,   k)
    timestamps = np.repeat(pos["timestamp"].values, k)
    sampled = rng.choice(all_product_ids, size=n_pos * k, replace=True)

    df_neg = pd.DataFrame({
        "user_id":    user_ids,
        "product_id": sampled,
        "label":      0.0,
        "timestamp":  timestamps,
    })

    pos_pairs = pd.MultiIndex.from_frame(pos[["user_id", "product_id"]])
    neg_pairs = pd.MultiIndex.from_frame(df_neg[["user_id", "product_id"]])
    df_neg = df_neg.loc[~neg_pairs.isin(pos_pairs)].reset_index(drop=True)

    return pd.concat([df_lbl, df_neg], ignore_index=True)


df_train_lbl = build_supervised(df_train)
df_valid_lbl = build_supervised(df_valid) if df_valid is not None else None

# random negative 1:4 추가 (project.md 명세)
all_pids = df_prods["product_id"].values
df_train_lbl = add_random_negatives(df_train_lbl, all_pids, k=4, seed=42)
if df_valid_lbl is not None:
    df_valid_lbl = add_random_negatives(df_valid_lbl, all_pids, k=4, seed=43)

# 메타데이터 조인 + cross/context feature 부착
df_train_full = df_train_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")
df_train_full = add_cross_context_features(df_train_full)

if df_valid_lbl is not None:
    df_valid_full = df_valid_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")
    df_valid_full = add_cross_context_features(df_valid_full)
else:
    df_valid_full = None

print(f"  Train pos:{int(df_train_full['label'].sum()):,} / neg:{int((1-df_train_full['label']).sum()):,}")
if df_valid_full is not None:
    print(f"  Valid pos:{int(df_valid_full['label'].sum()):,} / neg:{int((1-df_valid_full['label']).sum()):,}")

# ──────────────────────────────────────────────
# 3. 인코더 (train 기준 fit) — 모든 sparse feature
# ──────────────────────────────────────────────
# 기존 sparse + 새 cross/context (모두 categorical/binned 라 sparse 로 취급)
SPARSE_FEATURES = [
    "user_id",
    "product_id",
    "persona",
    "category_L1",
    "category_L2",
    "category_L3",
    "price_tier",
    # ── cross/context (v7) ──
    "user_pid_prev",   # 0/1/2/3
    "user_activity",   # 0/1/2
    "cat3_match",      # 0/1
    "tier_match",      # 0/1
    "hour_bin",        # 0/1/2/3
    "dow_weekend",     # 0/1
    # ── 추가 cross (v7a) ──
    "pid_popularity",  # 0..4 (5 bins)
    "user_cat3_view",  # 0..4 (5 bins)
]

encoders = {}
for feat in SPARSE_FEATURES:
    le = LabelEncoder()
    # product/category/tier 는 df_prods 전체 vocab 로 fit (평가 시 random neg vocab 일치)
    if feat == "product_id":
        le.fit(df_prods["product_id"].astype(str).values)
    elif feat in ["category_L1", "category_L2", "category_L3", "price_tier"]:
        le.fit(df_prods[feat].astype(str).values)
    # bin-encoded cross/context 는 명시적 vocab 로 fit (phase4 와 일치)
    elif feat == "user_pid_prev":
        le.fit(np.array(["0", "1", "2", "3"]))
    elif feat == "user_activity":
        le.fit(np.array(["0", "1", "2"]))
    elif feat in ["cat3_match", "tier_match", "dow_weekend"]:
        le.fit(np.array(["0", "1"]))
    elif feat == "hour_bin":
        le.fit(np.array(["0", "1", "2", "3"]))
    elif feat in ["pid_popularity", "user_cat3_view"]:
        le.fit(np.array(["0", "1", "2", "3", "4"]))
    else:  # user_id, persona
        le.fit(df_train_full[feat].astype(str).values)
    df_train_full[feat] = le.transform(df_train_full[feat].astype(str))
    encoders[feat] = le


def encode_with_oov(df: pd.DataFrame) -> np.ndarray:
    """미등록 값은 0 으로 처리 (OOV)."""
    X = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=int)
    for i, feat in enumerate(SPARSE_FEATURES):
        classes = encoders[feat].classes_
        vals = df[feat].astype(str).values
        mask = np.isin(vals, classes)
        X[mask, i] = encoders[feat].transform(vals[mask])
    return X


X_train = df_train_full[SPARSE_FEATURES].values.astype(int)
y_train = df_train_full["label"].values.astype(np.float32)

if df_valid_full is not None and len(df_valid_full) > 0:
    X_valid = encode_with_oov(df_valid_full)
    y_valid = df_valid_full["label"].values.astype(np.float32)
else:
    n_split = int(len(X_train) * 0.9)
    X_valid, y_valid = X_train[n_split:], y_train[n_split:]
    X_train, y_train = X_train[:n_split], y_train[:n_split]

feature_dims = {feat: len(encoders[feat].classes_) for feat in SPARSE_FEATURES}
print(f"  feature_dims: {feature_dims}")


# ──────────────────────────────────────────────
# 4. Dataset
# ──────────────────────────────────────────────
class DeepFMDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.long), torch.tensor(self.y[idx], dtype=torch.float32)


train_loader = DataLoader(DeepFMDataset(X_train, y_train), batch_size=512, shuffle=True)
valid_loader = DataLoader(DeepFMDataset(X_valid, y_valid), batch_size=512, shuffle=False)


# ──────────────────────────────────────────────
# 5. DeepFM
# ──────────────────────────────────────────────
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=24):  # v8c: 16 → 24 (capacity ↑)
        super().__init__()
        self.sparse_features = list(feature_dims.keys())
        self.embeddings = nn.ModuleDict({f: nn.Embedding(d, embedding_dim) for f, d in feature_dims.items()})
        self.linear     = nn.ModuleDict({f: nn.Embedding(d, 1) for f, d in feature_dims.items()})
        self.bias       = nn.Parameter(torch.zeros(1))

        mlp_in = len(feature_dims) * embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),     nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = [self.embeddings[f](x[:, i]) for i, f in enumerate(self.sparse_features)]
        lin = self.bias + sum(self.linear[f](x[:, i]) for i, f in enumerate(self.sparse_features))
        stk = torch.stack(emb, dim=1)
        fm  = 0.5 * torch.sum(stk.sum(1) ** 2 - (stk ** 2).sum(1), dim=1, keepdim=True)
        dnn = self.mlp(torch.cat(emb, dim=1))
        return torch.sigmoid(lin + fm + dnn).squeeze()


model = DeepFM(feature_dims).to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ──────────────────────────────────────────────
# 6. 학습 + valid AUC 모니터링
# ──────────────────────────────────────────────
def evaluate_auc(loader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            preds.extend(model(X_b.to(device)).cpu().numpy())
            labels.extend(y_b.numpy())
    if len(set(labels)) < 2:
        return 0.5
    return roc_auc_score(labels, preds)


EPOCHS = 15  # v8c: 10 → 15 (수렴 안정화)
print(f"DeepFM 학습 시작 (Epochs: {EPOCHS}, Features: {len(SPARSE_FEATURES)})...")
best_auc = 0.0
for ep in range(EPOCHS):
    model.train()
    total = 0.0
    for X_b, y_b in train_loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()

    val_auc = evaluate_auc(valid_loader)
    print(f"  Epoch {ep+1}/{EPOCHS} | Loss: {total/len(train_loader):.4f} | Valid AUC: {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        os.makedirs("data/models", exist_ok=True)
        torch.save(model.state_dict(), "data/models/deepfm.pth")
        print(f"    → 베스트 갱신, 저장")

print(f"\n[DONE] DeepFM 학습 완료! Best Valid AUC = {best_auc:.4f}")
print("   가중치: data/models/deepfm.pth")


# ──────────────────────────────────────────────
# 7. feature lookup 저장 (phase3/phase4 가 동일하게 재구성하기 위함)
# ──────────────────────────────────────────────
os.makedirs("data/models", exist_ok=True)

# user_top_cat3, user_top_tier, user_activity_map 을 CSV 저장
user_lookup = pd.DataFrame({
    "user_id": list(user_event_counts.index),
})
user_lookup["user_activity"] = user_lookup["user_id"].map(user_activity_map).fillna(0).astype(int)
user_lookup["top_cat3"]      = user_lookup["user_id"].map(user_top_cat3).fillna("unknown")
user_lookup["top_tier"]      = user_lookup["user_id"].map(user_top_tier).fillna("medium")
user_lookup.to_csv("data/models/deepfm_user_lookup.csv", index=False)

# pair_counts (user_id, product_id, cnt) → npz 로 저장
pair_keys = list(pair_counts.keys())
pd.DataFrame({
    "user_id":    [k[0] for k in pair_keys],
    "product_id": [k[1] for k in pair_keys],
    "cnt":        [pair_counts[k] for k in pair_keys],
}).to_csv("data/models/deepfm_pair_counts.csv", index=False)

# feature 메타 저장 (phase3/phase4 에서 동일 schema 사용)
with open("data/models/deepfm_feature_schema.json", "w", encoding="utf-8") as f:
    json.dump({
        "sparse_features": SPARSE_FEATURES,
        "feature_dims":    feature_dims,
        "activity_q33":    float(q33),
        "activity_q67":    float(q67),
    }, f, indent=2, ensure_ascii=False)

print("   user_lookup:  data/models/deepfm_user_lookup.csv")
print("   pair_counts:  data/models/deepfm_pair_counts.csv")
print("   schema:       data/models/deepfm_feature_schema.json")
