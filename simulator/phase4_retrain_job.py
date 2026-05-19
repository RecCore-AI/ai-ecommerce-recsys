"""
phase4_retrain_job.py  (CT 재학습 잡 — 15피처 schema 대응판)
─────────────────────────────────────────────────────────
역할 분리:
  - 이 파일은 신규 로그로 DeepFM 을 fine-tuning 만 담당
  - 모든 지표 측정은 phase4_offline_eval.py 가 담당 (단일 진실 원천)

CT 흐름 (ct_pipeline.py 가 호출):
  1. new_train_logs.csv → DeepFM fine-tune
  2. data/models/deepfm.pth 갱신
  3. phase4_offline_eval.py 실행 → metrics.json 갱신
  4. (ct_pipeline.py 가) /api/reload-model 호출

⚠️ 피처 스키마는 deepfm_feature_schema.json 을 단일 진실 공급원으로 로드한다.
   (학습된 deepfm.pth 는 v7+v7a 의 15개 sparse 피처 / embedding_dim 24 —
    구버전 7피처 정의로 load_state_dict 하면 크래시하므로 schema 기반으로 맞춘다.)
"""
import os
import sys
import csv
import json
import datetime
import subprocess
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 [Retrain Job] 시작 (Device: {device})")

DATA_DIR    = "data"
USERS_CSV   = f"{DATA_DIR}/users.csv"
PRODS_CSV   = f"{DATA_DIR}/products.csv"
TRAIN_CSV   = f"{DATA_DIR}/train_logs.csv"
NEW_CSV     = f"{DATA_DIR}/new_train_logs.csv"
MODEL_PATH  = f"{DATA_DIR}/models/deepfm.pth"
SCHEMA_PATH = f"{DATA_DIR}/models/deepfm_feature_schema.json"


# ──────────────────────────────────────────────
# DeepFM (phase2_deepfm.py 와 동일 — embedding_dim 24)
# ──────────────────────────────────────────────
class DeepFM(nn.Module):
    def __init__(self, feature_dims: dict, embedding_dim: int = 24):
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


class LogDS(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X, self.y = X, y

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, i: int):
        return torch.tensor(self.X[i], dtype=torch.long), torch.tensor(self.y[i], dtype=torch.float32)


# ──────────────────────────────────────────────
# binning 함수 (phase2_deepfm.py / phase4_offline_eval.py 와 동일)
# ──────────────────────────────────────────────
def pid_prev_bin(c: int) -> int:
    if c == 0: return 0
    if c == 1: return 1
    if c == 2: return 2
    return 3


def hour_bin(h: int) -> int:
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3


def pop_bin(c: int) -> int:
    if c < 5:   return 0
    if c < 20:  return 1
    if c < 100: return 2
    if c < 500: return 3
    return 4


def cat3_view_bin(c: int) -> int:
    if c == 0:  return 0
    if c < 5:   return 1
    if c < 20:  return 2
    if c < 100: return 3
    return 4


# ──────────────────────────────────────────────
# 1. 피처 스키마 + cross/context lookup (train_logs 기준 — 누수 방지)
# ──────────────────────────────────────────────
print("1️⃣  스키마 / cross-context lookup 복원 중...")
with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
    schema = json.load(f)
SPARSE       = schema["sparse_features"]                       # 15개
feature_dims = {f: schema["feature_dims"][f] for f in SPARSE}
q33, q67     = schema["activity_q33"], schema["activity_q67"]

df_users = pd.read_csv(USERS_CSV)
df_prods = pd.read_csv(PRODS_CSV)
df_train = pd.read_csv(TRAIN_CSV)
df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])

pair_counts      = df_train.groupby(["user_id", "product_id"]).size().to_dict()
pid_global_count = df_train.groupby("product_id").size().to_dict()
df_train_c3      = df_train.merge(df_prods[["product_id", "category_L3"]], on="product_id")
user_cat3_count  = df_train_c3.groupby(["user_id", "category_L3"]).size().to_dict()

df_train_pos = df_train[df_train["event_type"].isin(["cart", "purchase"])].merge(
    df_prods[["product_id", "category_L3", "price_tier"]], on="product_id")
user_top_cat3 = df_train_pos.groupby("user_id")["category_L3"].agg(
    lambda s: s.value_counts().index[0] if len(s) > 0 else "unknown").to_dict()
user_top_tier = df_train_pos.groupby("user_id")["price_tier"].agg(
    lambda s: s.value_counts().index[0] if len(s) > 0 else "medium").to_dict()

user_event_counts = df_train.groupby("user_id").size()
def activity_bin(c: int) -> int:
    if c < q33: return 0
    if c < q67: return 1
    return 2
user_activity_map = {uid: activity_bin(c) for uid, c in user_event_counts.items()}

prod_cat3_map = df_prods.set_index("product_id")["category_L3"].to_dict()
prod_tier_map = df_prods.set_index("product_id")["price_tier"].to_dict()

# 인코더 (phase2_deepfm.py 와 동일 vocab — fine-tune 호환 보장)
encoders = {}
for f in SPARSE:
    le = LabelEncoder()
    if f == "product_id":
        le.fit(df_prods["product_id"].astype(str).values)
    elif f in ["category_L1", "category_L2", "category_L3", "price_tier"]:
        le.fit(df_prods[f].astype(str).values)
    elif f == "user_pid_prev":
        le.fit(np.array(["0", "1", "2", "3"]))
    elif f == "user_activity":
        le.fit(np.array(["0", "1", "2"]))
    elif f in ["cat3_match", "tier_match", "dow_weekend"]:
        le.fit(np.array(["0", "1"]))
    elif f == "hour_bin":
        le.fit(np.array(["0", "1", "2", "3"]))
    elif f in ["pid_popularity", "user_cat3_view"]:
        le.fit(np.array(["0", "1", "2", "3", "4"]))
    elif f == "persona":
        le.fit(df_users["persona"].astype(str).values)
    else:  # user_id
        le.fit(df_users["user_id"].astype(str).values)
    encoders[f] = le
enc_maps = {f: {c: i for i, c in enumerate(encoders[f].classes_)} for f in SPARSE}


def add_cross_context(df: pd.DataFrame) -> pd.DataFrame:
    """phase2_deepfm.py / phase4_offline_eval.py 와 동일한 15피처 cross/context 생성."""
    df = df.copy()
    pairs = list(zip(df["user_id"].values, df["product_id"].values))
    df["user_pid_prev"] = [pid_prev_bin(pair_counts.get(k, 0)) for k in pairs]
    df["user_activity"] = df["user_id"].map(user_activity_map).fillna(0).astype(int)
    if "category_L3" not in df.columns:
        df["category_L3"] = df["product_id"].map(prod_cat3_map).fillna("unknown")
    else:
        df["category_L3"] = df["category_L3"].fillna("unknown")
    df["cat3_match"] = (df["category_L3"].astype(str) ==
                        df["user_id"].map(user_top_cat3).fillna("unknown").astype(str)).astype(int)
    if "price_tier" not in df.columns:
        df["price_tier"] = df["product_id"].map(prod_tier_map).fillna("medium")
    else:
        df["price_tier"] = df["price_tier"].fillna("medium")
    df["tier_match"] = (df["price_tier"].astype(str) ==
                        df["user_id"].map(user_top_tier).fillna("medium").astype(str)).astype(int)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"])
        df["hour_bin"]    = ts.dt.hour.map(hour_bin)
        df["dow_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
    else:
        df["hour_bin"]    = 1
        df["dow_weekend"] = 0
    df["pid_popularity"] = df["product_id"].map(pid_global_count).fillna(0).astype(int).map(pop_bin)
    uc_pairs = list(zip(df["user_id"].values, df["category_L3"].astype(str).values))
    df["user_cat3_view"] = [cat3_view_bin(user_cat3_count.get(k, 0)) for k in uc_pairs]
    return df


def encode(df: pd.DataFrame) -> np.ndarray:
    df = add_cross_context(df)
    X = np.zeros((len(df), len(SPARSE)), dtype=int)
    for i, f in enumerate(SPARSE):
        if f not in df.columns:
            continue
        X[:, i] = df[f].astype(str).map(enc_maps[f]).fillna(0).astype(int).to_numpy()
    return X


# ──────────────────────────────────────────────
# 2. 신규 로그 라벨링 + 인코딩
# ──────────────────────────────────────────────
print("2️⃣  신규 로그 전처리...")
if not os.path.exists(NEW_CSV):
    print(f"❌ {NEW_CSV} 없음 — 재학습할 신규 로그가 없습니다.")
    sys.exit(1)

df_new   = pd.read_csv(NEW_CSV)
new_full = df_new.merge(df_users, on="user_id").merge(df_prods, on="product_id")
new_full = new_full[new_full["event_type"].isin(["view", "cart", "purchase"])]
new_full["label"] = new_full["event_type"].isin(["cart", "purchase"]).astype(float)

if len(new_full) == 0:
    print("❌ 학습 가능한 신규 로그(view/cart/purchase)가 없습니다.")
    sys.exit(1)

X = encode(new_full)
y = new_full["label"].values.astype(np.float32)
print(f"   샘플: {len(y):,}건 (pos={int(y.sum()):,}, neg={int((1-y).sum()):,}) / 피처 {len(SPARSE)}개")

loader = DataLoader(LogDS(X, y), batch_size=512, shuffle=True)


# ──────────────────────────────────────────────
# 3. 기존 가중치 로드 + 파인튜닝
# ──────────────────────────────────────────────
print("3️⃣  파인튜닝...")
model = DeepFM(feature_dims).to(device)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print("   기존 가중치 로드 (15피처 / embedding_dim 24)")

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # 낮은 LR (파인튜닝)

EPOCHS = 5
model.train()
for ep in range(EPOCHS):
    total = 0.0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
        total += loss.item()
    print(f"   Epoch {ep+1}/{EPOCHS}  Loss: {total/len(loader):.4f}")

torch.save(model.state_dict(), MODEL_PATH)

# 모델 버전 보관 (명세 line 233: 재학습 후 모델 버전 관리)
version      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
versions_dir = f"{DATA_DIR}/models/versions"
os.makedirs(versions_dir, exist_ok=True)
versioned_path = f"{versions_dir}/deepfm_v{version}.pth"
torch.save(model.state_dict(), versioned_path)
print(f"💾  저장: {MODEL_PATH}  /  버전 보관: {versioned_path}")


# ──────────────────────────────────────────────
# 4. 평가는 phase4_offline_eval.py 에 위임 (단일 진실 원천)
# ──────────────────────────────────────────────
print("4️⃣  오프라인 평가 실행 중...")
eval_ok = False
try:
    subprocess.run([sys.executable, "phase4_offline_eval.py"], check=True)
    eval_ok = True
    print("✅ Retrain + Evaluation 완료")
except subprocess.CalledProcessError as e:
    print(f"⚠️ 평가 스크립트 실행 실패: {e}")
    print("   metrics.json 이 갱신되지 않았을 수 있습니다.")


# ──────────────────────────────────────────────
# 5. 모델 버전 레지스트리 기록 (버전 · 학습량 · 평가지표)
# ──────────────────────────────────────────────
rec = {}
if eval_ok:
    try:
        with open(f"{DATA_DIR}/metrics.json", "r", encoding="utf-8") as f:
            rec = json.load(f).get("recommend", {})
    except Exception:
        pass
registry = f"{DATA_DIR}/models/model_registry.csv"
is_new = not os.path.exists(registry)
with open(registry, "a", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    if is_new:
        w.writerow(["version", "timestamp", "finetune_samples",
                    "hitrate_50", "ndcg_50", "auc", "coverage"])
    w.writerow([f"v{version}", datetime.datetime.now().isoformat(timespec="seconds"),
                len(y), rec.get("hitrate_50"), rec.get("ndcg_50"),
                rec.get("auc"), rec.get("coverage")])
print(f"📦  모델 버전 등록: v{version}  →  {registry}")
