"""
phase2_two_tower.py  (개선판 v8 — 시퀀스 인코더 추가)
─────────────────────────────────────────────
변경 요지:
  ✅ [v2] User Tower 행동 피처 (cat1/tier 분포 + activity 버킷, cat1/tier 임베딩 SHARED)
  ✅ [v3] 임베딩 차원 비율 재조정 (USER/ITEM=64, SIDE=16)
  ✅ [v3] In-batch softmax + logQ correction
  ✅ [v3] EPOCHS 20, weight_decay, CosineAnnealingLR
  ✅ [v4] 명세 권장 1:4 네거티브 (하드 2 + 랜덤 2)
  ✅ [v8] User Tower 시퀀스 인코더 GRU 추가 (명세 line 151 충족)
       - user 별 train_logs 마지막 SEQ_LEN=10 행동 (view+cart+purchase 전체)
       - nn.GRU(item_dim, seq_hidden) → user MLP 입력에 결합
       - 단기 관심 신호 → Recall@300, HitRate@50, NDCG@50 향상 기대

Item Tower:
  item_idx + category_L1 + category_L2 + category_L3 + price_tier

Loss: sampling-bias-corrected in-batch softmax + explicit (hard+random) negatives
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import faiss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ──────────────────────────────────────────────
# 1. 데이터 로드 & 인코더
# ──────────────────────────────────────────────
print("데이터 로딩 중...")
df_users = pd.read_csv("data/users.csv")
df_prods = pd.read_csv("data/products.csv")
df_all_train = pd.read_csv("data/train_logs.csv")  # v8: 시퀀스 추출용 (전체 이벤트)
df_logs = df_all_train[df_all_train["event_type"].isin(["cart", "purchase"])].copy()

SPARSE_FEATS = ["persona", "category_L1", "category_L2", "category_L3", "price_tier"]

user_encoder = LabelEncoder().fit(df_users["user_id"].values)
prod_encoder = LabelEncoder().fit(df_prods["product_id"].values)

df_users["user_idx"] = user_encoder.transform(df_users["user_id"])
df_prods["prod_idx"] = prod_encoder.transform(df_prods["product_id"])

side_encoders = {}
for feat in SPARSE_FEATS:
    le = LabelEncoder()
    if feat == "persona":
        le.fit(df_users[feat].astype(str).values)
        df_users[f"{feat}_idx"] = le.transform(df_users[feat].astype(str))
    else:
        le.fit(df_prods[feat].astype(str).values)
        df_prods[f"{feat}_idx"] = le.transform(df_prods[feat].astype(str))
    side_encoders[feat] = le

df_logs = df_logs.merge(
    df_users[["user_id", "user_idx", "persona_idx"]], on="user_id", how="inner"
).merge(
    df_prods[[
        "product_id",
        "prod_idx",
        "category_L1_idx",
        "category_L2_idx",
        "category_L3_idx",
        "price_tier_idx"
    ]],
    on="product_id",
    how="inner"
)


num_users    = len(user_encoder.classes_)
num_prods    = len(prod_encoder.classes_)
num_persona  = len(side_encoders["persona"].classes_)
num_cat1 = len(side_encoders["category_L1"].classes_)
num_cat2 = len(side_encoders["category_L2"].classes_)
num_cat3 = len(side_encoders["category_L3"].classes_)
num_tier = len(side_encoders["price_tier"].classes_)
NUM_ACTIVITY = 3  # low / mid / high


print(f"  유저: {num_users:,} / 상품: {num_prods:,} / 페르소나: {num_persona} / 카테고리: {num_cat1}/{num_cat2}/{num_cat3} / 가격대: {num_tier}")
print(f"  학습용 로그: {len(df_logs):,}건")


# ──────────────────────────────────────────────
# 1-B. 유저 행동 피처 집계 (cart+purchase 기준)
#      - 카테고리 L1 분포 (num_users, num_cat1)
#      - 가격대 분포      (num_users, num_tier)
#      - 활동량 버킷      (num_users,)
# ──────────────────────────────────────────────
print("유저 행동 피처 집계 중...")

user_cat1_counts = np.zeros((num_users, num_cat1), dtype=np.float32)
np.add.at(
    user_cat1_counts,
    (df_logs["user_idx"].values, df_logs["category_L1_idx"].values),
    1.0,
)
row_sum = user_cat1_counts.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
user_cat1_dist = (user_cat1_counts / row_sum).astype(np.float32)

user_tier_counts = np.zeros((num_users, num_tier), dtype=np.float32)
np.add.at(
    user_tier_counts,
    (df_logs["user_idx"].values, df_logs["price_tier_idx"].values),
    1.0,
)
row_sum = user_tier_counts.sum(axis=1, keepdims=True)
row_sum[row_sum == 0] = 1.0
user_tier_dist = (user_tier_counts / row_sum).astype(np.float32)

activity_counts = np.zeros(num_users, dtype=np.int64)
np.add.at(activity_counts, df_logs["user_idx"].values, 1)
# 이력 0건은 별도 low 처리, 그 외 quantile bucket
nonzero = activity_counts[activity_counts > 0]
if len(nonzero) > 10:
    q33, q67 = np.quantile(nonzero, [0.33, 0.67])
else:
    q33, q67 = 1, 1
user_activity_idx = np.zeros(num_users, dtype=np.int64)
user_activity_idx[activity_counts >= q33] = 1
user_activity_idx[activity_counts >= q67] = 2

print(f"  cat1_dist: {user_cat1_dist.shape}  tier_dist: {user_tier_dist.shape}  activity 분포: {np.bincount(user_activity_idx, minlength=NUM_ACTIVITY).tolist()}")


# ──────────────────────────────────────────────
# 1-C. 카탈로그 피처 lookup + 카테고리별 상품 풀 (v4: 네거티브 샘플링용)
# ──────────────────────────────────────────────
df_prods_sorted = df_prods.sort_values("prod_idx").reset_index(drop=True)
prod_cat1 = df_prods_sorted["category_L1_idx"].values.astype(np.int64)
prod_cat2 = df_prods_sorted["category_L2_idx"].values.astype(np.int64)
prod_cat3 = df_prods_sorted["category_L3_idx"].values.astype(np.int64)
prod_tier = df_prods_sorted["price_tier_idx"].values.astype(np.int64)

cat1_to_prods = [
    np.where(prod_cat1 == c)[0].astype(np.int64) for c in range(num_cat1)
]
print(f"  cat1 풀 사이즈: {[len(p) for p in cat1_to_prods]}")

HARD_NEG = 2
RAND_NEG = 2
NUM_NEG  = HARD_NEG + RAND_NEG  # 명세 1:4 비율
SEQ_LEN  = 10                   # v8: user 시퀀스 길이


# ──────────────────────────────────────────────
# 1-D. user 별 시퀀스 추출 (v8 — 명세 line 151 "최근 행동 시퀀스")
#      - train_logs 전체 이벤트(view+cart+purchase) 에서 user 별 마지막 SEQ_LEN 행동
#      - timestamp 정렬 후 tail
#      - prod_idx 시퀀스 → GRU 인코더 입력
# ──────────────────────────────────────────────
print(f"user 별 시퀀스 추출 (SEQ_LEN={SEQ_LEN})...")
df_all_train["timestamp"] = pd.to_datetime(df_all_train["timestamp"])
df_all_sorted = (
    df_all_train.sort_values(["user_id", "timestamp"])
    .merge(df_users[["user_id", "user_idx"]], on="user_id")
    .merge(df_prods[["product_id", "prod_idx"]], on="product_id")
)
user_seq_arr = np.zeros((num_users, SEQ_LEN), dtype=np.int64)
# v9: user 별 prod_idx, timestamp 배열도 보관 → sample-time-aware sequence 용
user_pid_dict = {}
user_ts_dict  = {}
for u_idx_val, sub in df_all_sorted.groupby("user_idx"):
    pids_arr = sub["prod_idx"].values.astype(np.int64)
    ts_arr   = sub["timestamp"].values.astype("datetime64[ns]")
    user_pid_dict[int(u_idx_val)] = pids_arr
    user_ts_dict[int(u_idx_val)]  = ts_arr
    # 평가/서빙용 정적 시퀀스 (train_logs 마지막 N → test 시점 직전 = 자연스러운 cutoff)
    tail = pids_arr[-SEQ_LEN:] if len(pids_arr) >= SEQ_LEN else pids_arr
    pids = list(tail)
    if len(pids) < SEQ_LEN:
        pids = [0] * (SEQ_LEN - len(pids)) + pids
    user_seq_arr[int(u_idx_val)] = pids[-SEQ_LEN:]
print(f"  user_seq_arr: {user_seq_arr.shape}, nonzero rows: {(user_seq_arr.sum(axis=1) > 0).sum():,}")
# v9 sample-time-aware sequence 실패로 롤백 → buffer (정적 train 마지막 N) 만 사용


# ──────────────────────────────────────────────
# 2. Dataset (positive + 4 negatives: 2 hard + 2 random)
# ──────────────────────────────────────────────
class PairDataset(Dataset):
    def __init__(self, df, user_cat1_dist, user_tier_dist, user_activity_idx,
                 prod_cat1, prod_cat2, prod_cat3, prod_tier, cat1_to_prods,
                 num_prods, hard_neg=HARD_NEG, rand_neg=RAND_NEG):
        self.user_idx = df["user_idx"].values
        self.persona_idx = df["persona_idx"].values
        self.prod_idx = df["prod_idx"].values
        self.cat1_idx = df["category_L1_idx"].values
        self.cat2_idx = df["category_L2_idx"].values
        self.cat3_idx = df["category_L3_idx"].values
        self.tier_idx = df["price_tier_idx"].values
        self.user_cat1_dist = user_cat1_dist
        self.user_tier_dist = user_tier_dist
        self.user_activity_idx = user_activity_idx
        # 네거티브 샘플링용
        self.prod_cat1 = prod_cat1
        self.prod_cat2 = prod_cat2
        self.prod_cat3 = prod_cat3
        self.prod_tier = prod_tier
        self.cat1_to_prods = cat1_to_prods
        self.num_prods = num_prods
        self.hard_neg = hard_neg
        self.rand_neg = rand_neg

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, i: int) -> dict:
        u = self.user_idx[i]
        i_pos = self.prod_idx[i]
        cat1_pos = self.cat1_idx[i]

        # 하드 네거티브: 같은 cat_L1 풀에서 균일 샘플
        same_pool = self.cat1_to_prods[cat1_pos]
        hard = np.random.choice(same_pool, size=self.hard_neg, replace=True)
        # 랜덤 네거티브: 전체 카탈로그 균일 샘플
        rand = np.random.randint(0, self.num_prods, size=self.rand_neg)
        neg = np.concatenate([hard, rand]).astype(np.int64)  # (K,)

        return (
            torch.tensor(u, dtype=torch.long),
            torch.tensor(self.persona_idx[i], dtype=torch.long),
            torch.from_numpy(self.user_cat1_dist[u]),
            torch.from_numpy(self.user_tier_dist[u]),
            torch.tensor(self.user_activity_idx[u], dtype=torch.long),
            torch.tensor(i_pos, dtype=torch.long),
            torch.tensor(self.cat1_idx[i], dtype=torch.long),
            torch.tensor(self.cat2_idx[i], dtype=torch.long),
            torch.tensor(self.cat3_idx[i], dtype=torch.long),
            torch.tensor(self.tier_idx[i], dtype=torch.long),
            # 네거티브 (K,)
            torch.from_numpy(neg),
            torch.from_numpy(self.prod_cat1[neg]),
            torch.from_numpy(self.prod_cat2[neg]),
            torch.from_numpy(self.prod_cat3[neg]),
            torch.from_numpy(self.prod_tier[neg]),
        )


dataset = PairDataset(
    df_logs, user_cat1_dist, user_tier_dist, user_activity_idx,
    prod_cat1, prod_cat2, prod_cat3, prod_tier, cat1_to_prods,
    num_prods,
)
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True, drop_last=True)


# ──────────────────────────────────────────────
# 3. Two-Tower 모델 (행동 피처 + 차원 재조정 + cat1/tier 임베딩 공유)
# ──────────────────────────────────────────────
USER_DIM = 64   # user-id 임베딩 차원
ITEM_DIM = 64   # item-id 임베딩 차원
SIDE_DIM = 16   # 카테고리/속성류 임베딩 차원 (cat1, cat2, cat3, tier, persona, activity)
OUT_DIM  = 64   # 최종 retrieval 임베딩 차원 (FAISS 인덱스 호환 위해 유지)


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_prods, n_persona, n_cat1, n_cat2, n_cat3, n_tier,
                 n_activity=NUM_ACTIVITY, user_dim=USER_DIM, item_dim=ITEM_DIM,
                 side_dim=SIDE_DIM, out_dim=OUT_DIM, seq_len=SEQ_LEN):
        super().__init__()

        # User-only
        self.user_emb     = nn.Embedding(n_users, user_dim)
        self.persona_emb  = nn.Embedding(n_persona, side_dim)
        self.activity_emb = nn.Embedding(n_activity, side_dim)

        # Item-only
        self.item_emb = nn.Embedding(n_prods, item_dim)
        self.cat2_emb = nn.Embedding(n_cat2, side_dim)
        self.cat3_emb = nn.Embedding(n_cat3, side_dim)

        # SHARED 임베딩 (양 타워에서 모두 사용)
        self.cat1_emb = nn.Embedding(n_cat1, side_dim)
        self.tier_emb = nn.Embedding(n_tier, side_dim)

        # v8: 시퀀스 인코더 (item_emb 공유)
        self.seq_len    = seq_len
        self.seq_hidden = item_dim
        self.seq_gru    = nn.GRU(item_dim, self.seq_hidden, batch_first=True)
        # user_idx → 시퀀스(prod_idx 10개) lookup buffer (state_dict 에 같이 저장됨)
        self.register_buffer("user_seq", torch.zeros(n_users, seq_len, dtype=torch.long))

        # User MLP 입력: user + persona + cat1_pref + tier_pref + activity + seq
        user_in = user_dim + side_dim * 4 + self.seq_hidden
        # Item MLP 입력: item + cat1 + cat2 + cat3 + tier
        item_in = item_dim + side_dim * 4

        self.user_mlp = nn.Sequential(
            nn.Linear(user_in, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(item_in, 128),
            nn.ReLU(),
            nn.Linear(128, out_dim),
        )

    def encode_user(self, u_idx, persona_idx, cat1_dist, tier_dist, activity_idx, seq_idx=None) -> torch.Tensor:
        # v9: 학습 시 sample-aware seq_idx 전달, 평가/서빙 시 buffer fallback
        seq = seq_idx if seq_idx is not None else self.user_seq[u_idx]   # (B, T)
        seq_emb = self.item_emb(seq)                     # (B, T, item_dim)
        _, h = self.seq_gru(seq_emb)                     # h: (1, B, hidden)
        seq_vec = h.squeeze(0)                           # (B, hidden)

        cat_pref  = cat1_dist @ self.cat1_emb.weight    # (B, side_dim)
        tier_pref = tier_dist @ self.tier_emb.weight    # (B, side_dim)
        x = torch.cat([
            self.user_emb(u_idx),
            self.persona_emb(persona_idx),
            cat_pref,
            tier_pref,
            self.activity_emb(activity_idx),
            seq_vec,
        ], dim=-1)
        return F.normalize(self.user_mlp(x), p=2, dim=-1)

    def encode_item(self, p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx) -> torch.Tensor:
        x = torch.cat([
            self.item_emb(p_idx),
            self.cat1_emb(cat1_idx),
            self.cat2_emb(cat2_idx),
            self.cat3_emb(cat3_idx),
            self.tier_emb(tier_idx),
        ], dim=-1)
        return F.normalize(self.item_mlp(x), p=2, dim=-1)




model = TwoTowerModel(
    num_users,
    num_prods,
    num_persona,
    num_cat1,
    num_cat2,
    num_cat3,
    num_tier,
    NUM_ACTIVITY,
).to(device)

# v8: user_seq buffer 초기화 (학습 전에)
model.user_seq.copy_(torch.from_numpy(user_seq_arr).to(device))
print(f"  user_seq buffer 등록: {model.user_seq.shape}")

# ── 3번: logQ correction 용 item 빈도 분포 사전계산 ──
print("logQ correction 용 item 빈도 계산 중...")
item_freq = np.zeros(num_prods, dtype=np.float64)
np.add.at(item_freq, df_logs["prod_idx"].values, 1.0)
item_p = item_freq / item_freq.sum()
item_log_q = np.log(item_p + 1e-12).astype(np.float32)
item_log_q_t = torch.tensor(item_log_q, device=device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ──────────────────────────────────────────────
# 4. In-batch sampled softmax 학습 (+ logQ correction, CosineAnnealingLR)
# ──────────────────────────────────────────────
EPOCHS = 12   # v9: 20 → 12 (sample-aware seq IO 부담 + 충분히 수렴)
TEMPERATURE = 0.07

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

print(f"Two-Tower 학습 시작 (Epochs: {EPOCHS}, Loss: in-batch softmax + logQ + {NUM_NEG} explicit negatives)...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    n_batches  = 0

    for batch in dataloader:
        (u_idx, persona_idx, u_cat1_dist, u_tier_dist, u_activity_idx,
         p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx,
         neg_idx, neg_cat1, neg_cat2, neg_cat3, neg_tier) = batch

        u_idx = u_idx.to(device)
        persona_idx = persona_idx.to(device)
        u_cat1_dist = u_cat1_dist.to(device)
        u_tier_dist = u_tier_dist.to(device)
        u_activity_idx = u_activity_idx.to(device)
        p_idx = p_idx.to(device)
        cat1_idx = cat1_idx.to(device)
        cat2_idx = cat2_idx.to(device)
        cat3_idx = cat3_idx.to(device)
        tier_idx = tier_idx.to(device)
        neg_idx  = neg_idx.to(device)        # (B, K)
        neg_cat1 = neg_cat1.to(device)
        neg_cat2 = neg_cat2.to(device)
        neg_cat3 = neg_cat3.to(device)
        neg_tier = neg_tier.to(device)

        B = u_idx.size(0)
        K = neg_idx.size(1)

        # buffer 의 정적 시퀀스 사용 (학습/평가 분포 일치)
        u_vec = model.encode_user(u_idx, persona_idx, u_cat1_dist, u_tier_dist, u_activity_idx)
        i_vec = model.encode_item(p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx)

        # 네거티브 (B*K) 인코딩 → (B, K, D)
        neg_vec_flat = model.encode_item(
            neg_idx.reshape(-1),
            neg_cat1.reshape(-1),
            neg_cat2.reshape(-1),
            neg_cat3.reshape(-1),
            neg_tier.reshape(-1),
        )
        neg_vec = neg_vec_flat.view(B, K, -1)  # (B, K, D)

        # in-batch logits + logQ correction
        log_q_batch     = item_log_q_t[p_idx]                                      # (B,)
        logits_inbatch  = u_vec @ i_vec.T / TEMPERATURE - log_q_batch.unsqueeze(0) # (B, B)
        # 샘플된 네거티브 logits (logQ 미적용: positive 분포 아님)
        logits_sampled  = torch.einsum("bd,bkd->bk", u_vec, neg_vec) / TEMPERATURE # (B, K)

        logits = torch.cat([logits_inbatch, logits_sampled], dim=-1)               # (B, B+K)
        labels = torch.arange(B, device=device)                                    # positive at column b
        loss   = F.cross_entropy(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    print(f"  Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/n_batches:.4f} | lr: {scheduler.get_last_lr()[0]:.2e}")


# ──────────────────────────────────────────────
# 5. 모델/임베딩/유저 행동 피처 저장
# ──────────────────────────────────────────────
os.makedirs("data/models", exist_ok=True)
os.makedirs("data/indices", exist_ok=True)

torch.save({
    "state_dict": model.state_dict(),
    "num_users": num_users,
    "num_prods": num_prods,
    "num_persona": num_persona,
    "num_cat1": num_cat1,
    "num_cat2": num_cat2,
    "num_cat3": num_cat3,
    "num_tier": num_tier,
    "num_activity": NUM_ACTIVITY,
    "user_dim": USER_DIM,
    "item_dim": ITEM_DIM,
    "side_dim": SIDE_DIM,
    "out_dim":  OUT_DIM,
    "seq_len":  SEQ_LEN,
}, "data/models/two_tower.pth")
# v8: 시퀀스도 별도 저장 (phase3/phase4 가 모델 외에서도 활용)
np.save("data/models/two_tower_user_seq.npy", user_seq_arr)


# ──────────────────────────────────────────────
# 6. 전체 카탈로그 Item 임베딩 사전 계산 → FAISS
# ──────────────────────────────────────────────
print("전체 카탈로그 임베딩 추출 중...")
model.eval()
all_p_idx = torch.tensor(df_prods["prod_idx"].values, dtype=torch.long).to(device)
all_cat1_idx = torch.tensor(df_prods["category_L1_idx"].values, dtype=torch.long).to(device)
all_cat2_idx = torch.tensor(df_prods["category_L2_idx"].values, dtype=torch.long).to(device)
all_cat3_idx = torch.tensor(df_prods["category_L3_idx"].values, dtype=torch.long).to(device)
all_tier_idx = torch.tensor(df_prods["price_tier_idx"].values, dtype=torch.long).to(device)


with torch.no_grad():
    item_vecs = []
    bs = 2048
    for i in range(0, len(all_p_idx), bs):
        v = model.encode_item(
            all_p_idx[i:i+bs],
            all_cat1_idx[i:i+bs],
            all_cat2_idx[i:i+bs],
            all_cat3_idx[i:i+bs],
            all_tier_idx[i:i+bs],
        )
        item_vecs.append(v.cpu().numpy())
    item_emb = np.vstack(item_vecs).astype("float32")

faiss.normalize_L2(item_emb)
index = faiss.IndexFlatIP(item_emb.shape[1])
id_index = faiss.IndexIDMap(index)
id_index.add_with_ids(item_emb, df_prods["prod_idx"].values.astype("int64"))

faiss.write_index(id_index, "data/indices/candidate_item.index")

np.save("data/models/item_embeddings.npy", item_emb)
df_prods[["product_id", "prod_idx"]].to_csv("data/models/two_tower_prod_map.csv", index=False)

# ── 유저 매핑 + 활동량 버킷 (서빙/평가용) ──
user_map_df = df_users[["user_id", "user_idx", "persona_idx"]].copy()
user_map_df["activity_idx"] = user_activity_idx[df_users["user_idx"].values]
user_map_df.to_csv("data/models/two_tower_user_map.csv", index=False)

# ── 유저 행동 분포 행렬 (row = user_idx) ──
np.save("data/models/two_tower_user_cat1_dist.npy", user_cat1_dist)
np.save("data/models/two_tower_user_tier_dist.npy", user_tier_dist)

print("[DONE] Two-Tower 학습 + FAISS IndexIDMap 저장 완료!")
print("   - data/models/two_tower.pth")
print("   - data/indices/candidate_item.index  (IndexIDMap, IP)")
print("   - data/models/item_embeddings.npy")
print("   - data/models/two_tower_user_cat1_dist.npy")
print("   - data/models/two_tower_user_tier_dist.npy")
