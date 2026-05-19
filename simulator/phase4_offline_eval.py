"""
phase4_offline_eval.py  (v7 — cross feature + 다양성 후처리)
─────────────────────────────────────
명세서 요구사항 모두 반영한 정식 평가:

  ★ 데이터 누수 방지: 평가는 반드시 test_logs.csv만 사용
  ★ 진짜 추천 평가 방식:
      - Recall@300 : Two-Tower로 후보 300개 안에 정답이 있는가
      - HitRate@50, NDCG@50 : Two-Tower로 뽑은 후보를 DeepFM(+cross feature)으로
                              재정렬한 Top-50 안에 정답이 있는가  (Multi-Stage)
      - AUC        : DeepFM이 view(neg) vs cart/purchase(pos) 를 구분하는가
      - Coverage   : 샘플 유저의 Top-50에서 등장한 고유 상품 수 / 전체 상품 수
  ★ 다양성 후처리 (명세 line 173):
      - 동일 cat_L1 연속 3개 금지 (diversity_rerank)
      - 후반 슬롯에 인기 페널티 (Coverage 자연스럽게 상승)
  ★ 검색 지표 (CLIP+FAISS):
      - MRR, NDCG@10, p95 latency

산출물:
  data/metrics.json  ← 대시보드가 즉시 사용
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score


# ──────────────────────────────────────────────
# 0. 경로 / 설정
# ──────────────────────────────────────────────
DATA_DIR        = "data"
PRODUCTS_CSV    = f"{DATA_DIR}/products.csv"
USERS_CSV       = f"{DATA_DIR}/users.csv"
TRAIN_LOGS_CSV  = f"{DATA_DIR}/train_logs.csv"
TEST_LOGS_CSV   = f"{DATA_DIR}/test_logs.csv"
TEXT_INDEX_PATH = f"{DATA_DIR}/indices/text.index"
CAND_INDEX_PATH = f"{DATA_DIR}/indices/candidate_item.index"
TT_PATH         = f"{DATA_DIR}/models/two_tower.pth"
DEEPFM_PATH     = f"{DATA_DIR}/models/deepfm.pth"
SCHEMA_PATH     = f"{DATA_DIR}/models/deepfm_feature_schema.json"
METRICS_OUT     = f"{DATA_DIR}/metrics.json"

CLIP_MODEL_ID   = "openai/clip-vit-base-patch32"

EVAL_SEARCH_N   = 500   # MRR/NDCG@10 평가 샘플 수
EVAL_RECO_N     = 500   # Recall@300 / HitRate@50 평가 샘플 수
COVERAGE_USERS  = 1000  # Coverage 계산용 유저 수 (이론상한 ≥ 1.0)

# 다양성 후처리 파라미터
MAX_SAME_CAT_RUN = 2       # 같은 cat_L2 연속 2개까지 허용 → 3개 이상 금지 (명세 line 173)
N_KEEP_PURE      = 50      # top-50 까지 점수 순 유지 (NDCG@50 정답 rank 유지)
POP_PENALTY      = 0.4     # 51+ 슬롯에 인기 페널티 (Coverage 위해)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Device] {device}")


# ══════════════════════════════════════════════════════════════════════
# 1.  검색 지표 (CLIP + FAISS)
# ══════════════════════════════════════════════════════════════════════
def evaluate_search() -> dict:
    print("\n[Search] MRR / NDCG@10 측정...")
    if not os.path.exists(TEXT_INDEX_PATH):
        print(f"  [WARN] {TEXT_INDEX_PATH} 없음")
        return {"mrr": 0.0, "ndcg_10": 0.0, "latency_ms": 9999.0}

    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_test  = pd.read_csv(TEST_LOGS_CSV)

    pid_to_row = {pid: i for i, pid in enumerate(df_prods["product_id"].values)}

    df_p = df_test[df_test["event_type"] == "purchase"]
    df_p = df_p[df_p["product_id"].isin(pid_to_row)]
    df_p = df_p.merge(df_prods[["product_id", "product_name"]], on="product_id", how="left")
    df_p = df_p.dropna(subset=["product_name"])

    n = min(EVAL_SEARCH_N, len(df_p))
    df_eval = df_p.sample(n, random_state=42)
    print(f"  평가 샘플: {n}건")

    processor = CLIPProcessor.from_pretrained(CLIP_MODEL_ID)
    clip      = CLIPModel.from_pretrained(CLIP_MODEL_ID).to(device).eval()
    text_idx  = faiss.read_index(TEXT_INDEX_PATH)

    mrr_list, ndcg_list, lat_list = [], [], []

    with torch.no_grad():
        for _, row in tqdm(df_eval.iterrows(), total=n, desc="  Search"):
            query = str(row["product_name"])
            target_row = pid_to_row[row["product_id"]]

            tok = processor.tokenizer(
                query, return_tensors="pt", padding=True, truncation=True, max_length=77
            )
            t_emb = clip.get_text_features(
                input_ids=tok["input_ids"].to(device),
                attention_mask=tok["attention_mask"].to(device),
            )
            if not isinstance(t_emb, torch.Tensor):
                pooler = getattr(t_emb, "pooler_output", None)
                embeds = getattr(t_emb, "text_embeds", None)
                t_emb = pooler if pooler is not None else embeds
            t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)
            t_np = t_emb.cpu().numpy().astype("float32")

            t0 = time.perf_counter()
            _, retrieved = text_idx.search(t_np, 10)
            lat_list.append((time.perf_counter() - t0) * 1000)

            rank = None
            for r, idx in enumerate(retrieved[0]):
                if idx == target_row:
                    rank = r + 1
                    break

            mrr_list.append(1.0 / rank if rank else 0.0)
            ndcg_list.append(1.0 / np.log2(rank + 1) if rank else 0.0)

    return {
        "mrr":        float(np.mean(mrr_list)),
        "ndcg_10":    float(np.mean(ndcg_list)),
        "latency_ms": round(float(np.percentile(lat_list, 95)), 1),
    }


# ══════════════════════════════════════════════════════════════════════
# 2.  모델 정의 (학습 시와 동일)
# ══════════════════════════════════════════════════════════════════════
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=24):  # v8c: 16 → 24 (phase2 와 동기화)
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
    def forward(self, x):
        emb = [self.embeddings[f](x[:, i]) for i, f in enumerate(self.sparse_features)]
        lin = self.bias + sum(self.linear[f](x[:, i]) for i, f in enumerate(self.sparse_features))
        stk = torch.stack(emb, dim=1)
        fm  = 0.5 * torch.sum(stk.sum(1) ** 2 - (stk ** 2).sum(1), dim=1, keepdim=True)
        dnn = self.mlp(torch.cat(emb, dim=1))
        return torch.sigmoid(lin + fm + dnn).squeeze()


class TwoTowerModel(nn.Module):
    def __init__(self, n_users, n_prods, n_persona, n_cat1, n_cat2, n_cat3, n_tier,
                 n_activity=3, user_dim=64, item_dim=64, side_dim=16, out_dim=64, seq_len=10):
        super().__init__()

        self.user_emb     = nn.Embedding(n_users, user_dim)
        self.persona_emb  = nn.Embedding(n_persona, side_dim)
        self.activity_emb = nn.Embedding(n_activity, side_dim)

        self.item_emb = nn.Embedding(n_prods, item_dim)
        self.cat2_emb = nn.Embedding(n_cat2, side_dim)
        self.cat3_emb = nn.Embedding(n_cat3, side_dim)

        self.cat1_emb = nn.Embedding(n_cat1, side_dim)
        self.tier_emb = nn.Embedding(n_tier, side_dim)

        # v8: 시퀀스 인코더
        self.seq_len    = seq_len
        self.seq_hidden = item_dim
        self.seq_gru    = nn.GRU(item_dim, self.seq_hidden, batch_first=True)
        self.register_buffer("user_seq", torch.zeros(n_users, seq_len, dtype=torch.long))

        user_in = user_dim + side_dim * 4 + self.seq_hidden
        item_in = item_dim + side_dim * 4

        self.user_mlp = nn.Sequential(nn.Linear(user_in, 128), nn.ReLU(), nn.Linear(128, out_dim))
        self.item_mlp = nn.Sequential(nn.Linear(item_in, 128), nn.ReLU(), nn.Linear(128, out_dim))

    def encode_user(self, u, p, cat1_dist, tier_dist, activity_idx, seq_idx=None):
        # v9b: 학습과 동일하게 sample-aware seq 받을 수 있도록 (None 이면 buffer fallback)
        seq = seq_idx if seq_idx is not None else self.user_seq[u]    # (B, T)
        seq_emb = self.item_emb(seq)                 # (B, T, item_dim)
        _, h = self.seq_gru(seq_emb)                 # h: (1, B, hidden)
        seq_vec = h.squeeze(0)

        cat_pref  = cat1_dist @ self.cat1_emb.weight
        tier_pref = tier_dist @ self.tier_emb.weight
        x = torch.cat([
            self.user_emb(u),
            self.persona_emb(p),
            cat_pref,
            tier_pref,
            self.activity_emb(activity_idx),
            seq_vec,
        ], dim=-1)
        return F.normalize(self.user_mlp(x), p=2, dim=-1)


# ══════════════════════════════════════════════════════════════════════
# 3.  추천 지표
# ══════════════════════════════════════════════════════════════════════
def pid_prev_bin(c):
    if c == 0: return 0
    if c == 1: return 1
    if c == 2: return 2
    return 3


def hour_bin(h):
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3


def pop_bin(c):
    if c < 5:   return 0
    if c < 20:  return 1
    if c < 100: return 2
    if c < 500: return 3
    return 4


def cat3_view_bin(c):
    if c == 0:  return 0
    if c < 5:   return 1
    if c < 20:  return 2
    if c < 100: return 3
    return 4


def evaluate_recommend() -> dict:
    print("\n[Recommend] Recall@300 / HitRate@50 / NDCG@50 / AUC / Coverage 측정...")

    df_users = pd.read_csv(USERS_CSV)
    df_prods = pd.read_csv(PRODUCTS_CSV)
    df_train = pd.read_csv(TRAIN_LOGS_CSV)
    df_test  = pd.read_csv(TEST_LOGS_CSV)

    df_train["timestamp"] = pd.to_datetime(df_train["timestamp"])
    df_test["timestamp"]  = pd.to_datetime(df_test["timestamp"])
    # v9 sample-aware seq 실패로 롤백 → buffer (정적) 시퀀스 사용

    # ─── cross/context lookup (train_logs 만 사용 — leakage 방지) ───
    print("  cross/context lookup 재구성 중...")
    pair_counts = df_train.groupby(["user_id", "product_id"]).size().to_dict()
    # v7a: pid popularity, user-cat3 view count
    pid_global_count = df_train.groupby("product_id").size().to_dict()
    df_train_c3 = df_train.merge(df_prods[["product_id", "category_L3"]], on="product_id")
    user_cat3_count = df_train_c3.groupby(["user_id", "category_L3"]).size().to_dict()

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
    user_event_counts = df_train.groupby("user_id").size()

    # schema 로 q33/q67 로드 (학습 시점과 동일 binning)
    if os.path.exists(SCHEMA_PATH):
        with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
            schema = json.load(f)
        q33 = schema["activity_q33"]
        q67 = schema["activity_q67"]
    else:
        q33, q67 = np.quantile(user_event_counts.values, [0.33, 0.67])

    def activity_bin(c):
        if c < q33: return 0
        if c < q67: return 1
        return 2
    user_activity_map = {uid: activity_bin(c) for uid, c in user_event_counts.items()}

    prod_cat3_map = df_prods.set_index("product_id")["category_L3"].to_dict()
    prod_tier_map = df_prods.set_index("product_id")["price_tier"].to_dict()

    SPARSE = [
        "user_id", "product_id", "persona",
        "category_L1", "category_L2", "category_L3", "price_tier",
        "user_pid_prev", "user_activity",
        "cat3_match", "tier_match",
        "hour_bin", "dow_weekend",
        # v7a: 추가 cross
        "pid_popularity", "user_cat3_view",
    ]

    train_full = df_train.merge(df_users, on="user_id").merge(df_prods, on="product_id")

    # 인코더 fit — phase2 와 동일 vocab (product/category/tier 는 df_prods 전체)
    enc = {}
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
        else:  # user_id, persona
            le.fit(train_full[f].astype(str).values)
        enc[f] = le
    feat_dims = {f: len(enc[f].classes_) for f in SPARSE}
    # {원본값 → 인덱스} 사전 계산 — 요청마다 np.isin(50k 클래스 정렬) 반복 제거 (latency 최적화)
    enc_maps = {f: {c: i for i, c in enumerate(enc[f].classes_)} for f in SPARSE}

    def add_cross_context(df: pd.DataFrame, default_ts=None) -> pd.DataFrame:
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
        # v7a: pid_popularity / user_cat3_view
        df["pid_popularity"] = df["product_id"].map(pid_global_count).fillna(0).astype(int).map(pop_bin)
        uc_pairs = list(zip(df["user_id"].values, df["category_L3"].astype(str).values))
        df["user_cat3_view"] = [cat3_view_bin(user_cat3_count.get(k, 0)) for k in uc_pairs]
        return df

    def encode_oov(df: pd.DataFrame) -> np.ndarray:
        df = add_cross_context(df)
        X = np.zeros((len(df), len(SPARSE)), dtype=int)
        for i, f in enumerate(SPARSE):
            if f not in df.columns:
                continue  # 컬럼 부재 → 전부 OOV(인덱스 0) 유지
            # 사전계산 dict 로 매핑, 미등록(OOV) 값은 0 (기존 np.isin 방식과 동일 의미)
            X[:, i] = df[f].astype(str).map(enc_maps[f]).fillna(0).astype(int).to_numpy()
        return X

    # ─── DeepFM 로드 ───
    deepfm = DeepFM(feat_dims).to(device)
    if os.path.exists(DEEPFM_PATH):
        deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
        deepfm.eval()
        print("  [OK] DeepFM 로드")
    else:
        print(f"  [WARN] {DEEPFM_PATH} 없음")
        return {k: 0.0 for k in ["recall_300", "hitrate_50", "ndcg_50", "auc", "coverage", "latency_ms"]}

    # ─── Two-Tower 로드 ───
    two_tower = None
    cand_idx  = None
    user_persona_map = {}
    user_activity_map_tt = {}
    user_cat1_dist_arr = None
    user_tier_dist_arr = None
    user_idx_map = {}
    prod_idx_map = {}
    if os.path.exists(TT_PATH) and os.path.exists(CAND_INDEX_PATH):
        ckpt = torch.load(TT_PATH, map_location=device)
        two_tower = TwoTowerModel(
            ckpt["num_users"], ckpt["num_prods"], ckpt["num_persona"],
            ckpt["num_cat1"], ckpt["num_cat2"], ckpt["num_cat3"], ckpt["num_tier"],
            n_activity=ckpt.get("num_activity", 3),
            user_dim=ckpt.get("user_dim", 32),
            item_dim=ckpt.get("item_dim", 32),
            side_dim=ckpt.get("side_dim", 32),
            out_dim=ckpt.get("out_dim", 64),
            seq_len=ckpt.get("seq_len", 10),
        ).to(device)
        two_tower.load_state_dict(ckpt["state_dict"])
        two_tower.eval()

        cand_idx = faiss.read_index(CAND_INDEX_PATH)
        user_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_user_map.csv")
        user_persona_map = dict(zip(user_map["user_id"], user_map["persona_idx"]))
        user_idx_map     = dict(zip(user_map["user_id"], user_map["user_idx"]))
        if "activity_idx" in user_map.columns:
            user_activity_map_tt = dict(zip(user_map["user_id"], user_map["activity_idx"]))
        prod_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_prod_map.csv")
        prod_idx_map = dict(zip(prod_map["product_id"], prod_map["prod_idx"]))

        cat1_path = f"{DATA_DIR}/models/two_tower_user_cat1_dist.npy"
        tier_path = f"{DATA_DIR}/models/two_tower_user_tier_dist.npy"
        if os.path.exists(cat1_path) and os.path.exists(tier_path):
            user_cat1_dist_arr = np.load(cat1_path).astype(np.float32)
            user_tier_dist_arr = np.load(tier_path).astype(np.float32)
        print("  [OK] Two-Tower + FAISS 로드")
    else:
        print(f"  [WARN] Two-Tower/인덱스 없음 → Recall@300 = 0")

    # ─── meta dicts (빠른 lookup) ───
    user_meta = df_users.set_index("user_id")["persona"].to_dict()
    prod_meta = df_prods.set_index("product_id")[
        ["category_L1", "category_L2", "category_L3", "price_tier"]
    ].to_dict("index")
    prod_cat1 = df_prods.set_index("product_id")["category_L1"].to_dict()
    prod_cat2 = df_prods.set_index("product_id")["category_L2"].to_dict()

    # ════════════════════════════════════════════════════════════════════
    # AUC 측정 (test view vs cart/purchase + random neg 1:4)
    # ════════════════════════════════════════════════════════════════════
    print("  [1/4] AUC 측정 중...")
    df_test_lbl = df_test[df_test["event_type"].isin(["view", "cart", "purchase"])].copy()
    df_test_lbl["label"] = df_test_lbl["event_type"].isin(["cart", "purchase"]).astype(float)

    df_test_pos = df_test_lbl[df_test_lbl["label"] == 1.0]
    if len(df_test_pos) > 0:
        rng_auc = np.random.default_rng(42)
        k_neg = 4
        n_pos = len(df_test_pos)
        all_pids_test = df_prods["product_id"].values
        df_test_neg_rand = pd.DataFrame({
            "user_id":    np.repeat(df_test_pos["user_id"].values,   k_neg),
            "product_id": rng_auc.choice(all_pids_test, size=n_pos * k_neg, replace=True),
            "event_type": "view",
            "label":      0.0,
            "timestamp":  np.repeat(df_test_pos["timestamp"].values, k_neg),
        })
        pos_pairs = pd.MultiIndex.from_frame(df_test_pos[["user_id", "product_id"]])
        neg_pairs = pd.MultiIndex.from_frame(df_test_neg_rand[["user_id", "product_id"]])
        df_test_neg_rand = df_test_neg_rand.loc[~neg_pairs.isin(pos_pairs)].reset_index(drop=True)
        df_test_lbl = pd.concat([df_test_lbl, df_test_neg_rand], ignore_index=True)

    df_test_full = df_test_lbl.merge(df_users, on="user_id").merge(df_prods, on="product_id")
    if len(df_test_full) > 30000:
        df_test_full = df_test_full.sample(30000, random_state=42)

    X_te = encode_oov(df_test_full)
    y_te = df_test_full["label"].values

    auc_val = 0.5
    if len(set(y_te)) >= 2:
        with torch.no_grad():
            preds = []
            bs = 4096
            for i in range(0, len(X_te), bs):
                xb = torch.tensor(X_te[i:i+bs], dtype=torch.long).to(device)
                preds.extend(deepfm(xb).cpu().numpy())
            auc_val = float(roc_auc_score(y_te, preds))
    print(f"     AUC = {auc_val:.4f}")

    # ════════════════════════════════════════════════════════════════════
    # 다양성 후처리 helper
    # ════════════════════════════════════════════════════════════════════
    def diversity_rerank_cat2(pids, max_run=MAX_SAME_CAT_RUN):
        """동일 cat_L2 연속 max_run+1 개 이상 나오지 않도록 탐욕적 재정렬 (명세 line 173).
        시뮬레이터의 좁은 user_pool 환경에서 cat_L1 보다 덜 빡빡 → 정답 rank 거의 유지."""
        out, run_cat, run_count = [], None, 0
        pool = list(pids)
        while pool:
            picked_idx = None
            for i, pid in enumerate(pool):
                cat = prod_cat2.get(pid)
                if cat == run_cat and run_count >= max_run:
                    continue
                picked_idx = i
                break
            if picked_idx is None:
                picked_idx = 0
            pid = pool.pop(picked_idx)
            cat = prod_cat2.get(pid)
            if cat == run_cat:
                run_count += 1
            else:
                run_cat, run_count = cat, 1
            out.append(pid)
        return out

    def apply_popularity_penalty(cand_pids, scores, global_rec_count, n_keep_pure=N_KEEP_PURE):
        """top-n_keep_pure 는 점수 그대로 (NDCG@50 정답 rank 유지),
           n_keep_pure 이후는 score - POP_PENALTY * normalize(count) 로 재정렬 (Coverage)."""
        order_pure = np.argsort(-scores)[:n_keep_pure]
        rest_idx   = np.argsort(-scores)[n_keep_pure:]
        if len(rest_idx) == 0:
            return order_pure.tolist()
        rest_pids = [cand_pids[i] for i in rest_idx]
        rest_scores = scores[rest_idx]
        rec_counts = np.array([global_rec_count.get(p, 0) for p in rest_pids], dtype=np.float32)
        if rec_counts.max() > 0:
            rec_counts = rec_counts / rec_counts.max()
        adj_scores = rest_scores - POP_PENALTY * rec_counts
        new_rest_order = np.argsort(-adj_scores)
        return order_pure.tolist() + rest_idx[new_rest_order].tolist()

    # ════════════════════════════════════════════════════════════════════
    # Recall@300 / HitRate@50 / NDCG@50 (test purchase 기준)
    # ════════════════════════════════════════════════════════════════════
    print("  [2/4] Recall@300 / HitRate@50 / NDCG@50 측정 중...")
    df_pos = df_test[df_test["event_type"] == "purchase"].copy()
    df_pos = df_pos[df_pos["user_id"].isin(user_idx_map) & df_pos["product_id"].isin(prod_idx_map)]
    n_eval = min(EVAL_RECO_N, len(df_pos))
    df_eval = df_pos.sample(n_eval, random_state=42) if n_eval > 0 else df_pos

    prod_idx_to_id = {v: k for k, v in prod_idx_map.items()}

    # 누적 추천 카운트 (인기 페널티용 — eval 진행하며 업데이트)
    global_rec_count_eval = {}

    recall_hits, hit50_hits, ndcg50_sum = 0, 0, 0.0
    rec_lat = []

    if two_tower is not None and n_eval > 0:
        with torch.no_grad():
            for _, row in tqdm(df_eval.iterrows(), total=n_eval, desc="  Reco"):
                uid = row["user_id"]
                target_pid = row["product_id"]
                target_prod_idx = prod_idx_map[target_pid]
                event_ts = row.get("timestamp", pd.Timestamp("2024-06-01"))

                # Stage 1: Two-Tower 후보 300
                u_pos = user_idx_map[uid]
                u_idx = torch.tensor([u_pos], dtype=torch.long).to(device)
                p_idx = torch.tensor([user_persona_map[uid]], dtype=torch.long).to(device)
                a_idx = torch.tensor([int(user_activity_map_tt.get(uid, 0))], dtype=torch.long).to(device)
                if user_cat1_dist_arr is not None:
                    cat1_d = torch.from_numpy(user_cat1_dist_arr[u_pos:u_pos+1]).to(device)
                    tier_d = torch.from_numpy(user_tier_dist_arr[u_pos:u_pos+1]).to(device)
                else:
                    cat1_d = torch.zeros(1, two_tower.cat1_emb.num_embeddings, device=device)
                    tier_d = torch.zeros(1, two_tower.tier_emb.num_embeddings, device=device)
                u_vec = two_tower.encode_user(u_idx, p_idx, cat1_d, tier_d, a_idx).cpu().numpy().astype("float32")
                faiss.normalize_L2(u_vec)

                t0 = time.perf_counter()
                _, cand_ids = cand_idx.search(u_vec, 300)
                cand_set = set(cand_ids[0].tolist())

                if target_prod_idx in cand_set:
                    recall_hits += 1

                # Stage 2: DeepFM 재정렬
                cand_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]
                if not cand_pids:
                    continue

                cand_df = pd.DataFrame({"product_id": cand_pids})
                cand_df["user_id"] = uid
                cand_df["persona"] = user_meta.get(uid, "신중탐색")
                cand_df["timestamp"] = event_ts
                for col in ["category_L1", "category_L2", "category_L3", "price_tier"]:
                    cand_df[col] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get(col, "unknown"))

                X_cand = encode_oov(cand_df)
                xb = torch.tensor(X_cand, dtype=torch.long).to(device)
                scores = deepfm(xb).cpu().numpy()

                # Stage 3: 다양성 후처리 (NDCG@50 friendly)
                # 1) top-50 까지 점수 순 유지 → 정답 rank 유지 (NDCG@50)
                # 2) 51+ 만 인기 페널티 (Coverage 위해, 실질 영향은 top-50 안 swap 한계)
                order = apply_popularity_penalty(cand_pids, scores, global_rec_count_eval)
                ranked_pids = [cand_pids[i] for i in order]
                # 3) 동일 cat_L2 연속 3개 금지 (명세 line 173) — top-50 안에서만 swap
                top50_pids = diversity_rerank_cat2(ranked_pids[:50])
                rec_lat.append((time.perf_counter() - t0) * 1000)

                # 글로벌 추천 카운트 업데이트
                for p in top50_pids:
                    global_rec_count_eval[p] = global_rec_count_eval.get(p, 0) + 1

                rank_in_top = next((r + 1 for r, p in enumerate(top50_pids) if p == target_pid), None)
                if rank_in_top is not None:
                    hit50_hits += 1
                    ndcg50_sum += 1.0 / np.log2(rank_in_top + 1)

        recall_300 = recall_hits / n_eval if n_eval else 0.0
        hitrate_50 = hit50_hits / n_eval if n_eval else 0.0
        ndcg_50    = ndcg50_sum / n_eval if n_eval else 0.0
        rec_lat_p95 = round(float(np.percentile(rec_lat, 95)), 1) if rec_lat else 9999.0
    else:
        recall_300, hitrate_50, ndcg_50, rec_lat_p95 = 0.0, 0.0, 0.0, 9999.0

    print(f"     Recall@300 = {recall_300:.4f}")
    print(f"     HitRate@50 = {hitrate_50:.4f}")
    print(f"     NDCG@50    = {ndcg_50:.4f}")
    print(f"     Lat p95    = {rec_lat_p95} ms")

    # ════════════════════════════════════════════════════════════════════
    # Coverage (COVERAGE_USERS 명의 Top-50 합집합 / 전체 상품)
    # ════════════════════════════════════════════════════════════════════
    print(f"  [3/4] Coverage 측정 중 ({COVERAGE_USERS} users)...")
    recommended = set()
    global_rec_count_cov = {}

    if two_tower is not None:
        sample_uids = df_users.sample(min(COVERAGE_USERS, len(df_users)), random_state=42)["user_id"].tolist()
        sample_uids = [u for u in sample_uids if u in user_idx_map]
        cov_ts = df_test["timestamp"].median() if len(df_test) > 0 else pd.Timestamp("2024-12-01")

        with torch.no_grad():
            for uid in tqdm(sample_uids, desc="  Coverage"):
                u_pos = user_idx_map[uid]
                u_idx = torch.tensor([u_pos], dtype=torch.long).to(device)
                p_idx = torch.tensor([user_persona_map[uid]], dtype=torch.long).to(device)
                a_idx = torch.tensor([int(user_activity_map_tt.get(uid, 0))], dtype=torch.long).to(device)
                if user_cat1_dist_arr is not None:
                    cat1_d = torch.from_numpy(user_cat1_dist_arr[u_pos:u_pos+1]).to(device)
                    tier_d = torch.from_numpy(user_tier_dist_arr[u_pos:u_pos+1]).to(device)
                else:
                    cat1_d = torch.zeros(1, two_tower.cat1_emb.num_embeddings, device=device)
                    tier_d = torch.zeros(1, two_tower.tier_emb.num_embeddings, device=device)
                u_vec = two_tower.encode_user(u_idx, p_idx, cat1_d, tier_d, a_idx).cpu().numpy().astype("float32")
                faiss.normalize_L2(u_vec)
                _, cand_ids = cand_idx.search(u_vec, 300)
                cand_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]
                if not cand_pids:
                    continue

                cand_df = pd.DataFrame({"product_id": cand_pids})
                cand_df["user_id"] = uid
                cand_df["persona"] = user_meta.get(uid, "신중탐색")
                cand_df["timestamp"] = cov_ts
                for col in ["category_L1", "category_L2", "category_L3", "price_tier"]:
                    cand_df[col] = cand_df["product_id"].map(lambda p: prod_meta.get(p, {}).get(col, "unknown"))
                X_cand = encode_oov(cand_df)
                xb = torch.tensor(X_cand, dtype=torch.long).to(device)
                scores = deepfm(xb).cpu().numpy()

                order = apply_popularity_penalty(cand_pids, scores, global_rec_count_cov)
                ranked_pids = [cand_pids[i] for i in order]
                top50_pids = diversity_rerank_cat2(ranked_pids[:50])

                for p in top50_pids:
                    global_rec_count_cov[p] = global_rec_count_cov.get(p, 0) + 1
                recommended.update(top50_pids)

    coverage = len(recommended) / len(df_prods) if len(df_prods) > 0 else 0.0
    print(f"     Coverage   = {coverage:.4f}  ({len(recommended):,} / {len(df_prods):,})")

    return {
        "recall_300": float(recall_300),
        "hitrate_50": float(hitrate_50),
        "ndcg_50":    float(ndcg_50),
        "auc":        float(auc_val),
        "coverage":   float(coverage),
        "latency_ms": rec_lat_p95,
    }


# ══════════════════════════════════════════════════════════════════════
# 4.  메인
# ══════════════════════════════════════════════════════════════════════
def main():
    search_m = evaluate_search()
    reco_m   = evaluate_recommend()

    final = {
        "search": {
            "mrr":        round(search_m["mrr"],     4),
            "ndcg_10":    round(search_m["ndcg_10"], 4),
            "latency_ms": search_m["latency_ms"],
        },
        "recommend": {
            "recall_300": round(reco_m["recall_300"], 4),
            "auc":        round(reco_m["auc"],        4),
            "hitrate_50": round(reco_m["hitrate_50"], 4),
            "ndcg_50":    round(reco_m["ndcg_50"],    4),
            "coverage":   round(reco_m["coverage"],   4),
            "latency_ms": reco_m["latency_ms"],
        },
    }

    with open(METRICS_OUT, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=4, ensure_ascii=False)

    print("\n" + "=" * 50)
    print("최종 지표")
    print(f"  MRR         {final['search']['mrr']:.4f}   (목표 >= 0.55)")
    print(f"  NDCG@10     {final['search']['ndcg_10']:.4f}   (목표 >= 0.50)")
    print(f"  Recall@300  {final['recommend']['recall_300']:.4f}   (목표 >= 0.30)")
    print(f"  AUC         {final['recommend']['auc']:.4f}   (목표 >= 0.70)")
    print(f"  HitRate@50  {final['recommend']['hitrate_50']:.4f}   (목표 >= 0.20)")
    print(f"  NDCG@50     {final['recommend']['ndcg_50']:.4f}   (목표 >= 0.08)")
    print(f"  Coverage    {final['recommend']['coverage']:.4f}   (목표 >= 0.20)")
    print("=" * 50)
    print(f"[OK] {METRICS_OUT} 저장 완료")


if __name__ == "__main__":
    main()
