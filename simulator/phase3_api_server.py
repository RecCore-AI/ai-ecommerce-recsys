"""
phase3_api_server.py  (전면 개편판: 진짜 Multi-Stage 추천)
─────────────────────────────────────────────────────────
명세서 부합 사항:

  Stage 1 │ Candidate Generation (Two-Tower + FAISS)
            - User Tower(persona 포함) → FAISS IP search → 후보 300개
  Stage 2 │ Ranking (DeepFM)
            - User/Item/Cross/Context 피처
  Stage 3 │ Re-ranking (비즈니스 로직 + MAB)
            - 동일 카테고리 연속 3개 금지 (다양성)
            - 신규 유저(이력 5개 미만): 인기 + 트렌딩으로 폴백
            - 신규 상품(등록 7일 이내): 노출 부스팅
            - Epsilon-Greedy MAB: 상위 N개 중 1~2개를 탐색 슬롯으로

  + Session Encoder (간이 GRU): 최근 클릭 시퀀스 → 단기 관심 임베딩
                                 장기 선호(User Tower) ⊕ 단기 관심 결합
  + 응답 필수 필드: search_type/results/latency_ms/total_count
                   user_id/recommendations/pipeline_latency/session_context
"""
import os
import io
import json
import time
import random
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import redis
from datetime import datetime, timedelta
from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Literal
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from sklearn.preprocessing import LabelEncoder

import ko_fashion   # Query Understanding(보너스#1): 자연어 쿼리 → 가격/색상/카테고리 필터

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
DATA_DIR        = "data"
TEXT_INDEX_PATH = f"{DATA_DIR}/indices/text.index"
IMAGE_INDEX_PATH = f"{DATA_DIR}/indices/image.index"
CAND_INDEX_PATH = f"{DATA_DIR}/indices/candidate_item.index"
DEEPFM_PATH     = f"{DATA_DIR}/models/deepfm.pth"
TT_PATH         = f"{DATA_DIR}/models/two_tower.pth"

CANDIDATE_K     = 300       # Stage 1 후보 개수
RANK_TOP_N      = 50        # Stage 2 → Top-N 통과
SESSION_LEN     = 10        # Redis 세션 시퀀스 길이
NEW_PRODUCT_DAYS = 7        # 신규 상품 정의: 7일 이내
NEW_USER_HISTORY_THRESHOLD = 5  # 신규 유저 정의: 행동 5개 미만
EPSILON         = 0.1       # MAB exploration ratio
MAX_SAME_CATEGORY_RUN = 2   # 연속 3개 금지 → 같은 카테고리 연속 2개까지 허용
# 하이브리드(텍스트+이미지) 검색 late-fusion 가중치 (텍스트 : 이미지)
HYBRID_W_TEXT  = 0.5
HYBRID_W_IMAGE = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Multi-Stage Recommender API")

# 데모 페이지가 (필요 시) 다른 출처에서 호출돼도 동작하도록 CORS 허용.
# 같은 출처(서버가 직접 서빙)로 쓰면 사실상 영향 없음 — 발표 시 호스팅 유연성 확보용.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# 상품 이미지 원본 경로 (P0108775015 → 010/0108775015.jpg)
IMAGES_DIR = f"{DATA_DIR}/raw/images"
DEMO_HTML  = "demo.html"


# ──────────────────────────────────────────────
# 모델 정의 (학습 코드와 정확히 동일해야 함)
# ──────────────────────────────────────────────
class DeepFM(nn.Module):
    def __init__(self, feature_dims, embedding_dim=24):  # v8c: 16 → 24
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

        # SHARED
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
        self.out_dim  = out_dim

    def encode_user(self, u_idx, persona_idx, cat1_dist, tier_dist, activity_idx):
        seq = self.user_seq[u_idx]
        seq_emb = self.item_emb(seq)
        _, h = self.seq_gru(seq_emb)
        seq_vec = h.squeeze(0)

        cat_pref  = cat1_dist @ self.cat1_emb.weight
        tier_pref = tier_dist @ self.tier_emb.weight
        x = torch.cat([
            self.user_emb(u_idx),
            self.persona_emb(persona_idx),
            cat_pref,
            tier_pref,
            self.activity_emb(activity_idx),
            seq_vec,
        ], dim=-1)
        return F.normalize(self.user_mlp(x), p=2, dim=-1)

    def encode_item(self, p_idx, cat1_idx, cat2_idx, cat3_idx, tier_idx):
        x = torch.cat([
            self.item_emb(p_idx),
            self.cat1_emb(cat1_idx),
            self.cat2_emb(cat2_idx),
            self.cat3_emb(cat3_idx),
            self.tier_emb(tier_idx),
        ], dim=-1)
        return F.normalize(self.item_mlp(x), p=2, dim=-1)


class SessionGRU(nn.Module):
    """간이 세션 인코더: 최근 클릭한 item embedding 시퀀스를 GRU로 요약."""
    def __init__(self, item_emb_layer: nn.Embedding, hidden=64, out_dim=64):
        super().__init__()
        self.item_emb_layer = item_emb_layer
        self.gru = nn.GRU(item_emb_layer.embedding_dim, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, out_dim)

    def forward(self, item_idx_seq: torch.Tensor):
        # (1, T) → (1, T, emb) → (1, hidden)
        emb = self.item_emb_layer(item_idx_seq)
        _, h = self.gru(emb)
        return F.normalize(self.proj(h.squeeze(0)), p=2, dim=-1)


# ──────────────────────────────────────────────
# Redis
# ──────────────────────────────────────────────
try:
    redis_client = redis.Redis(host="redis", port=6379, db=0, decode_responses=True)
    redis_client.ping()
    print("✅ Redis 연결 성공")
except Exception:
    try:
        redis_client = redis.Redis(host="127.0.0.1", port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Redis 연결 성공 (localhost)")
    except Exception as e:
        print(f"⚠️ Redis 연결 실패: {e}")
        redis_client = None


# ──────────────────────────────────────────────
# 글로벌 자원 로드
# ──────────────────────────────────────────────
print("🚀 [1/5] CLIP & 텍스트 FAISS 로드...")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
text_faiss     = faiss.read_index(TEXT_INDEX_PATH)
image_faiss    = faiss.read_index(IMAGE_INDEX_PATH) if os.path.exists(IMAGE_INDEX_PATH) else text_faiss

# M-CLIP 텍스트→이미지 인코더 (한/영 의미검색). CLIP text.index 의 text↔text 정확매칭 한계를
# 보완: 다국어 텍스트를 CLIP 이미지공간에 정렬한 인코더로 image.index 를 검색해 '의미적으로
# 닮은' 상품을 회수, CLIP text.index 순위와 RRF 융합. 미설치/실패 시 CLIP 단독으로 graceful fallback.
try:
    import mclip_search as _mc
    mclip_enc = _mc.get_encoder()
    print("🚀 [1/5] M-CLIP 텍스트→이미지 인코더 로드 완료 (한/영 의미검색)")
except Exception as e:
    print(f"⚠️  M-CLIP 인코더 로드 실패({e}) — CLIP 단독 텍스트검색으로 동작.")
    mclip_enc = None

print("🚀 [2/5] 메타데이터 로드...")
df_prods = pd.read_csv(f"{DATA_DIR}/products.csv")
df_users = pd.read_csv(f"{DATA_DIR}/users.csv")
df_logs  = pd.read_csv(f"{DATA_DIR}/train_logs.csv")

# 인기 상품 (purchase 횟수) — 신규유저/MAB 폴백용
popular_pids = (
    df_logs[df_logs["event_type"] == "purchase"]["product_id"]
    .value_counts().head(200).index.tolist()
)

# 트렌딩 상품 (최근 30% 시점의 view)
df_logs_dt = df_logs.copy()
try:
    df_logs_dt["timestamp"] = pd.to_datetime(df_logs_dt["timestamp"])
    cutoff = df_logs_dt["timestamp"].quantile(0.7)
    trending_pids = (
        df_logs_dt[(df_logs_dt["timestamp"] >= cutoff) & (df_logs_dt["event_type"] == "view")]["product_id"]
        .value_counts().head(200).index.tolist()
    )
except Exception:
    trending_pids = popular_pids[:]

# 신규 상품 (등록 7일 이내) — train_logs 마지막 날짜 기준
try:
    latest_date = df_logs_dt["timestamp"].max()
    first_seen = df_logs_dt.groupby("product_id")["timestamp"].min()
    new_pids = first_seen[first_seen >= latest_date - pd.Timedelta(days=NEW_PRODUCT_DAYS)].index.tolist()
except Exception:
    new_pids = []

print(f"  인기 상품 {len(popular_pids)} / 트렌딩 {len(trending_pids)} / 신규 {len(new_pids)}")

print("🚀 [3/5] DeepFM 인코더/모델 로드...")
# feature schema 는 학습(phase2_deepfm.py)이 저장한 파일을 그대로 사용한다.
# → SPARSE_FEATURES / feature_dims 가 deepfm.pth 와 불일치하면 load_state_dict 가
#   실패하므로, 하드코딩 대신 schema 를 단일 진실 공급원으로 삼는다. (v7a 피처 15개 포함)
with open(f"{DATA_DIR}/models/deepfm_feature_schema.json", "r", encoding="utf-8") as f:
    _schema = json.load(f)
SPARSE_FEATURES = _schema["sparse_features"]
feature_dims    = {feat: _schema["feature_dims"][feat] for feat in SPARSE_FEATURES}

# v7a cross feature lookup: train_logs 기준 재계산 (phase2_deepfm.py 와 동일)
pid_global_count = df_logs.groupby("product_id").size().to_dict()
_logs_cat3 = df_logs.merge(df_prods[["product_id", "category_L3"]], on="product_id")
user_cat3_count = _logs_cat3.groupby(["user_id", "category_L3"]).size().to_dict()

encoders = {}
for f in SPARSE_FEATURES:
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

# {원본값 → 인덱스} 사전 계산 — 요청마다 np.isin(50k 클래스 정렬)을 반복하지 않도록 (latency 최적화)
encoder_maps = {f: {c: i for i, c in enumerate(encoders[f].classes_)} for f in SPARSE_FEATURES}

deepfm = DeepFM(feature_dims).to(device)
deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
deepfm.eval()

# v7: cross feature lookup 로드 (phase2_deepfm.py 가 저장)
pair_counts_dict = {}
user_top_cat3 = {}
user_top_tier = {}
user_activity_lookup = {}
try:
    pair_df = pd.read_csv(f"{DATA_DIR}/models/deepfm_pair_counts.csv")
    pair_counts_dict = dict(zip(zip(pair_df["user_id"], pair_df["product_id"]), pair_df["cnt"]))
    user_lookup_df = pd.read_csv(f"{DATA_DIR}/models/deepfm_user_lookup.csv")
    user_top_cat3 = dict(zip(user_lookup_df["user_id"], user_lookup_df["top_cat3"]))
    user_top_tier = dict(zip(user_lookup_df["user_id"], user_lookup_df["top_tier"]))
    user_activity_lookup = dict(zip(user_lookup_df["user_id"], user_lookup_df["user_activity"]))
    print(f"  cross feature lookup 로드: pairs={len(pair_counts_dict):,}, users={len(user_top_cat3):,}")
except Exception as e:
    print(f"  [WARN] cross lookup 로드 실패: {e}")

print("🚀 [4/5] Two-Tower 로드...")
two_tower = None
candidate_faiss = None
user_idx_map = {}
user_persona_map = {}
user_activity_map = {}
prod_idx_map = {}
prod_idx_to_id = {}
user_cat1_dist_arr = None  # (num_users, num_cat1)
user_tier_dist_arr = None  # (num_users, num_tier)

if os.path.exists(TT_PATH) and os.path.exists(CAND_INDEX_PATH):
    ckpt = torch.load(TT_PATH, map_location=device)

    two_tower = TwoTowerModel(
        n_users=ckpt["num_users"],
        n_prods=ckpt["num_prods"],
        n_persona=ckpt["num_persona"],
        n_cat1=ckpt["num_cat1"],
        n_cat2=ckpt["num_cat2"],
        n_cat3=ckpt["num_cat3"],
        n_tier=ckpt["num_tier"],
        n_activity=ckpt.get("num_activity", 3),
        user_dim=ckpt.get("user_dim", 32),
        item_dim=ckpt.get("item_dim", 32),
        side_dim=ckpt.get("side_dim", 32),
        out_dim=ckpt.get("out_dim", 64),
        seq_len=ckpt.get("seq_len", 10),
    ).to(device)

    two_tower.load_state_dict(ckpt["state_dict"])
    two_tower.eval()
    candidate_faiss = faiss.read_index(CAND_INDEX_PATH)

    user_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_user_map.csv")
    user_idx_map     = dict(zip(user_map["user_id"], user_map["user_idx"]))
    user_persona_map = dict(zip(user_map["user_id"], user_map["persona_idx"]))
    if "activity_idx" in user_map.columns:
        user_activity_map = dict(zip(user_map["user_id"], user_map["activity_idx"]))
    prod_map = pd.read_csv(f"{DATA_DIR}/models/two_tower_prod_map.csv")
    prod_idx_map     = dict(zip(prod_map["product_id"], prod_map["prod_idx"]))
    prod_idx_to_id   = {v: k for k, v in prod_idx_map.items()}

    cat1_path = f"{DATA_DIR}/models/two_tower_user_cat1_dist.npy"
    tier_path = f"{DATA_DIR}/models/two_tower_user_tier_dist.npy"
    if os.path.exists(cat1_path) and os.path.exists(tier_path):
        user_cat1_dist_arr = np.load(cat1_path).astype(np.float32)
        user_tier_dist_arr = np.load(tier_path).astype(np.float32)
        print(f"  ✅ 유저 행동 분포 로드: cat1 {user_cat1_dist_arr.shape}, tier {user_tier_dist_arr.shape}")
    print("  ✅ Two-Tower + FAISS IndexIDMap 로드")

print("🚀 [5/5] 세션 GRU 초기화...")
session_encoder = None
if two_tower is not None:
    session_encoder = SessionGRU(two_tower.item_emb).to(device).eval()

# 빠른 조회용
prod_meta_dict = df_prods.set_index("product_id").to_dict("index")
prod_cat3_map = {pid: m.get("category_L3") for pid, m in prod_meta_dict.items()}
prod_tier_map = {pid: m.get("price_tier")  for pid, m in prod_meta_dict.items()}
user_persona_str = df_users.set_index("user_id")["persona"].to_dict()
all_pids_set = set(df_prods["product_id"].values)

print("✅ 서버 준비 완료!")


# ──────────────────────────────────────────────
# 헬퍼
# ──────────────────────────────────────────────
def _pid_prev_bin(c: int) -> int:
    if c == 0: return 0
    if c == 1: return 1
    if c == 2: return 2
    return 3


def _hour_bin(h: int) -> int:
    if h < 6:  return 0
    if h < 12: return 1
    if h < 18: return 2
    return 3


def _pop_bin(c: int) -> int:
    """v7a: product 인기도 binning (phase2_deepfm.py pop_bin 과 동일)."""
    if c < 5:   return 0
    if c < 20:  return 1
    if c < 100: return 2
    if c < 500: return 3
    return 4


def _cat3_view_bin(c: int) -> int:
    """v7a: user-cat3 view 카운트 binning (phase2_deepfm.py cat3_view_bin 과 동일)."""
    if c == 0:  return 0
    if c < 5:   return 1
    if c < 20:  return 2
    if c < 100: return 3
    return 4


def _add_cross_context(df: pd.DataFrame) -> pd.DataFrame:
    """v7: phase2_deepfm.py 와 동일한 cross/context feature 추가."""
    df = df.copy()
    pairs = list(zip(df["user_id"].values, df["product_id"].values))
    df["user_pid_prev"] = [_pid_prev_bin(pair_counts_dict.get(k, 0)) for k in pairs]
    df["user_activity"] = df["user_id"].map(user_activity_lookup).fillna(0).astype(int)
    if "category_L3" not in df.columns:
        df["category_L3"] = df["product_id"].map(prod_cat3_map).fillna("unknown")
    df["cat3_match"] = (df["category_L3"].astype(str) ==
                        df["user_id"].map(user_top_cat3).fillna("unknown").astype(str)).astype(int)
    if "price_tier" not in df.columns:
        df["price_tier"] = df["product_id"].map(prod_tier_map).fillna("medium")
    df["tier_match"] = (df["price_tier"].astype(str) ==
                        df["user_id"].map(user_top_tier).fillna("medium").astype(str)).astype(int)
    now = datetime.now()
    df["hour_bin"]    = _hour_bin(now.hour)
    df["dow_weekend"] = int(now.weekday() >= 5)
    # v7a: pid_popularity / user_cat3_view (phase2_deepfm.py 와 동일 계산)
    df["pid_popularity"] = (
        df["product_id"].map(pid_global_count).fillna(0).astype(int).map(_pop_bin)
    )
    uc_pairs = list(zip(df["user_id"].values, df["category_L3"].astype(str).values))
    df["user_cat3_view"] = [_cat3_view_bin(user_cat3_count.get(k, 0)) for k in uc_pairs]
    return df


def encode_for_deepfm(df: pd.DataFrame) -> np.ndarray:
    df = _add_cross_context(df)
    X = np.zeros((len(df), len(SPARSE_FEATURES)), dtype=int)
    for i, f in enumerate(SPARSE_FEATURES):
        if f not in df.columns:
            continue  # 컬럼 부재 → 전부 OOV(인덱스 0) 유지
        # 사전계산 dict 로 매핑, 미등록(OOV) 값은 0 (기존 np.isin 방식과 동일 의미)
        X[:, i] = df[f].astype(str).map(encoder_maps[f]).fillna(0).astype(int).to_numpy()
    return X


def get_session_recent_pids(user_id: str) -> list:
    if redis_client is None:
        return []
    raw = redis_client.get(f"session:{user_id}")
    return json.loads(raw) if raw else []


def diversity_rerank(ranked_pids: list, max_run: int = MAX_SAME_CATEGORY_RUN) -> list:
    """
    동일 카테고리 연속 3개 이상 금지 (= 같은 카테고리 연속 2개까지 허용).
    탐욕적으로 다음 후보를 골라 카테고리 run 제약을 만족시킴.
    """
    out, run_cat, run_count = [], None, 0
    pool = list(ranked_pids)

    while pool and len(out) < len(ranked_pids):
        picked_idx = None
        for i, pid in enumerate(pool):
            cat = prod_meta_dict.get(pid, {}).get("category_L1")
            if cat == run_cat and run_count >= max_run:
                continue
            picked_idx = i
            break

        if picked_idx is None:
            picked_idx = 0  # 제약 풀고 그냥 픽

        pid = pool.pop(picked_idx)
        cat = prod_meta_dict.get(pid, {}).get("category_L1")
        if cat == run_cat:
            run_count += 1
        else:
            run_cat, run_count = cat, 1
        out.append(pid)

    return out


# ──────────────────────────────────────────────
# Pydantic
# ──────────────────────────────────────────────
class LogEvent(BaseModel):
    user_id: str
    product_id: str
    event_type: Literal["search", "view", "cart", "purchase"]
    timestamp: int


# ══════════════════════════════════════════════
# /api/search  (멀티모달 검색, 명세 필수 필드 준수)
# ══════════════════════════════════════════════
@app.post("/api/search")
async def personalized_search(
    user_id: str = Form("U000058a12d"),
    query: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    top_k: int = Form(10),
):
    t0 = time.perf_counter()
    if not query and not file:
        return {"error": "query 또는 file 중 하나는 필수입니다."}

    # Query Understanding(보너스#1): 가격/색상/카테고리 필터 추출. 가격 문구는 검색어에서 제거
    # ('5만원'의 '5'가 임베딩/매칭을 오염시키는 것 방지). 가격·색상은 결과 하드필터로 적용.
    qu_filters   = ko_fashion.extract_filters(query) if query else {}
    search_query = ko_fashion.strip_price(query) if query else query

    # 임베딩 생성
    text_emb, image_emb = None, None

    if query:
        # CLIP Text Encoder 로 쿼리 임베딩(명세 line 134). text.index 는 영문 product_name 기반.
        # project.md line 403(외부 클라우드 API 금지) 준수 — 외부 번역기 없이 로컬 인코딩만.
        tok = clip_processor.tokenizer(search_query, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        with torch.no_grad():
            tf = clip_model.get_text_features(input_ids=tok["input_ids"], attention_mask=tok["attention_mask"])
            if not isinstance(tf, torch.Tensor):
                pooler = getattr(tf, "pooler_output", None)
                embeds = getattr(tf, "text_embeds", None)
                tf = pooler if pooler is not None else embeds
            text_emb = tf / tf.norm(dim=-1, keepdim=True)

    if file:
        img_bytes = await file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        proc = clip_processor(images=[img], return_tensors="pt").to(device)
        with torch.no_grad():
            img_f = clip_model.get_image_features(pixel_values=proc["pixel_values"])
            if not isinstance(img_f, torch.Tensor):
                pooler = getattr(img_f, "pooler_output", None)
                embeds = getattr(img_f, "image_embeds", None)
                img_f = pooler if pooler is not None else embeds
            image_emb = img_f / img_f.norm(dim=-1, keepdim=True)

    user_info = df_users[df_users["user_id"] == user_id]
    persona = user_info["persona"].values[0] if not user_info.empty else "신규가입자"

    # 정규화 임베딩 → squared L2 거리를 코사인 유사도로 변환 (cos = 1 - d/2)
    def _cos(d):
        return max(0.0, 1.0 - float(d) / 2.0)

    def _to_results(idx_list, score_list):
        out = []
        for idx, sc in zip(idx_list, score_list):
            if int(idx) < 0:
                continue
            prod = df_prods.iloc[int(idx)]
            out.append({
                "product_id": prod["product_id"],
                "name":       prod["product_name"],
                "score":      round(float(sc), 4),
                "price":      int(prod["price"]),
            })
        return out

    # Query Understanding 하드필터: 가격(price 컬럼)·색상(상품명 명시)로 결과를 거른다.
    # 카테고리는 retrieval 이 처리하므로 표시만(부분일치 하드필터는 취약). 전부 걸러지면 원본 폴백.
    def _passes(idx) -> bool:
        prod = df_prods.iloc[int(idx)]
        price = int(prod["price"])
        if "price_max" in qu_filters and price > qu_filters["price_max"]:
            return False
        if "price_min" in qu_filters and price < qu_filters["price_min"]:
            return False
        color = qu_filters.get("color")
        if color and color not in str(prod["product_name"]).lower():
            return False
        return True

    def _apply_filters(ranked_full):
        if not qu_filters:
            return ranked_full
        kept = [(i, s) for i, s in ranked_full if _passes(i)]
        return kept if kept else ranked_full

    # M-CLIP 텍스트→이미지 회수(한/영 의미): search_query 를 image.index 에서 검색해 idx 순위 반환.
    def _mclip_rank(pool):
        if mclip_enc is None or not query:
            return []
        _, mi = image_faiss.search(mclip_enc.encode_query(search_query), pool)
        return [int(i) for i in mi[0] if int(i) >= 0]

    RRF_K, SCORE_SCALE = 60, 30   # RRF 융합 상수 / 표시용 0~1 환산

    # FAISS 유사도 검색 — 검색은 "유사 상품 찾기" 이므로 유사도 순으로 정렬한다.
    if query and file:
        # 하이브리드: CLIP text.index + CLIP image.index + M-CLIP(한/영 의미)를 RRF 융합.
        # 순위 기반 RRF 라 모달 간 스케일 차이에 무관하게 균형 결합된다.
        search_type = "hybrid"
        K = max(top_k * 20, 200)
        _, idx_i = image_faiss.search(image_emb.cpu().numpy().astype("float32"), K)
        fused = {}
        if not ko_fashion.has_korean(query or ""):   # 한글은 CLIP text↔text 노이즈 → 제외
            _, idx_t = text_faiss.search(text_emb.cpu().numpy().astype("float32"), K)
            for rank, idx in enumerate(idx_t[0]):
                if int(idx) < 0:
                    continue
                fused[int(idx)] = fused.get(int(idx), 0.0) + HYBRID_W_TEXT / (RRF_K + rank)
        for rank, idx in enumerate(idx_i[0]):
            if int(idx) < 0:
                continue
            fused[int(idx)] = fused.get(int(idx), 0.0) + HYBRID_W_IMAGE / (RRF_K + rank)
        for rank, idx in enumerate(_mclip_rank(K)):
            fused[idx] = fused.get(idx, 0.0) + HYBRID_W_TEXT / (RRF_K + rank)
        ranked = _apply_filters(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))[:top_k]
        results = _to_results([i for i, _ in ranked], [min(1.0, s * SCORE_SCALE) for _, s in ranked])
    elif query:
        # 텍스트 전용: CLIP text.index(어휘/정확매칭) + M-CLIP text→image(한/영 의미) RRF 융합.
        # 한글 쿼리는 CLIP text↔text 가 약하므로 M-CLIP 이 전담, 영어는 두 신호가 보강된다.
        search_type = "text"
        K = max(top_k * 20, 200)
        fused = {}
        if not ko_fashion.has_korean(query or ""):   # 한글은 CLIP text↔text 노이즈 → M-CLIP 전담
            _, idx_t = text_faiss.search(text_emb.cpu().numpy().astype("float32"), K)
            for rank, idx in enumerate(idx_t[0]):
                if int(idx) < 0:
                    continue
                fused[int(idx)] = fused.get(int(idx), 0.0) + 1.0 / (RRF_K + rank)
        for rank, idx in enumerate(_mclip_rank(K)):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (RRF_K + rank)
        ranked = _apply_filters(sorted(fused.items(), key=lambda kv: kv[1], reverse=True))[:top_k]
        results = _to_results([i for i, _ in ranked], [min(1.0, s * SCORE_SCALE) for _, s in ranked])
    else:
        # 이미지 전용: CLIP image.index
        search_type = "image"
        distances, indices = image_faiss.search(image_emb.cpu().numpy().astype("float32"), top_k)
        results = _to_results(indices[0], [_cos(d) for d in distances[0]])

    latency = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "search_type": search_type,
        "results": results,
        "latency_ms": latency,
        "total_count": len(results),
        "user_id": user_id,
        "persona": persona,
        "original_query": query,
        "parsed_filters": qu_filters,   # Query Understanding(보너스#1): 추출된 가격/색상/카테고리
    }


# ══════════════════════════════════════════════
# /api/log  (세션 추적)
# ══════════════════════════════════════════════
@app.post("/api/log")
async def receive_log(log: LogEvent):
    if redis_client is None:
        return {"message": "redis 미연결", "event": log.event_type}

    key = f"session:{log.user_id}"
    raw = redis_client.get(key)
    seq = json.loads(raw) if raw else []

    if log.event_type in ["search", "view", "cart", "purchase"]:
        if log.product_id in seq:
            seq.remove(log.product_id)
        seq.insert(0, log.product_id)
        seq = seq[:SESSION_LEN]
        redis_client.set(key, json.dumps(seq), ex=3600)

    # 클릭 횟수 카운트 (실시간 피처)
    redis_client.incr(f"click_count:{log.user_id}")
    redis_client.expire(f"click_count:{log.user_id}", 3600)

    return {"message": "ok", "event": log.event_type}


# ══════════════════════════════════════════════
# /api/recommend  (진짜 Multi-Stage)
# ══════════════════════════════════════════════
@app.get("/api/recommend")
async def recommend(
    user_id: str = Query("U000058a12d"),
    top_n: int = Query(10),
):
    t_total = time.perf_counter()

    # 컨텍스트 수집
    user_info  = df_users[df_users["user_id"] == user_id]
    persona    = user_info["persona"].values[0] if not user_info.empty else None
    recent_pids = get_session_recent_pids(user_id)
    history_count = len(recent_pids)

    # 신규 유저 판정 (명세 line 175: 행동 이력 5개 미만).
    # 학습된 유저(Two-Tower vocab 에 존재)는 시뮬레이터가 최소 5개 이상 행동을 부여하므로 신규가 아니다.
    # 미학습 user_id 이면서 세션 클릭도 5개 미만일 때만 콜드스타트로 본다.
    is_known_user = user_id in user_idx_map
    cold_start = (not is_known_user) and (history_count < NEW_USER_HISTORY_THRESHOLD)

    session_interest = None
    if recent_pids:
        cats = [prod_meta_dict.get(p, {}).get("category_L1") for p in recent_pids]
        cats = [c for c in cats if c]
        if cats:
            session_interest = max(set(cats), key=cats.count)

    # ──────── Stage 1: Candidate Generation ────────
    t_s1 = time.perf_counter()
    candidate_pids = []

    can_use_two_tower = (two_tower is not None) and (user_id in user_idx_map)
    if can_use_two_tower:
        u_pos = int(user_idx_map[user_id])
        u_idx = torch.tensor([u_pos], dtype=torch.long).to(device)
        p_idx = torch.tensor([user_persona_map[user_id]], dtype=torch.long).to(device)
        a_idx = torch.tensor([int(user_activity_map.get(user_id, 0))], dtype=torch.long).to(device)
        if user_cat1_dist_arr is not None:
            cat1_d = torch.from_numpy(user_cat1_dist_arr[u_pos:u_pos+1]).to(device)
            tier_d = torch.from_numpy(user_tier_dist_arr[u_pos:u_pos+1]).to(device)
        else:
            cat1_d = torch.zeros(1, two_tower.cat1_emb.num_embeddings, device=device)
            tier_d = torch.zeros(1, two_tower.tier_emb.num_embeddings, device=device)

        with torch.no_grad():
            u_vec = two_tower.encode_user(u_idx, p_idx, cat1_d, tier_d, a_idx)  # (1, D)

            # 단기 관심(세션) ⊕ 장기 선호(User Tower)
            if session_encoder is not None and recent_pids:
                seq_idx = [prod_idx_map[p] for p in recent_pids if p in prod_idx_map]
                if seq_idx:
                    seq_t = torch.tensor([seq_idx], dtype=torch.long).to(device)
                    s_vec = session_encoder(seq_t)  # (1, D)
                    u_vec = F.normalize(0.6 * u_vec + 0.4 * s_vec, p=2, dim=-1)

        u_np = u_vec.cpu().numpy().astype("float32")
        faiss.normalize_L2(u_np)
        _, cand_ids = candidate_faiss.search(u_np, CANDIDATE_K)
        candidate_pids = [prod_idx_to_id[i] for i in cand_ids[0] if i in prod_idx_to_id]

    # 콜드스타트 또는 Two-Tower 사용 불가 시: 인기 + 트렌딩으로 폴백
    if cold_start or not candidate_pids:
        fallback = list(dict.fromkeys(popular_pids + trending_pids))[:CANDIDATE_K]
        candidate_pids = fallback if not candidate_pids else candidate_pids
        # 부족하면 채우기
        if len(candidate_pids) < CANDIDATE_K:
            for p in fallback:
                if p not in candidate_pids:
                    candidate_pids.append(p)
                if len(candidate_pids) >= CANDIDATE_K:
                    break

    candidate_ms = round((time.perf_counter() - t_s1) * 1000, 1)

    # ──────── Stage 2: Ranking (DeepFM) ────────
    t_s2 = time.perf_counter()
    cand_df = pd.DataFrame({"product_id": candidate_pids})
    cand_df["user_id"] = user_id
    cand_df["persona"] = persona if persona else "신중탐색"
    cand_df = cand_df.merge(
        df_prods[["product_id", "category_L1", "price_tier"]],
        on="product_id", how="left"
    )
    cand_df["category_L1"] = cand_df["category_L1"].fillna("Ladieswear")
    cand_df["price_tier"]  = cand_df["price_tier"].fillna("medium")

    X_p = encode_for_deepfm(cand_df)
    with torch.no_grad():
        ranking_scores = deepfm(torch.tensor(X_p, dtype=torch.long).to(device)).cpu().numpy()
    cand_df["score"] = ranking_scores

    ranked = cand_df.sort_values("score", ascending=False).head(RANK_TOP_N)
    ranking_ms = round((time.perf_counter() - t_s2) * 1000, 1)

    # ──────── Stage 3: Re-ranking (다양성 + 신규상품 + MAB) ────────
    t_s3 = time.perf_counter()

    ranked_pids = ranked["product_id"].tolist()
    score_map = dict(zip(ranked["product_id"], ranked["score"]))

    # 1) 다양성 (동일 카테고리 연속 3개 금지)
    diverse_pids = diversity_rerank(ranked_pids)

    # 2) Top-N 슬롯 구성: exploitation + 신규상품 보장 슬롯 + MAB 탐색 슬롯
    new_pids_set = set(new_pids)

    # 신규상품 강제 노출 (명세 line 185): 등록 7일 이내 상품 1개를 슬롯으로 보장한다.
    # 신규상품은 행동 이력이 없어 Two-Tower 후보에 거의 안 잡히므로,
    # 후보 의존(기존 boost)이 아니라 강제 주입으로 노출 기회를 '보장' 한다.
    new_slot = []
    if new_pids and not cold_start:
        exploit_cats = {prod_meta_dict.get(p, {}).get("category_L1") for p in diverse_pids[:top_n]}
        new_cand = [p for p in new_pids if p not in diverse_pids]
        relevant = [p for p in new_cand if prod_meta_dict.get(p, {}).get("category_L1") in exploit_cats]
        pool = relevant if relevant else new_cand
        if pool:
            new_slot = [random.choice(pool)]

    # 슬롯 예산: 신규상품(0~1) + MAB(1) 만큼 exploitation 에서 차감
    reserve = len(new_slot) + 1
    final_pids = diverse_pids[:max(top_n - reserve, 1)]

    # MAB 탐색 슬롯 (Epsilon-Greedy)
    used = set(final_pids) | set(new_slot)
    explore_pool = [p for p in popular_pids + trending_pids if p not in used]
    random.shuffle(explore_pool)
    mab_count = max(1, top_n - len(final_pids) - len(new_slot))
    explore_picks = explore_pool[:mab_count]

    # 결과 조립: exploitation → 신규상품 슬롯 → MAB 슬롯
    final_recs = []
    for pid in final_pids:
        if cold_start:
            reason = "popular_fallback"
        elif pid in new_pids_set:
            reason = "new_product"
        elif session_interest and prod_meta_dict.get(pid, {}).get("category_L1") == session_interest:
            reason = "session_interest"
        else:
            reason = "personalized_deepfm"
        final_recs.append({
            "product_id": pid,
            "score": float(score_map.get(pid, 0.0)),
            "reason": reason,
            "is_exploration": False,
        })

    for pid in new_slot:
        final_recs.append({
            "product_id": pid,
            "score": 0.0,
            "reason": "new_product",
            "is_exploration": False,
        })

    for pid in explore_picks:
        final_recs.append({
            "product_id": pid,
            "score": 0.0,
            "reason": "mab_exploration",
            "is_exploration": True,
        })

    final_recs = final_recs[:top_n]

    reranking_ms = round((time.perf_counter() - t_s3) * 1000, 1)
    total_ms = round((time.perf_counter() - t_total) * 1000, 1)

    return {
        "user_id": user_id,
        "persona": persona,
        "recommendations": final_recs,
        "pipeline_latency": {
            "candidate_ms":  candidate_ms,
            "ranking_ms":    ranking_ms,
            "reranking_ms":  reranking_ms,
            "total_ms":      total_ms,
        },
        "session_context": {
            "recent_clicks":     recent_pids,
            "recent_clicked_item_ids": recent_pids,  # 호환
            "session_interest":  session_interest,
            "history_count":     history_count,
            "is_cold_start":     cold_start,
        },
    }


# ══════════════════════════════════════════════
# /api/reload-model  (CT 후 무중단 갱신)
# ══════════════════════════════════════════════
@app.post("/api/reload-model")
async def reload_model():
    try:
        deepfm.load_state_dict(torch.load(DEEPFM_PATH, map_location=device))
        deepfm.eval()
        return {"message": "✅ DeepFM 가중치 갱신 완료"}
    except Exception as e:
        return {"error": str(e)}


# ══════════════════════════════════════════════
# /api/health  (간단 헬스 체크)
# ══════════════════════════════════════════════
@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "two_tower_loaded": two_tower is not None,
        "deepfm_loaded": deepfm is not None,
        "redis_connected": redis_client is not None,
        "n_products": len(df_prods),
        "n_users": len(df_users),
    }


# ══════════════════════════════════════════════
# 데모 지원: 상품 이미지 서빙 + 비교 데모 페이지
# ══════════════════════════════════════════════
_TRANSPARENT_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4"
    "890000000a49444154789c6360000002000154a24f1f0000000049454e44ae426082"
)


@app.get("/images/{product_id}")
async def product_image(product_id: str):
    """상품 이미지 원본을 서빙한다. P0108775015 → data/raw/images/010/0108775015.jpg.
    파일이 없으면 1x1 투명 PNG 를 반환해 데모 그리드가 깨지지 않게 한다."""
    digits = product_id[1:] if product_id.startswith("P") else product_id
    folder = digits[:3]
    path = os.path.join(IMAGES_DIR, folder, f"{digits}.jpg")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg")
    return Response(content=_TRANSPARENT_PNG, media_type="image/png")


@app.get("/api/products")
async def products(ids: str = Query("")):
    """데모 지원용 — product_id 목록을 받아 이름/가격/카테고리를 돌려준다.
    (스펙 평가 대상인 /api/recommend 응답 스키마를 건드리지 않기 위한 보조 엔드포인트)"""
    out = []
    for pid in [x for x in ids.split(",") if x]:
        m = prod_meta_dict.get(pid)
        if m:
            out.append({
                "product_id":  pid,
                "name":        m.get("product_name"),
                "price":       int(m.get("price", 0)),
                "category_L1": m.get("category_L1"),
                "category_L3": m.get("category_L3"),
            })
    return {"products": out}


@app.get("/api/categories")
async def categories():
    """데모 지원용 — category_L1 목록과 상품 수를 돌려준다. (랜딩 페이지 카탈로그 탭용)"""
    vc = df_prods["category_L1"].value_counts()
    return {"categories": [{"name": str(k), "count": int(v)} for k, v in vc.items()]}


@app.get("/api/catalog")
async def catalog(
    category: str = Query(""),
    page: int = Query(1),
    page_size: int = Query(24),
    sort: Literal["popular", "price_asc", "price_desc", "random"] = Query("popular"),
):
    """데모 지원용 — 카테고리별 상품 목록(페이지네이션)을 돌려준다.
    추천/검색이 아닌 '카탈로그 브라우징' 용도. 평가 대상 스키마(/api/recommend)는 건드리지 않는다."""
    df = df_prods
    if category:
        df = df[df["category_L1"].astype(str) == category]

    if sort == "popular":
        order = df["product_id"].map(pid_global_count).fillna(0)
        df = df.assign(_pop=order).sort_values("_pop", ascending=False)
    elif sort == "price_asc":
        df = df.sort_values("price", ascending=True)
    elif sort == "price_desc":
        df = df.sort_values("price", ascending=False)
    elif sort == "random":
        df = df.sample(frac=1.0, random_state=random.randint(0, 1_000_000))

    total = len(df)
    start = max(page - 1, 0) * page_size
    page_df = df.iloc[start:start + page_size]

    out = []
    for _, prod in page_df.iterrows():
        out.append({
            "product_id":  prod["product_id"],
            "name":        prod["product_name"],
            "price":       int(prod["price"]),
            "price_tier":  prod["price_tier"],
            "category_L1": prod["category_L1"],
            "category_L3": prod["category_L3"],
        })
    return {
        "category": category or "all",
        "sort": sort,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_more": start + page_size < total,
        "products": out,
    }


@app.get("/api/user-history")
async def user_history(user_id: str = Query("U000058a12d"), n: int = Query(8)):
    """데모 지원용 — 해당 사용자가 과거에 담거나 구매한 상품(최근 n개)을 돌려준다.
    '같은 페르소나라도 개인 이력에 따라 추천이 달라진다'를 보여주기 위한 근거 표시용."""
    df = df_logs[(df_logs["user_id"] == user_id) &
                 (df_logs["event_type"].isin(["cart", "purchase"]))]
    pids = df.sort_values("timestamp")["product_id"].tolist() if "timestamp" in df.columns \
        else df["product_id"].tolist()
    seen, uniq = set(), []
    for p in reversed(pids):  # 최근 것부터, 중복 제거
        if p not in seen:
            seen.add(p); uniq.append(p)
        if len(uniq) >= n:
            break
    out = []
    for pid in uniq:
        m = prod_meta_dict.get(pid)
        if m:
            out.append({"product_id": pid, "name": m.get("product_name"),
                        "price": int(m.get("price", 0)), "category_L3": m.get("category_L3")})
    return {"user_id": user_id, "purchase_count": int(len(df)), "history": out}


@app.get("/", response_class=HTMLResponse)
async def demo_page():
    """발표용 라이브 비교 데모 페이지. 같은 출처에서 /api/* 를 호출한다."""
    if os.path.exists(DEMO_HTML):
        with open(DEMO_HTML, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>demo.html 이 없습니다.</h1>", status_code=404)