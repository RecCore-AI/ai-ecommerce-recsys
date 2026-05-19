"""
prepare_hm_data.py  v2 — 강화된 페르소나/유저 시뮬레이터
─────────────────────────────────────────────────────
기존 문제 (진단 결과):
  - purchase_logs 가 H&M 원본 거래에서 와서 페르소나/유저 신호 없음
  - 페르소나 간 cat_L3 JS divergence 0.015~0.21 (목표 0.3+)
  - 같은 유저 train↔test cat_L3 JS 0.406 vs random 0.450 (신호 9.8%)
  - product_id 재구매율 1.4%

해결:
  - 페르소나마다 narrow 한 cat_L3 + price_tier 조합 부여 (config.yaml)
  - 유저마다 그 안에서 1~2 카테고리 + 1 tier 픽 → narrow product pool (60~200개)
  - pool 의 15% 는 "core favorites" → 재구매로 product-id 신호 주입
  - 모든 행동 이벤트는 user_pool 에서 샘플 (train/test 일관성 확보)
  - 노이즈 10% 만 user_pool 밖에서 추출
"""
import os
import yaml
import pandas as pd
import numpy as np

print("[START] H&M + 강화 시뮬레이터 v2 시작...")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
rng = np.random.default_rng(RANDOM_SEED)

DATA_DIR = "data"
RAW_DIR  = "data/raw"
ARTICLES_PATH      = f"{RAW_DIR}/articles.csv"
CUSTOMERS_PATH     = f"{RAW_DIR}/customers.csv"
TRANSACTIONS_PATH  = f"{RAW_DIR}/transactions_train.csv"
CONFIG_PATH        = "config.yaml"

# ── 시뮬레이션 파라미터 ──
N_USERS  = 10000
TIME_START = pd.Timestamp("2024-01-01")
TIME_END   = pd.Timestamp("2024-12-31")

# 유저 활동량 버킷 (구매 횟수 범위)
ACTIVITY_RANGES = {
    "low":  (5,   20),
    "mid":  (20,  60),
    "high": (60, 200),
}
ACTIVITY_RATIOS = [0.30, 0.40, 0.30]
ACTIVITY_NAMES  = ["low", "mid", "high"]

# 유저 풀
POOL_MIN = 60
POOL_MAX = 200
CORE_RATIO = 0.25   # v8a: pool 의 25% (페르소나별 p_core 가 핵심 신호)

# v11: Zipf preference (purchase sampling 시 user 내부 product rank 명확화)
# 목적: candidate 300 안에서 정답 product 의 user_pid_prev 가 다른 product 대비 두드러지게
# core 안 alpha=1.1 → top-3 core 가 core 전체 행동의 ~55% 차지 (현실적 단골 패턴)
# pool 안 alpha=0.5 → 약한 기울기 (다양성 유지)
ZIPF_CORE_ALPHA = 1.1
ZIPF_POOL_ALPHA = 0.5

# Purchase 샘플링 가중치 (페르소나별 p_core 가 config 에 있으면 override)
P_CORE_DEFAULT  = 0.40   # 페르소나 정의 없을 때 fallback
P_NOISE = 0.10           # 10% : 전체 카탈로그 random (페르소나 공통)
# pool 비중 = 1 - p_core - P_NOISE  (페르소나별 동적 계산)

# 부정 퍼널 비율
BOUNCED_VIEW_RATIO   = 3.0  # v8a 설정 복원 (2.0 은 시퀀스 학습 신호 부족)
ABANDONED_CART_RATIO = 0.6
# bounced/abandoned 의 user_pool 외부 비율
NEG_NOISE_RATIO = 0.10

DEFAULT_PERSONA_RATIOS = {
    "트렌드세터": 0.15,
    "실용주의자": 0.25,
    "가성비추구": 0.25,
    "브랜드충성": 0.15,
    "충동구매":   0.10,
    "신중탐색":   0.10,
}
DEFAULT_PERSONA_CONFIGS = {
    "트렌드세터": {"cat_L3_pool": ["Garment Upper body"], "price_tiers": ["high"],
                "search_prob": 0.6, "view_to_cart": 0.15, "cart_to_purchase": 0.40, "p_core": 0.25},
    "실용주의자": {"cat_L3_pool": ["Underwear", "Socks & Tights"], "price_tiers": ["low","medium"],
                "search_prob": 0.5, "view_to_cart": 0.08, "cart_to_purchase": 0.50, "p_core": 0.55},
    "가성비추구": {"cat_L3_pool": ["Garment Upper body"], "price_tiers": ["low"],
                "search_prob": 0.7, "view_to_cart": 0.10, "cart_to_purchase": 0.50, "p_core": 0.40},
    "브랜드충성": {"cat_L3_pool": ["Garment Full body", "Garment Lower body"], "price_tiers": ["high"],
                "search_prob": 0.3, "view_to_cart": 0.12, "cart_to_purchase": 0.45, "p_core": 0.55},
    "충동구매":   {"cat_L3_pool": ["Accessories", "Swimwear"], "price_tiers": ["low","medium"],
                "search_prob": 0.2, "view_to_cart": 0.20, "cart_to_purchase": 0.30, "p_core": 0.25},
    "신중탐색":   {"cat_L3_pool": ["Shoes", "Nightwear"], "price_tiers": ["high","medium"],
                "search_prob": 0.8, "view_to_cart": 0.05, "cart_to_purchase": 0.20, "p_core": 0.35},
}


def load_config() -> tuple:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        ratios = cfg.get("persona_ratios", DEFAULT_PERSONA_RATIOS)
        raw_configs = cfg.get("persona_configs", {})
        configs = {}
        for persona, default in DEFAULT_PERSONA_CONFIGS.items():
            entry = raw_configs.get(persona, {})
            configs[persona] = {
                "cat_L3_pool":     entry.get("cat_L3_pool",     default["cat_L3_pool"]),
                "price_tiers":     entry.get("price_tiers",     default["price_tiers"]),
                "search_prob":     entry.get("search_prob",     default["search_prob"]),
                "view_to_cart":    entry.get("view_to_cart",    default["view_to_cart"]),
                "cart_to_purchase":entry.get("cart_to_purchase",default["cart_to_purchase"]),
                "p_core":          entry.get("p_core",          default.get("p_core", P_CORE_DEFAULT)),
            }
        print("config.yaml 로드 완료")
        return ratios, configs
    print("config.yaml 없음 → default 사용")
    return DEFAULT_PERSONA_RATIOS, DEFAULT_PERSONA_CONFIGS


def random_timestamps(n: int, start=TIME_START, end=TIME_END) -> pd.DatetimeIndex:
    if n <= 0:
        return pd.DatetimeIndex([])
    span = (end - start).total_seconds()
    sec  = rng.uniform(0, span, size=n)
    return pd.to_datetime(start) + pd.to_timedelta(sec, unit="s")


PERSONA_RATIOS, PERSONA_CONFIGS = load_config()
PERSONAS = list(PERSONA_RATIOS.keys())
PERSONA_PROBS = np.array([float(PERSONA_RATIOS[p]) for p in PERSONAS])
PERSONA_PROBS /= PERSONA_PROBS.sum()

# ───────────────────────────────────────────────
# 1-5. 상품 메타 (기존 로직 유지)
# ───────────────────────────────────────────────
print("1. H&M 원본 로드...")
articles  = pd.read_csv(ARTICLES_PATH, dtype={"article_id": str})
customers = pd.read_csv(CUSTOMERS_PATH)
trans     = pd.read_csv(TRANSACTIONS_PATH, dtype={"article_id": str})

print("2. 가격 산출...")
trans["real_price"] = trans["price"] * 1_000_000
item_avg_price = trans.groupby("article_id")["real_price"].mean().reset_index()

print("3. 상위 50K 상품 선정...")
user_counts = trans["customer_id"].value_counts()
active_customers = user_counts[user_counts >= 5].index
trans_active = trans[trans["customer_id"].isin(active_customers)].copy()
item_counts = trans_active["article_id"].value_counts()
top_items = item_counts.head(50000).index
final_articles = articles[articles["article_id"].isin(top_items)].copy()

print("4. price_tier 부여...")
final_articles = final_articles.merge(item_avg_price, on="article_id", how="left")
final_articles["real_price"] = final_articles["real_price"].fillna(final_articles["real_price"].median())
final_articles["price_tier"] = pd.qcut(
    final_articles["real_price"], q=3, labels=["low","medium","high"], duplicates="drop"
).astype(str).fillna("medium")

print("5. products.csv 작성...")
products = pd.DataFrame({
    "product_id": "P" + final_articles["article_id"],
    "category_L1": final_articles["index_group_name"].fillna("unknown"),
    "category_L2": final_articles["index_name"].fillna("unknown"),
    "category_L3": final_articles["product_group_name"].fillna("unknown"),
    "product_name": (
        final_articles["prod_name"].fillna("") + " - "
        + final_articles["colour_group_name"].fillna("") + " ("
        + final_articles["detail_desc"].fillna("") + ")"
    ),
    "price": final_articles["real_price"].astype(int),
    "price_tier": final_articles["price_tier"],
}).reset_index(drop=True)
products.to_csv(f"{DATA_DIR}/products.csv", index=False)
print(f"  products.csv: {len(products):,}개")

# ───────────────────────────────────────────────
# 6. 유저 생성 + 페르소나 할당
# ───────────────────────────────────────────────
print("6. users.csv 작성...")
trans_top = trans_active[trans_active["article_id"].isin(top_items)]
sampled_uids = pd.Series(trans_top["customer_id"].unique()).head(N_USERS)
users = pd.DataFrame({
    "user_id": "U" + sampled_uids.str[:10],
    "persona": rng.choice(PERSONAS, size=len(sampled_uids), p=PERSONA_PROBS),
})
users = users.drop_duplicates("user_id").reset_index(drop=True)
users.to_csv(f"{DATA_DIR}/users.csv", index=False)
print(f"  users.csv: {len(users):,}명")
print(users["persona"].value_counts().to_dict())

# ───────────────────────────────────────────────
# 7. 페르소나 → 실제 cat_L3 풀 resolve + 유저별 narrow pool 구축
# ───────────────────────────────────────────────
print("7. 유저별 narrow product pool 구축...")
all_cat_L3 = set(products["category_L3"].unique())
all_pids   = products["product_id"].values

persona_resolved = {}
for persona, cfg in PERSONA_CONFIGS.items():
    cats  = [c for c in cfg["cat_L3_pool"] if c in all_cat_L3]
    if not cats:
        cats = list(products["category_L3"].value_counts().head(2).index)
    tiers = [t for t in cfg["price_tiers"] if t in ["low","medium","high"]]
    if not tiers:
        tiers = ["medium"]
    pool_size = len(products[products["category_L3"].isin(cats) & products["price_tier"].isin(tiers)])
    persona_resolved[persona] = {"cats": cats, "tiers": tiers}
    print(f"  {persona:8s}  cats={cats}  tiers={tiers}  풀={pool_size:,}")

# 유저 활동량 할당
activity_choices = rng.choice(ACTIVITY_NAMES, size=len(users), p=ACTIVITY_RATIOS)

# 유저별 profile (cats, tier, pool, core, activity, persona)
user_profiles = {}
prods_by_cat_tier = {}  # cache (cat_L3, tier) → array of pids
for cat in all_cat_L3:
    for tier in ["low","medium","high"]:
        mask = (products["category_L3"]==cat) & (products["price_tier"]==tier)
        prods_by_cat_tier[(cat, tier)] = products.loc[mask, "product_id"].values

for i, (uid, persona, activity) in enumerate(zip(users["user_id"], users["persona"], activity_choices)):
    res = persona_resolved[persona]
    # 유저는 페르소나의 cat 중 1개 (40% 1개, 60% 가능하면 1-2개) 픽
    n_cats = 1 if len(res["cats"])==1 else rng.choice([1, 2], p=[0.5, 0.5])
    user_cats = list(rng.choice(res["cats"], size=min(n_cats, len(res["cats"])), replace=False))
    user_tier = rng.choice(res["tiers"])

    # 유저 풀 = (user_cats x user_tier) 조합 상품
    pool_chunks = [prods_by_cat_tier.get((c, user_tier), np.array([])) for c in user_cats]
    pool_pids   = np.concatenate(pool_chunks) if pool_chunks else np.array([])

    # 풀 크기 정규화
    if len(pool_pids) > POOL_MAX:
        idx = rng.choice(len(pool_pids), size=POOL_MAX, replace=False)
        pool_pids = pool_pids[idx]
    elif len(pool_pids) < POOL_MIN:
        # 페르소나 전체 cat x tier 조합 풀에서 보충
        broad_chunks = [
            prods_by_cat_tier.get((c, t), np.array([]))
            for c in res["cats"] for t in res["tiers"]
        ]
        broad_pids = np.concatenate(broad_chunks) if broad_chunks else np.array([])
        if len(broad_pids) >= POOL_MIN:
            need = min(POOL_MAX, len(broad_pids))
            idx = rng.choice(len(broad_pids), size=need, replace=False)
            pool_pids = broad_pids[idx]

    # 마지막 안전망: 전체 카탈로그에서 random
    if len(pool_pids) < POOL_MIN:
        pool_pids = rng.choice(all_pids, size=POOL_MIN, replace=False)

    # core favorites (15%)
    n_core = max(5, int(len(pool_pids) * CORE_RATIO))
    core_idx = rng.choice(len(pool_pids), size=min(n_core, len(pool_pids)), replace=False)
    core_pids = pool_pids[core_idx]

    # v11: Zipf preference weights — core 와 pool 의 product 순서가 곧 preference rank
    # core/pool 의 product 가 sample 될 때 1/(rank^alpha) 확률로 추출됨
    # core_pids[0] 이 user 의 #1 favorite, core_pids[1] 이 #2 favorite ... 식
    core_ranks = np.arange(1, len(core_pids) + 1, dtype=np.float64)
    core_weights = 1.0 / (core_ranks ** ZIPF_CORE_ALPHA)
    core_weights /= core_weights.sum()

    pool_ranks = np.arange(1, len(pool_pids) + 1, dtype=np.float64)
    pool_weights = 1.0 / (pool_ranks ** ZIPF_POOL_ALPHA)
    pool_weights /= pool_weights.sum()

    user_profiles[uid] = {
        "persona":  persona,
        "cats":     user_cats,
        "tier":     user_tier,
        "pool":     pool_pids,
        "core":     core_pids,
        "core_weights": core_weights,   # v11
        "pool_weights": pool_weights,   # v11
        "activity": activity,
        # v8a: 페르소나별 P_CORE (config 에서 페르소나별 차별화)
        "p_core":   float(PERSONA_CONFIGS[persona].get("p_core", P_CORE_DEFAULT)),
    }

pool_sizes = [len(p["pool"]) for p in user_profiles.values()]
core_sizes = [len(p["core"]) for p in user_profiles.values()]
print(f"  pool 사이즈: min={min(pool_sizes)}, p50={int(np.median(pool_sizes))}, max={max(pool_sizes)}")
print(f"  core 사이즈: min={min(core_sizes)}, p50={int(np.median(core_sizes))}, max={max(core_sizes)}")
print(f"  활동량 분포: {pd.Series(activity_choices).value_counts().to_dict()}")

# ───────────────────────────────────────────────
# 8. Purchase 이벤트 생성 (core 30% / pool 60% / noise 10%)
# ───────────────────────────────────────────────
print("8. Purchase 이벤트 생성...")
records_purchase = []
for uid, prof in user_profiles.items():
    low, high = ACTIVITY_RANGES[prof["activity"]]
    n_pur = int(rng.integers(low, high + 1))
    if n_pur <= 0:
        continue
    # v8a: 페르소나별 P_CORE 사용
    p_core_u = prof["p_core"]
    p_pool_u = 1.0 - p_core_u - P_NOISE
    r = rng.random(n_pur)
    pids = np.empty(n_pur, dtype=object)
    core_mask  = r < p_core_u
    pool_mask  = (r >= p_core_u) & (r < p_core_u + p_pool_u)
    noise_mask = r >= p_core_u + p_pool_u
    if core_mask.any():
        # v11: Zipf weighted — top-rank core 가 더 자주 sample (단골 신호)
        pids[core_mask]  = rng.choice(prof["core"],  size=int(core_mask.sum()),  replace=True,
                                     p=prof["core_weights"])
    if pool_mask.any():
        # v11: pool 도 약한 Zipf (다양성 보존하며 신호 추가)
        pids[pool_mask]  = rng.choice(prof["pool"],  size=int(pool_mask.sum()),  replace=True,
                                     p=prof["pool_weights"])
    if noise_mask.any():
        pids[noise_mask] = rng.choice(all_pids,      size=int(noise_mask.sum()), replace=True)
    ts = random_timestamps(n_pur)
    for pid, t in zip(pids, ts):
        records_purchase.append((uid, pid, "purchase", t, 0, 0, "success_purchase", 3.0))

COLS = ["user_id","product_id","event_type","timestamp",
        "is_bounced","is_abandoned","funnel_type","event_weight"]
purchase_logs = pd.DataFrame(records_purchase, columns=COLS)
print(f"  purchase: {len(purchase_logs):,}건")

# ───────────────────────────────────────────────
# 9. Success funnel (view → cart → purchase)
# ───────────────────────────────────────────────
print("9. Success funnel 생성...")
cart_for_purchase = purchase_logs.copy()
cart_for_purchase["event_type"]   = "cart"
cart_for_purchase["funnel_type"]  = "success_cart"
cart_for_purchase["event_weight"] = 2.0
cart_for_purchase["timestamp"]    = cart_for_purchase["timestamp"] - pd.to_timedelta(10, unit="m")

view_for_purchase = purchase_logs.copy()
view_for_purchase["event_type"]   = "view"
view_for_purchase["funnel_type"]  = "success_view"
view_for_purchase["event_weight"] = 1.0
view_for_purchase["timestamp"]    = view_for_purchase["timestamp"] - pd.to_timedelta(30, unit="m")
print(f"  cart(success): {len(cart_for_purchase):,}, view(success): {len(view_for_purchase):,}")

# ───────────────────────────────────────────────
# 10. 부정 퍼널 (bounced view, abandoned cart + 짝 view)
# ───────────────────────────────────────────────
print("10. Bounced view / Abandoned cart 생성...")
records_bounced  = []
records_aban     = []
records_aban_v   = []
for uid, prof in user_profiles.items():
    low, high = ACTIVITY_RANGES[prof["activity"]]
    n_pur = int(rng.integers(low, high + 1))
    n_bnc = int(n_pur * BOUNCED_VIEW_RATIO)
    n_aban = int(n_pur * ABANDONED_CART_RATIO)
    pool = prof["pool"]

    def sample_with_noise(n: int) -> np.ndarray:
        r = rng.random(n)
        out = np.empty(n, dtype=object)
        m = r < NEG_NOISE_RATIO
        if m.any():
            out[m] = rng.choice(all_pids, size=int(m.sum()), replace=True)
        nm = ~m
        if nm.any():
            out[nm] = rng.choice(pool, size=int(nm.sum()), replace=True)
        return out

    if n_bnc > 0:
        pids = sample_with_noise(n_bnc)
        ts   = random_timestamps(n_bnc)
        for pid, t in zip(pids, ts):
            records_bounced.append((uid, pid, "view", t, 1, 0, "bounced_view", 0.3))
    if n_aban > 0:
        pids = sample_with_noise(n_aban)
        ts   = random_timestamps(n_aban)
        for pid, t in zip(pids, ts):
            records_aban.append((uid, pid, "cart", t, 0, 1, "abandoned_cart", 1.2))
            records_aban_v.append((uid, pid, "view", t - pd.to_timedelta(20, unit="m"),
                                   0, 1, "abandoned_cart_view", 0.8))

bounced_views        = pd.DataFrame(records_bounced, columns=COLS)
abandoned_carts      = pd.DataFrame(records_aban,    columns=COLS)
abandoned_cart_views = pd.DataFrame(records_aban_v,  columns=COLS)
print(f"  bounced_view: {len(bounced_views):,}, abandoned_cart: {len(abandoned_carts):,}")

# ───────────────────────────────────────────────
# 11. Search 이벤트 (페르소나별 search_prob)
# ───────────────────────────────────────────────
print("11. Search 이벤트 생성...")
all_view_events = pd.concat(
    [view_for_purchase, bounced_views, abandoned_cart_views], ignore_index=True
)
search_chunks = []
for persona, cfg in PERSONA_CONFIGS.items():
    prob = float(cfg["search_prob"])
    persona_uids = users.loc[users["persona"]==persona, "user_id"]
    cand = all_view_events[all_view_events["user_id"].isin(persona_uids)]
    if len(cand) == 0 or prob <= 0:
        continue
    sample = cand.sample(frac=min(prob, 1.0), random_state=RANDOM_SEED + len(search_chunks))
    sample = sample.copy()
    sample["event_type"]   = "search"
    sample["funnel_type"]  = "search"
    sample["event_weight"] = 0.5
    sample["is_bounced"]   = 0
    sample["is_abandoned"] = 0
    sample["timestamp"]    = sample["timestamp"] - pd.to_timedelta(5, unit="m")
    search_chunks.append(sample)
search_logs = (pd.concat(search_chunks, ignore_index=True)
               if search_chunks else pd.DataFrame(columns=COLS))
print(f"  search: {len(search_logs):,}")

# ───────────────────────────────────────────────
# 12. 합산 + 정렬
# ───────────────────────────────────────────────
print("12. 전체 합치고 정렬...")
logs = pd.concat([
    search_logs, view_for_purchase, cart_for_purchase, purchase_logs,
    bounced_views, abandoned_cart_views, abandoned_carts
], ignore_index=True)
logs = logs[
    logs["user_id"].isin(users["user_id"])
    & logs["product_id"].isin(products["product_id"])
].copy()
logs["timestamp"] = pd.to_datetime(logs["timestamp"])
logs = logs.sort_values("timestamp").reset_index(drop=True)
print(f"  전체 이벤트: {len(logs):,}")
print(logs["event_type"].value_counts().to_dict())

# ───────────────────────────────────────────────
# 13. Train / Valid / Test 시간 분할 (8:1:1)
# ───────────────────────────────────────────────
print("13. 8:1:1 시간 분할...")
train_cutoff = logs["timestamp"].quantile(0.8)
valid_cutoff = logs["timestamp"].quantile(0.9)
train_logs = logs[logs["timestamp"] <= train_cutoff].copy()
valid_logs = logs[(logs["timestamp"] > train_cutoff) & (logs["timestamp"] <= valid_cutoff)].copy()
test_logs  = logs[logs["timestamp"] > valid_cutoff].copy()
train_logs.to_csv(f"{DATA_DIR}/train_logs.csv", index=False)
valid_logs.to_csv(f"{DATA_DIR}/valid_logs.csv", index=False)
test_logs.to_csv(f"{DATA_DIR}/test_logs.csv", index=False)
print(f"  train: {len(train_logs):,}  valid: {len(valid_logs):,}  test: {len(test_logs):,}")

# ───────────────────────────────────────────────
# 14. 데이터 신호 검증 (페르소나/유저 JS, 재구매율)
# ───────────────────────────────────────────────
print("14. 데이터 신호 검증...")
try:
    from scipy.spatial.distance import jensenshannon

    pos = logs[logs["event_type"].isin(["cart","purchase"])]
    pos = pos.merge(users, on="user_id").merge(
        products[["product_id","category_L3"]], on="product_id"
    )
    all_c3 = sorted(products["category_L3"].unique())

    # 페르소나 간 JS
    dists = {}
    for persona in PERSONAS:
        sub = pos[pos["persona"]==persona]
        vc = sub["category_L3"].value_counts(normalize=True)
        dists[persona] = np.array([vc.get(c, 0) for c in all_c3])
    print("  [페르소나 간 cat_L3 JS divergence (0=같음, 1=완전 다름)]")
    print("         " + "  ".join([p[:5] for p in PERSONAS]))
    for p1 in PERSONAS:
        print(f"  {p1[:5]:5s} ", end="")
        for p2 in PERSONAS:
            print(f"{jensenshannon(dists[p1], dists[p2]):.3f} ", end="")
        print()

    # 유저 train↔test 일관성
    train_pos = train_logs[train_logs["event_type"].isin(["cart","purchase"])].merge(
        products[["product_id","category_L3"]], on="product_id"
    )
    test_pos = test_logs[test_logs["event_type"]=="purchase"].merge(
        products[["product_id","category_L3"]], on="product_id"
    )
    tu = train_pos.groupby("user_id").size()
    te = test_pos.groupby("user_id").size()
    active = tu[tu>=20].index.intersection(te[te>=3].index)
    sample = list(active[:500])
    train_d = {}
    test_d = {}
    for uid in sample:
        a = train_pos[train_pos["user_id"]==uid]["category_L3"].value_counts(normalize=True)
        b = test_pos[test_pos["user_id"]==uid]["category_L3"].value_counts(normalize=True)
        train_d[uid] = np.array([a.get(c, 0) for c in all_c3])
        test_d[uid]  = np.array([b.get(c, 0) for c in all_c3])
    js_self = [jensenshannon(train_d[u], test_d[u]) for u in sample]
    js_rand = []
    for u in sample:
        other = sample[(sample.index(u)+1) % len(sample)]
        js_rand.append(jensenshannon(train_d[u], test_d[other]))
    print(f"  [유저 train↔test cat_L3 JS]  self={np.mean(js_self):.3f}  rand={np.mean(js_rand):.3f}"
          f"  → 신호={(np.mean(js_rand)-np.mean(js_self))/max(1e-9,np.mean(js_rand)):.3f}")

    # 재구매율
    repeat_rates = []
    for uid in sample:
        tr = set(train_pos[train_pos["user_id"]==uid]["product_id"])
        ts2 = set(test_pos[test_pos["user_id"]==uid]["product_id"])
        if ts2:
            repeat_rates.append(len(tr & ts2) / len(ts2))
    print(f"  [product_id 재구매율] 평균={np.mean(repeat_rates):.4f}")

    # v11: Zipf 효과 진단 — user 별 top-3 product 가 차지하는 cart/purchase 비중
    print("  [Zipf 진단] user 별 top-3 product 의 cart/purchase 점유율...")
    pos_all = logs[logs["event_type"].isin(["cart", "purchase"])]
    concentrations = []
    user_counts_pos = pos_all.groupby("user_id").size()
    active_users = user_counts_pos[user_counts_pos >= 20].index[:1000]
    for uid in active_users:
        sub = pos_all[pos_all["user_id"] == uid]
        if len(sub) < 10:
            continue
        top3_share = sub["product_id"].value_counts().head(3).sum() / len(sub)
        concentrations.append(top3_share)
    if concentrations:
        print(f"    top-3 점유율 평균={np.mean(concentrations):.3f}  "
              f"중앙값={np.median(concentrations):.3f}  "
              f"(v8a: ~0.10, v11 목표: 0.30+)")
except Exception as e:
    print(f"  검증 skip: {e}")

print("[DONE] 데이터 생성 + 검증 완료")
print("  - data/products.csv")
print("  - data/users.csv")
print("  - data/train_logs.csv")
print("  - data/valid_logs.csv")
print("  - data/test_logs.csv")
