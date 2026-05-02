import pandas as pd
import numpy as np
import time
import requests
import random
import os
import yaml

print("🚀 [Simulator] 고객 행동 시뮬레이터 가동 준비 중...")

# ---------------------------------------------------------
# 1. config.yaml 로드
# ---------------------------------------------------------
try:
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
        PERSONA_RATIOS = config_data['persona_ratios']
        PERSONA_CONFIG = config_data['persona_configs']
        # [역할 4 담당자에게] 엔드포인트(/api/log)는 역할 4와 합의 후 config.yaml에서 관리
        API_URL = config_data.get('api', {}).get('log_endpoint', 'http://api-server:8000/api/log')
    print(f"✅ config.yaml 설정 로드 완료  (API: {API_URL})")
except FileNotFoundError:
    print("⚠️ config.yaml 파일을 찾을 수 없습니다. 기본값으로 실행합니다.")
    PERSONA_RATIOS = {"트렌드세터": 1.0}
    PERSONA_CONFIG = {"트렌드세터": {"view_to_cart": 0.3, "cart_to_purchase": 0.5,
                                     "target_category": "Trending", "search_prob": 0.5}}
    API_URL = "http://api-server:8000/api/log"

# ---------------------------------------------------------
# 2. 기초 데이터 로드
# ---------------------------------------------------------
try:
    df_users = pd.read_csv('data/users.csv')
    df_prods = pd.read_csv('data/products.csv')
except FileNotFoundError:
    print("⚠️ users.csv 또는 products.csv 파일을 찾을 수 없습니다.")
    exit()

LOG_FILE = "data/new_train_logs.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("user_id,product_id,event_type,timestamp\n")

def post_event(user_id: str, product_id: str, event_type: str, timestamp: int) -> None:
    payload = {
        "user_id":    user_id,
        "product_id": product_id,
        "event_type": event_type,
        "timestamp":  timestamp,
    }
    try:
        requests.post(API_URL, json=payload, timeout=1)
    except requests.exceptions.RequestException:
        pass
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{user_id},{product_id},{event_type},{timestamp}\n")


log_count = 0
print("🔥 [Simulator] 실시간 트래픽 생성을 시작합니다! (Ctrl+C로 중단)")

# ---------------------------------------------------------
# 3. 무한 루프로 실시간 트래픽 생성
# ---------------------------------------------------------
try:
    while True:
        # A. 페르소나 선택
        personas      = list(PERSONA_RATIOS.keys())
        probabilities = list(PERSONA_RATIOS.values())
        chosen_persona = np.random.choice(personas, p=probabilities)

        user     = df_users[df_users['persona'] == chosen_persona].sample(1).iloc[0]
        user_id  = user['user_id']
        persona  = user['persona']

        # B. 타겟 카테고리 기반 상품 선택
        config         = PERSONA_CONFIG.get(persona, PERSONA_CONFIG["신중탐색"])
        target_keyword = config["target_category"]

        if target_keyword != "All":
            target_products = df_prods[df_prods['product_name'].str.contains(
                target_keyword, case=False, na=False)]
            if target_products.empty:
                target_products = df_prods
        else:
            target_products = df_prods

        # 가격 민감도(price_sensitivity) 기반 가격대 필터
        price_sensitivity = config.get("price_sensitivity", "medium")
        if price_sensitivity == "high":
            price_filtered = target_products[target_products['price_tier'] == 'low']
        elif price_sensitivity == "low":
            price_filtered = target_products[target_products['price_tier'].isin(['medium', 'high'])]
        else:
            price_filtered = target_products
        if not price_filtered.empty:
            target_products = price_filtered

        product    = target_products.sample(1).iloc[0]
        product_id = product['product_id']
        timestamp  = int(time.time())

        # C. search 이벤트 — 페르소나별 확률(search_prob)로 세션 시작
        search_prob = config.get("search_prob", 0.0)
        if random.random() < search_prob:
            post_event(user_id, product_id, "search", timestamp)

        # D. view → cart → purchase 퍼널
        event_type = "view"
        if random.random() < config["view_to_cart"]:
            event_type = "cart"
            if random.random() < config["cart_to_purchase"]:
                event_type = "purchase"

        post_event(user_id, product_id, event_type, timestamp)
        log_count += 1

        if log_count % 100 == 0:
            print(f"📊 [Simulator] 누적 로그: {log_count}건  "
                  f"(최근: {user_id} → {product_id} [{event_type}])")

        time.sleep(0.05)

except KeyboardInterrupt:
    print(f"\n🛑 [Simulator] 시뮬레이터 중지됨. 총 {log_count}건의 로그를 생성했습니다.")
