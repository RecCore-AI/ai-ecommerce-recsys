# Role 1 — 데이터 아키텍트 수정 내역

이 디렉토리의 파일은 기존 `simulator/` 내 원본을 개선한 버전입니다.  
원본은 `simulator/` 루트에 그대로 보존되어 있습니다.

---

## 파일 용도 및 실행 방법

### prepare_hm_data.py
**용도**: H&M 원본 데이터를 가공하여 학습에 필요한 CSV 파일 3종을 생성하는 **1회성 전처리 스크립트**입니다.

**선행 조건**:
- `data/raw/` 폴더에 `articles.csv`, `customers.csv`, `transactions_train.csv` 존재 필요

**실행**:
```bash
cd simulator/
python role1_data_architect/prepare_hm_data.py
```

**출력물**:
```
data/products.csv       ← 상품 50,000건 (카테고리 3단계, 가격 등급 포함)
data/users.csv          ← 유저 10,000명 (페르소나 할당)
data/train_logs.csv     ← 행동 로그 전체의 80% (학습용)
data/valid_logs.csv     ← 행동 로그 전체의 10% (검증용)
data/test_logs.csv      ← 행동 로그 전체의 10% (평가용, 가장 최근 데이터)
```

---

### simulator.py
**용도**: Docker 컨테이너 내에서 백그라운드로 상시 실행되며, 실시간 가짜 유저 행동 로그를 생성하는 **상시 실행 스크립트**입니다.  
`prepare_hm_data.py` 실행 후 생성된 `products.csv`, `users.csv`를 전제로 합니다.

**선행 조건**:
- `data/products.csv`, `data/users.csv` 존재 필요
- API 서버(`/api/log`)가 실행 중이어야 POST 요청이 정상 처리됨 (서버 미실행 시 로그 기록은 계속됨)

**실행**:
```bash
cd simulator/
python role1_data_architect/simulator.py
```

**동작 흐름**:
```
while True:
  1. config.yaml 기반 페르소나 선택
  2. 페르소나의 target_category + price_sensitivity로 상품 필터링
  3. search_prob 확률로 search 이벤트 발생 (선택)
  4. view → cart → purchase 퍼널 확률로 이벤트 결정
  5. API 서버에 JSON POST + new_train_logs.csv에 기록
  6. 0.05초 대기 (약 초당 20건)
```

**출력물**:
```
data/new_train_logs.csv  ← 실시간 누적 로그 (CT 파이프라인이 모니터링)
```

---

### config.yaml
**용도**: 시뮬레이터의 모든 설정값을 관리하는 **설정 파일**입니다. 코드 수정 없이 이 파일만 변경하여 시뮬레이터 동작을 조정할 수 있습니다.

**주요 설정 항목**:

| 항목 | 설명 |
|------|------|
| `api.log_endpoint` | 시뮬레이터가 POST할 API 주소 (역할 4와 합의 필요) |
| `persona_ratios` | 페르소나별 트래픽 분배 비율 (합계 = 1.0) |
| `view_to_cart` | view 이벤트가 cart로 전환될 확률 |
| `cart_to_purchase` | cart 이벤트가 purchase로 전환될 확률 |
| `target_category` | 해당 페르소나가 선호하는 상품 카테고리 키워드 |
| `search_prob` | 세션 시작 시 search 이벤트를 먼저 발생시킬 확률 |
| `price_sensitivity` | 상품 선택 시 가격대 필터 (`high` / `medium` / `low`) |

---

## 수정 파일 목록

| 파일 | 주요 변경 내용 |
|------|------|
| `prepare_hm_data.py` | 데이터 분할 방식 변경 |
| `simulator.py` | 이벤트 유형 추가, API 규격 수정, 가격 민감도 반영 |
| `config.yaml` | 신규 설정 항목 추가 |

---

## prepare_hm_data.py

### 변경 내용
- **기존**: 전체 로그를 무작위 셔플 후 `train_logs.csv` 단일 파일로 저장
- **변경**: timestamp 기준 시간순 정렬 후 **8:1:1 시간 기반 분할**로 3개 파일 저장

### 출력 파일
```
data/train_logs.csv   ← 전체의 80% (가장 오래된 데이터)
data/valid_logs.csv   ← 전체의 10%
data/test_logs.csv    ← 전체의 10% (가장 최근 데이터)
```

---

## simulator.py

### 변경 내용

**1. 이벤트 유형 추가**  
기존 `view / cart / purchase` 3종에서 `search` 이벤트를 추가했습니다.  
세션 시작 시 페르소나별 `search_prob` 확률로 `search` 이벤트를 먼저 발생시킵니다.

**2. API 요청 규격 변경**
- 기존: `requests.post(data=...)` — form data 방식, 필드명 `item_id`
- 변경: `requests.post(json=...)` — JSON 방식, 필드명 `product_id`

**3. 엔드포인트 config화**  
하드코딩된 `/api/click`을 제거하고 `config.yaml`의 `api.log_endpoint`에서 읽습니다.

**4. 가격 민감도 필터 추가**  
상품 선택 시 `config.yaml`의 `price_sensitivity`에 따라 `price_tier` 기반 필터를 적용합니다.

### 로그 스키마 (변경 없음)
```
user_id, product_id, event_type, timestamp
```
스키마는 기존과 동일합니다. `event_type` 값에 `"search"`가 추가되는 것뿐입니다.

---

## config.yaml

### 추가 항목

```yaml
api:
  log_endpoint: "http://api-server:8000/api/log"  # 역할 4와 합의 필요

persona_configs:
  "각 페르소나":
    search_prob: 0.0 ~ 0.8   # 세션 시작 시 search 이벤트 발생 확률
    price_sensitivity: "high | medium | low"
```

### 페르소나별 설정 요약

| 페르소나 | search_prob | price_sensitivity |
|------|------|------|
| 트렌드세터 | 0.60 | medium |
| 실용주의자 | 0.50 | medium |
| 가성비추구 | 0.70 | high (저가 선호) |
| 브랜드충성 | 0.30 | low (고가 선호) |
| 충동구매 | 0.20 | medium |
| 신중탐색 | 0.80 | high (저가 선호) |

---

## 다른 역할이 알아야 할 내용

### 역할 4 (서빙 & MLOps) — 필수 확인

**1. API 엔드포인트 합의**  
시뮬레이터는 `config.yaml`의 `api.log_endpoint`로 POST를 보냅니다.  
현재 값은 `/api/log`이며, 역할 4가 구현할 엔드포인트 이름과 반드시 맞춰야 합니다.  
결정 후 `config.yaml`의 `api.log_endpoint`만 수정하면 됩니다.

**2. POST JSON 스키마**  
시뮬레이터가 전송하는 JSON 구조입니다. 역할 4의 `/api/log` 엔드포인트는 이 스키마를 수신해야 합니다.

```json
{
  "user_id":    "U0123456789",
  "product_id": "P0123456789",
  "event_type": "search | view | cart | purchase",
  "timestamp":  1746000000
}
```

**3. new_train_logs.csv 초기화 필요**  
기존 `new_train_logs.csv`가 있다면 삭제 후 새 시뮬레이터를 실행해야 합니다.  
기존 파일 헤더(4컬럼)는 새 파일과 동일하므로 스키마 충돌은 없지만,  
`search` 이벤트가 포함된 깨끗한 상태에서 시작하는 것을 권장합니다.

**4. phase4_offline_eval.py 경로 수정**  
현재 `phase4_offline_eval.py`는 `train_logs.csv`에서 평가 샘플을 뽑고 있습니다.  
데이터 분할 이후에는 반드시 **`test_logs.csv`** 를 사용하도록 경로를 수정해야 합니다.

```python
# 수정 전
TRAIN_LOGS_CSV = os.path.join(DATA_DIR, "train_logs.csv")

# 수정 후
TEST_LOGS_CSV = os.path.join(DATA_DIR, "test_logs.csv")
```
