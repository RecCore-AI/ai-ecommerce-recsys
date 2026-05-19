# 멀티모달 검색 & Multi-Stage 추천 시스템

CLIP 기반 멀티모달 검색과 Two-Tower + DeepFM 기반 Multi-Stage 추천을 결합한 E-Commerce 추천 시스템입니다.
고객 행동 시뮬레이터로 6가지 페르소나의 검색·조회·구매 패턴을 생성하고, 검색·추천·MAB 탐색·세션 기반 실시간 추천·A/B 테스트·Continuous Training 까지 End-to-End 로 구현하여 Docker 로 배포합니다.

---

## 1. 프로젝트 개요

| 구분 | 내용 |
|---|---|
| 검색 | CLIP(`openai/clip-vit-base-patch32`) 텍스트·이미지 임베딩 + FAISS HNSW 인덱스. 텍스트/이미지/하이브리드 검색 |
| 추천 | **Stage 1** Two-Tower 후보 생성 → **Stage 2** DeepFM 랭킹 → **Stage 3** 재랭킹(다양성·신규상품·MAB) |
| 실시간성 | Redis 피처 스토어 + 세션 GRU 인코더로 현재 세션 클릭을 추천에 즉시 반영 |
| 평가 | 오프라인 지표(MRR·NDCG·Recall·AUC·HitRate·Coverage) + A/B 테스트(Chi-square) 대시보드 |
| 운영 | Continuous Training 모니터(성능 저하 감지·재학습 트리거) |
| 배포 | `docker-compose up` 한 번으로 Redis·API·대시보드·시뮬레이터·CT 모니터 기동 |

데이터: 상품 50,000건 · 사용자 10,000명 · 행동 로그 약 578만 건(시뮬레이터 생성, H&M 데이터 기반 파라미터 교정).

---

## 2. 시스템 아키텍처

```
┌──────────────────────── OFFLINE (학습 / 색인) ────────────────────────┐
│                                                                       │
│  prepare_hm_data.py ─→ products.csv · users.csv · {train/valid/test}_logs.csv │
│        │                                                              │
│        ├─→ phase1_embedding.py ─ CLIP ─→ indices/text.index, image.index    │
│        ├─→ phase2_two_tower.py ──────→ models/two_tower.pth, candidate_item.index │
│        └─→ phase2_deepfm.py ─────────→ models/deepfm.pth                │
│                              │                                        │
│                    phase4_offline_eval.py ─→ data/metrics.json         │
└───────────────────────────────────────────────────────────────────────┘
                               │  모델·인덱스 산출물
                               ▼
┌──────────────────────── ONLINE (서빙) ────────────────────────────────┐
│  api-server  (FastAPI, :8000)                                          │
│   ├ POST /api/search    CLIP → FAISS 유사도 검색 (text · image · hybrid)│
│   ├ GET  /api/recommend Stage1 Two-Tower→FAISS 후보 300                 │
│   │                     Stage2 DeepFM 랭킹 → top-50                     │
│   │                     Stage3 재랭킹 (다양성 + 신규상품 슬롯 + MAB)    │
│   ├ POST /api/log        세션 이벤트 기록                               │
│   └ GET  /api/health · POST /api/reload-model                          │
│                                                                        │
│  redis (:6379 피처 스토어)   dashboard (:8501)   simulator   ct-monitor │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 성능 (테스트셋, `random_seed=42`)

| 지표 | 측정 단계 | 목표 | 달성 | 베이스라인 대비 |
|---|---|---|---|---|
| MRR | 검색 (CLIP+FAISS) | ≥ 0.55 | **0.9268** | BM25 0.9259 → 동등 + 멀티모달 확장 |
| NDCG@10 | 검색 | ≥ 0.50 | **0.9432** | BM25 0.9447 → 동등 |
| Recall@300 | Stage1 Two-Tower 후보 | ≥ 0.30 | **0.4360** | — |
| AUC | Stage2 DeepFM 랭킹 | ≥ 0.70 | **0.8051** | — |
| HitRate@50 | Multi-Stage Top-50 | ≥ 0.20 | **0.3700** | 인기도 0.002 → **185×** |
| NDCG@50 | Multi-Stage Top-50 | ≥ 0.08 | **0.0922** | 인기도 0.0004 |
| Coverage | 추천 다양성 | ≥ 0.20 | **0.4118** | 인기도 0.001 → **412×** |

**7개 지표 전부 목표 달성.** 응답 시간: 추천 API p95 ≈ 138ms, 검색 API(텍스트) p50 ≈ 27ms — 모두 명세 200ms 이내.
상세 수치·계산식·재현 조건은 [`docs/`](docs/) 참고.

---

## 4. 디렉터리 구조

```
.
├── prepare_hm_data.py      # 시뮬레이터: 상품·유저·행동로그 생성 (config.yaml 기반)
├── simulator.py            # 실시간 행동 로그 생성 (서빙 중 동작)
├── config.yaml             # 페르소나·전이확률 등 시뮬레이터 설정
├── phase1_embedding.py     # CLIP 임베딩 → FAISS text/image 인덱스
├── phase2_two_tower.py     # Stage1 Two-Tower 후보 생성 모델 학습
├── phase2_deepfm.py        # Stage2 DeepFM 랭킹 모델 학습
├── phase3_api_server.py    # FastAPI 서빙 (검색·추천·세션 API)
├── phase4_offline_eval.py  # 오프라인 평가 → data/metrics.json
├── phase4_dashboard.py     # Streamlit 평가·A/B 테스트 대시보드
├── phase4_retrain_job.py   # 재학습 잡
├── ct_pipeline.py          # Continuous Training 모니터
├── docker-compose.yml      # redis·api-server·dashboard·simulator·ct-monitor
├── Dockerfile
├── requirements.txt
├── data/                   # 생성 데이터·모델·인덱스·metrics.json
└── docs/                   # 실험 리포트 (검색·추천·A/B·프로토콜)
```

---

## 5. 실행 방법

### 5-1. 전체 시스템 기동 (Docker)

```bash
docker compose up -d --build
```

기동 후:
- 검색·추천 API : http://localhost:8000  (`/docs` 에서 Swagger UI)
- 평가·A/B 대시보드 : http://localhost:8501
- 헬스 체크 : `curl http://localhost:8000/api/health`

> 최초 기동 시 api-server 가 CLIP 모델(약 600MB)을 내려받아 1~2분 소요됩니다.
> 코드(.py) 수정은 볼륨 마운트(`.:/app`)로 반영되므로 컨테이너 재시작만으로 충분합니다.

### 5-2. 데이터·모델 재생성 (선택 — 산출물은 `data/` 에 포함)

```bash
export PYTHONIOENCODING=utf-8
python -u prepare_hm_data.py      # 상품·유저·행동로그 생성
python -u phase1_embedding.py     # CLIP 임베딩 + FAISS 인덱스
python -u phase2_two_tower.py     # Two-Tower 학습
python -u phase2_deepfm.py        # DeepFM 학습
python -u phase4_offline_eval.py  # 오프라인 평가 → metrics.json
```

CPU 환경 기준 전체 약 30~40분 (`random_seed=42` 고정으로 재현 가능).

---

## 6. API 사용법

### 검색 — `POST /api/search`

```bash
# 텍스트 검색
curl -X POST http://localhost:8000/api/search -F "query=black hoodie" -F "top_k=10"
# 이미지 검색
curl -X POST http://localhost:8000/api/search -F "file=@sample.jpg" -F "top_k=10"
```

응답: `search_type`, `results[{product_id, name, score, price}]`, `latency_ms`, `total_count`

### 추천 — `GET /api/recommend`

```bash
curl "http://localhost:8000/api/recommend?user_id=U000058a12d&top_n=10"
```

응답: `user_id`, `recommendations[{product_id, score, reason, is_exploration}]`,
`pipeline_latency{candidate_ms, ranking_ms, reranking_ms, total_ms}`, `session_context{recent_clicks, session_interest}`

`reason` 값: `personalized_deepfm` · `session_interest` · `new_product` · `mab_exploration` · `popular_fallback`(신규 유저)

### 세션 기록 — `POST /api/log`

```bash
curl -X POST http://localhost:8000/api/log -H "Content-Type: application/json" \
  -d '{"user_id":"U000058a12d","product_id":"P0937915002","event_type":"view","timestamp":1700000000}'
```

---

## 7. 평가 재현

`data/metrics.json` 이 7개 지표의 실제 계산값입니다. 재현하려면:

```bash
python -u phase4_offline_eval.py   # test_logs.csv 만 사용 (데이터 누수 방지)
```

평가 프로토콜(데이터 분할·시드·네거티브 샘플링·지표 계산식·베이스라인)은 [`docs/01_실험_프로토콜.md`](docs/01_실험_프로토콜.md) 에 정리되어 있습니다.

---

## 8. 기술 스택

- Python 3.10 / PyTorch 2.x
- 멀티모달 인코딩: `transformers` (CLIP ViT-B/32)
- 벡터 검색: FAISS (HNSW)
- API: FastAPI + Uvicorn / 대시보드: Streamlit
- 피처 스토어: Redis
- 배포: Docker / Docker Compose

GPU 불필요 — CPU 환경에서 동작합니다.

---

## 9. 문서

| 문서 | 내용 |
|---|---|
| [docs/01_실험_프로토콜.md](docs/01_실험_프로토콜.md) | 데이터 구성·분할, 재현 시드/설정, 네거티브 샘플링, 지표 정의·계산식, 베이스라인 정의 |
| [docs/02_검색_리포트.md](docs/02_검색_리포트.md) | 멀티모달 검색 성능 (MRR·NDCG@10), BM25 베이스라인 대비, 의미적 정확성 검증 |
| [docs/03_추천_리포트.md](docs/03_추천_리포트.md) | Multi-Stage 추천 성능 (Recall·AUC·HitRate·NDCG·Coverage), 베이스라인 대비, 개인화 검증 |
| [docs/04_AB테스트_리포트.md](docs/04_AB테스트_리포트.md) | A/B 테스트 방법론·결과 (Chi-square, p-value, 95% 신뢰구간) |

---

## 10. 팀 구성 및 역할 분담

| 팀원 | 역할 | 주요 기여 영역 |
|---|---|---|
| 이명진 | 멀티모달 검색 및 벡터 인덱싱 리드 | Search Pipeline Lead |
| 전희상 | 데이터 시뮬레이션 및 데이터 아키텍트 | Data & Feature Architect |
| 권제우 | Multi-Stage 추천 모델링 리드 | Core RecSys ML Lead |
| 남관우 | 비즈니스 서빙 및 MLOps 리드 | Serving & MLOps Lead |

