⛴️
39_멀티모달 검색 및 Multi-Stage
추천 시스템 구축

1. 미션 소개
멀티모달 검색은 텍스트와 이미지를 동시에 이해하여 상품을 찾는 기술이다. 무신사의 스냅
이미지 검색, 네이버 스마트렌즈처럼 "이 옷과 비슷한 상품 찾기" 기능에 활용된다. 개인화
추천은 YouTube, Pinterest, 쿠팡 등에서 사용하는 Multi-Stage Recommender로,
Candidate Generation → Ranking → Re-ranking 3단계를 거쳐 수억 개 상품 중 사용
자별 Top-N을 밀리초 내에 추출한다.
이 미션에서는 고객 행동 시뮬레이터를 직접 구현하여 6가지 페르소나의 검색, 조회, 구매
패턴을 생성한다. CLIP 기반 멀티모달 검색 엔진과 Two-Tower + DeepFM 기반 Multi-
Stage 추천 시스템을 구축하고, MAB(Multi-Armed Bandit) 탐색 전략과 세션 기반 실시
간 추천, A/B 테스트 시뮬레이션까지 포함한 End-to-End 시스템을 Docker로 배포한다.
최종 결과물은 GitHub 레포지토리에 코드와 문서를 업로드하여 URL을 제출한다.
이 경험을 통해 Contrastive Learning 기반 멀티모달 임베딩의 원리, ANN 인덱스의 속도-
정확도 Trade-off, Feature Interaction을 학습하는 Ranking 모델의 구조, Exploration
vs Exploitation 균형, 오프라인/온라인 평가 지표의 차이와 불일치 원인을 이해하게 된다.

2. 최종 결과물
다음 4가지 기능이 정상 동작하는 시스템 1개를 완성하고, GitHub 레포지토리 URL을 제출
한다.

1. 멀티모달 검색 API

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 1



입력/요청: POST /api/search에 텍스트, 이미지, 또는 텍스트+이미지 조합 전송
출력/화면: 유사 상품 Top-K(기본값 K=10), 유사도 점수, 검색 유형, 응답 시간이
JSON으로 반환된다

2. 개인화 추천 API

입력/요청: GET /api/recommend?user_id=U1234&top_n=10

출력/화면: 추천 상품 Top-N(기본값 N=10), 추천 사유(reason), MAB 탐색 슬롯
표시, 각 Stage별 처리 시간이 JSON으로 반환된다

3. 평가 및 A/B 테스트 대시보드
입력/요청: 브라우저에서 http://localhost:8501  접속
출력/화면: 검색 품질(MRR, NDCG), 추천 성능(HitRate, Coverage), A/B 테스
트 결과(전환율, p-value, 신뢰구간)가 차트로 표시된다

4. 시뮬레이터 및 시스템 실행
입력/요청: docker-compose up  명령 실행
출력/화면: 전체 시스템(Redis, API 서버, 대시보드)이 한 번에 실행되고, 시뮬레이
터가 행동 로그를 실시간 생성한다

5. API 응답 필수 필드 규격
모든 API 엔드포인트는 아래 필수 응답 필드를 반드시 포함해야 한다. 이는 자동 채
점 및 동료 평가의 일관성을 보장하기 위한 최소 규격이다.

검색 API 필수 응답 필드

필드명 타입 설명
search_type string "text" | "image" | "hybrid"

results[] array product_id, name, score, price 포함
latency_ms float 검색 API 전체 응답 시간 (ms)

total_count int 검색된 전체 결과 수

추천 API 필수 응답 필드

필드명 타입 설명
user_id string 요청 사용자 ID

product_id, score, reason,
recommendations[] array

is_exploration 포함

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 2



candidate_ms, ranking_ms,
pipeline_latency object

reranking_ms, total_ms

session_context object recent_clicks, session_interest

3. 과제 목표
이 과제를 마친 후, 학습자는 아래를 스스로 설명할 수 있어야 한다.

CLIP 모델이 Contrastive Learning으로 텍스트-이미지를 동일 벡터 공간에 매핑하는
원리를 설명할 수 있다.

Two-Tower 모델에서 User/Item Tower를 분리하여 Item 임베딩을 사전 계산
(Offline)하고 ANN으로 검색(Online)하는 구조의 장단점을 이해한다.

FAISS IVF+PQ 인덱스가 클러스터링과 양자화를 통해 속도와 정확도(Recall) 사이에
서 Trade-off를 가지는 원리를 설명할 수 있다.

Candidate Generation(Recall 중심)과 Ranking(Precision 중심)의 역할 차이와 분
리 이유를 구분할 수 있다.

MAB(Multi-Armed Bandit)가 Exploration과 Exploitation 사이에서 균형을 맞추는
방식을 이해한다.

오프라인 평가(HitRate, NDCG)와 온라인 평가(CTR, CVR)가 측정하는 것의 차이, 그
리고 두 지표가 불일치할 수 있는 이유를 이해한다.

4. 기능 요구 사항
다음 요구사항을 모두 만족해야 한다.

1. 고객 행동 시뮬레이터
상품 5만 건 이상이 생성되어야 한다.

카테고리는 3단계 계층 구조(대분류 → 중분류 → 소분류)를 포함해야 한다.

고객 1만 명 이상이 생성되어야 한다.

최소 6가지 페르소나(트렌드세터, 실용주의자, 가성비추구, 브랜드충성, 충동구매,
신중탐색)가 정의되어야 한다.

페르소나별로 선호 카테고리, 가격 민감도, 전환율이 차별화되어야 한다.

행동 로그 100만 건 이상이 생성되어야 한다.

이벤트 유형은 search, view, cart, purchase를 포함해야 한다.

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 3



시뮬레이터 설정(페르소나 비율, 전이 확률 등)은 config 파일로 변경 가능해야 한
다.

2. 멀티모달 검색 엔진
텍스트 쿼리를 CLIP Text Encoder로 임베딩해야 한다.

이미지를 CLIP Image Encoder로 임베딩해야 한다.

텍스트 전용, 이미지 전용, 텍스트+이미지 하이브리드 검색이 모두 동작해야 한다.

FAISS 인덱스(IVF+PQ 또는 HNSW)를 사용해야 한다.

검색 응답 시간이 200ms 이내여야 한다.

테스트셋에서 MRR ≥ 0.55를 달성해야 한다.

테스트셋에서 NDCG@10 ≥ 0.50을 달성해야 한다.

3. Multi-Stage 추천 엔진
Stage 1: Candidate Generation (Two-Tower)

User Tower는 사용자의 최근 행동 시퀀스와 프로필 피처를 입력으로 받아야
한다.

Item Tower는 상품의 카테고리, 속성, 가격을 입력으로 받아야 한다.

Item 임베딩은 Offline에서 사전 계산하여 FAISS 인덱스에 저장해야 한다.

FAISS 인덱스로 후보 300개 이상을 100ms 이내에 추출해야 한다.

테스트셋에서 Recall@300 ≥ 0.30을 달성해야 한다.

Stage 2: Ranking (DeepFM 또는 Wide&Deep)

후보 상품에 대해 CTR/CVR을 예측하는 Ranking 모델이 구현되어야 한다.

입력 피처는 User 피처, Item 피처, Cross 피처, Context 피처를 포함해야 한
다.

Ranking 모델의 AUC ≥ 0.70을 달성해야 한다.

Stage 3: Re-ranking (비즈니스 로직 + MAB)

동일 카테고리 상품이 연속 3개 이상 나오지 않도록 다양성을 조절해야 한다.

신규 사용자(행동 이력 5개 미만)는 인기 상품 + 트렌딩 상품으로 대체해야 한
다.

추천 결과 상위 N개 중 1~2개는 MAB 알고리즘(Epsilon-Greedy 또는 UCB)
을 통해 탐색(Exploration) 슬롯으로 할당해야 한다.

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 4



신규 상품(등록 7일 이내)에 노출 기회를 보장하는 로직이 구현되어야 한다.

세션 기반 실시간 추천
현재 세션의 클릭 이벤트를 실시간으로 추천에 반영해야 한다.

세션 인코더(GRU 또는 Transformer)로 단기 관심을 임베딩해야 한다.

장기 선호(User Tower)와 단기 관심(Session Encoder)을 결합하여 추천해
야 한다.

추천 성능 평가
테스트셋에서 HitRate@50 ≥ 0.20을 달성해야 한다.

테스트셋에서 NDCG@50 ≥ 0.08을 달성해야 한다.

Coverage(추천된 고유 상품 수 / 전체 상품 수) ≥ 0.20을 달성해야 한다.

추천 API 전체 응답 시간이 200ms 이내여야 한다.

4. Feature Store 및 서빙 시스템
사용자별 실시간 피처(최근 조회 상품, 세션 내 클릭 수)가 Redis에 저장되어야 한
다.

피처 조회 시간이 10ms 이내여야 한다.

docker-compose up  명령으로 전체 시스템이 한 번에 실행되어야 한다.

API 서버와 대시보드가 각각 독립 컨테이너로 실행되어야 한다.

5. A/B 테스트 및 평가 시스템
두 가지 이상의 추천 전략을 비교하는 A/B 테스트 시뮬레이션이 구현되어야 한다.

통계적 유의성 검정(Chi-square 또는 Z-test)이 구현되어야 한다.

p-value와 95% 신뢰구간이 출력되어야 한다.

대시보드에서 검색 품질, 추천 성능, A/B 테스트 결과를 시각화해야 한다.

6. Continuous Training 파이프라인 (모니터링 및 재학습)

추천 성능 지표(HitRate, CTR)를 주기적으로 모니터링하는 스크립트가 구현되어
야 한다.

성능 지표가 임계값 이하로 떨어지면 알림을 출력해야 한다.

새로운 로그가 일정량(예: 10,000건) 쌓이면 재학습을 트리거하는 로직이 구현되어
야 한다.

재학습 후 모델 버전을 관리하는 방식이 구현되어야 한다.

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 5



7. 문서화 및 제출
README.md에 프로젝트 개요, 아키텍처 다이어그램, 실행 방법이 포함되어야 한
다.

docs/ 폴더에 실험 결과 리포트(검색, 추천, A/B 테스트)가 포함되어야 한다.

GitHub 레포지토리 URL을 제출해야 한다.

레포지토리는 Public이거나 평가자에게 접근 권한이 부여되어야 한다.

8. 평가 프로토콜 및 재현성 조건
모든 성능 지표는 아래 조건에 따라 산출해야 하며, docs/실험 리포트에 반드시 포
함해야 한다.   

항목 규격
상품 50,000건 + 이미지/텍스트 메타 포함, 사용자 10,000명, 행동 로

데이터 구성 그 1,000,000건 (시뮬레이터 생성)

train / valid / test = 8 / 1 / 1 (시간 기반 분할 권장: 최근 데이터를
데이터 분할

test로 배정)

random_seed = 42 (시뮬레이터, 데이터 분할, 모델 학습 모두 동일
재현용 시드 시드 사용)

Negative Two-Tower: 랜덤 네거티브 1:4 비율 / Ranking: 노출 후 비클릭
Sampling implicit negative 사용

검색: BM25 텍스트 단독 검색 / 추천: Two-Tower 단독(Ranking 없
베이스라인 이 후보만) / 인기도 기반 추천

Offline top-k 기준 (검색 k=10, 추천 k=50), 평가 스크립트를
지표 산출 기준

src/evaluation/metrics.py에 제공
(1) 데이터 분할 방식 (2) 지표 정의/계산식 (3) 재현용 시드/설정값 (4)

리포트 필수 포함 베이스라인 대비 개선율

학습 데이터셋 추천
본 미션은 시뮬레이터 기반으로 설계되어 있지만, 실제 데이터로 학습하거나 시뮬레이
터의 현실성을 높이기 위해 아래 데이터셋을 활용할 수 있습니다.

1. 필수 활용 데이터셋

데이터셋 설명 규모 활용 방법

H&M 패션 구매 이력 Mul
 + 상 ti-Stage 추천 검

Recommendations 품 이미지 3,100만 거래 증, 시뮬레이터 파라미
터 교정

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 6



데이터셋 설명 규모 활용 방법
이커머스 행동 로그

Markov Chain 전이
Retailrocket (view, cart, 270만 이벤트 확률 추정

purchase)

패션 이미지 + 속성
DeepFashion 80만 이미지 CLIP 임베딩 품질 검

레이블 증

다운로드 링크:

H&M Kaggle: https://www.kaggle.com/competitions/h-and-m-
personalized-fashion-recommendations

Retailrocket: 
https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset

DeepFashion: 
https://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html

2. 데이터 라이선스 및 주의사항

데이터셋 라이선스 상업적 사용
H&M Kaggle Kaggle Terms ⚠️ 대회 목적
Retailrocket CC0 ✅ 자유롭게 사용 가능
DeepFashion 연구용 ❌ 학술 목적만 허용

구현 지침
1. 환경 구축 + 시뮬레이터 구현
2. CLIP 임베딩 + FAISS 인덱스 + 검색 API

3. Two-Tower Candidate Generation

4. DeepFM Ranking + 추천 API

5. Re-ranking 비즈니스 로직 + MAB + 세션 추천
6. A/B 테스트 + 대시보드 + Feature Store

7. CT 파이프라인 + 통합 테스트 + 문서화

5. 보너스 과제 (선택)
1. Query Understanding (속성 자동 추출)

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 7



자연어 쿼리에서 속성 필터(가격, 색상, 카테고리)를 자동 추출한다.

예시: "10만원 이하 검정 니트" → price_max=100000, color=black,
category=니트
배움 포인트: 검색 품질에서 Query Understanding의 역할을 체감한다.

2. 추천 다양성 최적화 (MMR)

Maximal Marginal Relevance(MMR)를 적용하여 추천 다양성을 최적화한다.

λ 파라미터로 Relevance-Diversity Trade-off를 조절한다.

배움 포인트: 정확도와 다양성의 Trade-off를 정량적으로 분석한다.

3. 모델 경량화 (ONNX)

추천 모델을 ONNX로 변환하고 추론 속도를 비교한다.

전체 응답 시간 100ms 이내를 목표로 최적화한다.

배움 포인트: 모델 경량화가 서빙 성능에 미치는 영향을 이해한다.

4. Thompson Sampling MAB

Epsilon-Greedy 대신 Thompson Sampling을 구현하여 MAB 성능을 비교한
다.

배움 포인트: 다양한 MAB 알고리즘의 수렴 속도와 Regret 차이를 체감한다.

6. 개발 환경
Python 3.10 이상
PyTorch 2.0 이상
FastAPI (API 서버)

Streamlit (대시보드)

7. 제약 사항
필수 라이브러리
벡터 검색: FAISS (faiss-cpu 또는 faiss-gpu)

멀티모달 인코딩: transformers (CLIP)

Ranking 모델: PyTorch 직접 구현 또는 deepctr-torch

Feature Store: Redis (redis-py)

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 8



인프라
Docker 및 Docker Compose 필수
GPU는 선택 사항 (CPU 환경에서도 동작해야 함)

외부 클라우드 API 사용 금지 (로컬에서 완결)

범위 제한
결제, 배송, 재고 관리 기능은 구현하지 않는다.

사용자 인증은 단순 ID 기반으로 대체한다.

실시간 이벤트 스트리밍(Kafka 등)은 이 과제 범위가 아니다. (이후 미션에서 다룬다)

코드 품질
설정 분리: 하이퍼파라미터, 파일 경로 등은 코드 내 하드코딩하지 않고 config.yaml 또
는 .env 파일로 분리하여 관리해야 한다.

타입 힌트: 주요 함수에 타입 힌트를 작성해야 한다.

제출 형태
GitHub 레포지토리 URL 제출
레포지토리에 README.md, docker-compose.yml, 소스 코드, docs/ 포함 필수
레포지토리는 Public 또는 평가자 접근 권한 부여

로컬 Docker 실행 최소/권장 사양

항목 최소 사양 권장 사양
RAM 16GB 32GB 이상
CPU 4코어 이상 8코어 이상
저장공간 50GB 이상 (SSD 권장) 100GB 이상 (NVMe SSD)

GPU 불필요 (CPU 모드 지원) NVIDIA GPU 8GB+ (CLIP 인코딩 가속)

Docker 메모리 8GB 할당 16GB 할당

리소스 부족 시 대응 전략 

시뮬레이터 로그 규모 downscale(예: 상품 10K, 로그 200K)

CLIP 배치 인코딩(batch=32)

FAISS 인덱스 구축 단계 분리
Redis maxmemory 정책 설정(allkeys-lru)

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 9



Docker 리소스 제한 예시: docker-compose에 mem_limit: 4g 설정
응답 시간 측정 조건 가이드

200ms / 100ms / 10ms 등의 응답 시간 기준은 아래 조건에서 측정한다
측정 방법: warm-up 10회 호출 후, 100회 호출의 p95(95번째 백분위) 기준
동시 요청: 단건(single request) 기준. 동시 요청 테스트는 보너스 과제로 수행
하드웨어: 최소 사양(RAM 16GB, CPU 4코어) 기준. GPU 사용 시 별도 명시
허용 오차: CPU 전용 환경에서 ±50ms 허용. 단, docs/리포트에 실측 환경을 반
드시 기록
최적화 참고: ONNX 변환, 양자화, 캐싱, FAISS 인덱스 파라미터 튜닝(nprobe 조
절) 등을 "성능 개선 체크리스트"로 시도하고 결과를 문서화

8. 결과 예시
아래는 정답이 아니라 참고 예시다. 실제 문구, 디자인, 구현 방식은 달라도 된다.

시스템 실행 화면 예시  

$ docker-compose up

[+] Running 4/4

 ✔ Container redis          Started (0.5s)

 ✔ Container api-server     Started (3.1s)

 ✔ Container dashboard      Started (1.8s)

 ✔ Container simulator      Started (2.0s)

[api-server] FAISS 인덱스 로딩 완료 (50,000 상품)

[api-server] Two-Tower 모델 로딩 완료
[api-server] DeepFM Ranking 모델 로딩 완료
[api-server] http://localhost:8000 에서 API 서버 시작
[dashboard] http://localhost:8501 에서 대시보드 시작
[simulator] 행동 로그 생성 시작...

검색 API 응답 예시  

POST http://localhost:8000/api/search

{

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 10



  "query_text": "검정 오버핏 후드티",

  "top_k": 10

}

Response:

{

  "search_type": "text",

  "results": [

    {"product_id": "P12345", "name": "오버핏 후드 스웨트셔
츠", "score": 0.91, "price": 59000},

    {"product_id": "P23456", "name": "루즈핏 블랙 후드", 

"score": 0.87, "price": 45000}

  ],

  "latency_ms": 42

}

추천 API 응답 예시 

GET http://localhost:8000/api/recommend?user_id=U1234&to

p_n=10

Response:

{

  "user_id": "U1234",

  "recommendations": [

    {"product_id": "P11111", "score": 0.85, "reason": "r

ecent_view", "is_exploration": false},

    {"product_id": "P22222", "score": 0.82, "reason": "s

imilar_users", "is_exploration": false},

    {"product_id": "P99999", "score": 0.45, "reason": "m

ab_exploration", "is_exploration": true}

  ],

  "pipeline_latency": {

    "candidate_ms": 45,

    "ranking_ms": 62,

    "reranking_ms": 12,

    "total_ms": 127

  },

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 11



  "session_context": {

    "recent_clicks": ["P001", "P002"],

    "session_interest": "아우터"

  }

}

A/B 테스트 결과 예시 

========== A/B 테스트 결과 ==========

실험 기간: 시뮬레이션 30일
샘플 수: Control 5,000명, Treatment 5,000명

┌─────────────────────────────────────────────┐

│ Metric          │ Control │ Treatment │ Lift │

├─────────────────────────────────────────────┤

│ CVR             │ 3.2%    │ 4.1%      │+28%  │

│ MAB Exploration │ 0%      │ 15%       │ -    │

└─────────────────────────────────────────────┘

통계적 유의성:

  - p-value: 0.003 (< 0.05, 유의함)

  - 95% CI: [0.4%, 1.4%]

Continuous Training 알림 예시 

[CT Monitor] 2024-01-15 09:00:00

현재 HitRate@50: 0.18 (임계값: 0.20 이하)

성능 저하 감지! 재학습을 권장합니다.

[CT Trigger] 신규 로그 12,345건 축적 (임계값: 10,000건)

모델 재학습 트리거됨...

재학습 완료. 모델 버전: v2.1 → v2.2

9.  동료 평가 질문 예시
기능 동작 검증
[Context 1: 시스템 실행]

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 12



docker-compose up  명령으로 모든 컨테이너가 정상 시작되는가?

검색 API, 추천 API, 대시보드가 각각 응답하는가?

[Context 2: Cold Start 상황]
"오늘 막 가입하여 클릭 이력이 전혀 없는 신규 사용자가 접속했습니다."

신규 사용자 요청 시 적절한 폴백 결과가 반환되는가?

[Context 3: MAB 탐색]
"추천 결과에서 신규 상품 노출이 필요합니다."

추천 결과에 MAB Exploration 슬롯(is_exploration=true)이 포함되어 있는가?

[Context 4: 응답 시간]

검색 API 응답 시간이 200ms 이내인가?

추천 API 응답 시간이 200ms 이내인가?

[Context 5: 정량 성능 지표 검증]

리포트에 데이터 분할 방식(train/valid/test 비율)이 명시되어 있는가?

검색 성능: MRR ≥ 0.55, NDCG@10 ≥ 0.50 달성 여부가 수치로 보고되었는가?

추천 성능: Recall@300 ≥ 0.30, HitRate@50 ≥ 0.20, NDCG@50 ≥ 0.08 달성 여
부가 보고되었는가?

Ranking 모델: AUC ≥ 0.70 달성 여부가 보고되었는가?

Coverage ≥ 0.20 달성 여부가 보고되었는가?

Feature Store 피처 조회 시간이 10ms 이내인지 확인되었는가?

[Context 6: Re-ranking]

동일 카테고리 상품이 연속 3개 이상 노출되지 않는가? (샘플 3건 이상 확인)

신규 상품(등록 7일 이내)이 추천 결과에 포함되는가?

코드 구조 및 설계
[Context: 로직 변경]
"추천 알고리즘을 '유사도순'에서 '최신순'으로 변경해야 합니다."

전체 코드를 수정하지 않고, config나 전략 패턴으로 로직을 교체할 수 있는가?

[Context: Asset 변경]
"시뮬레이터의 상품 데이터(페르소나 비율, 카테고리 구조)가 변경되었습니다."

config 파일 수정만으로 시뮬레이터가 변경된 설정을 반영하는가?

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 13



역할 분리: simulator, search, recommendation, serving 모듈이 폴더로 분리되어
있는가?

문서화: README.md에 실행 방법과 아키텍처가 설명되어 있는가?

[Context : 팀 역할 및 기여도]

README 또는 기획서에 팀원별 역할 분담이 명시되어 있는가?

개인별 주요 기여 영역(구현 모듈, 문서 작성 등)이 요약되어 있는가?

핵심 기술 원리 적용
멀티모달 임베딩: CLIP으로 텍스트/이미지가 동일 벡터로 변환되어 검색에 사용되는

가?

Two-Tower 구조: User/Item Tower가 분리되고, Item 임베딩이 FAISS에 저장되어
있는가?

Ranking 모델: DeepFM 또는 Wide&Deep으로 Feature Interaction이 학습되는가?

MAB 탐색: Exploration 슬롯이 추천 결과에 포함되어 있는가?

세션 기반 추천: 현재 세션 클릭이 추천에 실시간 반영되는가?

A/B 테스트: 통계적 유의성 검정(p-value, 신뢰구간)이 구현되었는가?

CT 파이프라인: 성능 저하 시 재학습 트리거 로직이 구현되었는가?

39_멀티모달 검색 및 Multi-Stage 추천 시스템 구축 14