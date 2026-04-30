# 🚀 [팀 가이드] 추천 시스템 프로젝트 실행 및 학습 매뉴얼


## 📋 1. 사전 준비 (Prerequisites)
*   **하드 디스크**: 최소 **50GB~60GB** 이상의 여유 공간 (32GB 데이터 + 도커 이미지 + 인덱스 파일)
*   **도커**: Docker Desktop 설치 및 실행 필수

---

## 📥 2. 데이터 세팅 (Data Setup)
1.  아래 링크에서 **32GB 원본 데이터**를 다운로드하세요.
    *   🔗 [https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations]
2.  프로젝트의 `simulator/data/raw/` 폴더 안에 모든 데이터를 압축 해제합니다.(data 폴터를 새로 만들어주세요.)
    *   **경로 주의**: `simulator/data/raw/images/...` 형태가 되어야 합니다.

---

## 🏗️ 3. 전체 파이프라인 학습 (Training Pipeline)
원본 데이터에서 검색 인덱스와 추천 모델을 생성하는 과정입니다. 순서대로 실행하세요.

| 단계 | 실행 파일 | 설명 | 주요 목표 지표 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `phase1_embedding.py` | CLIP 모델을 이용한 이미지/텍스트 벡터화 | $MRR \ge 0.55$ |
| **Phase 2** | `phase2_two_tower.py` | 후보 추출을 위한 Two-Tower 모델 학습 | $Recall@300 \ge 0.30$ |
| **Phase 3** | `phase2_deepfm.py` | 정교한 랭킹을 위한 DeepFM 모델 학습 | $AUC \ge 0.70$ |

> **💡 Note**: 각 단계가 완료되면 `data/indices/`와 `data/models/` 폴더에 파일이 생성됩니다. 이 파일들이 있어야 시스템이 돌아갑니다.

---

## 🐳 4. 시스템 실행 (Docker)
학습이 완료되었다면 이제 도커를 이용해 전체 서비스를 띄웁니다.

```
.dockerignore가 있는 폴더 위치에서 터미널을 실행시켜서
docker-compose up --build  를 입력해주세요.(반드시 docker desktop을 먼저 설치해주세요.)
```

### 📍 서비스 접속 정보
*   **Search API**: `http://localhost:8000/docs` (Swagger를 통한 API 테스트 가능)
*   **Admin Dashboard**: `http://localhost:8501` (전체 시스템 지표 모니터링)

---

## 🧠 5. 시스템 핵심 로직 이해하기
팀원들이 코드 분석할 때 참고할 핵심 키워드입니다.

*   **멀티모달 검색**: CLIP으로 이미지와 텍스트의 유사도를 계산하고 FAISS로 초고속 검색을 수행합니다.
*   **Multi-Stage 추천**:
    1.  **Candidate Generation**: Two-Tower 모델로 빠르게 후보군 추출
    2.  **Ranking**: DeepFM으로 유저의 클릭 확률(CTR) 정밀 예측
    3.  **Re-ranking**: MAB 알고리즘을 통한 탐색(Exploration) 적용
*   **실시간 처리**: Redis를 사용하여 유저의 세션 로그와 피처를 실시간으로 관리합니다.

---
대시보드에서 A/B테스트 수행한 방법: config.yaml에 있는 
persona_ratios:
  "트렌드세터": 0.15
  "실용주의자": 0.25
  "가성비추구": 0.25
  "브랜드충성": 0.15
  "충동구매":   0.10
  "신중탐색":   0.10
  를 다음과 같이 설정해서 먼저 현실적인 분포에 대한 구매 전환율 🚀 [팀 가이드] 추천 시스템 프로젝트 실행 및 학습 매뉴얼


## 📋 1. 사전 준비 (Prerequisites)
*   **하드 디스크**: 최소 **50GB~60GB** 이상의 여유 공간 (32GB 데이터 + 도커 이미지 + 인덱스 파일)
*   **도커**: Docker Desktop 설치 및 실행 필수

---

## 📥 2. 데이터 세팅 (Data Setup)
1.  아래 링크에서 **32GB 원본 데이터**를 다운로드하세요.
    *   🔗 [https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations]
2.  프로젝트의 `simulator/data/raw/` 폴더 안에 모든 데이터를 압축 해제합니다.(data 폴터를 새로 만들어주세요.)
    *   **경로 주의**: `simulator/data/raw/images/...` 형태가 되어야 합니다.

---

## 🏗️ 3. 전체 파이프라인 학습 (Training Pipeline)
원본 데이터에서 검색 인덱스와 추천 모델을 생성하는 과정입니다. 순서대로 실행하세요.

| 단계 | 실행 파일 | 설명 | 주요 목표 지표 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | `phase1_embedding.py` | CLIP 모델을 이용한 이미지/텍스트 벡터화 | $MRR \ge 0.55$ |
| **Phase 2** | `phase2_two_tower.py` | 후보 추출을 위한 Two-Tower 모델 학습 | $Recall@300 \ge 0.30$ |
| **Phase 3** | `phase2_deepfm.py` | 정교한 랭킹을 위한 DeepFM 모델 학습 | $AUC \ge 0.70$ |

> **💡 Note**: 각 단계가 완료되면 `data/indices/`와 `data/models/` 폴더에 파일이 생성됩니다. 이 파일들이 있어야 시스템이 돌아갑니다.

---

## 🐳 4. 시스템 실행 (Docker)
학습이 완료되었다면 이제 도커를 이용해 전체 서비스를 띄웁니다.

```
.dockerignore가 있는 폴더 위치에서 터미널을 실행시켜서
docker-compose up --build  를 입력해주세요.(반드시 docker desktop을 먼저 설치해주세요.)
```

### 📍 서비스 접속 정보
*   **Search API**: `http://localhost:8000/docs` (Swagger를 통한 API 테스트 가능)
*   **Admin Dashboard**: `http://localhost:8501` (전체 시스템 지표 모니터링)

---

## 🧠 5. 시스템 핵심 로직 이해하기
팀원들이 코드 분석할 때 참고할 핵심 키워드입니다.

*   **멀티모달 검색**: CLIP으로 이미지와 텍스트의 유사도를 계산하고 FAISS로 초고속 검색을 수행합니다.
*   **Multi-Stage 추천**:
    1.  **Candidate Generation**: Two-Tower 모델로 빠르게 후보군 추출
    2.  **Ranking**: DeepFM으로 유저의 클릭 확률(CTR) 정밀 예측
    3.  **Re-ranking**: MAB 알고리즘을 통한 탐색(Exploration) 적용
*   **실시간 처리**: Redis를 사용하여 유저의 세션 로그와 피처를 실시간으로 관리합니다.

---
대시보드 화면에서 A/B테스트 진행 방법
Step 1: 정상 분포로 초기 데이터 생성
# config.yaml
persona_ratios:
  "트렌드세터": 0.15
  "실용주의자": 0.25
  "가성비추구": 0.25
  "브랜드충성": 0.15
  "충동구매":   0.10
  "신중탐색":   0.10
시뮬레이터를 돌려서 new_train_logs.csv에 로그를 쌓고, CT가 재학습 완료되면 이때의 CVR을 기록해 두세요. 이게 A그룹 전환율이 됩니다.

Step 2: 트렌드 주입으로 CVR 변화 유도
yaml# config.yaml 수정
persona_ratios:
  "트렌드세터": 0.80  # Jeans가 매우 유행했다고 가정
  ...
시뮬레이터를 다시 돌리면 Jeans 구매가 폭발적으로 쌓여요. CT가 이걸 감지해서 DeepFM을 재학습하면, 모델이 Jeans를 더 잘 추천하게 되고 CVR이 올라갑니다. 이게 B그룹 전환율이 돼요.

Step 3: 대시보드 Tab 3 A/B 테스트
A그룹 전환율 = Step 1에서 archive CVR  (예: 3.2%)
B그룹 전환율 = Step 2에서 archive CVR  (예: 5.8%)
이 두 값을 넣고 검정 실행 → p-value < 0.05 → "DeepFM 재학습이 통계적으로 유의미하게 CVR을 올렸다" 증명 완료.
  
