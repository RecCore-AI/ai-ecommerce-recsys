# Search Module

이 디렉토리는 캡스톤 프로젝트의 2번 역할인 멀티모달 검색 및 벡터 인덱싱 파트를 담당한다.

## 현재 구현 상태

현재 products.csv에는 상품명, 설명, 이미지 경로가 없고 다음 컬럼만 존재한다.

- product_id
- category_L1
- category_L2
- category_L3
- price
- price_tier

따라서 현재 버전은 카테고리와 가격 정보를 이용한 TF-IDF 기반 텍스트 검색 baseline이다.

## 파일 구조

```text
search/
├─ README.md
├─ search_engine.py
├─ run_search.py
├─ evaluate_search.py
└─ artifacts/
   ├─ text_vectorizer.pkl
   └─ baseline_matrix.npy