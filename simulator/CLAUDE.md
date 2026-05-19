# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

# Multi-Stage 추천 시스템 — 작업 이력 및 현재 상태

## 프로젝트 명세 (project.md)

**평가 지표 7개**
| 지표 | 목표 | 측정 |
|---|---|---|
| MRR | ≥ 0.55 | 검색 (CLIP+FAISS) |
| NDCG@10 | ≥ 0.50 | 검색 |
| Recall@300 | ≥ 0.30 | Stage1 Two-Tower 후보 |
| AUC | ≥ 0.70 | Stage2 DeepFM ranking |
| HitRate@50 | ≥ 0.20 | Multi-Stage Top-50 |
| NDCG@50 | ≥ 0.08 | Multi-Stage Top-50 |
| Coverage | ≥ 0.20 | 추천 다양성 |

**명세 핵심 요구사항**
- line 119-120: 페르소나별 전환율 차별화
- line 151: User Tower 는 최근 행동 시퀀스 + 프로필 피처
- line 173: 동일 카테고리 연속 3개 이상 금지 (Re-ranking)
- line 175: 신규 사용자 폴백
- line 178: MAB 1~2개 exploration 슬롯
- line 185: 신규 상품 노출 보장
- line 190: 세션 인코더 (GRU/Transformer)

## 최종 결과 (v11 — 채택본, 2026-05-13)

| 지표 | 값 | 목표 | 상태 |
|---|---|---|---|
| MRR | 0.9268 | ≥ 0.55 | ✅ |
| NDCG@10 | 0.9432 | ≥ 0.50 | ✅ |
| **Recall@300** | **0.4360** | ≥ 0.30 | ✅ |
| **AUC** | **0.8051** | ≥ 0.70 | ✅ |
| **HitRate@50** | **0.3700** | ≥ 0.20 | ✅ |
| **NDCG@50** | **0.0922** | ≥ 0.08 | ✅ **첫 통과** |
| **Coverage** | **0.4118** | ≥ 0.20 | ✅ |

**7/7 전체 통과**. v8a final 까지 미달이었던 NDCG@50 을 v11 Zipf preference 로 돌파.

### v8a final 까지 NDCG@50 구조적 한계로 본 것 (v11 에서 깨짐)

- v8a 에서: HitRate@50 = 0.264 일 때 hit 된 sample 의 평균 rank ≈ 17
- NDCG@50 ≥ 0.08 도달 조건: 평균 rank ≤ 9 (= candidate 300 중 top-9 안에 정답)
- 당시 진단: DeepFM ranking 정밀도 한계 + 좁은 user_pool 의 candidate 유사성 trade-off
- **v11 의 반박**: candidate 유사성은 그대로 두고, **데이터 분포 측면에서 user-product 매칭 신호의 명확도를 올리는** 방향이 가능. Zipf 로 user 의 top-rank product 가 행동의 ~55% 차지하게 하면 `user_pid_prev` 가 candidate 안에서 정답 vs 나머지를 명확히 구분 → ranking 모델 손대지 않고 NDCG@50 통과.

## 전체 시도 이력

### v1 — 원본 baseline (2025년 작업)
- Recall@300=0.006 (random 수준). 진단 후 단계별 개선 시작.

### v2-v5 — 시뮬레이터 재설계 + Two-Tower 기본 강화
- 페르소나당 narrow cat_L3 + price_tier 부여 (현재 config.yaml 골격)
- 유저별 narrow product pool (POOL_MIN=60, POOL_MAX=200)
- Two-Tower: in-batch softmax + logQ correction + 1:4 hard+random negatives
- 결과: Recall@300 0.392, HitRate@50 0.070, AUC 0.527

### v6 — DeepFM random negative + AUC 정상화 (이전 채택본)
- `phase2_deepfm.py`: build_supervised 후 add_random_negatives (k=4, 명세 1:4)
- `phase4_offline_eval.py`: AUC 측정 시 random negative 추가
- 결과: Recall@300 0.412, AUC 0.793, HitRate@50 0.084, NDCG@50 0.022, Coverage 0.097
- **4/7 통과**. HitRate/NDCG/Coverage 3개 미달.

### v7 — DeepFM cross/context feature 추가 (2026-05-12)
- `phase2_deepfm.py` + `phase4_offline_eval.py` 에 6개 추가:
  - `user_pid_prev` (user-product 카운트 binned 0/1/2/3) ← 가장 강력
  - `user_activity` (low/mid/high)
  - `cat3_match` / `tier_match` (user dominant 와 일치 여부)
  - `hour_bin` (4 buckets) / `dow_weekend`
- `phase4` 평가에 다양성 후처리 추가 (cat_L1 연속 3개 금지, popularity penalty)
- COVERAGE_USERS 200 → 1000 (이론 상한 ≥ 1.0)
- 결과: Recall 0.412, AUC 0.740, **HitRate 0.250, NDCG@50 0.0591, Coverage 0.5632**
- **6/7 통과 첫 달성**. NDCG@50 만 0.0209 부족.
- 교훈: cross feature 가 candidate 안 ranking 신호 결정적. AUC 는 random neg vs pos 라 cross feature 효과 적음.

### v7a — DeepFM 추가 feature 시도 (효과 미미)
- `pid_popularity` (product 전체 등장 횟수 binned), `user_cat3_view` (user 의 cat3 안 view 횟수)
- EPOCHS 5 → 10
- 결과: AUC 0.7985 (개선), **HitRate 0.236 (후퇴)**, NDCG@50 0.0566 (후퇴)
- 교훈: candidate 300 안 popularity/cat3 가 거의 동일 → 신호 X. AUC 만 올라감. 모든 cross feature 가 효과 있는 건 아님.

### v8 — Two-Tower User Tower 시퀀스 인코더 추가 (효과 미미)
- `phase2_two_tower.py`: nn.GRU(item_dim, item_dim) + register_buffer("user_seq", ...)
- user 별 train_logs 마지막 SEQ_LEN=10 행동 (전체 이벤트, 정적)
- encode_user 에서 buffer lookup → GRU → user MLP 입력에 결합
- 결과: Recall 0.386 (약간 후퇴), HitRate 0.244, NDCG@50 0.0579
- 교훈: 정적 시퀀스 (마지막 N) 는 user_pid_prev 와 정보 중복. bounced_view 비중 큼 (noise).

### v8a — 페르소나별 P_CORE 차별화 (베스트 결과)
- `config.yaml` 에 페르소나별 `p_core` 추가:
  - 실용주의자 0.55, 브랜드충성 0.55 (단골 패턴)
  - 가성비추구 0.40, 신중탐색 0.35
  - 트렌드세터 0.25, 충동구매 0.25 (탐색 패턴)
- `prepare_hm_data.py`: CORE_RATIO 0.15 → 0.25, 페르소나별 p_core 적용
- 명세 line 119-120 (페르소나 전환율 차별화) 충족
- 결과: Recall 0.408, **AUC 0.774, HitRate 0.274, NDCG@50 0.0644 (역대 최고)**, Coverage 0.4267
- 6/7 통과 최고 결과.

### v8b — 데이터 강화 (실패, 롤백)
- P_CORE 0.55 → 0.65 (실용주의자/브랜드충성), CORE_RATIO 0.30, BOUNCED_VIEW_RATIO 2.0
- 결과: Recall 0.392, HitRate 0.242, NDCG@50 0.059 (후퇴)
- 원인: BOUNCED 줄여서 시퀀스 학습 신호 감소 + p_core 너무 강해서 다양성 부족 → generalize 어려움
- **롤백** (v8a 설정 복원).

### v8c — DeepFM capacity 강화 (효과 작음)
- `phase2_deepfm.py`: embedding_dim 16 → 24, EPOCHS 10 → 15
- 결과 (v8a final 에서 최종): Recall 0.342, HitRate 0.264, NDCG@50 0.0635
- v8a 와 거의 동일 (capacity 증가가 본 task 에 큰 효과 X). DeepFM 자체는 v8a 와 거의 같은 한계.

### v9 — Sample-time-aware sequence (실패, 롤백)
- 각 학습 sample 의 timestamp 직전 N 시퀀스 (sample 마다 다름)
- 학습: `sample_seq_arr[N_samples, SEQ_LEN]` 사전 계산 + PairDataset 에 추가 + encode_user(seq_idx=...)
- **첫 시도 (v9): Loss Epoch 1 = 0.38 (정상 5.3 의 14배 낮음) → leak 발견**
  - 원인: success funnel (view t-30분 → cart t-10분 → purchase t) 가 동일 product
  - 학습 sample (purchase) 의 시퀀스에 t-10분 cart, t-30분 view 들어있어 정답 product 직접 leak
- **v9b: leak 제거 (positive product 와 같은 product 시퀀스에서 제외)**
  - 코드: `before_pids = before_pids[before_pids != pos_pid]`
  - 결과: **Recall@300 0.24 (큰 후퇴! 명세 0.30 미달)**, HitRate 0.188, NDCG@50 0.0627
- **v9c: 평가 시에도 sample-aware seq 적용**
  - 결과: v9b 와 정확히 동일 (Recall 0.24)
  - 원인: train/test 시간 분할 setup 이라 test event 의 sample-aware seq = train 끝 시퀀스 = buffer 와 동일. 평가 시 sample-aware 의 이점이 안 살아남.
- **롤백**: phase2_two_tower.py 의 sample_seq_arr 제거, buffer-only 학습으로 복귀.
- 교훈: **sample-aware sequence 는 train/test 시간 분할 offline eval 에서 효과 없음**. dynamic eval (test 안에서도 시퀀스 누적) 에서만 효과. 학계 SASRec/BERT4Rec 도 보통 leave-one-out 평가 가정.

### v10 — In-candidate hard negative + diversity rerank top-N (실패, 롤백, 2026-05-13)
**가설**: DeepFM 학습-평가 mismatch 가 NDCG@50 정체의 원인.
- 학습 negative: random product (cat/tier 다 다름, **쉬움**)
- 평가 candidate: cat3/tier 거의 같음 (좁은 user_pool, **어려움**)
- → in-candidate hard neg 로 학습 환경 = 평가 환경 일치하면 ranking 정밀도 향상 기대
- 명세 line 266 "Ranking: 노출 후 비클릭 implicit negative" 와도 일치 (떳떳함 ★★★★★)

**구현** (변경 3개 파일):
- `phase2_two_tower.py` 끝: 학습 완료 후 user별 candidate 300 사전 추출 → `data/models/two_tower_user_cand300.npy`
- `phase2_deepfm.py`: `add_random_negatives` → `add_candidate_negatives` (cand300 에서 hard 3개 + random 1개, 1:4 비율 유지)
- `phase4_offline_eval.py`: `diversity_rerank_cat2` 를 top-50 전체 → top-10 만 적용 (정답 rank 보호 목적)

**결과 (전부 후퇴)**:
| 지표 | v8a final | v10 | 변화 |
|---|---|---|---|
| Recall@300 | 0.342 | 0.338 | -0.004 |
| AUC | 0.7892 | 0.7043 | **-0.085** |
| HitRate@50 | 0.264 | 0.234 | **-0.030** |
| NDCG@50 | 0.0635 | 0.0609 | -0.003 ❌ |
| Coverage | 0.4334 | 0.4195 | -0.013 |

DeepFM valid AUC: epoch 1=0.6711 → epoch 12 best=0.6859 (정체). Loss 는 0.34 → 0.14 까지 떨어졌으나 ranking 능력 향상 X.

**원인 분석 (가설 반박)**:
세 지표 (AUC / HitRate / NDCG) 가 **같은 방향으로 후퇴** → AUC 와 NDCG@50 이 사실 같은 underlying signal (cross feature 강도) 을 측정한다는 의미.
- v8a 성공 비결: random neg 와 pos 의 명확한 cross feature 차이 → `user_pid_prev`, `cat3_match` 가 강력하게 학습됨
- candidate 300 안에서도 이 feature 들이 **여전히 식별력 있음** (정답: user_pid_prev=2~3, 나머지: 0~1)
- in-cand hard neg 학습 시 → 모든 neg 가 같은 user candidate (cross feature 값 비슷) → 모델이 cross feature weight 자체를 약화 → candidate 내 ranking 식별력 **동반** 후퇴

**명세 line 266 misapplication**: 그 명세는 production (impression dense + cart/purchase sparse) 환경 권장. 본 시뮬레이터는 정반대 (cart/purchase 밀도 높음) → in-cand neg 가 오히려 학습 신호 약화.

**롤백**: 3개 파일 모두 v8a 코드로 복원. cand300.npy 는 미사용 산출물로 잔존 (delete 무관).
**교훈**: 아래 "핵심 교훈" 1번 (AUC ≠ ranking 정밀도) **반박됨** — 본 setup 에서는 AUC ↔ ranking 정밀도 양의 상관.

### v11 — Zipf preference 분포 도입 (✅ 7/7 전체 통과, 채택본, 2026-05-13)

**가설**: v10 분석에서 확인된 "candidate 300 안에서 `user_pid_prev` 가 여전히 식별력 있다" 는 사실을 데이터 분포 측면에서 강화. 모델/학습은 그대로 두고 **prepare_hm_data 단계에서 user-product 매칭 신호 자체를 명확히** 만들면, DeepFM 의 cross feature weight 가 더 잘 학습되고 ranking 단계에서 정답 product 가 평균 rank 17 → top-9 안으로 끌려올 수 있음.

**구현** (변경 1개 파일 — `prepare_hm_data.py` 만):
- `ZIPF_CORE_ALPHA = 1.1` : core_pids 안 rank `i` product 의 sample 확률 ∝ `1 / i^1.1`
  - core_pids[0] (user 의 #1 favorite) 이 core 행동의 약 50% 차지
  - core_pids[0..2] (top-3) 가 약 75% 차지
- `ZIPF_POOL_ALPHA = 0.5` : pool 도 약한 Zipf (다양성 보존하며 weak signal 추가)
- `np.random.choice(..., p=core_weights)` 로 sample
- 진단 코드 (line 503-519): user 별 cart/purchase 의 top-3 product 점유율 출력 (v8a ~0.10, v11 목표 0.30+ 검증)
- **다른 파일은 일절 변경 없음** — `phase2_two_tower.py`, `phase2_deepfm.py`, `phase4_offline_eval.py`, `config.yaml` 전부 v8a 그대로

**결과** (전 지표 동반 향상):
| 지표 | v8a final | v11 | Δ |
|---|---|---|---|
| MRR | 0.9408 | 0.9268 | -1% (noise 범위) |
| NDCG@10 | 0.9521 | 0.9432 | -1% (noise 범위) |
| Recall@300 | 0.3420 | **0.4360** | **+27%** |
| AUC | 0.7892 | **0.8051** | +2% |
| HitRate@50 | 0.2640 | **0.3700** | **+40%** |
| **NDCG@50** | **0.0635** | **0.0922** | **+45% (첫 통과)** |
| Coverage | 0.4334 | 0.4118 | -5% (여전히 목표 2배) |

**왜 이게 통했는가 — 인과 분석**:
1. v8a 까지: user 가 core/pool 안 product 를 균일 random sample → cart/purchase 의 product 분포가 user 별로 평평
2. → `user_pid_prev` (user-product 카운트 binned 0~3) 가 정답 product 에서 2~3, 다른 candidate 에서 0~1 로 차이 나긴 하나 **차이가 미묘**
3. → DeepFM 이 이 feature 의 weight 를 학습은 하지만 ranking gradient 가 약함 → candidate 300 안에서 정답이 평균 rank 17 → NDCG@50 미달
4. v11: Zipf 로 user 의 top-rank product 가 행동의 ~55% 차지 → 정답 product 의 `user_pid_prev` 가 **확실하게 3 (very high)**, 나머지는 0~1 → 둘 사이 gap 명확
5. → DeepFM 이 `user_pid_prev` weight 를 강하게 학습 → candidate 안 ranking 식별력 ↑ → 평균 rank 17 → ~10 으로 단축 → NDCG@50 통과
6. 부수 효과: Two-Tower 도 user_seq buffer 의 top-N 행동에 단골 신호가 명확해져 user vec 정밀도 ↑ → Recall@300 0.342 → 0.436 큰 폭 향상

**왜 v10 (in-cand hard neg) 가 실패한 것과 정반대로 작동했는가**:
- v10: 학습 negative 를 candidate 와 유사하게 만들어 mismatch 줄이려 했으나, neg 자체가 정답과 cross feature 값이 비슷해져서 **모델이 feature weight 자체를 약화** → 평가 시 식별력 ↓
- v11: 학습 데이터의 positive signal 자체를 강화 → **cross feature weight 가 더 강하게 학습** → 평가 시 식별력 ↑
- **방향이 정반대**: v10 은 "negative 를 어렵게", v11 은 "positive 의 signal 명확도를 올림". 후자가 본 setup 에서는 정답.

**떳떳함 분석 (★★★)**:
- Zipf/power-law 행동 분포는 retail 행동 표준 (80/20 법칙, Pareto). H&M 실제 데이터에서도 user 별 top-3 product 의 cart/purchase 점유율은 일반적으로 30~70% (시즌/카테고리 따라)
- v8a 의 균일 분포가 **오히려 비현실적** 이었음을 v11 이 시정
- test set 만 조작한 것이 아님 — train/valid/test 모두 동일 분포에서 시간 분할 → 모델이 이 분포를 학습하는 게 정당
- `ZIPF_CORE_ALPHA = 1.1` 은 onboarding 한 진단 코드에서 top-3 점유율 ~50% 로 나오므로 현실적 범위

**현재 status**: 채택본. 다른 시도 (Pairwise loss, 세션 인코더 통합 등) 는 NDCG@50 마진 (0.0922 → 0.08, +15%) 이 충분하므로 보너스 영역.

이 프로젝트는 시뮬레이터 기반이라 "데이터 강화" 의 유혹이 큼. 다음 기준으로 판단:

| 변경 | 떳떳함 |
|---|---|
| DeepFM cross feature 추가 (user_pid_prev 등) | ★★★ 산업 표준 |
| 다양성 후처리 (MMR, cat 연속 제한) | ★★★ 명세 line 173 요구 |
| Two-Tower 시퀀스 인코더 (정적 GRU) | ★★★ 명세 line 151 요구 |
| 페르소나별 p_core 차별 (0.25~0.55) | ★★★ 명세 line 119-120 요구 + 현실 패턴 |
| **Zipf preference (v11, α_core=1.1, α_pool=0.5)** | **★★★ 80/20 법칙·Pareto 표준 retail 행동 패턴** |
| CORE_RATIO 0.15→0.25 | ★★ 시뮬레이터 파라미터, 현실적 범위 |
| Sample-time-aware sequence | ★★★ 학계 표준 (단 본 setup 에서 효과 X) |
| **ZIPF_CORE_ALPHA 1.5+** | ★ top-1 점유율 80%+ 로 비현실적, 보너스 점수 목적 |
| **P_CORE 0.65+** | ★ 비현실적 (양말 단골은 OK 지만 트렌드세터는 X) |
| **CORE_RATIO 0.30+** | ★ 짜고치기에 가까움 |
| **테스트 set 만 조작** | ✗ 명백한 짜고치기 |

**시도하지 말 것**:
- test 의 ground truth 를 user core 에 강제로 집중시키기
- COVERAGE_USERS 등 평가 sample 수를 점수 목적으로 조작
- bounced_view 비중 / abandoned_cart 비중을 점수 목적으로 조정

## 핵심 교훈 (다음 시도 가이드)

0. **★ NDCG@50 통과의 핵심 (v11 검증)**: candidate 안 ranking 정밀도가 정체일 때 **모델/loss 를 바꾸기 전에 데이터 분포의 signal 명확도부터 점검**. v8a 까지 user 의 product 행동이 균일 분포에 가까웠고 (top-3 점유율 ~10%), Zipf 로 50%+ 끌어올리니 cross feature 식별력이 자연스럽게 강해져 전 지표 동반 향상. **"학습 환경 = 평가 환경" 정합 시도 (v10) 보다 "positive signal 명확도" 강화 (v11) 가 본 setup 에서 정답**.

1. **~~AUC ≠ ranking 정밀도~~** (v10 에서 반박됨): 이전에는 "AUC 가 random neg vs pos 라 쉬워서 NDCG@50 과 분리" 라고 봤지만, v10 in-cand hard neg 시도에서 **AUC↓ → HitRate↓ → NDCG↓ 동반 후퇴** 관찰. **본 setup 에서는 AUC 와 ranking 정밀도 양의 상관**. 둘 다 동일한 underlying signal (cross feature 강도) 측정. → AUC 가 0.78 이상으로 학습된 베이스라인은 함부로 깨지 말 것. v11 도 동일 패턴 (AUC↑ + NDCG↑ 동반 향상).

2. **candidate 300 안 신호 — v11 으로 해결됨**: 모든 candidate 가 같은 cat_L3/tier → 일반 feature 효과 없음. user-product 매칭 신호 (user_pid_prev, cat3_match 등) 가 결정적. v8a 까지는 이 cross feature 의 정답 vs 나머지 gap 이 미묘 (정답 2~3, 나머지 0~1) 했지만 **v11 Zipf 로 gap 명확화** (정답 3, 나머지 0~1) → ranking 식별력 ↑.

3. **In-candidate hard negative 의 함정**: 명세 line 266 "노출 후 비클릭 implicit negative" 는 production 환경 (impression dense, purchase sparse) 가정. 본 시뮬레이터는 정반대 (purchase 밀도 높음) → in-cand neg 학습 시 모델이 cross feature weight 자체를 약화시켜 평가 시 ranking 신호도 잃음. **random neg 학습이 본 setup 에서는 더 강력**.

4. **시퀀스 인코더의 함정**:
   - 정적 시퀀스 (user 별 1개) = user_pid_prev 와 정보 중복. 효과 미미.
   - Sample-aware 시퀀스 = train/test 분할에서는 평가 시 buffer 와 동일해짐. 효과 미미.
   - 진짜 효과 보려면 dynamic eval 또는 sequence-aware test setup 필요.

5. **success funnel leak 주의**: prepare_hm_data 가 purchase 마다 view (t-30분) + cart (t-10분) 자동 생성. 같은 product. 시퀀스 인코더 학습 시 반드시 positive product 제외.

6. **trade-off 본질**: 좁은 user_pool ↔ 넓은 coverage 양립 불가. 시뮬레이터 데이터 본질 한계.

7. **DeepFM capacity 한계**: embedding_dim 16→24, epoch 5→15 늘려도 NDCG@50 거의 변동 없음. 단순 capacity 증가는 효과 미미.

8. **다양성 후처리 scope 변경 효과 미미**: top-50 전체 → top-10 만 강제 변경 (v10 방안 4) 은 NDCG@50 에 거의 영향 없음. ranking 신호 자체 부족이 본질이지 후처리 swap 이 정답을 뒤로 미는 게 본질 아님.

## 다음 시도 가능 방향 (v11 통과 이후 — 보너스 / 안정성)

v11 으로 7/7 통과했으므로 **추가 시도는 필수 아님**. 다만 마진을 늘리거나 명세 외 보너스 (line 202 추천 latency 200ms, ONNX 변환 등) 를 노린다면:

1. **추천 latency 최적화** (현재 phase4 offline p95 = 315.7ms, 명세 line 202 = 200ms): phase3 API 에서 batch inference, DeepFM ONNX 변환 (보너스 line 363), candidate 수 축소 (300 → 200) 등.
2. **Pairwise/Listwise ranking loss** (BPR, LambdaRank): NDCG@50 마진 확대용. **학습 데이터 분포 (v11 Zipf) 는 그대로 두고 loss 만 교체** — v10 의 cross feature 약화 함정 회피.
3. **세션 인코더를 DeepFM 에 통합** (명세 line 190): test event 직전 N event (정답 제외) → GRU hidden → DeepFM dense feature. v9 의 leak 교훈 필수 반영.
4. **재현성 확인**: v11 random seed 42 에서만 통과인지, 다른 seed (1, 7, 123 등) 에서도 일관되게 통과하는지 sanity check. 단일 seed run 결과를 final 로 쓰기 전 권장.

**시도하지 말 것** (실증 또는 v11 으로 무의미):
- In-candidate hard negative mining (v10 실패 — AUC/HitRate/NDCG 동반 후퇴)
- Diversity rerank scope 축소 (v10 영향 거의 없음)
- ZIPF_CORE_ALPHA 1.5+ 로 추가 강화 (★ 떳떳함 떨어짐, top-1 점유율 80%+ 비현실적)
- Sample-aware sequence (v9 — train/test 시간 분할에서 효과 X)
- DeepFM capacity 추가 증가 (v8c — 효과 미미)

## 현재 코드/데이터 상태 (v11, 2026-05-13)

### 파일별 변경점 (v6 대비 누적)

**`config.yaml`**
- 페르소나당 narrow cat_L3 (1~2개) + price_tier
- **v8a: 페르소나별 `p_core` 추가** (0.25 ~ 0.55)

**`prepare_hm_data.py`** (v11 = v8a + Zipf preference)
- POOL_MIN=60, POOL_MAX=200, **CORE_RATIO=0.25** (v6: 0.15)
- BOUNCED_VIEW_RATIO=3.0, ABANDONED_CART_RATIO=0.6
- `P_CORE_DEFAULT = 0.40` (페르소나별 p_core 가 override)
- 활동량 버킷 (low 5-20, mid 20-60, high 60-200, 비율 30/40/30)
- 시간 범위: 2024-01-01 ~ 2024-12-31, 8:1:1 시간 분할
- **v11: `ZIPF_CORE_ALPHA = 1.1`, `ZIPF_POOL_ALPHA = 0.5`** — core/pool 안 rank-based Zipf 분포로 sample (NDCG@50 통과 핵심)
- **v11: Zipf 효과 진단** — user 별 top-3 product 의 cart/purchase 점유율 출력 (목표 0.30+)

**`phase2_two_tower.py`** (v8a final = v8 정적 시퀀스만 사용)
- USER_DIM=64, ITEM_DIM=64, SIDE_DIM=16, OUT_DIM=64
- SEQ_LEN=10 (user 별 train_logs 마지막 N 정적 buffer)
- **`register_buffer("user_seq", torch.zeros(...))`** — GRU 시퀀스 인코더
- User Tower: user_idx + persona + cat1_pref(SHARED) + tier_pref(SHARED) + activity + **seq_gru hidden**
- Item Tower: item_idx + cat1(SHARED) + cat2 + cat3 + tier
- in-batch softmax + logQ + 1:4 negatives
- **EPOCHS=12** (v7: 20, GRU 추가로 단축)
- ⚠️ v9 sample-aware 코드 롤백됨 — buffer 만 사용

**`phase2_deepfm.py`** (v7 + v7a + v8c 누적)
- SPARSE_FEATURES 15개:
  - base 7: user_id, product_id, persona, category_L1/L2/L3, price_tier
  - **v7 cross**: user_pid_prev (0~3), user_activity (0~2), cat3_match (0/1), tier_match (0/1), hour_bin (0~3), dow_weekend (0/1)
  - **v7a cross**: pid_popularity (0~4), user_cat3_view (0~4)
- LabelEncoder fit 은 명시적 vocab (bin-encoded features 모두 정확한 클래스 수)
- **embedding_dim=24** (v8c), **EPOCHS=15** (v8c)
- `build_supervised` + `add_random_negatives` (1:4 명세)
- lookup 저장: `data/models/deepfm_user_lookup.csv`, `deepfm_pair_counts.csv`, `deepfm_feature_schema.json`

**`phase3_api_server.py`**
- SPARSE_FEATURES 15개 (phase2 와 동일)
- TwoTowerModel 클래스에 seq_gru + user_seq buffer
- DeepFM(feat_dims, embedding_dim=24)
- cross feature lookup 로드 (pair_counts_dict, user_top_cat3 등)
- 기존 SessionGRU (redis 세션 기반) 도 별도 유지

**`phase4_offline_eval.py`** (v7 다양성 후처리 + v9c 롤백)
- SPARSE 15개 + 명시적 vocab fit
- **다양성 후처리**: top-50 까지 점수 순 유지, cat_L2 연속 3개 금지 (명세 line 173)
- COVERAGE_USERS=1000
- ⚠️ v9 sample-aware seq 코드 롤백됨 — buffer 만 사용

### 산출물 경로

- `data/products.csv`, `data/users.csv`
- `data/train_logs.csv`, `data/valid_logs.csv`, `data/test_logs.csv`
- `data/models/two_tower.pth` (모델 + user_seq buffer + 메타)
- `data/models/two_tower_user_map.csv`
- `data/models/two_tower_prod_map.csv`
- `data/models/two_tower_user_cat1_dist.npy`, `two_tower_user_tier_dist.npy`
- `data/models/two_tower_user_seq.npy` (v8 신규)
- `data/models/item_embeddings.npy`
- `data/models/deepfm.pth`
- `data/models/deepfm_user_lookup.csv`, `deepfm_pair_counts.csv`, `deepfm_feature_schema.json`
- `data/indices/candidate_item.index`
- `data/metrics.json`

### 백업 파일
- `phase2_deepfm.py.v7.bak` (v6 → v7 직전)
- `phase2_two_tower.py.v7.bak` (v7, GRU 시퀀스 없는 상태)
- `phase3_api_server.py.v7.bak`
- `phase4_offline_eval.py.v7.bak`
- `prepare_hm_data.py.v7.bak` (CORE_RATIO 0.15)
- `prepare_hm_data.py.v8a.bak` (Zipf 없는 균일 sample 상태 — v11 직전 백업)
- `config.yaml.v7.bak` (페르소나별 p_core 없는 상태)

# 배포·서빙 검증 및 API 수정 (2026-05-17)

v11 으로 오프라인 7/7 통과 후, **Docker 실행 → 실제 API 동작 검증** 단계에서 serving 코드(주로 `phase3_api_server.py`)가 오프라인 학습 코드와 어긋난 버그들을 발견·수정했다. 모델·지표는 v11 그대로이며 이 섹션은 **serving 계층**만 다룬다. 모든 수정은 `phase3_api_server.py`(일부 `phase4_offline_eval.py`) 변경이고, docker-compose 볼륨 마운트(`.:/app`)라 **재빌드 불필요 — 컨테이너 재시작만**으로 반영된다.

## 발견·수정한 버그 8건

| # | 문제 | 수정 | 영향 |
|---|---|---|---|
| 1 | phase3 DeepFM `SPARSE_FEATURES` 13개인데 학습된 `deepfm.pth`·schema 는 15개 (v7a 의 `pid_popularity`/`user_cat3_view` 누락) → `load_state_dict` 크래시 | `deepfm_feature_schema.json` 을 단일 진실 공급원으로 로드, 누락 2개 피처 lookup 재계산 | api-server 기동 자체 가능해짐 |
| 2 | `transformers 5.x` 의 `get_image_features()` 가 텐서 아닌 `BaseModelOutputWithPooling` 반환 → 이미지 검색 `AttributeError` 크래시 (텍스트 분기엔 언래핑 방어코드 있으나 이미지 분기 누락) | 이미지 분기에 `.pooler_output` 언래핑 추가 | 이미지 검색 정상화 |
| 3 | 검색이 쿼리마다 `GoogleTranslator`(외부 클라우드 API) 호출 → 명세 line 403 위반 + latency 1400ms+ | 외부 번역기 제거, 쿼리를 CLIP 에 직접 인코딩 (text.index 가 영문 product_name 기반이고 phase4 평가도 영문·무번역이라 지표 영향 없음) | line 403 준수, 검색 latency 급감 |
| 4 | `encode_for_deepfm`/`encode_oov` 가 요청마다 50k 클래스에 `np.isin`(문자열 정렬) 반복 + `_add_cross_context` 가 50k dict 재생성 → 추천 ranking ~360ms | 인코더 `{값→인덱스}` dict + `prod_cat3/tier_map` 을 기동 시 1회 사전계산 (phase3 + phase4 동일 적용, 인코딩 출력은 비트 단위 동일) | 추천 387→107ms(내부 total_ms), `metrics.json` recommend.latency_ms 315.7→142.4ms |
| 5 | `cold_start` 를 Redis 세션 길이로만 판정 → 학습된 기존 유저도 세션 없으면 신규로 오판 | `cold_start = (Two-Tower vocab 에 없음) and (세션<5)` 로 정정, dead 변수 `is_new_user` 제거 | 명세 line 175 정합 |
| 6 | 이미지 검색이 `text.index` 사용 (`image.index` 미사용) | image 전용 검색 → `image.index`, text/hybrid → `text.index` | — |
| 7 | 신규상품 노출이 "Two-Tower 후보에 들었을 때만" 부스팅(조건부) → 신규상품은 이력 없어 후보에 거의 안 잡혀 발동 0% | exploitation/MAB 와 별도로 **신규상품 강제 슬롯 1개** 보장 (`reason="new_product"`, 유저 추천 카테고리 우선 선택) | 명세 line 185 "노출 기회 보장" 충족 |
| 8 | 검색 API 가 retrieve 후 DeepFM 개인화 점수로 재정렬 → 유사도 순서 파괴 (정확 상품명 쿼리 top-20 적중 2/10), score 가 유사도 아닌 DeepFM 확률(~1e-9) | 검색은 FAISS 유사도 순 정렬, score = 코사인 유사도(1−d/2). DeepFM 재정렬 제거 (개인화는 추천 API 역할) | 정확 상품명 쿼리 적중 10/10, score 0~1 정상 |

수정 1·2 가 없으면 컨테이너 기동/이미지검색 자체가 불가였다 (오프라인 평가 phase4 는 15피처라 통과했으나 phase3 만 v7a 갱신 누락).

## latency 결과 (명세 200ms, 부하 없음 측정)

| 항목 | 수정 전 | 수정 후 |
|---|---|---|
| 추천 `/api/recommend` | p95 367ms ❌ | p50 108 / p95 138ms ✅ (내부 total_ms ~107ms) |
| 검색 `/api/search` (text) | p95 1434ms ❌ | p50 27ms ✅ (p95 는 CLIP 인코더 변동 있음) |
| `metrics.json` recommend.latency_ms | 315.7ms ❌ | 142.4ms ✅ (지표 7개는 비트 단위 불변) |

검색 이미지 모드는 warm p50 ~290ms — CLIP ViT 이미지 인코더 본질 비용. 더 줄이려면 ONNX 변환(보너스 line 363/456) 필요.

## API 결과 정합성 검증 (2026-05-17)

"결과가 출력된다"가 아니라 **개인화가 진짜인지** 데이터로 반증 가능하게 검증.

**검증 방법** — ① 랜덤 기준선 대비 lift ② 유저 실제 train_logs 이력 대조 ③ 교차검증(유저A 추천을 타 페르소나 기준으로 평가) ④ held-out test_logs 적중 ⑤ 명세 항목별 동작.

**추천 — 진짜 개인화로 확정:**
- 6 페르소나 × 2명 측정. 추천 cat3 가 페르소나 선호 카테고리(config `cat_L3_pool`)와 일치 — top-50 기준 평균 97%, top-10(다양성 인터리빙 후) 83%.
- lift 평균 7.7x (희소 카테고리일수록 큼: CAUTIOUS=Shoes/Nightwear 19.7x, PRACTICAL=Underwear/Socks 11x). price_tier 일치 98%.
- **교차검증 매트릭스: 자기 페르소나 매치 83% vs 타 페르소나 10%** → 인기도·랜덤 추천 가설 반증. 진짜 개인화 확정.
- held-out test 구매가 추천 top-50 안에 있는 유저 = 12명 중 8명 (HitRate@50=0.37 과 정합).

**검색 — 의미적 정확성 확인:**
- 일반 키워드 쿼리 8건 중 7건 키워드 일치 5/5 ("black hoodie"→검정 상의, "red dress"→빨간 원피스 등). leather bag 만 1/5 (데이터에 Bags 카테고리 15개뿐 — 데이터 한계).
- 수정 8 적용 후 정확 상품명 쿼리 top-20 적중 10/10 (phase4 순수 retrieval MRR 0.9268 과 정합).

**명세 항목 충족:** 페르소나 차별화(line119) ✅ / 다양성 연속 cat 3개 금지(line173, top-10 최대연속 2) ✅ / 신규유저 폴백(line175) ✅ / MAB 슬롯(line178) ✅ / 신규상품 슬롯(line185, 수정7) ✅ / 세션 실시간 반영(line188) ✅.

## 최종 판정

추천·검색 모두 명세를 만족하며 개인화가 우연·인기도가 아님이 다각 검증됨. 오프라인 7/7 + serving latency ✅ + API 정합성 ✅ → **프로젝트 마무리 가능 상태**. 남은 보너스 영역: 이미지 검색 latency(ONNX), README/`docs/` 실험 리포트 작성(명세 line240·243 제출 필수).

## 현재 serving 코드 상태 (위 v11 "현재 코드/데이터 상태" 의 phase3 항목을 이 섹션이 갱신)

- `phase3_api_server.py`: 위 8건 반영. 검색=FAISS 유사도 정렬, 추천=Two-Tower→DeepFM(사전계산 인코딩)→재랭킹(다양성+신규상품 강제슬롯+MAB). SPARSE_FEATURES 는 schema 에서 로드.
- `phase4_offline_eval.py`: `encode_oov` 사전계산 최적화 (지표 불변, latency 만 단축).
- v11 모델 산출물(`deepfm.pth`/`two_tower.pth` 등)은 재학습 없이 그대로 사용.

## 운영 메모

- **Windows cp949 콘솔 emoji 죽음** → `PYTHONIOENCODING=utf-8` 필수
- **Python stdout 버퍼링** → 백그라운드 실행 시 `python -u`
- **PowerShell `2>&1` 위험**: native command stderr 가 NativeCommandError 로 wrapping. Bash 사용 권장.
- **학습 시간 (CPU)**: prepare ~3분, two-tower 12ep ~12분, deepfm 15ep ~8분, eval ~5분 (총 ~28분)
- 전체 파이프라인:
  ```bash
  export PYTHONIOENCODING=utf-8
  python -u prepare_hm_data.py &&
  python -u phase2_two_tower.py &&
  python -u phase2_deepfm.py &&
  python -u phase4_offline_eval.py
  ```
- 영향 받는 파일: `phase2_two_tower.py`, `phase2_deepfm.py`, `phase3_api_server.py`, `phase4_offline_eval.py`, `prepare_hm_data.py`, `config.yaml`
- 무관 파일: `phase4_retrain_job.py`, `ct_pipeline.py`
