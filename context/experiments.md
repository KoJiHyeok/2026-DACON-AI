# Experiments Log

> 규칙: 모든 실험은 (가설 → 변경점 → 로컬 CV → 리더보드) 순으로 기록.
> CV는 세션 프리픽스 GroupKFold Macro-F1. 리더보드 제출은 일 10회 예산 관리.
> **점수 판정은 LB 실측 또는 세션 group-split만** — accuracy·누수 split·단일 holdout 금지 (w112 핸드오프 계승).

## 현재 최고 기록

| 구분 | Macro-F1 | 비고 |
|---|---|---|
| 리더보드 (우리 팀) | **0.7242** | 동료 제출 (2026-07-05, 구성 확인 필요). 이전 최고 w112 0.7208 / 내 재건 3-way 0.71884 |
| 커트라인(12등) | 0.77665 | 2026.07.05 기준 (07.04 0.77585 → +0.0008/일 속도로 상승 중) — 갭 **−0.056** |
| 순위 | 81등 | 핸드오프 시점 |

## 실험 기록 (w112까지의 여정 — 전부 LB 실측, 핸드오프 §1에서 계승)

| # | 날짜 | 가설/행동 | 변경점 | 로컬 CV | LB | 결론 |
|---|---|---|---|---|---|---|
| 0 | ~07.03 | 공개 베이스라인 | current_prompt만 TF-IDF+LogReg | ~0.43 (group-split) | - | 기준점. history·meta를 버림 |
| 1 | ~07.03 | 3필드 전부 사용 | linear: E_+seq 피처 + LinearSVC(C=0.1) | - | ~0.673 solo | 피처만으로 +0.24 |
| 2 | ~07.03 | 이종 view 스태킹 | stacker: AAR 4-view SGD + transition prior | holdout 0.7098 | ~0.671 solo | linear와 유사 수준 |
| 3 | ~07.03 | 사전학습 인코더 | multilingual-e5-base 파인튜닝 (full 70k, s42, max_len 384) | - | ~0.701 solo | 단일 최강 성분 |
| 4 | ~07.03 | 이종 blend | 3-way uniform [1,1,1] (v1 인코더) | - | 0.7130 | blend 시작 |
| 5 | ~07.04 | 인코더 업그레이드 | v1 → v2 s42 (full 70k) | - | 0.7190 | +0.0060 |
| 6 | ~07.04 | 인코더 지분↑ | weights [1,1,1.5] (enc 0.43) | - | 0.7200 | +0.0010 |
| 7 | 07.04 | 인코더 지분↑↑ | **weights [1,1,2] (enc 0.50) = w112** 🏆 | - | **0.7208** | +0.0008, 체감 중. [1,1,2.5]는 로컬 하락 시작점 → 후순위 |
| 8 | 07.05 | R4: explore 4클래스는 계층 분류가 낫다 (밤샘 task2, 프로토타입 = linear 단독) | 1단계 family gate(F1 0.984) → 2단계 explore 전용 분류기. flat 0.6638 → override 0.6812 → strict route **0.6883** | 5-fold SGKF **+0.0246** (explore 4클래스 각 +0.020~0.028) | 미제출 | **생존** (독립 리뷰 통과: 누수 없음·수치 재현 확인). 단 **조건부** — hard-label 스왑이라 확률 blend와 비호환 → 확률 레벨(family 확률 마스킹) 재설계 + override/strict 각각 LB 게이트 후 승격. 리포트: night/2026-07-05/task2_report.md |
| 9 | 07.05 | R3: 첫 스텝(history=0) prior 보정 (밤샘 task2) | first-step log-prior bias, λ 그리드 | 최적 λ=0.125에서 **+0.0008** (첫스텝 자체 F1은 하락) | 미제출 | **보류** — calib_v1과 같은 유형(분포 피팅 bias)이라 LB 비전이 위험 대비 이득 없음 |
| 10 | 07.05 | w112 재조립: 원본 인코더(fp16)로 3-way 복원 | ai-2026 draft(linear+stacker+weights[1,1,2]) + artifacts/enc_v2_s42 fp16 → submit/ 스테이징 (커밋 e4cd2b4) | - (재현 제출) | **0.71884** | 기준선 복구 (팀 w112 0.7208 대비 −0.002 — fp16 재변환 or 체크포인트 미세 차이 추정, 추적 비용 대비 무가치). 제출 대장 #1 |
| 12 | 07.05 | **세션 형제 행 라벨 복원**: step k 라벨 == step k+1 history 마지막 assistant_action.name | train 검증 스크립트 (스크래치) | **train 58,326/58,326 쌍 100.00% 성립** (sim·au 계열 모두, gap1~6 231,664쌍 무예외) | **0.71884 (델타 0)** | **이득 0 확정** — test는 세션당 1스텝 샘플링(형제 행 없음). 상위권 0.79도 순수 모델링이라는 뜻. 폴백 안전하므로 보험 코드는 script.py에 잔류 (D-008). 점수 기대는 인코더 v3·R4로 이동 |
| 11 | 07.05 | blend 그리드 도구 (밤샘 task3) | scripts/blend/{collect_probs,grid_blend}.py — 성분별 holdout 확률 npz + 가중 그리드 | ⚠️ 수치 판정 불가 — stacker가 full-train 아티팩트라 holdout 누수 (0.7385는 오염값) | 미제출 | 도구만 채택. 사용하려면 성분 전부를 85% split로 재학습한 npz 필요 |
| 13 | 07.05 | **로컬 LB 시뮬레이션 리그 완성** | 평가행 = job3 holdout_base.npz의 9,969행, linear·stacker는 정직 OOF(artifacts/oof) id 조인 | 3-way [1,1,2] = **0.71726** vs 실제 LB 0.71884 (**오차 0.0016**) | - | 채택 — 후보 스크리닝은 LB 소모 없이 리그에서. 단 #15의 축별 한계 참고 |
| 14 | 07.05 | R4 확률 레벨 통합 (r4-integrate) | V1 override(gate행만 explore 질량 재배분) / V2 soft route(전 family 재조정) — 3-way 리그 베이스 | V1 **−0.0073**, V2 **−0.0158** (explore 4클래스 전부 하락) | 미제출 | **폐기** — linear 단독의 +0.0246은 강한 베이스에서 역전. 인코더+stacker가 explore를 specialist보다 잘 풂. 재시도 금지행 |
| 15 | 07.05 | serialize v3 (args+hist12+lang+elapsed) 프록시 A/B | scripts/encoding/serialize_ab.py — TF-IDF+LinearSVC 3-fold, 70k | v2 0.5017 / v3 **−0.031** / v3_no_args −0.026 / v3_hist6(v2+args) **−0.0036**, explore4 개선 없음(grep −0.007) | 미제출 (GPU 미투입) | **폐기** — hist 확장은 명백한 해, args는 explore를 못 올림(TF-IDF가 유리한 리터럴 신호인데도) → GPU 정당화 실패. 부산물: 실제 토크나이저 실측(v3는 384에서 29.3% 잘림 — chars/4 근사는 1.64배 과소) |
| 16 | 07.05 | blend 가중 재튜닝 (리그 그리드) | [2,2,1.75](enc 지분 0.30)가 로컬 +0.0093 | 로컬 0.72654 | 미제출 | **기각** — 핸드오프 §5 편향 재확인: 리그는 enc 지분 축에서 기울기가 LB와 반대 (LB 실측: 지분 0.33→0.50 단조 상승). 리그는 성분 추가/제거 판정용으로만, enc 지분 축은 LB만 신뢰 |
| 17 | 07.05 | meta-selector (코덱스 1순위 아이디어) | scripts/meta/meta_selector.py — 27피처(마진·불일치·rank) LogReg, 확신 임계 override, 9,969행 중첩 그룹 5-fold | **전 임계·전 설정 마이너스** (θ=0.9에서도 −0.003, fixed:broken = 1:8~17). oracle 상한 0.80163은 실재 확인 | 미제출 | **폐기** — 파생 피처에는 oracle 갭을 열 정보가 없음. R4와 동일 교훈: 약한 메타로 블렌드 못 이김. (후속 후보로 '성분 신뢰 3-way 분류'가 제안됐으나 저순위 보류) |
| 18 | 07.05 | 4-way: e5-small을 encoder_2로 추가 | submit/model/encoder_2 (fp16 235MB), 인코더 블록 uniform 평균, weights [1,1,2], zip 757MB | - | **0.71280 (−0.006)** | **폐기 (uniform 블록)** — small(솔로 할인 −0.019 가족)이 base와 5:5 평균되며 최강 성분을 희석. 제출 대장 #3. 가중 블록(base≫small)은 holdout_small.npz 도착 후 리그에서만 재검토 |
| 19 | 07.05 | history 유무 버킷별 blend 가중 (동료 리포 notes/bucket-blend-2026-07-04.md + bucket-weights JSON) | 동료 JSON을 리그에서 우리 성분으로 평가: hist_empty=[0.75,0.5,2], hist_present=[0.75,1,0.75] | 리그 **0.72477 (+0.0075** vs flat 0.71726**)**. 이득 전부 history-있음 버킷의 enc 지분 하향(0.30)에서 발생 | - | ⚠️ **판단 오류 기록**: "동료 0.7242 = 이 레시피"는 정황 추론(0.7200+프록시 델타 0.0036≈0.7236)이었을 뿐 LB 실측 아님 — 동료의 +0.0036도 **자기 프록시 측정치**였다. 이걸 "LB 검증"으로 격상해 #16 금지에 예외를 뚫은 것이 #20 실패의 원인 |
| 20 | 07.05 | 버킷 정밀화: history_presence(2버킷) → history_len3(0 / 1-4 / 5+) | script.py에 history_len3 스킴 추가 + model/bucket_weights.json: hist_0=[0.75,0.5,2], hist_1_4=[0.75,1,1], hist_5_plus=[0.75,1.5,0.75] | 리그 **0.72679** (flat 대비 +0.0095, 반반 안정성도 통과) | **0.71270 (−0.0061)** | **폐기** — 리그 +0.0095 → LB −0.0061 완전 역전. **enc 지분 축 신기루 3차 확인** (핸드오프 §5, #16에 이어). 반반 안정성 체크도 이 신기루는 못 거른다(리그 내부 일관성일 뿐). 대장 #4. 교훈: enc 지분을 낮추는 어떤 스킴(전역·버킷 불문)도 리그 점수 무효 — 85% 프록시 enc가 실전 enc보다 약해서 생기는 구조적 편향 |
| 21 | 07.05 | e5-small 가중 블록 (holdout_small.npz 도착, #18 후속) | 리그에서 버킷 3-way 위에 base:small 블록 비율 0.95:0.05~0.5:0.5 + 4번째 독립 성분 w4 0.1~0.5 전수 | small solo 0.59374 (base 0.70509). **전 구간 단조 마이너스** — 최소 희석(0.95:0.05)조차 −0.0006, w4=0.1도 −0.0001 | 미제출 | **e5-small 완전 폐기** — 어떤 가중으로도 기여 없음. #18 LB −0.006과 정합. 이질 성분은 밤샘 mdeberta로 |
| 22 | 07.06 night | char-ngram LinearSVC 독립 4번째 성분 (밤샘 task1) | `scripts/components/char_svm/train_oof.py`, char_wb(2,5) max_features=300k + LinearSVC(C=0.1, balanced), 3-fold SGKF OOF + 리그 w4 add-test | OOF **0.59369**, explore4 0.41445. 리그 baseline 0.717259, best w4=1.0 → **0.718163 (+0.00090)** | 미제출 | **FAIL/폐기** — PASS 기준 +0.002 미달. 28.2% disagreement로 다양성은 있으나 성분이 약해 3-way 보정 신호가 부족. reviewer가 FAIL 판정 타당성 확인(조인 assert·그리드 정확, w4 축은 오히려 리그에 유리한 방향인데도 미달) |
| 23 | 07.06 night | AU/SIM id-prefix 분리 + AU 전용 linear 라우팅 (밤샘 task3) | `sess_au` 5,025행만으로 TF-IDF+LinearSVC, 세션 Group 3-fold OOF → 리그 holdout AU 682행의 **예측을 교체**(하드 라우팅), SIM은 3-way 유지 | AU OOF **0.68001**, 리그 AU `0.51381→0.69035`, all **+0.00935**. 독립 검증: ① tester 전 수치 소수점 재현 ② reviewer가 OOF의 holdout 혼입(fold train의 ~13%) 발견 → **완전 격리 재검증(비holdout 4,343행만 학습)에서 오히려 +0.01143** ③ AU 약세는 전 성분 공통(lin 0.544/stk 0.492/enc 0.509 vs SIM 0.67~0.72) — enc 프록시 고유 아님 | 미제출 | **생존 — LB 게이트 후보 (회의적 해석 필수)**. ⚠️ 리그 델타 +0.0094는 #20 신기루(+0.0095→LB −0.0061)와 크기가 같고, AU 라우팅도 'AU 행에서 enc 지분 0'인 스킴이라 구조 유사 — 단 이번엔 성분 간 재가중이 아니라 **AU 전용 학습이 전 성분을 +0.15 이상 이기는** 마진이라 부호 반전 여지가 작다고 판단. 추가 위험: 로컬 test 스텁(5행)에 sess_au 0건 — test에 AU가 없으면 델타 0(무해). LB 게이트 1회로 판정 |

## ❌ 폐기 확정 — 재시도 금지 (검증 후 버린 것, 핸드오프 §6)

| 레버 | LB/결과 | 왜 |
|---|---|---|
| seed soup (soup2/3: s42+s7 가중치 평균) | 0.697 | seed별 head 초기화가 다른 basin → 파괴적 간섭 |
| R4 계층 분류 (explore gate+specialist) | 리그 −0.007~−0.016 | linear 단독에서만 +0.0246, 3-way 베이스에서 역전 — blend가 이미 explore를 더 잘 풂 (#14) |
| serialize 확장 (hist12 / args 추가) | 프록시 −0.031 / −0.004 | 길이 희석 실해 + args가 explore F1을 못 올림. hist12는 384 토큰에서 83% 잘림 문제도 (#15) |
| 세션 형제 행 라벨 복원 | LB 델타 0 | test가 세션당 1스텝 샘플링 — 복원 대상 없음 (#12, 보험 코드는 잔류) |
| e5-small 성분 추가 (모든 형태) | LB −0.006 / 리그 전 가중 마이너스 | uniform·가중 블록·4번째 독립 성분 전부 마이너스 — solo 0.594가 너무 약함 (#18·#21). 완전 폐기 |
| **enc 지분 낮추기 — 전역·버킷 불문 전면 금지** | LB 실측 −0.0061 (#20) | 리그 기울기가 enc 지분 축에서 LB와 반대 — **세 번 확인** (핸드오프 §5, #16, #20 LB 실측). 버킷별도 예외 아님 (#19의 '예외' 판단은 정황 추론을 LB 검증으로 오독한 오류). 리그는 오직 성분 추가/제거 판정용, enc 지분이 변하는 모든 스킴은 LB 게이트 필수 |
| calib_v1 (enc T=1.34 + class bias) | 0.7169 | holdout +0.005가 LB −0.002로 비전이 — train 분포 피팅 bias는 분포 이동에 취약 |
| flat 피처 추가 F~W | 이득 없음 | |
| stacker 변형 4종 | 이득 없음 | |
| cascade base | +0.0019 | 노이즈 수준 |
| max_len 512 추론 | ±0.000 | |
| 스키마에 없는 환각 피처 | 무효 | 입력에 존재하지 않는 필드 |
| 옛 회차(2025) 전략 | 무관 | Macro-F1 14-class 지표 불일치 |

## 로컬 CV → LB 할인율 (핸드오프 §5 — 반드시 할인해서 읽기)

| 성분 가족 | 할인 |
|---|---:|
| linear | −0.002 |
| encoder (base) | −0.015 |
| encoder (small/e8) | −0.019 |
| stacker | −0.033 |

- 로컬 그리드는 최적 enc 지분을 LB보다 **낮게** 잡는다 (로컬 최적 0.33 vs LB 최적 0.50) → 그리드는 방향·랭킹 선택기로만, 오른쪽 보정해서 읽기.
- 검증 프로토콜: **StratifiedGroupKFold**, 그룹키 = `-step_\d+$` 제거 (9,429 세션), 중요 판정은 3-fold.
- **per-class F1 항상 확인**: 약점 = 탐색 계열(read_file·grep_search·list_directory·glob_pattern). respond_only·write_file은 이미 1.0 — 건드리지 말 것.

## Agent Handoff Log

Sub-agent 작업 결과는 필요할 때 아래 형식으로 남긴다.

```text
Role:
Goal:
Files changed:
Validation:
Experiment log entry:
Open questions:
Next recommended owner:
```

<!-- (task2 브랜치의 중복 기록 섹션은 본 테이블 #8·#9로 통합됨 — 상세는 context/night/2026-07-05/task2_report.md) -->
