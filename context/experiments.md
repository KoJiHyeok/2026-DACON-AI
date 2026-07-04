# Experiments Log

> 규칙: 모든 실험은 (가설 → 변경점 → 로컬 CV → 리더보드) 순으로 기록.
> CV는 세션 프리픽스 GroupKFold Macro-F1. 리더보드 제출은 일 10회 예산 관리.
> **점수 판정은 LB 실측 또는 세션 group-split만** — accuracy·누수 split·단일 holdout 금지 (w112 핸드오프 계승).

## 현재 최고 기록

| 구분 | Macro-F1 | 비고 |
|---|---|---|
| 리더보드 (우리) | **0.7208** | **w112** = 3-way 앙상블 weights [1,1,2] — [핸드오프](../legacy/w112_handoff.md) |
| 커트라인(12등) | 0.77585 | 2026.07.04 기준 (핸드오프 시점 0.77426) — 갭 **−0.055** |
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

## ❌ 폐기 확정 — 재시도 금지 (검증 후 버린 것, 핸드오프 §6)

| 레버 | LB/결과 | 왜 |
|---|---|---|
| seed soup (soup2/3: s42+s7 가중치 평균) | 0.697 | seed별 head 초기화가 다른 basin → 파괴적 간섭 |
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

## 2026-07-05 task2 — R4/R3 local CV probes

| # | 날짜 | 가설/행동 | 변경점 | 로컬 CV | LB | 결론 |
|---|---|---|---|---|---|---|
| 8 | 07.05 | R4 explore 계층 분류가 flat 14-way보다 탐색 4클래스 약점을 줄일 수 있다 | `scripts/hierarchy/proto_hier.py`: E_+seq-like sparse features + `LinearSVC(C=0.1, balanced)`, 1차 family gate, 2차 explore branch | flat 0.66378 → explore override 0.68124 → strict family route 0.68834. explore 4-class Macro-F1 0.51042 → 0.53538 | - | 로컬 강한 양성. LB 게이트용 후보로 승격 |
| 9 | 07.05 | R3 첫 스텝 class-wise prior bias가 history 없음 구간을 보정한다 | `scripts/hierarchy/first_step_bias.py`: train-fold first-step log prior shift를 `n_history==0` valid row에만 lambda 적용 | best lambda 0.125: 0.66378 → 0.66460(+0.00082). first-step subset Macro-F1은 0.42136 → 0.40053로 하락 | - | 약한/위험한 보정. `calib_v1` 실패 전례 때문에 단독 채택 금지, LB 게이트 필수 |
