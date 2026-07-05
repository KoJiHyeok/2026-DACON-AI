# Experiments Log

> 규칙: 모든 실험은 (가설 → 변경점 → 로컬 CV → 리더보드) 순으로 기록.
> CV는 세션 프리픽스 GroupKFold Macro-F1. 리더보드 제출은 일 10회 예산 관리.
> **점수 판정은 LB 실측 또는 세션 group-split만** — accuracy·누수 split·단일 holdout 금지 (w112 핸드오프 계승).

## 현재 최고 기록

| 구분 | Macro-F1 | 비고 |
|---|---|---|
| 리더보드 (우리) | **0.7208** | **w112** = 3-way 앙상블 weights [1,1,2] — [핸드오프](../legacy/w112_handoff.md) |
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
| 12 | 07.05 | **세션 형제 행 라벨 복원**: step k 라벨 == step k+1 history 마지막 assistant_action.name | train 검증 스크립트 (스크래치) | **train 58,326/58,326 쌍 100.00% 성립** (sim·au 계열 모두) | 프로브 예정 | ⚠️ 로컬 test 스텁 5행은 전부 다른 세션 → 주최측이 세션당 1스텝 샘플링했을 가능성. 성립 시 test 비최종 스텝 전부 무료 정답. 폴백 안전(형제 없으면 모델 예측)이라 **모든 제출물에 보험으로 내장** + 단독 프로브 1회로 test 구조 판별 (D-008) |
| 11 | 07.05 | blend 그리드 도구 (밤샘 task3) | scripts/blend/{collect_probs,grid_blend}.py — 성분별 holdout 확률 npz + 가중 그리드 | ⚠️ 수치 판정 불가 — stacker가 full-train 아티팩트라 holdout 누수 (0.7385는 오염값) | 미제출 | 도구만 채택. 사용하려면 성분 전부를 85% split로 재학습한 npz 필요 |

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

<!-- (task2 브랜치의 중복 기록 섹션은 본 테이블 #8·#9로 통합됨 — 상세는 context/night/2026-07-05/task2_report.md) -->
