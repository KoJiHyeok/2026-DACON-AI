# Experiments Log

> 규칙: 모든 실험은 (가설 → 변경점 → 로컬 CV → 리더보드) 순으로 기록.
> CV는 세션 프리픽스 GroupKFold Macro-F1. 리더보드 제출은 일 10회 예산 관리.

## 현재 최고 기록

| 구분 | Macro-F1 | 비고 |
|---|---|---|
| 로컬 CV | - | |
| 리더보드 | - | |
| 커트라인(12등) | 0.77585 | 2026.07.04 기준 |

## 실험 기록

| # | 날짜 | 가설 | 변경점 | 로컬 CV | LB | 결론 |
|---|---|---|---|---|---|---|
| 0 | | 베이스라인 재현으로 파이프라인 검증 | TF-IDF+LogReg | | | |

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
