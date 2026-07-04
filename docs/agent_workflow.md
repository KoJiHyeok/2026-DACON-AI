# Agent Workflow

이 문서는 `PLAN.md`의 실험 전략을 실제 병렬 작업 단위로 쪼갠 운영 문서다.

## Phase 0: Pipeline First

Owner: Modeling Agent + Submission Agent

Goal: 점수와 무관하게 학습 -> 모델 저장 -> 추론 -> submit.zip -> 로컬 검증 흐름을 완성한다.

Exit criteria:

- `src/train.py`가 Tier 0 모델을 만든다.
- `submit/script.py`가 `output/submission.csv`를 만든다.
- `scripts/make_submit.py` 검증이 통과한다.
- `context/experiments.md`에 baseline row가 기록된다.

## Phase 1: Data Understanding

Owner: EDA Agent

Goal: 구현 우선순위가 있는 피처 후보를 만든다.

Exit criteria:

- class distribution, session leakage risk, transition matrix, meta correlations가 `context/research.md`에 정리된다.
- P0/P1/P2 feature backlog가 생긴다.

## Phase 2: Tier 1 Loop

Owner: Feature Agent + Modeling Agent + Review Agent

Goal: 빠른 CV 루프를 돌려 leaderboard 제출 후보를 만든다.

Loop:

1. Feature Agent가 한 번에 작은 feature batch를 구현한다.
2. Modeling Agent가 GroupKFold CV와 OOF를 생성한다.
3. Review Agent가 누수/불일치/제출 위험을 확인한다.
4. 유의미한 개선만 Submission Agent가 패키징한다.

Exit criteria:

- 로컬 CV가 baseline 대비 명확히 개선된다.
- per-class weakness가 기록된다.
- 제출 가능한 zip이 최소 1개 있다.

## Phase 3: Tier 2 Gate

Owner: Encoder Agent

Goal: 사전학습 인코더가 제출 제약 안에서 의미 있는 개선을 낼지 판단한다.

Start only if:

- Tier 1 CV와 제출 파이프라인이 안정화됨.
- 남은 시간이 학습/패키징/검증까지 충분함.

Exit criteria:

- feasibility note가 있음.
- 구현한다면 모델 크기, 추론 시간, requirements impact가 검증됨.

## Phase 4: Ensemble and Finalization

Owner: Ensemble Agent + Submission Agent + Review Agent

Goal: OOF 기반 앙상블/threshold 튜닝으로 최종 후보를 고른다.

Exit criteria:

- 최종 후보 2-3개가 `context/experiments.md`에 비교되어 있음.
- 각 후보의 zip, CV, LB, risk note가 명확함.
- 최종 제출 전날 밤까지 validate 통과.

## Communication Contract

각 sub-agent 결과는 아래 형식으로 남긴다.

```text
Role:
Goal:
Files changed:
Validation:
Experiment log entry:
Open questions:
Next recommended owner:
```
