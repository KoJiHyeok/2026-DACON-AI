# Ensemble Agent

## Mission

OOF predictions를 사용해 Tier 1/Tier 2 모델 조합과 class-wise calibration/threshold를 테스트한다.

## Read First

- `agents/common.md`
- `context/experiments.md`
- Available OOF/model metadata under `submit/model/` or experiment outputs

## Scope

- Probability averaging or weighted averaging.
- Class-wise threshold/bias tuning for Macro-F1.
- Confusion matrix and per-class deltas.
- Stability check across folds.

## Deliverables

- Selected ensemble recipe with weights/biases.
- Reproducible artifact saved with model metadata.
- `context/experiments.md` entry showing per-class tradeoffs.

## Do Not

- Do not tune on public leaderboard feedback alone.
- Do not keep ensemble components that cannot be reproduced or packaged.
