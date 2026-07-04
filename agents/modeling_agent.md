# Modeling Agent

## Mission

Tier 0/1 모델을 빠르게 재현 가능하게 학습하고, 신뢰 가능한 GroupKFold Macro-F1을 만든다.

## Read First

- `agents/common.md`
- `src/features.py`
- `src/train.py`
- `scripts/validate_submit.py`
- `context/experiments.md`

## Scope

- Tier 0: official TF-IDF + LogisticRegression baseline reproduction.
- Tier 1: TF-IDF/char n-gram + meta/transition features + linear/GBDT candidates.
- OOF predictions and fold metrics.
- Save artifacts under `submit/model/`.

## Required Implementation Properties

- Deterministic seed.
- GroupKFold by `session_id`.
- Macro-F1 over all `ACTION_CLASSES`.
- Configurable experiment name.
- Persist enough metadata to reproduce:
  - git commit or dirty status
  - package versions
  - feature mode/config
  - fold scores

## Deliverables

- Working `src/train.py` command.
- Model artifacts in `submit/model/`.
- Experiment row in `context/experiments.md`.

## Do Not

- Do not optimize against the 5-row local `test.jsonl`.
- Do not treat public leaderboard gains as reliable without local CV support.
