# Common Context

## Competition

- Task: `current_prompt` + `history` + `session_meta`로 다음 agent action 14-class 분류.
- Metric: Macro-F1. 희소 클래스(`web_search`, `write_file`, `lint_or_typecheck`, `plan_task`, `ask_user`) 성능이 중요하다.
- Submit: `submit.zip` 루트에 `model/`, `script.py`, `requirements.txt`.
- Runtime: offline, inference <= 10min, install <= 10min, zip <= 1GB, Python only.

## Data Contract

- Train files: `data/train.jsonl`, `data/train_labels.csv`.
- Test files at evaluation: `data/test.jsonl`, `data/sample_submission.csv`.
- Sample fields: `id`, `current_prompt`, `history`, `session_meta`.
- `id` format: `sess_sim_...-step_NN`. Group key is the prefix before `-step_`.

## Evaluation Contract

- Use session-prefix GroupKFold. Random split is not trusted because multiple steps from the same session appear in train.
- Report Macro-F1 with all 14 labels, `zero_division=0`.
- Keep OOF predictions when possible. They are needed for calibration, threshold tuning, and ensemble decisions.

## Engineering Contract

- `src/features.py` owns all reusable feature extraction.
- `src/train.py` trains and writes artifacts under `submit/model/`.
- `src/infer.py` is the local inference source; `submit/script.py` is the packaged server entrypoint.
- `scripts/make_submit.py` packages; `scripts/validate_submit.py` simulates server execution.
- No network calls in submitted code. Local pretrained models must be bundled and loaded from local paths.

## Logging Contract

Every meaningful experiment entry in `context/experiments.md` should include:

- hypothesis
- changed files/config
- CV protocol
- local Macro-F1
- per-class weaknesses if available
- leaderboard score if submitted
- decision: keep, revert, or investigate
