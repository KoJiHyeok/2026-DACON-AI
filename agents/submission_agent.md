# Submission Agent

## Mission

제출 패키지를 만들고, 평가 서버 계약을 로컬에서 최대한 엄격하게 검증한다.

## Read First

- `agents/common.md`
- `submit/script.py`
- `submit/requirements.txt`
- `scripts/make_submit.py`
- `scripts/validate_submit.py`

## Checklist

- `submit.zip` root contains only the expected runtime files.
- `script.py` creates `output/submission.csv`.
- Output columns and id order match `sample_submission.csv`.
- All predictions are in `ACTION_CLASSES`.
- Requirements are pinned with `==`.
- No network calls.
- Zip <= 1GB.
- Inference time <= 10min, with local warning if >8min.

## Deliverables

- Validated `submit/submit.zip`.
- Validation output summary.
- Submit candidate row in `context/experiments.md`.

## Do Not

- Do not change model behavior unless packaging requires it.
- Do not include training-only files or raw data in submit.zip.
