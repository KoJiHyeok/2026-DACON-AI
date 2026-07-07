# merge080 regression test report

## Verdict

PASS. Added focused regression coverage for the risky merge080 mBERT path without changing `submit_candidates/merge080/script.py`.

## Coverage

- `tests/test_merge080_script.py` loads `submit_candidates/merge080/script.py` with local stubs for `torch`, `transformers`, `joblib`, and `scipy`.
- The loader isolates `src` imports by clearing cached `src*` modules and prepending `submit_candidates/merge080` to `sys.path`, then restores the prior interpreter state.
- `encoder_predict(..., align_by_label=True)` is tested with fake logits and an alphabetically ordered `id2label`, verifying that output columns are reordered back to `ACTIONS`.
- `main()` mBERT blend block is tested with patched dependencies:
  - `MBERT_MIX=0.2` over config `mix=0.0` produces `(1-w) * final + w * mbert` and renormalizes rows to 1.
  - `mix=0.0` leaves the final probabilities unchanged and does not call the mBERT encoder path.
- `validate_mbert` coverage:
  - positive mix plus missing directory raises `FileNotFoundError`;
  - zero mix skips validation;
  - bad label set raises `RuntimeError`;
  - `MBERT_MIX` env overrides config mix in both skip and fail-fast directions.

## Production diff

No production code was changed. `submit_candidates/merge080/script.py` remains byte-for-byte untouched in this task, so there is no algebraic behavior change to justify.

## Verification

- `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest -q tests\test_merge080_script.py`
  - `8 passed in 0.92s`
- `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest -q tests\`
  - `16 passed in 4.50s`

## Operational note

The requested git commits could not be created in this sandbox. `git add` and `git commit` both fail because Git cannot create `C:/dev/2026-AI-DACON/.git/worktrees/task1/index.lock`; the external gitdir is read-only to this session.
