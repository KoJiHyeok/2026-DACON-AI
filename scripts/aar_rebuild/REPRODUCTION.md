# AAR reproduction report

## Status

The rebuild is a CPU-only, consumer-compatible implementation of the lost
`train_tscar.py` path. The surviving metadata records only `4-view SGD
candidates + greedy_blend inner selection + oof_stack_validation(logreg)` and
`stack_folds=3`; the original work2 repository and its AAR binaries were not
present at the supplied paths.

The trainer uses four views (`full`, `prompt_context`, `history`, `action`),
`SGDClassifier(loss="log_loss")`, fixed seed 42, a configurable TF-IDF cap
(default 50,000), and session groups obtained from `id.rsplit("-step_", 1)[0]`.
These are reconstruction choices, not claims of byte-level parity.

## Validation

The included `tests/test_aar_rebuild.py` trains a 42-row, 3-session fixture,
checks deterministic three-fold OOF metrics, verifies artifact keys, and loads
saved component probabilities into the saved logistic stacker. A full run is:

```powershell
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/aar_rebuild/train_aar.py
```

The command writes `aar_models.joblib` and `aar_config.json` to the selected
output directory (default `submit/model`). The historical reference was OOF
approximately 0.71 and exp #2 holdout 0.7098; no honest numeric comparison or
row-match rate can be reported without the missing original OOF/model files.
The trainer prints its measured three-fold OOF Macro-F1.

Two full-data attempts were made with the supplied CPU environment (first at
180k features/max-iter 40, then at 50k/max-iter 3); both exceeded the 10-minute
local execution window before writing a completed artifact. The focused fixture
run completes in about 37 seconds and proves the same serialization path.

## Parity limits

Exact prediction parity is not recoverable because the original view builder,
SGD alpha/iteration schedule, vectorizer limits, candidate list, greedy tie
breaking, logreg regularization, initialization details, sklearn version at
training time, and original AAR artifact are missing. `submit/aar_infer.py`
was left unchanged and remains the compatibility oracle.
