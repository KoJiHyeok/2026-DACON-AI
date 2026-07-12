# AAR reproduction report

## Status: corrected re-implementation

The original `train_tscar.py` source file (from the deleted `work2`
checkout referenced in `artifacts/oof/oof_rebuild_2026_07_04/meta.json`) is
still not on disk. However, **the trained artifacts it produced are on disk
and were not lost**:

* `submit/model/stacker/aar_config.json` (1.4 KB)
* `submit/model/stacker/aar_models.joblib` (47 MB)

Both were opened directly (`json.load` / `joblib.load`) to recover the exact
recipe. See `SCHEMA.md` for the full structural writeup. In short: 3 text
SGD components (`prompt_context_sgd`, `prompt_sgd`, `action_sgd`, each an
`sklearn.pipeline.Pipeline` of TF-IDF + `SGDClassifier(loss="log_loss")`
with per-component `alpha`/`max_features` read off the live objects) plus a
`transition_prior` rule-based empirical-frequency component, hstacked (in
`stacker_components` order) into a `LogisticRegression(C=1.0,
class_weight="balanced", max_iter=500, solver="lbfgs", random_state=42)`.

## Prior attempt (retracted)

A previous pass at this rebuild asserted the original 31.5 MB AAR binary and
its JSON config "did not exist at the supplied paths" and invented a
different, unverified 4-view all-sklearn recipe (`full/prompt_context/
history/action` views, components named `sgd_<view>`, no transition
component, uniform weight 1.0, a config schema that didn't match the real
one). That assertion was false — both files are present at the paths named
in this task's briefing — and was made without opening either file. This
report and the schema file replace that draft; the "not present" language
has been fully removed, not just amended.

## Reproduction as validated in this pass

### 1. Structural fidelity (exact)

`train_aar` writes an artifact dict with the same 5 top-level shapes as the
original (`components` dict with the 3 correct names, `transition` dict with
the 5 correct keys, `stacker` a `LogisticRegression`, `actions`,
`metadata`), and an `aar_config.json` with the real `stacker_components`
order and component `weight` values copied from the live config. This was
verified two ways:

* `tests/test_aar_rebuild.py` fits a small fixture and asserts every key and
  shape.
* A live smoke run (see below) produces a `model/aar_models.joblib` that
  `submit/aar_infer.py`'s unmodified `predict_aar()` loads and scores
  **without any code changes to the consumer** — confirmed by actually
  calling `aar_infer.predict_aar(records, texts, prompt_texts, config)` in a
  fresh working directory pointed at the smoke artifact and checking the
  returned labels are all valid `ACTIONS`.

### 2. Row-match rate against the real original artifact (measured, not estimated)

A subset trainer run (3000 of the 70000 rows, `max_iter=25` — the real
value read off the live SGD objects) was compared against the **actual
original `aar_models.joblib`** (copied read-only from
`submit/model/stacker/`) on the same first 200 rows of `train.jsonl`, both
going through the unmodified `aar_infer.predict_aar` path:

```
row match rate (rebuilt 3000-row subset vs. real full-70k artifact): 0.765
```

This number is expected to rise substantially with the full 70k rows (the
subset model has ~4% of the real model's training data and materially
smaller effective TF-IDF vocabularies), and is reported honestly as a
subset-vs-full comparison, not a full-vs-full one — no full 70k retraining
was run in this pass (see "Full run not executed" below).

Macro-F1 was **not** used for this comparison beyond a sanity check, because
these 200 rows are in-sample training rows for both artifacts (not a held-out
split), so per-row F1 there mostly reflects memorization rather than
generalization. The row-match rate between two independently fit
same-architecture models is the informative number here.

### 3. OOF Macro-F1 vs. the 0.7098 target (subset, explicitly labeled)

Using the trainer's own internal 3-fold session-group OOF evaluation
(`GroupKFold` on `id.rsplit("-step_", 1)[0]`, matching the recipe note in
`meta.json`):

| rows (of 70000) | stacked OOF macro-F1 | wall time |
|---|---|---|
| 3000  | 0.599 | 42.4 s |
| 8000  | 0.638 | 101.3 s |
| 70000 (target, from `aar_config.json`) | 0.7098 | not run |

The upward trend with more rows (0.599 → 0.638 as rows go 3000 → 8000) is
consistent with converging toward the 0.7098 target as data grows to the
full 70000 rows; it is not itself evidence of exact parity, only of the
recipe moving in the right direction at the right order of magnitude.

### 4. Full 70k-row training time: extrapolated, not measured

Two real data points (3000 rows -> 42.4 s wall, 8000 rows -> 101.3 s wall)
give a power-law fit exponent of ~0.89 (sub-linear scaling, as expected
since TF-IDF vocabularies saturate against their `max_features` caps).
Extrapolating to 70000 rows:

* Power-law extrapolation: ~695 s (~11.6 min)
* Linear extrapolation from the 8000-row point: ~886 s (~14.8 min)

So a full 70k-row training run is estimated at **roughly 12-15 minutes**
wall time on this machine. This is training-time only and is separate from
the submission's 10-minute **inference** budget, which `aar_infer.py`
already satisfies unchanged. The full run was deliberately not executed in
this pass per the task instructions; the decision to spend ~15 minutes on a
full retrain is left to the main session.

## Parity limits (honest, narrowed from the retracted draft)

Recoverable from the surviving files and verified in this pass: component
names, view assignment, exact TF-IDF/SGD hyperparameters per component,
transition table group names/weights/blend formula, stacker hyperparameters
and input column order. These are no longer "estimates" -- they are read
directly from the live objects.

Not recoverable, even with the real artifacts in hand:

* The original row iteration/shuffle order used during SGD fitting (SGD
  fits are order-sensitive; a different row order under the same seed can
  produce different final coefficients even with identical hyperparameters).
* Whether the original transition table was built over the full 70000 rows
  or some other split before being frozen into the artifact (the value
  itself was recovered exactly for the *whole-artifact* case — the `global`
  vector and `groups` tables in the real `aar_models.joblib` are directly
  copyable if ever needed — but this rebuild recomputes them from
  `train.jsonl`/`train_labels.csv` rather than reusing the original tables
  verbatim, since the goal is a re-trainable pipeline, not a byte copy).
* The original sklearn version at training time (irrelevant to functional
  parity here since `aar_infer.py`'s consumer contract is what's actually
  exercised at inference, and that contract is verified compatible).

## How to run

```powershell
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/aar_rebuild/train_aar.py
```

Writes `aar_models.joblib` and `aar_config.json` to `--output` (default
`submit/model`). Supports `--limit N` to subsample rows for a fast smoke
run (used throughout this report), and `AAR_*` environment variable
fallbacks for every flag (no required argparse arguments).
