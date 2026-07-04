# task2 report — R4 explore hierarchy + R3 first-step prior

## Scope

- Data: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv`
- Split: `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)`, group key = `id` without trailing `-step_\d+`
- Baseline: E_+seq-like sparse features + `LinearSVC(C=0.1, class_weight="balanced", max_iter=1000)`
- Scripts:
  - `scripts/hierarchy/proto_hier.py`
  - `scripts/hierarchy/first_step_bias.py`

`LinearSVC` emitted convergence warnings at `max_iter=1000`; the same setting is used for all compared variants, so the local deltas are still apples-to-apples. Treat absolute scores as prototype-CV numbers, not a production ensemble estimate.

## R4: flat 14-way vs hierarchy

| model | 5-fold Macro-F1 | std | explore 4-class Macro-F1 | std |
|---|---:|---:|---:|---:|
| A. flat 14-way | 0.66378 | 0.00165 | 0.51042 | 0.00545 |
| B1. family gate -> explore override only | 0.68124 | 0.00097 | 0.53129 | 0.00352 |
| B2. family gate -> strict family route | 0.68834 | 0.00162 | 0.53538 | 0.00335 |

Stage-1 family classifier quality:

| mean metric | value |
|---|---:|
| family Macro-F1 | 0.98377 |
| explore precision | 0.98587 |
| explore recall | 0.98134 |
| valid explore rate | 0.41117 |

Explore per-class F1, mean across folds:

| action | flat | override | strict route | route delta vs flat |
|---|---:|---:|---:|---:|
| read_file | 0.48536 | 0.51183 | 0.51344 | +0.02808 |
| grep_search | 0.56927 | 0.58713 | 0.58956 | +0.02029 |
| list_directory | 0.41405 | 0.43643 | 0.43785 | +0.02379 |
| glob_pattern | 0.57299 | 0.58975 | 0.60069 | +0.02771 |

Interpretation: R4 survives fold-valid testing. The first-stage family decision is almost saturated, so the R1 caveat ("last2/last1 only helps after explore is known") is addressed by the hierarchy. The strict route is best locally, but it changes non-explore decision boundaries too; the lower-risk next probe is to package both override and strict-route variants and LB-gate them separately.

## R3: first-step prior/bias

Train-only first-step prior shift is large and directionally matches R1:

| action | global rate | first-step rate | log shift |
|---|---:|---:|---:|
| write_file | 0.02116 | 0.07867 | +1.312 |
| list_directory | 0.06184 | 0.20200 | +1.182 |
| plan_task | 0.03827 | 0.12456 | +1.179 |
| apply_patch | 0.06890 | 0.00067 | -4.453 |
| respond_only | 0.07397 | 0.00856 | -2.143 |
| edit_file | 0.15959 | 0.02533 | -1.837 |

Bias grid result, applied only to validation rows with `n_history == 0`:

| lambda | Macro-F1 | first-step Macro-F1 | non-first Macro-F1 |
|---:|---:|---:|---:|
| 0.125 | 0.66460 | 0.40053 | 0.66341 |
| 0.000 | 0.66378 | 0.42136 | 0.66341 |
| 0.250 | 0.66370 | 0.36690 | 0.66341 |
| 0.500 | 0.66096 | 0.29593 | 0.66341 |

The tiny best local setting (`lambda=0.125`, +0.00082 overall) is not robust-looking: first-step subset Macro-F1 falls, stronger values degrade quickly, and the old `calib_v1` lesson was exactly that train/holdout-fitted bias can fail on LB. This should be treated as a `calib_v1`-style risky calibration knob, not as a ready production change. If used at all, gate it behind a holdout/LB probe and keep the strength very small.

## Recommendation

1. Promote R4 to the next implementation probe. Package two variants: explore override and strict family route. The local gain is large enough that LB validation is worth a submission slot.
2. Do not ship R3 by itself. At most test `lambda=0.125` as a small ablation after the hierarchy route is settled.
3. Keep the report caveat: these are local 5-fold group-split results, not leaderboard evidence.

## Artifacts

- `scripts/hierarchy/_out/proto_hier_summary.csv`
- `scripts/hierarchy/_out/proto_hier_fold_metrics.csv`
- `scripts/hierarchy/_out/proto_hier_per_class_f1.csv`
- `scripts/hierarchy/_out/proto_hier_split_stats.csv`
- `scripts/hierarchy/_out/first_step_bias_summary.csv`
- `scripts/hierarchy/_out/first_step_bias_fold_metrics.csv`
- `scripts/hierarchy/_out/first_step_bias_per_class_f1.csv`
- `scripts/hierarchy/_out/first_step_bias_prior_shift.csv`
