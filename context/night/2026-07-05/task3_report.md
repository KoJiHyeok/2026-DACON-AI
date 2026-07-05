# task3 report: 2-way holdout blend

## 요약

- 구현:
  - `scripts/blend/collect_probs.py`
  - `scripts/blend/grid_blend.py`
- 실행 로그: `context/night/2026-07-05/task3_run.log`
- 산출 NPZ:
  - `context/night/2026-07-05/holdout_linear.npz`
  - `context/night/2026-07-05/holdout_stacker.npz`
- grid 결과: `context/night/2026-07-05/grid_blend_linear_stacker.json`

## Holdout split

- split: `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)` 첫 fold
- group: `id.rsplit("-step_", 1)[0]`
- stratify: `action`
- rows: train 59,999 / holdout 10,001 (14.287%)
- sessions: train 8,079 / holdout 1,350 / overlap 0
- 주의: `valid_frac=0.15`는 `round(1 / 0.15) = 7`이라 실제 holdout은 약 14.3%다. `colab/holdout_eval.py`와 같은 방식.

## Component scores

| component | source | holdout Macro-F1 | caveat |
|---|---:|---:|---|
| linear | `linear_pipeline/features.py`, `E_+seq`, `LinearSVC` | 0.632926 | 85% split only fit, honest holdout |
| stacker | `C:\dev\dacon-agent-action-api-boost\model\aar_*` | 0.738490 | existing full artifact inference; leakage-prone |

Linear fit emitted a sklearn `ConvergenceWarning`; probabilities were still saved. If this score is used for decision-making, rerun with higher `--linear-max-iter`.

## Grid result

Search space: each weight 0..3, step 0.25, excluding all-zero.

| rank | weights `[linear, stacker]` | Macro-F1 |
|---:|---:|---:|
| 1 | `[0.25, 2.5]` | 0.739693 |
| 2 | `[0.25, 3.0]` | 0.739613 |
| 3 | `[0.25, 2.75]` | 0.739598 |
| 4 | `[0.25, 2.25]` | 0.739596 |
| 5 | `[0.25, 2.0]` | 0.739295 |
| 6 | `[0.25, 1.5]` | 0.739053 |
| 7 | `[0.5, 3.0]` | 0.739053 |
| 8 | `[0.75, 2.75]` | 0.738950 |
| 9 | `[0.25, 1.75]` | 0.738932 |
| 10 | `[0.25, 1.0]` | 0.738886 |

Equal 2-way `[1, 1]`: 0.732777. On this holdout, adding a small amount of linear to stacker improves slightly over stacker alone, but equal weighting is worse.

## sim/au check

Holdout composition:

- `sess_sim_*`: 9,320 rows (93.19%)
- `sess_au_*`: 681 rows (6.81%)

Macro-F1 by bucket:

| bucket | linear | stacker | best `[0.25,2.5]` | equal `[1,1]` |
|---|---:|---:|---:|---:|
| all | 0.632926 | 0.738490 | 0.739693 | 0.732777 |
| sess_sim | 0.638168 | 0.750017 | 0.751367 | 0.742121 |
| sess_au | 0.489808 | 0.530596 | 0.528233 | 0.546964 |

The global best weight is driven by the dominant `sess_sim_*` bucket and slightly hurts `sess_au_*` versus stacker alone. Equal `[1,1]` is worse overall but better on `sess_au_*`.

## Encoder warning

`grid_blend.py` aligns `actions` and `ids`, but encoder/LB weights should not be trusted unless encoder probabilities are collected on this exact grouped holdout. A Colab encoder NPZ from a different split or class order will produce misleading weights even if its LB score is strong.

## Conclusion

- The scripts now provide the requested local CPU workflow for linear/stacker NPZ collection and grid search.
- Best observed 2-way local holdout weight: `[linear=0.25, stacker=2.5]`, Macro-F1 0.739693.
- Do not treat the stacker score or the best blend as an unbiased holdout result until the stacker can be retrained on the same 85% split. Current stacker artifact likely saw the holdout labels during its original training.
