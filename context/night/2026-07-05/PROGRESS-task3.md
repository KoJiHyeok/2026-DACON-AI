# task3 progress

## 2026-07-05

- Started task3 blend holdout workflow.
- Confirmed project rule: session-prefix grouped validation is mandatory.
- Added `scripts/blend/collect_probs.py` for common StratifiedGroupKFold 85/15 holdout probability collection.
- Added `scripts/blend/grid_blend.py` for aligned NPZ blend grid search and top-k reporting.
- Ran local linear refit + stacker artifact inference into holdout NPZs.
- Ran 2-way grid search. Best `[linear, stacker] = [0.25, 2.5]`, Macro-F1 0.739693.
- Wrote `task3_report.md` and `task3.DONE`.
