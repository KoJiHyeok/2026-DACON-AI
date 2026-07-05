# task3 progress

## Checklist

- [x] Loaded `C:\dev\2026-AI-DACON\CLAUDE.md` and task instructions.
- [x] Add AU/SIM analysis script.
- [x] Run AU/SIM distributions and component diagnostics.
- [x] Add AU-only linear probe script.
- [x] Run AU-only linear probe.
- [ ] Write `task3_report.md`.
- [ ] Create `task3.DONE`.
- [ ] Independent review/test evidence.
- [ ] Final commit.

## Notes

- Working tree starts clean.
- Read-only external inputs:
  - `C:\dev\2026-AI-DACON\data`
  - `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04`
  - `context/night/2026-07-05/holdout_base.npz`
- Added `scripts/au/common.py`, `scripts/au/analyze.py`, and `scripts/au/probe_au_linear.py`.
- Syntax check passed: `python -m py_compile scripts\au\*.py`.
- `scripts/au/analyze.py` completed. Join assert: 3-way blend Macro-F1 `0.7172592175`.
- Holdout rows: 9,969 total, AU 682 (6.84%), SIM 9,287.
- Component AU macro-F1: linear 0.5437, stacker 0.4920, encoder 0.5087, blend 0.5138.
- `scripts/au/probe_au_linear.py` completed. AU-only 3-fold OOF Macro-F1 `0.680006`.
- Routing probe on league holdout: blend all `0.717259` -> AU-specialist hybrid `0.726613`, delta `+0.009354`.

## Next resume point

Write `task3_report.md`, then create DONE and run independent review/test.
