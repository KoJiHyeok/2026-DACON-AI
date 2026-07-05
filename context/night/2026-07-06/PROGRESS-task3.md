# task3 progress

## Checklist

- [x] Loaded `C:\dev\2026-AI-DACON\CLAUDE.md` and task instructions.
- [x] Add AU/SIM analysis script.
- [ ] Run AU/SIM distributions and component diagnostics.
- [x] Add AU-only linear probe script.
- [ ] Run AU-only linear probe.
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
- Added `scripts/au/analyze.py` and `scripts/au/probe_au_linear.py`.
- Syntax check passed: `python -m py_compile scripts\au\*.py`.

## Next resume point

Run `scripts/au/analyze.py`, then inspect AU/SIM distributions and component failures.
