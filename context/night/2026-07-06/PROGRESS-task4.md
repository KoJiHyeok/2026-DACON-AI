# task4 progress

## Checklist

- [x] Loaded `C:\dev\2026-AI-DACON\CLAUDE.md` and `context/night/2026-07-06/task4.md`.
- [x] Confirmed task3 baseline and holdout leakage caveat from `report_task3.md`.
- [x] Add isolated AU routing grid script under `scripts/au2/`.
- [x] Run syntax check.
- [x] Run baseline soft alpha grid.
- [x] Run AU-only C/features grid.
- [x] Run all-nonholdout sample-weight grid.
- [x] Write `report_task4.md`.
- [x] Create `task4.DONE`.
- [x] Independent reviewer/tester pass.
- [ ] Final commit (blocked by sandbox permission on external `.git` worktree).

## Notes

- Working tree started clean.
- `scripts/au/` and `submit/` are read-only for this task.
- Evaluation must reproduce 3-way `[1,1,2]` league baseline `0.7172592175`.
- Training for task4 candidates excludes every id in `holdout_base.npz`; AU evaluation uses the 682 holdout AU rows only.
- Soft alpha axis done: `au_only_word_char_C0.5` with alpha `0.9` scored league Macro-F1 `0.7325434019`, delta `+0.0059303543` vs task3 hard baseline `0.7266130476`.
- Same isolated hard route (`alpha=1.0`) scored `0.7286929405`; the soft gain over same-protocol hard is `+0.0038504614`.
- Commit attempt after soft axis failed: `.git` points outside the writable sandbox (`C:\dev\2026-AI-DACON\.git\worktrees\task4`), so `index.lock` cannot be created under current permissions.
- AU-only grid done: best is `au_only_char_C1` with alpha `0.9`, league Macro-F1 `0.7331367018`, delta `+0.0065236542` vs task3 hard baseline.
- SIM/all-nonholdout weight axis done: best is `auWeight10` with alpha `0.7`, league Macro-F1 `0.7231096234`, delta `-0.0035034242`; this axis fails.
- Aggregated outputs written: `night_out/task4/summary_all.json`, `route_rows.csv`, `per_class_best_vs_blend.csv`.
- Report written: `context/night/2026-07-06/report_task4.md`.
- DONE written: `context/night/2026-07-06/task4.DONE`.
- Independent reviewer PASS: no blocking findings; verified holdout exclusion, 3-way assert, AU-only replacement, artifact/report consistency, and no forbidden prior/blend/submit source changes. Low note: ignored `submit/__pycache__/au_route.cpython-313.pyc` exists from import.
- Independent tester PASS: py_compile, collector rerun, summary/DONE consistency assertions, and `git status --short -- submit` all passed.

## Next resume point

Task complete except final git commit, which is blocked by current sandbox permissions for the external git worktree.
