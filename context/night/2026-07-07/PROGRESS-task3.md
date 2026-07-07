# task3 progress

## Checklist

- [x] Read `CLAUDE.md`, `task3.md`, and AU routing precedent.
- [x] Implement shared subroute utilities and 4-way join sanity.
- [x] Run join sanity.
- [ ] Commit checkpoints (blocked: gitdir is outside writable workspace).
- [x] Run first-pass sweep and record screening candidates.
- [x] Run specialist probes for screened groups.
- [x] Write `report_subroute.md`.
- [x] Create `task3.DONE`.
- [ ] Final commit (blocked: gitdir is outside writable workspace).

## Current notes

- Working tree started clean.
- `scripts/subroute/` did not exist at start; task3 will add it.
- Required sanity gates passed via `scripts/subroute/sweep.py --sanity-only`:
  - 3-way macro-F1 = `0.71725922`
  - 4-way macro-F1 = `0.72254583`
- `sess_au` and `turn_index == 0` must be excluded from candidate routing.
- Checkpoint commit attempt after sanity failed: `.git` points to `C:/dev/2026-AI-DACON/.git/worktrees/task3`; creating `index.lock` there returned `Permission denied` under the current sandbox.
- First-pass sweep output:
  - `night_out/task3_subroute/sweep.json`
  - `night_out/task3_subroute/sweep_rows.csv`
  - `night_out/task3_subroute/sweep_screen_pass.csv`
- Screen pass candidates in weakness order:
  1. `cross:open_files_empty&git_dirty=false` — holdout `1324`, train(nonholdout) `8140`, group F1 `0.583398`, delta `-0.139147`, turn0 share `0.0`
  2. `open_files_empty` — holdout `3243`, train `18604`, group F1 `0.633040`, delta `-0.089505`, turn0 share `0.0`
  3. `turn_index>=12` — holdout `608`, train `3657`, group F1 `0.633106`, delta `-0.089440`, turn0 share `0.0`
  4. `git_dirty=false` — holdout `1891`, train `11886`, group F1 `0.635397`, delta `-0.087148`, turn0 share `0.0`
  5. `turn_index>=8` — holdout `2158`, train `13548`, group F1 `0.660674`, delta `-0.061872`, turn0 share `0.0`
  6. `turn_index>=10` — holdout `1174`, train `7369`, group F1 `0.670218`, delta `-0.052328`, turn0 share `0.0`
  7. `history_len>=10` — holdout `3822`, train `23311`, group F1 `0.687130`, delta `-0.035416`, turn0 share `0.0`
- Probe output:
  - `night_out/task3_subroute/probe.json`
  - `night_out/task3_subroute/probe_route_rows.csv`
- Probe result: no candidate reaches `+0.005` LB gate or `+0.002` report-only band. Best final delta is `+0.000584` (`turn_index>=12`, alpha `0.6`), and all specialist hard margins are negative, so all are `discard_info_limited`.
- Report written: `context/night/2026-07-07/report_subroute.md`.
- DONE written: `context/night/2026-07-07/task3.DONE`.
- Independent tester: `py_compile` PASS, `sweep.py --sanity-only` PASS, `sweep.json` screen pass count `7`, `probe.json` results count `7`, best delta `+0.000583783`, no LB/report candidate.
- Independent reviewer: no blocking issues. Cross-check confirmed report matches artifacts, masks exclude `sess_au`, screened/probed candidates have `0` turn0 rows, and all hard margins are negative. Non-blocking note: scripts rely on stored holdout split being group-disjoint; reviewer verified actual split overlap is `0`.

## Next resume point

No modeling work remains. If write access to `C:/dev/2026-AI-DACON/.git/worktrees/task3` is restored, commit the task3 files.
