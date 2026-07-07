# PROGRESS-task2

- worktree: `C:\dev\night\2026-07-08\task2`
- lane: linear replacement candidates; outputs under `night_out/linear2/`
- fold contract: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\fold_indices.json`

## Status

- Completed. Baseline reproduction passed tolerance, 9 sweep configs completed, and every linear replacement candidate was discarded.

| variant | oof_f1 | league | delta | decision |
|---|---:|---:|---:|---|
| char_2-5_mf120k_C1 | 0.620063 | 0.738260 | -0.000513 | discard |
| char_3-5_mf120k_C1 | 0.618173 | 0.738181 | -0.000592 | discard |
| char_2-4_mf120k_C1 | 0.616013 | 0.737933 | -0.000839 | discard |
| char_3-5_mf120k_C2 | 0.613350 | 0.737882 | -0.000891 | discard |
| char_3-5_mf120k_C0.5 | 0.618553 | 0.737865 | -0.000907 | discard |
| union_2-5_mf120k_C1 | 0.617453 | 0.737716 | -0.001056 | discard |
| union_3-5_mf120k_C1 | 0.616445 | 0.737655 | -0.001117 | discard |
| char_3-5_mf200k_C1 | 0.617955 | 0.737590 | -0.001183 | discard |
| char_3-5_mf300k_C1 | 0.617785 | 0.737590 | -0.001183 | discard |

## Baseline

- rebuilt OOF macro-F1: `0.663895`
- reference OOF macro-F1: `0.663307`
- delta: `+0.000588` (`< 0.002`, pass)
- reference B4+soft-AU league baseline: `0.738772`

## Next resume point

No resume needed. See `context/night/2026-07-08/report_linear2.md` and `task2.DONE`.
