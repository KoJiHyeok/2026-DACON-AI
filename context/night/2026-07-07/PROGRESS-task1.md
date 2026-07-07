# task1 progress

## Checklist

- [x] enc_block_weights regression tests
- [x] league4 rebuild baseline
- [x] league4 block ratio grid
- [x] league4 AU alpha grid
- [x] report and DONE marker
- [x] reviewer/tester verification
- [ ] git commits (blocked by sandbox permission on external git metadata)

## Log

- Started from `context/night/2026-07-07/task1.md`; no prior progress file existed.
- Confirmed `submit/script.py` already reads `enc_block_weights.json` with `utf-8-sig`.
- Added `tests/test_enc_block_weights.py` for BOM JSON, length mismatch, negative weights, missing config, and env override.
- `pytest` is not installed in `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` or the local conda Python; direct harness against `submit/script.py::enc_block_weights` passed all five cases.
- Git commit attempt failed because `.git/worktrees/task1/index.lock` is under `C:\dev\2026-AI-DACON`, outside the writable sandbox.
- Added `scripts/league4/common.py` and `rebuild.py`; `rebuild.py` passed sanity joins and wrote `night_out/league4/rebuild.{json,csv}`.
- Rebuild scores: 3-way `0.717259`, raw 4-way `[1.2,0.8]` `0.722546`, 4-way + soft-AU alpha 0.9 `0.738772`.
- Added and ran `scripts/league4/grid_block.py`; wrote `night_out/league4/grid_block.{json,csv}`.
- Block best was e5 `1.15` / mBERT `0.85`, final `0.739046`, delta `+0.000273` vs B4.
- Added and ran `scripts/league4/grid_alpha.py`; wrote `night_out/league4/grid_alpha.{json,csv}`.
- Alpha best stayed at `0.90`, final `0.738772`; alpha `0.95` was `-0.000389`, alpha `1.00` was `-0.003025`.
- Wrote `context/night/2026-07-07/report_league4.md` and `context/night/2026-07-07/task1.DONE`.
- Local verification: `py_compile scripts/league4/*.py` passed; `tests/test_features.py` fallback passed; direct `enc_block_weights` harness passed.
- `pytest` remains unavailable: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest ...` -> `No module named pytest`.
- Reviewer result: script/data/report review found no leakage or join issue, but marked FAIL because final git commit is impossible in this sandbox and the DoD includes a commit.
- Tester result: PASS for venv availability, fallback test, py_compile, rebuild rerun, and output inspection; FAIL/NOT RUN only for pytest due missing package.
- Cleaned tester-created temporary validation files; remaining untracked outputs are the authored task artifacts and `night_out/league4/` result artifacts.

## Next resume point

Only unresolved item is the requested git commit; it requires write access to `C:\dev\2026-AI-DACON\.git\worktrees\task1`.
