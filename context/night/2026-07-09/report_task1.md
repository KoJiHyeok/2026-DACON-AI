# task1 report - serialize contract regression test

## Verdict

PASS. Added `tests/test_serialize_contract.py` to lock the duplicated `_bucket()` and `serialize()` helpers against silent drift. No production serialization code was changed.

## What the test checks

- Parses source files with Python `ast`; target scripts are never imported, so training/inference dependencies are not loaded.
- Finds exactly one `FunctionDef` named `_bucket` and one named `serialize` in each target file.
- Removes a function docstring node before dumping, then compares `ast.dump(..., annotate_fields=True)` against `submit/script.py` as the reference.
- Checks char-cap/source invariants inside each `serialize()` source span:
  - `[:800]` for query text
  - `[:120]` for `result_summary`
  - `[:200]` for user content
  - `open_files[:5]`
  - `reversed(hist[-max_hist:])`
- Proves the guard is not always-pass by mutating the reference `serialize()` AST constant `120` to `121` and asserting the comparison returns false.

## Target results

| File | `_bucket()` AST | `serialize()` AST | char-cap guards |
|---|---:|---:|---:|
| `submit/script.py` | reference | reference | PASS |
| `colab/encoder_v2_s42_repro.py` | PASS | PASS | PASS |
| `colab/mdeberta_finetune.py` | PASS | PASS | PASS |
| `colab/encoder_e5_holdout85_maxhist.py` | PASS | PASS | PASS |

## Verification

- Focused run: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest tests/test_serialize_contract.py -q`
  - `11 passed in 0.43s`
- Full run: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest tests -q`
  - `27 passed in 27.65s`
  - Log: `context/night/2026-07-09/task1_run.log`

## Limitations

- The AST comparison intentionally ignores comments, whitespace, line numbers, and function docstrings.
- The source-level char-cap checks are presence guards inside `serialize()`, not a full behavioral property test over generated samples.
- If all four copies are intentionally changed together, AST equality will pass; the char-cap checks only catch required-pattern removal or spelling drift.
- Reviewer/tester subagents were not spawned because the available multi-agent tool policy permits spawning only when the user explicitly requests delegation or subagents. This report records that gap; pytest verification was still completed.

## Operational note

The requested git commit could not be created from this sandbox. `git add`/`git commit` fail because Git cannot create `C:/dev/2026-AI-DACON/.git/worktrees/task1/index.lock`; the external gitdir is read-only to this session.
