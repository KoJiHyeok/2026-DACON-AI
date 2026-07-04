# task1 report — two_way linear+stacker

## Summary

Built a 2-way submission candidate at `submit_candidates/two_way/` using the existing linear champion and AAR stacker artifacts. The encoder block from `ensemble/script_3way.py` was removed, so the default blend is:

```text
prediction = argmax((linear_proba + stacker_proba) / 2)
```

Optional override remains available via `ENS_WEIGHTS="lin,stk"` or `model/weights.json`, but no weights file is included. Default behavior is uniform `[1, 1]`.

## Assembly

Source repository was read-only: `C:\dev\dacon-agent-action-api-boost`.

Copied artifacts:

| Source | Destination | Purpose |
|---|---|---|
| `linear_pipeline\submit\model\model.pkl` | `submit_candidates\two_way\model\linear\model.pkl` | linear champion pipeline + class bias |
| `model\aar_config.json` | `submit_candidates\two_way\model\stacker\aar_config.json` | AAR stacker component config |
| `model\aar_models.joblib` | `submit_candidates\two_way\model\stacker\aar_models.joblib` | AAR component and stacker models |

Not copied:

| File | Reason |
|---|---|
| `model\model.joblib` | used by stacker solo fallback script, not by AAR stacker path in `aar_config.json` |
| `model\prompt_model.joblib` | same as above |
| encoder directories | intentionally excluded for this 2-way candidate |
| `torch`, `transformers` dependencies | encoder removed, not required |

`submit_candidates/two_way/script.py` is self-contained. It vendors the required linear feature builders and AAR inference helpers because the project submit gate packages only `script.py`, `requirements.txt`, and `model/`.

`requirements.txt` intentionally contains comments only, following the team policy not to repin server baseline packages.

## Verification

Smoke test:

```powershell
$env:ENS_DATA='C:\dev\2026-AI-DACON\data'
$env:ENS_OUT='C:\dev\night\2026-07-05\task1\root_dir_probe'
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe script.py
```

Result: success, 5-row `submission.csv` generated.

Zip validation:

```powershell
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe C:\dev\2026-AI-DACON\scripts\validate_submit.py task1_two_way_submit.zip --data-dir C:\dev\2026-AI-DACON\data
```

Result: 12/12 PASS. Full log: `context/night/2026-07-05/task1_validate.log`.

Key validation numbers:

- zip size: 39.5 MB
- offline inference: 52.8 seconds
- output rows: 5/5
- id order: PASS
- action labels: PASS

## Expected LB

Known solo references from handoff:

- linear solo LB: 0.6732
- stacker solo LB: 0.6708
- uniform linear+stacker estimate: around 0.69

This candidate should be materially below the known 3-way w112 result (LB 0.7208), because the encoder block was the strongest solo component and is intentionally absent here.

## Recommendation

Worth keeping as a validated fallback and packaging rehearsal. I would not submit it over w112 unless the morning encoder rebuild is unavailable and a clean 2-way fallback is needed. Its main value is that the script/model layout and 12-check path are now proven; adding the encoder back for 3-way should be a smaller integration step.

## Commit status

Commit was attempted but blocked by the current sandbox permissions. This worktree's `.git` file points to `C:\dev\2026-AI-DACON\.git\worktrees\task1`, and `git add` failed while creating `index.lock` there with `Permission denied`. No zip is staged or retained; the remaining manual step is to run `git add` and `git commit` from an environment with write access to that external gitdir.
