# Reproducibility playbook v2 report

## Outcome

The reproduction playbook and verifier now describe submission #14, public LB
0.77089. The active inference composition is linear + AAR + one Qwen2.5-0.5B
encoder block, outer weights `[1,1,2]`, encoder-block weights `[1.0]`, and
soft-AU alpha 0.9. `encoder_2`/mBERT was removed from the active component set.

The current authoritative staging tree was read only. No file under `submit/`,
the 2026-07-13 worktree, or the main worktree was modified; no zip, network
call, push, or submission was performed.

## Changes

- `docs/repro_playbook.md` was rebuilt around champion #14. It includes the
  exact Qwen full-training command, hist12 serialization, fast-AAR and
  length-sorted batching, the T4 rehearsal cross-reference, and the six current
  deployed artifact hashes.
- The feature-source ambiguity is explicit: `submit/features.py` is the
  deployed linear/pickle contract imported by `submit/script.py`;
  `src/features.py` is training-side/general source and must not be substituted.
- `scripts/repro_rehearsal/verify.py` now exposes exactly four components:
  `aar`, `linear`, `qwen-encoder`, and `au`. It rejects `encoder_2`, checks
  Qwen2 fp16/24-layer/14-label config, max_hist 12, both weight vectors,
  fast-AAR invocation, length sorting and row-order restoration, deployed
  feature import, soft-AU default, and all six exact deployed hashes.
- Existing 2026-07-13 AAR/linear CPU rehearsal outputs are reused read-only.
  AAR/linear/AU OOF manifests are fully rehashed on every run.
- Qwen full mode has no surviving local run JSON or SHA256 manifest. The
  verifier therefore reports its training evidence as `documented-only` while
  separately passing the immutable deployed-byte and runtime-config contract.

## Deployed snapshot checked

| Component | Files | Result |
|---|---:|---|
| AAR | model, config, `fast_aar.py` | 3/3 size + SHA256 PASS |
| linear | deployed pickle | 1/1 size + SHA256 PASS |
| Qwen encoder | fp16 safetensors | 1/1 size + SHA256 PASS |
| AU | deployed pickle | 1/1 size + SHA256 PASS |

The Qwen safetensors measurement was 988,122,712 bytes with SHA256
`0e9f798c58b4334861376a2f8372ee0d41d38a88e3fc38854427488e54e5a056`.
The fast-AAR vendor measurement was 17,401 bytes with SHA256
`7ccbb898a9e9103ff889c603be94378db04bef264c11dea0a18fa0b6132b2042`.
All other exact values are in playbook section 7 and the verifier constants.

## Validation

Environment: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (the requested
server-mirror environment).

```text
python -m py_compile scripts/repro_rehearsal/verify.py tests/test_repro_rehearsal.py
PASS (exit 0)

python -m pytest -q tests/test_repro_rehearsal.py
..........                                                               [100%]
10 passed in 0.15s
```

The tests cover mixed-log parsing, PowerShell UTF-16 logs, tolerance boundaries,
manifest corruption, a valid Qwen-only runtime contract, rejection of
`encoder_2`, deployed hash drift, the exact component set, concise-status gap
preservation, and no-required-argument parsing.

Final static audit results:

```text
static_audit=pass
reported_json_matches_verifier_summary=true
components=aar,linear,qwen-encoder,au
authoritative_submit_tracked_status=clean
git diff --check: PASS
```

## Full `verify.py` stdout

Command:

```powershell
& C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\repro_rehearsal\verify.py
```

Output (complete):

```json
{
  "status": "pass",
  "champion": {
    "submission": 14,
    "public_lb": 0.77089
  },
  "external_root": "C:\\dev\\2026-AI-DACON",
  "repro_root": "C:\\dev\\night\\2026-07-13\\task2\\out_repro",
  "components": [
    {
      "component": "aar",
      "status": "pass",
      "checks": {
        "deployed_artifacts": {
          "status": "pass",
          "files_checked": 3
        },
        "metric": {
          "status": "pass",
          "files_checked": 1,
          "measured": 0.7034006994681244,
          "expected": 0.7034,
          "tolerance": 0.005
        },
        "repro_files": {
          "status": "pass",
          "files_checked": 2
        },
        "artifact_manifest": {
          "status": "pass",
          "files_checked": 6
        }
      }
    },
    {
      "component": "linear",
      "status": "pass",
      "checks": {
        "deployed_artifacts": {
          "status": "pass",
          "files_checked": 1
        },
        "metric": {
          "status": "pass",
          "files_checked": 1,
          "measured": 0.6636584439296592,
          "expected": 0.663895,
          "tolerance": 0.005
        },
        "repro_files": {
          "status": "pass",
          "files_checked": 2
        },
        "artifact_manifest": {
          "status": "pass",
          "files_checked": 6
        }
      }
    },
    {
      "component": "qwen-encoder",
      "status": "pass",
      "checks": {
        "deployed_artifacts": {
          "status": "pass",
          "files_checked": 1
        },
        "runtime_contract": {
          "status": "pass",
          "checks_passed": 15,
          "checks_total": 15,
          "training_evidence": "documented-only"
        }
      }
    },
    {
      "component": "au",
      "status": "pass",
      "checks": {
        "deployed_artifacts": {
          "status": "pass",
          "files_checked": 1
        },
        "metric": {
          "status": "pass",
          "files_checked": 1,
          "measured": 0.7031536844011531,
          "expected": 0.703154,
          "tolerance": 0.005
        },
        "artifact_manifest": {
          "status": "pass",
          "files_checked": 7
        }
      }
    }
  ]
}
```

## Routing and repository-state notes

- Routed audit requested: `gpt-5.6-sol`, reasoning `high`, sandbox
  `read-only`, `ROUTED_TASK=1`.
- Routed result: no model response. The local sandbox blocked both WebSocket
  and HTTPS access to `api.openai.com`; validation therefore relied on the
  deterministic verifier, focused tests, and local static inspection.
- Routed branch/commit: N/A (read-only attempt).
- Working branch: `night/2026-07-14/task1`, base `0554a88`.
- Commit attempts are blocked by filesystem policy: Git needs to create
  `C:\dev\2026-AI-DACON\.git\worktrees\task11\index.lock`, which is outside
  the writable workspace root and returns `Permission denied`. The worktree
  files are complete, but no honest commit SHA can be reported from this
  sandbox.

## Remaining evidence gap

To upgrade Qwen from `documented-only` training evidence to manifest-verified,
the control owner should export the full run console/config and a SHA256
manifest from `~/out/qwen05i_2ep_full`, then tie the fp16 model hash to the
deployed `0e9f798c...` artifact. This is evidence preservation only; no model
retraining or resubmission is required by this task.
