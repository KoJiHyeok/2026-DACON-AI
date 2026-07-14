# Champion #14 reproducibility playbook

This playbook describes the deployed submission #14 champion (public LB
**0.77089**, submission commit `d16fbc8` and later main state): linear + AAR +
a **single Qwen2.5-0.5B encoder block**, blended with three-slot weights
`[1, 1, 2]`, then soft-AU routing with alpha 0.9. The inference path uses
length-sorted Qwen batches and `fast_aar.py`. mBERT/`encoder_2` is not an active
component.

This document trains or verifies components. It does not authorize packaging,
submission, or changes under `submit/`. Before any future large model-family
change, also follow the mandatory [Colab T4 rehearsal contract](t4_rehearsal.md).

## 1. Evidence levels and fixed inputs

Use these labels when reporting a rehearsal result:

| Label | Meaning |
|---|---|
| measured-now | Executed in the 2026-07-13 CPU rehearsal or hashed directly from the current staging tree |
| manifest-verified | Existing run JSON and every file named by its SHA256 manifest were checked |
| documented-only | Recipe or metric comes from tracked records but lacks a surviving local run manifest |

Fixed local inputs:

```text
DATA=C:\dev\2026-AI-DACON\data
PYTHON=C:\dev\2026-AI-DACON\.venv\Scripts\python.exe
EXTERNAL=C:\dev\2026-AI-DACON
PRIOR_REPRO=C:\dev\night\2026-07-13\task2\out_repro
REPRO=<this worktree>\out_repro
```

PowerShell setup for a new CPU reproduction:

```powershell
$env:DATA = 'C:\dev\2026-AI-DACON\data'
$env:PYTHON = 'C:\dev\2026-AI-DACON\.venv\Scripts\python.exe'
$env:EXTERNAL = 'C:\dev\2026-AI-DACON'
$env:REPRO = (Join-Path (Get-Location) 'out_repro')
```

The CPU environment measured on 2026-07-13 was Python 3.13.13, NumPy 2.5.0,
SciPy 1.18.0, scikit-learn 1.8.0, and joblib 1.5.3 on Windows 11. AAR and
`fast_aar.py` require scikit-learn 1.8.0; the private TF-IDF implementation is
part of the verified inference contract. The GPU/server record names PyTorch
2.7.1, Transformers 5.13.1, and Accelerate 1.14.0. `requirements.txt` still
uses `transformers>=4.51`, so the historical environment is evidence rather
than a complete package lock.

## 2. Exact deployed inference contract

The active model graph is:

```text
linear probability ─────┐
AAR probability ────────┼─ weighted blend [1, 1, 2]
Qwen probability ───────┘        │
  encoder block = [encoder]       └─ sess_au rows only:
  enc_block_weights = [1.0]          0.9 * AU + 0.1 * blend
```

The required staging facts are:

- `submit/model/encoder` is the only encoder directory; `encoder_2` must be
  absent. Its config is `Qwen2ForSequenceClassification`, `model_type=qwen2`,
  fp16, 24 layers, and 14 labels.
- `submit/model/encoder/serialize_config.json` is `{"max_hist": 12}`. The
  inference serializer consumes the newest 12 history entries.
- `submit/model/enc_block_weights.json` is `[1.0]`; the outer blend weights in
  `submit/model/weights.json` are `[1.0, 1.0, 2.0]` for linear, AAR, and the
  Qwen block.
- AU is not a fourth outer-blend slot. On `sess_au` rows it applies
  `0.9 * P_au + 0.1 * P_blend`; all other rows retain the three-slot blend.
- `submit/script.py` calls `fast_aar.fast_predict_proba`. The vendor
  `aar_infer.py` remains the serialization/artifact contract and fallback
  reference. The optimized path was verified on 5,000 rows with 100% argmax
  agreement and probability max absolute error 0.0.
- Qwen inference sorts serialized inputs by length before batching and writes
  each batch back to its original row indices. This changes padding work, not
  output order.

### Which `features.py` is authoritative?

`submit/features.py` is the **deployed linear inference contract**. The staged
`script.py` prepends its own directory to `sys.path` and executes
`import features as F`, so replacing it with `src/features.py` can silently
change the feature vector paired with the deployed pickle. `src/features.py`
is the training-side/general project source; it is not a drop-in deployment
substitute. The linear reproduction command below intentionally uses the
tracked `submit/features.py` feature set.

## 3. Required execution order

1. Confirm inputs, the six deployed hashes in section 7, and free disk space.
   Keep external data, artifacts, and `submit/` read-only.
2. Run AAR alone. It is CPU-heavy and historical timing became invalid when
   another trainer competed for CPU.
3. Run linear after AAR. It builds the feature dataframe once and performs
   three full fits plus inner class-bias fits.
4. On a GPU host, train the Qwen full model using the exact two-epoch hist12
   recipe. Do not add the retired mBERT block.
5. Train AU only on `sess_au` rows and apply alpha 0.9 only on that mask.
6. Run the verifier. Then run a full 30,000-row Colab T4 rehearsal according
   to `docs/t4_rehearsal.md`; a five-row local smoke cannot establish the
   ten-minute runtime gate.

## 4. CPU components

### AAR stacker — measured-now target

Run from the repository root in PowerShell:

```powershell
& $env:PYTHON scripts/aar_rebuild/train_aar.py `
  --data "$env:DATA\train.jsonl" `
  --labels "$env:DATA\train_labels.csv" `
  --output "$env:REPRO\aar\model" `
  --seed 42 --max-iter 25 --folds 3 `
  *> "$env:REPRO\aar\train.combined.log"
```

Contract: session prefix (`-step_N` removed) + `GroupKFold(3)`; three TF-IDF
SGD log-loss views and a transition prior feed a balanced logistic-regression
stacker. Expected files are `aar_models.joblib` and `aar_config.json`. Parse
`stacked_oof_macro_f1` from the final JSON object in the log. Accept
**0.7034 ± 0.005**. The 2026-07-13 rehearsal measured 0.7034006995 in
1,008.5 seconds.

This is a corrected reimplementation, not byte reproduction. The deleted
original trainer's SGD row/shuffle order and sklearn version are unrecoverable.
The surviving original artifact reported 0.7098; the rebuild anchor previously
matched original predictions on 957/1000 sampled rows.

The deployed predictor uses `submit/fast_aar.py`, not a changed model. Its
speed path retains the fitted vectorizers/classifiers and exact probability
contract; any sklearn version change requires the 5,000-row equivalence gate
again.

### linear — measured-now target

```powershell
& $env:PYTHON scripts/linear2/baseline_repro.py `
  --data-dir $env:DATA `
  --oof-dir "$env:EXTERNAL\artifacts\oof\oof_rebuild_2026_07_04" `
  --out-dir "$env:REPRO\linear" `
  --all-folds --evaluate --tune-bias --force `
  --seed 42 --bias-seed 43 --max-iter 1000 --close-tol 0.005
```

Contract: saved three session-group folds from the 2026-07-04 OOF; feature set
`E_+seq` from `submit/features.py`; balanced `LinearSVC(C=0.1)`; inner
`GroupShuffleSplit(test_size=0.2, random_state=43)` for class bias. Expected
outputs are `fold{1,2,3}_{probs.npy,meta.json}`, `repro_probs.npy`, and
`summary.json`. Accept OOF Macro-F1 **0.663895 ± 0.005**. The 2026-07-13
rehearsal measured 0.6636584439 in 1,305.7 seconds.

## 5. Qwen2.5-0.5B encoder — active GPU component

The tracked runner is `colab/mdeberta_finetune.py`; the `MDEB_*` prefix is a
historical runner name and does not mean mBERT is deployed. The full 70k
Qwen2.5-0.5B-Instruct two-epoch recipe is:

```bash
source ~/venv-speed/bin/activate
export MDEB_MODEL=Qwen/Qwen2.5-0.5B-Instruct
export MDEB_MODE=full MDEB_MAXHIST=12
export MDEB_SEED=42 MDEB_EPOCHS=2
export MDEB_BATCH=8 MDEB_ACCUM=2 MDEB_MAXLEN=384 MDEB_LR=2e-5
export MDEB_GRAD_CKPT=1 MDEB_DATA_DIR=~/data
export MDEB_OUT=~/out/qwen05i_2ep_full
python colab/mdeberta_finetune.py
```

Training is fp32 with gradient checkpointing on A5000; the runner writes
`model_fp32/` first and creates the deployed `model_fp16/` only afterward.
The effective batch size is 16. Hist12 serialization, max length 384, seed 42,
lr 2e-5, and two epochs are contractual. The 4-epoch variant overfit by ep3
(0.74569) and is not the champion recipe.

Tracked experiment #52 records holdout85 solo Macro-F1 0.75932 for instruct-2ep
(base-2ep 0.75941) and the complete hybrid at 0.76760, +0.01160 over the prior
holdout blend. Submission #14 produced LB 0.77089. These scores are
**documented-only** here: full mode emitted no local run JSON, and the server
directory `~/out/qwen05i_2ep_full` has no local SHA256 manifest in this
workstation. The current deployed safetensors hash in section 7 is therefore
the immutable local byte anchor, not proof of byte-identical retraining.

### Runtime gate

Submission #13 exceeded the T4 ten-minute limit. With length-sorted batching
and `fast_aar`, the 30,000-row Colab T4 rehearsal measured 515 seconds:
6.6s load, 7.0s linear, 29.7s AAR, 471.5s Qwen. Submission #14 then completed
server scoring successfully. These optimizations are part of the champion
contract, not optional post-processing.

## 6. AU char_wb specialist

The tracked five-fold evidence generator is:

```powershell
& $env:PYTHON scripts/oof5/gen_oof_au.py --c 1.0 --max-iter 5000 --seed 42
```

It uses `au_route.serialize`, char_wb TF-IDF 3–5 grams with 120,000 maximum
features, and balanced `LinearSVC(C=1.0)`. Expected evidence is five sparse AU
OOF NPZs, `au_row_mask.npz`, `run_oof_au.json`, and `SHA256SUMS` under
`artifacts/experiments/oof_au/`. Coverage is 5,025/70,000 rows. Accept pooled
AU-subset Macro-F1 **0.703154 ± 0.005**. Recorded five-fold wall time is 114s.

The deployed full specialist is `submit/model/au_linear/model.pkl`. Its exact
full-train CLI is not preserved, so the OOF generator proves the recipe and
split behavior but not byte-identical deployed-model regeneration.

## 7. Deployed artifacts measured on 2026-07-14

These six hashes were measured directly from the current authoritative
`C:\dev\2026-AI-DACON\submit` staging tree. They are a snapshot, not permission
to alter the files.

| Component | File | Bytes | SHA256 |
|---|---|---:|---|
| linear | `submit/model/linear/model.pkl` | 8,354,283 | `ebc07c26455c3e2e93d4d662abb3ba14876636904f8c69288fd58179c72a8877` |
| AAR | `submit/model/stacker/aar_models.joblib` | 47,420,632 | `31b10456d072ce0e7e4a868c1f02fe7451cf90fe786de53104fed3dec1e0ed6d` |
| AAR | `submit/model/stacker/aar_config.json` | 1,402 | `f7eb7d95003bf003cf6cdd68c593547b486bb7a67ae201c644589a70fb362e27` |
| AAR fast path | `submit/fast_aar.py` | 17,401 | `7ccbb898a9e9103ff889c603be94378db04bef264c11dea0a18fa0b6132b2042` |
| Qwen encoder | `submit/model/encoder/model.safetensors` | 988,122,712 | `0e9f798c58b4334861376a2f8372ee0d41d38a88e3fc38854427488e54e5a056` |
| AU | `submit/model/au_linear/model.pkl` | 17,673,981 | `bc01eb659eca930bcad238d9210beb6c2c72d11b4cdbb778fc136a0bd98725e0` |

There is deliberately no `encoder_2` row. The previous e5/mBERT hashes belong
to retired champion configurations and must not be accepted for #14.

## 8. Machine verification

The no-argument command reuses the immutable 2026-07-13 CPU rehearsal outputs
for AAR/linear while checking current manifests and deployed bytes:

```powershell
& $env:PYTHON scripts/repro_rehearsal/verify.py
& $env:PYTHON scripts/repro_rehearsal/verify.py --component qwen-encoder
& $env:PYTHON scripts/repro_rehearsal/verify.py --component aar
& $env:PYTHON scripts/repro_rehearsal/verify.py --output "$env:REPRO\verification.json"
& $env:PYTHON scripts/repro_rehearsal/verify.py --details
```

For a newly generated CPU rehearsal, pass
`--repro-root "$env:REPRO"`. The verifier checks:

- active component names are AAR, linear, Qwen encoder, and AU;
- exact sizes and SHA256 for all six section-7 artifacts;
- absence of `encoder_2` and exact Qwen/config/blend/serialize contracts;
- `fast_aar` invocation, length-sorted batching, original-order restoration,
  deployed `features.py` import, and soft-AU default;
- fresh AAR/linear metrics and required outputs; and
- all files named by the surviving AAR/linear/AU OOF manifests.

A top-level `"status": "pass"` means those claims passed. Stdout is concise by
default; `--details` prints every file row, and `--output` always writes that
detailed report. A pass does not erase the
Qwen training-evidence gap; that component reports
`"training_evidence": {"status": "documented-only", ...}` separately.

## 9. Known irreproducibility gaps

- Qwen full mode has no surviving local run JSON, console log, or SHA256
  manifest tying `~/out/qwen05i_2ep_full` to the deployed fp16 bytes.
- AAR's original source, SGD row order, and training sklearn version are lost;
  the corrected rebuild is metric reproduction rather than byte reproduction.
- Linear reconstruction depends on saved folds and the exact staged
  `submit/features.py`; swapping in `src/features.py` invalidates the contract.
- AU has a complete five-fold evidence CLI but no preserved standalone
  full-train CLI/build record for its deployed pickle.
- GPU records do not freeze Python, CUDA, driver, hardware, and all packages.
- SHA256 protects stored bytes from later mutation. It cannot prove that a
  generator paired each probability row with the correct id at creation time.
