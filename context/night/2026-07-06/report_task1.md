# task1 report - char-ngram SVM fourth component

## Summary

Verdict: **FAIL**.

The independent char-ngram SVM is a valid OOF component, but it does not clear
the add-test gate. Best league result is w4=1.0 with +0.000903 over the
validated 3-way baseline, below the required +0.002 threshold.

## Method

- Data: `C:\dev\2026-AI-DACON\data\train.jsonl` + `train_labels.csv`
- Split: 3-fold `StratifiedGroupKFold`, group key = id with `-step_\d+$` removed
- Text view: full `current_prompt`, compact `act:<name>` history tokens, action-order pairs, and bucketed `session_meta`/workspace fields
- Vectorizer: `TfidfVectorizer(analyzer="char_wb", ngram_range=(2,5), max_features=300000, sublinear_tf=True, dtype=float32)`
- Classifier: `LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)`
- Probability conversion: `softmax(decision_function)` aligned to the 14 action classes

## OOF Results

| Fold | Valid rows | Features | Macro-F1 | Explore4 Macro-F1 |
|---:|---:|---:|---:|---:|
| 0 | 23,334 | 248,213 | 0.595070 | 0.411814 |
| 1 | 23,333 | 248,042 | 0.592800 | 0.415487 |
| 2 | 23,333 | 248,074 | 0.593134 | 0.415899 |

Overall OOF Macro-F1: **0.593688**  
Explore4 OOF Macro-F1: **0.414448**

Explore4 per-class F1:

| Class | F1 |
|---|---:|
| read_file | 0.374675 |
| grep_search | 0.405477 |
| list_directory | 0.437338 |
| glob_pattern | 0.440304 |

## League Add-Test

Sanity baseline `[linear, stacker, encoder] = [1,1,2]`: **0.717259**
(expected 0.71726).

| w4 | Macro-F1 | Delta vs baseline |
|---:|---:|---:|
| 0.25 | 0.717086 | -0.000173 |
| 0.50 | 0.717055 | -0.000204 |
| 0.75 | 0.717953 | +0.000694 |
| 1.00 | 0.718163 | +0.000903 |

Pass rule: any w4 delta >= +0.002. No tested w4 clears it, so the component is
not an LB gate candidate.

## Diversity

On the 9,969 league rows:

- char vs existing linear prediction disagreement: 0.281673
- char solo Macro-F1: 0.597528
- existing linear solo Macro-F1: 0.667650
- stacker solo Macro-F1: 0.705661
- encoder solo Macro-F1: 0.705089

Interpretation: diversity exists, but the component is too weak to add enough
corrective signal to the already strong 3-way blend.

## Artifacts

- `night_out/task1/oof_fold0.npz`
- `night_out/task1/oof_fold1.npz`
- `night_out/task1/oof_fold2.npz`
- `night_out/task1/char_oof.npz`
- `night_out/task1/char_oof_probs.npy`
- `night_out/task1/summary.json`
- `night_out/task1/char_svm_full.pkl`
- `night_out/task1/char_svm_full_meta.json`

The full-train artifact was produced for completeness, but because the verdict
is FAIL it should not be promoted into a submission without a separate LB gate
decision.

## Repro Commands

```powershell
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/components/char_svm/train_oof.py --fold 0
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/components/char_svm/train_oof.py --fold 1
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/components/char_svm/train_oof.py --fold 2
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/components/char_svm/train_oof.py --evaluate --train-full
```

## Notes

- A PowerShell watcher around the final fold matched its own command line and
  returned a nonzero wrapper status after outputs had already been written.
  The saved fold files were parsed by the final OOF assembly, and no
  `train_oof.py` Python process remained before reporting.
