# task5 report - first-step(hist=0) routing probe

## Verdict

**FAIL.** The first-step subgroup is weak for the current league components, so the probe was warranted, but the best whole-league routing delta is only `+0.001137`, below the task5 PASS threshold `+0.005`.

Hard replacement is actively unsafe: alpha `1.0` (equivalent to using the hist_0 specialist prediction on all first-step rows) changes whole-league Macro-F1 from `0.717259` to `0.703534`, delta `-0.013726`. Soft alpha `0.7` is the best variant, but the margin is too small and should be treated as noise / encoder-share mirage risk, not a submission candidate.

## Reproduction

- Analysis: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\firststep\analyze.py`
- Probe: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\firststep\probe_firststep.py`
- Join assert: fixed 3-way league blend reproduced `0.7172592174830689`.
- Hist_0 predicate: `len(sample.get("history") or []) == 0`.
- Serializer: `submit/au_route.py::serialize`, loaded read-only.

Artifacts:

- `night_out/task5/firststep_analysis_summary.json`
- `night_out/task5/firststep_component_macro_f1.csv`
- `night_out/task5/firststep_component_per_class_f1.csv`
- `night_out/task5/firststep_label_distribution.csv`
- `night_out/task5/firststep_probe_results.json`
- `night_out/task5/firststep_probe_routing_eval.csv`
- `night_out/task5/firststep_probe_per_class_f1.csv`

## Data

| split | rows | sessions |
|---|---:|---:|
| train all | 70,000 | - |
| train hist_0 | 9,000 | 9,000 |
| nonholdout hist_0 train pool | 7,707 | 7,707 |
| league holdout all | 9,969 | - |
| league holdout hist_0 | 1,293 | - |

Holdout hist_0 share is `12.97%`.

Top train hist_0 labels:

| action | count | rate |
|---|---:|---:|
| list_directory | 1,818 | 20.20% |
| read_file | 1,488 | 16.53% |
| plan_task | 1,121 | 12.46% |
| grep_search | 1,046 | 11.62% |
| run_bash | 1,011 | 11.23% |
| write_file | 708 | 7.87% |
| ask_user | 629 | 6.99% |
| glob_pattern | 567 | 6.30% |

`apply_patch` is nearly absent in first-step rows: only `6 / 9,000` train hist_0 rows and `0 / 1,293` holdout hist_0 rows.

## Component Diagnosis

This is not the expected "encoder only is strong" shape. All components are poor on holdout hist_0, and linear is only slightly ahead.

| component | all Macro-F1 | hist_0 Macro-F1 | non_hist_0 Macro-F1 |
|---|---:|---:|---:|
| linear | 0.667650 | 0.413642 | 0.671261 |
| stacker | 0.705661 | 0.402247 | 0.705270 |
| encoder_proxy | 0.705089 | 0.399726 | 0.707238 |
| blend | 0.717259 | 0.402078 | 0.718872 |

Selected holdout hist_0 per-class behavior:

| action | n | linear | encoder_proxy | blend |
|---|---:|---:|---:|---:|
| list_directory | 266 | 0.501 | 0.547 | 0.545 |
| read_file | 209 | 0.304 | 0.019 | 0.019 |
| plan_task | 157 | 0.608 | 0.737 | 0.750 |
| grep_search | 148 | 0.044 | 0.000 | 0.000 |
| run_bash | 142 | 0.873 | 0.898 | 0.904 |
| ask_user | 100 | 0.356 | 0.316 | 0.313 |
| glob_pattern | 92 | 0.020 | 0.000 | 0.000 |
| write_file | 97 | 0.984 | 0.990 | 0.995 |
| lint_or_typecheck | 14 | 0.222 | 0.133 | 0.133 |
| respond_only | 12 | 1.000 | 1.000 | 1.000 |
| web_search | 7 | 0.000 | 0.000 | 0.000 |
| apply_patch | 0 | 0.000 | 0.000 | 0.000 |

The blend/encoder path is almost blind to first-step `read_file`, `grep_search`, and `glob_pattern`; however, those local fixes are not enough to improve the 14-class whole-league score safely.

## Specialist Probe

Setup:

- Train rows: nonholdout hist_0 only (`7,707` rows).
- Evaluation rows: league holdout hist_0 only (`1,293` rows).
- CV: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.
- Model: `FeatureUnion(word 1-2 80k + char_wb 3-5 120k) + LinearSVC(C=0.5, class_weight=balanced)`.
- Holdout specialist probability: mean of the three fold-model softmax probabilities. No fold trains on any league holdout row.
- No blend-weight retuning outside hist_0; no class prior or bias correction.

CV:

| fold | valid rows | Macro-F1 |
|---|---:|---:|
| 1 | 2,569 | 0.428137 |
| 2 | 2,569 | 0.421132 |
| 3 | 2,569 | 0.428186 |
| OOF | 7,707 | 0.426023 |

Holdout hist_0 local Macro-F1:

| variant | Macro-F1 |
|---|---:|
| blend | 0.402078 |
| specialist hard | 0.422666 |
| soft alpha 0.5 | 0.410436 |
| soft alpha 0.7 | 0.423344 |
| soft alpha 1.0 | 0.422666 |

Whole-league routing:

| variant | all Macro-F1 | delta vs blend | hist_0 Macro-F1 |
|---|---:|---:|---:|
| fixed blend | 0.717259 | 0.000000 | 0.402078 |
| soft alpha 0.5 | 0.717606 | +0.000347 | 0.410436 |
| soft alpha 0.7 | 0.718396 | +0.001137 | 0.423344 |
| soft alpha 1.0 / hard | 0.703534 | -0.013726 | 0.422666 |

Best per-class hist_0 changes for alpha `0.7`:

| action | n | blend | alpha 0.7 | delta |
|---|---:|---:|---:|---:|
| read_file | 209 | 0.019 | 0.262 | +0.243 |
| list_directory | 266 | 0.545 | 0.568 | +0.023 |
| glob_pattern | 92 | 0.000 | 0.043 | +0.043 |
| ask_user | 100 | 0.313 | 0.328 | +0.016 |
| edit_file | 34 | 0.971 | 0.971 | +0.000 |
| write_file | 97 | 0.995 | 0.995 | +0.000 |
| grep_search | 148 | 0.000 | 0.000 | +0.000 |
| run_tests | 15 | 0.000 | 0.000 | +0.000 |
| plan_task | 157 | 0.750 | 0.736 | -0.014 |
| run_bash | 142 | 0.904 | 0.892 | -0.012 |

The specialist fixes some `read_file`/exploration blindness, but it is not a strong replacement model. Hard replacement changes many first-step class predictions enough to hurt whole-league class F1 even though local hist_0 macro rises slightly.

## Decision

FAIL. Do not promote a first-step route.

Reasons:

- Required threshold is `+0.005`; best observed whole-league delta is only `+0.001137`.
- Hard override is negative (`-0.013726`), so the route is not robust.
- The subgroup is label-skewed and has near-zero `apply_patch`; local hist_0 macro improvements do not translate cleanly to the full 14-class league metric.
- Given the known encoder-share mirage from prior experiments, a sub-`+0.005` soft gain is not credible enough for LB gating.

Suggested next work, if revisited: treat first-step as a feature/bias signal inside the existing blend or stacker rather than an override route. Do not replace encoder/blend predictions wholesale for hist_0 rows.

## Operational Note

The requested final git commit could not be created in this sandbox because Git needs to write `C:\dev\2026-AI-DACON\.git\worktrees\task5\index.lock`, and that external gitdir is read-only for this session. The worktree files and artifacts were written successfully.
