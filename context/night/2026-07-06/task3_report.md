# task3 report - AU/SIM split analysis and AU specialist probe

## Verdict

**PASS for LB gate candidate.** The fixed 3-way league baseline reproduced at Macro-F1 `0.7172592175`. Routing only `sess_au*` holdout rows to an honest AU-only linear OOF model changed league holdout Macro-F1 from `0.717259` to `0.726613`, delta `+0.009354`, above the `+0.002` PASS threshold.

This is not the same mechanism as the prior failed R4 specialist/meta-selector work: the AU specialist is stronger on AU (`0.690347` on the same holdout AU rows) than the fixed blend (`0.513806`). It is also not a calib_v1/R3 prior-shift result: no class prior or bias was fit; the probe used text/meta features and session-group OOF predictions.

LB gate is still mandatory. The candidate should be implemented as explicit id routing (`id.startswith("sess_au")`) with no blend weight tuning and no class-prior patching.

## Reproduction

- Analysis: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\au\analyze.py`
- Probe: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\au\probe_au_linear.py`
- Key artifacts:
  - `context/night/2026-07-06/au_analysis_summary.json`
  - `context/night/2026-07-06/au_component_per_class_f1.csv`
  - `context/night/2026-07-06/au_error_samples.md`
  - `context/night/2026-07-06/task3_probe_au_linear.json`

## Distribution

| split | SIM rows | AU rows | AU share | SIM sessions | AU sessions |
|---|---:|---:|---:|---:|---:|
| train | 64,975 | 5,025 | 7.18% | 8,330 | 1,099 |
| league holdout | 9,287 | 682 | 6.84% | 1,197 | 153 |

AU label distribution is not just a smaller copy of SIM. AU is much more `read_file` heavy and much less `glob_pattern`/`list_directory` heavy.

| class | AU count | AU % | SIM count | SIM % |
|---|---:|---:|---:|---:|
| read_file | 1,291 | 25.69% | 7,966 | 12.26% |
| edit_file | 882 | 17.55% | 10,289 | 15.84% |
| grep_search | 550 | 10.95% | 9,362 | 14.41% |
| run_bash | 496 | 9.87% | 4,572 | 7.04% |
| run_tests | 331 | 6.59% | 4,230 | 6.51% |
| respond_only | 346 | 6.89% | 4,832 | 7.44% |
| lint_or_typecheck | 281 | 5.59% | 2,002 | 3.08% |
| apply_patch | 191 | 3.80% | 4,632 | 7.13% |
| plan_task | 169 | 3.36% | 2,510 | 3.86% |
| web_search | 118 | 2.35% | 1,155 | 1.78% |
| list_directory | 109 | 2.17% | 4,220 | 6.49% |
| ask_user | 94 | 1.87% | 2,607 | 4.01% |
| glob_pattern | 89 | 1.77% | 5,195 | 8.00% |
| write_file | 78 | 1.55% | 1,403 | 2.16% |

Field differences:

| field | SIM | AU | read |
|---|---:|---:|---|
| mean history_len | 7.08 | 5.04 | AU is earlier/shorter context |
| mean turn_index | 5.35 | 3.72 | AU has more early-turn rows |
| mean elapsed_session_sec | 526.67 | 307.57 | AU sessions are shorter |
| mean current_prompt_words | 12.97 | 10.94 | AU prompts are shorter |
| prompt guessed Korean | 71.13% | 76.28% | both Korean-heavy; AU slightly more |
| `last_ci_status=passed` | 38.86% | 55.44% | AU has more clean/passed workspaces |
| `user_tier=enterprise` | 14.71% | 35.00% | AU meta mix differs |

## Current 3-Way Behavior

| component | all Macro-F1 | SIM Macro-F1 | AU Macro-F1 |
|---|---:|---:|---:|
| linear | 0.667650 | 0.672857 | 0.543719 |
| stacker | 0.705661 | 0.716994 | 0.491988 |
| encoder | 0.705089 | 0.716902 | 0.508700 |
| blend | 0.717259 | 0.729707 | 0.513806 |

The blend is SIM-driven. On AU, the base linear is actually better than stacker, encoder, and the 3-way blend. The AU problem is concentrated in exploration/file-navigation classes plus `apply_patch` and `run_bash`.

| class | AU n | AU linear | AU stacker | AU encoder | AU blend | SIM blend |
|---|---:|---:|---:|---:|---:|---:|
| glob_pattern | 8 | 0.182 | 0.095 | 0.080 | 0.083 | 0.650 |
| list_directory | 11 | 0.133 | 0.145 | 0.099 | 0.105 | 0.511 |
| apply_patch | 28 | 0.281 | 0.243 | 0.263 | 0.256 | 0.898 |
| ask_user | 14 | 0.370 | 0.308 | 0.333 | 0.417 | 0.628 |
| grep_search | 75 | 0.387 | 0.362 | 0.453 | 0.441 | 0.604 |
| run_bash | 72 | 0.620 | 0.435 | 0.500 | 0.455 | 0.823 |
| read_file | 180 | 0.623 | 0.613 | 0.452 | 0.490 | 0.536 |
| lint_or_typecheck | 36 | 0.585 | 0.511 | 0.526 | 0.526 | 0.570 |
| plan_task | 26 | 0.588 | 0.531 | 0.553 | 0.553 | 0.680 |
| run_tests | 45 | 0.580 | 0.536 | 0.604 | 0.571 | 0.770 |
| edit_file | 126 | 0.774 | 0.721 | 0.745 | 0.739 | 0.949 |

Full AU/SIM per-class F1 matrix:

| class | AU n | SIM n | linear AU/SIM | stacker AU/SIM | encoder AU/SIM | blend AU/SIM |
|---|---:|---:|---:|---:|---:|---:|
| apply_patch | 28 | 638 | 0.281 / 0.845 | 0.243 / 0.898 | 0.263 / 0.858 | 0.256 / 0.898 |
| ask_user | 14 | 379 | 0.370 / 0.562 | 0.308 / 0.591 | 0.333 / 0.611 | 0.417 / 0.628 |
| edit_file | 126 | 1,454 | 0.774 / 0.918 | 0.721 / 0.939 | 0.745 / 0.926 | 0.739 / 0.949 |
| glob_pattern | 8 | 775 | 0.182 / 0.591 | 0.095 / 0.626 | 0.080 / 0.639 | 0.083 / 0.650 |
| grep_search | 75 | 1,398 | 0.387 / 0.558 | 0.362 / 0.588 | 0.453 / 0.599 | 0.441 / 0.604 |
| lint_or_typecheck | 36 | 284 | 0.585 / 0.488 | 0.511 / 0.590 | 0.526 / 0.560 | 0.526 / 0.570 |
| list_directory | 11 | 640 | 0.133 / 0.431 | 0.145 / 0.506 | 0.099 / 0.510 | 0.105 / 0.511 |
| plan_task | 26 | 344 | 0.588 / 0.561 | 0.531 / 0.628 | 0.553 / 0.671 | 0.553 / 0.680 |
| read_file | 180 | 1,104 | 0.623 / 0.479 | 0.613 / 0.492 | 0.452 / 0.531 | 0.490 / 0.536 |
| respond_only | 42 | 692 | 0.988 / 0.999 | 0.976 / 1.000 | 1.000 / 0.999 | 1.000 / 0.999 |
| run_bash | 72 | 617 | 0.620 / 0.771 | 0.435 / 0.807 | 0.500 / 0.817 | 0.455 / 0.823 |
| run_tests | 45 | 608 | 0.580 / 0.710 | 0.536 / 0.762 | 0.604 / 0.763 | 0.571 / 0.770 |
| web_search | 13 | 161 | 0.500 / 0.523 | 0.556 / 0.626 | 0.514 / 0.557 | 0.556 / 0.606 |
| write_file | 6 | 193 | 1.000 / 0.984 | 0.857 / 0.985 | 1.000 / 0.992 | 1.000 / 0.992 |

## AU Error Review

Top AU blend confusion pairs:

| true | pred | count |
|---|---:|---:|
| read_file | list_directory | 75 |
| edit_file | apply_patch | 40 |
| grep_search | list_directory | 33 |
| read_file | grep_search | 30 |
| run_bash | lint_or_typecheck | 27 |

Qualitative read from `au_error_samples.md`:

- `read_file -> list_directory`: "package.json scripts..." asks to open the file first, but the wording sounds like inventory/listing.
- `edit_file -> apply_patch`: "selectTheme return type..." is a direct fix request, but the blend over-prefers patch-like edits.
- `grep_search -> list_directory`: asking whether an export route exists in `app.py` is a search intent, but gets treated as directory exploration.
- `read_file -> grep_search`: asking where `session` is defined after reading auth code is ambiguous between reading a known file and searching.
- `run_bash -> lint_or_typecheck`: "run iOS again" after Podfile editing is a shell command, not typecheck, but CI/test language pulls it toward validation classes.

The main AU pathology is over-predicting broad exploration (`list_directory`, `grep_search`) for Korean short prompts where the correct action is often `read_file`, and confusing edit mechanisms (`edit_file` vs `apply_patch`).

## AU Specialist Probe

Probe setup:

- Rows: 5,025 AU train rows, 1,099 AU sessions.
- CV: `StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)`.
- Model: `FeatureUnion(word 1-2 + char_wb 3-5) + LinearSVC(C=0.5, class_weight=balanced)`.
- Fold Macro-F1: `0.693827`, `0.673527`, `0.671855`.
- AU-only OOF Macro-F1 over all AU rows: `0.680006`.

Routing evaluation on the same 9,969-row league holdout:

| metric | value |
|---|---:|
| fixed 3-way all Macro-F1 | 0.717259 |
| fixed 3-way AU Macro-F1 | 0.513806 |
| AU-linear OOF on holdout AU | 0.690347 |
| hybrid all Macro-F1 | 0.726613 |
| hybrid delta | +0.009354 |

Per-class AU changes on holdout:

| class | n | blend | AU linear | delta |
|---|---:|---:|---:|---:|
| glob_pattern | 8 | 0.083 | 0.800 | +0.717 |
| list_directory | 11 | 0.105 | 0.727 | +0.622 |
| apply_patch | 28 | 0.256 | 0.604 | +0.347 |
| read_file | 180 | 0.490 | 0.754 | +0.264 |
| grep_search | 75 | 0.441 | 0.680 | +0.239 |
| run_bash | 72 | 0.455 | 0.570 | +0.115 |
| edit_file | 126 | 0.739 | 0.678 | -0.061 |
| respond_only | 42 | 1.000 | 0.721 | -0.279 |
| write_file | 6 | 1.000 | 0.909 | -0.091 |

The probe is not uniformly better, but it fixes the exact low-F1 AU classes by enough to dominate losses on `edit_file`, `respond_only`, and `write_file`.

## Decision

PASS. The local evidence supports an LB-gate candidate: train a dedicated AU linear model, route `sess_au*` rows to it, and leave SIM rows on the existing 3-way blend. This should be tested on LB before any merge into the submission lane.

Guardrails for the candidate:

- Do not tune blend weights or encoder share.
- Do not add class-prior or bias calibration.
- Keep GroupKFold/OOF validation by session prefix.
- Include a fallback path if test has no `sess_au*` rows or if AU model loading fails.
