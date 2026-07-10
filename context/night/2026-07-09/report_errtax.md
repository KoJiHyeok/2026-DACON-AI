# task2 errtax report

## Summary

- Holdout rows: `9969` (`sim=9287`, `au=682`).
- Final mirror: 4-way `[linear, stacker, e5=1.2, mBERT=0.8]` + soft-AU `alpha=0.9`.
- Macro-F1: `0.738772`; expected `0.73877` tolerance `5e-4` passed.
- AU cache hit: `True` from `C:\dev\night\2026-07-09\task2\night_out\league4`.

## Weakest classes

| class | support | precision | recall | F1 | macro gap if perfect |
| --- | --- | --- | --- | --- | --- |
| list_directory | 651 | 0.428426 | 0.648233 | 0.515892 | 0.034579 |
| read_file | 1284 | 0.544180 | 0.604361 | 0.572694 | 0.030522 |
| lint_or_typecheck | 320 | 0.606811 | 0.612500 | 0.609642 | 0.027883 |
| grep_search | 1473 | 0.711879 | 0.545146 | 0.617455 | 0.027325 |
| ask_user | 393 | 0.663793 | 0.587786 | 0.623482 | 0.026894 |
| web_search | 174 | 0.587379 | 0.695402 | 0.636842 | 0.025940 |
| glob_pattern | 783 | 0.738170 | 0.597701 | 0.660550 | 0.024246 |
| plan_task | 370 | 0.685185 | 0.700000 | 0.692513 | 0.021963 |
| run_tests | 653 | 0.775449 | 0.793262 | 0.784254 | 0.015410 |
| run_bash | 689 | 0.825702 | 0.811321 | 0.818448 | 0.012968 |
| apply_patch | 666 | 0.851444 | 0.929429 | 0.888729 | 0.007948 |
| edit_file | 1580 | 0.959974 | 0.925949 | 0.942655 | 0.004096 |
| write_file | 199 | 0.975369 | 0.994975 | 0.985075 | 0.001066 |
| respond_only | 734 | 0.989218 | 1.000000 | 0.994580 | 0.000387 |

## Top confusions

| true | pred | count | share errors | share true | explore |
| --- | --- | --- | --- | --- | --- |
| grep_search | read_file | 378 | 0.145273 | 0.256619 | True |
| read_file | list_directory | 225 | 0.086472 | 0.175234 | True |
| grep_search | list_directory | 205 | 0.078786 | 0.139172 | True |
| read_file | grep_search | 199 | 0.076480 | 0.154984 | True |
| list_directory | read_file | 142 | 0.054573 | 0.218126 | True |
| glob_pattern | list_directory | 129 | 0.049577 | 0.164751 | True |
| glob_pattern | read_file | 116 | 0.044581 | 0.148148 | True |
| edit_file | apply_patch | 103 | 0.039585 | 0.065190 | False |
| ask_user | plan_task | 101 | 0.038816 | 0.256997 | False |
| grep_search | glob_pattern | 78 | 0.029977 | 0.052953 | True |

## Explore-cluster confusions

| true | pred | count | share errors | share true |
| --- | --- | --- | --- | --- |
| grep_search | read_file | 378 | 0.145273 | 0.256619 |
| read_file | list_directory | 225 | 0.086472 | 0.175234 |
| grep_search | list_directory | 205 | 0.078786 | 0.139172 |
| read_file | grep_search | 199 | 0.076480 | 0.154984 |
| list_directory | read_file | 142 | 0.054573 | 0.218126 |
| glob_pattern | list_directory | 129 | 0.049577 | 0.164751 |
| glob_pattern | read_file | 116 | 0.044581 | 0.148148 |
| grep_search | glob_pattern | 78 | 0.029977 | 0.052953 |
| glob_pattern | grep_search | 66 | 0.025365 | 0.084291 |
| read_file | glob_pattern | 56 | 0.021522 | 0.043614 |

## SIM vs AU

| slice | rows | macro-F1 |
| --- | --- | --- |
| all | 9969 | 0.738772 |
| sim | 9287 | 0.735676 |
| au | 682 | 0.770168 |

Weakest class per slice:

### sim

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| list_directory | 640 | 0.424897 | 0.645312 | 0.512407 |
| read_file | 1104 | 0.505130 | 0.579710 | 0.539857 |
| lint_or_typecheck | 284 | 0.599278 | 0.584507 | 0.591800 |
| grep_search | 1398 | 0.711962 | 0.532189 | 0.609087 |
| ask_user | 379 | 0.659763 | 0.588391 | 0.622036 |

### au

| class | support | precision | recall | F1 |
| --- | --- | --- | --- | --- |
| apply_patch | 28 | 0.470588 | 0.571429 | 0.516129 |
| ask_user | 14 | 0.800000 | 0.571429 | 0.666667 |
| run_bash | 72 | 0.739130 | 0.708333 | 0.723404 |
| lint_or_typecheck | 36 | 0.652174 | 0.833333 | 0.731707 |
| run_tests | 45 | 0.733333 | 0.733333 | 0.733333 |

## Macro-F1 gap contribution

The last column is `(1 - class_F1) / 14`: the maximum macro-F1 lift if that class became perfect while all other classes stayed fixed.

| class | F1 | max macro lift |
| --- | --- | --- |
| list_directory | 0.515892 | 0.034579 |
| read_file | 0.572694 | 0.030522 |
| lint_or_typecheck | 0.609642 | 0.027883 |
| grep_search | 0.617455 | 0.027325 |
| ask_user | 0.623482 | 0.026894 |
| web_search | 0.636842 | 0.025940 |
| glob_pattern | 0.660550 | 0.024246 |
| plan_task | 0.692513 | 0.021963 |
| run_tests | 0.784254 | 0.015410 |
| run_bash | 0.818448 | 0.012968 |
| apply_patch | 0.888729 | 0.007948 |
| edit_file | 0.942655 | 0.004096 |
| write_file | 0.985075 | 0.001066 |
| respond_only | 0.994580 | 0.000387 |

## Deterministic-key purity scan

These are not candidate scores. They only show whether observed error sets concentrate under simple deterministic keys enough to justify a later specialist-routing probe.

| target | feature | value | bucket rows | target rows | target rate | coverage |
| --- | --- | --- | --- | --- | --- | --- |
| explore_errors | last_action | list_directory | 630 | 251 | 0.398413 | 0.145761 |
| explore_errors | history_len | 0 | 1293 | 372 | 0.287703 | 0.216028 |
| explore_errors | last_action | none | 1293 | 372 | 0.287703 | 0.216028 |
| explore_errors | open_files | 0 | 3468 | 992 | 0.286044 | 0.576074 |
| explore_errors | prompt_mentions_file_search | True | 1560 | 425 | 0.272436 | 0.246806 |
| explore_errors | last_action | glob_pattern | 722 | 195 | 0.270083 | 0.113240 |
| explore_errors | last_action | plan_task | 344 | 91 | 0.264535 | 0.052846 |
| explore_errors | turn_index | 1-3 | 3764 | 942 | 0.250266 | 0.547038 |
| explore_errors | prompt_has_question | True | 1582 | 388 | 0.245259 | 0.225319 |
| explore_errors | elapsed_sec | <=120 | 367 | 88 | 0.239782 | 0.051103 |

Interpretation:

- Error mass is real: `2,602` holdout rows are wrong, and true read/grep/list/glob mistakes account for `1,722` of them (`66.2%`). The top 10 explore-cluster confusion pairs alone cover `1,594` rows (`61.3%` of all errors).
- Single-key purity is not AU-grade yet. The best simple key for all explore errors is `last_action=list_directory`, with `251/630` target rows (`39.8%`, `14.6%` coverage). For the largest pair `grep_search -> read_file`, the same key gives `128/630` (`20.3%`, `33.9%` coverage).
- The first-step signal is visible (`history_len=0` covers `372` explore errors), but exp #25 already killed first-step routing. Treat it as a diagnostic stratum only, not a lever.

## Report-only next levers

1. `last_action=list_directory` explore subtype probe. This is the closest soft-AU-shaped lead: it concentrates explore errors at `39.8%` and captures one third of the largest `grep_search -> read_file` pair. Next step should be purity-only first: inspect whether a non-derived key inside this bucket separates "show file contents" from "search/list around file".
2. Read/grep/list/glob boundary audit from examples. The biggest pairs are all within the exploration cluster (`grep_search -> read_file` 378, `read_file -> list_directory` 225, `grep_search -> list_directory` 205, `read_file -> grep_search` 199). This should not revive the banned R4 hierarchy; only a narrow pair-specific route with a deterministic key and specialist margin should advance.
3. `lint_or_typecheck` vs execution-command residual audit. `lint_or_typecheck` is the third weakest class (`F1=0.6096`, max macro lift `+0.0279`) and mostly falls into `run_tests` (73) or `run_bash` (45). Current purity is weak (`prompt_mentions_test=True` only `35/1257`), so this is lower priority and should stop at report-only unless a cleaner key appears.

## Artifacts

- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_summary.json`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_per_class_f1.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_confusion.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_slice_macro_f1.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_slice_per_class_f1.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_macro_gap.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_key_purity.csv`
- `C:\dev\night\2026-07-09\task2\night_out\errtax\errtax_error_examples.md`
