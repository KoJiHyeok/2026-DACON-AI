# Template Override R2 Report

## Verdict

**R2 current_prompt template override is rejected.** Under the required nonholdout-only mining gate, there are no deployable templates:

| gate | templates | holdout changed | fixed | broken | macro-F1 delta | decision |
|---|---:|---:|---:|---:|---:|---|
| Strict: nonholdout_n >= 20, purity >= 0.995, non-respond_only, non-AU target | 0 | 0 | 0 | 0 | +0.000000 | reject |
| Diagnostic only: min_n >= 2, apply all changed templates | 65 | 66 | 12 | 44 | -0.003825 | reject |
| Diagnostic only: min_n >= 2, holdout-safe cherry-pick | 12 | 12 | 12 | 0 | +0.000943 | reject |

The deploy gate is empty, and even an invalid holdout-label cherry-pick stays below the +0.002 report threshold. This closes R1's "hold for a small high-frequency template probe" item as **final discard**.

## Method

Scripts added:

- `scripts/tmpl_override/mine.py`
- `scripts/tmpl_override/judge.py`
- `scripts/tmpl_override/common.py`

`mine.py` uses the R1 masking order from `scripts/analysis/template_forensics.py`: quoted spans, paths/filenames, numbers, and camelCase-ish identifiers are masked. It also collapses whitespace for stable CSV keys; this did not change the R1 reproduction counts. The R1 source did not lowercase; an optional `r1_lower` mode was also checked and did not create strict candidates.

Purity is computed only on nonholdout rows. The holdout split is:

| split | rows |
|---|---:|
| train total | 70,000 |
| holdout excluded from mining | 9,969 |
| nonholdout used for purity | 60,031 |

## R1 Reproduction

Full-train R1 check matched the report exactly:

| metric | value |
|---|---:|
| unique normalized templates | 62,421 |
| duplicate templates | 4,319 |
| rows in duplicate templates | 11,898 |
| purity >= 0.99 templates | 1,727 |
| purity >= 0.99 rows | 5,181 |
| non-respond_only rows in those templates | 2,606 |

Nonholdout-only mining reduces the high-purity non-respond_only duplicate coverage to 2,162 rows. The largest high-purity non-respond_only template has only 11 nonholdout rows (`다시 빌드` -> `run_bash`). The only non-respond_only templates with nonholdout_n >= 20 are ask_user-ish prompts with purity 0.773 and 0.667, so they fail the purity gate.

## Judge

Baseline came from `scripts/league4/common.py`:

| model | macro-F1 | AU macro-F1 | non-AU macro-F1 |
|---|---:|---:|---:|
| 4-way + soft-AU | 0.738772 | 0.770168 | 0.735676 |

Holdout target rows are split into 682 AU and 9,287 non-AU. `judge.py` excludes AU rows from overrides to avoid double intervention with the AU route.

Strict gate result:

- mine candidates: 0
- templates that would change a non-AU holdout prediction: 0
- fixed / broken / wrong_to_wrong: 0 / 0 / 0
- delta vs B4+soft-AU: +0.000000

Relaxed diagnostic (`min_n=2`) shows why the R1 small templates should not be promoted:

- 996 mined templates, but only 65 would change any non-AU holdout prediction.
- Applying all changed templates breaks 44 already-correct rows while fixing 12: net -32 rows, macro-F1 -0.003825.
- Holdout-safe cherry-picking can make fixed 12 / broken 0, but that uses holdout labels and still gives only +0.000943.

## Script.py Integration

No `submit/script.py` patch is recommended.

If a future run ever produces real candidates, the deploy shape should be a tiny immutable `{normalized_prompt: action}` dictionary plus the same R1 normalizer in inference. Apply it only after the final blend argmax, only for non-AU routed rows, and only when the override action differs from the current prediction. This R2 run produced no template that justifies adding that branch.

## Artifacts

Generated locally, not required for submission:

- `night_out/tmpl_override/mine_summary.json`
- `night_out/tmpl_override/judge_summary.json`
- `night_out/tmpl_override/template_stats.csv`
- `night_out/tmpl_override_relaxed/judge_summary.json`
