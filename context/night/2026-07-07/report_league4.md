# League4 task1 report

## Summary

- Baseline B4 is the current submitted configuration: 4-way `[linear, stacker, e5=1.2, mBERT=0.8]` plus soft-AU `alpha=0.9`.
- Local B4 score: `0.738772` Macro-F1.
- Submission gate for a new candidate: B4 `+0.005` or higher, i.e. `>= 0.743772` in this league.
- Result: no candidate passes the gate. Keep `[1.2,0.8]` and `alpha=0.9`.

## Inputs and checks

- Holdout rows: `9,969`; AU rows: `682`; non-AU rows: `9,287`.
- OOF join sanity:
  - 3-way `(lin + stk + 2*e5) / 4`: `0.717259`.
  - 4-way `(lin + stk + 1.2*e5 + 0.8*mBERT) / 4`: `0.722546`.
- AU model: `char_wb(3,5)` TF-IDF, `max_features=120000`, `LinearSVC(C=1.0, class_weight=balanced)`.
- AU training protocol: nonholdout `sess_au` rows only (`4,343` rows, `946` sessions); holdout AU rows are excluded.

## Rebuild

| Variant | All | AU | non-AU |
|---|---:|---:|---:|
| 3-way e5x2 | 0.717259 | 0.513806 | 0.729707 |
| 4-way `[1.2,0.8]` raw | 0.722546 | 0.511765 | 0.735676 |
| 4-way `[1.2,0.8]` + soft-AU `alpha=0.9` | 0.738772 | 0.770168 | 0.735676 |

B4 is `0.738772`. The 4-way raw blend improves the non-AU slice, while AU is still supplied by the isolated AU route.

## Block ratio grid

Total encoder-block weight is fixed at `2.0`; only the internal e5/mBERT split changes. Soft-AU `alpha=0.9` is applied after the raw blend.

| e5 | mBERT | Raw all | Raw non-AU | Final | Delta vs B4 | Half1 | Half2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.40 | 0.60 | 0.720788 | 0.733778 | 0.737138 | -0.001634 | 0.729879 | 0.744083 |
| 1.35 | 0.65 | 0.721030 | 0.733988 | 0.737257 | -0.001516 | 0.729789 | 0.744428 |
| 1.30 | 0.70 | 0.721453 | 0.734468 | 0.737698 | -0.001075 | 0.729727 | 0.745386 |
| 1.25 | 0.75 | 0.721692 | 0.734767 | 0.738063 | -0.000710 | 0.730008 | 0.745905 |
| 1.20 | 0.80 | 0.722546 | 0.735676 | 0.738772 | +0.000000 | 0.730812 | 0.746473 |
| 1.15 | 0.85 | 0.722631 | 0.735968 | 0.739046 | +0.000273 | 0.730696 | 0.747123 |
| 1.10 | 0.90 | 0.722089 | 0.735431 | 0.738558 | -0.000214 | 0.730267 | 0.746559 |
| 1.05 | 0.95 | 0.721778 | 0.735034 | 0.738127 | -0.000646 | 0.729941 | 0.746068 |
| 1.00 | 1.00 | 0.721926 | 0.735110 | 0.738190 | -0.000583 | 0.729607 | 0.746619 |

Best block point is e5 `1.15`, mBERT `0.85`, final `0.739046`, delta `+0.000273`. This is report-only noise, not a submission candidate.

## AU alpha grid

Fixed raw blend is `[1.2,0.8]`.

| Alpha | All | AU | non-AU | Delta vs B4 | Changed AU argmax |
|---:|---:|---:|---:|---:|---:|
| 0.70 | 0.731502 | 0.638827 | 0.735676 | -0.007271 | 132 |
| 0.75 | 0.733715 | 0.669501 | 0.735676 | -0.005057 | 158 |
| 0.80 | 0.734957 | 0.691783 | 0.735676 | -0.003815 | 187 |
| 0.85 | 0.736626 | 0.723501 | 0.735676 | -0.002146 | 217 |
| 0.90 | 0.738772 | 0.770168 | 0.735676 | +0.000000 | 266 |
| 0.95 | 0.738384 | 0.773608 | 0.735676 | -0.000389 | 327 |
| 1.00 | 0.735747 | 0.744158 | 0.735676 | -0.003025 | 384 |

Best alpha remains `0.90`. `0.95` improves AU-only F1 but loses global Macro-F1 because the argmax changes hurt the full-class balance.

## Decision

No new submission candidate. The current 4-way recipe `[1.2,0.8]` with soft-AU `alpha=0.9` remains the best validated configuration in this scan.

Artifacts:

- `scripts/league4/rebuild.py`
- `scripts/league4/grid_block.py`
- `scripts/league4/grid_alpha.py`
- `night_out/league4/rebuild.{json,csv}`
- `night_out/league4/grid_block.{json,csv}`
- `night_out/league4/grid_alpha.{json,csv}`
