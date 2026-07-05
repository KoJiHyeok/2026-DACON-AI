# task5 progress

## 2026-07-06

- Read repository instructions from top-level `CLAUDE.md`; the AGENTS-requested nested `2026-AI-DACON\CLAUDE.md` path was not present in this worktree.
- Created isolated first-step scripts under `scripts/firststep/`; did not modify `scripts/au*`, `scripts/components`, or `submit/`.
- Ran analysis:
  - Fixed 3-way join reproduced: `0.7172592174830689`.
  - Holdout rows: `9,969`; holdout hist_0 rows: `1,293` (`12.97%`).
  - Train hist_0 rows: `9,000`; nonholdout hist_0 rows available for specialist training: `7,707`.
  - Holdout hist_0 Macro-F1 by component:
    - linear: `0.413642`
    - stacker: `0.402247`
    - encoder_proxy: `0.399726`
    - blend: `0.402078`
- Analysis does not show an encoder-only strong segment; all components are weak on hist_0, so the specialist probe is warranted.

## Next

- Completed `scripts/firststep/probe_firststep.py` with nonholdout hist_0 training only and soft alphas `{0.5,0.7,1.0}`.
- Probe:
  - 3-fold nonholdout hist_0 OOF Macro-F1: `0.426023`.
  - Holdout hist_0 blend Macro-F1: `0.402078`.
  - Holdout hist_0 specialist hard Macro-F1: `0.422666`.
  - Best soft holdout hist_0 Macro-F1: alpha `0.7`, `0.423344`.
  - Whole-league best delta: alpha `0.7`, `+0.001137`.
  - Hard alpha `1.0` whole-league delta: `-0.013726`.
- Verdict: FAIL. Best whole-league delta is below the required `+0.005`; hard replacement is harmful despite local hist_0 improvement.

## Remaining

- Write `context/night/2026-07-06/report_task5.md`.
- Write `context/night/2026-07-06/task5.DONE`.
- Attempt final commit; current sandbox blocks writing Git index under the external worktree gitdir.
