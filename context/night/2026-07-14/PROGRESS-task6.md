# PROGRESS-task6

- task: CX-C au_linear upgrade candidate
- worktree: `C:\dev\night\2026-07-14\task6`
- branch: `night/2026-07-14/task6`
- owned paths: `scripts/cx_au2/**`, `tests/test_cx_au2.py`, this task's night report/progress/DONE
- forbidden: `submit/**`, canonical context ledgers, official submission, push
- model routing: Sol (`gpt-5.6-sol`, xhigh, read-only) attempted; blocked by sandbox network policy before response

## Status

- Read ownership and submission/context gates.
- Verified deployed artifact contract: `{"union": TfidfVectorizer, "clf": LinearSVC}`.
- Verified inference contract: `submit/au_route.py::predict_proba` calls `union.transform`, then `clf.decision_function`, then softmax.
- Verified all ticket inputs exist, including sklearn 1.8.0 environment and Qwen holdout NPZ.
- Implemented `scripts/cx_au2/{common,train_au2,eval_au2}.py` and `tests/test_cx_au2.py`.
- Static compile PASS.
- Focused pytest PASS: `3 passed` under Python 3.13 / sklearn 1.8.0.
- Real 5-variant x 5-fold training PASS: baseline char-C1 won pooled OOF (`0.680667`).
- No candidate `model.pkl` was written because no improvement beat baseline (conditional DoD).
- Frozen w3/alpha=.85 evaluation PASS: all five deltas are exactly zero; recommendation `do_not_swap`.
- Report and `task6.DONE` written.

## Design constraints

- Remove every frozen holdout id before AU-only CV and final refit.
- Select variants only by session-prefix GroupKFold OOF Macro-F1.
- Keep at most five predeclared variants and keep candidate pickle compatible with `au_route.predict_proba`.
- Evaluate the selected candidate once on the frozen holdout using `(lin + stk + 3*qwen) / 5` and soft-AU alpha `0.85`.

## Next resume point

Run final regression tests and diff audit, then commit the owned-path handoff if clean.
