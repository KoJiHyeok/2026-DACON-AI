# -*- coding: utf-8 -*-
"""Generate 5-fold session-group OOF for the AAR stacker component.

Recipe (scripts/aar_rebuild/train_aar.py, validated 2026-07-12: 70k full
train 828s, OOF 0.7034 with the script's native 3-fold GroupKFold):
  - 3 text-view SGD classifiers (prompt_context_sgd, prompt_sgd, action_sgd)
    with the exact COMPONENT_SPECS hyperparameters from aar_config.json.
  - a rule-based transition_prior component (aar_transition_predict_proba).
  - final stacking view = hstack of the 4 component OOF probs (14*4=56 cols)
    fed to LogisticRegression(C=1.0, max_iter=500, solver=lbfgs,
    class_weight=balanced, random_state=seed).

train_aar.py itself is single-level: it builds one global GroupKFold(3) OOF
matrix for the 4 base components, then fits ONE stacker on that whole
matrix and reports `stacker.fit(X, y); stacker.predict(X)` as "OOF" (i.e.
out-of-fold for the base components, in-sample for the stacker). Reusing
that exact global-fit shape at the outer-fold level would let the stacker
see each row it is later asked to predict, so this script keeps the base
components' OOF discipline byte-for-byte (never fit on data used to predict
it) and extends the same discipline one level up: for outer fold k, the
stacker is fit on the base-component OOF rows from the OTHER 4 outer folds
only, then applied to fold k's base-component OOF rows. Base components for
fold k are trained once on all rows outside fold k -- there is no inner
nesting, so total component fits stay at 5 folds x 3 SGD fits = 15, the
same order of magnitude as the original 3-fold run (3x3=9), not the ~60
fits an inner-CV design would need.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submit"))
from scripts.oof5 import common as C
from scripts.aar_rebuild import train_aar as AARTRAIN
import aar_infer as AAR  # noqa: E402

OUT_DIR = C.ROOT / "artifacts" / "experiments" / "oof_aar"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=25)
    args = parser.parse_args()

    t_start = time.time()
    fold_map = C.load_fold_map()
    print("[load] train.jsonl + labels via AAR.load_training")
    records, y = AARTRAIN.load_training(C.DATA_DIR / "train.jsonl", C.DATA_DIR / "train_labels.csv")
    ids = np.asarray([str(r.get("id", "")) for r in records], dtype=object)
    outer_folds = C.fold_assignment_for_ids(ids, fold_map)
    n_outer = int(outer_folds.max()) + 1
    n = len(records)
    print(f"[fold] n_outer_folds={n_outer} rows={n}")

    print("[views] building text views once (full corpus)")
    views = AARTRAIN.build_views(records)

    # Pass 1: base-component OOF, exactly matching train_aar.py's per-fold
    # fit/predict (fold != k trains, fold == k predicted), just against the
    # 5-fold external fold_map instead of the script's own GroupKFold(3).
    base_oof = {name: np.zeros((n, len(AAR.ACTIONS)), dtype=np.float64) for name in AARTRAIN.COMPONENT_SPECS}
    base_oof_transition = np.zeros((n, len(AAR.ACTIONS)), dtype=np.float64)
    component_fold_reports = []

    for fold in range(n_outer):
        t0 = time.time()
        va_mask = outer_folds == fold
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]
        print(f"[components fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")

        for offset, name in enumerate(AARTRAIN.COMPONENT_SPECS):
            view = AARTRAIN.COMPONENT_SPECS[name]["view"]
            model = AARTRAIN._text_pipeline(
                name, args.seed + fold * 100 + offset,
                AARTRAIN.COMPONENT_SPECS[name]["alpha"], args.max_iter,
            )
            train_values = [views[view][i] for i in tr_idx]
            model.fit(train_values, y[tr_idx])
            valid_values = [views[view][i] for i in va_idx]
            base_oof[name][va_idx] = AARTRAIN._proba(model, valid_values)
            del model
            gc.collect()

        train_records = [records[i] for i in tr_idx]
        transition_spec = AARTRAIN.build_transition_spec(train_records, y[tr_idx])
        valid_records = [records[i] for i in va_idx]
        base_oof_transition[va_idx] = AAR.aar_transition_predict_proba(transition_spec, valid_records)

        elapsed = time.time() - t0
        component_fold_reports.append({"fold": fold, "train_rows": int(len(tr_idx)), "valid_rows": int(len(va_idx)), "elapsed_sec": round(elapsed, 3)})
        print(f"[components fold {fold}] elapsed={elapsed:.1f}s")

    base_stack_matrix = np.hstack(
        [base_oof[name] for name in AARTRAIN.COMPONENT_SPECS] + [base_oof_transition]
    ).astype(np.float32)
    del base_oof, base_oof_transition
    gc.collect()

    # Pass 2: for outer fold k, fit the stacker on the OTHER 4 folds' rows
    # of base_stack_matrix (never including fold k), then predict_proba
    # fold k's rows -- so the final AAR-component OOF row is produced by a
    # stacker that never trained on it, matching the task's OOF discipline
    # one level up from the base components.
    fold_reports = []
    all_ids_out = []
    all_probs_out = []
    all_y_out = []

    for fold in range(n_outer):
        t0 = time.time()
        va_mask = outer_folds == fold
        tr_idx = np.where(~va_mask)[0]
        va_idx = np.where(va_mask)[0]

        stacker = LogisticRegression(
            C=1.0, max_iter=500, solver="lbfgs", class_weight="balanced", random_state=args.seed,
        )
        stacker.fit(base_stack_matrix[tr_idx], y[tr_idx])
        probs_14 = stacker.predict_proba(base_stack_matrix[va_idx])
        probs = C.align_probs(np.asarray(probs_14, dtype=np.float64), [str(c) for c in stacker.classes_], C.ACTIONS)

        fold_ids = ids[va_idx]
        fold_y = y[va_idx]
        macro = C.macro_f1_probs(probs, fold_y, C.ACTIONS)
        elapsed = time.time() - t0

        C.save_fold_npz(
            OUT_DIR / f"oof_aar_fold{fold}.npz",
            ids=fold_ids, probs=probs, y_true=fold_y, fold=fold,
        )
        fold_reports.append({
            "fold": fold, "stacker_train_rows": int(len(tr_idx)), "valid_rows": int(len(va_idx)),
            "macro_f1": macro, "elapsed_sec": round(elapsed, 3),
        })
        all_ids_out.append(fold_ids)
        all_probs_out.append(probs)
        all_y_out.append(fold_y)
        print(f"[stacker fold {fold}] macro_f1={macro:.6f} elapsed={elapsed:.1f}s")

        del stacker, probs_14, probs
        gc.collect()

    pooled_ids = np.concatenate(all_ids_out)
    pooled_probs = np.vstack(all_probs_out)
    pooled_y = np.concatenate(all_y_out)
    C.verify_coverage(pooled_ids, fold_map)
    pooled_macro = C.macro_f1_probs(pooled_probs, pooled_y, C.ACTIONS)

    total_elapsed = time.time() - t_start
    run_meta = {
        "component": "aar_stacker",
        "recipe": {
            "trainer": "scripts/aar_rebuild/train_aar.py (validated 2026-07-12: 70k full train 828s, OOF 0.7034, matches submit/model/stacker/aar_config.json spec)",
            "components": list(AARTRAIN.COMPONENT_SPECS.keys()) + ["transition_prior"],
            "component_specs": AARTRAIN.COMPONENT_SPECS,
            "stacker": "LogisticRegression(C=1.0, max_iter=500, solver=lbfgs, class_weight=balanced, random_state={})".format(args.seed),
            "outer_fold_protocol": "5-fold from artifacts/experiments/oof_h12/fold_map.csv (replaces train_aar.py's native 3-fold GroupKFold)",
        },
        "deviation_from_train_aar": {
            "what": "train_aar.py fits ONE global LogisticRegression stacker on the whole base-component OOF "
                    "matrix (all folds pooled) and reports in-sample stacker.predict() on that same matrix as "
                    "its 'OOF' score. Reusing that shape per outer fold here would let the stacker train on rows "
                    "it is then asked to predict (fold k's rows would be in the pooled fit set that produces "
                    "fold k's own prediction), which breaks the parity OOF discipline this task requires.",
            "fix": "Two-pass design. Pass 1: base 4 components (3 SGD text views + transition_prior) produce OOF "
                   "exactly as train_aar.py does -- for fold k, train only on rows with fold_map fold != k, "
                   "predict fold == k. Pass 2: for each fold k, the LogisticRegression stacker is fit ONLY on the "
                   "base-component OOF rows belonging to the OTHER 4 folds, then predict_proba's fold k's rows. "
                   "This keeps every output row (both component-level and stacker-level) unseen by any model that "
                   "produced it, at the cost of refitting a cheap 56-column LogisticRegression 5 times instead of "
                   "once -- negligible compared to the 3 SGD text-view fits per fold.",
            "cost": "Base components: 5 folds x 3 SGD fits = 15 fits, same order of magnitude as train_aar.py's "
                    "native 3 folds x 3 fits = 9. No inner-CV nesting was needed.",
        },
        "seed": args.seed,
        "fold_map_sha256": C.FOLD_MAP_SHA256,
        "n_outer_folds": n_outer,
        "rows": int(n),
        "component_fold_reports": component_fold_reports,
        "stacker_fold_reports": fold_reports,
        "pooled_macro_f1": pooled_macro,
        "total_elapsed_sec": round(total_elapsed, 3),
    }
    C.write_json(OUT_DIR / "run_oof_aar.json", run_meta)
    C.write_sha256sums(OUT_DIR, [f"oof_aar_fold{f}.npz" for f in range(n_outer)] + ["run_oof_aar.json"])
    print(f"[done] pooled_macro_f1={pooled_macro:.6f} total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
