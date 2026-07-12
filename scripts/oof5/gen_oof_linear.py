# -*- coding: utf-8 -*-
"""Generate 5-fold session-group OOF for the linear component.

Recipe (validated in exp #32, scripts/linear2/baseline_repro.py):
  submit/features.py, feature set "E_+seq", LinearSVC(C=0.1, class_weight=
  balanced, max_iter=1000), decision_function -> softmax, plus per-fold
  class-bias coordinate tuning (mirrors src/oof_lab_2026_07_03.py::
  coordinate_tune, reimplemented in scripts/linear2/baseline_repro.py).
  That repro reproduced the reference linear OOF at 0.663895 vs reference
  0.663307 (delta +0.000588, within the 0.002 tolerance gate).

This script reuses that exact recipe but swaps the fold assignment for the
5-fold artifacts/experiments/oof_h12/fold_map.csv so all 5 champion
components share one fold split.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.oof5 import common as C

OUT_DIR = C.ROOT / "artifacts" / "experiments" / "oof_linear"
FEATURES_PATH = C.ROOT / "submit" / "features.py"


def import_features():
    spec = importlib.util.spec_from_file_location("oof5_submit_features", FEATURES_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["oof5_submit_features"] = module
    spec.loader.exec_module(module)
    return module


def encode_y(y_true: np.ndarray, classes: list[str]) -> np.ndarray:
    index = {str(label): i for i, label in enumerate(classes)}
    return np.asarray([index[str(label)] for label in y_true], dtype=np.int32)


def macro_from_pred_int(y_int: np.ndarray, pred_int: np.ndarray, n_classes: int) -> float:
    cm = np.bincount(y_int * n_classes + pred_int, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def macro_from_scores(scores: np.ndarray, bias: np.ndarray, y_int: np.ndarray, n_classes: int) -> float:
    pred_int = np.argmax(np.asarray(scores) + bias.reshape(1, -1), axis=1)
    return macro_from_pred_int(y_int, pred_int, n_classes)


def tune_class_bias(scores: np.ndarray, y_true: np.ndarray, classes: list[str]) -> dict[str, float]:
    """Mirror scripts/linear2/baseline_repro.py::tune_class_bias exactly."""
    labels = [str(c) for c in classes]
    y_int = encode_y(y_true, labels)
    bias = np.zeros(len(labels), dtype=float)
    best = macro_from_scores(scores, bias, y_int, len(labels))
    grids = [
        [-2.0, -1.0, -0.5, -0.25, 0.25, 0.5, 1.0, 2.0],
        [-0.6, -0.3, -0.15, 0.15, 0.3, 0.6],
        [-0.2, -0.1, -0.05, 0.05, 0.1, 0.2],
    ]
    for grid in grids:
        improved = True
        passes = 0
        while improved:
            passes += 1
            improved = False
            for col in range(len(labels)):
                base_value = bias[col]
                local_best = best
                local_value = base_value
                for delta in grid:
                    bias[col] = base_value + delta
                    score = macro_from_scores(scores, bias, y_int, len(labels))
                    if score > local_best:
                        local_best = score
                        local_value = bias[col]
                bias[col] = local_value
                if local_best > best + 1e-6:
                    best = local_best
                    improved = True
            if passes >= 6:
                break
    return {label: float(value) for label, value in zip(labels, bias)}


def decision_probs(pipe, frame, actions, class_bias):
    classes = [str(c) for c in pipe.named_steps["clf"].classes_]
    scores = pipe.decision_function(frame)
    if class_bias:
        bias = np.asarray([float(class_bias.get(str(label), 0.0)) for label in classes], dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64) + bias.reshape(1, -1)
    return C.align_probs(C.softmax(scores), classes, actions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--bias-test-size", type=float, default=0.2)
    parser.add_argument("--bias-seed", type=int, default=43)
    parser.add_argument("--no-bias-tuning", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    fold_map = C.load_fold_map()
    print("[load] train.jsonl + labels")
    samples, ids, y, groups = C.load_train()
    folds = C.fold_assignment_for_ids(ids, fold_map)
    n_folds = int(folds.max()) + 1
    print(f"[fold] n_folds={n_folds} rows={len(ids)}")

    F = import_features()
    print("[features] build dataframe once")
    df = F.build_dataframe(samples)

    fold_reports = []
    all_ids_out = []
    all_probs_out = []
    all_y_out = []

    for fold in range(n_folds):
        t0 = time.time()
        va_mask = folds == fold
        tr_mask = ~va_mask
        tr_idx = np.where(tr_mask)[0]
        va_idx = np.where(va_mask)[0]
        print(f"[fold {fold}] train={len(tr_idx)} valid={len(va_idx)}")

        bias_map = None
        if not args.no_bias_tuning:
            sub_rel, tune_rel = next(
                GroupShuffleSplit(n_splits=1, test_size=args.bias_test_size, random_state=args.bias_seed).split(
                    np.zeros(len(tr_idx)), y[tr_idx], groups[tr_idx]
                )
            )
            sub_idx = tr_idx[sub_rel]
            tune_idx = tr_idx[tune_rel]
            pipe_bias = F.build_pipeline(F.FEATURE_SETS["E_+seq"], clf="svc", C=0.1, max_iter=args.max_iter)
            pipe_bias.fit(df.iloc[sub_idx], y[sub_idx])
            classes = [str(c) for c in pipe_bias.named_steps["clf"].classes_]
            tune_scores = pipe_bias.decision_function(df.iloc[tune_idx])
            bias_map = tune_class_bias(tune_scores, y[tune_idx], classes)
            del pipe_bias, tune_scores
            gc.collect()

        pipe = F.build_pipeline(F.FEATURE_SETS["E_+seq"], clf="svc", C=0.1, max_iter=args.max_iter)
        pipe.fit(df.iloc[tr_idx], y[tr_idx])
        probs = decision_probs(pipe, df.iloc[va_idx], C.ACTIONS, bias_map)

        fold_ids = ids[va_idx]
        fold_y = y[va_idx]
        macro = C.macro_f1_probs(probs, fold_y, C.ACTIONS)
        elapsed = time.time() - t0

        C.save_fold_npz(
            OUT_DIR / f"oof_linear_fold{fold}.npz",
            ids=fold_ids, probs=probs, y_true=fold_y, fold=fold,
        )
        fold_reports.append({
            "fold": fold, "train_rows": int(len(tr_idx)), "valid_rows": int(len(va_idx)),
            "macro_f1": macro, "elapsed_sec": round(elapsed, 3),
            "bias_tuning": bool(not args.no_bias_tuning),
        })
        all_ids_out.append(fold_ids)
        all_probs_out.append(probs)
        all_y_out.append(fold_y)
        print(f"[fold {fold}] macro_f1={macro:.6f} elapsed={elapsed:.1f}s")

        del pipe, probs
        gc.collect()

    pooled_ids = np.concatenate(all_ids_out)
    pooled_probs = np.vstack(all_probs_out)
    pooled_y = np.concatenate(all_y_out)
    C.verify_coverage(pooled_ids, fold_map)
    pooled_macro = C.macro_f1_probs(pooled_probs, pooled_y, C.ACTIONS)

    total_elapsed = time.time() - t_start
    run_meta = {
        "component": "linear",
        "recipe": {
            "features": str(FEATURES_PATH),
            "feature_set": "E_+seq",
            "clf": "LinearSVC(C=0.1, class_weight=balanced, max_iter={})".format(args.max_iter),
            "proba_conversion": "decision_function -> softmax",
            "bias_tuning": bool(not args.no_bias_tuning),
            "provenance": "scripts/linear2/baseline_repro.py (validated exp #32, OOF 0.663895 vs reference 0.663307)",
        },
        "seed_note": "LinearSVC has no random_state param in this build (dual=True deterministic solver); "
                     "bias-tuning GroupShuffleSplit uses random_state=43 matching baseline_repro.py default",
        "fold_map_sha256": C.FOLD_MAP_SHA256,
        "n_folds": n_folds,
        "rows": int(len(ids)),
        "fold_reports": fold_reports,
        "pooled_macro_f1": pooled_macro,
        "total_elapsed_sec": round(total_elapsed, 3),
    }
    C.write_json(OUT_DIR / "run_oof_linear.json", run_meta)
    C.write_sha256sums(OUT_DIR, [f"oof_linear_fold{f}.npz" for f in range(n_folds)] + ["run_oof_linear.json"])
    print(f"[done] pooled_macro_f1={pooled_macro:.6f} total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
