# -*- coding: utf-8 -*-
"""Reconstruct or audit the champion linear OOF.

The original OOF metadata names ``submit/features.py`` + ``E_+seq`` +
``LinearSVC(C=0.1)`` and per-fold class-bias tuning. The hidden trainer that
created the 2026-07-04 artifact is not present in this worktree, so this script
has two jobs:

* audit the reference OOF and write a machine-readable baseline summary;
* optionally run the documented public recipe on the saved folds for a drift
  check. If the drift is too large, downstream sweeps keep using the reference
  OOF as the baseline, per task instructions.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.linear2 import common as C
else:
    from . import common as C


def import_submit_features():
    return C.import_module_from_path("linear2_submit_features", C.SUBMIT_DIR / "features.py")


def decision_probs(pipe: Any, frame: Any, actions: Sequence[str], class_bias: dict[str, float] | None = None) -> np.ndarray:
    classes = [str(c) for c in pipe.named_steps["clf"].classes_]
    scores = pipe.decision_function(frame)
    if class_bias:
        bias = np.asarray([float(class_bias.get(str(label), 0.0)) for label in classes], dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64) + bias.reshape(1, -1)
    return C.align_probs(C.softmax(scores), classes, actions)


def encode_y(y_true: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    index = {str(label): i for i, label in enumerate(classes)}
    return np.asarray([index[str(label)] for label in y_true], dtype=np.int32)


def macro_from_pred_int(y_int: np.ndarray, pred_int: np.ndarray, n_classes: int) -> float:
    cm = np.bincount(
        y_int * n_classes + pred_int,
        minlength=n_classes * n_classes,
    ).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros_like(tp), where=denom > 0)
    return float(f1.mean())


def macro_from_scores(scores: np.ndarray, bias: np.ndarray, y_int: np.ndarray, n_classes: int) -> float:
    pred_int = np.argmax(np.asarray(scores) + bias.reshape(1, -1), axis=1)
    return macro_from_pred_int(y_int, pred_int, n_classes)


def tune_class_bias(scores: np.ndarray, y_true: np.ndarray, classes: Sequence[str]) -> tuple[dict[str, float], dict[str, float]]:
    """Mirror ``src/oof_lab_2026_07_03.py::coordinate_tune`` exactly."""
    labels = [str(c) for c in classes]
    y_int = encode_y(y_true, labels)
    bias = np.zeros(len(labels), dtype=float)
    base = macro_from_scores(scores, bias, y_int, len(labels))
    best = base
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
    return {label: float(value) for label, value in zip(labels, bias)}, {"base_macro_f1": base, "tuned_macro_f1": best}


def fit_fold(
    *,
    fold: dict[str, Any],
    df: Any,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    F = import_submit_features()
    fold_no = int(fold["fold"])
    out_dir = args.out_dir / "baseline_repro"
    out_dir.mkdir(parents=True, exist_ok=True)
    fold_path = out_dir / f"fold{fold_no}_probs.npy"
    meta_path = out_dir / f"fold{fold_no}_meta.json"
    if fold_path.exists() and meta_path.exists() and not args.force:
        print(f"[fold {fold_no}] cache hit {fold_path}")
        return C.read_json(meta_path)

    tr_idx = fold["train_idx"]
    va_idx = fold["valid_idx"]
    t0 = time.time()
    bias_map: dict[str, float] | None = None
    bias_meta: dict[str, Any] = {"enabled": bool(args.tune_bias)}
    if args.tune_bias:
        sub_rel, tune_rel = next(
            GroupShuffleSplit(n_splits=1, test_size=args.bias_test_size, random_state=args.bias_seed).split(
                np.zeros(len(tr_idx)), y[tr_idx], groups[tr_idx]
            )
        )
        sub_idx = tr_idx[sub_rel]
        tune_idx = tr_idx[tune_rel]
        pipe_bias = F.build_pipeline(F.FEATURE_SETS["E_+seq"], clf="svc", C=0.1, max_iter=args.max_iter)
        print(f"[fold {fold_no}] fit bias split sub={len(sub_idx)} tune={len(tune_idx)}")
        pipe_bias.fit(df.iloc[sub_idx], y[sub_idx])
        classes = [str(c) for c in pipe_bias.named_steps["clf"].classes_]
        tune_scores = pipe_bias.decision_function(df.iloc[tune_idx])
        bias_map, tune_scores_meta = tune_class_bias(tune_scores, y[tune_idx], classes)
        bias_meta.update(
            {
                "random_state": args.bias_seed,
                "test_size": args.bias_test_size,
                "subtrain_rows": int(len(sub_idx)),
                "tune_rows": int(len(tune_idx)),
                "class_bias": bias_map,
                **tune_scores_meta,
            }
        )

    print(f"[fold {fold_no}] refit outer train={len(tr_idx)} valid={len(va_idx)}")
    pipe = F.build_pipeline(F.FEATURE_SETS["E_+seq"], clf="svc", C=0.1, max_iter=args.max_iter)
    pipe.fit(df.iloc[tr_idx], y[tr_idx])
    probs = decision_probs(pipe, df.iloc[va_idx], C.ACTIONS, bias_map)
    macro = C.macro_f1_probs(probs, y[va_idx], C.ACTIONS)
    np.save(fold_path, probs.astype(np.float32))
    meta = {
        "fold": fold_no,
        "train_rows": int(len(tr_idx)),
        "valid_rows": int(len(va_idx)),
        "macro_f1": macro,
        "per_class_f1": C.per_class_f1(probs, y[va_idx], C.ACTIONS),
        "n_features": int(pipe.named_steps["feat"].transform(df.iloc[va_idx[:1]]).shape[1]),
        "classes_seen": [str(c) for c in pipe.named_steps["clf"].classes_],
        "bias": bias_meta,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    C.write_json(meta_path, meta)
    print(f"[fold {fold_no}] macro_f1={macro:.6f} elapsed={meta['elapsed_sec']:.1f}s")
    return meta


def assemble_repro(
    *,
    folds: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = args.out_dir / "baseline_repro"
    probs = np.zeros((len(ids), len(C.ACTIONS)), dtype=np.float32)
    seen = np.zeros(len(ids), dtype=bool)
    fold_rows = []
    for fold in folds:
        fold_no = int(fold["fold"])
        path = out_dir / f"fold{fold_no}_probs.npy"
        if not path.exists():
            raise FileNotFoundError(path)
        valid_idx = fold["valid_idx"]
        probs[valid_idx] = np.load(path).astype(np.float32)
        seen[valid_idx] = True
        fold_rows.append(C.read_json(out_dir / f"fold{fold_no}_meta.json"))
    if not seen.all():
        raise AssertionError(f"missing OOF rows: {int((~seen).sum())}")

    ref_probs, _, row_ids, ref_y = C.load_reference_oof(args.oof_dir)
    if not np.array_equal(ref_y, y):
        raise AssertionError("reference y_true does not match train_labels order")
    macro = C.macro_f1_probs(probs, y, C.ACTIONS)
    ref_macro = C.macro_f1_probs(ref_probs, y, C.ACTIONS)
    delta = macro - ref_macro
    diff = {
        "max_abs_prob_diff": float(np.max(np.abs(probs.astype(np.float64) - ref_probs))),
        "mean_abs_prob_diff": float(np.mean(np.abs(probs.astype(np.float64) - ref_probs))),
        "argmax_disagreement": float(np.mean(C.labels_from_probs(probs) != C.labels_from_probs(ref_probs))),
    }
    league = C.evaluate_lin_replacement(lin_probs_all=probs, row_ids=row_ids)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "close" if abs(delta) < args.close_tol else "not_close_reference_oof_retained",
        "close_tol": args.close_tol,
        "recipe": {
            "features": str(C.SUBMIT_DIR / "features.py"),
            "feature_set": "E_+seq",
            "clf": "LinearSVC(C=0.1, class_weight=balanced)",
            "bias_tuning": bool(args.tune_bias),
            "source": r"C:\dev\Second-Brain-Project\Hoseo\ai-2026\src\oof_lab_2026_07_03.py",
            "note": "trainer logic mirrored from the original OOF generator; current worktree submit/features.py is used",
        },
        "n_rows": int(len(ids)),
        "oof_macro_f1": macro,
        "reference_oof_macro_f1": ref_macro,
        "delta_vs_reference_oof_macro_f1": delta,
        "diff_vs_reference": diff,
        "folds": fold_rows,
        "league": league,
    }
    np.save(out_dir / "repro_probs.npy", probs.astype(np.float32))
    C.write_json(out_dir / "summary.json", summary)
    return summary


def audit_existing(args: argparse.Namespace) -> dict[str, Any]:
    ref_probs, classes, row_ids, y_true = C.load_reference_oof(args.oof_dir)
    folds = C.load_saved_folds(args.oof_dir)
    fold_summaries = []
    for fold in folds:
        valid_idx = fold["valid_idx"]
        fold_summaries.append(
            {
                "fold": int(fold["fold"]),
                "train_rows": int(len(fold["train_idx"])),
                "valid_rows": int(len(valid_idx)),
                "reference_macro_f1": C.macro_f1_probs(ref_probs[valid_idx], y_true[valid_idx], C.ACTIONS),
                "artifact_meta": C.read_json(args.oof_dir / f"linear_fold{int(fold['fold'])}_meta.json"),
            }
        )
    league = C.evaluate_lin_replacement(lin_probs_all=ref_probs, row_ids=row_ids)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "status": "reference_oof_audited",
        "reason_repro_not_claimed": "audit-only mode does not refit; run --all-folds --evaluate --tune-bias for reconstruction",
        "classes": classes,
        "n_rows": int(len(row_ids)),
        "reference_oof_macro_f1": C.macro_f1_probs(ref_probs, y_true, C.ACTIONS),
        "reference_per_class_f1": C.per_class_f1(ref_probs, y_true, C.ACTIONS),
        "folds": fold_summaries,
        "league": league,
        "source_meta": C.read_json(args.oof_dir / "meta.json"),
    }
    out_dir = args.out_dir / "baseline_repro"
    C.write_json(out_dir / "summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=C.OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--tune-bias", action="store_true")
    parser.add_argument("--bias-test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bias-seed", type=int, default=43)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--close-tol", type=float, default=0.002)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not any([args.audit_only, args.fold is not None, args.all_folds, args.evaluate]):
        parser.error("choose --audit-only, --fold K, --all-folds, or --evaluate")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        summary = audit_existing(args)
        print(
            "[audit] reference_oof={:.6f} softAU={:.6f}".format(
                summary["reference_oof_macro_f1"],
                summary["league"]["league_macro_f1"],
            )
        )
        return

    print("[load] train")
    samples, ids, y, groups = C.load_train(args.data_dir)
    folds = C.load_saved_folds(args.oof_dir, groups=groups)
    F = import_submit_features()
    print("[features] build dataframe once")
    df = F.build_dataframe(samples)

    selected = []
    if args.fold is not None:
        selected = [fold for fold in folds if int(fold["fold"]) == int(args.fold)]
        if not selected:
            raise ValueError(f"unknown fold {args.fold}")
    elif args.all_folds:
        selected = folds
    for fold in selected:
        fit_fold(fold=fold, df=df, y=y, groups=groups, args=args)

    if args.evaluate:
        summary = assemble_repro(folds=folds, ids=ids, y=y, args=args)
        print(
            "[summary] oof={:.6f} ref={:.6f} delta={:+.6f} status={}".format(
                summary["oof_macro_f1"],
                summary["reference_oof_macro_f1"],
                summary["delta_vs_reference_oof_macro_f1"],
                summary["status"],
            )
        )


if __name__ == "__main__":
    main()
