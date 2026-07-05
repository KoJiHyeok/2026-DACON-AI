# -*- coding: utf-8 -*-
"""Probe an honest history==0 linear specialist against the fixed league blend."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

from common import (
    ACTION_CLASSES,
    DEFAULT_HOLDOUT_BASE,
    DEFAULT_OOF_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_TRAIN_LABELS,
    align_probs,
    is_hist0,
    load_league_components,
    load_submit_serializer,
    load_train_records,
    macro_f1_from_pred,
    per_class_rows,
    predict_labels,
    session_id,
    softmax,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--holdout-base", type=Path, default=DEFAULT_HOLDOUT_BASE)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=0.5)
    parser.add_argument("--alphas", default="0.5,0.7,1.0")
    return parser.parse_args()


def build_features(train_texts: list[str], valid_texts: list[str]):
    union = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=80_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=1,
                    max_features=120_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                ),
            ),
        ]
    )
    x_train = union.fit_transform(train_texts)
    x_valid = union.transform(valid_texts)
    if not sparse.issparse(x_train) or not sparse.issparse(x_valid):
        raise TypeError("TF-IDF feature union should return sparse matrices")
    return union, x_train, x_valid


def align_model_scores(clf: LinearSVC, x_matrix, actions: Sequence[str]) -> np.ndarray:
    probs = softmax(np.asarray(clf.decision_function(x_matrix)))
    return align_probs(probs, [str(c) for c in clf.classes_], actions)


def f1_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, actions: Sequence[str]) -> dict[str, Any]:
    return {
        "variant": name,
        "rows": int(len(y_true)),
        "macro_f1": macro_f1_from_pred(y_true, y_pred, actions),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    records = load_train_records(args.train_jsonl, args.labels_csv)
    league = load_league_components(args.holdout_base, args.oof_dir)
    holdout_ids = {str(sample_id) for sample_id in league["ids"]}
    by_id = {str(row["id"]): row for row in records}
    serialize = load_submit_serializer()

    ids = np.asarray([str(row["id"]) for row in records], dtype=object)
    y = np.asarray([str(row["action"]) for row in records], dtype=object)
    groups = np.asarray([session_id(sample_id) for sample_id in ids], dtype=object)
    hist0_mask_all = np.asarray([is_hist0(row) for row in records], dtype=bool)
    holdout_mask_all = np.asarray([sample_id in holdout_ids for sample_id in ids], dtype=bool)
    train_idx = np.where(hist0_mask_all & ~holdout_mask_all)[0]
    if len(train_idx) == 0:
        raise ValueError("no nonholdout first-step rows available for training")

    holdout_hist0 = np.asarray([is_hist0(by_id[str(sample_id)]) for sample_id in league["ids"]], dtype=bool)
    holdout_samples = [by_id[str(sample_id)] for sample_id in league["ids"][holdout_hist0]]
    if not holdout_samples:
        raise ValueError("no first-step rows in holdout")

    train_texts_all = [serialize(records[int(i)]) for i in train_idx]
    train_y = y[train_idx]
    train_groups = groups[train_idx]
    holdout_texts = [serialize(sample) for sample in holdout_samples]

    splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    train_oof = np.zeros((len(train_idx), len(ACTION_CLASSES)), dtype=np.float64)
    holdout_fold_probs: list[np.ndarray] = []
    fold_rows = []

    for fold, (tr_pos, va_pos) in enumerate(
        splitter.split(np.zeros(len(train_y)), train_y, groups=train_groups),
        start=1,
    ):
        overlap = set(train_groups[tr_pos]) & set(train_groups[va_pos])
        if overlap:
            raise AssertionError(f"fold {fold} group leakage: {len(overlap)} groups")
        x_train_texts = [train_texts_all[int(i)] for i in tr_pos]
        x_valid_texts = [train_texts_all[int(i)] for i in va_pos]
        union, x_train, x_valid = build_features(x_train_texts, x_valid_texts)
        clf = LinearSVC(C=args.c, class_weight="balanced", max_iter=5000, random_state=args.seed)
        clf.fit(x_train, train_y[tr_pos])

        valid_probs = align_model_scores(clf, x_valid, ACTION_CLASSES)
        train_oof[va_pos] = valid_probs
        valid_pred = predict_labels(valid_probs, ACTION_CLASSES)

        x_holdout = union.transform(holdout_texts)
        holdout_fold_probs.append(align_model_scores(clf, x_holdout, ACTION_CLASSES))
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(len(tr_pos)),
                "valid_rows": int(len(va_pos)),
                "valid_sessions": int(len(set(train_groups[va_pos]))),
                "macro_f1": macro_f1_from_pred(train_y[va_pos], valid_pred, ACTION_CLASSES),
                "classes_in_train": [str(c) for c in clf.classes_],
            }
        )
        print(
            f"[fold {fold}] train={len(tr_pos)} valid={len(va_pos)} "
            f"groups={len(set(train_groups[va_pos]))} macro_f1={fold_rows[-1]['macro_f1']:.6f}",
            flush=True,
        )

    oof_pred = predict_labels(train_oof, ACTION_CLASSES)
    holdout_specialist = np.mean(np.stack(holdout_fold_probs, axis=0), axis=0)
    holdout_specialist = align_probs(holdout_specialist, ACTION_CLASSES, league["actions"])

    actions = league["actions"]
    y_holdout = league["y_true"]
    blend_probs = league["components"]["blend"]
    blend_pred = predict_labels(blend_probs, actions)
    specialist_pred_hist0 = predict_labels(holdout_specialist, actions)
    y_hist0 = y_holdout[holdout_hist0]
    blend_pred_hist0 = blend_pred[holdout_hist0]

    variant_rows = [
        f1_row("blend_hist0", y_hist0, blend_pred_hist0, actions),
        f1_row("specialist_hist0", y_hist0, specialist_pred_hist0, actions),
    ]
    all_eval_rows = []
    per_class = []
    per_class.extend(per_class_rows(y_hist0, blend_pred_hist0, actions, {"variant": "blend_hist0"}))
    per_class.extend(per_class_rows(y_hist0, specialist_pred_hist0, actions, {"variant": "specialist_hist0"}))

    base_all = macro_f1_from_pred(y_holdout, blend_pred, actions)
    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    for alpha in alphas:
        soft_probs = blend_probs.copy()
        soft_probs[holdout_hist0] = alpha * holdout_specialist + (1.0 - alpha) * blend_probs[holdout_hist0]
        soft_pred = predict_labels(soft_probs, actions)
        soft_pred_hist0 = soft_pred[holdout_hist0]
        variant_name = f"soft_alpha_{alpha:g}"
        variant_rows.append(f1_row(f"{variant_name}_hist0", y_hist0, soft_pred_hist0, actions))
        all_score = macro_f1_from_pred(y_holdout, soft_pred, actions)
        all_eval_rows.append(
            {
                "variant": variant_name,
                "alpha": float(alpha),
                "blend_all_macro_f1": float(base_all),
                "hybrid_all_macro_f1": float(all_score),
                "hybrid_delta": float(all_score - base_all),
                "hist0_macro_f1": macro_f1_from_pred(y_hist0, soft_pred_hist0, actions),
            }
        )
        per_class.extend(per_class_rows(y_hist0, soft_pred_hist0, actions, {"variant": f"{variant_name}_hist0"}))

    pd.DataFrame(fold_rows).to_csv(args.out_dir / "firststep_probe_fold_metrics.csv", index=False)
    pd.DataFrame(variant_rows).to_csv(args.out_dir / "firststep_probe_hist0_macro_f1.csv", index=False)
    pd.DataFrame(all_eval_rows).to_csv(args.out_dir / "firststep_probe_routing_eval.csv", index=False)
    pd.DataFrame(per_class).to_csv(args.out_dir / "firststep_probe_per_class_f1.csv", index=False)

    best = max(all_eval_rows, key=lambda row: row["hybrid_delta"])
    return {
        "inputs": {
            "train_jsonl": str(args.train_jsonl),
            "labels_csv": str(args.labels_csv),
            "holdout_base": str(args.holdout_base),
            "oof_dir": str(args.oof_dir),
            "n_splits": int(args.n_splits),
            "seed": int(args.seed),
            "c": float(args.c),
            "alphas": alphas,
            "model": "submit.au_route.serialize + FeatureUnion(word 1-2 80k + char_wb 3-5 120k) + LinearSVC(C, balanced)",
            "holdout_prediction": "mean probability ensemble of the 3 fold models; all fold training excludes league holdout rows",
        },
        "join_assert_blend_macro_f1": float(league["blend_f1"]),
        "data": {
            "train_rows": int(len(records)),
            "train_hist0_rows": int(hist0_mask_all.sum()),
            "nonholdout_hist0_train_rows": int(len(train_idx)),
            "nonholdout_hist0_train_sessions": int(len(set(train_groups))),
            "holdout_rows": int(len(league["ids"])),
            "holdout_hist0_rows": int(holdout_hist0.sum()),
            "holdout_hist0_share": float(holdout_hist0.mean()),
        },
        "cv": {
            "folds": fold_rows,
            "nonholdout_hist0_oof_macro_f1": macro_f1_from_pred(train_y, oof_pred, ACTION_CLASSES),
            "nonholdout_hist0_oof_per_class": per_class_rows(train_y, oof_pred, ACTION_CLASSES),
        },
        "holdout_hist0_macro_f1": variant_rows,
        "routing_eval": all_eval_rows,
        "best_variant": best,
        "decision_rule": {
            "pass_threshold_hybrid_delta": 0.005,
            "passes": bool(best["hybrid_delta"] >= 0.005),
        },
        "output_files": [
            "firststep_probe_fold_metrics.csv",
            "firststep_probe_hist0_macro_f1.csv",
            "firststep_probe_routing_eval.csv",
            "firststep_probe_per_class_f1.csv",
            "firststep_probe_results.json",
        ],
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    result = run_probe(args)
    (args.out_dir / "firststep_probe_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    best = result["best_variant"]
    print(
        f"[probe] cv_oof={result['cv']['nonholdout_hist0_oof_macro_f1']:.6f} "
        f"best={best['variant']} delta={best['hybrid_delta']:+.6f} "
        f"pass={result['decision_rule']['passes']}"
    )
    print(f"[save] {args.out_dir / 'firststep_probe_results.json'}")


if __name__ == "__main__":
    main()
