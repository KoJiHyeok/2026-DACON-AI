# -*- coding: utf-8 -*-
"""Leak-free 5-variant GroupKFold sweep for the sess_au linear specialist."""
from __future__ import annotations

import argparse
import platform
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

if __package__:
    from . import common
else:
    import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.DATA_DIR)
    parser.add_argument("--holdout-npz", type=Path, default=common.HOLDOUT_NPZ)
    parser.add_argument("--candidate-dir", type=Path, default=common.CANDIDATE_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=common.ACTIONS, average="macro", zero_division=0))


def run_cv(
    samples: list[dict],
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
) -> list[dict]:
    if len(common.VARIANTS) > 5:
        raise AssertionError("CX-C multiple-comparison gate allows at most five variants")
    au_route = common.load_au_route()
    texts = np.asarray([au_route.serialize(sample) for sample in samples], dtype=object)
    splitter = GroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = {variant.name: np.empty(len(y), dtype=object) for variant in common.VARIANTS}
    covered = np.zeros(len(y), dtype=np.int8)
    fold_rows: dict[str, list[dict]] = defaultdict(list)

    by_kind: dict[str, list[common.Variant]] = defaultdict(list)
    for variant in common.VARIANTS:
        by_kind[variant.feature_kind].append(variant)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(texts, y, groups)):
        common.assert_group_disjoint(groups[train_idx], groups[valid_idx], f"CV fold {fold}")
        covered[valid_idx] += 1
        for feature_kind, variants in by_kind.items():
            started = time.time()
            vectorizer = common.make_vectorizer(feature_kind)
            x_train = vectorizer.fit_transform(texts[train_idx].tolist())
            x_valid = vectorizer.transform(texts[valid_idx].tolist())
            vectorize_sec = time.time() - started
            for variant in variants:
                fit_started = time.time()
                clf = common.make_classifier(variant, seed)
                clf.fit(x_train, y[train_idx])
                pred = clf.predict(x_valid)
                oof[variant.name][valid_idx] = pred
                fold_rows[variant.name].append(
                    {
                        "fold": fold,
                        "train_rows": int(len(train_idx)),
                        "valid_rows": int(len(valid_idx)),
                        "train_sessions": int(len(set(groups[train_idx].tolist()))),
                        "valid_sessions": int(len(set(groups[valid_idx].tolist()))),
                        "n_features": int(x_train.shape[1]),
                        "macro_f1": macro_f1(y[valid_idx], pred),
                        "vectorize_sec_shared": round(vectorize_sec, 3),
                        "fit_predict_sec": round(time.time() - fit_started, 3),
                    }
                )
                print(
                    f"[fold {fold}] {variant.name}: "
                    f"F1={fold_rows[variant.name][-1]['macro_f1']:.6f} features={x_train.shape[1]}"
                )
    if not np.all(covered == 1):
        raise AssertionError(f"OOF coverage must equal one for every row; values={np.unique(covered)}")

    results = []
    for variant in common.VARIANTS:
        pred = np.asarray(oof[variant.name], dtype=object)
        results.append(
            {
                "variant": common.variant_payload(variant),
                "oof_macro_f1": macro_f1(y, pred),
                "folds": fold_rows[variant.name],
                "mean_fold_macro_f1": float(np.mean([row["macro_f1"] for row in fold_rows[variant.name]])),
                "std_fold_macro_f1": float(np.std([row["macro_f1"] for row in fold_rows[variant.name]])),
            }
        )
    return results


def main() -> None:
    args = parse_args()
    started = time.time()
    holdout = common.load_holdout(args.holdout_npz)
    samples, ids, y, groups = common.load_train(args.data_dir)
    au_samples, au_ids, au_y, au_groups = common.select_nonholdout_au(
        samples, ids, y, groups, holdout["ids"]
    )
    holdout_au_ids = [str(x) for x in holdout["ids"] if common.is_au(str(x))]
    print(
        f"[data] nonholdout AU={len(au_ids)} rows/{len(set(au_groups.tolist()))} sessions; "
        f"frozen holdout AU={len(holdout_au_ids)} rows"
    )

    results = run_cv(au_samples, au_y, au_groups, args.folds, args.seed)
    by_name = {row["variant"]["name"]: row for row in results}
    baseline = by_name["baseline_char_C1"]
    best = max(results, key=lambda row: (row["oof_macro_f1"], row["variant"]["name"]))
    delta = float(best["oof_macro_f1"] - baseline["oof_macro_f1"])
    improved = best["variant"]["name"] != "baseline_char_C1" and delta > 0.0

    candidate_path = args.candidate_dir / "model.pkl"
    artifact_status = "not_written_baseline_won"
    artifact_sha = None
    if improved:
        selected = common.variant_from_payload(best["variant"])
        artifact = common.fit_artifact(au_samples, au_y, selected, args.seed)
        common.dump_artifact(candidate_path, artifact)
        artifact_sha = common.sha256(candidate_path)
        artifact_status = "written_oof_winner"

    summary = {
        "protocol": {
            "selection": f"GroupKFold(n_splits={args.folds}, shuffle=True, random_state={args.seed})",
            "scope": "sess_au rows after frozen holdout id and session exclusion",
            "metric": "pooled 14-class Macro-F1",
            "holdout_labels_used_for_selection": False,
            "variant_count": len(common.VARIANTS),
        },
        "inputs": {
            "data_dir": str(args.data_dir),
            "holdout_npz": str(args.holdout_npz),
            "holdout_npz_sha256": common.sha256(args.holdout_npz),
            "deployed_au_model": str(common.CURRENT_AU_MODEL),
            "deployed_au_model_sha256": common.sha256(common.CURRENT_AU_MODEL),
        },
        "rows": {
            "all_train": int(len(ids)),
            "all_au": int(np.sum([common.is_au(str(x)) for x in ids])),
            "nonholdout_au": int(len(au_ids)),
            "nonholdout_au_sessions": int(len(set(au_groups.tolist()))),
            "holdout": int(len(holdout["ids"])),
            "holdout_au": int(len(holdout_au_ids)),
        },
        "results": results,
        "baseline": baseline,
        "selected": best,
        "selected_oof_delta_vs_baseline": delta,
        "selected_beats_baseline": improved,
        "candidate": {
            "path": str(candidate_path),
            "status": artifact_status,
            "sha256": artifact_sha,
            "train_scope": "nonholdout sess_au only",
            "artifact_contract": {"keys": ["union", "clf"], "consumer": "submit/au_route.py::predict_proba"},
        },
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
            "numpy": np.__version__,
        },
        "elapsed_sec": round(time.time() - started, 3),
    }
    common.write_json(args.candidate_dir / "train_summary.json", summary)
    common.write_json(
        args.candidate_dir / "candidate_status.json",
        {
            "selected_variant": best["variant"]["name"],
            "selected_beats_baseline": improved,
            "oof_delta": delta,
            "model_written": improved,
            "model_sha256": artifact_sha,
        },
    )
    print(
        f"[done] selected={best['variant']['name']} OOF={best['oof_macro_f1']:.6f} "
        f"delta={delta:+.6f} candidate={artifact_status}"
    )


if __name__ == "__main__":
    main()
