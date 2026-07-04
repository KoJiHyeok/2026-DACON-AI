"""Prototype R3: first-step class-wise prior/bias adjustment.

For each StratifiedGroupKFold fold:
  1. Fit the same flat 14-way LinearSVC baseline as proto_hier.py.
  2. Estimate a train-only first-step prior shift:
       log P(action | history_len == 0) - log P(action)
  3. Add lambda * prior_shift only to validation rows with no history.

This mirrors the intended state-conditioned prior probe without using the
validation labels to estimate the bias vector. The lambda grid is local-CV
model selection only; any deployment still needs a leaderboard gate because
the old calib_v1 global bias improved holdout but hurt LB.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

from proto_hier import ACTIONS, build_classifier, build_preprocessor, load_training_frame, resolve_data_root


DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "scripts" / "hierarchy" / "_out"


def _aligned_scores(clf, x_valid_matrix) -> np.ndarray:
    scores = clf.decision_function(x_valid_matrix)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    classes = list(clf.classes_)
    aligned = np.full((x_valid_matrix.shape[0], len(ACTIONS)), -1e9, dtype=float)
    for src_idx, cls in enumerate(classes):
        aligned[:, ACTIONS.index(cls)] = scores[:, src_idx]
    return aligned


def _prior_shift(y_train: np.ndarray, first_mask: np.ndarray, alpha: float) -> np.ndarray:
    total = len(y_train)
    first_total = int(first_mask.sum())
    global_counts = np.asarray([(y_train == action).sum() for action in ACTIONS], dtype=float)
    first_counts = np.asarray([((y_train == action) & first_mask).sum() for action in ACTIONS], dtype=float)
    global_prob = (global_counts + alpha) / (total + alpha * len(ACTIONS))
    first_prob = (first_counts + alpha) / (first_total + alpha * len(ACTIONS))
    return np.log(first_prob) - np.log(global_prob)


def run_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    out_dir: Path,
    lambdas: list[float],
    alpha: float,
    only_folds: set[int] | None = None,
) -> None:
    groups = df["session_id"].to_numpy()
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_rows = []
    per_class_rows = []
    prior_rows = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(df, y, groups), start=1):
        if only_folds and fold not in only_folds:
            continue
        x_train = df.iloc[train_idx].reset_index(drop=True)
        x_valid = df.iloc[valid_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_valid = y[valid_idx]

        preprocessor = build_preprocessor()
        x_train_matrix = preprocessor.fit_transform(x_train)
        x_valid_matrix = preprocessor.transform(x_valid)
        clf = build_classifier()
        clf.fit(x_train_matrix, y_train)
        scores = _aligned_scores(clf, x_valid_matrix)
        first_train = x_train["n_history"].to_numpy() == 0
        first_valid = x_valid["n_history"].to_numpy() == 0
        bias = _prior_shift(y_train, first_train, alpha=alpha)

        for action, value in zip(ACTIONS, bias):
            prior_rows.append(
                {
                    "fold": fold,
                    "action": action,
                    "log_prior_shift_first_vs_global": value,
                    "train_global_rate": float(np.mean(y_train == action)),
                    "train_first_step_rate": float(np.mean(y_train[first_train] == action))
                    if np.any(first_train)
                    else 0.0,
                }
            )

        for lam in lambdas:
            adjusted = scores.copy()
            adjusted[first_valid] += lam * bias
            pred = np.asarray([ACTIONS[i] for i in np.argmax(adjusted, axis=1)])
            fold_rows.append(
                {
                    "fold": fold,
                    "lambda": lam,
                    "macro_f1": f1_score(y_valid, pred, labels=ACTIONS, average="macro", zero_division=0),
                    "first_step_macro_f1": f1_score(
                        y_valid[first_valid],
                        pred[first_valid],
                        labels=ACTIONS,
                        average="macro",
                        zero_division=0,
                    )
                    if np.any(first_valid)
                    else 0.0,
                    "non_first_macro_f1": f1_score(
                        y_valid[~first_valid],
                        pred[~first_valid],
                        labels=ACTIONS,
                        average="macro",
                        zero_division=0,
                    )
                    if np.any(~first_valid)
                    else 0.0,
                    "n_first_valid": int(first_valid.sum()),
                }
            )
            class_f1 = f1_score(y_valid, pred, labels=ACTIONS, average=None, zero_division=0)
            for action, value in zip(ACTIONS, class_f1):
                per_class_rows.append(
                    {
                        "fold": fold,
                        "lambda": lam,
                        "action": action,
                        "f1": value,
                    }
                )

        baseline = next(row for row in reversed(fold_rows) if row["fold"] == fold and row["lambda"] == 0.0)
        best = max((row for row in fold_rows if row["fold"] == fold), key=lambda row: row["macro_f1"])
        print(
            f"fold {fold}: baseline={baseline['macro_f1']:.5f} "
            f"best_lambda={best['lambda']} best={best['macro_f1']:.5f} "
            f"first_rows={baseline['n_first_valid']}",
            flush=True,
        )
        _write_outputs(out_dir, fold_rows, per_class_rows, prior_rows, partial=True)

    _write_outputs(out_dir, fold_rows, per_class_rows, prior_rows, partial=False)


def _write_outputs(
    out_dir: Path,
    fold_rows: list[dict],
    per_class_rows: list[dict],
    prior_rows: list[dict],
    partial: bool,
) -> None:
    pd.DataFrame(fold_rows).to_csv(out_dir / "first_step_bias_fold_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / "first_step_bias_per_class_f1.csv", index=False)
    pd.DataFrame(prior_rows).to_csv(out_dir / "first_step_bias_prior_shift.csv", index=False)
    if partial:
        pd.DataFrame(fold_rows).to_csv(out_dir / "first_step_bias_fold_metrics_partial.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(out_dir / "first_step_bias_per_class_f1_partial.csv", index=False)
        pd.DataFrame(prior_rows).to_csv(out_dir / "first_step_bias_prior_shift_partial.csv", index=False)

    summary = (
        pd.DataFrame(fold_rows)
        .groupby("lambda", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            first_step_macro_f1_mean=("first_step_macro_f1", "mean"),
            non_first_macro_f1_mean=("non_first_macro_f1", "mean"),
        )
        .sort_values("macro_f1_mean", ascending=False)
    )
    summary.to_csv(out_dir / "first_step_bias_summary.csv", index=False)
    if not partial:
        print(summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; keeps first N rows")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--only-folds", default=None, help="Comma-separated fold numbers to run, e.g. 1,3")
    parser.add_argument(
        "--lambdas",
        default="0,0.125,0.25,0.5,0.75,1,1.5,2",
        help="Comma-separated first-step bias strengths",
    )
    args = parser.parse_args()
    lambdas = [float(x) for x in args.lambdas.split(",") if x.strip()]

    data_root = resolve_data_root(args.data_root)
    print(f"data_root={data_root}")
    df, y = load_training_frame(data_root, max_rows=args.max_rows)
    print(f"loaded rows={len(df):,} groups={df['session_id'].nunique():,}")
    only_folds = {int(x) for x in args.only_folds.split(",")} if args.only_folds else None
    run_cv(
        df,
        y,
        n_splits=args.folds,
        seed=args.seed,
        out_dir=Path(args.out_dir),
        lambdas=lambdas,
        alpha=args.alpha,
        only_folds=only_folds,
    )


if __name__ == "__main__":
    main()
