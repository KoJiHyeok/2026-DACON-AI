"""Rebuild the lost AAR stacker using only CPU-friendly sklearn models.

The implementation follows the surviving recipe description: four SGD view
candidates, inner greedy blend selection, and a three-group-fold OOF logistic
stack validation.  It deliberately shares all serialization helpers with the
checked-in inference consumer by importing ``submit/aar_infer.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("AAR_PROJECT_ROOT", "C:/dev/2026-AI-DACON"))
if str(ROOT / "submit") not in sys.path:
    sys.path.insert(0, str(ROOT / "submit"))
import aar_infer as AAR  # noqa: E402


VIEWS = ("full", "prompt_context", "history", "action")


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_training(data_path: Path, labels_path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = AAR.read_jsonl(data_path)
    import pandas as pd
    labels = pd.read_csv(labels_path)
    if "id" not in labels or "action" not in labels:
        raise ValueError("labels CSV must contain id and action columns")
    by_id = dict(zip(labels["id"].astype(str), labels["action"].astype(str)))
    y = np.asarray([by_id.get(str(r.get("id", "")), "") for r in records], dtype=object)
    if np.any(y == "") or len(y) != len(records):
        raise ValueError("training records and labels are not aligned by id")
    unknown = sorted(set(y) - set(AAR.ACTIONS))
    if unknown:
        raise ValueError(f"unknown action labels: {unknown}")
    return records, y


def build_views(records: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    texts = [AAR.record_to_text(r) for r in records]
    prompts = [AAR.record_to_prompt_text(r) for r in records]
    return AAR.aar_views(records, texts, prompts)


def _pipeline(seed: int, alpha: float, max_iter: int, max_features: int) -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_features=max_features,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("clf", SGDClassifier(
            loss="log_loss", penalty="l2", alpha=alpha, max_iter=max_iter,
            tol=1e-3, class_weight="balanced", random_state=seed,
            early_stopping=False,
        )),
    ])


def _vectorizer(max_features: int) -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                           max_features=max_features, sublinear_tf=True,
                           strip_accents="unicode")


def _classifier(seed: int, alpha: float, max_iter: int) -> SGDClassifier:
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=alpha,
                         max_iter=max_iter, tol=1e-3, class_weight="balanced",
                         random_state=seed, early_stopping=False)


def _proba(model: Any, values: Sequence[Any]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=np.float64)
    out = np.zeros((len(raw), len(AAR.ACTIONS)), dtype=np.float64)
    for j, label in enumerate(getattr(model, "classes_", AAR.ACTIONS)):
        if str(label) in AAR.ACTIONS:
            out[:, AAR.ACTIONS.index(str(label))] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def greedy_blend(probas: Sequence[np.ndarray], y: np.ndarray, steps: int = 10) -> tuple[np.ndarray, list[float]]:
    """Select a nonnegative simplex blend by coordinate-greedy macro-F1."""
    if not probas:
        raise ValueError("at least one candidate is required")
    n = len(probas)
    weights = np.zeros(n, dtype=np.int64)
    current = np.zeros_like(probas[0], dtype=np.float64)
    best = -1.0
    for _ in range(max(1, steps)):
        winner = None
        winner_score = best
        for i, proba in enumerate(probas):
            trial = (current * weights.sum() + proba) / (weights.sum() + 1.0)
            score = f1_score(y, [AAR.ACTIONS[j] for j in trial.argmax(1)], average="macro")
            if score > winner_score + 1e-12:
                winner, winner_score = i, score
        if winner is None:
            break
        current = (current * weights.sum() + probas[winner]) / (weights.sum() + 1.0)
        weights[winner] += 1
        best = winner_score
    if not weights.sum():
        weights[0] = 1
        current = probas[0]
    return current, (weights / weights.sum()).tolist()


def train_aar(records: list[dict[str, Any]], y: np.ndarray, output_dir: Path,
              *, seed: int = 42, alpha: float = 3e-5, max_iter: int = 40,
              folds: int = 3, max_features: int = 50_000) -> dict[str, Any]:
    ids = np.asarray([str(r.get("id", "")) for r in records], dtype=object)
    groups = np.asarray([x.rsplit("-step_", 1)[0] for x in ids], dtype=object)
    views = build_views(records)
    # Fit each vocabulary once, matching the surviving build_views-once note.
    # Reusing the sparse matrices avoids refitting four large vocabularies for
    # every fold while keeping the SGD fit itself group-separated.
    matrices = {}
    for view in VIEWS:
        vectorizer = _vectorizer(max_features)
        matrices[view] = vectorizer.fit_transform(views[view])
    n = len(records)
    oof = {view: np.zeros((n, len(AAR.ACTIONS)), dtype=np.float64) for view in VIEWS}
    splitter = GroupKFold(n_splits=folds)
    fold_scores: list[float] = []
    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(n), y, groups)):
        candidates = []
        for offset, view in enumerate(VIEWS):
            model = _classifier(seed + fold * 100 + offset, alpha, max_iter)
            model.fit(matrices[view][train_idx], y[train_idx])
            oof[view][valid_idx] = _proba(model, matrices[view][valid_idx])
            candidates.append(oof[view][valid_idx])
        _, weights = greedy_blend(candidates, y[valid_idx], steps=10)
        blend = sum(p * w for p, w in zip(candidates, weights))
        fold_scores.append(float(f1_score(y[valid_idx], [AAR.ACTIONS[j] for j in blend.argmax(1)], average="macro")))

    matrix = np.hstack([oof[v] for v in VIEWS]).astype(np.float32)
    # sklearn 1.8 removed the legacy ``multi_class`` constructor argument;
    # lbfgs still selects multinomial handling automatically for this target.
    stacker = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                 random_state=seed)
    stacker.fit(matrix, y)
    final_components = {}
    for offset, view in enumerate(VIEWS):
        model = _pipeline(seed + offset, alpha, max_iter, max_features)
        model.fit(views[view], y)
        final_components[f"sgd_{view}"] = model
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"components": final_components, "stacker": stacker,
                "actions": list(AAR.ACTIONS), "metadata": {"seed": seed, "views": list(VIEWS)}}
    joblib.dump(artifact, output_dir / "aar_models.joblib", compress=3)
    config = {"enabled": True, "model_file": "aar_models.joblib", "use_stacker": True,
              "components": [{"name": f"sgd_{v}", "kind": "sgd", "view": v, "weight": 1.0} for v in VIEWS],
              "stacker_components": [f"sgd_{v}" for v in VIEWS], "use_bias": False,
              "class_bias": [0.0] * len(AAR.ACTIONS)}
    (output_dir / "aar_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return {"rows": n, "folds": folds, "group_count": int(len(set(groups))),
            "fold_macro_f1": fold_scores, "oof_macro_f1": float(np.mean(fold_scores)),
            "views": list(VIEWS), "max_features": max_features,
            "output": str(output_dir)}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("AAR_DATA", str(DATA_ROOT / "data" / "train.jsonl")))
    parser.add_argument("--labels", default=os.getenv("AAR_LABELS", str(DATA_ROOT / "data" / "train_labels.csv")))
    parser.add_argument("--output", default=os.getenv("AAR_OUTPUT", str(ROOT / "submit" / "model")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("AAR_SEED", "42")))
    parser.add_argument("--alpha", type=float, default=float(os.getenv("AAR_ALPHA", "3e-5")))
    parser.add_argument("--max-iter", type=int, default=int(os.getenv("AAR_MAX_ITER", "40")))
    parser.add_argument("--max-features", type=int, default=int(os.getenv("AAR_MAX_FEATURES", "50000")))
    parser.add_argument("--folds", type=int, default=int(os.getenv("AAR_FOLDS", "3")))
    args = parser.parse_args(argv)
    records, y = load_training(Path(args.data), Path(args.labels))
    result = train_aar(records, y, Path(args.output), seed=args.seed, alpha=args.alpha,
                        max_iter=args.max_iter, folds=args.folds, max_features=args.max_features)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
