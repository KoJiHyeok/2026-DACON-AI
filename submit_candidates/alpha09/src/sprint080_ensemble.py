from __future__ import annotations

from typing import Dict, Mapping, Sequence

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

from .constants import ACTIONS


def _normalize(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64)
    out = np.clip(out, 0.0, None)
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / out.shape[1]), where=row_sum > 0)


def weighted_average(component_probas: Mapping[str, np.ndarray], weights: Mapping[str, float] | None = None) -> np.ndarray:
    if not component_probas:
        raise ValueError("component_probas is empty")
    first = next(iter(component_probas.values()))
    out = np.zeros_like(first, dtype=np.float64)
    total = 0.0
    for name, proba in component_probas.items():
        weight = float(weights.get(name, 1.0) if weights else 1.0)
        if weight <= 0:
            continue
        if proba.shape != first.shape:
            raise ValueError(f"component {name} shape mismatch: {proba.shape} != {first.shape}")
        out += weight * _normalize(proba)
        total += weight
    if total <= 0:
        raise ValueError("sum of weights must be positive")
    return _normalize(out / total)


def stack_feature_matrix(component_probas: Mapping[str, np.ndarray]) -> np.ndarray:
    if not component_probas:
        raise ValueError("component_probas is empty")
    arrays = []
    n = None
    for name in sorted(component_probas):
        arr = _normalize(component_probas[name])
        if n is None:
            n = arr.shape[0]
        if arr.shape[0] != n or arr.shape[1] != len(ACTIONS):
            raise ValueError(f"component {name} has invalid shape {arr.shape}")
        maxp = arr.max(axis=1, keepdims=True)
        part = np.partition(arr, -2, axis=1)
        margin = (part[:, -1] - part[:, -2]).reshape(-1, 1)
        entropy = (-(arr * np.log(arr + 1e-12)).sum(axis=1)).reshape(-1, 1)
        arrays.extend([arr, maxp, margin, entropy])
    return np.hstack(arrays)


def fit_logreg_stacker_oof(
    component_probas: Mapping[str, np.ndarray],
    y_int: np.ndarray,
    fold_ids: np.ndarray,
    *,
    seed: int = 42,
) -> np.ndarray:
    x = stack_feature_matrix(component_probas)
    out = np.zeros((len(y_int), len(ACTIONS)), dtype=np.float64)
    for fold in sorted(set(int(v) for v in fold_ids)):
        valid = fold_ids == fold
        train = ~valid
        model = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, random_state=seed)
        model.fit(x[train], y_int[train])
        raw = model.predict_proba(x[valid])
        aligned = np.zeros((valid.sum(), len(ACTIONS)), dtype=np.float64)
        for src_idx, label in enumerate(model.classes_):
            aligned[:, int(label)] = raw[:, src_idx]
        out[valid] = aligned
    return _normalize(out)


def fit_extratrees_stacker_oof(
    component_probas: Mapping[str, np.ndarray],
    y_int: np.ndarray,
    fold_ids: np.ndarray,
    *,
    seed: int = 42,
    n_estimators: int = 600,
) -> np.ndarray:
    x = stack_feature_matrix(component_probas)
    out = np.zeros((len(y_int), len(ACTIONS)), dtype=np.float64)
    for fold in sorted(set(int(v) for v in fold_ids)):
        valid = fold_ids == fold
        train = ~valid
        model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
            max_features="sqrt",
            min_samples_leaf=2,
        )
        model.fit(x[train], y_int[train])
        raw = model.predict_proba(x[valid])
        aligned = np.zeros((valid.sum(), len(ACTIONS)), dtype=np.float64)
        for src_idx, label in enumerate(model.classes_):
            aligned[:, int(label)] = raw[:, src_idx]
        out[valid] = aligned
    return _normalize(out)


def evaluate_proba(y_true_int: Sequence[int], proba: np.ndarray) -> Dict[str, object]:
    from tools.fast_f1 import fast_confusion_matrix, fast_macro_f1, fast_per_class_f1

    y = np.asarray(y_true_int, dtype=np.int64)
    pred = np.asarray(proba).argmax(axis=1).astype(np.int64)
    per_class = fast_per_class_f1(y, pred, len(ACTIONS))
    cm = fast_confusion_matrix(y, pred, len(ACTIONS))
    pairs = []
    for i, true_label in enumerate(ACTIONS):
        for j, pred_label in enumerate(ACTIONS):
            if i != j and cm[i, j] > 0:
                pairs.append({"true": true_label, "pred": pred_label, "count": int(cm[i, j])})
    pairs.sort(key=lambda x: x["count"], reverse=True)
    return {
        "macro_f1": fast_macro_f1(y, pred, len(ACTIONS)),
        "class_f1": {action: float(per_class[idx]) for idx, action in enumerate(ACTIONS)},
        "class_worst_f1": float(per_class.min()),
        "confusion_pairs": pairs[:50],
    }
