from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

from .constants import ACTIONS


def _model_classes(model: object) -> Sequence[str] | None:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(x) for x in classes]

    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        clf = named_steps.get("clf")
        classes = getattr(clf, "classes_", None)
        if classes is not None:
            return [str(x) for x in classes]
    return None


def predict_proba_aligned(
    model: object,
    texts: Sequence[str],
    actions: Sequence[str] = ACTIONS,
) -> np.ndarray:
    """Return probabilities in the canonical ACTIONS order."""
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(texts), dtype=np.float32)
        classes = _model_classes(model)
    else:
        preds = [str(x) for x in model.predict(texts)]
        raw = np.zeros((len(preds), len(actions)), dtype=np.float32)
        action_to_idx = {a: i for i, a in enumerate(actions)}
        for i, pred in enumerate(preds):
            if pred in action_to_idx:
                raw[i, action_to_idx[pred]] = 1.0
        classes = list(actions)

    if raw.shape[1] == len(actions) and not classes:
        return raw

    action_to_idx = {a: i for i, a in enumerate(actions)}
    aligned = np.zeros((raw.shape[0], len(actions)), dtype=np.float32)
    for src_idx, label in enumerate(classes or actions):
        dst_idx = action_to_idx.get(str(label))
        if dst_idx is not None and src_idx < raw.shape[1]:
            aligned[:, dst_idx] = raw[:, src_idx]
    return aligned


def weighted_average(
    parts: Iterable[Tuple[np.ndarray, float]],
) -> np.ndarray:
    total: np.ndarray | None = None
    weight_sum = 0.0
    for proba, weight in parts:
        if weight <= 0:
            continue
        arr = np.asarray(proba, dtype=np.float32)
        total = arr * weight if total is None else total + arr * weight
        weight_sum += weight

    if total is None or weight_sum <= 0:
        raise ValueError("At least one positive-weight probability matrix is required.")
    return total / weight_sum


def labels_from_proba(
    proba: np.ndarray,
    actions: Sequence[str] = ACTIONS,
) -> list[str]:
    indices = np.asarray(proba).argmax(axis=1)
    return [actions[int(i)] for i in indices]
