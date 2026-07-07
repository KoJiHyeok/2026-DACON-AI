from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np
from sklearn.metrics import f1_score

from .constants import ACTIONS


def apply_class_bias(proba: np.ndarray, bias: Sequence[float]) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(proba, dtype=np.float32), 1e-12, 1.0))
    logits = logits + np.asarray(bias, dtype=np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_bias_vector(
    proba: np.ndarray,
    class_bias: Sequence[float] | None = None,
    rule_bias: np.ndarray | Sequence[Sequence[float]] | None = None,
    specialist_bias: np.ndarray | Sequence[Sequence[float]] | None = None,
) -> np.ndarray:
    logits = np.log(np.clip(np.asarray(proba, dtype=np.float32), 1e-12, 1.0))
    if class_bias is not None:
        logits = logits + np.asarray(class_bias, dtype=np.float32)
    if rule_bias is not None:
        logits = logits + np.asarray(rule_bias, dtype=np.float32)
    if specialist_bias is not None:
        logits = logits + np.asarray(specialist_bias, dtype=np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def fit_rule_bias(
    proba: np.ndarray,
    y_true: Sequence[str],
    rule_hits: np.ndarray,
    actions: Sequence[str] = ACTIONS,
    candidates: Sequence[float] = (-0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 0.75),
) -> Tuple[list[float], float]:
    hits = np.asarray(rule_hits, dtype=np.float32)
    if hits.shape != proba.shape:
        raise ValueError(f"rule_hits shape {hits.shape} must match proba shape {proba.shape}")
    bias = np.zeros(len(actions), dtype=np.float32)
    best = macro_f1(y_true, proba, actions)
    for class_idx in range(len(actions)):
        local_best = best
        local_value = 0.0
        for value in candidates:
            trial_bias = np.zeros_like(hits)
            trial_bias[:, class_idx] = hits[:, class_idx] * float(value)
            score = macro_f1(y_true, apply_bias_vector(proba, rule_bias=trial_bias), actions)
            if score > local_best + 1e-7:
                local_best = score
                local_value = float(value)
        if local_best > best + 1e-7:
            bias[class_idx] = local_value
            best = local_best
    return bias.astype(float).tolist(), best


def labels_from_proba(proba: np.ndarray, actions: Sequence[str] = ACTIONS) -> list[str]:
    return [actions[int(i)] for i in np.asarray(proba).argmax(axis=1)]


def macro_f1(y_true: Sequence[str], proba: np.ndarray, actions: Sequence[str] = ACTIONS) -> float:
    return float(f1_score(y_true, labels_from_proba(proba, actions), average="macro"))


def fit_class_bias(
    proba: np.ndarray,
    y_true: Sequence[str],
    actions: Sequence[str] = ACTIONS,
    rounds: int = 5,
) -> Tuple[list[float], float]:
    bias = np.zeros(len(actions), dtype=np.float32)
    best = macro_f1(y_true, proba, actions)
    deltas = [-0.85, -0.60, -0.40, -0.25, -0.16, -0.10, -0.06, 0.06, 0.10, 0.16, 0.25, 0.40, 0.60, 0.85]
    for _ in range(rounds):
        changed = False
        for idx in range(len(actions)):
            current = float(bias[idx])
            local_score = best
            local_value = current
            for delta in deltas:
                trial = bias.copy()
                trial[idx] = current + float(delta)
                score = macro_f1(y_true, apply_class_bias(proba, trial), actions)
                if score > local_score + 1e-7:
                    local_score = score
                    local_value = float(trial[idx])
            if local_score > best + 1e-7:
                bias[idx] = local_value
                best = local_score
                changed = True
        if not changed:
            break
    return bias.astype(float).tolist(), best
