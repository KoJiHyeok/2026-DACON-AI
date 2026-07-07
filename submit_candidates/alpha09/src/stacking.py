from __future__ import annotations

from typing import Dict, Sequence

import numpy as np


def proba_stats(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=np.float32)
    sorted_arr = np.sort(arr, axis=1)
    max_p = sorted_arr[:, -1]
    second_p = sorted_arr[:, -2] if arr.shape[1] > 1 else np.zeros_like(max_p)
    margin = max_p - second_p
    entropy = -(arr * np.log(np.clip(arr, 1e-12, 1.0))).sum(axis=1) / np.log(arr.shape[1])
    return np.column_stack([max_p, margin, entropy]).astype(np.float32)


def top_one_hot(proba: np.ndarray, rank: int = 1) -> np.ndarray:
    arr = np.asarray(proba, dtype=np.float32)
    order = np.argsort(arr, axis=1)
    idx = order[:, -rank]
    out = np.zeros_like(arr, dtype=np.float32)
    out[np.arange(arr.shape[0]), idx] = 1.0
    return out


def build_stack_matrix(
    component_names: Sequence[str],
    component_probas: Dict[str, np.ndarray],
    include_stats: bool = True,
    include_top_ids: bool = True,
) -> np.ndarray:
    parts = []
    for name in component_names:
        proba = np.asarray(component_probas[name], dtype=np.float32)
        parts.append(proba)
        if include_stats:
            parts.append(proba_stats(proba))
        if include_top_ids:
            parts.append(top_one_hot(proba, 1))
            parts.append(top_one_hot(proba, 2))
    return np.hstack(parts).astype(np.float32)
