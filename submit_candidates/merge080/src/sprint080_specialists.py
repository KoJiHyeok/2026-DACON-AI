from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np

from .constants import ACTIONS, ACTION_TO_ID
from .sprint080_ensemble import evaluate_proba


GROUPS: Dict[str, list[str]] = {
    "group_a": ["read_file", "grep_search", "list_directory", "glob_pattern"],
    "edit": ["edit_file", "write_file", "apply_patch"],
    "execute": ["run_bash", "run_tests", "lint_or_typecheck"],
    "dialogue": ["ask_user", "plan_task", "respond_only", "web_search"],
}


@dataclass
class SpecialistResult:
    enabled: bool
    macro_before: float
    macro_after: float
    fixed: int
    broken: int
    net_gain: int
    applied: int
    reason: str


def _same_group_topk(proba: np.ndarray, group_ids: set[int], top_k: int = 3) -> np.ndarray:
    top = np.argsort(-proba, axis=1)[:, :top_k]
    return np.array([sum(int(x in group_ids) for x in row) >= 2 for row in top], dtype=bool)


def apply_group_specialist(
    base_proba: np.ndarray,
    specialist_proba: np.ndarray,
    group_actions: Sequence[str],
    *,
    margin_threshold: float = 0.12,
) -> np.ndarray:
    out = np.asarray(base_proba, dtype=np.float64).copy()
    group_ids = {ACTION_TO_ID[a] for a in group_actions}
    top = np.argsort(-base_proba, axis=1)[:, :2]
    margin = base_proba[np.arange(len(base_proba)), top[:, 0]] - base_proba[np.arange(len(base_proba)), top[:, 1]]
    mask = _same_group_topk(base_proba, group_ids, top_k=3) & (margin <= margin_threshold)
    for row_idx in np.where(mask)[0]:
        best_group = max(group_ids, key=lambda idx: specialist_proba[row_idx, idx])
        out[row_idx, list(group_ids)] = 0.0
        out[row_idx, best_group] = 1.0
    return out


def evaluate_specialist_gate(
    y_true_int: np.ndarray,
    base_proba: np.ndarray,
    specialist_proba: np.ndarray,
    group_actions: Sequence[str],
    *,
    margin_threshold: float = 0.12,
) -> SpecialistResult:
    base_pred = np.asarray(base_proba).argmax(axis=1)
    corrected = apply_group_specialist(base_proba, specialist_proba, group_actions, margin_threshold=margin_threshold)
    corr_pred = corrected.argmax(axis=1)
    before = evaluate_proba(y_true_int, base_proba)["macro_f1"]
    after = evaluate_proba(y_true_int, corrected)["macro_f1"]
    changed = base_pred != corr_pred
    fixed = int(np.sum(changed & (base_pred != y_true_int) & (corr_pred == y_true_int)))
    broken = int(np.sum(changed & (base_pred == y_true_int) & (corr_pred != y_true_int)))
    enabled = bool(after > before)
    return SpecialistResult(
        enabled=enabled,
        macro_before=float(before),
        macro_after=float(after),
        fixed=fixed,
        broken=broken,
        net_gain=fixed - broken,
        applied=int(np.sum(changed)),
        reason="enabled_by_oof_gain" if enabled else "disabled_no_oof_gain",
    )
