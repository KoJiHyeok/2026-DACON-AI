from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from .action_policy_features import (
    edit_apply_write_features,
    file_navigation_features,
    response_planning_features,
    run_test_lint_features,
)
from .constants import ACTIONS
from .ensemble import predict_proba_aligned


DEFAULT_PAIRS: tuple[tuple[str, str], ...] = (
    ("edit_file", "apply_patch"),
    ("edit_file", "write_file"),
    ("grep_search", "read_file"),
    ("run_bash", "run_tests"),
    ("lint_or_typecheck", "run_tests"),
    ("ask_user", "respond_only"),
    ("plan_task", "respond_only"),
    ("web_search", "grep_search"),
    ("glob_pattern", "list_directory"),
)


def pair_key(left: str, right: str) -> str:
    a, b = sorted([left, right])
    return f"{a}__{b}"


def make_specialist_model(alpha: float = 3e-5) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 3),
            min_df=2,
            max_features=90_000,
            sublinear_tf=True,
            token_pattern=r"(?u)\b[^\s]+\b",
        )),
        ("char", TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            max_features=60_000,
            sublinear_tf=True,
        )),
    ])
    clf = SGDClassifier(
        loss="log_loss",
        alpha=alpha,
        penalty="l2",
        max_iter=35,
        tol=1e-4,
        class_weight="balanced",
        n_jobs=-1,
        random_state=777,
    )
    return Pipeline([("features", features), ("clf", clf)])


def train_specialists(
    texts: Sequence[str],
    labels: Sequence[str],
    pairs: Iterable[Tuple[str, str]] = DEFAULT_PAIRS,
) -> Dict[str, object]:
    models: Dict[str, object] = {}
    y = np.asarray(labels)
    for left, right in pairs:
        mask = np.isin(y, [left, right])
        if int(mask.sum()) < 50:
            continue
        model = make_specialist_model()
        model.fit([texts[i] for i in np.where(mask)[0]], y[mask].tolist())
        models[pair_key(left, right)] = model
    return models


def apply_specialists(
    proba: np.ndarray,
    texts: Sequence[str],
    specialists: Dict[str, object],
    margin_threshold: float = 0.12,
    confidence_threshold: float = 0.58,
    actions: Sequence[str] = ACTIONS,
) -> np.ndarray:
    if not specialists:
        return proba
    adjusted = np.asarray(proba, dtype=np.float32).copy()
    order = np.argsort(adjusted, axis=1)
    for row_idx in range(adjusted.shape[0]):
        top1 = int(order[row_idx, -1])
        top2 = int(order[row_idx, -2])
        margin = float(adjusted[row_idx, top1] - adjusted[row_idx, top2])
        if margin > margin_threshold:
            continue
        key = pair_key(actions[top1], actions[top2])
        model = specialists.get(key)
        if model is None:
            continue
        spec = predict_proba_aligned(model, [texts[row_idx]], actions)[0]
        chosen = int(spec.argmax())
        conf = float(spec[chosen])
        if chosen in (top1, top2) and conf >= confidence_threshold:
            adjusted[row_idx, top1] *= 0.92
            adjusted[row_idx, top2] *= 0.92
            adjusted[row_idx, chosen] += conf * 0.25
            adjusted[row_idx] /= max(float(adjusted[row_idx].sum()), 1e-12)
    return adjusted


class PolicySpecialist:
    target_actions: tuple[str, ...] = ()
    feature_fn: Callable[[Dict[str, Any]], Dict[str, float]]

    def __init__(
        self,
        n_estimators: int = 240,
        min_samples_leaf: int = 2,
        random_state: int = 2026,
    ) -> None:
        self.vectorizer = DictVectorizer(sparse=False)
        self.model = ExtraTreesClassifier(
            n_estimators=n_estimators,
            max_features="sqrt",
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
        self.classes_: list[str] = []
        self.is_fitted = False

    def _features(self, records: Sequence[Dict[str, Any]]) -> list[Dict[str, float]]:
        return [self.feature_fn(record) for record in records]

    def fit(self, records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "PolicySpecialist":
        y = np.asarray([str(label) for label in labels])
        mask = np.isin(y, self.target_actions)
        unique = sorted(set(y[mask].tolist()))
        if int(mask.sum()) < max(8, len(self.target_actions) * 2) or len(unique) < 2:
            self.classes_ = unique
            self.is_fitted = False
            return self
        filtered_records = [records[int(i)] for i in np.where(mask)[0]]
        x = self.vectorizer.fit_transform(self._features(filtered_records))
        self.model.fit(x, y[mask].tolist())
        self.classes_ = [str(x) for x in self.model.classes_]
        self.is_fitted = True
        return self

    def predict_proba(self, records: Sequence[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(records), len(ACTIONS)), dtype=np.float32)
        if not records:
            return out
        if not self.is_fitted:
            target_indices = [ACTIONS.index(action) for action in self.target_actions if action in ACTIONS]
            if target_indices:
                out[:, target_indices] = 1.0 / len(target_indices)
            return out
        x = self.vectorizer.transform(self._features(records))
        raw = np.asarray(self.model.predict_proba(x), dtype=np.float32)
        for source_idx, label in enumerate(self.classes_):
            if label in ACTIONS:
                out[:, ACTIONS.index(label)] = raw[:, source_idx]
        row_sums = out.sum(axis=1, keepdims=True)
        missing = row_sums[:, 0] <= 0
        if np.any(missing):
            target_indices = [ACTIONS.index(action) for action in self.target_actions if action in ACTIONS]
            if target_indices:
                missing_rows = np.where(missing)[0]
                out[np.ix_(missing_rows, target_indices)] = 1.0 / len(target_indices)
        return out

    def maybe_override(
        self,
        records: Sequence[Dict[str, Any]],
        base_proba: np.ndarray,
        margin_threshold: float = 0.15,
        confidence_threshold: float = 0.42,
        y_true: Sequence[str] | None = None,
    ) -> tuple[np.ndarray, Dict[str, int]]:
        adjusted = np.asarray(base_proba, dtype=np.float32).copy()
        if adjusted.size == 0:
            return adjusted, {"eligible": 0, "applied": 0, "corrected": 0, "damaged": 0}
        target_indices = {ACTIONS.index(action) for action in self.target_actions if action in ACTIONS}
        spec = self.predict_proba(records)
        order = np.argsort(adjusted, axis=1)
        stats = {"eligible": 0, "applied": 0, "corrected": 0, "damaged": 0}
        truth = [str(x) for x in y_true] if y_true is not None else None
        for row_idx in range(adjusted.shape[0]):
            top1 = int(order[row_idx, -1])
            top2 = int(order[row_idx, -2])
            if top1 not in target_indices or top2 not in target_indices:
                continue
            margin = float(adjusted[row_idx, top1] - adjusted[row_idx, top2])
            if margin > margin_threshold:
                continue
            stats["eligible"] += 1
            chosen = int(spec[row_idx].argmax())
            conf = float(spec[row_idx, chosen])
            if chosen not in target_indices or conf < confidence_threshold:
                continue
            before = int(adjusted[row_idx].argmax())
            adjusted[row_idx, list(target_indices)] *= 0.82
            adjusted[row_idx, chosen] += 0.35 * conf
            adjusted[row_idx] /= max(float(adjusted[row_idx].sum()), 1e-12)
            after = int(adjusted[row_idx].argmax())
            if after != before:
                stats["applied"] += 1
                if truth is not None:
                    true_idx = ACTIONS.index(truth[row_idx]) if truth[row_idx] in ACTIONS else -1
                    if before != true_idx and after == true_idx:
                        stats["corrected"] += 1
                    elif before == true_idx and after != true_idx:
                        stats["damaged"] += 1
        return adjusted, stats


class FileNavigationSpecialist(PolicySpecialist):
    target_actions = ("read_file", "grep_search", "list_directory", "glob_pattern")
    feature_fn = staticmethod(file_navigation_features)


class EditApplyWriteSpecialist(PolicySpecialist):
    target_actions = ("edit_file", "apply_patch", "write_file")
    feature_fn = staticmethod(edit_apply_write_features)


class RunTestLintSpecialist(PolicySpecialist):
    target_actions = ("run_bash", "run_tests", "lint_or_typecheck")
    feature_fn = staticmethod(run_test_lint_features)


class ResponsePlanningSpecialist(PolicySpecialist):
    target_actions = ("ask_user", "respond_only", "plan_task")
    feature_fn = staticmethod(response_planning_features)
