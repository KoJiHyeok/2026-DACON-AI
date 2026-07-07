from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .candidate_ranker import ACTIONS, ACTION_GROUPS, ACTION_TO_GROUP, ACTION_TO_IDX


GROUP_A = ("read_file", "grep_search", "list_directory", "glob_pattern")
GROUP_A_IDS = tuple(ACTION_TO_IDX[action] for action in GROUP_A)


def _record_text(record: Mapping[str, Any]) -> str:
    parts = [str(record.get("current_prompt", ""))]
    for turn in record.get("history", []) or []:
        if isinstance(turn, Mapping):
            parts.extend([
                str(turn.get("content", "")),
                str(turn.get("name", "")),
                str(turn.get("result_summary", "")),
            ])
    return "\n".join(parts).lower()


def _action_sequence(record: Mapping[str, Any]) -> List[str]:
    seq: List[str] = []
    for turn in record.get("history", []) or []:
        if isinstance(turn, Mapping) and turn.get("role") == "assistant_action":
            name = str(turn.get("name", ""))
            if name:
                seq.append(name)
    return seq


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def switch_intent_flags(record: Mapping[str, Any]) -> Dict[str, int]:
    text = _record_text(record)
    return {
        "has_search_word": int(any(x in text for x in ("search", "grep", "find", "검색", "찾아", "찾기", "참조", "어디서", "쓰이는지"))),
        "has_read_word": int(any(x in text for x in ("read", "open", "cat ", "show content", "열어", "읽어", "내용", "보여줘", "확인"))),
        "has_directory_word": int(any(x in text for x in ("directory", "folder", "tree", "list", "ls", "폴더", "디렉토리", "목록", "구조"))),
        "has_glob_pattern": int(bool(re.search(r"[*?]\.?\w*|\*\.[a-zA-Z0-9]+", text)) or any(x in text for x in ("glob", "wildcard", "패턴", "확장자"))),
        "has_file_path": int(bool(re.search(r"(?<!\w)(?:[\w.-]+[/\\])+[\w.-]+\.[a-zA-Z0-9]+", text))),
        "has_extension": int(bool(re.search(r"\.[a-zA-Z0-9]{1,8}\b", text))),
        "has_reference_word": int(any(x in text for x in ("reference", "usage", "symbol", "function", "class", "참조", "정의", "함수", "클래스", "호출"))),
        "has_run_word": int(any(x in text for x in ("run", "execute", "bash", "shell", "terminal", "실행", "돌려"))),
        "has_test_word": int(any(x in text for x in ("pytest", "unittest", "test", "테스트", "검증"))),
        "has_lint_word": int(any(x in text for x in ("lint", "mypy", "ruff", "eslint", "typecheck", "타입"))),
        "has_edit_word": int(any(x in text for x in ("edit", "modify", "fix", "patch", "수정", "고쳐", "패치"))),
        "has_question_word": int("?" in text or any(x in text for x in ("clarify", "which", "선택", "확인", "물어"))),
        "has_plan_word": int(any(x in text for x in ("plan", "approach", "architecture", "roadmap", "계획", "설계", "단계"))),
    }


def build_switch_targets(
    y_true: Sequence[int],
    base_pred: Sequence[int],
    ranker_pred: Sequence[int],
    mode: str = "flip_only",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y_true, dtype=np.int32)
    base = np.asarray(base_pred, dtype=np.int32)
    ranker = np.asarray(ranker_pred, dtype=np.int32)
    targets = np.zeros(len(y), dtype=np.int8)
    weights = np.ones(len(y), dtype=np.float64)
    outcomes: List[str] = []
    for i, (true_idx, base_idx, ranker_idx) in enumerate(zip(y, base, ranker)):
        base_ok = int(base_idx) == int(true_idx)
        ranker_ok = int(ranker_idx) == int(true_idx)
        if ranker_ok and not base_ok:
            outcomes.append("ranker_wins")
            targets[i] = 1
        elif base_ok and not ranker_ok:
            outcomes.append("base_wins")
            targets[i] = 0
        elif base_ok and ranker_ok:
            outcomes.append("same_correct")
            targets[i] = 0
            weights[i] = 0.15 if mode == "all_records_low_weight_same" else 0.0
        else:
            outcomes.append("same_wrong")
            targets[i] = 0
            weights[i] = 0.25 if mode == "all_records_low_weight_same" else 0.0
        if mode == "hard_cases_only" and int(base_idx) == int(ranker_idx):
            weights[i] = 0.0
    if mode == "flip_only":
        keep = np.isin(np.asarray(outcomes, dtype=object), ["ranker_wins", "base_wins"])
        weights = weights * keep.astype(np.float64)
    return targets, weights, np.asarray(outcomes, dtype=object)


def _margin(row: np.ndarray) -> float:
    if row.size < 2:
        return 0.0
    top = np.partition(row, -2)[-2:]
    return float(top[1] - top[0])


def _rank(row: np.ndarray, action_idx: int) -> int:
    order = np.argsort(-row, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(row) + 1)
    return int(ranks[int(action_idx)])


def switch_feature_rows(
    records: Sequence[Mapping[str, Any]],
    base_pred: Sequence[int],
    ranker_pred: Sequence[int],
    base_proba: np.ndarray,
    ranker_action_scores: np.ndarray,
    candidate_counts: Sequence[int],
    component_top1: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    base = np.asarray(base_pred, dtype=np.int32)
    ranker = np.asarray(ranker_pred, dtype=np.int32)
    counts = np.asarray(candidate_counts, dtype=np.int32)
    rows: List[Dict[str, Any]] = []
    for i, record in enumerate(records):
        base_idx = int(base[i])
        ranker_idx = int(ranker[i])
        base_action = ACTIONS[base_idx]
        ranker_action = ACTIONS[ranker_idx]
        base_row = base_proba[i]
        ranker_row = ranker_action_scores[i]
        seq = _action_sequence(record)
        row: Dict[str, Any] = {
            "base_top1_action": base_idx,
            "ranker_top1_action": ranker_idx,
            f"base_top1_action={base_action}": 1,
            f"ranker_top1_action={ranker_action}": 1,
            f"base_prediction_group={ACTION_TO_GROUP.get(base_action, 'other')}": 1,
            f"ranker_prediction_group={ACTION_TO_GROUP.get(ranker_action, 'other')}": 1,
            "base_top1_score": float(base_row[base_idx]),
            "ranker_top1_score": float(ranker_row[ranker_idx]),
            "base_ranker_score": float(base_row[ranker_idx]),
            "ranker_base_score": float(ranker_row[base_idx]),
            "ranker_minus_base_score": float(ranker_row[ranker_idx] - base_row[base_idx]),
            "ranker_target_minus_base_target": float(ranker_row[ranker_idx] - base_row[ranker_idx]),
            "base_top1_margin": _margin(base_row),
            "ranker_top1_margin": _margin(ranker_row),
            "base_ranker_action_same": int(base_idx == ranker_idx),
            "candidate_count": int(counts[i]),
            "base_ranker_rank": _rank(base_row, ranker_idx),
            "ranker_base_rank": _rank(ranker_row, base_idx),
            "is_group_a_case": int(base_idx in GROUP_A_IDS or ranker_idx in GROUP_A_IDS),
            "is_base_group_a": int(base_idx in GROUP_A_IDS),
            "is_ranker_group_a": int(ranker_idx in GROUP_A_IDS),
            "history_len": len(record.get("history", []) or []),
            "last_action=none": 1 if not seq else 0,
            "component_entropy": 0.0,
            "component_agreement_count": 0,
        }
        if seq:
            row[f"last_action={seq[-1]}"] = 1
        if len(seq) >= 2:
            row[f"last_2_actions={seq[-2]}>{seq[-1]}"] = 1
        for group, actions in ACTION_GROUPS.items():
            row[f"base_is_{group}"] = int(base_action in actions)
            row[f"ranker_is_{group}"] = int(ranker_action in actions)
        row.update(switch_intent_flags(record))
        pair = f"{base_action}->{ranker_action}"
        pair_flags = {
            "grep_search->read_file": "is_grep_vs_read",
            "read_file->grep_search": "is_read_vs_grep",
            "read_file->list_directory": "is_read_vs_list",
            "list_directory->read_file": "is_list_vs_read",
            "list_directory->glob_pattern": "is_list_vs_glob",
            "glob_pattern->list_directory": "is_glob_vs_list",
            "run_bash->run_tests": "is_run_vs_test",
            "run_tests->run_bash": "is_run_vs_test",
            "run_tests->lint_or_typecheck": "is_test_vs_lint",
            "lint_or_typecheck->run_tests": "is_test_vs_lint",
            "edit_file->apply_patch": "is_edit_vs_patch",
            "apply_patch->edit_file": "is_edit_vs_patch",
            "ask_user->respond_only": "is_ask_vs_respond",
            "respond_only->ask_user": "is_ask_vs_respond",
            "plan_task->respond_only": "is_plan_vs_respond",
            "respond_only->plan_task": "is_plan_vs_respond",
        }
        for flag in set(pair_flags.values()):
            row[flag] = 0
        if pair in pair_flags:
            row[pair_flags[pair]] = 1
        if component_top1 is not None:
            values = np.asarray(component_top1[i], dtype=np.int32)
            row["component_agreement_count"] = int((values == ranker_idx).sum())
            counts_for_entropy = np.bincount(values, minlength=len(ACTIONS)).astype(np.float64)
            row["component_entropy"] = _entropy(counts_for_entropy)
        rows.append(row)
    return rows


def _guarded_threshold(row: Mapping[str, Any], threshold: float, guard_mode: str) -> float:
    if guard_mode != "group_a_conservative" or not row.get("is_group_a_case"):
        return threshold
    base_idx = int(row["base_top1_action"])
    ranker_idx = int(row["ranker_top1_action"])
    base_action = ACTIONS[base_idx]
    ranker_action = ACTIONS[ranker_idx]
    adjusted = threshold
    if base_action == "list_directory" and ranker_action == "read_file":
        adjusted += 0.15
    if base_action == "grep_search" and ranker_action == "read_file" and (row.get("has_search_word") or row.get("has_reference_word")):
        adjusted += 0.10
    if base_action == "read_file" and ranker_action == "list_directory" and not row.get("has_directory_word"):
        adjusted += 0.15
    if ranker_action == "list_directory" and not row.get("has_directory_word"):
        adjusted += 0.10
    if ranker_action == "grep_search" and (row.get("has_search_word") or row.get("has_reference_word")):
        adjusted -= 0.05
    return min(max(adjusted, 0.0), 0.99)


def apply_switch_predictions(
    records: Sequence[Mapping[str, Any]],
    base_pred: Sequence[int],
    ranker_pred: Sequence[int],
    switch_scores: Sequence[float],
    threshold: float,
    guard_mode: str = "none",
    feature_rows: Sequence[Mapping[str, Any]] | None = None,
) -> np.ndarray:
    base = np.asarray(base_pred, dtype=np.int32)
    ranker = np.asarray(ranker_pred, dtype=np.int32)
    scores = np.asarray(switch_scores, dtype=np.float64)
    if feature_rows is None:
        dummy = np.zeros((len(base), len(ACTIONS)), dtype=np.float64)
        for i, (b, r) in enumerate(zip(base, ranker)):
            dummy[i, int(b)] = 1.0
            dummy[i, int(r)] = max(dummy[i, int(r)], float(scores[i]))
        feature_rows = switch_feature_rows(records, base, ranker, dummy, dummy, np.ones(len(base), dtype=np.int32))
    pred = base.copy()
    for i, row in enumerate(feature_rows):
        if int(base[i]) == int(ranker[i]):
            continue
        if float(scores[i]) >= _guarded_threshold(row, threshold, guard_mode):
            pred[i] = int(ranker[i])
    return pred


class RankerSwitch:
    def __init__(self, model_type: str = "lr", random_state: int = 42, **params: Any) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.params = params
        self.vectorizer: DictVectorizer | None = None
        self.model: Any = None
        self.constant_score: float | None = None

    def _make_model(self) -> Any:
        if self.model_type == "sgd":
            return SGDClassifier(loss="log_loss", max_iter=int(self.params.get("max_iter", 1000)), tol=1e-3, class_weight="balanced", random_state=self.random_state)
        if self.model_type == "et":
            return ExtraTreesClassifier(n_estimators=int(self.params.get("n_estimators", 200)), min_samples_leaf=int(self.params.get("min_samples_leaf", 3)), max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=self.random_state)
        if self.model_type == "rf":
            return RandomForestClassifier(n_estimators=int(self.params.get("n_estimators", 160)), min_samples_leaf=int(self.params.get("min_samples_leaf", 3)), max_features="sqrt", class_weight="balanced_subsample", n_jobs=-1, random_state=self.random_state)
        if self.model_type == "hgb":
            return HistGradientBoostingClassifier(max_iter=int(self.params.get("max_iter", 80)), learning_rate=float(self.params.get("learning_rate", 0.07)), l2_regularization=float(self.params.get("l2_regularization", 0.05)), random_state=self.random_state)
        return LogisticRegression(max_iter=int(self.params.get("max_iter", 400)), class_weight="balanced", n_jobs=-1)

    def fit(
        self,
        rows: Sequence[Mapping[str, Any]],
        targets: Sequence[int],
        sample_weight: Sequence[float] | None = None,
    ) -> "RankerSwitch":
        y = np.asarray(targets, dtype=np.int8)
        weights = None if sample_weight is None else np.asarray(sample_weight, dtype=np.float64)
        if weights is not None:
            keep = weights > 0
            rows = [rows[i] for i in np.where(keep)[0]]
            y = y[keep]
            weights = weights[keep]
        if len(y) == 0:
            self.constant_score = 0.0
            return self
        if len(np.unique(y)) == 1:
            self.constant_score = float(y[0])
            return self
        dense = self.model_type == "hgb"
        self.vectorizer = DictVectorizer(sparse=not dense)
        x = self.vectorizer.fit_transform(rows)
        self.model = self._make_model()
        if weights is None:
            self.model.fit(x, y)
        else:
            self.model.fit(x, y, sample_weight=weights)
        self.constant_score = None
        return self

    def predict_switch_scores(self, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.constant_score is not None:
            return np.full(len(rows), self.constant_score, dtype=np.float64)
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("RankerSwitch must be fitted before prediction")
        x = self.vectorizer.transform(rows)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            if proba.shape[1] == 1:
                return np.zeros(x.shape[0], dtype=np.float64)
            return proba[:, 1].astype(np.float64)
        decision = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))

    def select(
        self,
        records: Sequence[Mapping[str, Any]],
        base_pred: Sequence[int],
        ranker_pred: Sequence[int],
        switch_scores: Sequence[float],
        threshold: float,
        guard_mode: str = "none",
        feature_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> np.ndarray:
        return apply_switch_predictions(records, base_pred, ranker_pred, switch_scores, threshold, guard_mode, feature_rows)
