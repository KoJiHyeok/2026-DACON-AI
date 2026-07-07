from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from .constants import ACTIONS


ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
N_ACTIONS = len(ACTIONS)

ACTION_GROUPS: Dict[str, Tuple[str, ...]] = {
    "group_a_file_navigation": ("read_file", "grep_search", "list_directory", "glob_pattern"),
    "group_b_editing": ("edit_file", "write_file", "apply_patch"),
    "group_c_execution": ("run_bash", "run_tests", "lint_or_typecheck"),
    "group_d_user_or_plan": ("ask_user", "plan_task"),
    "group_e_web_or_response": ("web_search", "respond_only"),
}

ACTION_TO_GROUP = {
    action: group
    for group, actions in ACTION_GROUPS.items()
    for action in actions
}


def _record_text(record: Mapping[str, Any]) -> str:
    parts = [str(record.get("current_prompt", ""))]
    for turn in record.get("history", []) or []:
        if isinstance(turn, Mapping):
            parts.append(str(turn.get("content", "")))
            parts.append(str(turn.get("name", "")))
            parts.append(str(turn.get("result_summary", "")))
    return "\n".join(parts).lower()


def _action_sequence(record: Mapping[str, Any]) -> List[str]:
    seq: List[str] = []
    for turn in record.get("history", []) or []:
        if isinstance(turn, Mapping) and turn.get("role") == "assistant_action":
            name = str(turn.get("name", ""))
            if name:
                seq.append(name)
    return seq


def _rank_vector(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks.astype(np.float32)


def _entropy(values: np.ndarray) -> float:
    total = float(values.sum())
    if total <= 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def intent_flags(record: Mapping[str, Any]) -> Dict[str, int]:
    text = _record_text(record)
    flags = {
        "has_file_path": int(bool(re.search(r"(?<!\w)(?:[\w.-]+/)+[\w.-]+\.[a-zA-Z0-9]+", text))),
        "has_extension": int(bool(re.search(r"\.[a-zA-Z0-9]{1,8}\b", text))),
        "has_glob_pattern": int(bool(re.search(r"[*?]\.?\w*|\*\.[a-zA-Z0-9]+", text))),
        "has_directory_word": int(any(x in text for x in ("directory", "folder", "tree", "디렉토리", "폴더", "목록", "구조"))),
        "has_search_word": int(any(x in text for x in ("search", "grep", "find", "찾아", "검색"))),
        "has_reference_word": int(any(x in text for x in ("reference", "usage", "where used", "참조", "쓰이는지", "호출"))),
        "has_read_word": int(any(x in text for x in ("read", "open", "cat ", "show content", "열어", "읽어", "내용", "보여"))),
        "has_edit_word": int(any(x in text for x in ("edit", "modify", "change", "fix", "수정", "고쳐", "바꿔"))),
        "has_patch_word": int(any(x in text for x in ("patch", "diff", "apply_patch", "패치"))),
        "has_write_word": int(any(x in text for x in ("write", "create", "new file", "만들", "생성", "작성"))),
        "has_test_word": int(any(x in text for x in ("pytest", "unittest", "test", "테스트", "검증"))),
        "has_lint_word": int(any(x in text for x in ("lint", "mypy", "ruff", "eslint", "typecheck", "타입"))),
        "has_run_word": int(any(x in text for x in ("run", "execute", "돌려", "실행"))),
        "has_shell_word": int(any(x in text for x in ("bash", "shell", "terminal", "command", "npm ", "pip ", "python "))),
        "has_question_word": int(any(x in text for x in ("?", "which", "clarify", "선택", "확인", "물어"))),
        "has_plan_word": int(any(x in text for x in ("plan", "approach", "architecture", "roadmap", "계획", "설계", "단계"))),
        "has_web_word": int(any(x in text for x in ("web", "internet", "latest", "look it up", "검색해", "최신"))),
        "has_url": int(bool(re.search(r"https?://", text))),
        "has_error_word": int(any(x in text for x in ("error", "exception", "traceback", "failed", "crash", "에러", "실패", "깨져"))),
        "has_stacktrace": int(any(x in text for x in ("traceback", "stack trace", "exception:", "error:"))),
        "has_symbol_word": int(any(x in text for x in ("function", "class", "method", "symbol", "함수", "클래스", "정의"))),
        "has_list_word": int(any(x in text for x in ("list", "ls", "tree", "목록", "구조"))),
        "has_wildcard_word": int(any(x in text for x in ("glob", "wildcard", "pattern", "패턴", "확장자", "*. "))),
    }
    return flags


def build_candidate_lists(
    component_probas: Mapping[str, np.ndarray],
    source_names: Sequence[str],
    cap: int | str,
    extra_mask: np.ndarray | None = None,
) -> List[List[int]]:
    if not source_names:
        raise ValueError("source_names must not be empty")
    first = component_probas[source_names[0]]
    n_samples, n_classes = first.shape
    if cap == "all":
        return [list(range(n_classes)) for _ in range(n_samples)]
    k = max(1, min(int(cap), n_classes))
    masks = np.zeros((n_samples, n_classes), dtype=bool)
    best_rank = np.full((n_samples, n_classes), 999.0, dtype=np.float32)
    mean_score = np.zeros((n_samples, n_classes), dtype=np.float64)
    for source in source_names:
        proba = component_probas[source]
        if proba.shape != (n_samples, n_classes):
            raise ValueError(f"{source} shape mismatch: {proba.shape}")
        top = np.argpartition(proba, -k, axis=1)[:, -k:]
        ranks = np.argsort(np.argsort(-proba, axis=1), axis=1) + 1
        row = np.arange(n_samples)[:, None]
        masks[row, top] = True
        best_rank = np.minimum(best_rank, ranks.astype(np.float32))
        mean_score += proba
    mean_score /= float(len(source_names))
    if extra_mask is not None:
        if extra_mask.shape != masks.shape:
            raise ValueError(f"extra_mask shape mismatch: {extra_mask.shape}")
        masks |= extra_mask
    candidate_lists: List[List[int]] = []
    for i in range(n_samples):
        ids = np.where(masks[i])[0].tolist()
        ids.sort(key=lambda idx: (best_rank[i, idx], -mean_score[i, idx], idx))
        candidate_lists.append([int(idx) for idx in ids])
    return candidate_lists


def candidate_oracle_prediction(
    candidate_lists: Sequence[Sequence[int]],
    y_true_int: np.ndarray,
    base_pred_int: np.ndarray,
) -> np.ndarray:
    pred = np.asarray(base_pred_int, dtype=np.int32).copy()
    for i, candidates in enumerate(candidate_lists):
        true_idx = int(y_true_int[i])
        if true_idx in candidates:
            pred[i] = true_idx
    return pred


class CandidateRowBuilder:
    def build_train_rows(
        self,
        records: Sequence[Mapping[str, Any]],
        labels: Sequence[str],
        component_probas: Mapping[str, np.ndarray],
        candidate_lists: Sequence[Sequence[int]],
        meta: Mapping[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
        y = np.asarray([ACTION_TO_IDX[str(label)] for label in labels], dtype=np.int32)
        rows, row_record_indices, row_action_indices = self.build_inference_rows(records, component_probas, candidate_lists, meta)
        targets = np.asarray([1 if int(action_idx) == int(y[int(record_idx)]) else 0 for record_idx, action_idx in zip(row_record_indices, row_action_indices)], dtype=np.int8)
        return rows, targets, row_record_indices, row_action_indices

    def build_inference_rows(
        self,
        records: Sequence[Mapping[str, Any]],
        component_probas: Mapping[str, np.ndarray],
        candidate_lists: Sequence[Sequence[int]],
        meta: Mapping[str, Any] | None = None,
    ) -> Tuple[List[Dict[str, Any]], np.ndarray, np.ndarray]:
        meta = meta or {}
        base_name = str(meta.get("base_proba_name") or ("meta_et" if "meta_et" in component_probas else "meta"))
        source_topk = int(meta.get("source_topk", 5))
        component_names = list(component_probas)
        rows: List[Dict[str, Any]] = []
        row_record_indices: List[int] = []
        row_action_indices: List[int] = []

        ranks_by_component = {name: np.apply_along_axis(_rank_vector, 1, proba) for name, proba in component_probas.items()}
        top1_by_component = {name: np.argmax(proba, axis=1) for name, proba in component_probas.items()}
        base_proba = component_probas[base_name]
        base_pred = np.argmax(base_proba, axis=1)
        base_sorted = np.sort(base_proba, axis=1)

        for i, (record, candidates) in enumerate(zip(records, candidate_lists)):
            flags = intent_flags(record)
            seq = _action_sequence(record)
            history_len = len(record.get("history", []) or [])
            base_candidate_scores = np.asarray([base_proba[i, c] for c in candidates], dtype=np.float64) if candidates else np.asarray([0.0])
            record_best = float(base_candidate_scores.max())
            record_mean = float(base_candidate_scores.mean())
            record_median = float(np.median(base_candidate_scores))
            record_std = float(base_candidate_scores.std()) or 1.0
            agreement_counts = Counter(int(top1_by_component[name][i]) for name in component_names)
            agreement_values = np.asarray(list(agreement_counts.values()), dtype=np.float64)
            agreement_entropy = _entropy(agreement_values)
            top1_margin = float(base_sorted[i, -1] - base_sorted[i, -2]) if base_sorted.shape[1] >= 2 else 0.0
            top2_margin = float(base_sorted[i, -2] - base_sorted[i, -3]) if base_sorted.shape[1] >= 3 else 0.0

            for action_idx in candidates:
                action_idx = int(action_idx)
                action = ACTIONS[action_idx]
                row: Dict[str, Any] = {
                    "candidate_action_id": action_idx,
                    f"candidate_action_name={action}": 1,
                    f"candidate_action_group={ACTION_TO_GROUP.get(action, 'other')}": 1,
                    "candidate_count": len(candidates),
                    "history_len": history_len,
                    "base_top1_margin": top1_margin,
                    "base_top2_margin": top2_margin,
                    "agreement_count_top1": agreement_counts.get(action_idx, 0),
                    "agreement_entropy": agreement_entropy,
                    "base_is_record_top1": int(base_pred[i] == action_idx),
                    "last_action=none": 1 if not seq else 0,
                }
                if seq:
                    row[f"last_action={seq[-1]}"] = 1
                if len(seq) >= 2:
                    row[f"last_2_actions={seq[-2]}>{seq[-1]}"] = 1
                if len(seq) >= 3:
                    row[f"last_3_actions={seq[-3]}>{seq[-2]}>{seq[-1]}"] = 1
                for group, actions in ACTION_GROUPS.items():
                    row[f"is_{group}"] = int(action in actions)
                row.update(flags)

                source_count = 0
                rank_sum = 0.0
                best_source_rank = 999.0
                proba_values = []
                for name in component_names:
                    proba = component_probas[name][i]
                    score = float(proba[action_idx])
                    rank = float(ranks_by_component[name][i, action_idx])
                    sorted_scores = np.sort(proba)
                    top1 = float(sorted_scores[-1])
                    top2 = float(sorted_scores[-2]) if len(sorted_scores) >= 2 else 0.0
                    row[f"{name}_proba"] = score
                    row[f"{name}_rank"] = rank
                    row[f"{name}_inv_rank"] = 1.0 / rank
                    row[f"{name}_is_top1"] = int(rank <= 1)
                    row[f"{name}_is_top2"] = int(rank <= 2)
                    row[f"{name}_is_top3"] = int(rank <= 3)
                    row[f"{name}_is_top5"] = int(rank <= 5)
                    row[f"{name}_score_minus_top1"] = score - top1
                    row[f"{name}_score_minus_top2"] = score - top2
                    row[f"{name}_entropy"] = _entropy(proba)
                    in_topk = int(rank <= source_topk)
                    row[f"in_{name}_topk"] = in_topk
                    if in_topk:
                        source_count += 1
                        rank_sum += rank
                        best_source_rank = min(best_source_rank, rank)
                    proba_values.append(score)

                base_score = float(base_proba[i, action_idx])
                row["candidate_score"] = base_score
                row["candidate_score_minus_record_best"] = base_score - record_best
                row["candidate_score_minus_record_mean"] = base_score - record_mean
                row["candidate_score_minus_record_median"] = base_score - record_median
                row["candidate_score_zscore_within_record"] = (base_score - record_mean) / record_std
                row["source_count"] = source_count
                row["best_source_rank"] = best_source_rank if source_count else 999.0
                row["mean_source_rank"] = rank_sum / source_count if source_count else 999.0
                row["component_score_mean"] = float(np.mean(proba_values))
                row["component_score_std"] = float(np.std(proba_values))
                rows.append(row)
                row_record_indices.append(i)
                row_action_indices.append(action_idx)

        return rows, np.asarray(row_record_indices, dtype=np.int32), np.asarray(row_action_indices, dtype=np.int32)


class CandidateRanker:
    def __init__(self, model_type: str = "et", random_state: int = 42, **params: Any) -> None:
        self.model_type = model_type
        self.random_state = random_state
        self.params = params
        self.vectorizer: DictVectorizer | None = None
        self.model: Any = None

    def _make_model(self) -> Any:
        model_type = self.model_type
        if model_type == "lr":
            return LogisticRegression(max_iter=int(self.params.get("max_iter", 300)), class_weight="balanced", n_jobs=-1)
        if model_type == "sgd":
            return SGDClassifier(loss="log_loss", max_iter=int(self.params.get("max_iter", 1000)), tol=1e-3, class_weight="balanced", random_state=self.random_state)
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=int(self.params.get("n_estimators", 160)),
                min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),
                max_features=self.params.get("max_features", "sqrt"),
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=self.random_state,
            )
        if model_type == "hgb":
            return HistGradientBoostingClassifier(
                max_iter=int(self.params.get("max_iter", 120)),
                learning_rate=float(self.params.get("learning_rate", 0.06)),
                l2_regularization=float(self.params.get("l2_regularization", 0.02)),
                random_state=self.random_state,
            )
        return ExtraTreesClassifier(
            n_estimators=int(self.params.get("n_estimators", 180)),
            min_samples_leaf=int(self.params.get("min_samples_leaf", 2)),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=-1,
            random_state=self.random_state,
        )

    def fit(
        self,
        candidate_rows: Sequence[Mapping[str, Any]],
        candidate_targets: Sequence[int],
        groups: Sequence[Any] | None = None,
        sample_weight: Sequence[float] | None = None,
    ) -> "CandidateRanker":
        dense = self.model_type == "hgb"
        self.vectorizer = DictVectorizer(sparse=not dense)
        x = self.vectorizer.fit_transform(candidate_rows)
        self.model = self._make_model()
        y = np.asarray(candidate_targets, dtype=np.int8)
        if sample_weight is not None:
            self.model.fit(x, y, sample_weight=np.asarray(sample_weight, dtype=np.float64))
        else:
            self.model.fit(x, y)
        return self

    def predict_candidate_scores(self, candidate_rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
        if self.vectorizer is None or self.model is None:
            raise RuntimeError("CandidateRanker must be fitted before prediction")
        x = self.vectorizer.transform(candidate_rows)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(x)
            if proba.shape[1] == 1:
                return np.zeros(x.shape[0], dtype=np.float64)
            return proba[:, 1].astype(np.float64)
        decision = self.model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-decision))

    def select_actions(
        self,
        records: Sequence[Mapping[str, Any]],
        candidate_lists: Sequence[Sequence[int]],
        candidate_scores: Sequence[float],
        row_record_indices: Sequence[int],
        row_action_indices: Sequence[int],
        fallback: Sequence[int] | None = None,
    ) -> np.ndarray:
        return self.select_actions_from_scores(candidate_lists, candidate_scores, row_record_indices, row_action_indices, fallback)

    @staticmethod
    def select_actions_from_scores(
        candidate_lists: Sequence[Sequence[int]],
        candidate_scores: Sequence[float],
        row_record_indices: Sequence[int],
        row_action_indices: Sequence[int],
        fallback: Sequence[int] | None = None,
    ) -> np.ndarray:
        n_records = len(candidate_lists)
        if fallback is None:
            selected = np.asarray([int(cands[0]) if cands else 0 for cands in candidate_lists], dtype=np.int32)
        else:
            selected = np.asarray(fallback, dtype=np.int32).copy()
        best_scores = np.full(n_records, -math.inf, dtype=np.float64)
        for score, record_idx, action_idx in zip(candidate_scores, row_record_indices, row_action_indices):
            ridx = int(record_idx)
            value = float(score)
            if value > best_scores[ridx]:
                best_scores[ridx] = value
                selected[ridx] = int(action_idx)
        return selected
