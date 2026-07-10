from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer

from .common import ACTIONS


# Isolated from the removed alpha09 snapshot's sprint080_features.py.
FEATURE_SOURCE = "submit_candidates/alpha09/src/sprint080_features.py"
FEATURE_SOURCE_COMMIT = "4400df449a672aceffbe15eac0773e4a76c46a3a"
FEATURE_SOURCE_SHA256 = "356cff34fe6bd658938a49c5bdd5e5ea64f13e0973f94d95621db116d0bf81c3"

KOREAN_KEYWORDS = {
    "has_read_word": ["읽어", "열어", "열", "보여", "내용", "확인"],
    "has_search_word": ["찾아", "검색", "어디서", "쓰이는지", "참조", "정의"],
    "has_list_word": ["목록", "구조", "폴더", "디렉토리"],
    "has_edit_word": ["수정", "고쳐", "바꿔", "개선"],
    "has_patch_word": ["패치", "diff", "apply_patch"],
    "has_run_word": ["실행", "돌려", "명령"],
    "has_test_word": ["테스트", "pytest", "unittest"],
    "has_lint_word": ["lint", "린트", "타입체크", "typecheck", "mypy", "ruff", "eslint"],
    "has_ask_word": ["물어", "질문", "확인해줘"],
    "has_plan_word": ["계획", "설계", "로드맵", "단계"],
    "has_web_word": ["최신", "웹검색", "인터넷", "검색해"],
}
ENGLISH_KEYWORDS = {
    "has_read_word": ["read", "open", "show content", "cat"],
    "has_search_word": ["search", "grep", "find", "reference", "definition", "usage", "symbol"],
    "has_list_word": ["list", "tree", "ls", "directory", "folder", "structure"],
    "has_edit_word": ["edit", "modify", "fix", "change"],
    "has_patch_word": ["patch", "diff", "apply patch"],
    "has_run_word": ["run", "execute", "bash", "shell", "terminal", "command"],
    "has_test_word": ["test", "pytest", "unittest"],
    "has_lint_word": ["lint", "typecheck", "type check", "mypy", "ruff", "eslint"],
    "has_ask_word": ["ask", "clarify", "question", "choose"],
    "has_plan_word": ["plan", "design", "architecture", "roadmap", "steps"],
    "has_web_word": ["web", "latest", "internet", "lookup", "online"],
}

PATH_RE = re.compile(r"(?:(?:[A-Za-z]:)?[./\\]?[\\\w.-]+[/\\])+[\w.-]+\.[A-Za-z0-9]{1,8}")
EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}\b")
COMMAND_RE = re.compile(
    r"\b(?:python|pip|pytest|npm|pnpm|yarn|git|ruff|mypy|eslint|bash|sh)\b",
    re.I,
)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _history(record: dict[str, Any]) -> list[dict[str, Any]]:
    history = record.get("history")
    return history if isinstance(history, list) else []


def action_sequence(record: dict[str, Any]) -> list[str]:
    sequence: list[str] = []
    for item in _history(record):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            name = item.get("name") or item.get("action") or item.get("tool")
            if name:
                sequence.append(str(name))
    return sequence


def _count_user_turns(history: Iterable[dict[str, Any]]) -> int:
    return sum(
        1
        for item in history
        if isinstance(item, dict) and item.get("role") == "user"
    )


def _workspace(record: dict[str, Any]) -> dict[str, Any]:
    meta = record.get("session_meta")
    workspace = meta.get("workspace") if isinstance(meta, dict) else {}
    return workspace if isinstance(workspace, dict) else {}


def _flatten_action_payloads(record: dict[str, Any]) -> str:
    parts: list[str] = []
    for item in _history(record):
        if not isinstance(item, dict) or item.get("role") != "assistant_action":
            continue
        parts.append(_text(item.get("args")))
        parts.append(_text(item.get("result_summary")))
    return " ".join(parts)


def structured_features(record: dict[str, Any]) -> dict[str, float]:
    prompt = _text(record.get("current_prompt"))
    prompt_lower = prompt.lower()
    history = _history(record)
    sequence = action_sequence(record)
    workspace = _workspace(record)
    raw_open_files = workspace.get("open_files")
    open_files = raw_open_files if isinstance(raw_open_files, list) else []
    payload = _flatten_action_payloads(record)
    payload_lower = payload.lower()

    features: dict[str, float] = {
        "history_length": float(len(history)),
        "user_turn_count": float(_count_user_turns(history)),
        "assistant_action_count": float(len(sequence)),
        "open_files_count": float(len(open_files)),
        "file_extension_count": float(
            len(EXT_RE.findall(prompt + " " + " ".join(map(str, open_files))))
        ),
        "git_dirty": float(bool(workspace.get("git_dirty"))),
        "loc": float(workspace.get("loc") or 0),
        "current_prompt_length": float(len(prompt)),
        "current_prompt_token_count": float(len(prompt.split())),
        "has_file_path": float(bool(PATH_RE.search(prompt))),
        "has_extension": float(bool(EXT_RE.search(prompt))),
        "has_glob_pattern": float("*." in prompt or "*" in prompt or "glob" in prompt_lower),
        "args_path_token_count": float(len(PATH_RE.findall(payload))),
        "args_extension_count": float(len(EXT_RE.findall(payload))),
        "args_command_token_count": float(len(COMMAND_RE.findall(payload))),
        "result_summary_has_error": float(
            any(
                word in payload_lower
                for word in ["error", "failed", "traceback", "exception", "실패", "에러"]
            )
        ),
    }
    for key in sorted(set(KOREAN_KEYWORDS) | set(ENGLISH_KEYWORDS)):
        words = KOREAN_KEYWORDS.get(key, []) + ENGLISH_KEYWORDS.get(key, [])
        features[key] = float(any(word.lower() in prompt_lower for word in words))

    last_action = sequence[-1] if sequence else "none"
    previous_action = sequence[-2] if len(sequence) >= 2 else "none"
    features[f"last_action={last_action}"] = 1.0
    features[f"prev_action={previous_action}"] = 1.0
    for first, second in zip(sequence[-6:], sequence[-5:]):
        features[f"action_bigram={first}>{second}"] = 1.0
    for first, second, third in zip(sequence[-6:], sequence[-5:], sequence[-4:]):
        features[f"action_trigram={first}>{second}>{third}"] = 1.0

    features[f"last_ci_status={workspace.get('last_ci_status', 'none')}"] = 1.0
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict):
        for language, value in language_mix.items():
            if isinstance(value, (int, float)):
                features[f"language_mix={language}"] = float(value)
    return features


def feature_dicts(records: Iterable[dict[str, Any]]) -> list[dict[str, float]]:
    return [structured_features(record) for record in records]


def numeric_feature_names(actions: Sequence[str] = ACTIONS) -> tuple[str, ...]:
    labels = tuple(str(action) for action in actions)
    return (
        *(f"baseline_prob={action}" for action in labels),
        *(f"e5_prob={action}" for action in labels),
        "baseline_max",
        "e5_max",
        "baseline_margin",
        "e5_margin",
        "baseline_entropy",
        "e5_entropy",
    )


def _validate_component(values: np.ndarray, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.number) or raw.ndim != 2:
        raise ValueError(f"{name} must be a numeric matrix")
    array = np.asarray(raw, dtype=np.float64)
    if array.shape[1] != len(ACTIONS):
        raise ValueError(f"{name} must have {len(ACTIONS)} columns")
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} contains invalid probabilities")
    if not np.allclose(array.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"{name} rows do not sum to one")
    return array


def teammate_numeric_features(baseline: np.ndarray, e5: np.ndarray) -> np.ndarray:
    baseline_array = _validate_component(baseline, "baseline")
    e5_array = _validate_component(e5, "e5")
    if baseline_array.shape != e5_array.shape:
        raise ValueError("baseline and e5 shapes differ")

    baseline_part = np.partition(baseline_array, -2, axis=1)
    e5_part = np.partition(e5_array, -2, axis=1)
    return np.hstack(
        [
            baseline_array,
            e5_array,
            baseline_array.max(axis=1, keepdims=True),
            e5_array.max(axis=1, keepdims=True),
            (baseline_part[:, -1] - baseline_part[:, -2]).reshape(-1, 1),
            (e5_part[:, -1] - e5_part[:, -2]).reshape(-1, 1),
            (-(baseline_array * np.log(baseline_array + 1e-12)).sum(axis=1)).reshape(-1, 1),
            (-(e5_array * np.log(e5_array + 1e-12)).sum(axis=1)).reshape(-1, 1),
        ]
    )


def build_teammate_matrix(
    baseline: np.ndarray,
    e5: np.ndarray,
    records: Sequence[dict[str, Any]],
    *,
    vectorizer: DictVectorizer | None = None,
    fit_vectorizer: bool = False,
) -> tuple[sparse.csr_matrix, DictVectorizer]:
    numeric = teammate_numeric_features(baseline, e5)
    if len(records) != numeric.shape[0]:
        raise ValueError("records length does not match probability rows")
    dictionaries = feature_dicts(records)
    if fit_vectorizer:
        if vectorizer is not None:
            raise ValueError("vectorizer must be None when fit_vectorizer=True")
        vectorizer = DictVectorizer(sparse=True)
        structured = vectorizer.fit_transform(dictionaries)
    else:
        if vectorizer is None:
            raise ValueError("a fitted vectorizer is required for transform")
        structured = vectorizer.transform(dictionaries)
    matrix = sparse.hstack(
        [sparse.csr_matrix(numeric), structured],
        format="csr",
    )
    return matrix, vectorizer
