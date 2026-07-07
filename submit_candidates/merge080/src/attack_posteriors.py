from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .constants import ACTIONS
from .sprint080_features import action_sequence


PATH_RE = re.compile(r"(?:(?:[A-Za-z]:)?[./\\]?[\\\w.-]+[/\\])+[\w.-]+(?:\.[A-Za-z0-9]{1,8})?")
EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}\b")
NUM_RE = re.compile(r"\b\d+\b")
QUOTE_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
SPACE_RE = re.compile(r"\s+")


KEYWORD_GROUPS = {
    "read": ["read", "open", "cat", "show content", "열어", "읽어", "내용", "보여"],
    "search": ["search", "grep", "find", "reference", "usage", "definition", "찾아", "검색", "참조", "정의", "쓰이는"],
    "list": ["list", "tree", "ls", "directory", "folder", "structure", "목록", "구조", "폴더", "디렉토리"],
    "glob": ["glob", "wildcard", "pattern", "extension", "*.", "패턴", "확장자", "전체"],
    "edit": ["edit", "modify", "fix", "change", "수정", "고쳐", "변경"],
    "patch": ["patch", "diff", "apply_patch", "패치"],
    "write": ["write", "create", "new file", "생성", "새 파일", "만들"],
    "test": ["test", "pytest", "unittest", "테스트"],
    "lint": ["lint", "typecheck", "mypy", "ruff", "eslint", "타입", "린트"],
    "run": ["run", "execute", "shell", "terminal", "bash", "실행", "명령"],
    "ask": ["ask", "clarify", "question", "확인", "질문", "물어"],
    "plan": ["plan", "design", "architecture", "roadmap", "계획", "설계"],
    "web": ["web", "latest", "internet", "online", "검색해", "최신"],
}


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_prompt(text: Any) -> str:
    out = _text(text).lower()
    out = QUOTE_RE.sub("<quote>", out)
    out = PATH_RE.sub("<path>", out)
    out = EXT_RE.sub("<ext>", out)
    out = NUM_RE.sub("<num>", out)
    out = SPACE_RE.sub(" ", out).strip()
    return out[:320]


def turn_bucket(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta") if isinstance(record.get("session_meta"), dict) else {}
    turn = meta.get("turn_index", 0)
    try:
        turn_i = int(turn)
    except Exception:
        return "turn_na"
    if turn_i <= 1:
        return "turn_0_1"
    if turn_i <= 3:
        return "turn_2_3"
    if turn_i <= 6:
        return "turn_4_6"
    return "turn_7p"


def prompt_keyword_group(record: Dict[str, Any]) -> str:
    text = _text(record.get("current_prompt")).lower()
    hits = []
    for group, words in KEYWORD_GROUPS.items():
        if any(word in text for word in words):
            hits.append(group)
    return "+".join(hits[:3]) if hits else "none"


def _history_action_names(record: Dict[str, Any]) -> List[str]:
    return action_sequence(record)


def _result_keywords(record: Dict[str, Any]) -> str:
    history = record.get("history") if isinstance(record.get("history"), list) else []
    text_parts = []
    for item in history[-6:]:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            text_parts.append(_text(item.get("result_summary")))
    text = " ".join(text_parts).lower()
    keys = []
    for token in ["error", "failed", "passed", "traceback", "found", "no matches", "files", "diff", "pytest"]:
        if token in text:
            keys.append(token.replace(" ", "_"))
    return "+".join(keys[:4]) if keys else "none"


def _args_extension_signature(record: Dict[str, Any]) -> str:
    history = record.get("history") if isinstance(record.get("history"), list) else []
    blob = []
    for item in history[-8:]:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            blob.append(_text(item.get("args")))
            blob.append(_text(item.get("result_summary")))
    text = " ".join(blob).lower()
    exts = sorted(set(EXT_RE.findall(text)))[:5]
    return "+".join(exts) if exts else "no_ext"


def memory_signatures(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    prompt_sig = normalize_prompt(record.get("current_prompt"))
    seq = _history_action_names(record)
    last = seq[-1] if seq else "none"
    seq_sig = ">".join(seq[-5:]) if seq else "none"
    hist_actions = ">".join(seq[-10:]) if seq else "none"
    return [
        ("prompt", prompt_sig),
        ("prompt_last", f"{prompt_sig}|last={last}"),
        ("prompt_seq", f"{prompt_sig}|seq={seq_sig}"),
        ("prompt_hist_actions", f"{prompt_sig}|hist={hist_actions}"),
        ("prompt_turn", f"{prompt_sig}|turn={turn_bucket(record)}"),
        ("args_ext", f"{_args_extension_signature(record)}|kw={prompt_keyword_group(record)}"),
        ("result_kw", f"{_result_keywords(record)}|last={last}|kw={prompt_keyword_group(record)}"),
    ]


def transition_signatures(record: Dict[str, Any]) -> List[Tuple[str, str]]:
    seq = _history_action_names(record)
    last = seq[-1] if seq else "none"
    prev = seq[-2] if len(seq) >= 2 else "none"
    last2 = ">".join(seq[-2:]) if len(seq) >= 2 else last
    last3 = ">".join(seq[-3:]) if len(seq) >= 3 else last2
    meta = record.get("session_meta") if isinstance(record.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    ci = workspace.get("last_ci_status", "none")
    dirty = int(bool(workspace.get("git_dirty")))
    open_bucket = "open0" if not open_files else ("open1_2" if len(open_files) <= 2 else "open3p")
    kw = prompt_keyword_group(record)
    return [
        ("last", last),
        ("prev_last", f"{prev}>{last}"),
        ("last2", last2),
        ("last3", last3),
        ("last_kw", f"{last}|kw={kw}"),
        ("turn_last", f"{turn_bucket(record)}|{last}"),
        ("ci_dirty_last", f"ci={ci}|dirty={dirty}|{last}"),
        ("open_last", f"{open_bucket}|{last}"),
    ]


class PosteriorModel:
    def __init__(
        self,
        kind: str,
        min_count: int = 2,
        min_confidence: float = 0.55,
        alpha: float = 0.25,
    ) -> None:
        if kind not in {"memory", "transition"}:
            raise ValueError("kind must be 'memory' or 'transition'")
        self.kind = kind
        self.min_count = int(min_count)
        self.min_confidence = float(min_confidence)
        self.alpha = float(alpha)
        self.tables: Dict[str, Dict[str, List[int]]] = {}
        self.global_counts: List[int] = [0] * len(ACTIONS)

    def _signatures(self, record: Dict[str, Any]) -> List[Tuple[str, str]]:
        return memory_signatures(record) if self.kind == "memory" else transition_signatures(record)

    def fit(self, records: Sequence[Dict[str, Any]], y: Sequence[int]) -> "PosteriorModel":
        tables: Dict[str, Dict[str, Counter[int]]] = defaultdict(lambda: defaultdict(Counter))
        global_counter: Counter[int] = Counter()
        for record, label in zip(records, y):
            label_i = int(label)
            global_counter[label_i] += 1
            for source, key in self._signatures(record):
                tables[source][key][label_i] += 1
        self.tables = {
            source: {key: [int(counter.get(i, 0)) for i in range(len(ACTIONS))] for key, counter in table.items()}
            for source, table in tables.items()
        }
        self.global_counts = [int(global_counter.get(i, 0)) for i in range(len(ACTIONS))]
        return self

    def _counts_to_proba(self, counts: Sequence[int]) -> np.ndarray:
        arr = np.asarray(counts, dtype=np.float64) + self.alpha
        return arr / arr.sum()

    @staticmethod
    def _entropy(proba: np.ndarray) -> float:
        return float(-(proba * np.log(proba + 1e-12)).sum())

    def predict_one(self, record: Dict[str, Any]) -> Tuple[np.ndarray, float, str, int]:
        best_proba = self._counts_to_proba(self.global_counts)
        best_conf = 0.0
        best_source = "global"
        best_count = int(sum(self.global_counts))
        for source, key in self._signatures(record):
            counts = self.tables.get(source, {}).get(key)
            if not counts:
                continue
            count = int(sum(counts))
            if count < self.min_count:
                continue
            proba = self._counts_to_proba(counts)
            conf = float(proba.max())
            if conf < self.min_confidence:
                continue
            # Prefer high confidence, then higher support.
            rank = (conf, math.log1p(count))
            best_rank = (best_conf, math.log1p(best_count))
            if rank > best_rank:
                best_proba = proba
                best_conf = conf
                best_source = source
                best_count = count
        return best_proba.astype(np.float64), best_conf, best_source, best_count

    def predict_proba(self, records: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        probas = np.zeros((len(records), len(ACTIONS)), dtype=np.float64)
        conf = np.zeros(len(records), dtype=np.float64)
        sources: List[str] = []
        counts = np.zeros(len(records), dtype=np.int64)
        for i, record in enumerate(records):
            proba, c, source, count = self.predict_one(record)
            probas[i] = proba
            conf[i] = c
            sources.append(source)
            counts[i] = count
        return probas, conf, sources, counts

