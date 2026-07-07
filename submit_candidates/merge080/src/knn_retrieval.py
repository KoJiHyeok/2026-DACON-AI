from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List

import numpy as np

from .constants import ACTIONS
from .sprint080_features import action_sequence

PATH_RE = re.compile(r"(?:(?:[A-Za-z]:)?[./\\]?[\\\w.-]+[/\\])+[\w.-]+(?:\.[A-Za-z0-9]{1,8})?")
EXT_RE = re.compile(r"\.[A-Za-z0-9]{1,8}\b")
NUM_RE = re.compile(r"\b\d+\b")
QUOTE_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
SPACE_RE = re.compile(r"\s+")


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def normalize_text(value: Any, max_chars: int = 1600) -> str:
    text = as_text(value).lower()
    text = QUOTE_RE.sub(" <quote> ", text)
    text = PATH_RE.sub(" <path> ", text)
    text = EXT_RE.sub(" <ext> ", text)
    text = NUM_RE.sub(" <num> ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text[:max_chars]


def last_user_message(record: Dict[str, Any]) -> str:
    history = record.get("history") if isinstance(record.get("history"), list) else []
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") != "assistant_action":
            return as_text(item.get("content"))
    return ""


def last_actions(record: Dict[str, Any], n: int) -> str:
    seq = action_sequence(record)
    if not seq:
        return "none"
    return " > ".join(seq[-n:])


def compact_meta(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta") if isinstance(record.get("session_meta"), dict) else {}
    ws = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    open_files = ws.get("open_files") if isinstance(ws.get("open_files"), list) else []
    language_mix = ws.get("language_mix") if isinstance(ws.get("language_mix"), dict) else {}
    bits = [
        f"tier={meta.get('user_tier', 'na')}",
        f"lang={meta.get('language_pref', 'na')}",
        f"turn={meta.get('turn_index', 'na')}",
        f"ci={ws.get('last_ci_status', 'na')}",
        f"dirty={int(bool(ws.get('git_dirty')))}",
        f"open={min(len(open_files), 5)}",
        f"loc={bucket(ws.get('loc'), (100, 1000, 5000, 20000, 100000))}",
    ]
    for lang, value in sorted(language_mix.items()):
        if isinstance(value, (int, float)) and value > 0.15:
            bits.append(f"mix={lang}")
    return " ".join(bits)


def bucket(value: Any, edges: tuple[int, ...]) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "na"
    for idx, edge in enumerate(edges):
        if numeric < edge:
            return f"b{idx}"
    return f"b{len(edges)}"


def path_extension_signature(record: Dict[str, Any]) -> str:
    chunks = [as_text(record.get("current_prompt"))]
    history = record.get("history") if isinstance(record.get("history"), list) else []
    for item in history[-10:]:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            chunks.append(as_text(item.get("args")))
            chunks.append(as_text(item.get("result_summary")))
    blob = " ".join(chunks).lower()
    exts = sorted(set(EXT_RE.findall(blob)))[:8]
    path_count = len(PATH_RE.findall(blob))
    return f"ext={' '.join(exts) if exts else 'none'} path_count={min(path_count, 5)}"


def result_keywords(record: Dict[str, Any]) -> str:
    history = record.get("history") if isinstance(record.get("history"), list) else []
    blob = " ".join(
        as_text(item.get("result_summary"))
        for item in history[-10:]
        if isinstance(item, dict) and item.get("role") == "assistant_action"
    ).lower()
    keys = [
        "error",
        "failed",
        "passed",
        "traceback",
        "found",
        "no matches",
        "files",
        "diff",
        "pytest",
        "mypy",
        "ruff",
        "eslint",
        "modified",
    ]
    return " ".join(k.replace(" ", "_") for k in keys if k in blob) or "none"


def keyword_group(record: Dict[str, Any]) -> str:
    text = as_text(record.get("current_prompt")).lower()
    groups: List[str] = []
    mapping = {
        "read": ["read", "open", "cat", "show", "view", "열어", "읽어", "내용", "확인", "보여"],
        "search": ["search", "grep", "find", "where", "reference", "usage", "definition", "symbol", "찾", "검색", "참조", "정의", "쓰이는"],
        "list": ["list", "tree", "ls", "directory", "folder", "structure", "목록", "폴더", "디렉토리", "구조"],
        "glob": ["glob", "wildcard", "pattern", "*.", "extension", "패턴", "확장자"],
        "edit": ["edit", "modify", "fix", "change", "수정", "고쳐", "변경"],
        "patch": ["patch", "diff", "apply_patch", "패치"],
        "run": ["run", "execute", "bash", "shell", "terminal", "command", "실행"],
        "test": ["test", "pytest", "unittest", "테스트"],
        "lint": ["lint", "typecheck", "mypy", "ruff", "eslint"],
        "ask": ["ask", "clarify", "question", "물어", "확인"],
        "plan": ["plan", "design", "architecture", "roadmap", "계획", "설계"],
        "web": ["web", "latest", "internet", "online", "최신"],
    }
    for name, words in mapping.items():
        if any(word in text for word in words):
            groups.append(name)
    return " ".join(groups[:4]) if groups else "none"


def retrieval_text(record: Dict[str, Any], view: str = "hybrid") -> str:
    prompt = normalize_text(record.get("current_prompt"), 1200)
    last_user = normalize_text(last_user_message(record), 800)
    seq = last_actions(record, 16)
    last = last_actions(record, 1)
    last2 = last_actions(record, 2)
    meta = compact_meta(record)
    path_sig = path_extension_signature(record)
    result_sig = result_keywords(record)
    kw = keyword_group(record)
    if view == "prompt":
        return f"prompt {prompt}"
    if view == "prompt_last":
        return f"prompt {prompt} last {last} kw {kw}"
    if view == "sequence":
        return f"seq {seq} last2 {last2} kw {kw} meta {meta}"
    if view == "result_path":
        return f"prompt {prompt} path {path_sig} result {result_sig} last {last}"
    return "\n".join(
        [
            f"prompt {prompt}",
            f"last_user {last_user}",
            f"kw {kw}",
            f"last {last}",
            f"last2 {last2}",
            f"seq {seq}",
            f"meta {meta}",
            f"path {path_sig}",
            f"result {result_sig}",
        ]
    )


def entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    return -(clipped * np.log(clipped)).sum(axis=1)


def posterior_from_neighbors(labels: np.ndarray, similarities: np.ndarray, num_classes: int = len(ACTIONS)) -> np.ndarray:
    sims = np.clip(similarities.astype(np.float64), 0.0, None)
    if sims.sum() <= 0:
        sims = np.ones_like(sims, dtype=np.float64)
    out = np.zeros(num_classes, dtype=np.float64)
    for label, sim in zip(labels, sims):
        out[int(label)] += float(sim)
    total = out.sum()
    if total <= 0:
        out[:] = 1.0 / num_classes
    else:
        out /= total
    return out
