from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Tuple


def _clean_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            s = str(value)
    else:
        s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        return s[: max_chars // 2] + " … " + s[-max_chars // 2 :]
    return s


def _flatten_meta(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                items.extend(_flatten_meta(v, key))
            elif isinstance(v, list):
                if len(v) <= 8:
                    items.append((key, "|".join(_clean_text(x, 80) for x in v)))
                else:
                    items.append((key, f"list_len={len(v)} first=" + "|".join(_clean_text(x, 60) for x in v[:5])))
            else:
                items.append((key, _clean_text(v, 120)))
    return items


def serialize_session_meta(meta: Dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return "[SESSION_META] none"
    parts = [f"{k}={v}" for k, v in _flatten_meta(meta)]
    return "[SESSION_META] " + " ; ".join(parts)


def extract_action_sequence(history: Iterable[Dict[str, Any]]) -> List[str]:
    seq: List[str] = []
    for h in history or []:
        if isinstance(h, dict) and h.get("role") == "assistant_action":
            name = h.get("name") or h.get("action") or h.get("tool")
            if name:
                seq.append(str(name))
    return seq


def serialize_history(history: List[Dict[str, Any]] | None, max_turns: int = 12) -> str:
    if not isinstance(history, list) or not history:
        return "[HISTORY] empty"
    recent = history[-max_turns:]
    lines: List[str] = ["[HISTORY_RECENT]"]
    for i, h in enumerate(recent):
        if not isinstance(h, dict):
            lines.append(f"turn_{i}: {_clean_text(h, 500)}")
            continue
        role = _clean_text(h.get("role", "unknown"), 50)
        if role == "assistant_action":
            name = _clean_text(h.get("name") or h.get("action") or h.get("tool") or "unknown", 80)
            args = _clean_text(h.get("args", ""), 500)
            result = _clean_text(h.get("result_summary", h.get("result", "")), 500)
            lines.append(f"turn_{i}: role=assistant_action action={name} args={args} result={result}")
        else:
            content = _clean_text(h.get("content", h), 800)
            lines.append(f"turn_{i}: role={role} content={content}")
    seq = extract_action_sequence(history)
    if seq:
        lines.append("[ACTION_SEQUENCE] " + " > ".join(seq[-16:]))
        lines.append("[LAST_ACTION] " + seq[-1])
    return "\n".join(lines)


def keyword_tokens(prompt: str) -> str:
    p = prompt.lower()
    pairs = {
        "kw_test": ["test", "pytest", "unittest", "unit test", "테스트", "검증", "확인"],
        "kw_lint": ["lint", "typecheck", "type check", "mypy", "ruff", "eslint", "타입", "린트"],
        "kw_search": ["search", "grep", "find", "찾", "검색", "어디", "정의", "참조", "reference"],
        "kw_file": ["file", "파일", "열어", "읽", "수정", "패치", "저장", "open", "read"],
        "kw_dir": ["directory", "folder", "폴더", "디렉토리", "목록", "구조", "tree", "list"],
        "kw_web": ["web", "internet", "latest", "최신", "검색해서", "웹", "사이트", "뉴스", "lookup"],
        "kw_ask": ["?", "어떻게", "뭐", "무엇", "확인해", "which", "what", "how", "clarify"],
        "kw_run": ["run", "execute", "bash", "shell", "terminal", "실행", "터미널", "명령어", "커맨드"],
        "kw_plan": ["plan", "설계", "구조", "계획", "로드맵", "단계", "아키텍처", "approach"],
    }
    tokens: List[str] = []
    for token, words in pairs.items():
        if any(w in p for w in words):
            tokens.append(token)
    return " ".join(tokens)


def record_to_text(record: Dict[str, Any]) -> str:
    current_prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    meta = record.get("session_meta")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    meta_text = serialize_session_meta(meta if isinstance(meta, dict) else None)
    hist_text = serialize_history(history if isinstance(history, list) else None)
    last_action = seq[-1] if seq else "none"
    feature_line = "[DERIVED] " + " ".join([
        f"history_len={len(history) if isinstance(history, list) else 0}",
        f"last_action={last_action}",
        keyword_tokens(current_prompt),
    ])
    return "\n".join([
        f"[ID] {_clean_text(record.get('id', ''), 120)}",
        "[CURRENT_PROMPT] " + current_prompt,
        feature_line,
        meta_text,
        hist_text,
    ])


def record_to_prompt_text(record: Dict[str, Any]) -> str:
    current_prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    return "\n".join([
        "[CURRENT_PROMPT] " + current_prompt,
        "[DERIVED] " + " ".join([
            f"history_len={len(history) if isinstance(history, list) else 0}",
            f"last_action={last_action}",
            keyword_tokens(current_prompt),
        ]),
    ])
