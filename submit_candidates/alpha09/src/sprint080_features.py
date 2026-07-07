from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List


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
COMMAND_RE = re.compile(r"\b(?:python|pip|pytest|npm|pnpm|yarn|git|ruff|mypy|eslint|bash|sh)\b", re.I)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _history(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    history = record.get("history")
    return history if isinstance(history, list) else []


def action_sequence(record: Dict[str, Any]) -> List[str]:
    seq: List[str] = []
    for item in _history(record):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            name = item.get("name") or item.get("action") or item.get("tool")
            if name:
                seq.append(str(name))
    return seq


def _count_user_turns(history: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for item in history if isinstance(item, dict) and item.get("role") == "user")


def _workspace(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("session_meta")
    workspace = meta.get("workspace") if isinstance(meta, dict) else {}
    return workspace if isinstance(workspace, dict) else {}


def _flatten_action_payloads(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for item in _history(record):
        if not isinstance(item, dict) or item.get("role") != "assistant_action":
            continue
        parts.append(_text(item.get("args")))
        parts.append(_text(item.get("result_summary")))
    return " ".join(parts)


def structured_features(record: Dict[str, Any]) -> Dict[str, float]:
    prompt = _text(record.get("current_prompt"))
    prompt_lower = prompt.lower()
    hist = _history(record)
    seq = action_sequence(record)
    workspace = _workspace(record)
    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    payload = _flatten_action_payloads(record)
    payload_lower = payload.lower()

    feats: Dict[str, float] = {
        "history_length": float(len(hist)),
        "user_turn_count": float(_count_user_turns(hist)),
        "assistant_action_count": float(len(seq)),
        "open_files_count": float(len(open_files)),
        "file_extension_count": float(len(EXT_RE.findall(prompt + " " + " ".join(map(str, open_files))))),
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
        "result_summary_has_error": float(any(w in payload_lower for w in ["error", "failed", "traceback", "exception", "실패", "에러"])),
    }

    all_keyword_keys = sorted(set(KOREAN_KEYWORDS) | set(ENGLISH_KEYWORDS))
    for key in all_keyword_keys:
        words = KOREAN_KEYWORDS.get(key, []) + ENGLISH_KEYWORDS.get(key, [])
        feats[key] = float(any(word.lower() in prompt_lower for word in words))

    last_action = seq[-1] if seq else "none"
    prev_action = seq[-2] if len(seq) >= 2 else "none"
    feats[f"last_action={last_action}"] = 1.0
    feats[f"prev_action={prev_action}"] = 1.0
    for a, b in zip(seq[-6:], seq[-5:]):
        feats[f"action_bigram={a}>{b}"] = 1.0
    for a, b, c in zip(seq[-6:], seq[-5:], seq[-4:]):
        feats[f"action_trigram={a}>{b}>{c}"] = 1.0

    last_ci = workspace.get("last_ci_status", "none")
    feats[f"last_ci_status={last_ci}"] = 1.0
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict):
        for lang, value in language_mix.items():
            if isinstance(value, (int, float)):
                feats[f"language_mix={lang}"] = float(value)
    return feats


def feature_dicts(records: Iterable[Dict[str, Any]]) -> List[Dict[str, float]]:
    return [structured_features(record) for record in records]
