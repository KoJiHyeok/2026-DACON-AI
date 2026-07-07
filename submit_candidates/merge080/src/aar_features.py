from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterable, List, Tuple


def clean_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + " ... " + text[-half:]
    return text


def flatten_meta(obj: Any, prefix: str = "") -> List[Tuple[str, Any]]:
    items: List[Tuple[str, Any]] = []
    if isinstance(obj, dict):
        for key, value in sorted(obj.items(), key=lambda kv: str(kv[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                items.extend(flatten_meta(value, name))
            elif isinstance(value, list):
                items.append((name, "|".join(clean_text(x, 80) for x in value[:10])))
                items.append((name + ".len", len(value)))
            else:
                items.append((name, value))
    return items


def action_sequence(record: Dict[str, Any]) -> List[str]:
    history = record.get("history")
    if not isinstance(history, list):
        return []
    out: List[str] = []
    for item in history:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            name = item.get("name") or item.get("action") or item.get("tool")
            if name:
                out.append(str(name))
    return out


def last_user_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            return clean_text(item.get("content", ""), 1000)
    return ""


def prompt_text(record: Dict[str, Any]) -> str:
    return clean_text(record.get("current_prompt", ""), 2500)


def _bucket_number(value: Any, bounds: Iterable[float]) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    for bound in bounds:
        if x <= bound:
            return f"le_{int(bound)}"
    return f"gt_{int(list(bounds)[-1])}"


def _workspace(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        return {}
    workspace = meta.get("workspace")
    return workspace if isinstance(workspace, dict) else {}


def keyword_flags(text: str) -> Dict[str, int]:
    lower = text.lower()
    groups = {
        "read_file": ("read", "open", "show", "cat ", "view", "inspect file", "file content"),
        "grep_search": ("grep", "rg ", "search", "find", "reference", "defined", "where is"),
        "list_directory": ("ls", "tree", "folder", "directory", "list files", "structure"),
        "glob_pattern": ("glob", "*.py", "*.js", "*.ts", "*.json", "all files", "pattern"),
        "edit_file": ("edit", "fix", "change", "update", "modify", "replace", "refactor"),
        "write_file": ("write file", "create file", "new file", "save as", "generate file"),
        "apply_patch": ("patch", "diff", "apply_patch", "apply patch"),
        "run_bash": ("run ", "execute", "bash", "shell", "terminal", "command", "npm ", "pip ", "python "),
        "run_tests": ("test", "pytest", "unittest", "coverage", "spec", "happy path"),
        "lint_or_typecheck": ("lint", "typecheck", "type check", "mypy", "ruff", "eslint", "tsc "),
        "ask_user": ("?", "which", "choose", "clarify", "confirm", "what do you", "should i"),
        "plan_task": ("plan", "roadmap", "approach", "strategy", "steps", "architecture"),
        "web_search": ("web", "internet", "latest", "today", "news", "lookup", "browse", "search online"),
        "respond_only": ("explain", "tell me", "answer", "summarize", "describe", "why"),
    }
    return {name: int(any(token in lower for token in tokens)) for name, tokens in groups.items()}


def metadata_features(record: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    workspace = _workspace(record)
    history = record.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    seq = action_sequence(record)
    last_action = seq[-1] if seq else "none"
    prompt = prompt_text(record)

    for name in ("user_tier", "language_pref"):
        features[f"{name}={clean_text(meta.get(name), 40)}"] = 1.0
    for name in ("git_dirty", "last_ci_status"):
        features[f"workspace.{name}={clean_text(workspace.get(name), 40)}"] = 1.0

    budget = meta.get("budget_tokens_remaining")
    turn = meta.get("turn_index")
    elapsed = meta.get("elapsed_session_sec")
    loc = workspace.get("loc")
    open_files = workspace.get("open_files")
    open_count = len(open_files) if isinstance(open_files, list) else 0

    features[f"budget_bin={_bucket_number(budget, [256, 512, 1024, 2048, 4096, 8192, 16384])}"] = 1.0
    features[f"turn_bin={_bucket_number(turn, [0, 1, 2, 4, 8, 16, 32])}"] = 1.0
    features[f"elapsed_bin={_bucket_number(elapsed, [30, 60, 120, 300, 600, 1200, 2400])}"] = 1.0
    features[f"loc_bin={_bucket_number(loc, [100, 1000, 5000, 20000, 100000])}"] = 1.0
    features[f"history_len={history_len}"] = 1.0
    features[f"action_count={len(seq)}"] = 1.0
    features[f"open_count={open_count}"] = 1.0
    features[f"last_action={last_action}"] = 1.0
    if len(seq) >= 2:
        features[f"last2={seq[-2]}>{seq[-1]}"] = 1.0

    for key, value in (workspace.get("language_mix") or {}).items():
        try:
            features[f"langmix={key}"] = float(value)
        except (TypeError, ValueError):
            continue

    for path in open_files[:8] if isinstance(open_files, list) else []:
        suffix = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "none"
        features[f"open_ext={suffix}"] = 1.0

    features["num_budget_log"] = math.log1p(float(budget or 0.0)) / 12.0
    features["num_turn_log"] = math.log1p(float(turn or 0.0)) / 4.0
    features["num_elapsed_log"] = math.log1p(float(elapsed or 0.0)) / 8.0
    features["num_loc_log"] = math.log1p(float(loc or 0.0)) / 12.0
    features["num_prompt_len"] = min(len(prompt), 2000) / 2000.0
    features["num_prompt_qmark"] = float("?" in prompt)
    features["num_history_len"] = history_len / 12.0
    return features


def rule_features(record: Dict[str, Any]) -> Dict[str, float]:
    prompt = prompt_text(record)
    last_user = last_user_text(record)
    seq = action_sequence(record)
    joined = prompt + "\n" + last_user
    flags = keyword_flags(joined)
    features: Dict[str, float] = {f"rule_{key}": float(value) for key, value in flags.items()}
    lower = joined.lower()
    features["has_code_fence"] = float("```" in joined)
    features["has_file_path"] = float(bool(re.search(r"[\w./\\-]+\.(py|js|ts|tsx|json|csv|md|yml|yaml|txt)", lower)))
    features["has_shell_op"] = float(any(x in lower for x in ("&&", "||", "npm ", "pip ", "pytest", "python ")))
    features["has_latest_word"] = float(any(x in lower for x in ("latest", "today", "current", "recent")))
    features["starts_question"] = float(lower.strip().startswith(("what", "why", "how", "which", "can ", "should ")))
    if seq:
        features[f"after_{seq[-1]}"] = 1.0
    return features


def record_to_history_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    if not isinstance(history, list) or not history:
        return "[HISTORY] empty\n[CURRENT_PROMPT] " + prompt_text(record)
    lines = ["[CURRENT_PROMPT] " + prompt_text(record), "[HISTORY_FOCUSED]"]
    for idx, item in enumerate(history[-10:]):
        if not isinstance(item, dict):
            lines.append(f"turn_{idx} raw={clean_text(item, 400)}")
            continue
        role = clean_text(item.get("role", "unknown"), 40)
        if role == "assistant_action":
            name = clean_text(item.get("name") or item.get("action") or item.get("tool") or "unknown", 80)
            result = clean_text(item.get("result_summary", item.get("result", "")), 500)
            lines.append(f"assistant_action={name} result={result}")
        else:
            lines.append(f"{role}={clean_text(item.get('content', ''), 700)}")
    return "\n".join(lines)


def record_to_action_text(record: Dict[str, Any]) -> str:
    seq = action_sequence(record)
    history = record.get("history")
    parts = [f"hist_len={len(history) if isinstance(history, list) else 0}"]
    if not seq:
        parts.append("last_action=none")
    for action in seq[-12:]:
        parts.append(f"act_{action}")
    if seq:
        parts.append(f"last_action={seq[-1]}")
    for left, right in zip(seq[-12:], seq[-11:]):
        parts.append(f"pair_{left}>{right}")
    parts.extend(f"kw_{name}" for name, value in keyword_flags(prompt_text(record)).items() if value)
    return " ".join(parts)


def record_to_meta_text(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    parts = []
    for key, value in flatten_meta(meta):
        if isinstance(value, (int, float)):
            parts.append(f"{key}_bin={_bucket_number(value, [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768])}")
        else:
            parts.append(f"{key}={clean_text(value, 120)}")
    seq = action_sequence(record)
    if seq:
        parts.append("last_action=" + seq[-1])
    return "[META] " + " ; ".join(parts)


def record_to_prompt_context_text(record: Dict[str, Any]) -> str:
    seq = action_sequence(record)
    return "\n".join([
        "[CURRENT_PROMPT] " + prompt_text(record),
        "[LAST_USER] " + last_user_text(record),
        "[ACTIONS] " + " > ".join(seq[-8:]) if seq else "[ACTIONS] none",
        record_to_meta_text(record),
        record_to_action_text(record),
    ])


def transition_keys(record: Dict[str, Any]) -> Dict[str, str]:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    workspace = _workspace(record)
    history = record.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    seq = action_sequence(record)
    last_action = seq[-1] if seq else "none"
    flags = keyword_flags(prompt_text(record))
    active_flags = [name for name, value in flags.items() if value]
    first_flag = active_flags[0] if active_flags else "none"
    keys = {
        "last_action": last_action,
        "history_len": str(history_len),
        "language_pref": clean_text(meta.get("language_pref", "none"), 40),
        "ci_dirty": f"{workspace.get('last_ci_status', 'none')}|{workspace.get('git_dirty', 'none')}",
        "prompt_rule": first_flag,
        "last_action_rule": f"{last_action}|{first_flag}",
    }
    if len(seq) >= 2:
        keys["last2"] = f"{seq[-2]}>{seq[-1]}"
    else:
        keys["last2"] = "none"
    return keys
