from __future__ import annotations

import re
from typing import Any, Dict, List

from .aar_features import (
    action_sequence,
    clean_text,
    keyword_flags,
    last_user_text,
    metadata_features,
    prompt_text,
    record_to_action_text,
    record_to_history_text,
    record_to_meta_text,
    record_to_prompt_context_text,
    rule_features,
)
from .serialize import record_to_prompt_text, record_to_text
from .template_signature import intent_template_signature


def intent_signature(record: Dict[str, Any]) -> str:
    return intent_template_signature(prompt_text(record))


def session_group_id(record_id: Any) -> str:
    text = str(record_id)
    return text.rsplit("-step_", 1)[0] if "-step_" in text else text


def step_id(record_id: Any) -> str:
    text = str(record_id)
    return text.rsplit("-step_", 1)[1] if "-step_" in text else "NA"


def workspace_summary(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta")
    workspace = meta.get("workspace") if isinstance(meta, dict) else None
    if not isinstance(workspace, dict):
        workspace = {}
    open_files = workspace.get("open_files")
    if isinstance(open_files, list):
        suffixes = []
        for path in open_files[:12]:
            value = str(path)
            suffixes.append(value.rsplit(".", 1)[-1].lower() if "." in value else "none")
    else:
        suffixes = []
    language_mix = workspace.get("language_mix")
    lang = []
    if isinstance(language_mix, dict):
        lang = [f"{k}:{round(float(v), 2)}" for k, v in sorted(language_mix.items()) if isinstance(v, (int, float))]
    return " ".join([
        f"loc={workspace.get('loc', 'NA')}",
        f"dirty={workspace.get('git_dirty', 'NA')}",
        f"ci={workspace.get('last_ci_status', 'NA')}",
        "open_ext=" + "|".join(suffixes),
        "lang=" + "|".join(lang),
    ])


def last_user_assistant_pair(record: Dict[str, Any]) -> str:
    history = record.get("history")
    if not isinstance(history, list):
        return "[PAIR] empty"
    last_user = ""
    last_action = ""
    last_result = ""
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "user":
            last_user = clean_text(item.get("content", ""), 900)
        elif item.get("role") == "assistant_action":
            last_action = clean_text(item.get("name") or item.get("action") or item.get("tool") or "", 80)
            last_result = clean_text(item.get("result_summary", item.get("result", "")), 500)
    return "\n".join([
        "[LAST_USER] " + last_user,
        "[LAST_ASSISTANT_ACTION] " + last_action,
        "[LAST_RESULT] " + last_result,
        "[CURRENT_PROMPT] " + prompt_text(record),
    ])


def compressed_state(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    history = record.get("history")
    seq = action_sequence(record)
    prompt = prompt_text(record)
    flags = [name for name, value in keyword_flags(prompt).items() if value]
    return " ".join([
        f"step={step_id(record.get('id', ''))}",
        f"turn={meta.get('turn_index', 'NA')}",
        f"tier={meta.get('user_tier', 'NA')}",
        f"lang_pref={meta.get('language_pref', 'NA')}",
        f"hist_len={len(history) if isinstance(history, list) else 0}",
        f"user_turns={sum(1 for x in history if isinstance(x, dict) and x.get('role') == 'user') if isinstance(history, list) else 0}",
        f"assistant_actions={len(seq)}",
        f"last={seq[-1] if seq else 'none'}",
        f"last2={'>'.join(seq[-2:]) if len(seq) >= 2 else 'none'}",
        "rules=" + "|".join(flags),
        "workspace=" + workspace_summary(record),
    ])


def prompt_last_action(record: Dict[str, Any]) -> str:
    seq = action_sequence(record)
    return "\n".join([
        "[CURRENT_PROMPT] " + prompt_text(record),
        f"[STEP] {step_id(record.get('id', ''))}",
        f"[LAST_ACTION] {seq[-1] if seq else 'none'}",
        f"[ACTION_SEQUENCE] {' > '.join(seq[-10:]) if seq else 'none'}",
        "[RULE_HINTS] " + " ".join(name for name, value in keyword_flags(prompt_text(record)).items() if value),
    ])


def prompt_workspace(record: Dict[str, Any]) -> str:
    return "\n".join([
        "[CURRENT_PROMPT] " + prompt_text(record),
        "[WORKSPACE] " + workspace_summary(record),
        "[META] " + record_to_meta_text(record),
    ])


def make_views(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "full": record_to_text(record),
        "prompt": record_to_prompt_text(record),
        "history_recent": record_to_history_text(record),
        "action_sequence": record_to_action_text(record),
        "metadata_text": record_to_meta_text(record),
        "prompt_last_action": prompt_last_action(record),
        "prompt_workspace": prompt_workspace(record),
        "last_user_assistant_pair": last_user_assistant_pair(record),
        "compressed_state": compressed_state(record),
        "intent_signature": intent_signature(record),
        "prompt_context": record_to_prompt_context_text(record),
        "meta_dict": metadata_features(record),
        "rule_dict": rule_features(record),
    }


def build_view_matrix(records: List[Dict[str, Any]], view: str) -> List[Any]:
    if view == "full":
        return [record_to_text(r) for r in records]
    if view == "prompt":
        return [record_to_prompt_text(r) for r in records]
    if view == "history_recent":
        return [record_to_history_text(r) for r in records]
    if view == "action_sequence":
        return [record_to_action_text(r) for r in records]
    if view == "metadata_text":
        return [record_to_meta_text(r) for r in records]
    if view == "prompt_last_action":
        return [prompt_last_action(r) for r in records]
    if view == "prompt_workspace":
        return [prompt_workspace(r) for r in records]
    if view == "last_user_assistant_pair":
        return [last_user_assistant_pair(r) for r in records]
    if view == "compressed_state":
        return [compressed_state(r) for r in records]
    if view == "intent_signature":
        return [intent_signature(r) for r in records]
    if view == "prompt_context":
        return [record_to_prompt_context_text(r) for r in records]
    if view == "meta_dict":
        return [metadata_features(r) for r in records]
    if view == "rule_dict":
        return [rule_features(r) for r in records]
    raise KeyError(f"Unknown view: {view}")


def matched_rule_names(record: Dict[str, Any]) -> List[str]:
    flags = rule_features(record)
    names = []
    for key, value in flags.items():
        if key.startswith("rule_") and value:
            names.append(re.sub(r"^rule_", "", key))
    return names
