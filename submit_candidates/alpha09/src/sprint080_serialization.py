from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List


def _bucket(v: Any, edges: tuple[int, ...] = (1000, 10000, 50000, 100000)) -> str:
    if v is None:
        return "na"
    try:
        value = float(v)
    except (TypeError, ValueError):
        return "na"
    for idx, edge in enumerate(edges):
        if value < edge:
            return f"b{idx}"
    return f"b{len(edges)}"


def _action_sequence(record: Dict[str, Any]) -> List[str]:
    seq: List[str] = []
    history = record.get("history")
    if not isinstance(history, list):
        return seq
    for item in history:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            name = item.get("name") or item.get("action") or item.get("tool")
            if name:
                seq.append(str(name))
    return seq


def _clean(value: Any, max_chars: int = 1000) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + " ..." + text[-half:]


def session_group_id(record_id: Any) -> str:
    text = str(record_id)
    return text.rsplit("-step_", 1)[0] if "-step_" in text else text


def serialize_e5_compatible(record: Dict[str, Any], max_hist: int = 6) -> str:
    """Serialization contract used by the submitted e5s42gpuopt encoder line."""
    session_meta = record.get("session_meta") or {}
    workspace = session_meta.get("workspace") if isinstance(session_meta, dict) else {}
    if not isinstance(workspace, dict):
        workspace = {}

    parts: List[str] = []
    parts.append(
        f"tier={session_meta.get('user_tier', '?')} "
        f"lang={session_meta.get('language_pref', '?')} "
        f"turn={session_meta.get('turn_index', '?')} "
        f"ci={workspace.get('last_ci_status', '?')} "
        f"dirty={int(bool(workspace.get('git_dirty')))} "
        f"budget={_bucket(session_meta.get('budget_tokens_remaining'))} "
        f"loc={_bucket(workspace.get('loc'))}"
    )
    open_files = workspace.get("open_files") or []
    if isinstance(open_files, list) and open_files:
        parts.append("open: " + " ".join(str(x) for x in open_files[:5]))
    parts.append("query: " + str(record.get("current_prompt") or "")[:800])

    history = record.get("history") or []
    items: List[str] = []
    if isinstance(history, list):
        for item in reversed(history[-max_hist:]):
            if not isinstance(item, dict):
                continue
            if item.get("role") == "assistant_action":
                items.append(f"act:{item.get('name')} r:{str(item.get('result_summary') or '')[:120]}")
            else:
                items.append(f"user: {str(item.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    return "\n".join(parts)


def serialize_s1_prompt(record: Dict[str, Any]) -> str:
    return "[PROMPT] " + _clean(record.get("current_prompt"), 1600)


def serialize_s2_prompt_recent(record: Dict[str, Any], max_hist: int = 6) -> str:
    history = record.get("history") or []
    lines = [serialize_s1_prompt(record), "[HISTORY]"]
    if isinstance(history, list):
        for item in history[-max_hist:]:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "assistant_action":
                lines.append(
                    "assistant_action "
                    + _clean(item.get("name"), 80)
                    + " "
                    + _clean(item.get("result_summary"), 300)
                )
            else:
                lines.append("user " + _clean(item.get("content"), 400))
    return "\n".join(lines)


def serialize_s3_compressed_full(record: Dict[str, Any], max_hist: int = 12) -> str:
    session_meta = record.get("session_meta") or {}
    workspace = session_meta.get("workspace") if isinstance(session_meta, dict) else {}
    history = record.get("history") or []
    lines = [serialize_s2_prompt_recent(record, max_hist=max_hist)]
    lines.append("[SESSION_META] " + _clean(session_meta, 900))
    lines.append("[WORKSPACE] " + _clean(workspace, 900))
    lines.append("[ACTION_SEQUENCE] " + " > ".join(_action_sequence(record)[-16:]))
    return "\n".join(lines)


def serialize_s4_tagged_policy(record: Dict[str, Any]) -> str:
    session_meta = record.get("session_meta") or {}
    workspace = session_meta.get("workspace") if isinstance(session_meta, dict) else {}
    seq = _action_sequence(record)
    last = seq[-1] if seq else "none"
    prev = seq[-2] if len(seq) >= 2 else "none"
    return "\n".join(
        [
            "[PROMPT] " + _clean(record.get("current_prompt"), 1600),
            "[LAST_ACTION] " + last,
            "[PREV_ACTION] " + prev,
            "[ACTION_SEQUENCE] " + " > ".join(seq[-16:]),
            "[SESSION] "
            + " ".join(
                [
                    f"tier={session_meta.get('user_tier', 'na')}",
                    f"lang={session_meta.get('language_pref', 'na')}",
                    f"turn={session_meta.get('turn_index', 'na')}",
                    f"budget={_bucket(session_meta.get('budget_tokens_remaining'))}",
                ]
            ),
            "[WORKSPACE] "
            + " ".join(
                [
                    f"ci={workspace.get('last_ci_status', 'na') if isinstance(workspace, dict) else 'na'}",
                    f"dirty={int(bool(workspace.get('git_dirty'))) if isinstance(workspace, dict) else 0}",
                    f"loc={_bucket(workspace.get('loc')) if isinstance(workspace, dict) else 'na'}",
                ]
            ),
        ]
    )


def serialize_xrb_tagged(record: Dict[str, Any], max_hist: int = 12) -> str:
    """Tagged multilingual view for XLM-R style encoders.

    The tags intentionally separate the user intent, previous tool policy,
    and environment state so a non-E5 encoder can learn a different boundary
    than the compact e5-compatible line.
    """
    session_meta = record.get("session_meta") or {}
    workspace = session_meta.get("workspace") if isinstance(session_meta, dict) else {}
    if not isinstance(workspace, dict):
        workspace = {}
    seq = _action_sequence(record)
    last = seq[-1] if seq else "none"

    history = record.get("history") or []
    history_lines: List[str] = []
    if isinstance(history, list):
        for item in history[-max_hist:]:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "assistant_action":
                history_lines.append(
                    "[ASSISTANT_ACTION] "
                    + _clean(item.get("name"), 80)
                    + " [ARGS] "
                    + _clean(item.get("args"), 350)
                    + " [RESULT] "
                    + _clean(item.get("result_summary"), 350)
                )
            else:
                history_lines.append("[USER] " + _clean(item.get("content"), 600))

    meta_bits = [
        f"user_tier={session_meta.get('user_tier', 'na')}",
        f"language_pref={session_meta.get('language_pref', 'na')}",
        f"budget_bucket={_bucket(session_meta.get('budget_tokens_remaining'))}",
        f"turn_index={session_meta.get('turn_index', 'na')}",
        f"elapsed_bucket={_bucket(session_meta.get('elapsed_session_sec'), (60, 300, 900, 1800, 3600))}",
        f"last_ci_status={workspace.get('last_ci_status', 'na')}",
        f"git_dirty={int(bool(workspace.get('git_dirty')))}",
        f"loc_bucket={_bucket(workspace.get('loc'))}",
    ]
    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict):
        mix = " ".join(f"{k}:{round(float(v), 3)}" for k, v in sorted(language_mix.items()) if isinstance(v, (int, float)))
        if mix:
            meta_bits.append(f"language_mix={mix}")
    open_files = workspace.get("open_files") or []
    if isinstance(open_files, list) and open_files:
        meta_bits.append("open_files=" + " ".join(_clean(x, 120) for x in open_files[:8]))

    return "\n".join(
        [
            "[CURRENT_PROMPT] " + _clean(record.get("current_prompt"), 1800),
            "[HISTORY] " + " \n".join(history_lines),
            "[SESSION_META] " + " ".join(meta_bits),
            "[LAST_ACTION] " + last,
            "[ACTION_SEQUENCE] " + " > ".join(seq[-20:]),
        ]
    )


SERIALIZERS = {
    "e5_compatible": serialize_e5_compatible,
    "S1": serialize_s1_prompt,
    "S2": serialize_s2_prompt_recent,
    "S3": serialize_s3_compressed_full,
    "S4": serialize_s4_tagged_policy,
    "current_recent_history": serialize_s2_prompt_recent,
    "xrb_tagged": serialize_xrb_tagged,
}


def serialize_record(record: Dict[str, Any], serialization: str) -> str:
    try:
        fn = SERIALIZERS[serialization]
    except KeyError as exc:
        raise KeyError(f"Unknown sprint_080 serialization: {serialization}") from exc
    return fn(record)


def serialization_hash(records: Iterable[Dict[str, Any]], serialization: str) -> str:
    h = hashlib.sha256()
    h.update(serialization.encode("utf-8"))
    for record in records:
        h.update(b"\n---record---\n")
        h.update(serialize_record(record, serialization).encode("utf-8", errors="replace"))
    return h.hexdigest()
