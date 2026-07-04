"""
Shared data loading for simulator forensics round 1.

Loads data/train.jsonl + data/train_labels.csv, flattens each row into a
DataFrame with derived state columns (session key, step number, last/last2
action + args, history length, etc). Caches the flattened frame to a pickle
next to the source data so repeated analysis scripts don't re-parse the 98MB
jsonl file every time.

Usage:
    from common import load_frame
    df = load_frame()
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
CACHE_PATH = DATA_DIR / "_forensics_cache.pkl"

STEP_RE = re.compile(r"-step_(\d+)$")


def session_key(row_id: str) -> str:
    return STEP_RE.sub("", row_id)


def step_num(row_id: str) -> int:
    m = STEP_RE.search(row_id)
    return int(m.group(1)) if m else -1


def _args_sig(args: dict | None) -> str:
    """Compact signature of an action's args dict (keys only, sorted)."""
    if not args:
        return ""
    return ",".join(sorted(args.keys()))


def _row_features(obj: dict[str, Any]) -> dict[str, Any]:
    rid = obj["id"]
    meta = obj.get("session_meta", {}) or {}
    ws = meta.get("workspace", {}) or {}
    history = obj.get("history", []) or []

    # Walk history to find last / last2 assistant_action turns and last user turn.
    action_turns = [h for h in history if h.get("role") == "assistant_action"]
    user_turns = [h for h in history if h.get("role") == "user"]

    last_action = action_turns[-1]["name"] if action_turns else "none"
    last2_action = action_turns[-2]["name"] if len(action_turns) >= 2 else "none"
    last3_action = action_turns[-3]["name"] if len(action_turns) >= 3 else "none"
    last_args = action_turns[-1].get("args") if action_turns else None
    last_args_sig = _args_sig(last_args)
    last_result_summary = action_turns[-1].get("result_summary", "") if action_turns else ""
    last_user_msg = user_turns[-1].get("content", "") if user_turns else ""

    lang_mix = ws.get("language_mix", {}) or {}
    top_lang = max(lang_mix, key=lang_mix.get) if lang_mix else "none"

    return {
        "id": rid,
        "session_key": session_key(rid),
        "step": step_num(rid),
        "current_prompt": obj.get("current_prompt", ""),
        "history_len": len(history),
        "n_action_turns": len(action_turns),
        "n_user_turns": len(user_turns),
        "last_action": last_action,
        "last2_action": last2_action,
        "last3_action": last3_action,
        "last_args_sig": last_args_sig,
        "last_args": json.dumps(last_args, ensure_ascii=False) if last_args else "",
        "last_result_summary": last_result_summary,
        "last_user_msg": last_user_msg,
        "user_tier": meta.get("user_tier", "none"),
        "language_pref": meta.get("language_pref", "none"),
        "budget_tokens_remaining": meta.get("budget_tokens_remaining"),
        "turn_index": meta.get("turn_index"),
        "elapsed_session_sec": meta.get("elapsed_session_sec"),
        "loc": ws.get("loc"),
        "git_dirty": ws.get("git_dirty"),
        "last_ci_status": ws.get("last_ci_status", "none"),
        "open_files": ws.get("open_files", []) or [],
        "n_open_files": len(ws.get("open_files", []) or []),
        "top_lang": top_lang,
        "n_langs": len(lang_mix),
    }


def build_frame() -> pd.DataFrame:
    rows = []
    with TRAIN_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append(_row_features(obj))
    df = pd.DataFrame(rows)
    labels = pd.read_csv(TRAIN_LABELS)
    df = df.merge(labels, on="id", how="left")
    return df


def load_frame(force_rebuild: bool = False) -> pd.DataFrame:
    if CACHE_PATH.exists() and not force_rebuild:
        return pd.read_pickle(CACHE_PATH)
    df = build_frame()
    df.to_pickle(CACHE_PATH)
    return df


if __name__ == "__main__":
    df = load_frame(force_rebuild=True)
    print(df.shape)
    print(df.dtypes)
    print(df.head())
