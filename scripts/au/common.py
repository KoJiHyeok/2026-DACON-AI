# -*- coding: utf-8 -*-
"""Shared helpers for AU/SIM diagnostics."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import f1_score


ACTION_CLASSES = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

DEFAULT_TRAIN_JSONL = Path(r"C:\dev\2026-AI-DACON\data\train.jsonl")
DEFAULT_TRAIN_LABELS = Path(r"C:\dev\2026-AI-DACON\data\train_labels.csv")
DEFAULT_HOLDOUT_BASE = Path(r"C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz")
DEFAULT_OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
DEFAULT_OUT_DIR = Path("context/night/2026-07-06")

STEP_RE = re.compile(r"-step_(\d+)$")
HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def step_num(sample_id: str) -> int:
    match = STEP_RE.search(str(sample_id))
    return int(match.group(1)) if match else -1


def bucket_from_id(sample_id: str) -> str:
    return "au" if str(sample_id).startswith("sess_au") else "sim"


def read_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {row["id"]: row["action"] for row in csv.DictReader(f)}


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_train_records(train_jsonl: Path, labels_csv: Path) -> list[dict[str, Any]]:
    labels = read_labels(labels_csv)
    records: list[dict[str, Any]] = []
    for obj in iter_jsonl(train_jsonl):
        rid = str(obj["id"])
        obj["action"] = labels.get(rid)
        records.append(obj)
    return records


def dominant_lang(workspace: dict[str, Any]) -> str:
    mix = workspace.get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return str(max(mix.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)[0])


def _num_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def prompt_lang_guess(text: str) -> str:
    if not text:
        return "empty"
    n = len(text)
    hangul = len(HANGUL_RE.findall(text))
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    if hangul / max(n, 1) >= 0.05:
        return "ko"
    if non_ascii / max(n, 1) >= 0.05:
        return "non_ascii"
    return "ascii"


def flatten_record(obj: dict[str, Any]) -> dict[str, Any]:
    rid = str(obj["id"])
    meta = obj.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = obj.get("history") or []
    if not isinstance(history, list):
        history = []

    action_turns = [h for h in history if isinstance(h, dict) and h.get("role") == "assistant_action"]
    user_turns = [h for h in history if isinstance(h, dict) and h.get("role") == "user"]
    prompt = str(obj.get("current_prompt") or "")
    prompt_chars = len(prompt)
    hangul_chars = len(HANGUL_RE.findall(prompt))
    non_ascii_chars = sum(1 for ch in prompt if ord(ch) > 127)
    open_files = workspace.get("open_files") or []
    if not isinstance(open_files, list):
        open_files = []

    return {
        "id": rid,
        "bucket": bucket_from_id(rid),
        "session": session_id(rid),
        "step": step_num(rid),
        "action": obj.get("action"),
        "history_len": len(history),
        "n_action_turns": len(action_turns),
        "n_user_turns": len(user_turns),
        "last_action": str(action_turns[-1].get("name")) if action_turns else "none",
        "current_prompt_len": prompt_chars,
        "current_prompt_words": len(prompt.split()),
        "current_prompt_hangul_frac": hangul_chars / max(prompt_chars, 1),
        "current_prompt_non_ascii_frac": non_ascii_chars / max(prompt_chars, 1),
        "prompt_lang_guess": prompt_lang_guess(prompt),
        "user_tier": str(meta.get("user_tier") or "none"),
        "language_pref": str(meta.get("language_pref") or "none"),
        "budget_tokens_remaining": _num_or_nan(meta.get("budget_tokens_remaining")),
        "turn_index": _num_or_nan(meta.get("turn_index")),
        "elapsed_session_sec": _num_or_nan(meta.get("elapsed_session_sec")),
        "workspace_loc": _num_or_nan(workspace.get("loc")),
        "git_dirty": bool(workspace.get("git_dirty")),
        "last_ci_status": str(workspace.get("last_ci_status") or "none"),
        "n_open_files": len(open_files),
        "top_lang": dominant_lang(workspace),
        "n_langs": len(workspace.get("language_mix") or {}),
    }


def load_league_components(
    holdout_base: Path = DEFAULT_HOLDOUT_BASE,
    oof_dir: Path = DEFAULT_OOF_DIR,
    expected_blend_f1: float = 0.71726,
    tolerance: float = 5e-5,
) -> dict[str, Any]:
    enc_npz = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc_npz["ids"]])
    enc_probs = np.asarray(enc_npz["probs"], dtype=np.float64)
    y_true = np.asarray([str(x) for x in enc_npz["y_true"]])
    actions = [str(x) for x in enc_npz["actions"]]

    classes = json.loads((oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(action) for action in actions]
    row_index = {str(row_id): i for i, row_id in enumerate(row_ids)}
    rows = [row_index[sample_id] for sample_id in ids]

    linear = np.load(oof_dir / "linear_probs.npy")[:, col][rows]
    stacker = np.load(oof_dir / "stacker_probs.npy")[:, col][rows]
    blend = (linear + stacker + 2.0 * enc_probs) / 4.0
    score = macro_f1(blend, y_true, actions)
    if abs(score - expected_blend_f1) > tolerance:
        raise AssertionError(
            f"3-way join check failed: got {score:.8f}, expected {expected_blend_f1:.5f}"
        )

    return {
        "ids": ids,
        "y_true": y_true,
        "actions": actions,
        "components": {
            "linear": linear,
            "stacker": stacker,
            "encoder": enc_probs,
            "blend": blend,
        },
        "blend_f1": score,
    }


def predict_labels(probs: np.ndarray, actions: list[str]) -> np.ndarray:
    labels = np.asarray(actions)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1(probs: np.ndarray, y_true: np.ndarray, actions: list[str]) -> float:
    return float(f1_score(y_true, predict_labels(probs, actions), average="macro", zero_division=0))


def truncate(text: Any, limit: int = 500) -> str:
    value = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."
