from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .constants import ACTIONS
from .sprint080_oof import normalize_proba


def is_au(sample_id: Any) -> bool:
    return str(sample_id).startswith("sess_au")


def compact_json(value: Any, limit: int = 240) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def serialize(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta") if isinstance(record.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    parts = [
        "id_prefix=sess_au" if is_au(record.get("id", "")) else "id_prefix=sess_sim",
        f"user_tier={meta.get('user_tier', '')}",
        f"language_pref={meta.get('language_pref', '')}",
        f"turn_index={meta.get('turn_index', '')}",
        f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}",
        f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}",
        f"git_dirty={workspace.get('git_dirty', '')}",
        f"last_ci_status={workspace.get('last_ci_status', '')}",
        f"loc={workspace.get('loc', '')}",
        "language_mix=" + compact_json(workspace.get("language_mix") or {}),
        "open_files=" + " ".join(str(x) for x in (workspace.get("open_files") or [])),
        "current_prompt=" + str(record.get("current_prompt") or ""),
    ]
    for item in record.get("history") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", ""))
        if role == "assistant_action":
            parts.append(
                "history_action="
                + str(item.get("name", ""))
                + " args="
                + compact_json(item.get("args") or {}, 180)
                + " result="
                + str(item.get("result_summary", ""))
            )
        else:
            parts.append(f"history_{role}=" + str(item.get("content", "")))
    return "\n".join(parts)


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z -= z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return normalize_proba(exp)


def align_proba(raw: np.ndarray, classes: Sequence[Any]) -> np.ndarray:
    out = np.zeros((raw.shape[0], len(ACTIONS)), dtype=np.float64)
    dst = {a: i for i, a in enumerate(ACTIONS)}
    for j, cls in enumerate(classes):
        i = None
        if isinstance(cls, (int, np.integer)) and 0 <= int(cls) < len(ACTIONS):
            i = int(cls)
        else:
            text = str(cls)
            if text.isdigit() and 0 <= int(text) < len(ACTIONS):
                i = int(text)
            else:
                i = dst.get(text)
        if i is not None and j < raw.shape[1]:
            out[:, i] = raw[:, j]
    return normalize_proba(out)


def predict_proba(artifact: Dict[str, Any], records: Iterable[Dict[str, Any]]) -> Tuple[np.ndarray, List[str]]:
    texts = [serialize(record) for record in records]
    x = artifact["union"].transform(texts)
    clf = artifact["clf"]
    if hasattr(clf, "decision_function"):
        raw = softmax(np.asarray(clf.decision_function(x), dtype=np.float64))
    elif hasattr(clf, "predict_proba"):
        raw = np.asarray(clf.predict_proba(x), dtype=np.float64)
    else:
        pred = clf.predict(x)
        classes = [str(c) for c in clf.classes_]
        raw = np.zeros((len(pred), len(classes)), dtype=np.float64)
        idx = {c: i for i, c in enumerate(classes)}
        for r, label in enumerate(pred):
            raw[r, idx[str(label)]] = 1.0
    classes = [str(c) for c in clf.classes_]
    return align_proba(raw, classes), classes
