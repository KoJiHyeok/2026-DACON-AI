# -*- coding: utf-8 -*-
"""Shared helpers for first-step (history == []) diagnostics."""
from __future__ import annotations

import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support


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

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "train.jsonl"
DEFAULT_TRAIN_LABELS = DEFAULT_DATA_DIR / "train_labels.csv"
DEFAULT_HOLDOUT_BASE = Path(r"C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz")
DEFAULT_OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
DEFAULT_OUT_DIR = Path("night_out/task5")
EXPECTED_BLEND_F1 = 0.7172592174830689

STEP_RE = re.compile(r"-step_(\d+)$")


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


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


def is_hist0(sample: dict[str, Any]) -> bool:
    history = sample.get("history") or []
    return isinstance(history, list) and len(history) == 0


def load_submit_serializer():
    """Load submit/au_route.py serialize() without importing submit as a package."""
    module_path = REPO_ROOT / "submit" / "au_route.py"
    spec = importlib.util.spec_from_file_location("submit_au_route_for_firststep", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load serializer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    old_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write_bytecode
    return module.serialize


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_probs(
    probs: np.ndarray,
    src_classes: Sequence[str],
    dst_classes: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    out = np.full((probs.shape[0], len(dst_classes)), fill_value, dtype=np.float64)
    for dst_i, cls in enumerate(dst_classes):
        if cls in src:
            out[:, dst_i] = probs[:, src.index(cls)]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.ravel() <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(dst_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def predict_labels(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray(list(actions), dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1_from_pred(y_true: np.ndarray, y_pred: np.ndarray, actions: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(actions), average="macro", zero_division=0))


def macro_f1_from_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> float:
    return macro_f1_from_pred(y_true, predict_labels(probs, actions), actions)


def per_class_rows(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    actions: Sequence[str],
    prefix: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(actions),
        zero_division=0,
    )
    base = dict(prefix or {})
    rows = []
    for action, p, r, f, s in zip(actions, precision, recall, f1, support):
        row = dict(base)
        row.update(
            {
                "action": str(action),
                "support": int(s),
                "pred_count": int((y_pred == action).sum()),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "scarce_support_le_14": bool(s <= 14),
            }
        )
        rows.append(row)
    return rows


def load_league_components(
    holdout_base: Path = DEFAULT_HOLDOUT_BASE,
    oof_dir: Path = DEFAULT_OOF_DIR,
    expected_blend_f1: float = EXPECTED_BLEND_F1,
    tolerance: float = 5e-8,
) -> dict[str, Any]:
    enc_npz = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc_npz["ids"]], dtype=object)
    enc_probs = np.asarray(enc_npz["probs"], dtype=np.float64)
    y_true = np.asarray([str(x) for x in enc_npz["y_true"]], dtype=object)
    actions = [str(x) for x in enc_npz["actions"]]

    classes = json.loads((oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(action) for action in actions]
    row_index = {str(row_id): i for i, row_id in enumerate(row_ids)}
    rows = [row_index[sample_id] for sample_id in ids]

    linear = np.load(oof_dir / "linear_probs.npy")[:, col][rows]
    stacker = np.load(oof_dir / "stacker_probs.npy")[:, col][rows]
    blend = (linear + stacker + 2.0 * enc_probs) / 4.0
    score = macro_f1_from_probs(blend, y_true, actions)
    if abs(score - expected_blend_f1) > tolerance:
        raise AssertionError(
            f"3-way join check failed: got {score:.10f}, expected {expected_blend_f1:.10f}"
        )

    return {
        "ids": ids,
        "y_true": y_true,
        "actions": actions,
        "components": {
            "linear": linear,
            "stacker": stacker,
            "encoder_proxy": enc_probs,
            "blend": blend,
        },
        "blend_f1": score,
    }
