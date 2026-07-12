# -*- coding: utf-8 -*-
"""Shared helpers for the OOF5-GEN lane (D-014 Lane A phase 1).

Generates session-group 5-fold OOF probability matrices for linear, AU
(char_wb C1 soft-routing specialist), and AAR stacker components on the
EXACT same fold assignment as artifacts/experiments/oof_h12/fold_map.csv,
so a 5-component parity stacker experiment becomes possible.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score

ROOT = Path(r"C:\dev\2026-AI-DACON")
DATA_DIR = ROOT / "data"
REF_OOF_DIR = ROOT / "artifacts" / "experiments" / "oof_h12"
FOLD_MAP_PATH = REF_OOF_DIR / "fold_map.csv"
FOLD_MAP_SHA256 = "56074c16c400fbccc389e15c01c05adc4db810533516340f15e9826dd44fe295"

# Column ordering matches artifacts/experiments/oof_h12/oof_fold0.npz's
# `actions` array exactly (alphabetical).
ACTIONS = [
    "apply_patch",
    "ask_user",
    "edit_file",
    "glob_pattern",
    "grep_search",
    "lint_or_typecheck",
    "list_directory",
    "plan_task",
    "read_file",
    "respond_only",
    "run_bash",
    "run_tests",
    "web_search",
    "write_file",
]

SEED = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fold_map() -> None:
    actual = sha256_file(FOLD_MAP_PATH)
    if actual != FOLD_MAP_SHA256:
        raise AssertionError(
            f"fold_map.csv SHA256 mismatch: got {actual}, expected {FOLD_MAP_SHA256}"
        )


def load_fold_map() -> dict[str, int]:
    verify_fold_map()
    out: dict[str, int] = {}
    with FOLD_MAP_PATH.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        id_col = "id" if "id" in reader.fieldnames else reader.fieldnames[0]
        fold_col = "fold" if "fold" in reader.fieldnames else reader.fieldnames[1]
        for row in reader:
            out[str(row[id_col])] = int(row[fold_col])
    return out


def load_train(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as f:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    ids = np.asarray([str(s["id"]) for s in samples], dtype=object)
    y = np.asarray([labels[str(s["id"])] for s in samples], dtype=object)
    groups = np.asarray([str(s["id"]).rsplit("-step_", 1)[0] for s in samples], dtype=object)
    return samples, ids, y, groups


def fold_assignment_for_ids(ids: Sequence[str], fold_map: dict[str, int]) -> np.ndarray:
    missing = [str(i) for i in ids if str(i) not in fold_map]
    if missing:
        raise AssertionError(f"{len(missing)} ids missing from fold_map (e.g. {missing[:3]})")
    return np.asarray([fold_map[str(i)] for i in ids], dtype=np.int64)


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
    dst_classes: Sequence[str] = ACTIONS,
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    out = np.full((probs.shape[0], len(dst_classes)), fill_value, dtype=np.float64)
    for dst_i, label in enumerate(dst_classes):
        if str(label) in src:
            out[:, dst_i] = probs[:, src.index(str(label))]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.ravel() <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(dst_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def labels_from_probs(probs: np.ndarray, actions: Sequence[str] = ACTIONS) -> np.ndarray:
    return np.asarray([str(a) for a in actions], dtype=object)[np.asarray(probs).argmax(axis=1)]


def macro_f1_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str] = ACTIONS) -> float:
    return float(f1_score(y_true, labels_from_probs(probs, actions), labels=list(actions), average="macro", zero_division=0))


def save_fold_npz(path: Path, *, ids: np.ndarray, probs: np.ndarray, y_true: np.ndarray, fold: int) -> None:
    """Save one fold's OOF slice with the same key schema as oof_h12 reference.

    ids and probs MUST be produced from the same iteration/loop so row i of
    probs corresponds to row i of ids (known trust boundary per task brief).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ids=np.asarray(ids, dtype=object),
        probs=np.asarray(probs, dtype=np.float32),
        y_true=np.asarray(y_true, dtype=object),
        actions=np.asarray(ACTIONS, dtype=object),
        fold=np.full(len(ids), fold, dtype=np.int64),
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_sha256sums(dir_path: Path, filenames: Sequence[str]) -> None:
    lines = []
    for name in filenames:
        digest = sha256_file(dir_path / name)
        lines.append(f"{digest}  {name}")
    (dir_path / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def verify_coverage(all_ids: np.ndarray, fold_map: dict[str, int]) -> None:
    expected = set(fold_map.keys())
    got = set(str(x) for x in all_ids)
    if got != expected:
        missing = expected - got
        extra = got - expected
        raise AssertionError(f"coverage mismatch: missing={len(missing)} extra={len(extra)}")
    if len(all_ids) != len(set(str(x) for x in all_ids)):
        raise AssertionError("duplicate ids in concatenated OOF")


class Timer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.time() - self.t0
