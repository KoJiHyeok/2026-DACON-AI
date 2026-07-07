# -*- coding: utf-8 -*-
"""Shared helpers for current_prompt template override R2.

The default normalizer mirrors `scripts/analysis/template_forensics.py` from
forensics R1.  Lowercasing is intentionally optional because the R1 source did
not lowercase before grouping, even though later notes summarized the process
more loosely.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
HOLDOUT_BASE = ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"
OUT_DIR = ROOT / "night_out" / "tmpl_override"

PATH_RE = re.compile(r"[\w\-]+(?:/[\w\-.]+)+|\b[\w\-]+\.[a-zA-Z]{1,6}\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUOTED_RE = re.compile(r"'[^']{1,60}'|\"[^\"]{1,60}\"")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:[A-Z][a-z0-9_]*){1,}\b")
STEP_RE = re.compile(r"-step_\d+$")


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def normalize_prompt(text: str, normalizer: str = "r1") -> str:
    """Return the R1 prompt template key.

    `r1` is the exact R1 placeholder order plus whitespace collapse.
    `r1_lower` additionally lowercases after placeholdering.
    """
    if normalizer not in {"r1", "r1_lower"}:
        raise ValueError(f"unknown normalizer: {normalizer}")
    t = str(text or "")
    t = QUOTED_RE.sub("<QUOTED>", t)
    t = PATH_RE.sub("<PATH>", t)
    t = NUM_RE.sub("<NUM>", t)
    t = IDENT_RE.sub("<IDENT>", t)
    t = " ".join(t.strip().split())
    if normalizer == "r1_lower":
        t = t.lower()
    return t


def load_train_records(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    labels_path = data_dir / "train_labels.csv"
    jsonl_path = data_dir / "train.jsonl"
    with labels_path.open(newline="", encoding="utf-8") as f:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}

    rows: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            sample_id = str(obj["id"])
            rows.append(
                {
                    "id": sample_id,
                    "session_id": session_id(sample_id),
                    "current_prompt": str(obj.get("current_prompt", "")),
                    "action": labels[sample_id],
                }
            )
    return rows


def load_holdout_ids(path: Path = HOLDOUT_BASE) -> set[str]:
    z = np.load(path, allow_pickle=True)
    return {str(x) for x in z["ids"]}


def summarize_records(
    records: Iterable[dict[str, Any]],
    normalizer: str = "r1",
    min_rows: int = 1,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n_rows": 0,
            "sessions": set(),
            "action_counts": Counter(),
            "example_prompt": "",
        }
    )
    for row in records:
        template = normalize_prompt(str(row["current_prompt"]), normalizer)
        rec = grouped[template]
        rec["n_rows"] += 1
        rec["sessions"].add(str(row["session_id"]))
        rec["action_counts"][str(row["action"])] += 1
        if not rec["example_prompt"]:
            rec["example_prompt"] = str(row["current_prompt"])

    out: list[dict[str, Any]] = []
    for template, rec in grouped.items():
        n_rows = int(rec["n_rows"])
        if n_rows < min_rows:
            continue
        action_counts: Counter[str] = rec["action_counts"]
        top_action, top_count = action_counts.most_common(1)[0]
        out.append(
            {
                "template": template,
                "n_rows": n_rows,
                "n_sessions": len(rec["sessions"]),
                "top_action": top_action,
                "top_count": int(top_count),
                "purity": float(top_count / n_rows),
                "action_counts": dict(sorted(action_counts.items())),
                "example_prompt": rec["example_prompt"],
            }
        )
    out.sort(key=lambda r: (-int(r["n_rows"]), str(r["template"])))
    return out


def merge_holdout_counts(
    nonholdout_stats: list[dict[str, Any]],
    holdout_stats: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    holdout_by_template = {str(row["template"]): row for row in holdout_stats}
    rows: list[dict[str, Any]] = []
    for row in nonholdout_stats:
        template = str(row["template"])
        hold = holdout_by_template.get(template)
        merged = {
            "template": template,
            "nonholdout_n": int(row["n_rows"]),
            "nonholdout_sessions": int(row["n_sessions"]),
            "top_action": str(row["top_action"]),
            "top_count": int(row["top_count"]),
            "purity": float(row["purity"]),
            "holdout_n": int(hold["n_rows"]) if hold else 0,
            "nonholdout_action_counts": row["action_counts"],
            "example_prompt": str(row["example_prompt"]),
        }
        rows.append(merged)
    rows.sort(key=lambda r: (-int(r["nonholdout_n"]), -int(r["holdout_n"]), str(r["template"])))
    return rows


def filter_mine_candidates(
    rows: Iterable[dict[str, Any]],
    min_nonholdout_n: int = 20,
    min_purity: float = 0.995,
    exclude_respond_only: bool = True,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if int(row["nonholdout_n"]) < min_nonholdout_n:
            continue
        if float(row["purity"]) < min_purity:
            continue
        if exclude_respond_only and str(row["top_action"]) == "respond_only":
            continue
        out.append(dict(row))
    return out


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, set):
        return sorted(str(v) for v in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = [json_ready(row) for row in rows]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"[save] {path}")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                    if isinstance(value, (dict, list))
                    else value
                    for key, value in row.items()
                }
            )
    print(f"[save] {path}")
