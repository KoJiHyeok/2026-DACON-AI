from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def find_first_existing(candidates: Iterable[str | Path]) -> Path:
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist: " + ", ".join(map(str, candidates)))


def load_train(data_dir: str | Path) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    data_dir = Path(data_dir)
    train_jsonl = find_first_existing([
        data_dir / "train.jsonl",
        data_dir / "data" / "train.jsonl",
        Path("open") / "train.jsonl",
        Path("open") / "data" / "train.jsonl",
    ])
    labels_csv = find_first_existing([
        data_dir / "train_labels.csv",
        data_dir / "data" / "train_labels.csv",
        Path("open") / "train_labels.csv",
        Path("open") / "data" / "train_labels.csv",
    ])
    records = read_jsonl(train_jsonl)
    labels = pd.read_csv(labels_csv)
    if not {"id", "action"}.issubset(labels.columns):
        raise ValueError("train_labels.csv must contain id, action columns")
    return records, labels


def load_test() -> List[Dict[str, Any]]:
    test_jsonl = find_first_existing([
        "data/test.jsonl",
        "open/test.jsonl",
        "open/data/test.jsonl",
        "./test.jsonl",
    ])
    return read_jsonl(test_jsonl)
