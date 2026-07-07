from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .constants import ACTIONS


WEIGHT_FILES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model", "vocab.txt")


def required_encoder_artifacts(model_dir: Path) -> Dict[str, Any]:
    model_dir = Path(model_dir)
    existing = []
    missing = []
    config_ok = (model_dir / "config.json").exists()
    weights_ok = any((model_dir / name).exists() for name in WEIGHT_FILES)
    tokenizer_ok = any((model_dir / name).exists() for name in TOKENIZER_FILES)
    if config_ok:
        existing.append("config.json")
    else:
        missing.append("config.json")
    if weights_ok:
        existing.append("model weights")
    else:
        missing.append("model.safetensors_or_pytorch_model.bin")
    if tokenizer_ok:
        existing.append("tokenizer files")
    else:
        missing.append("tokenizer files")
    return {
        "model_dir": str(model_dir),
        "ready": not missing,
        "existing": existing,
        "missing": missing,
    }


def normalize_proba(proba: np.ndarray) -> np.ndarray:
    arr = np.asarray(proba, dtype=np.float64)
    row_sum = arr.sum(axis=1, keepdims=True)
    return np.divide(arr, row_sum, out=np.zeros_like(arr), where=row_sum > 0)


def weighted_blend(components: Mapping[str, np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    out = None
    total = 0.0
    for name, proba in components.items():
        weight = float(weights.get(name, 0.0))
        if weight <= 0:
            continue
        arr = normalize_proba(proba)
        out = arr * weight if out is None else out + arr * weight
        total += weight
    if out is None or total <= 0:
        raise ValueError("At least one positive component weight is required")
    return normalize_proba(out / total)


def encoder_gate(valid_macro_f1: float) -> str:
    score = float(valid_macro_f1)
    if score >= 0.70:
        return "target_encoder"
    if score >= 0.69:
        return "strong_encoder"
    if score >= 0.67:
        return "candidate_encoder"
    if score >= 0.66:
        return "usable_encoder"
    return "failed"


def discover_encoder_runs(root: Path) -> list[Dict[str, Any]]:
    root = Path(root)
    runs: list[Dict[str, Any]] = []
    if not root.exists():
        return runs
    metrics_paths = []
    if (root / "metrics.json").exists():
        metrics_paths.append(root / "metrics.json")
    metrics_paths.extend(sorted(root.glob("*/metrics.json")))
    for metrics_path in metrics_paths:
        run_dir = metrics_path.parent
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        proba_path = run_dir / "oof_proba.npy"
        if not proba_path.exists():
            proba_path = run_dir / "valid_proba.npy"
        proba_shape = None
        if proba_path.exists():
            proba_shape = list(np.load(proba_path, allow_pickle=True, mmap_mode="r").shape)
        run_name = str(metrics.get("run_name") or run_dir.name)
        runs.append({
            "run_name": run_name,
            "run_dir": str(run_dir),
            "metrics_path": str(metrics_path),
            "proba_path": str(proba_path) if proba_path.exists() else "",
            "proba_shape": proba_shape,
            "valid_macro_f1": metrics.get("valid_macro_f1"),
            "gate": encoder_gate(float(metrics.get("valid_macro_f1", 0.0))),
            "model_artifacts": required_encoder_artifacts(run_dir / "model"),
        })
    return runs
