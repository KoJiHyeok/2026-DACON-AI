from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .constants import ACTIONS


RESERVED_OUTPUTS = {"submit.zip", "e5s42gpuopt.zip", "submit_encoder_e5_s42_gpuopt.zip"}


def safe_zip_name(name: str) -> str:
    if not name.endswith(".zip"):
        raise ValueError("DACON submit candidate name must end with .zip")
    if len(name) > 30:
        raise ValueError(f"DACON zip name must be 30 chars or fewer: {name}")
    if Path(name).name in RESERVED_OUTPUTS:
        raise ValueError(f"Refusing to overwrite protected submit artifact: {name}")
    return name


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_oof_array(arr: np.ndarray, n_samples: int, n_actions: int = len(ACTIONS), name: str = "oof") -> None:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if arr.shape != (n_samples, n_actions):
        raise ValueError(f"{name} must have shape ({n_samples}, {n_actions}), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf")
    row_sum = arr.sum(axis=1)
    if np.any(row_sum <= 0):
        raise ValueError(f"{name} contains non-positive probability rows")


def merge_saved_fold_probas(
    out_dir: str | Path,
    n_samples: int,
    n_actions: int,
    fold_ids: np.ndarray | None = None,
) -> tuple[np.ndarray, list[int]]:
    out_dir = Path(out_dir)
    oof = np.zeros((n_samples, n_actions), dtype=np.float32)
    covered_folds: list[int] = []
    for fold_dir in sorted(out_dir.glob("fold*")):
        if not fold_dir.is_dir():
            continue
        suffix = fold_dir.name.replace("fold", "", 1)
        if not suffix.isdigit():
            continue
        fold = int(suffix)
        idx_path = fold_dir / "valid_indices.npy"
        proba_path = fold_dir / "valid_proba.npy"
        if not idx_path.exists() or not proba_path.exists():
            continue
        valid_idx = np.load(idx_path).astype(np.int64)
        proba = np.load(proba_path)
        if proba.shape != (len(valid_idx), n_actions):
            raise ValueError(f"{fold_dir} proba shape {proba.shape} does not match indices {len(valid_idx)}")
        if np.any(valid_idx < 0) or np.any(valid_idx >= n_samples):
            raise ValueError(f"{fold_dir} contains indices outside 0..{n_samples - 1}")
        if fold_ids is not None and len(fold_ids) >= n_samples:
            expected = np.asarray(fold_ids)[valid_idx]
            if not np.all(expected == fold):
                raise ValueError(f"{fold_dir} valid_indices do not match fold_ids for fold {fold}")
        oof[valid_idx] = normalize_proba(proba).astype(np.float32)
        covered_folds.append(fold)
    return oof, covered_folds


def fold_output_complete(out_dir: str | Path, fold: int) -> bool:
    fold_dir = Path(out_dir) / f"fold{fold}"
    return (fold_dir / "valid_proba.npy").exists() and (fold_dir / "valid_indices.npy").exists()


def normalize_proba(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float64).copy()
    out = np.clip(out, 0.0, None)
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / out.shape[1]), where=row_sum > 0)


def load_oof(path: str | Path, n_samples: int | None = None) -> np.ndarray:
    arr = np.load(path)
    if n_samples is not None:
        validate_oof_array(arr, n_samples, len(ACTIONS), str(path))
    return normalize_proba(arr)


def find_reproducible_baseline_oof(root: str | Path = ".") -> Dict[str, Any]:
    root = Path(root)
    candidates = [
        root / "cache" / "repro_0681153_no_group" / "oof_best.npy",
        root / "cache" / "oof_35k_quick" / "oof_best.npy",
    ]
    for path in candidates:
        if path.exists():
            arr = np.load(path, mmap_mode="r")
            return {
                "name": "aar_et_reproducible_35k",
                "path": str(path),
                "shape": list(arr.shape),
                "note": "Previous exact 0.681153 OOF is missing; this is the auditable local baseline.",
            }
    return {
        "name": "missing",
        "path": "",
        "shape": [],
        "note": "No reproducible AAR/ET OOF cache found.",
    }


def write_component_manifest(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload.setdefault("action_order", ACTIONS)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
