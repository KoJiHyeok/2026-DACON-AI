"""Fit fold-honest temperature and class-bias calibration for Qwen.

Each validation session is calibrated only with parameters fitted on the
other four GroupKFold folds.  The full-data refit is written separately as a
deployment candidate and is never used for the OOF recommendation score.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = Path(r"C:\dev\2026-AI-DACON")
DEFAULT_QWEN = DEFAULT_EXTERNAL_ROOT / "colab_out" / "qwen_i2ep_h85.npz"
DEFAULT_HOLDOUT = DEFAULT_EXTERNAL_ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"
DEFAULT_COMMON = DEFAULT_EXTERNAL_ROOT / "scripts" / "league4" / "common.py"
DEFAULT_CANDIDATE = Path(__file__).with_name("calib_candidate.json")
DEFAULT_SUMMARY = Path(__file__).with_name("fit_summary.json")


@dataclass(frozen=True)
class Calibration:
    temperature: float
    class_bias: np.ndarray


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_calibration(probs: np.ndarray, temperature: float, class_bias: Sequence[float]) -> np.ndarray:
    """Mirror submit/script.py: softmax(log(clip(p))/T + bias)."""
    p = np.asarray(probs, dtype=np.float64)
    bias = np.asarray(class_bias, dtype=np.float64)
    if p.ndim != 2 or bias.shape != (p.shape[1],):
        raise ValueError(f"shape mismatch: probs={p.shape}, class_bias={bias.shape}")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(bias)):
        raise ValueError("probabilities and bias must be finite")
    return softmax(np.log(np.clip(p, 1e-12, None)) / float(temperature) + bias.reshape(1, -1))


def build_group_folds(ids: Sequence[str], n_splits: int = 5) -> list[tuple[np.ndarray, np.ndarray]]:
    ids_arr = np.asarray([str(x) for x in ids], dtype=object)
    groups = np.asarray([session_id(x) for x in ids_arr], dtype=object)
    splitter = GroupKFold(n_splits=n_splits)
    folds = [(tr.astype(np.int64), va.astype(np.int64)) for tr, va in splitter.split(ids_arr, groups=groups)]
    seen = np.zeros(len(ids_arr), dtype=np.int8)
    for tr, va in folds:
        if set(groups[tr]).intersection(groups[va]):
            raise AssertionError("session leakage between train and validation")
        seen[va] += 1
    if not np.all(seen == 1):
        raise AssertionError("each row must occur in exactly one validation fold")
    return folds


def _nll(probs: np.ndarray, y_idx: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y_idx)), y_idx], 1e-15, None)).mean())


def _fit_temperature(
    probs: np.ndarray,
    y_idx: np.ndarray,
    bias: np.ndarray,
    bounds: tuple[float, float],
) -> float:
    logp = np.log(np.clip(probs, 1e-12, None))

    def objective(t: float) -> float:
        return _nll(softmax(logp / float(t) + bias.reshape(1, -1)), y_idx)

    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-7})
    if not result.success:
        raise RuntimeError(f"temperature optimization failed: {result.message}")
    return float(result.x)


def _fit_centered_bias(
    probs: np.ndarray,
    y_idx: np.ndarray,
    temperature: float,
    ridge: float,
    initial: np.ndarray,
) -> np.ndarray:
    n_classes = probs.shape[1]
    logp_scaled = np.log(np.clip(probs, 1e-12, None)) / float(temperature)
    init = np.asarray(initial, dtype=np.float64)
    init = init - init.mean()
    theta0 = init[:-1]

    def unpack(theta: np.ndarray) -> np.ndarray:
        return np.concatenate([theta, [-float(theta.sum())]])

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        bias = unpack(theta)
        calibrated = softmax(logp_scaled + bias.reshape(1, -1))
        loss = _nll(calibrated, y_idx) + 0.5 * ridge * float(np.dot(bias, bias))
        grad_bias = calibrated
        grad_bias[np.arange(len(y_idx)), y_idx] -= 1.0
        grad_bias = grad_bias.mean(axis=0) + ridge * bias
        grad_theta = grad_bias[:-1] - grad_bias[-1]
        return loss, grad_theta

    result = minimize(objective, theta0, method="L-BFGS-B", jac=True, options={"maxiter": 500, "ftol": 1e-12})
    if not result.success:
        raise RuntimeError(f"class-bias optimization failed: {result.message}")
    bias = unpack(np.asarray(result.x, dtype=np.float64))
    return bias - bias.mean()


def fit_calibration(
    probs: np.ndarray,
    y_idx: np.ndarray,
    ridge: float = 1e-3,
    temperature_bounds: tuple[float, float] = (0.05, 5.0),
    max_rounds: int = 20,
) -> Calibration:
    """Alternate scalar temperature fitting and intercept-only bias fitting."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y_idx, dtype=np.int64)
    if p.ndim != 2 or len(p) != len(y):
        raise ValueError("probs and y_idx row counts must match")
    if np.any(y < 0) or np.any(y >= p.shape[1]):
        raise ValueError("y_idx contains an invalid class index")
    bias = np.zeros(p.shape[1], dtype=np.float64)
    temperature = _fit_temperature(p, y, bias, temperature_bounds)
    for _ in range(max_rounds):
        previous_t, previous_bias = temperature, bias.copy()
        bias = _fit_centered_bias(p, y, temperature, ridge, bias)
        temperature = _fit_temperature(p, y, bias, temperature_bounds)
        if abs(temperature - previous_t) < 1e-7 and np.max(np.abs(bias - previous_bias)) < 1e-7:
            break
    bias = _fit_centered_bias(p, y, temperature, ridge, bias)
    return Calibration(temperature=temperature, class_bias=bias)


def calibration_payload(calibration: Calibration, actions: Sequence[str]) -> dict:
    if len(actions) != len(calibration.class_bias):
        raise ValueError("actions and class_bias lengths differ")
    return {
        "temperature": float(calibration.temperature),
        "class_bias": {str(action): float(value) for action, value in zip(actions, calibration.class_bias)},
    }


def write_calibration_json(path: Path, calibration: Calibration, actions: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(calibration_payload(calibration, actions), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_calibration_json(path: Path, actions: Sequence[str]) -> Calibration:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if set(payload) != {"temperature", "class_bias"} or not isinstance(payload["class_bias"], dict):
        raise ValueError("calibration JSON must contain only temperature and class_bias")
    missing = [str(a) for a in actions if str(a) not in payload["class_bias"]]
    extra = sorted(set(payload["class_bias"]) - {str(a) for a in actions})
    if missing or extra:
        raise ValueError(f"class_bias keys mismatch: missing={missing}, extra={extra}")
    return Calibration(
        temperature=float(payload["temperature"]),
        class_bias=np.asarray([float(payload["class_bias"][str(a)]) for a in actions], dtype=np.float64),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sessions_sha256(values: Sequence[str]) -> str:
    data = "\n".join(sorted({str(v) for v in values})).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def load_external_common(path: Path):
    module_name = "cx_calib_external_league_common"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import league common from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_aligned_qwen(qwen_path: Path, holdout_path: Path, common_path: Path):
    common = load_external_common(common_path)
    holdout = np.load(holdout_path, allow_pickle=True)
    ids = np.asarray([str(x) for x in holdout["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in holdout["y_true"]], dtype=object)
    actions = [str(x) for x in holdout["actions"]]
    probs = common.align_npz_probs(qwen_path, ids, y_true, actions)
    if probs.shape != (len(ids), len(actions)) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("aligned Qwen probabilities have an invalid shape or row sum")
    return ids, y_true, actions, probs


def _score_record(probs: np.ndarray, y_idx: np.ndarray, actions: Sequence[str]) -> dict[str, float]:
    pred_idx = probs.argmax(axis=1)
    labels = list(range(len(actions)))
    return {
        "nll": _nll(probs, y_idx),
        "macro_f1": float(f1_score(y_idx, pred_idx, labels=labels, average="macro", zero_division=0)),
    }


def run(args: argparse.Namespace) -> dict:
    ids, y_true, actions, probs = load_aligned_qwen(args.qwen, args.holdout_base, args.common)
    action_index = {action: i for i, action in enumerate(actions)}
    y_idx = np.asarray([action_index[str(y)] for y in y_true], dtype=np.int64)
    groups = np.asarray([session_id(x) for x in ids], dtype=object)
    folds = build_group_folds(ids, n_splits=args.n_splits)
    oof = np.empty_like(probs, dtype=np.float64)
    fold_rows = []

    for fold, (train_idx, valid_idx) in enumerate(folds):
        calibration = fit_calibration(probs[train_idx], y_idx[train_idx], ridge=args.ridge)
        calibrated_train = apply_calibration(probs[train_idx], calibration.temperature, calibration.class_bias)
        calibrated_valid = apply_calibration(probs[valid_idx], calibration.temperature, calibration.class_bias)
        oof[valid_idx] = calibrated_valid
        train_sessions = groups[train_idx]
        valid_sessions = groups[valid_idx]
        row = {
            "fold": fold,
            "n_train_rows": int(len(train_idx)),
            "n_valid_rows": int(len(valid_idx)),
            "n_train_sessions": int(len(set(train_sessions))),
            "n_valid_sessions": int(len(set(valid_sessions))),
            "validation_sessions_sha256": sessions_sha256(valid_sessions),
            **calibration_payload(calibration, actions),
            "train_before": _score_record(probs[train_idx], y_idx[train_idx], actions),
            "train_after": _score_record(calibrated_train, y_idx[train_idx], actions),
            "valid_before": _score_record(probs[valid_idx], y_idx[valid_idx], actions),
            "valid_after": _score_record(calibrated_valid, y_idx[valid_idx], actions),
        }
        fold_rows.append(row)
        print(
            f"fold {fold}: T={calibration.temperature:.6f} "
            f"valid NLL {row['valid_before']['nll']:.6f}->{row['valid_after']['nll']:.6f} "
            f"F1 {row['valid_before']['macro_f1']:.6f}->{row['valid_after']['macro_f1']:.6f}"
        )

    full = fit_calibration(probs, y_idx, ridge=args.ridge)
    full_probs = apply_calibration(probs, full.temperature, full.class_bias)
    write_calibration_json(args.output, full, actions)
    summary = {
        "schema_version": 1,
        "protocol": (
            f"GroupKFold(n_splits={args.n_splits}, group=id prefix before -step_); "
            f"fold parameters fit on the other {args.n_splits - 1} folds"
        ),
        "n_splits": args.n_splits,
        "ridge": args.ridge,
        "actions": actions,
        "n_rows": int(len(ids)),
        "n_sessions": int(len(set(groups))),
        "inputs": {
            "qwen_npz": str(args.qwen),
            "qwen_sha256": sha256_file(args.qwen),
            "holdout_base": str(args.holdout_base),
            "holdout_base_sha256": sha256_file(args.holdout_base),
        },
        "folds": fold_rows,
        "oof_before": _score_record(probs, y_idx, actions),
        "oof_after": _score_record(oof, y_idx, actions),
        "full_refit": {
            **calibration_payload(full, actions),
            "before": _score_record(probs, y_idx, actions),
            "after": _score_record(full_probs, y_idx, actions),
            "usage": "deployment candidate only; excluded from recommendation metrics",
        },
        "candidate_json": str(args.output),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"wrote {args.summary}")
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--holdout-base", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--output", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--ridge", type=float, default=1e-3)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
