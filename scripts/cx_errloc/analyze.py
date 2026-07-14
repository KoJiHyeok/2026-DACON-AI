# -*- coding: utf-8 -*-
"""Deterministic CX-B holdout error-localization analysis.

Compares the deployed old and new Qwen blend surfaces, writes every target
class transition row, and evaluates only a small predeclared set of soft
probability-space interventions. It never mutates the source repository.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import f1_score


DEFAULT_SOURCE_ROOT = Path(r"C:\dev\2026-AI-DACON")
TARGETS = ("list_directory", "glob_pattern")
OLD_QWEN_WEIGHT = 2.0
OLD_AU_ALPHA = 0.90
NEW_QWEN_WEIGHT = 3.0
NEW_AU_ALPHA = 0.85


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--qwen-npz", type=Path, default=None)
    parser.add_argument("--au-cache", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=here / "analysis.json")
    parser.add_argument("--output-csv", type=Path, default=here / "transition_rows.csv")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_common(source_root: Path) -> Any:
    path = source_root / "scripts" / "league4" / "common.py"
    spec = importlib.util.spec_from_file_location("cx_errloc_league4_common", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    # common.py imports submit/au_route after adding its own ROOT to sys.path.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def labels_from_probs(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray([str(action) for action in actions], dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def class_f1(y_true: np.ndarray, pred: np.ndarray, actions: Sequence[str]) -> dict[str, float]:
    values = f1_score(
        y_true,
        pred,
        labels=[str(action) for action in actions],
        average=None,
        zero_division=0,
    )
    return {str(action): float(value) for action, value in zip(actions, values)}


def macro_f1(y_true: np.ndarray, pred: np.ndarray, actions: Sequence[str]) -> float:
    return float(
        f1_score(
            y_true,
            pred,
            labels=[str(action) for action in actions],
            average="macro",
            zero_division=0,
        )
    )


def load_au_probs(path: Path, data: Any) -> np.ndarray:
    with np.load(path, allow_pickle=True) as artifact:
        ids = np.asarray([str(value) for value in artifact["ids"]], dtype=object)
        actions = [str(value) for value in artifact["actions"]]
        probs = np.asarray(artifact["probs"], dtype=np.float64)
    assert_unique(ids, "AU cache ids")
    assert_unique(actions, "AU cache actions")
    expected_ids = data.ids[data.au_mask]
    if not np.array_equal(ids, expected_ids):
        raise AssertionError("AU cache ids are not in exact holdout AU row order")
    if actions != data.actions:
        raise AssertionError("AU cache actions are not in holdout action order")
    data_module_assert_probs(probs, len(ids), len(actions), "AU cache")
    return probs


def data_module_assert_probs(probs: np.ndarray, n_rows: int, n_classes: int, name: str) -> None:
    if probs.shape != (n_rows, n_classes):
        raise AssertionError(f"{name} shape {probs.shape} != {(n_rows, n_classes)}")
    if not np.all(np.isfinite(probs)):
        raise AssertionError(f"{name} contains non-finite values")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError(f"{name} row probabilities do not sum to 1")


def assert_unique(values: Sequence[Any], name: str) -> None:
    text = [str(value) for value in values]
    if len(set(text)) != len(text):
        raise AssertionError(f"{name} contains duplicates")


def validate_qwen_contract(path: Path, data: Any) -> dict[str, bool]:
    with np.load(path, allow_pickle=True) as artifact:
        ids = [str(value) for value in artifact["ids"]]
        actions = [str(value) for value in artifact["actions"]]
        y_true = np.asarray([str(value) for value in artifact["y_true"]], dtype=object)
        probs = np.asarray(artifact["probs"], dtype=np.float64)
    assert_unique(ids, "Qwen ids")
    assert_unique(actions, "Qwen actions")
    if set(ids) != set(str(value) for value in data.ids):
        raise AssertionError("Qwen id set does not exactly match holdout")
    if set(actions) != set(data.actions):
        raise AssertionError("Qwen action set does not exactly match holdout")
    data_module_assert_probs(probs, len(ids), len(actions), "Qwen source")
    row_index = {sample_id: index for index, sample_id in enumerate(ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in data.ids], dtype=np.int64)
    if not np.array_equal(y_true[rows], data.y_true):
        raise AssertionError("Qwen y_true does not match holdout after id alignment")
    return {
        "unique_ids": True,
        "exact_id_set": True,
        "unique_actions": True,
        "exact_action_set": True,
        "y_true_after_id_alignment": True,
    }


def apply_qwen_log_bias(
    qwen: np.ndarray,
    actions: Sequence[str],
    class_bias: Mapping[str, float] | None,
) -> np.ndarray:
    if not class_bias:
        return np.asarray(qwen, dtype=np.float64)
    logits = np.log(np.clip(np.asarray(qwen, dtype=np.float64), 1e-15, 1.0))
    for action, bias in class_bias.items():
        if action not in actions:
            raise ValueError(f"unknown action in class bias: {action}")
        logits[:, list(actions).index(action)] += float(bias)
    logits -= logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def surface_probs(
    data: Any,
    qwen: np.ndarray,
    au_probs: np.ndarray,
    qwen_weight: float,
    au_alpha: float,
) -> np.ndarray:
    if qwen_weight <= 0:
        raise ValueError("qwen_weight must be positive")
    if not 0.0 <= au_alpha <= 1.0:
        raise ValueError("au_alpha must be in [0, 1]")
    out = (data.lin + data.stk + qwen_weight * qwen) / (2.0 + qwen_weight)
    out = np.asarray(out, dtype=np.float64).copy()
    out[data.au_mask] = au_alpha * au_probs + (1.0 - au_alpha) * out[data.au_mask]
    data_module_assert_probs(out, len(data.ids), len(data.actions), "surface")
    return out


def surface_metrics(data: Any, probs: np.ndarray) -> dict[str, Any]:
    pred = labels_from_probs(probs, data.actions)
    stats: dict[str, dict[str, float | int]] = {}
    for action in data.actions:
        truth = data.y_true == action
        guessed = pred == action
        tp = int(np.count_nonzero(truth & guessed))
        fp = int(np.count_nonzero(~truth & guessed))
        fn = int(np.count_nonzero(truth & ~guessed))
        stats[action] = {
            "support": int(np.count_nonzero(truth)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
            "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        }
    return {
        "macro_f1": macro_f1(data.y_true, pred, data.actions),
        "correct_rows": int(np.count_nonzero(pred == data.y_true)),
        "class_f1": class_f1(data.y_true, pred, data.actions),
        "class_stats": stats,
    }


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def transition_rows(
    data: Any,
    qwen: np.ndarray,
    old_probs: np.ndarray,
    new_probs: np.ndarray,
) -> list[dict[str, Any]]:
    old_pred = labels_from_probs(old_probs, data.actions)
    new_pred = labels_from_probs(new_probs, data.actions)
    components = {"lin": data.lin, "stk": data.stk, "qwen": qwen}
    rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_col = data.actions.index(target)
        true_target = data.y_true == target
        masks = (
            ("old_correct_new_wrong", true_target & (old_pred == target) & (new_pred != target)),
            ("old_wrong_new_correct", true_target & (old_pred != target) & (new_pred == target)),
        )
        for transition, mask in masks:
            for index in np.flatnonzero(mask):
                sample = data.samples_by_id[str(data.ids[index])]
                meta = sample.get("session_meta") or {}
                workspace = meta.get("workspace") or {}
                row: dict[str, Any] = {
                    "holdout_index": int(index),
                    "id": str(data.ids[index]),
                    "target": target,
                    "transition": transition,
                    "old_pred": str(old_pred[index]),
                    "new_pred": str(new_pred[index]),
                    "is_au": bool(data.au_mask[index]),
                    "turn_index": meta.get("turn_index"),
                    "history_len": len(sample.get("history") or []),
                    "current_prompt": str(sample.get("current_prompt") or ""),
                    "workspace_loc": workspace.get("loc"),
                    "workspace_git_dirty": workspace.get("git_dirty"),
                    "workspace_open_files_count": len(workspace.get("open_files") or []),
                    "workspace_open_files": compact_json(workspace.get("open_files") or []),
                    "workspace_language_mix": compact_json(workspace.get("language_mix") or {}),
                    "old_target_prob": float(old_probs[index, target_col]),
                    "new_target_prob": float(new_probs[index, target_col]),
                    "old_margin": float(
                        old_probs[index, target_col]
                        - np.max(np.delete(old_probs[index], target_col))
                    ),
                    "new_margin": float(
                        new_probs[index, target_col]
                        - np.max(np.delete(new_probs[index], target_col))
                    ),
                }
                for name, probs in components.items():
                    row[f"{name}_pred"] = str(data.actions[int(probs[index].argmax())])
                    row[f"{name}_target_prob"] = float(probs[index, target_col])
                    competitor = str(new_pred[index] if transition == "old_correct_new_wrong" else old_pred[index])
                    competitor_col = data.actions.index(competitor)
                    row[f"{name}_target_minus_competitor"] = float(
                        probs[index, target_col] - probs[index, competitor_col]
                    )
                rows.append(row)
    rows.sort(key=lambda row: (TARGETS.index(str(row["target"])), str(row["transition"]), int(row["holdout_index"])))
    return rows


def numeric_summary(values: Iterable[int | float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def summarize_transitions(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for target in TARGETS:
        summary[target] = {}
        for transition in ("old_correct_new_wrong", "old_wrong_new_correct"):
            subset = [row for row in rows if row["target"] == target and row["transition"] == transition]
            other_key = "new_pred" if transition == "old_correct_new_wrong" else "old_pred"
            summary[target][transition] = {
                "count": len(subset),
                "ids": [str(row["id"]) for row in subset],
                "au_rows": sum(bool(row["is_au"]) for row in subset),
                "non_au_rows": sum(not bool(row["is_au"]) for row in subset),
                "confusion_direction": dict(sorted(Counter(str(row[other_key]) for row in subset).items())),
                "turn_index": numeric_summary(row["turn_index"] for row in subset),
                "turn_1_rows": sum(row["turn_index"] == 1 for row in subset),
                "history_len": numeric_summary(row["history_len"] for row in subset),
                "empty_history_rows": sum(row["history_len"] == 0 for row in subset),
                "workspace_loc": numeric_summary(row["workspace_loc"] for row in subset),
                "empty_open_files_rows": sum(row["workspace_open_files_count"] == 0 for row in subset),
                "component_target_votes": {
                    name: sum(row[f"{name}_pred"] == target for row in subset)
                    for name in ("lin", "stk", "qwen")
                },
            }
    return summary


def evaluate_candidate(
    name: str,
    data: Any,
    qwen: np.ndarray,
    au_probs: np.ndarray,
    baseline_probs: np.ndarray,
    qwen_weight: float = NEW_QWEN_WEIGHT,
    au_alpha: float = NEW_AU_ALPHA,
    qwen_log_bias: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    adjusted_qwen = apply_qwen_log_bias(qwen, data.actions, qwen_log_bias)
    probs = surface_probs(data, adjusted_qwen, au_probs, qwen_weight, au_alpha)
    pred = labels_from_probs(probs, data.actions)
    baseline_pred = labels_from_probs(baseline_probs, data.actions)
    metrics = surface_metrics(data, probs)
    baseline_metrics = surface_metrics(data, baseline_probs)
    class_delta = {
        action: metrics["class_f1"][action] - baseline_metrics["class_f1"][action]
        for action in data.actions
    }
    target_row_changes: dict[str, dict[str, int]] = {}
    for target in TARGETS:
        target_mask = data.y_true == target
        target_row_changes[target] = {
            "fixed_vs_new": int(np.count_nonzero(target_mask & (baseline_pred != target) & (pred == target))),
            "lost_vs_new": int(np.count_nonzero(target_mask & (baseline_pred == target) & (pred != target))),
        }
    return {
        "name": name,
        "formula": {
            "qwen_weight": qwen_weight,
            "au_alpha": au_alpha,
            "qwen_log_bias": dict(qwen_log_bias or {}),
        },
        "macro_f1": metrics["macro_f1"],
        "macro_f1_delta_vs_new": metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "correct_rows": metrics["correct_rows"],
        "correct_rows_delta_vs_new": metrics["correct_rows"] - baseline_metrics["correct_rows"],
        "prediction_rows_changed_vs_new": int(np.count_nonzero(pred != baseline_pred)),
        "correctness_transitions_vs_new": {
            "fixed": int(np.count_nonzero((baseline_pred != data.y_true) & (pred == data.y_true))),
            "lost": int(np.count_nonzero((baseline_pred == data.y_true) & (pred != data.y_true))),
        },
        "class_f1_delta_vs_new": class_delta,
        "target_class_f1_delta_vs_new": {target: class_delta[target] for target in TARGETS},
        "target_row_changes_vs_new": target_row_changes,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    qwen_path = (args.qwen_npz or source_root / "colab_out" / "qwen_i2ep_h85.npz").resolve()
    au_path = (
        args.au_cache
        or source_root / "night_out" / "league4" / "au_charwb_C1_holdout_probs.npz"
    ).resolve()
    common = load_common(source_root)
    data = common.load_league_data()
    assert_unique(data.ids, "holdout ids")
    assert_unique(data.train_ids, "train ids")
    qwen_contract = validate_qwen_contract(qwen_path, data)
    qwen = common.align_npz_probs(qwen_path, data.ids, data.y_true, data.actions)
    data_module_assert_probs(qwen, len(data.ids), len(data.actions), "Qwen")
    au_probs = load_au_probs(au_path, data)

    old_probs = surface_probs(data, qwen, au_probs, OLD_QWEN_WEIGHT, OLD_AU_ALPHA)
    new_probs = surface_probs(data, qwen, au_probs, NEW_QWEN_WEIGHT, NEW_AU_ALPHA)
    old_metrics = surface_metrics(data, old_probs)
    new_metrics = surface_metrics(data, new_probs)
    rows = transition_rows(data, qwen, old_probs, new_probs)

    # Three small, predeclared soft interventions plus two falsification controls.
    proposals = [
        evaluate_candidate(
            "qwen_log_bias_list_directory_p0p08",
            data,
            qwen,
            au_probs,
            new_probs,
            qwen_log_bias={"list_directory": 0.08},
        ),
        evaluate_candidate(
            "qwen_log_bias_glob_pattern_p0p10",
            data,
            qwen,
            au_probs,
            new_probs,
            qwen_log_bias={"glob_pattern": 0.10},
        ),
        evaluate_candidate(
            "qwen_log_bias_both_p0p08",
            data,
            qwen,
            au_probs,
            new_probs,
            qwen_log_bias={"list_directory": 0.08, "glob_pattern": 0.08},
        ),
    ]
    controls = [
        evaluate_candidate(
            "control_qwen_weight_2p8",
            data,
            qwen,
            au_probs,
            new_probs,
            qwen_weight=2.8,
        ),
        evaluate_candidate(
            "control_au_alpha_0p90",
            data,
            qwen,
            au_probs,
            new_probs,
            au_alpha=0.90,
        ),
    ]

    input_paths = {
        "qwen_holdout": qwen_path,
        "holdout_base": Path(common.HOLDOUT_BASE),
        "au_cache": au_path,
        "train_jsonl": source_root / "data" / "train.jsonl",
        "train_labels": source_root / "data" / "train_labels.csv",
        "linear_probs": Path(common.OOF_DIR) / "linear_probs.npy",
        "stacker_probs": Path(common.OOF_DIR) / "stacker_probs.npy",
        "row_ids": Path(common.OOF_DIR) / "row_ids.json",
        "classes": Path(common.OOF_DIR) / "classes.json",
        "league_common": source_root / "scripts" / "league4" / "common.py",
        "au_route": source_root / "submit" / "au_route.py",
    }
    payload = {
        "task_id": "CX-B",
        "determinism": "No randomness; exact id/y_true/action alignment; sorted transition output.",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in input_paths.items()
        },
        "alignment": {
            "holdout_rows": len(data.ids),
            "classes": len(data.actions),
            "actions": data.actions,
            "au_rows": int(np.count_nonzero(data.au_mask)),
            "qwen_rows_aligned": len(qwen),
            "au_cache_rows_aligned": len(au_probs),
            "qwen_contract": qwen_contract,
        },
        "surfaces": {
            "old": {
                "formula": "soft_AU((lin + stk + 2*qwen)/4, alpha=0.90)",
                **old_metrics,
            },
            "new": {
                "formula": "soft_AU((lin + stk + 3*qwen)/5, alpha=0.85)",
                **new_metrics,
            },
            "delta_new_minus_old": {
                "macro_f1": new_metrics["macro_f1"] - old_metrics["macro_f1"],
                "correct_rows": new_metrics["correct_rows"] - old_metrics["correct_rows"],
                "class_f1": {
                    action: new_metrics["class_f1"][action] - old_metrics["class_f1"][action]
                    for action in data.actions
                },
            },
        },
        "transitions": summarize_transitions(rows),
        "proposals": proposals,
        "falsification_controls": controls,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_csv, rows)
    print(f"old_macro_f1={old_metrics['macro_f1']:.10f}")
    print(f"new_macro_f1={new_metrics['macro_f1']:.10f}")
    print(f"delta={new_metrics['macro_f1'] - old_metrics['macro_f1']:+.10f}")
    for target in TARGETS:
        lost = payload["transitions"][target]["old_correct_new_wrong"]["count"]
        gained = payload["transitions"][target]["old_wrong_new_correct"]["count"]
        print(f"{target}: lost={lost} gained={gained}")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_csv}")


if __name__ == "__main__":
    main()
