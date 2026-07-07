# -*- coding: utf-8 -*-
"""Shared utilities for task3 subpopulation-routing sweeps.

This module intentionally embeds the task1 4-way join recipe so `sweep.py` and
`probe.py` evaluate exactly the same holdout rows and class order.
"""
from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
if str(SUBMIT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMIT_DIR))

DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
HOLDOUT_BASE = ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"
MBERT_HOLDOUT = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz")
OUT_DIR = ROOT / "night_out" / "task3_subroute"

EXPECTED_3WAY = 0.71726
EXPECTED_4WAY = 0.72255
SANITY_TOL = 5e-4
DEFAULT_ALPHAS = (0.5, 0.6, 0.7, 0.8, 0.9)
SOFT_AU_ALPHA = 0.9
SCREEN_WEAK_DELTA = -0.03
SCREEN_MIN_HOLDOUT = 300
SCREEN_MIN_TRAIN = 3000

STEP_RE = re.compile(r"-step_(\d+)$")


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def step_num(sample_id: str) -> int:
    match = STEP_RE.search(str(sample_id))
    return int(match.group(1)) if match else -1


def is_au(sample_id: str) -> bool:
    return str(sample_id).startswith("sess_au")


def id_family(sample_id: str) -> str:
    parts = str(sample_id).split("_")
    if len(parts) >= 2 and parts[0] == "sess":
        return f"{parts[0]}_{parts[1]}"
    return parts[0] if parts else "unknown"


def safe_value(value: Any) -> str:
    if value is None or value == "":
        return "missing"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def read_labels(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        return {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}


def load_train(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    labels = read_labels(data_dir / "train_labels.csv")
    ids = np.asarray([str(sample["id"]) for sample in samples], dtype=object)
    y = np.asarray([labels[str(sample["id"])] for sample in samples], dtype=object)
    groups = np.asarray([session_id(str(sample_id)) for sample_id in ids], dtype=object)
    return samples, ids, y, groups


def predict_labels(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray([str(a) for a in actions], dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1_labels(y_true: np.ndarray, pred: np.ndarray) -> float:
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def macro_f1_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> float:
    return macro_f1_labels(y_true, predict_labels(probs, actions))


def align_probs(
    probs: np.ndarray,
    src_classes: Sequence[str],
    dst_classes: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    dst = [str(c) for c in dst_classes]
    out = np.full((probs.shape[0], len(dst)), fill_value, dtype=np.float64)
    for dst_i, cls in enumerate(dst):
        if cls in src:
            out[:, dst_i] = probs[:, src.index(cls)]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.ravel() <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(dst)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def _load_oof_component(
    *,
    holdout_ids: np.ndarray,
    actions: Sequence[str],
    oof_dir: Path,
    filename: str,
) -> np.ndarray:
    classes = json.loads((oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(str(action)) for action in actions]
    row_index = {str(row_id): i for i, row_id in enumerate(row_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in holdout_ids], dtype=np.int64)
    return np.load(oof_dir / filename)[:, col][rows].astype(np.float64)


def _load_npz_probs_aligned(path: Path, holdout_ids: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    src_ids = np.asarray([str(x) for x in z["ids"]], dtype=object)
    src_actions = [str(x) for x in z["actions"]]
    src_y = np.asarray([str(x) for x in z["y_true"]], dtype=object)
    row_index = {str(sample_id): i for i, sample_id in enumerate(src_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in holdout_ids], dtype=np.int64)
    aligned_y = src_y[rows]
    if not np.array_equal(aligned_y, y_true):
        raise AssertionError(f"{path} y_true mismatch after id join")
    probs = np.asarray(z["probs"], dtype=np.float64)[rows]
    return align_probs(probs, src_actions, actions)


def load_league(
    holdout_base: Path = HOLDOUT_BASE,
    oof_dir: Path = OOF_DIR,
    mbert_holdout: Path = MBERT_HOLDOUT,
) -> dict[str, Any]:
    enc = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in enc["y_true"]], dtype=object)
    actions = [str(x) for x in enc["actions"]]
    e5 = np.asarray(enc["probs"], dtype=np.float64)

    lin = _load_oof_component(holdout_ids=ids, actions=actions, oof_dir=oof_dir, filename="linear_probs.npy")
    stk = _load_oof_component(holdout_ids=ids, actions=actions, oof_dir=oof_dir, filename="stacker_probs.npy")
    mbert = _load_npz_probs_aligned(mbert_holdout, ids, y_true, actions)

    blend3 = (lin + stk + 2.0 * e5) / 4.0
    blend4 = (lin + stk + 1.2 * e5 + 0.8 * mbert) / 4.0
    score3 = macro_f1_probs(blend3, y_true, actions)
    score4 = macro_f1_probs(blend4, y_true, actions)
    if abs(score3 - EXPECTED_3WAY) > SANITY_TOL:
        raise AssertionError(f"3-way sanity failed: got {score3:.8f}, expected {EXPECTED_3WAY:.5f}")
    if abs(score4 - EXPECTED_4WAY) > SANITY_TOL:
        raise AssertionError(f"4-way sanity failed: got {score4:.8f}, expected {EXPECTED_4WAY:.5f}")

    return {
        "ids": ids,
        "y_true": y_true,
        "actions": actions,
        "linear": lin,
        "stacker": stk,
        "e5": e5,
        "mbert": mbert,
        "blend3": blend3,
        "blend4": blend4,
        "blend3_score": score3,
        "blend4_score": score4,
        "blend4_pred": predict_labels(blend4, actions),
    }


def num_or_nan(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return float("nan")


def budget_bucket(value: Any) -> str:
    val = num_or_nan(value)
    if math.isnan(val):
        return "missing"
    if val < 1000:
        return "0000_1000"
    if val < 10000:
        return "1000_10000"
    if val < 50000:
        return "10000_50000"
    return "50000_plus"


def flatten_sample(sample: dict[str, Any]) -> dict[str, Any]:
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    if not isinstance(history, list):
        history = []
    open_files = workspace.get("open_files") or []
    if not isinstance(open_files, list):
        open_files = []
    sample_id = str(sample.get("id") or "")
    return {
        "id": sample_id,
        "id_family": id_family(sample_id),
        "session": session_id(sample_id),
        "step": step_num(sample_id),
        "is_au": is_au(sample_id),
        "user_tier": safe_value(meta.get("user_tier")),
        "language_pref": safe_value(meta.get("language_pref")),
        "last_ci_status": safe_value(workspace.get("last_ci_status")),
        "git_dirty": bool(workspace.get("git_dirty")),
        "budget_bucket": budget_bucket(meta.get("budget_tokens_remaining")),
        "budget_tokens_remaining": num_or_nan(meta.get("budget_tokens_remaining")),
        "workspace_loc": num_or_nan(workspace.get("loc")),
        "turn_index": num_or_nan(meta.get("turn_index")),
        "open_files_empty": len(open_files) == 0,
        "history_len": len(history),
    }


def _finite_values(flats: Sequence[dict[str, Any]], field: str, non_au_only: bool = True) -> np.ndarray:
    values = []
    for row in flats:
        if non_au_only and row["is_au"]:
            continue
        val = row.get(field)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and not math.isnan(float(val)):
            values.append(float(val))
    return np.asarray(values, dtype=np.float64)


def candidate_name(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "id_family":
        return f"id_family={spec['value']}"
    if kind == "field_equals":
        value = spec["value"]
        if isinstance(value, bool):
            value = "true" if value else "false"
        return f"{spec['field']}={value}"
    if kind == "numeric_ge":
        return f"{spec['field']}>={spec['threshold_label']}"
    if kind == "numeric_le":
        return f"{spec['field']}<={spec['threshold_label']}"
    if kind == "flag_true":
        return str(spec["field"])
    if kind == "and":
        return "cross:" + "&".join(candidate_name(part) for part in spec["parts"])
    raise ValueError(f"unknown candidate kind: {kind}")


def candidate_family(spec: dict[str, Any]) -> str:
    kind = spec["kind"]
    if kind == "field_equals":
        return str(spec["field"])
    if kind in {"numeric_ge", "numeric_le", "flag_true"}:
        return str(spec["field"])
    if kind == "id_family":
        return "id_family"
    if kind == "and":
        return "cross"
    return kind


def candidate_mask(spec: dict[str, Any], flats: Sequence[dict[str, Any]]) -> np.ndarray:
    kind = spec["kind"]
    if kind == "id_family":
        return np.asarray([row["id_family"] == spec["value"] for row in flats], dtype=bool)
    if kind == "field_equals":
        return np.asarray([row.get(spec["field"]) == spec["value"] for row in flats], dtype=bool)
    if kind == "numeric_ge":
        threshold = float(spec["threshold"])
        out = []
        for row in flats:
            val = row.get(spec["field"])
            out.append(isinstance(val, (int, float)) and not math.isnan(float(val)) and float(val) >= threshold)
        return np.asarray(out, dtype=bool)
    if kind == "numeric_le":
        threshold = float(spec["threshold"])
        out = []
        for row in flats:
            val = row.get(spec["field"])
            out.append(isinstance(val, (int, float)) and not math.isnan(float(val)) and float(val) <= threshold)
        return np.asarray(out, dtype=bool)
    if kind == "flag_true":
        return np.asarray([bool(row.get(spec["field"])) for row in flats], dtype=bool)
    if kind == "and":
        masks = [candidate_mask(part, flats) for part in spec["parts"]]
        if not masks:
            return np.zeros(len(flats), dtype=bool)
        out = masks[0].copy()
        for mask in masks[1:]:
            out &= mask
        return out
    raise ValueError(f"unknown candidate kind: {kind}")


def build_base_candidate_specs(samples: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flats = [flatten_sample(sample) for sample in samples]
    specs: list[dict[str, Any]] = []

    for value in sorted({row["id_family"] for row in flats}):
        if value not in {"sess_sim", "sess_au"}:
            specs.append({"kind": "id_family", "value": value})

    for field in ("user_tier", "language_pref", "last_ci_status", "git_dirty", "budget_bucket"):
        values = sorted({row[field] for row in flats}, key=lambda x: str(x))
        for value in values:
            specs.append({"kind": "field_equals", "field": field, "value": value})

    loc_values = _finite_values(flats, "workspace_loc")
    if loc_values.size:
        p10 = float(np.quantile(loc_values, 0.10))
        p90 = float(np.quantile(loc_values, 0.90))
        specs.append(
            {
                "kind": "numeric_le",
                "field": "workspace_loc",
                "threshold": p10,
                "threshold_label": f"p10_{int(round(p10))}",
            }
        )
        specs.append(
            {
                "kind": "numeric_ge",
                "field": "workspace_loc",
                "threshold": p90,
                "threshold_label": f"p90_{int(round(p90))}",
            }
        )

    for threshold in (8, 10, 12):
        specs.append(
            {
                "kind": "numeric_ge",
                "field": "turn_index",
                "threshold": float(threshold),
                "threshold_label": str(threshold),
            }
        )

    specs.append({"kind": "flag_true", "field": "open_files_empty"})
    specs.append(
        {
            "kind": "numeric_ge",
            "field": "history_len",
            "threshold": 10.0,
            "threshold_label": "10",
        }
    )
    return specs


@dataclass
class DataContext:
    samples: list[dict[str, Any]]
    ids: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    flats: list[dict[str, Any]]
    holdout_samples: list[dict[str, Any]]
    holdout_flats: list[dict[str, Any]]
    holdout_id_set: set[str]
    nonholdout_mask: np.ndarray
    train_non_au_mask: np.ndarray
    holdout_non_au_mask: np.ndarray


def build_context(samples: list[dict[str, Any]], ids: np.ndarray, y: np.ndarray, groups: np.ndarray, holdout_ids: np.ndarray) -> DataContext:
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    holdout_samples = [sample_by_id[str(sample_id)] for sample_id in holdout_ids]
    flats = [flatten_sample(sample) for sample in samples]
    holdout_flats = [flatten_sample(sample) for sample in holdout_samples]
    holdout_id_set = {str(sample_id) for sample_id in holdout_ids}
    nonholdout_mask = np.asarray([str(sample_id) not in holdout_id_set for sample_id in ids], dtype=bool)
    train_non_au_mask = np.asarray([not is_au(str(sample_id)) for sample_id in ids], dtype=bool)
    holdout_non_au_mask = np.asarray([not is_au(str(sample_id)) for sample_id in holdout_ids], dtype=bool)
    return DataContext(
        samples=samples,
        ids=ids,
        y=y,
        groups=groups,
        flats=flats,
        holdout_samples=holdout_samples,
        holdout_flats=holdout_flats,
        holdout_id_set=holdout_id_set,
        nonholdout_mask=nonholdout_mask,
        train_non_au_mask=train_non_au_mask,
        holdout_non_au_mask=holdout_non_au_mask,
    )


def score_candidate(
    *,
    spec: dict[str, Any],
    ctx: DataContext,
    league: dict[str, Any],
    screen_weak_delta: float = SCREEN_WEAK_DELTA,
    screen_min_holdout: int = SCREEN_MIN_HOLDOUT,
    screen_min_train: int = SCREEN_MIN_TRAIN,
) -> dict[str, Any]:
    train_mask_raw = candidate_mask(spec, ctx.flats)
    holdout_mask_raw = candidate_mask(spec, ctx.holdout_flats)
    train_mask = train_mask_raw & ctx.train_non_au_mask
    train_nonholdout_mask = train_mask & ctx.nonholdout_mask
    holdout_mask = holdout_mask_raw & ctx.holdout_non_au_mask
    holdout_turn0 = np.asarray(
        [row.get("turn_index") == 0.0 for row in ctx.holdout_flats],
        dtype=bool,
    )
    train_turn0 = np.asarray(
        [row.get("turn_index") == 0.0 for row in ctx.flats],
        dtype=bool,
    )

    holdout_rows = int(holdout_mask.sum())
    if holdout_rows:
        group_pred = predict_labels(league["blend4"][holdout_mask], league["actions"])
        group_f1 = macro_f1_labels(league["y_true"][holdout_mask], group_pred)
    else:
        group_f1 = float("nan")
    delta = group_f1 - float(league["blend4_score"]) if not math.isnan(group_f1) else float("nan")
    train_nonholdout_rows = int(train_nonholdout_mask.sum())
    screen_pass = (
        holdout_rows >= screen_min_holdout
        and train_nonholdout_rows >= screen_min_train
        and not math.isnan(delta)
        and delta <= screen_weak_delta
    )
    return {
        "name": candidate_name(spec),
        "family": candidate_family(spec),
        "spec": spec,
        "holdout_rows": holdout_rows,
        "holdout_share": holdout_rows / max(int(ctx.holdout_non_au_mask.sum()), 1),
        "holdout_turn0_rows": int((holdout_mask & holdout_turn0).sum()),
        "holdout_turn0_share": int((holdout_mask & holdout_turn0).sum()) / max(holdout_rows, 1),
        "train_rows_all": int(train_mask.sum()),
        "train_nonholdout_rows": train_nonholdout_rows,
        "train_nonholdout_turn0_rows": int((train_nonholdout_mask & train_turn0).sum()),
        "blend4_group_macro_f1": group_f1,
        "delta_vs_overall_4way": delta,
        "screen_pass": bool(screen_pass),
    }


def build_cross_specs(
    *,
    base_specs: Sequence[dict[str, Any]],
    base_rows: Sequence[dict[str, Any]],
    ctx: DataContext,
    league: dict[str, Any],
    max_cross: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    spec_by_name = {candidate_name(spec): spec for spec in base_specs}
    weak_rows = [
        row
        for row in base_rows
        if row.get("screen_pass")
        and row.get("family") not in {"workspace_loc", "turn_index", "history_len"}
    ]
    weak_rows = sorted(weak_rows, key=lambda row: (float(row["delta_vs_overall_4way"]), -int(row["holdout_rows"])))
    cross_specs: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for i, left in enumerate(weak_rows[:8]):
        for right in weak_rows[i + 1 : 8]:
            if left["family"] == right["family"]:
                continue
            parts = [spec_by_name[str(left["name"])], spec_by_name[str(right["name"])]]
            name = "cross:" + "&".join(candidate_name(part) for part in parts)
            if name in seen:
                continue
            seen.add(name)
            spec = {"kind": "and", "parts": parts}
            row = score_candidate(spec=spec, ctx=ctx, league=league)
            if row["holdout_rows"] >= SCREEN_MIN_HOLDOUT and row["train_nonholdout_rows"] >= SCREEN_MIN_TRAIN:
                cross_specs.append(spec)
                cross_rows.append(row)
            if len(cross_specs) >= max_cross:
                return cross_specs, cross_rows
    return cross_specs, cross_rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def save_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        print(f"[save] {path}")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            safe_row = dict(row)
            if isinstance(safe_row.get("spec"), (dict, list)):
                safe_row["spec"] = json.dumps(safe_row["spec"], ensure_ascii=False, sort_keys=True)
            writer.writerow(safe_row)
    print(f"[save] {path}")
