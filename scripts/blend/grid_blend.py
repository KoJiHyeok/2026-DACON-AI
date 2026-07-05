# -*- coding: utf-8 -*-
"""Grid-search probability blend weights on aligned holdout NPZ files."""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score


DEFAULT_ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    ids = d["ids"].astype(str)
    probs = np.asarray(d["probs"], dtype=np.float64)
    y_true = d["y_true"].astype(str)
    actions = [str(x) for x in d["actions"]] if "actions" in d.files else list(DEFAULT_ACTIONS)
    meta = {}
    if "meta_json" in d.files:
        meta = json.loads(str(np.asarray(d["meta_json"]).item()))
    return {"path": path, "ids": ids, "probs": probs, "y_true": y_true, "actions": actions, "meta": meta}


def align_probs(probs: np.ndarray, src_actions: list[str], dst_actions: list[str]) -> np.ndarray:
    missing = sorted(set(dst_actions) - set(src_actions))
    if missing:
        raise ValueError(f"missing classes in npz: {missing}")
    idx = [src_actions.index(a) for a in dst_actions]
    return probs[:, idx]


def reorder_to_reference(item, ref_ids: np.ndarray, ref_y: np.ndarray):
    pos = {sample_id: i for i, sample_id in enumerate(item["ids"])}
    ref_set = set(ref_ids)
    missing = [sample_id for sample_id in ref_ids if sample_id not in pos]
    extra = [sample_id for sample_id in item["ids"] if sample_id not in ref_set]
    if missing or extra:
        raise ValueError(
            f"{item['path']} id mismatch: missing={len(missing)} extra={len(extra)}"
        )
    order = np.array([pos[sample_id] for sample_id in ref_ids])
    y = item["y_true"][order]
    if not np.array_equal(y, ref_y):
        raise ValueError(f"{item['path']} y_true mismatch after id alignment")
    return item["probs"][order]


def score(probs: np.ndarray, y_true: np.ndarray, actions: list[str]) -> float:
    preds = np.array(actions)[probs.argmax(axis=1)]
    return float(f1_score(y_true, preds, average="macro", zero_division=0))


def normalized_blend(mats: list[np.ndarray], weights: tuple[float, ...]) -> np.ndarray:
    denom = float(sum(weights))
    if denom <= 0:
        raise ValueError("weight sum must be positive")
    out = np.zeros_like(mats[0], dtype=np.float64)
    for mat, weight in zip(mats, weights):
        out += float(weight) * mat
    return out / denom


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("npz", nargs="+", type=Path, help="holdout probability npz files")
    p.add_argument("--names", default=None, help="comma-separated names; defaults to file stems")
    p.add_argument("--min-weight", type=float, default=0.0)
    p.add_argument("--max-weight", type=float, default=3.0)
    p.add_argument("--step", type=float, default=0.25)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--actions", default=",".join(DEFAULT_ACTIONS))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    actions = [x.strip() for x in args.actions.split(",") if x.strip()]
    items = [load_npz(path) for path in args.npz]
    names = (
        [x.strip() for x in args.names.split(",")]
        if args.names else [item["path"].stem.replace("holdout_", "") for item in items]
    )
    if len(names) != len(items):
        raise ValueError("--names count must match npz count")

    ref_ids = items[0]["ids"]
    ref_y = items[0]["y_true"]
    mats = []
    for item in items:
        probs = reorder_to_reference(item, ref_ids, ref_y)
        probs = align_probs(probs, item["actions"], actions)
        mats.append(probs)

    print(f"[data] rows={len(ref_y)} components={names}")
    for name, mat in zip(names, mats):
        print(f"[single] {name}: macro_f1={score(mat, ref_y, actions):.6f}")

    if any("enc" in name.lower() or "encoder" in str(item["path"]).lower() for name, item in zip(names, items)):
        print(
            "[warning] encoder LB weights are not reliable unless encoder probabilities "
            "come from the same grouped holdout split and label/action order is aligned."
        )

    grid = np.round(
        np.arange(args.min_weight, args.max_weight + args.step / 2.0, args.step), 10
    )
    rows = []
    for weights in itertools.product(grid, repeat=len(mats)):
        weights = tuple(float(w) for w in weights)
        if sum(weights) <= 0:
            continue
        blended = normalized_blend(mats, weights)
        rows.append({
            "weights": weights,
            "macro_f1": score(blended, ref_y, actions),
        })
    rows.sort(key=lambda r: (-r["macro_f1"], r["weights"]))
    top = rows[: args.top_k]

    print(f"[grid] searched={len(rows)} step={args.step:g} range=[{args.min_weight:g},{args.max_weight:g}]")
    print("[top]")
    for rank, row in enumerate(top, 1):
        weight_str = ",".join(f"{w:g}" for w in row["weights"])
        print(f"{rank:02d}\tmacro_f1={row['macro_f1']:.6f}\tweights=[{weight_str}]")

    baseline = tuple(1.0 for _ in mats)
    baseline_score = score(normalized_blend(mats, baseline), ref_y, actions)
    print(f"[baseline] weights={[1.0 for _ in mats]} macro_f1={baseline_score:.6f}")

    result = {
        "components": names,
        "paths": [str(item["path"]) for item in items],
        "rows": int(len(ref_y)),
        "actions": actions,
        "grid": {"min": args.min_weight, "max": args.max_weight, "step": args.step},
        "single": {name: score(mat, ref_y, actions) for name, mat in zip(names, mats)},
        "baseline_equal_weights_macro_f1": baseline_score,
        "top": [{"rank": i + 1, "macro_f1": r["macro_f1"], "weights": list(r["weights"])} for i, r in enumerate(top)],
        "meta": {name: item["meta"] for name, item in zip(names, items)},
        "warning": (
            "Do not over-trust LB-derived encoder weights unless encoder probabilities "
            "were collected on the same grouped holdout IDs."
        ),
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[save] {args.out}")


if __name__ == "__main__":
    main()
