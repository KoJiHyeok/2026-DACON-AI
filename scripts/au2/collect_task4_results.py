# -*- coding: utf-8 -*-
"""Collect task4 partial JSON outputs into final report tables."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

THIS = Path(__file__).resolve()
ROOT = THIS.parents[2]
sys.path.insert(0, str(THIS.parent))
import task4_grid as grid  # noqa: E402


OUT_DIR = ROOT / "night_out" / "task4"


def load_json(name: str) -> dict[str, Any]:
    with (OUT_DIR / name).open(encoding="utf-8") as f:
        return json.load(f)


def write_json(name: str, value: Any) -> None:
    path = OUT_DIR / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
    path = OUT_DIR / name
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {path}")


def run_lookup(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(run["variant"]): run for run in runs}


def recompute_per_class(best: dict[str, Any], run: dict[str, Any]) -> list[dict[str, Any]]:
    samples, ids, y, _groups = grid.load_train(grid.DATA_DIR)
    holdout = grid.load_holdout(grid.HOLDOUT_BASE, grid.OOF_DIR)
    holdout_id_set = set(str(x) for x in holdout["ids"])
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    au_mask = np.asarray([grid.au_route.is_au(str(sample_id)) for sample_id in holdout["ids"]], dtype=bool)
    holdout_au_ids = [str(x) for x in holdout["ids"][au_mask]]
    holdout_au_samples = [sample_by_id[sample_id] for sample_id in holdout_au_ids]
    nonholdout = np.asarray([str(sample_id) not in holdout_id_set for sample_id in ids], dtype=bool)

    if str(best["variant"]).startswith("au_only_"):
        train_idx = np.asarray(
            [i for i, sample_id in enumerate(ids) if nonholdout[i] and grid.au_route.is_au(str(sample_id))],
            dtype=np.int64,
        )
        sample_weight = None
    else:
        train_idx = np.asarray([i for i in range(len(ids)) if nonholdout[i]], dtype=np.int64)
        sample_weight = np.ones(len(train_idx), dtype=np.float64)
        au_weight = float(run.get("au_sample_weight", 1.0))
        for pos, original_idx in enumerate(train_idx):
            if grid.au_route.is_au(samples[int(original_idx)].get("id", "")):
                sample_weight[pos] = au_weight

    result = grid.fit_predict(
        [samples[int(i)] for i in train_idx],
        y[train_idx],
        holdout_au_samples,
        feature_kind=str(run["feature_kind"]),
        c_value=float(run["c"]),
        sample_weight=sample_weight,
        seed=42,
    )
    au_probs = grid.align_probs(result["probs"], grid.ACTIONS, holdout["actions"])
    alpha = float(best["alpha"])
    mixed = alpha * au_probs + (1.0 - alpha) * holdout["blend"][au_mask]
    pred_au = grid.predict_from_probs(mixed, holdout["actions"])
    return grid.per_class_rows(
        name=f"{best['variant']}|alpha={grid.alpha_key(alpha)}",
        pred_au=pred_au,
        blend_pred_au=holdout["blend_pred"][au_mask],
        y_au=holdout["y_true"][au_mask],
        actions=holdout["actions"],
    )


def main() -> None:
    soft = load_json("soft_alpha.json")
    au_grid = load_json("au_grid_partial.json")
    sim_weight = load_json("sim_weight_partial.json")

    runs = list(au_grid.get("runs", [])) + list(sim_weight.get("runs", []))
    route_rows = list(au_grid.get("route_rows", [])) + list(sim_weight.get("route_rows", []))
    route_rows_sorted = sorted(route_rows, key=lambda row: float(row["league_macro_f1"]), reverse=True)
    best = route_rows_sorted[0]
    runs_by_variant = run_lookup(runs)
    per_class = recompute_per_class(best, runs_by_variant[str(best["variant"])])
    per_class_sorted = sorted(per_class, key=lambda row: float(row["delta_f1"]), reverse=True)

    hard_reference = next(
        row
        for row in route_rows
        if row["variant"] == "au_only_word_char_C0.5" and abs(float(row["alpha"]) - 1.0) < 1e-12
    )
    same_variant_hard = next(
        row
        for row in route_rows
        if row["variant"] == best["variant"] and abs(float(row["alpha"]) - 1.0) < 1e-12
    )

    summary = {
        "inputs": au_grid["inputs"],
        "split": au_grid["split"],
        "baseline": au_grid["baseline"],
        "soft_axis_file": "soft_alpha.json",
        "au_grid_file": "au_grid_partial.json",
        "sim_weight_file": "sim_weight_partial.json",
        "runs": runs,
        "route_rows": route_rows_sorted,
        "best": {
            **best,
            "passes_requested_gate": bool(float(best["delta_vs_task3_hard"]) >= grid.PASS_THRESHOLD),
            "decision_line": grid.PASS_BASELINE + grid.PASS_THRESHOLD,
            "delta_vs_isolated_word_char_hard": float(best["league_macro_f1"])
            - float(hard_reference["league_macro_f1"]),
            "delta_vs_same_variant_hard": float(best["league_macro_f1"])
            - float(same_variant_hard["league_macro_f1"]),
        },
        "references": {
            "soft_axis_best": max(soft["route_rows"], key=lambda row: float(row["league_macro_f1"])),
            "isolated_word_char_hard": hard_reference,
            "same_variant_hard": same_variant_hard,
        },
        "per_class_best_vs_blend": per_class_sorted,
    }

    write_json("summary_all.json", summary)
    write_csv("route_rows.csv", route_rows_sorted)
    write_csv("per_class_best_vs_blend.csv", per_class_sorted)
    print(
        "[done] best={variant} alpha={alpha:g} score={score:.6f} "
        "delta_task3={delta:+.6f}".format(
            variant=best["variant"],
            alpha=float(best["alpha"]),
            score=float(best["league_macro_f1"]),
            delta=float(best["delta_vs_task3_hard"]),
        )
    )


if __name__ == "__main__":
    main()
