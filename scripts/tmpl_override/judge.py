# -*- coding: utf-8 -*-
"""Judge mined template overrides against the 4-way + soft-AU holdout league."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import numpy as np

import common as tmpl_common


def load_league_common() -> Any:
    path = tmpl_common.ROOT / "scripts" / "league4" / "common.py"
    spec = importlib.util.spec_from_file_location("league4_common_for_tmpl_override", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load league common from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=tmpl_common.DATA_DIR)
    parser.add_argument("--holdout-base", type=Path, default=tmpl_common.HOLDOUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=tmpl_common.OUT_DIR)
    parser.add_argument("--normalizer", choices=["r1", "r1_lower"], default="r1")
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--purity", type=float, default=0.995)
    parser.add_argument("--include-respond-only", action="store_true")
    parser.add_argument("--force-au", action="store_true")
    return parser.parse_args()


def score_labels(league_common: Any, data: Any, pred: np.ndarray) -> dict[str, float]:
    return {
        "macro_f1": league_common.macro_f1_labels(data.y_true, pred, data.actions),
        "au_macro_f1": league_common.macro_f1_labels(data.y_true[data.au_mask], pred[data.au_mask], data.actions),
        "non_au_macro_f1": league_common.macro_f1_labels(
            data.y_true[data.non_au_mask], pred[data.non_au_mask], data.actions
        ),
    }


def evaluate_override(
    league_common: Any,
    data: Any,
    base_pred: np.ndarray,
    templates: np.ndarray,
    template: str,
    action: str,
) -> dict[str, Any]:
    target_mask = (templates == template) & data.non_au_mask
    changed_mask = target_mask & (base_pred != action)
    pred = base_pred.copy()
    pred[changed_mask] = action
    y = data.y_true
    fixed = int(((base_pred != y) & (pred == y) & changed_mask).sum())
    broken = int(((base_pred == y) & (pred != y) & changed_mask).sum())
    wrong_to_wrong = int(((base_pred != y) & (pred != y) & changed_mask).sum())
    score = league_common.macro_f1_labels(y, pred, data.actions)
    return {
        "holdout_non_au_matches": int(target_mask.sum()),
        "changed_rows": int(changed_mask.sum()),
        "fixed": fixed,
        "broken": broken,
        "wrong_to_wrong": wrong_to_wrong,
        "macro_f1": score,
    }


def apply_combined(
    league_common: Any,
    data: Any,
    base_pred: np.ndarray,
    templates: np.ndarray,
    selected: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, Any]]:
    pred = base_pred.copy()
    changed_mask = np.zeros(len(pred), dtype=bool)
    for row in selected:
        action = str(row["top_action"])
        mask = (templates == str(row["template"])) & data.non_au_mask & (pred != action)
        pred[mask] = action
        changed_mask |= mask
    y = data.y_true
    return pred, {
        "templates": len(selected),
        "changed_rows": int(changed_mask.sum()),
        "fixed": int(((base_pred != y) & (pred == y) & changed_mask).sum()),
        "broken": int(((base_pred == y) & (pred != y) & changed_mask).sum()),
        "wrong_to_wrong": int(((base_pred != y) & (pred != y) & changed_mask).sum()),
        **score_labels(league_common, data, pred),
    }


def main() -> None:
    args = parse_args()
    league_common = load_league_common()

    rows = tmpl_common.load_train_records(args.data_dir)
    holdout_ids = tmpl_common.load_holdout_ids(args.holdout_base)
    holdout_rows = [row for row in rows if row["id"] in holdout_ids]
    nonholdout_rows = [row for row in rows if row["id"] not in holdout_ids]
    table = tmpl_common.merge_holdout_counts(
        tmpl_common.summarize_records(nonholdout_rows, args.normalizer, min_rows=1),
        tmpl_common.summarize_records(holdout_rows, args.normalizer, min_rows=1),
    )
    mine_candidates = tmpl_common.filter_mine_candidates(
        table,
        min_nonholdout_n=args.min_n,
        min_purity=args.purity,
        exclude_respond_only=not args.include_respond_only,
    )

    print("[load] league components")
    data = league_common.load_league_data()
    au = league_common.train_or_load_au_probs(data, league_common.OUT_DIR, force=args.force_au)
    four = league_common.four_way_blend(data, league_common.BASE_E5_WEIGHT, league_common.BASE_MBERT_WEIGHT)
    base_probs = league_common.apply_soft_au(data, four, au["probs"], league_common.DEFAULT_ALPHA)
    base_pred = league_common.predict_from_probs(base_probs, data.actions)
    base_scores = score_labels(league_common, data, base_pred)
    holdout_templates = np.asarray(
        [
            tmpl_common.normalize_prompt(
                str(data.samples_by_id[str(sample_id)].get("current_prompt", "")),
                args.normalizer,
            )
            for sample_id in data.ids
        ],
        dtype=object,
    )

    judged: list[dict[str, Any]] = []
    for candidate in mine_candidates:
        result = evaluate_override(
            league_common,
            data,
            base_pred,
            holdout_templates,
            str(candidate["template"]),
            str(candidate["top_action"]),
        )
        delta = float(result["macro_f1"] - base_scores["macro_f1"])
        row = {
            **candidate,
            **result,
            "delta_vs_base": delta,
            "passes_judge_gate": bool(result["changed_rows"] > 0),
            "passes_safe_gate": bool(
                result["changed_rows"] > 0
                and result["fixed"] > 0
                and result["broken"] == 0
                and delta > 0
            ),
        }
        judged.append(row)
    judged.sort(
        key=lambda r: (
            not bool(r["passes_safe_gate"]),
            -float(r["delta_vs_base"]),
            -int(r["fixed"]),
            int(r["broken"]),
            -int(r["changed_rows"]),
        )
    )

    judge_gate = [row for row in judged if bool(row["passes_judge_gate"])]
    safe_gate = [row for row in judged if bool(row["passes_safe_gate"])]
    _, combined_judge = apply_combined(league_common, data, base_pred, holdout_templates, judge_gate)
    _, combined_safe = apply_combined(league_common, data, base_pred, holdout_templates, safe_gate)
    combined_judge["delta_vs_base"] = combined_judge["macro_f1"] - base_scores["macro_f1"]
    combined_safe["delta_vs_base"] = combined_safe["macro_f1"] - base_scores["macro_f1"]

    summary = {
        "normalizer": args.normalizer,
        "gate": {
            "min_nonholdout_n": args.min_n,
            "min_purity": args.purity,
            "exclude_respond_only": not args.include_respond_only,
            "exclude_au_targets": True,
            "safe_gate": "changed_rows>0, fixed>0, broken==0, individual macro delta>0",
        },
        "rows": {
            "train": len(rows),
            "holdout": len(holdout_rows),
            "nonholdout": len(nonholdout_rows),
            "league_holdout": int(len(data.ids)),
            "league_holdout_au": int(data.au_mask.sum()),
            "league_holdout_non_au": int(data.non_au_mask.sum()),
        },
        "baseline": {
            **base_scores,
            "expected_b4_soft_au_from_task": 0.73877,
            "au_cache_hit": bool(au["cache_hit"]),
            "au_model": au["meta"],
        },
        "candidate_counts": {
            "mine_candidates": len(mine_candidates),
            "judge_gate_templates": len(judge_gate),
            "safe_gate_templates": len(safe_gate),
        },
        "combined_judge_gate": combined_judge,
        "combined_safe_gate": combined_safe,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tmpl_common.write_json(args.out_dir / "judge_summary.json", summary)
    tmpl_common.write_csv(args.out_dir / "judge_templates.csv", judged)
    tmpl_common.write_csv(args.out_dir / "judge_safe_templates.csv", safe_gate)

    print(
        "[base] "
        f"B4+softAU={base_scores['macro_f1']:.6f} "
        f"au={base_scores['au_macro_f1']:.6f} nonAU={base_scores['non_au_macro_f1']:.6f}"
    )
    print(
        "[judge] "
        f"mine={len(mine_candidates)} changed_templates={len(judge_gate)} "
        f"safe_templates={len(safe_gate)} "
        f"safe_delta={combined_safe['delta_vs_base']:+.6f} "
        f"safe_fixed={combined_safe['fixed']} safe_broken={combined_safe['broken']}"
    )


if __name__ == "__main__":
    main()
