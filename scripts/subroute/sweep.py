# -*- coding: utf-8 -*-
"""First-pass subpopulation sweep for task3.

Outputs group sizes and 4-way blend macro-F1 for candidate routing groups.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from common import (
    DATA_DIR,
    DEFAULT_ALPHAS,
    HOLDOUT_BASE,
    MBERT_HOLDOUT,
    OOF_DIR,
    OUT_DIR,
    SCREEN_MIN_HOLDOUT,
    SCREEN_MIN_TRAIN,
    SCREEN_WEAK_DELTA,
    build_base_candidate_specs,
    build_context,
    build_cross_specs,
    load_league,
    load_train,
    save_csv,
    save_json,
    score_candidate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--holdout-base", type=Path, default=HOLDOUT_BASE)
    parser.add_argument("--mbert-holdout", type=Path, default=MBERT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--min-holdout", type=int, default=SCREEN_MIN_HOLDOUT)
    parser.add_argument("--min-train", type=int, default=SCREEN_MIN_TRAIN)
    parser.add_argument("--weak-delta", type=float, default=SCREEN_WEAK_DELTA)
    parser.add_argument("--sanity-only", action="store_true")
    return parser.parse_args()


def run_sweep(args: argparse.Namespace) -> dict[str, Any]:
    print("[load] train and 4-way league components")
    samples, ids, y, groups = load_train(args.data_dir)
    league = load_league(args.holdout_base, args.oof_dir, args.mbert_holdout)
    ctx = build_context(samples, ids, y, groups, league["ids"])
    print(
        "[sanity] 3way={:.8f} 4way={:.8f} holdout={} non_au_holdout={}".format(
            league["blend3_score"],
            league["blend4_score"],
            len(league["ids"]),
            int(ctx.holdout_non_au_mask.sum()),
        )
    )

    summary: dict[str, Any] = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "oof_dir": str(args.oof_dir),
            "holdout_base": str(args.holdout_base),
            "mbert_holdout": str(args.mbert_holdout),
            "alphas_for_probe": list(DEFAULT_ALPHAS),
        },
        "screen": {
            "weak_delta_vs_overall_4way_lte": args.weak_delta,
            "min_holdout_rows": args.min_holdout,
            "min_train_nonholdout_rows": args.min_train,
            "sess_au_excluded_from_all_candidates": True,
            "turn_index_0_route_excluded": True,
        },
        "baseline": {
            "holdout_rows": int(len(league["ids"])),
            "holdout_non_au_rows": int(ctx.holdout_non_au_mask.sum()),
            "train_rows": int(len(ids)),
            "train_nonholdout_rows": int(ctx.nonholdout_mask.sum()),
            "train_nonholdout_non_au_rows": int((ctx.nonholdout_mask & ctx.train_non_au_mask).sum()),
            "blend3_macro_f1": float(league["blend3_score"]),
            "blend4_macro_f1": float(league["blend4_score"]),
        },
        "rows": [],
        "screen_pass_rows": [],
        "cross_rows": [],
    }
    if args.sanity_only:
        return summary

    base_specs = build_base_candidate_specs(samples)
    base_rows = [
        score_candidate(
            spec=spec,
            ctx=ctx,
            league=league,
            screen_weak_delta=args.weak_delta,
            screen_min_holdout=args.min_holdout,
            screen_min_train=args.min_train,
        )
        for spec in base_specs
    ]
    cross_specs, cross_rows = build_cross_specs(base_specs=base_specs, base_rows=base_rows, ctx=ctx, league=league)
    del cross_specs
    rows = sorted(base_rows + cross_rows, key=lambda row: (float(row["delta_vs_overall_4way"]), -int(row["holdout_rows"])))
    summary["rows"] = rows
    summary["screen_pass_rows"] = [row for row in rows if row["screen_pass"]]
    summary["cross_rows"] = cross_rows
    return summary


def main() -> None:
    args = parse_args()
    summary = run_sweep(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.sanity_only:
        save_json(args.out_dir / "sanity.json", summary)
        print("[done] sanity only")
        return
    save_json(args.out_dir / "sweep.json", summary)
    save_csv(args.out_dir / "sweep_rows.csv", summary["rows"])
    save_csv(args.out_dir / "sweep_screen_pass.csv", summary["screen_pass_rows"])
    print(
        "[done] rows={} screen_pass={}".format(
            len(summary["rows"]),
            len(summary["screen_pass_rows"]),
        )
    )


if __name__ == "__main__":
    main()
