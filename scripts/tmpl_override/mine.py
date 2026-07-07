# -*- coding: utf-8 -*-
"""Mine high-purity current_prompt templates without holdout leakage."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=common.DATA_DIR)
    parser.add_argument("--holdout-base", type=Path, default=common.HOLDOUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=common.OUT_DIR)
    parser.add_argument("--normalizer", choices=["r1", "r1_lower"], default="r1")
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--purity", type=float, default=0.995)
    parser.add_argument("--include-respond-only", action="store_true")
    return parser.parse_args()


def high_purity_summary(stats: list[dict], total_rows: int, threshold: float = 0.99) -> dict:
    selected = [row for row in stats if int(row["n_rows"]) >= 2 and float(row["purity"]) >= threshold]
    top_dist = Counter(str(row["top_action"]) for row in selected)
    rows_by_action = Counter()
    for row in selected:
        rows_by_action[str(row["top_action"])] += int(row["n_rows"])
    non_respond_rows = sum(int(row["n_rows"]) for row in selected if str(row["top_action"]) != "respond_only")
    return {
        "threshold": threshold,
        "templates": len(selected),
        "rows": sum(int(row["n_rows"]) for row in selected),
        "coverage": sum(int(row["n_rows"]) for row in selected) / total_rows,
        "sessions": None,
        "non_respond_only_rows": non_respond_rows,
        "non_respond_only_coverage": non_respond_rows / total_rows,
        "templates_by_top_action": dict(sorted(top_dist.items())),
        "rows_by_top_action": dict(sorted(rows_by_action.items())),
    }


def main() -> None:
    args = parse_args()
    rows = common.load_train_records(args.data_dir)
    holdout_ids = common.load_holdout_ids(args.holdout_base)
    holdout_rows = [row for row in rows if row["id"] in holdout_ids]
    nonholdout_rows = [row for row in rows if row["id"] not in holdout_ids]
    if len(rows) != len(holdout_rows) + len(nonholdout_rows):
        raise AssertionError("holdout split accounting failed")

    all_stats = common.summarize_records(rows, args.normalizer, min_rows=1)
    nonholdout_stats = common.summarize_records(nonholdout_rows, args.normalizer, min_rows=1)
    holdout_stats = common.summarize_records(holdout_rows, args.normalizer, min_rows=1)
    table = common.merge_holdout_counts(nonholdout_stats, holdout_stats)
    candidates = common.filter_mine_candidates(
        table,
        min_nonholdout_n=args.min_n,
        min_purity=args.purity,
        exclude_respond_only=not args.include_respond_only,
    )

    duplicate_all = [row for row in all_stats if int(row["n_rows"]) > 1]
    duplicate_nonholdout = [row for row in nonholdout_stats if int(row["n_rows"]) > 1]
    summary = {
        "normalizer": args.normalizer,
        "paths": {
            "data_dir": str(args.data_dir),
            "holdout_base": str(args.holdout_base),
            "out_dir": str(args.out_dir),
        },
        "rows": {
            "train": len(rows),
            "holdout": len(holdout_rows),
            "nonholdout": len(nonholdout_rows),
        },
        "all_train_repro_check": {
            "unique_templates": len(all_stats),
            "duplicate_templates": len(duplicate_all),
            "rows_in_duplicate_templates": sum(int(row["n_rows"]) for row in duplicate_all),
            "purity_ge_0_99": high_purity_summary(all_stats, len(rows), threshold=0.99),
        },
        "nonholdout_mining": {
            "unique_templates": len(nonholdout_stats),
            "duplicate_templates": len(duplicate_nonholdout),
            "rows_in_duplicate_templates": sum(int(row["n_rows"]) for row in duplicate_nonholdout),
            "purity_ge_0_99": high_purity_summary(nonholdout_stats, len(nonholdout_rows), threshold=0.99),
            "mine_gate": {
                "min_nonholdout_n": args.min_n,
                "min_purity": args.purity,
                "exclude_respond_only": not args.include_respond_only,
                "templates": len(candidates),
                "nonholdout_rows": sum(int(row["nonholdout_n"]) for row in candidates),
                "holdout_rows": sum(int(row["holdout_n"]) for row in candidates),
                "templates_with_holdout_rows": sum(1 for row in candidates if int(row["holdout_n"]) > 0),
            },
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    common.write_json(args.out_dir / "mine_summary.json", summary)
    common.write_csv(args.out_dir / "template_stats.csv", table)
    common.write_csv(args.out_dir / "mine_candidates.csv", candidates)

    repro = summary["all_train_repro_check"]["purity_ge_0_99"]
    gate = summary["nonholdout_mining"]["mine_gate"]
    print(
        "[repro] all train purity>=0.99: "
        f"{repro['templates']} templates, {repro['rows']} rows, "
        f"non-respond_only={repro['non_respond_only_rows']}"
    )
    print(
        "[mine] "
        f"holdout={len(holdout_rows)} nonholdout={len(nonholdout_rows)} "
        f"gate_templates={gate['templates']} holdout_rows={gate['holdout_rows']}"
    )


if __name__ == "__main__":
    main()
