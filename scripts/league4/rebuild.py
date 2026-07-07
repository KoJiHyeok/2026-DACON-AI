# -*- coding: utf-8 -*-
"""Rebuild the 4-way local league baseline and current soft-AU mirror."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=common.OUT_DIR)
    parser.add_argument("--force-au", action="store_true", help="Retrain AU holdout probabilities even if cached.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    print("[load] league components")
    data = common.load_league_data()
    print(f"[data] rows={len(data.ids)} au={int(data.au_mask.sum())} non_au={int(data.non_au_mask.sum())}")

    print("[au] char_wb C=1 nonholdout-AU probabilities")
    au = common.train_or_load_au_probs(data, args.out_dir, force=args.force_au)
    print(f"[au] cache_hit={au['cache_hit']} train_rows={au['meta'].get('train_rows')}")

    three = common.three_way_blend(data)
    four = common.four_way_blend(data)
    four_soft = common.apply_soft_au(data, four, au["probs"], common.DEFAULT_ALPHA)

    rows = [
        {"name": "3way_e5x2", **common.score_bundle(data, three)},
        {"name": "4way_e5_1.2_mbert_0.8", **common.score_bundle(data, four)},
        {
            "name": "4way_e5_1.2_mbert_0.8_soft_au_a0.9",
            **common.score_bundle(data, four_soft),
            **common.half_scores(data, four_soft),
        },
    ]
    baseline = rows[-1]["macro_f1"]
    payload = {
        "inputs": {
            "holdout_base": str(common.HOLDOUT_BASE),
            "mbert_holdout": str(common.MBERT_HOLDOUT),
            "oof_dir": str(common.OOF_DIR),
            "data_dir": str(common.DATA_DIR),
            "alpha": common.DEFAULT_ALPHA,
            "e5_weight": common.BASE_E5_WEIGHT,
            "mbert_weight": common.BASE_MBERT_WEIGHT,
        },
        "sanity": {
            "expected_3way": common.EXPECTED_3WAY,
            "expected_4way": common.EXPECTED_4WAY,
            "actual_3way": rows[0]["macro_f1"],
            "actual_4way": rows[1]["macro_f1"],
        },
        "au_model": au["meta"],
        "rows": rows,
        "baseline_b4_soft_au": baseline,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    common.write_json(args.out_dir / "rebuild.json", payload)
    common.write_csv(args.out_dir / "rebuild.csv", rows)
    print("[summary]")
    for row in rows:
        print(
            "  {name}: all={macro_f1:.6f} au={au_macro_f1:.6f} non_au={non_au_macro_f1:.6f}".format(**row)
        )
    print(f"[done] B4={baseline:.6f} elapsed={payload['elapsed_sec']:.1f}s")


if __name__ == "__main__":
    main()
