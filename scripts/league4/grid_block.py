# -*- coding: utf-8 -*-
"""Grid mBERT/e5 weights inside the encoder block with total encoder weight fixed at 2."""
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
    data = common.load_league_data()
    au = common.train_or_load_au_probs(data, args.out_dir, force=args.force_au)

    base_blend = common.four_way_blend(data, common.BASE_E5_WEIGHT, common.BASE_MBERT_WEIGHT)
    base_soft = common.apply_soft_au(data, base_blend, au["probs"], common.DEFAULT_ALPHA)
    base_scores = common.score_bundle(data, base_soft)
    base_raw = common.score_bundle(data, base_blend)
    b4 = base_scores["macro_f1"]

    rows = []
    for i in range(60, 101, 5):
        mbert_weight = i / 100.0
        e5_weight = 2.0 - mbert_weight
        raw = common.four_way_blend(data, e5_weight, mbert_weight)
        soft = common.apply_soft_au(data, raw, au["probs"], common.DEFAULT_ALPHA)
        scores_raw = common.score_bundle(data, raw, prefix="raw_")
        scores_soft = common.score_bundle(data, soft, prefix="soft_au_")
        halves = common.half_scores(data, soft)
        row = {
            "e5_weight": round(e5_weight, 2),
            "mbert_weight": round(mbert_weight, 2),
            "alpha": common.DEFAULT_ALPHA,
            "raw_macro_f1": scores_raw["raw_macro_f1"],
            "raw_non_au_macro_f1": scores_raw["raw_non_au_macro_f1"],
            "soft_au_macro_f1": scores_soft["soft_au_macro_f1"],
            "delta_vs_b4": scores_soft["soft_au_macro_f1"] - b4,
            "delta_raw_vs_current_raw": scores_raw["raw_macro_f1"] - base_raw["macro_f1"],
            "delta_non_au_vs_current": scores_raw["raw_non_au_macro_f1"] - base_raw["non_au_macro_f1"],
            **halves,
            "half_gap_abs": abs(halves["half1_macro_f1"] - halves["half2_macro_f1"]),
        }
        rows.append(row)

    best = max(rows, key=lambda r: (r["soft_au_macro_f1"], r["raw_non_au_macro_f1"]))
    payload = {
        "baseline": {
            "e5_weight": common.BASE_E5_WEIGHT,
            "mbert_weight": common.BASE_MBERT_WEIGHT,
            "alpha": common.DEFAULT_ALPHA,
            "b4_soft_au_macro_f1": b4,
            **base_scores,
            "raw_current_macro_f1": base_raw["macro_f1"],
            "raw_current_non_au_macro_f1": base_raw["non_au_macro_f1"],
        },
        "au_model": au["meta"],
        "rows": rows,
        "best": best,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    common.write_json(args.out_dir / "grid_block.json", payload)
    common.write_csv(args.out_dir / "grid_block.csv", rows)
    print("[grid_block]")
    for row in rows:
        print(
            "  e5={e5_weight:.2f} mbert={mbert_weight:.2f} raw={raw_macro_f1:.6f} "
            "nonAU={raw_non_au_macro_f1:.6f} final={soft_au_macro_f1:.6f} "
            "dB4={delta_vs_b4:+.6f} h1={half1_macro_f1:.6f} h2={half2_macro_f1:.6f}".format(**row)
        )
    print(
        "[best] e5={e5_weight:.2f} mbert={mbert_weight:.2f} final={soft_au_macro_f1:.6f} "
        "delta={delta_vs_b4:+.6f}".format(**best)
    )


if __name__ == "__main__":
    main()
