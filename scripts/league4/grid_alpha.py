# -*- coding: utf-8 -*-
"""Grid soft-AU alpha on top of the current 4-way blend."""
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
    blend = common.four_way_blend(data, common.BASE_E5_WEIGHT, common.BASE_MBERT_WEIGHT)
    base_soft = common.apply_soft_au(data, blend, au["probs"], common.DEFAULT_ALPHA)
    b4 = common.macro_f1_probs(base_soft, data.y_true, data.actions)
    raw_scores = common.score_bundle(data, blend)

    rows = []
    for i in range(70, 101, 5):
        alpha = i / 100.0
        soft = common.apply_soft_au(data, blend, au["probs"], alpha)
        scores = common.score_bundle(data, soft)
        blend_pred_au = common.predict_from_probs(blend[data.au_mask], data.actions)
        soft_pred_au = common.predict_from_probs(soft[data.au_mask], data.actions)
        row = {
            "alpha": round(alpha, 2),
            **scores,
            "delta_vs_b4": scores["macro_f1"] - b4,
            "changed_au_vs_4way": int((blend_pred_au != soft_pred_au).sum()),
            **common.half_scores(data, soft),
        }
        row["half_gap_abs"] = abs(row["half1_macro_f1"] - row["half2_macro_f1"])
        rows.append(row)

    best = max(rows, key=lambda r: (r["macro_f1"], r["au_macro_f1"]))
    payload = {
        "baseline": {
            "e5_weight": common.BASE_E5_WEIGHT,
            "mbert_weight": common.BASE_MBERT_WEIGHT,
            "alpha": common.DEFAULT_ALPHA,
            "b4_soft_au_macro_f1": b4,
            "raw_4way_macro_f1": raw_scores["macro_f1"],
            "raw_4way_au_macro_f1": raw_scores["au_macro_f1"],
            "raw_4way_non_au_macro_f1": raw_scores["non_au_macro_f1"],
        },
        "au_model": au["meta"],
        "rows": rows,
        "best": best,
        "elapsed_sec": round(time.time() - t0, 3),
    }
    common.write_json(args.out_dir / "grid_alpha.json", payload)
    common.write_csv(args.out_dir / "grid_alpha.csv", rows)
    print("[grid_alpha]")
    for row in rows:
        print(
            "  alpha={alpha:.2f} all={macro_f1:.6f} au={au_macro_f1:.6f} "
            "nonAU={non_au_macro_f1:.6f} dB4={delta_vs_b4:+.6f} changedAU={changed_au_vs_4way}".format(**row)
        )
    print("[best] alpha={alpha:.2f} final={macro_f1:.6f} delta={delta_vs_b4:+.6f}".format(**best))


if __name__ == "__main__":
    main()
