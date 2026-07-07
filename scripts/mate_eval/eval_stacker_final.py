# -*- coding: utf-8 -*-
"""Evaluate teammate-style stacker-as-final variants on the fixed holdout."""
from __future__ import annotations

import argparse
import time
from typing import Any

import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common.add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    holdout = common.load_components(
        holdout_base=args.holdout_base,
        oof_dir=args.oof_dir,
        mbert_holdout=args.mbert_holdout,
    )
    au = common.fit_au_char_c1(
        data_dir=args.data_dir,
        holdout=holdout,
        cache_path=args.au_cache,
        refresh=args.refresh_au,
    )

    comp = holdout["components"]
    blend4 = comp["blend4"]
    reference = common.macro_f1_probs(blend4, holdout["y_true"], holdout["actions"])
    variants = {
        "linear": comp["linear"],
        "stacker_final": comp["stacker"],
        "e5": comp["e5"],
        "mbert": comp["mbert"],
        "blend3_sanity": comp["blend3"],
        "blend4": blend4,
        f"stacker_final_soft_au_a{args.alpha:g}": common.apply_soft_au(
            comp["stacker"], holdout=holdout, au_probs=au["probs"], alpha=args.alpha
        ),
        f"blend4_soft_au_a{args.alpha:g}": common.apply_soft_au(
            blend4, holdout=holdout, au_probs=au["probs"], alpha=args.alpha
        ),
    }

    summary_rows = [
        common.score_variant(name, probs, holdout=holdout, reference_score=reference)
        for name, probs in variants.items()
    ]
    blend4_pred = common.predict_labels(blend4, holdout["actions"])
    per_class_rows: list[dict[str, Any]] = []
    for name, probs in variants.items():
        per_class_rows.extend(
            common.per_class_variant_rows(name, probs, holdout=holdout, baseline_pred=blend4_pred)
        )

    result = {
        "inputs": {
            **holdout["paths"],
            "data_dir": str(args.data_dir),
            "au_cache": str(args.au_cache),
            "alpha": float(args.alpha),
        },
        "sanity": holdout["sanity"],
        "split": {
            "holdout_rows": int(len(holdout["ids"])),
            "au_rows": int(holdout["au_mask"].sum()),
            "sim_rows": int((~holdout["au_mask"]).sum()),
        },
        "au_model": au["meta"],
        "au_from_cache": bool(au["from_cache"]),
        "summary": summary_rows,
        "elapsed_sec": round(time.time() - started, 3),
    }

    out_prefix = args.out_dir / "mate_stacker_final"
    common.write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_rows)
    common.write_csv(out_prefix.with_name(out_prefix.name + "_per_class.csv"), per_class_rows)
    common.write_json(out_prefix.with_name(out_prefix.name + ".json"), result)
    common.print_score_table(summary_rows)
    print(f"[done] elapsed={result['elapsed_sec']:.3f}s")


if __name__ == "__main__":
    main()
