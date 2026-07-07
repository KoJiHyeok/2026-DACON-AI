# -*- coding: utf-8 -*-
"""Evaluate teammate read/grep/list log-bias on fixed holdout variants."""
from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

import common


BASE_BIAS = {
    "read_file": 0.1,
    "grep_search": -0.1,
    "list_directory": -0.18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    common.add_common_args(parser)
    return parser.parse_args()


def score_biased(
    *,
    base_name: str,
    bias_name: str,
    probs: np.ndarray,
    bias: dict[str, float],
    holdout: dict[str, Any],
    baseline_pred: np.ndarray,
    baseline_score: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    actions = holdout["actions"]
    y_true = holdout["y_true"]
    mask = holdout["au_mask"]
    pred = common.labels_from_log_bias(probs, actions, bias)
    score = common.macro_f1_labels(y_true, pred, actions)
    row = {
        "base": base_name,
        "bias": bias_name,
        "macro_f1": score,
        "delta_vs_unbiased": score - baseline_score,
        "au_macro_f1": common.macro_f1_labels(y_true[mask], pred[mask], actions),
        "sim_macro_f1": common.macro_f1_labels(y_true[~mask], pred[~mask], actions),
        "changed_vs_unbiased": int(np.sum(pred != baseline_pred)),
        "read_bias": float(bias.get("read_file", 0.0)),
        "grep_bias": float(bias.get("grep_search", 0.0)),
        "list_bias": float(bias.get("list_directory", 0.0)),
    }

    base_pc = {r["class"]: r for r in common.per_class_f1(y_true, baseline_pred, actions)}
    pc_rows = []
    for pc in common.per_class_f1(y_true, pred, actions):
        old = base_pc[pc["class"]]
        pc_rows.append(
            {
                "base": base_name,
                "bias": bias_name,
                "class": pc["class"],
                "support": pc["support"],
                "baseline_f1": old["f1"],
                "biased_f1": pc["f1"],
                "delta_f1": pc["f1"] - old["f1"],
                "baseline_pred_count": old["pred_count"],
                "biased_pred_count": pc["pred_count"],
                "delta_pred_count": pc["pred_count"] - old["pred_count"],
            }
        )
    return row, pc_rows


def one_at_a_time_biases() -> list[tuple[str, dict[str, float]]]:
    rows: list[tuple[str, dict[str, float]]] = []
    for cls, base_value in BASE_BIAS.items():
        short = cls.replace("_file", "").replace("_search", "").replace("_directory", "")
        for scale in (0.5, 1.5):
            rows.append((f"{short}_only_{scale:g}x", {cls: base_value * scale}))
    return rows


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
    bases = {
        "stacker_final": comp["stacker"],
        "blend4": comp["blend4"],
        f"blend4_soft_au_a{args.alpha:g}": common.apply_soft_au(
            comp["blend4"], holdout=holdout, au_probs=au["probs"], alpha=args.alpha
        ),
    }
    bias_variants: list[tuple[str, dict[str, float]]] = [
        ("mate_0.5x", common.scaled_bias(0.5)),
        ("mate_1.0x", common.scaled_bias(1.0)),
        ("mate_1.5x", common.scaled_bias(1.5)),
    ]
    bias_variants.extend(one_at_a_time_biases())

    summary_rows: list[dict[str, Any]] = []
    per_class_rows: list[dict[str, Any]] = []
    for base_name, probs in bases.items():
        baseline_pred = common.predict_labels(probs, holdout["actions"])
        baseline_score = common.macro_f1_labels(holdout["y_true"], baseline_pred, holdout["actions"])
        summary_rows.append(
            {
                "base": base_name,
                "bias": "none",
                "macro_f1": baseline_score,
                "delta_vs_unbiased": 0.0,
                "au_macro_f1": common.macro_f1_labels(
                    holdout["y_true"][holdout["au_mask"]],
                    baseline_pred[holdout["au_mask"]],
                    holdout["actions"],
                ),
                "sim_macro_f1": common.macro_f1_labels(
                    holdout["y_true"][~holdout["au_mask"]],
                    baseline_pred[~holdout["au_mask"]],
                    holdout["actions"],
                ),
                "changed_vs_unbiased": 0,
                "read_bias": 0.0,
                "grep_bias": 0.0,
                "list_bias": 0.0,
            }
        )
        for bias_name, bias in bias_variants:
            row, pc_rows = score_biased(
                base_name=base_name,
                bias_name=bias_name,
                probs=probs,
                bias=bias,
                holdout=holdout,
                baseline_pred=baseline_pred,
                baseline_score=baseline_score,
            )
            summary_rows.append(row)
            per_class_rows.extend(pc_rows)

    result = {
        "inputs": {
            **holdout["paths"],
            "data_dir": str(args.data_dir),
            "au_cache": str(args.au_cache),
            "alpha": float(args.alpha),
        },
        "sanity": holdout["sanity"],
        "bias_family_note": "log(p + 1e-12) plus class constants; argmax only",
        "base_bias": BASE_BIAS,
        "summary": summary_rows,
        "au_model": au["meta"],
        "au_from_cache": bool(au["from_cache"]),
        "elapsed_sec": round(time.time() - started, 3),
    }

    out_prefix = args.out_dir / "mate_logbias"
    common.write_csv(out_prefix.with_name(out_prefix.name + "_summary.csv"), summary_rows)
    common.write_csv(out_prefix.with_name(out_prefix.name + "_per_class.csv"), per_class_rows)
    common.write_json(out_prefix.with_name(out_prefix.name + ".json"), result)

    print("base,bias,macro_f1,delta_vs_unbiased,au_macro_f1,sim_macro_f1,changed_vs_unbiased")
    for row in summary_rows:
        print(
            "{base},{bias},{macro},{delta},{au},{sim},{changed}".format(
                base=row["base"],
                bias=row["bias"],
                macro=common.format_float(row["macro_f1"]),
                delta=f"{float(row['delta_vs_unbiased']):+.9f}",
                au=common.format_float(row["au_macro_f1"]),
                sim=common.format_float(row["sim_macro_f1"]),
                changed=row["changed_vs_unbiased"],
            )
        )
    print(f"[done] elapsed={result['elapsed_sec']:.3f}s")


if __name__ == "__main__":
    main()
