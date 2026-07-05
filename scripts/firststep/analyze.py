# -*- coding: utf-8 -*-
"""Analyze first-step holdout behavior across league components."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

from common import (
    DEFAULT_HOLDOUT_BASE,
    DEFAULT_OOF_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_TRAIN_LABELS,
    is_hist0,
    load_league_components,
    load_train_records,
    macro_f1_from_pred,
    per_class_rows,
    predict_labels,
    session_id,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--holdout-base", type=Path, default=DEFAULT_HOLDOUT_BASE)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def write_label_distribution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    scopes = {
        "train_all": np.ones(len(df), dtype=bool),
        "train_hist0": df["hist0"].to_numpy(dtype=bool),
        "train_non_hist0": ~df["hist0"].to_numpy(dtype=bool),
        "nonholdout_hist0": (df["hist0"] & ~df["in_holdout"]).to_numpy(dtype=bool),
        "holdout_hist0": (df["hist0"] & df["in_holdout"]).to_numpy(dtype=bool),
    }
    rows = []
    for scope, mask in scopes.items():
        part = df.loc[mask]
        total = int(part.shape[0])
        counts = Counter(part["action"])
        for action, count in sorted(counts.items()):
            rows.append(
                {
                    "scope": scope,
                    "action": str(action),
                    "count": int(count),
                    "scope_total": total,
                    "rate": float(count / total) if total else 0.0,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "firststep_label_distribution.csv", index=False)
    return out


def component_scores(league: dict, holdout_hist0: np.ndarray, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = league["ids"]
    y_true = league["y_true"]
    actions = league["actions"]
    masks = {
        "all": np.ones(len(ids), dtype=bool),
        "hist0": holdout_hist0,
        "non_hist0": ~holdout_hist0,
    }

    macro_rows = []
    per_rows = []
    for component, probs in league["components"].items():
        preds = predict_labels(probs, actions)
        for segment, mask in masks.items():
            if not np.any(mask):
                continue
            macro_rows.append(
                {
                    "component": component,
                    "segment": segment,
                    "rows": int(mask.sum()),
                    "macro_f1": macro_f1_from_pred(y_true[mask], preds[mask], actions),
                }
            )
            per_rows.extend(
                per_class_rows(
                    y_true[mask],
                    preds[mask],
                    actions,
                    prefix={"component": component, "segment": segment, "rows": int(mask.sum())},
                )
            )

    macro_df = pd.DataFrame(macro_rows)
    per_df = pd.DataFrame(per_rows)
    macro_df.to_csv(out_dir / "firststep_component_macro_f1.csv", index=False)
    per_df.to_csv(out_dir / "firststep_component_per_class_f1.csv", index=False)
    return macro_df, per_df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_train_records(args.train_jsonl, args.labels_csv)
    league = load_league_components(args.holdout_base, args.oof_dir)
    holdout_ids = {str(sample_id) for sample_id in league["ids"]}
    by_id = {str(row["id"]): row for row in records}

    missing_records = [sample_id for sample_id in league["ids"] if str(sample_id) not in by_id]
    if missing_records:
        raise ValueError(f"holdout ids missing from train records: {missing_records[:5]}")

    holdout_hist0 = np.asarray([is_hist0(by_id[str(sample_id)]) for sample_id in league["ids"]], dtype=bool)
    df = pd.DataFrame(
        [
            {
                "id": str(row["id"]),
                "session": session_id(str(row["id"])),
                "hist0": is_hist0(row),
                "in_holdout": str(row["id"]) in holdout_ids,
                "action": str(row["action"]),
            }
            for row in records
        ]
    )

    label_df = write_label_distribution(df, args.out_dir)
    macro_df, per_df = component_scores(league, holdout_hist0, args.out_dir)

    hist0_macro = macro_df[macro_df["segment"] == "hist0"].sort_values("macro_f1", ascending=False)
    summary = {
        "inputs": {
            "train_jsonl": str(args.train_jsonl),
            "labels_csv": str(args.labels_csv),
            "holdout_base": str(args.holdout_base),
            "oof_dir": str(args.oof_dir),
        },
        "join_assert_blend_macro_f1": float(league["blend_f1"]),
        "train_rows": int(len(df)),
        "train_hist0_rows": int(df["hist0"].sum()),
        "train_hist0_sessions": int(df.loc[df["hist0"], "session"].nunique()),
        "nonholdout_hist0_rows": int((df["hist0"] & ~df["in_holdout"]).sum()),
        "nonholdout_hist0_sessions": int(df.loc[df["hist0"] & ~df["in_holdout"], "session"].nunique()),
        "holdout_rows": int(len(league["ids"])),
        "holdout_hist0_rows": int(holdout_hist0.sum()),
        "holdout_hist0_share": float(holdout_hist0.mean()),
        "holdout_hist0_label_counts": {
            str(k): int(v)
            for k, v in Counter(league["y_true"][holdout_hist0]).most_common()
        },
        "component_macro_f1": macro_df.to_dict(orient="records"),
        "hist0_component_ranking": hist0_macro.to_dict(orient="records"),
        "output_files": [
            "firststep_label_distribution.csv",
            "firststep_component_macro_f1.csv",
            "firststep_component_per_class_f1.csv",
            "firststep_analysis_summary.json",
        ],
    }
    (args.out_dir / "firststep_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print(f"[done] wrote analysis outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
