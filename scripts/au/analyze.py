# -*- coding: utf-8 -*-
"""AU/SIM distribution and holdout component diagnostics."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

from common import (
    DEFAULT_HOLDOUT_BASE,
    DEFAULT_OOF_DIR,
    DEFAULT_OUT_DIR,
    DEFAULT_TRAIN_JSONL,
    DEFAULT_TRAIN_LABELS,
    bucket_from_id,
    flatten_record,
    load_league_components,
    load_train_records,
    predict_labels,
    truncate,
)


NUMERIC_FIELDS = [
    "history_len",
    "n_action_turns",
    "n_user_turns",
    "current_prompt_len",
    "current_prompt_words",
    "current_prompt_hangul_frac",
    "current_prompt_non_ascii_frac",
    "budget_tokens_remaining",
    "turn_index",
    "elapsed_session_sec",
    "workspace_loc",
    "n_open_files",
    "n_langs",
]

CATEGORICAL_FIELDS = [
    "user_tier",
    "language_pref",
    "prompt_lang_guess",
    "git_dirty",
    "last_ci_status",
    "top_lang",
    "last_action",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--labels-csv", type=Path, default=DEFAULT_TRAIN_LABELS)
    parser.add_argument("--holdout-base", type=Path, default=DEFAULT_HOLDOUT_BASE)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def write_label_distribution(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    counts = df.groupby(["bucket", "action"], observed=True).size().rename("count").reset_index()
    totals = df.groupby("bucket", observed=True).size().rename("bucket_total").reset_index()
    out = counts.merge(totals, on="bucket")
    out["rate_in_bucket"] = out["count"] / out["bucket_total"]
    total_by_action = df.groupby("action", observed=True).size().rename("total_action_count").reset_index()
    au_by_action = (
        df[df["bucket"] == "au"].groupby("action", observed=True).size().rename("au_action_count").reset_index()
    )
    out = out.merge(total_by_action, on="action", how="left").merge(au_by_action, on="action", how="left")
    out["au_action_count"] = out["au_action_count"].fillna(0).astype(int)
    out["au_share_of_action"] = out["au_action_count"] / out["total_action_count"]
    out.sort_values(["action", "bucket"]).to_csv(out_dir / "au_train_label_distribution.csv", index=False)
    return out


def write_numeric_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for field in NUMERIC_FIELDS:
        for bucket, part in df.groupby("bucket", observed=True):
            values = pd.to_numeric(part[field], errors="coerce").dropna()
            rows.append(
                {
                    "field": field,
                    "bucket": bucket,
                    "count": int(values.shape[0]),
                    "mean": float(values.mean()) if not values.empty else np.nan,
                    "median": float(values.median()) if not values.empty else np.nan,
                    "std": float(values.std(ddof=0)) if not values.empty else np.nan,
                    "p10": float(values.quantile(0.10)) if not values.empty else np.nan,
                    "p90": float(values.quantile(0.90)) if not values.empty else np.nan,
                    "min": float(values.min()) if not values.empty else np.nan,
                    "max": float(values.max()) if not values.empty else np.nan,
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "au_field_numeric_summary.csv", index=False)
    return out


def write_categorical_summary(df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for field in CATEGORICAL_FIELDS:
        counts = df.groupby(["bucket", field], observed=True).size().rename("count").reset_index()
        totals = df.groupby("bucket", observed=True).size().rename("bucket_total").reset_index()
        counts = counts.merge(totals, on="bucket")
        counts["rate_in_bucket"] = counts["count"] / counts["bucket_total"]
        counts["field"] = field
        counts = counts.rename(columns={field: "value"})
        rows.append(counts[["field", "bucket", "value", "count", "bucket_total", "rate_in_bucket"]])
    out = pd.concat(rows, ignore_index=True)
    out.sort_values(["field", "bucket", "count"], ascending=[True, True, False]).to_csv(
        out_dir / "au_field_categorical_summary.csv", index=False
    )
    return out


def component_scores(league: dict, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    ids = league["ids"]
    y_true = league["y_true"]
    actions = league["actions"]
    buckets = np.asarray([bucket_from_id(sample_id) for sample_id in ids])

    macro_rows = []
    per_class_rows = []
    masks = {
        "all": np.ones(len(ids), dtype=bool),
        "sim": buckets == "sim",
        "au": buckets == "au",
    }
    for component, probs in league["components"].items():
        preds = predict_labels(probs, actions)
        for bucket, mask in masks.items():
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true[mask], preds[mask], labels=actions, zero_division=0
            )
            macro_rows.append(
                {
                    "component": component,
                    "bucket": bucket,
                    "rows": int(mask.sum()),
                    "macro_f1": float(np.mean(f1)),
                }
            )
            for action, p, r, f, s in zip(actions, precision, recall, f1, support):
                per_class_rows.append(
                    {
                        "component": component,
                        "bucket": bucket,
                        "action": action,
                        "support": int(s),
                        "precision": float(p),
                        "recall": float(r),
                        "f1": float(f),
                    }
                )
    macro_df = pd.DataFrame(macro_rows)
    per_class_df = pd.DataFrame(per_class_rows)
    macro_df.to_csv(out_dir / "au_component_macro_f1.csv", index=False)
    per_class_df.to_csv(out_dir / "au_component_per_class_f1.csv", index=False)
    return macro_df, per_class_df


def write_confusion_and_samples(records: list[dict], league: dict, out_dir: Path) -> pd.DataFrame:
    ids = league["ids"]
    y_true = league["y_true"]
    actions = league["actions"]
    blend_preds = predict_labels(league["components"]["blend"], actions)
    au_mask = np.asarray([bucket_from_id(sample_id) == "au" for sample_id in ids])

    cm = confusion_matrix(y_true[au_mask], blend_preds[au_mask], labels=actions)
    rows = []
    for i, true_label in enumerate(actions):
        for j, pred_label in enumerate(actions):
            if true_label == pred_label:
                continue
            count = int(cm[i, j])
            if count:
                rows.append({"true": true_label, "pred": pred_label, "count": count})
    conf_df = pd.DataFrame(rows).sort_values("count", ascending=False)
    conf_df.to_csv(out_dir / "au_blend_confusion_top.csv", index=False)

    by_id = {str(row["id"]): row for row in records}
    selected: list[str] = []
    for row in conf_df.head(10).itertuples(index=False):
        pair_mask = au_mask & (y_true == row.true) & (blend_preds == row.pred)
        for sample_id in ids[pair_mask]:
            if sample_id not in selected:
                selected.append(str(sample_id))
                break
        if len(selected) >= 5:
            break
    if len(selected) < 5:
        for sample_id in ids[au_mask & (y_true != blend_preds)]:
            if sample_id not in selected:
                selected.append(str(sample_id))
            if len(selected) >= 5:
                break

    pred_lookup = {sample_id: pred for sample_id, pred in zip(ids, blend_preds)}
    true_lookup = {sample_id: true for sample_id, true in zip(ids, y_true)}
    sample_lines = [
        "# AU blend error samples",
        "",
        "These are selected from the top AU confusion pairs in the 9,969-row league holdout.",
        "",
    ]
    for sample_id in selected[:5]:
        sample = by_id[sample_id]
        meta = sample.get("session_meta") or {}
        workspace = meta.get("workspace") or {}
        history = sample.get("history") or []
        sample_lines.extend(
            [
                f"## {sample_id}",
                "",
                f"- true: `{true_lookup[sample_id]}`",
                f"- blend_pred: `{pred_lookup[sample_id]}`",
                f"- meta: tier={meta.get('user_tier')} lang={meta.get('language_pref')} "
                f"turn={meta.get('turn_index')} ci={workspace.get('last_ci_status')} "
                f"dirty={workspace.get('git_dirty')} open_files={len(workspace.get('open_files') or [])}",
                f"- current_prompt: {truncate(sample.get('current_prompt'), 900)}",
                "",
                "Recent history:",
                "",
            ]
        )
        for item in history[-6:]:
            role = item.get("role") if isinstance(item, dict) else "unknown"
            if role == "assistant_action":
                text = f"assistant_action {item.get('name')} -> {truncate(item.get('result_summary'), 240)}"
            else:
                text = f"user -> {truncate(item.get('content'), 360)}"
            sample_lines.append(f"- {text}")
        sample_lines.append("")
    (out_dir / "au_error_samples.md").write_text("\n".join(sample_lines), encoding="utf-8")
    return conf_df


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records = load_train_records(args.train_jsonl, args.labels_csv)
    df = pd.DataFrame([flatten_record(row) for row in records])
    df.to_csv(args.out_dir / "au_train_flat_features.csv", index=False)

    label_df = write_label_distribution(df, args.out_dir)
    numeric_df = write_numeric_summary(df, args.out_dir)
    categorical_df = write_categorical_summary(df, args.out_dir)

    league = load_league_components(args.holdout_base, args.oof_dir)
    macro_df, per_class_df = component_scores(league, args.out_dir)
    conf_df = write_confusion_and_samples(records, league, args.out_dir)

    train_counts = df["bucket"].value_counts().to_dict()
    holdout_buckets = Counter(bucket_from_id(sample_id) for sample_id in league["ids"])
    summary = {
        "train_rows": int(df.shape[0]),
        "train_bucket_counts": {str(k): int(v) for k, v in train_counts.items()},
        "holdout_rows": int(len(league["ids"])),
        "holdout_bucket_counts": {str(k): int(v) for k, v in holdout_buckets.items()},
        "join_assert_blend_macro_f1": float(league["blend_f1"]),
        "component_macro_f1": macro_df.to_dict(orient="records"),
        "top_au_confusions": conf_df.head(5).to_dict(orient="records"),
        "output_files": [
            "au_train_flat_features.csv",
            "au_train_label_distribution.csv",
            "au_field_numeric_summary.csv",
            "au_field_categorical_summary.csv",
            "au_component_macro_f1.csv",
            "au_component_per_class_f1.csv",
            "au_blend_confusion_top.csv",
            "au_error_samples.md",
        ],
    }
    (args.out_dir / "au_analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:4000])
    print(f"[done] wrote analysis outputs to {args.out_dir}")


if __name__ == "__main__":
    main()
