# -*- coding: utf-8 -*-
"""Error taxonomy for the current league4 + soft-AU baseline.

This script intentionally delegates all probability joins to
scripts/league4/common.py. It only computes diagnostics from the already
materialized league holdout probabilities.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "league4"))

import common as league  # noqa: E402


EXPECTED_FINAL_MACRO = 0.73877
EXPLORE_ACTIONS = {"read_file", "grep_search", "list_directory", "glob_pattern"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "night_out" / "errtax")
    parser.add_argument("--report-path", type=Path, default=ROOT / "context" / "night" / "2026-07-09" / "report_errtax.md")
    parser.add_argument("--au-out-dir", type=Path, default=league.OUT_DIR)
    parser.add_argument("--force-au", action="store_true", help="retrain AU holdout probabilities even if cached")
    return parser.parse_args()


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {path}")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def family_from_id(sample_id: str) -> str:
    if str(sample_id).startswith("sess_au_"):
        return "au"
    if str(sample_id).startswith("sess_sim_"):
        return "sim"
    return "other"


def safe_text(value: Any, limit: int = 220) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def top_language(workspace: dict[str, Any]) -> str:
    mix = workspace.get("language_mix") or {}
    if isinstance(mix, dict) and mix:
        return str(max(mix.items(), key=lambda item: float(item[1] or 0))[0])
    return "unknown"


def numeric_bucket(value: Any, cuts: Sequence[tuple[float, str]], default: str = "unknown") -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    for upper, label in cuts:
        if number <= upper:
            return label
    return cuts[-1][1] if cuts else default


def history_last_user(history: Sequence[Any]) -> str:
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            return safe_text(item.get("content"), 280)
    return ""


def history_last_action(history: Sequence[Any]) -> str:
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            return str(item.get("name") or "none")
    return "none"


def feature_values(sample_id: str, sample: dict[str, Any]) -> dict[str, str]:
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    open_files = workspace.get("open_files") or []
    prompt = str(sample.get("current_prompt") or "").lower()
    return {
        "family": family_from_id(sample_id),
        "history_len": numeric_bucket(len(history) // 2, [(0, "0"), (2, "1-2"), (6, "3-6"), (999, "7+")]),
        "user_tier": str(meta.get("user_tier") or "unknown"),
        "language_pref": str(meta.get("language_pref") or "unknown"),
        "git_dirty": str(workspace.get("git_dirty")),
        "last_ci_status": str(workspace.get("last_ci_status") or "unknown"),
        "top_lang": top_language(workspace),
        "last_action": history_last_action(history),
        "turn_index": numeric_bucket(meta.get("turn_index"), [(0, "0"), (3, "1-3"), (7, "4-7"), (999, "8+")]),
        "elapsed_sec": numeric_bucket(
            meta.get("elapsed_session_sec"),
            [(120, "<=120"), (600, "121-600"), (1800, "601-1800"), (10**9, "1801+")],
        ),
        "budget_tokens": numeric_bucket(
            meta.get("budget_tokens_remaining"),
            [(20_000, "<=20k"), (80_000, "20k-80k"), (160_000, "80k-160k"), (10**9, "160k+")],
        ),
        "open_files": numeric_bucket(len(open_files), [(0, "0"), (2, "1-2"), (6, "3-6"), (999, "7+")]),
        "prompt_has_question": str("?" in prompt),
        "prompt_mentions_test": str(any(token in prompt for token in ("test", "pytest", "ci", "테스트"))),
        "prompt_mentions_file_search": str(
            any(token in prompt for token in ("grep", "search", "find", "list", "glob", "read", "파일", "검색"))
        ),
    }


def per_class_metrics(
    y_true: np.ndarray,
    pred: np.ndarray,
    actions: Sequence[str],
    slice_name: str,
    mask: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    y_slice = y_true[mask]
    p_slice = pred[mask]
    precision, recall, f1, support = precision_recall_fscore_support(
        y_slice, p_slice, labels=list(actions), zero_division=0
    )
    macro = float(f1_score(y_slice, p_slice, labels=list(actions), average="macro", zero_division=0))
    summary = {"slice": slice_name, "rows": int(mask.sum()), "macro_f1": macro}
    rows: list[dict[str, Any]] = []
    for action, p, r, f, s in zip(actions, precision, recall, f1, support):
        rows.append(
            {
                "slice": slice_name,
                "action": str(action),
                "support": int(s),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "macro_gap_if_perfect": float((1.0 - f) / len(actions)),
            }
        )
    return summary, rows


def confusion_rows(y_true: np.ndarray, pred: np.ndarray, actions: Sequence[str]) -> list[dict[str, Any]]:
    cm = confusion_matrix(y_true, pred, labels=list(actions))
    supports = cm.sum(axis=1)
    total_errors = int((y_true != pred).sum())
    rows: list[dict[str, Any]] = []
    for i, true_label in enumerate(actions):
        for j, pred_label in enumerate(actions):
            if i == j:
                continue
            count = int(cm[i, j])
            if not count:
                continue
            rows.append(
                {
                    "true": str(true_label),
                    "pred": str(pred_label),
                    "count": count,
                    "share_of_errors": float(count / total_errors) if total_errors else 0.0,
                    "share_of_true": float(count / supports[i]) if supports[i] else 0.0,
                    "explore_cluster": bool(true_label in EXPLORE_ACTIONS or pred_label in EXPLORE_ACTIONS),
                }
            )
    rows.sort(key=lambda row: (int(row["count"]), float(row["share_of_true"])), reverse=True)
    return rows


def example_rows(
    data: league.LeagueData,
    pred: np.ndarray,
    confusion: Sequence[dict[str, Any]],
    max_examples: int = 12,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    pairs = [row for row in confusion if row["explore_cluster"]] + [row for row in confusion if not row["explore_cluster"]]
    for pair in pairs:
        if len(selected) >= max_examples:
            break
        pair_mask = (data.y_true == pair["true"]) & (pred == pair["pred"])
        for idx in np.flatnonzero(pair_mask):
            sample_id = str(data.ids[int(idx)])
            if sample_id in used_ids:
                continue
            sample = data.samples_by_id[sample_id]
            history = sample.get("history") or []
            meta = sample.get("session_meta") or {}
            workspace = meta.get("workspace") or {}
            selected.append(
                {
                    "id": sample_id,
                    "family": family_from_id(sample_id),
                    "true": str(data.y_true[int(idx)]),
                    "pred": str(pred[int(idx)]),
                    "history_turns": len(history) // 2,
                    "last_action": history_last_action(history),
                    "user_tier": str(meta.get("user_tier") or ""),
                    "language_pref": str(meta.get("language_pref") or ""),
                    "top_lang": top_language(workspace),
                    "last_ci_status": str(workspace.get("last_ci_status") or ""),
                    "current_prompt": safe_text(sample.get("current_prompt"), 500),
                    "last_user": history_last_user(history),
                }
            )
            used_ids.add(sample_id)
            break
    return selected


def write_examples_markdown(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    lines = [
        "# Error Examples",
        "",
        "One row is selected from each high-count confusion pair, prioritizing read/grep/list/glob pairs.",
        "",
    ]
    for row in rows:
        lines.extend(
            [
                f"## {row['id']}",
                "",
                f"- family: `{row['family']}`",
                f"- true -> pred: `{row['true']}` -> `{row['pred']}`",
                f"- history_turns: `{row['history_turns']}`, last_action: `{row['last_action']}`",
                f"- meta: tier=`{row['user_tier']}`, lang=`{row['language_pref']}`, top_lang=`{row['top_lang']}`, ci=`{row['last_ci_status']}`",
                f"- current_prompt: {row['current_prompt']}",
                f"- last_user: {row['last_user']}",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {path}")


def purity_rows(
    data: league.LeagueData,
    pred: np.ndarray,
    targets: Sequence[tuple[str, np.ndarray]],
    min_bucket_rows: int = 40,
) -> list[dict[str, Any]]:
    feature_by_id = {
        str(sample_id): feature_values(str(sample_id), data.samples_by_id[str(sample_id)]) for sample_id in data.ids
    }
    feature_names = list(next(iter(feature_by_id.values())).keys())
    out: list[dict[str, Any]] = []
    total_rows = len(data.ids)
    for target_name, target_mask in targets:
        target_total = int(target_mask.sum())
        if not target_total:
            continue
        for feature in feature_names:
            bucket_counts: Counter[str] = Counter()
            target_counts: Counter[str] = Counter()
            for idx, sample_id in enumerate(data.ids):
                value = feature_by_id[str(sample_id)][feature]
                bucket_counts[value] += 1
                if bool(target_mask[idx]):
                    target_counts[value] += 1
            for value, bucket_n in bucket_counts.items():
                if bucket_n < min_bucket_rows:
                    continue
                target_n = target_counts[value]
                if not target_n:
                    continue
                out.append(
                    {
                        "target": target_name,
                        "feature": feature,
                        "value": value,
                        "bucket_rows": int(bucket_n),
                        "target_rows": int(target_n),
                        "target_rate_in_bucket": float(target_n / bucket_n),
                        "target_coverage": float(target_n / target_total),
                        "bucket_share": float(bucket_n / total_rows),
                    }
                )
    out.sort(
        key=lambda row: (
            float(row["target_rate_in_bucket"]),
            int(row["target_rows"]),
            float(row["target_coverage"]),
        ),
        reverse=True,
    )
    return out


def md_table(rows: Sequence[dict[str, Any]], columns: Sequence[tuple[str, str]], limit: int | None = None) -> list[str]:
    if limit is not None:
        rows = rows[:limit]
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        cells = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def build_report(
    path: Path,
    summary: dict[str, Any],
    per_class: Sequence[dict[str, Any]],
    slice_macro: Sequence[dict[str, Any]],
    slice_per_class: Sequence[dict[str, Any]],
    confusion: Sequence[dict[str, Any]],
    purity: Sequence[dict[str, Any]],
) -> None:
    weak = sorted([row for row in per_class if row["slice"] == "all"], key=lambda row: float(row["f1"]))
    gap = sorted([row for row in per_class if row["slice"] == "all"], key=lambda row: float(row["macro_gap_if_perfect"]), reverse=True)
    explore_conf = [row for row in confusion if row["explore_cluster"]]
    purity_preview = purity[:10]

    lines = [
        "# task2 errtax report",
        "",
        "## Summary",
        "",
        f"- Holdout rows: `{summary['rows']}` (`sim={summary['sim_rows']}`, `au={summary['au_rows']}`).",
        f"- Final mirror: 4-way `[linear, stacker, e5=1.2, mBERT=0.8]` + soft-AU `alpha={summary['alpha']}`.",
        f"- Macro-F1: `{summary['macro_f1']:.6f}`; expected `{EXPECTED_FINAL_MACRO:.5f}` tolerance `5e-4` passed.",
        f"- AU cache hit: `{summary['au_cache_hit']}` from `{summary['au_out_dir']}`.",
        "",
        "## Weakest classes",
        "",
        *md_table(
            weak,
            [
                ("action", "class"),
                ("support", "support"),
                ("precision", "precision"),
                ("recall", "recall"),
                ("f1", "F1"),
                ("macro_gap_if_perfect", "macro gap if perfect"),
            ],
            limit=14,
        ),
        "",
        "## Top confusions",
        "",
        *md_table(
            confusion,
            [
                ("true", "true"),
                ("pred", "pred"),
                ("count", "count"),
                ("share_of_errors", "share errors"),
                ("share_of_true", "share true"),
                ("explore_cluster", "explore"),
            ],
            limit=10,
        ),
        "",
        "## Explore-cluster confusions",
        "",
        *md_table(
            explore_conf,
            [
                ("true", "true"),
                ("pred", "pred"),
                ("count", "count"),
                ("share_of_errors", "share errors"),
                ("share_of_true", "share true"),
            ],
            limit=10,
        ),
        "",
        "## SIM vs AU",
        "",
        *md_table(slice_macro, [("slice", "slice"), ("rows", "rows"), ("macro_f1", "macro-F1")]),
        "",
        "Weakest class per slice:",
        "",
    ]
    for slice_name in ("sim", "au"):
        rows = sorted(
            [row for row in slice_per_class if row["slice"] == slice_name],
            key=lambda row: float(row["f1"]),
        )[:5]
        lines.extend([f"### {slice_name}", ""])
        lines.extend(
            md_table(
                rows,
                [
                    ("action", "class"),
                    ("support", "support"),
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("f1", "F1"),
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Macro-F1 gap contribution",
            "",
            "The last column is `(1 - class_F1) / 14`: the maximum macro-F1 lift if that class became perfect while all other classes stayed fixed.",
            "",
            *md_table(
                gap,
                [
                    ("action", "class"),
                    ("f1", "F1"),
                    ("macro_gap_if_perfect", "max macro lift"),
                ],
                limit=14,
            ),
            "",
            "## Deterministic-key purity scan",
            "",
            "These are not candidate scores. They only show whether observed error sets concentrate under simple deterministic keys enough to justify a later specialist-routing probe.",
            "",
            *md_table(
                purity_preview,
                [
                    ("target", "target"),
                    ("feature", "feature"),
                    ("value", "value"),
                    ("bucket_rows", "bucket rows"),
                    ("target_rows", "target rows"),
                    ("target_rate_in_bucket", "target rate"),
                    ("target_coverage", "coverage"),
                ],
            ),
            "",
            "## Report-only next levers",
            "",
            "1. Pair-specific deterministic-key audit for the largest confusion pair. This follows the soft-AU pattern only if a non-derived key isolates a high-purity subset; it is not an enc-weight, calibration, threshold, prior, seed-soup, serialize, linear-replacement, or encoder-diversity retry.",
            "2. Explore-cluster boundary audit using actual error examples before modeling. The report data says whether read/grep/list/glob mistakes are a few repeated boundaries or diffuse; only the repeated-boundary case should advance to a later routing probe.",
            "3. Slice-specific residual audit for the weakest SIM or AU class. If the weakness is one slice only, a future probe can test a deterministic subroute inside that slice; if both slices share the failure, do not route and treat it as model information loss.",
            "",
            "## Artifacts",
            "",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_summary.json'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_per_class_f1.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_confusion.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_slice_macro_f1.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_slice_per_class_f1.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_macro_gap.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_key_purity.csv'}`",
            f"- `{ROOT / 'night_out' / 'errtax' / 'errtax_error_examples.md'}`",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {path}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("[load] league data")
    data = league.load_league_data()
    print(f"[load] rows={len(data.ids)} actions={len(data.actions)}")

    print("[au] load/train AU holdout probabilities")
    au = league.train_or_load_au_probs(data, args.au_out_dir, force=args.force_au)
    blend = league.four_way_blend(data)
    final = league.apply_soft_au(data, blend, au["probs"], league.DEFAULT_ALPHA)
    pred = league.predict_from_probs(final, data.actions)
    macro = league.macro_f1_labels(data.y_true, pred, data.actions)
    print(f"[score] final macro_f1={macro:.8f}")
    if abs(macro - EXPECTED_FINAL_MACRO) > 5e-4:
        raise AssertionError(f"final macro-F1 {macro:.8f} is not within 5e-4 of {EXPECTED_FINAL_MACRO:.5f}")

    sim_mask = np.asarray([family_from_id(sample_id) == "sim" for sample_id in data.ids], dtype=bool)
    au_mask = np.asarray([family_from_id(sample_id) == "au" for sample_id in data.ids], dtype=bool)
    all_mask = np.ones(len(data.ids), dtype=bool)
    if int(sim_mask.sum() + au_mask.sum()) != len(data.ids):
        raise AssertionError("holdout has ids outside sess_sim_/sess_au_ families")
    if not np.array_equal(au_mask, data.au_mask):
        raise AssertionError("id-prefix AU mask differs from au_route.is_au mask")

    slice_macro: list[dict[str, Any]] = []
    per_class: list[dict[str, Any]] = []
    for slice_name, mask in (("all", all_mask), ("sim", sim_mask), ("au", au_mask)):
        summary_row, rows = per_class_metrics(data.y_true, pred, data.actions, slice_name, mask)
        slice_macro.append(summary_row)
        per_class.extend(rows)

    all_per_class = [row for row in per_class if row["slice"] == "all"]
    macro_gap = sorted(
        [
            {
                "action": row["action"],
                "f1": row["f1"],
                "macro_gap_if_perfect": row["macro_gap_if_perfect"],
            }
            for row in all_per_class
        ],
        key=lambda row: float(row["macro_gap_if_perfect"]),
        reverse=True,
    )

    conf = confusion_rows(data.y_true, pred, data.actions)
    examples = example_rows(data, pred, conf)

    targets: list[tuple[str, np.ndarray]] = []
    for pair in conf[:10]:
        target_name = f"{pair['true']}->{pair['pred']}"
        target_mask = (data.y_true == pair["true"]) & (pred == pair["pred"])
        targets.append((target_name, target_mask))
    weak_classes = sorted(all_per_class, key=lambda row: float(row["f1"]))[:5]
    for row in weak_classes:
        cls = str(row["action"])
        targets.append((f"{cls}_errors", (data.y_true == cls) & (pred != cls)))
    targets.append(("explore_errors", np.isin(data.y_true, list(EXPLORE_ACTIONS)) & (data.y_true != pred)))
    purity = purity_rows(data, pred, targets)

    write_csv(args.out_dir / "errtax_per_class_f1.csv", sorted(per_class, key=lambda row: (row["slice"], float(row["f1"]))))
    write_csv(args.out_dir / "errtax_confusion.csv", conf)
    write_csv(args.out_dir / "errtax_slice_macro_f1.csv", slice_macro)
    write_csv(args.out_dir / "errtax_slice_per_class_f1.csv", sorted(per_class, key=lambda row: (row["slice"], float(row["f1"]))))
    write_csv(args.out_dir / "errtax_macro_gap.csv", macro_gap)
    write_csv(args.out_dir / "errtax_error_examples.csv", examples)
    write_examples_markdown(args.out_dir / "errtax_error_examples.md", examples)
    write_csv(args.out_dir / "errtax_key_purity.csv", purity)

    summary = {
        "rows": int(len(data.ids)),
        "sim_rows": int(sim_mask.sum()),
        "au_rows": int(au_mask.sum()),
        "actions": [str(action) for action in data.actions],
        "macro_f1": float(macro),
        "expected_macro_f1": EXPECTED_FINAL_MACRO,
        "alpha": float(league.DEFAULT_ALPHA),
        "au_cache_hit": bool(au.get("cache_hit", False)),
        "au_meta": au.get("meta", {}),
        "au_out_dir": str(args.au_out_dir),
        "weakest_classes": sorted(all_per_class, key=lambda row: float(row["f1"]))[:5],
        "top_confusions": conf[:10],
        "slice_macro_f1": slice_macro,
        "top_purity": purity[:10],
        "output_dir": str(args.out_dir),
        "report_path": str(args.report_path),
    }
    write_json(args.out_dir / "errtax_summary.json", summary)
    build_report(args.report_path, summary, per_class, slice_macro, per_class, conf, purity)

    print("[summary]")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
