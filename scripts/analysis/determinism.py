"""
Item 1: Determinism analysis.

For a range of state definitions, compute P(action | state) purity (share of
the majority action within each state bucket) and coverage (share of all rows
covered by that bucket). Report purity>=0.95 and >=0.99 bucket sets and their
cumulative coverage -- the headline "rule ceiling" number.

Also reports unique session count per bucket so single-session buckets (which
look "pure" only because they repeat within one simulated session) are
visible separately.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_frame

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
OUT_DIR.mkdir(exist_ok=True)


def bucket_purity(df: pd.DataFrame, state_cols: list[str], min_rows: int = 5) -> pd.DataFrame:
    """Return one row per state-bucket with n_rows, n_sessions, top_action, purity."""
    g = df.groupby(state_cols, dropna=False)
    recs = []
    for key, sub in g:
        if len(sub) < min_rows:
            continue
        vc = sub["action"].value_counts()
        top_action = vc.index[0]
        purity = vc.iloc[0] / len(sub)
        n_sessions = sub["session_key"].nunique()
        recs.append({
            "state": key if isinstance(key, tuple) else (key,),
            "n_rows": len(sub),
            "n_sessions": n_sessions,
            "top_action": top_action,
            "purity": purity,
        })
    out = pd.DataFrame(recs)
    return out.sort_values("n_rows", ascending=False)


def coverage_at_threshold(bucket_df: pd.DataFrame, total_rows: int, thresh: float) -> tuple[int, float, int]:
    sel = bucket_df[bucket_df["purity"] >= thresh]
    n_rows = sel["n_rows"].sum()
    return len(sel), n_rows / total_rows, sel["n_rows"].sum()


def add_engineered_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["history_present"] = np.where(df["history_len"] == 0, "empty", "present")

    def family4(a):
        if a == "none":
            return "none"
        if a in {"read_file", "grep_search", "list_directory", "glob_pattern"}:
            return "explore"
        if a in {"edit_file", "write_file", "apply_patch", "run_bash", "run_tests", "lint_or_typecheck"}:
            return "mutate_validate"
        return "coordinate"

    df["last_action_family4"] = df["last_action"].map(family4)
    df["last2_action_family4"] = df["last2_action"].map(family4)

    def turn_bin(t):
        if t <= 1:
            return "turn_01"
        if t <= 3:
            return "turn_02_03"
        if t <= 6:
            return "turn_04_06"
        return "turn_07_plus"

    df["turn_index_bin"] = df["turn_index"].map(turn_bin)
    df["history_len_bin"] = pd.cut(
        df["history_len"], bins=[-1, 0, 2, 4, 8, 100],
        labels=["0", "1-2", "3-4", "5-8", "9+"]
    ).astype(str)

    def budget_bin(b):
        if b < 5000:
            return "b_lt5k"
        if b < 20000:
            return "b_5k_20k"
        if b < 60000:
            return "b_20k_60k"
        if b < 120000:
            return "b_60k_120k"
        return "b_120k_plus"

    df["budget_bin"] = df["budget_tokens_remaining"].map(budget_bin)

    # keyword flags on current_prompt useful as state components
    cp = df["current_prompt"].fillna("")
    df["prompt_has_glob_char"] = cp.str.contains(r"[*]\.\w+|\*\*", regex=True)
    df["prompt_has_quoted"] = cp.str.contains(r"['\"][^'\"]{2,40}['\"]", regex=True)
    df["prompt_has_path"] = cp.str.contains(r"[\w\-/]+\.[a-zA-Z]{1,5}\b|/", regex=True)

    return df


def main():
    df = load_frame()
    df = add_engineered_cols(df)
    total = len(df)

    state_defs = {
        "last_action": ["last_action"],
        "last_action+last2": ["last_action", "last2_action"],
        "last_action+last2+last3": ["last_action", "last2_action", "last3_action"],
        "last_action_family4": ["last_action_family4"],
        "last_action+last_args_sig": ["last_action", "last_args_sig"],
        "last_ci_status+last_action": ["last_ci_status", "last_action"],
        "history_len_bin": ["history_len_bin"],
        "history_len_bin+last_action": ["history_len_bin", "last_action"],
        "turn_index": ["turn_index"],
        "turn_index_bin": ["turn_index_bin"],
        "turn_index+last_action": ["turn_index", "last_action"],
        "step+last_action": ["step", "last_action"],
        "last_action+git_dirty": ["last_action", "git_dirty"],
        "last_action+budget_bin": ["last_action", "budget_bin"],
        "last_action+user_tier": ["last_action", "user_tier"],
        "last_action+language_pref": ["last_action", "language_pref"],
        "history_present": ["history_present"],
        "history_present+last_action_family4": ["history_present", "last_action_family4"],
        "last_action+last2+turn_index_bin": ["last_action", "last2_action", "turn_index_bin"],
        "last_action+prompt_glob": ["last_action", "prompt_has_glob_char"],
        "last_action+last_result_prefix": ["last_action", "last_result_summary"],
    }

    summary_rows = []
    bucket_tables = {}
    for name, cols in state_defs.items():
        bt = bucket_purity(df, cols, min_rows=5)
        bucket_tables[name] = bt
        for thresh in (0.95, 0.99, 1.0):
            n_buckets, cov, n_rows = coverage_at_threshold(bt, total, thresh)
            summary_rows.append({
                "state_def": name,
                "threshold": thresh,
                "n_buckets": n_buckets,
                "n_rows_covered": n_rows,
                "coverage": cov,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_DIR / "determinism_summary.csv", index=False)
    print("=== Coverage summary (purity threshold) ===")
    print(summary.pivot(index="state_def", columns="threshold", values="coverage").round(4))

    # Global "rule ceiling": union of rows covered by >=0.99-purity buckets across
    # the best few individual state defs (non-overlapping accounting via row-level max).
    # Build a per-row "is this row in some >=0.99 purity bucket for definition X" flag
    # for the most promising state defs, then take the row-level OR (best achievable
    # coverage if we could freely choose which rule fires per row -- upper bound).
    candidate_defs = [
        "last_action+last2+last3", "last_action+last2", "last_action+last_args_sig",
        "history_len_bin+last_action", "last_action+last_result_prefix",
        "step+last_action", "last_action_family4",
    ]
    row_flags = pd.DataFrame(index=df.index)
    for name in candidate_defs:
        cols = state_defs[name]
        bt = bucket_tables[name]
        pure = bt[bt["purity"] >= 0.99]
        pure_keys = set(pure["state"])
        key_series = list(zip(*[df[c] for c in cols])) if len(cols) > 1 else list(df[cols[0]])
        if len(cols) == 1:
            key_series = [(k,) for k in key_series]
        row_flags[name] = [k in pure_keys for k in key_series]

    any_pure = row_flags.any(axis=1)
    print(f"\n=== Union upper bound across {len(candidate_defs)} candidate defs at purity>=0.99 ===")
    print(f"rows covered: {any_pure.sum()} / {total} = {any_pure.mean():.4f}")

    # Save top buckets per definition (for the report table) -- purity>=0.99, sorted by n_rows
    top_buckets_records = []
    for name, bt in bucket_tables.items():
        pure = bt[bt["purity"] >= 0.99].sort_values("n_rows", ascending=False).head(15)
        for _, r in pure.iterrows():
            top_buckets_records.append({
                "state_def": name,
                "state": r["state"],
                "n_rows": r["n_rows"],
                "n_sessions": r["n_sessions"],
                "top_action": r["top_action"],
                "purity": round(r["purity"], 4),
            })
    top_buckets = pd.DataFrame(top_buckets_records)
    top_buckets.to_csv(OUT_DIR / "top_pure_buckets.csv", index=False)

    # Also: for last_action alone (the simplest deployable rule), full breakdown
    la = bucket_tables["last_action"].sort_values("n_rows", ascending=False)
    la.to_csv(OUT_DIR / "last_action_buckets.csv", index=False)
    print("\n=== last_action alone: purity per bucket ===")
    print(la[["state", "n_rows", "n_sessions", "top_action", "purity"]].to_string(index=False))

    with open(OUT_DIR / "determinism_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "total_rows": total,
            "n_sessions": df["session_key"].nunique(),
            "union_upper_bound_rows": int(any_pure.sum()),
            "union_upper_bound_coverage": float(any_pure.mean()),
            "candidate_defs": candidate_defs,
        }, f, indent=2)

    print("\nSaved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
