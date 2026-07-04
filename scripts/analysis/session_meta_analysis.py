"""
Item 5: session_meta determinism.

- last_ci_status=failed -> run_tests/edit_file conditional distribution.
- git_dirty effect on action distribution.
- budget_tokens_remaining bucket effect.
- turn_index / step effect (does the policy change over the session?).
- First-step rows (history empty, step_00/01) as a separate regime.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import load_frame

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
OUT_DIR.mkdir(exist_ok=True)


def dist_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return (
        df.groupby(group_col)["action"]
        .value_counts(normalize=True)
        .unstack(fill_value=0)
        .round(4)
    )


def main():
    df = load_frame()

    print("=== last_ci_status effect (all rows) ===")
    print(dist_table(df, "last_ci_status").to_string())
    print("\nrow counts:")
    print(df["last_ci_status"].value_counts())

    print("\n=== last_ci_status effect, conditioned on last_action having failure-relevant context ===")
    # restrict to rows where last_action in mutate/validate family (where CI status is actionable)
    relevant = df[df["last_action"].isin(["edit_file", "write_file", "apply_patch", "run_tests", "lint_or_typecheck"])]
    print(dist_table(relevant, "last_ci_status").to_string())
    print(relevant["last_ci_status"].value_counts())

    print("\n=== git_dirty effect ===")
    print(dist_table(df, "git_dirty").to_string())

    print("\n=== budget_tokens_remaining bucket effect ===")
    df["budget_bucket"] = pd.cut(
        df["budget_tokens_remaining"],
        bins=[-1, 2000, 5000, 10000, 20000, 40000, 80000, 130000, 1e9],
        labels=["<2k", "2-5k", "5-10k", "10-20k", "20-40k", "40-80k", "80-130k", "130k+"],
    )
    print(dist_table(df, "budget_bucket").to_string())
    print(df["budget_bucket"].value_counts().sort_index())

    print("\n=== turn_index effect (selected turns) ===")
    ti = dist_table(df, "turn_index")
    print(ti.to_string())

    print("\n=== step number effect (from id, selected) ===")
    st = dist_table(df, "step")
    print(st.head(15).to_string())

    print("\n=== First-step rows (history_len==0) vs rest: action distribution ===")
    df["is_first_step"] = df["history_len"] == 0
    print(dist_table(df, "is_first_step").to_string())
    print(df["is_first_step"].value_counts())

    # cross: does budget correlate with respond_only (session wrapping up)?
    print("\n=== respond_only rate by budget bucket ===")
    ro_rate = df.groupby("budget_bucket")["action"].apply(lambda s: (s == "respond_only").mean())
    print(ro_rate.round(4))

    print("\n=== respond_only rate by turn_index ===")
    ro_rate_t = df.groupby("turn_index")["action"].apply(lambda s: (s == "respond_only").mean())
    print(ro_rate_t.round(4))

    # elapsed_session_sec
    print("\n=== elapsed_session_sec bucket effect ===")
    df["elapsed_bucket"] = pd.qcut(df["elapsed_session_sec"], 8, duplicates="drop")
    print(dist_table(df, "elapsed_bucket").to_string())

    # user_tier / language_pref
    print("\n=== user_tier effect ===")
    print(dist_table(df, "user_tier").to_string())
    print("\n=== language_pref effect ===")
    print(dist_table(df, "language_pref").to_string())

    dist_table(df, "last_ci_status").to_csv(OUT_DIR / "meta_last_ci_status.csv")
    dist_table(df, "budget_bucket").to_csv(OUT_DIR / "meta_budget_bucket.csv")
    dist_table(df, "turn_index").to_csv(OUT_DIR / "meta_turn_index.csv")
    dist_table(df, "is_first_step").to_csv(OUT_DIR / "meta_first_step.csv")


if __name__ == "__main__":
    main()
