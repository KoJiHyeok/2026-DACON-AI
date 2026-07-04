"""
Item 6: Simulator artifacts.

- Session termination: for each session, is the row with the highest observed
  step number disproportionately respond_only? (Session arcs may always end
  on a wrap-up turn.)
- turn_index vs step alignment (should be identical if step := turn_index).
- budget_tokens_remaining / elapsed_session_sec as a function of turn_index
  (deterministic formula vs noisy).
- id numeric fields (date, session serial) vs label -- sanity check for
  accidental leakage from generation order.
- args-schema-is-deterministic-per-action-name note (found while building
  exploration signals) recorded here for completeness.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_frame

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
OUT_DIR.mkdir(exist_ok=True)

ID_RE = re.compile(r"sess_sim_(\d+)_(\d+)-step_(\d+)")


def main():
    df = load_frame()

    # --- Session termination ---
    max_step = df.groupby("session_key")["step"].transform("max")
    df["is_last_observed_step"] = df["step"] == max_step
    session_sizes = df.groupby("session_key")["step"].nunique()
    print(f"Sessions: {df['session_key'].nunique()}, step-count distribution:")
    print(session_sizes.describe())
    print("\nStep count value counts (top 15):")
    print(session_sizes.value_counts().sort_index().head(15))

    print("\n=== Action distribution: last observed step in session vs earlier steps ===")
    last_dist = df[df["is_last_observed_step"]]["action"].value_counts(normalize=True)
    earlier_dist = df[~df["is_last_observed_step"]]["action"].value_counts(normalize=True)
    cmp = pd.DataFrame({"last_step": last_dist, "earlier_steps": earlier_dist}).fillna(0).round(4)
    print(cmp.to_string())
    n_last = df["is_last_observed_step"].sum()
    respond_only_rate_last = (df[df["is_last_observed_step"]]["action"] == "respond_only").mean()
    print(f"\nrespond_only rate among last-observed-step rows: {respond_only_rate_last:.4f} (n={n_last})")
    print(f"respond_only rate among earlier rows: {(df[~df['is_last_observed_step']]['action']=='respond_only').mean():.4f}")

    # is this "is_last_observed_step" signal just proxying for high turn_index/low budget
    # (which we already found matters), or does it add lift on TOP of those?
    print("\n=== respond_only rate: is_last_observed_step x turn_index_bin ===")
    df["turn_bin5"] = pd.cut(df["turn_index"], [0, 3, 6, 9, 100], labels=["1-3", "4-6", "7-9", "10+"])
    piv = df.groupby(["turn_bin5", "is_last_observed_step"])["action"].apply(lambda s: (s == "respond_only").mean())
    print(piv.unstack().round(4).to_string())

    # --- turn_index vs step alignment ---
    print("\n=== turn_index == step agreement rate ===")
    print((df["turn_index"] == df["step"]).mean())
    mismatch = df[df["turn_index"] != df["step"]]
    print(f"mismatched rows: {len(mismatch)}")
    if len(mismatch):
        print(mismatch[["id", "turn_index", "step"]].head(10))

    # --- budget / elapsed as function of turn_index: check determinism ---
    print("\n=== budget_tokens_remaining by turn_index: mean/std (std=0 would mean deterministic formula) ===")
    print(df.groupby("turn_index")["budget_tokens_remaining"].agg(["mean", "std", "count"]).round(1).head(15))

    print("\n=== elapsed_session_sec by turn_index: mean/std ===")
    print(df.groupby("turn_index")["elapsed_session_sec"].agg(["mean", "std", "count"]).round(1).head(15))

    # --- id numeric parts vs label (leakage sanity check) ---
    parsed = df["id"].str.extract(ID_RE)
    parsed.columns = ["date_part", "serial_part", "step_part"]
    df["id_date"] = parsed["date_part"]
    df["id_serial"] = pd.to_numeric(parsed["serial_part"], errors="coerce")
    print("\n=== unique id_date values (should be a single sim-generation date if all one batch) ===")
    print(df["id_date"].value_counts())

    print("\n=== correlation-ish check: id_serial mod 14 vs action (should look ~uniform if no leak) ===")
    df["serial_mod14"] = df["id_serial"] % 14
    print(pd.crosstab(df["serial_mod14"], df["action"], normalize="index").round(3).to_string())

    # chi-square-ish quick check via max deviation from marginal
    marginal = df["action"].value_counts(normalize=True)
    ct = pd.crosstab(df["serial_mod14"], df["action"], normalize="index")
    max_dev = (ct - marginal).abs().max().max()
    print(f"\nmax abs deviation from marginal across serial_mod14 buckets: {max_dev:.4f}")

    # --- args schema determinism note ---
    print("\n=== args-key schema per action name (from last_args_sig, deterministic mapping check) ===")
    schema = df.groupby("last_action")["last_args_sig"].agg(lambda s: s.value_counts(normalize=True).head(3).to_dict())
    for action, d in schema.items():
        print(f"  {action}: {d}")

    cmp.to_csv(OUT_DIR / "session_termination_dist.csv")
    session_sizes.to_csv(OUT_DIR / "session_size_dist.csv")


if __name__ == "__main__":
    main()
