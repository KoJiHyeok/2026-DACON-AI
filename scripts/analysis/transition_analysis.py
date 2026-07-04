"""
Item 3: Transition structure.

- last_action -> action 14x14 transition matrix, row-normalized.
- Cells where conditioning on last2_action lifts purity meaningfully above the
  last_action-only row max.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from common import load_frame

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
OUT_DIR.mkdir(exist_ok=True)

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash", "run_tests",
    "lint_or_typecheck", "ask_user", "plan_task", "web_search", "respond_only",
]


def main():
    df = load_frame()
    last_states = ["none"] + ACTIONS

    ct = pd.crosstab(df["last_action"], df["action"])
    ct = ct.reindex(index=last_states, columns=ACTIONS, fill_value=0)
    ct = ct[ct.sum(axis=1) > 0]
    row_norm = ct.div(ct.sum(axis=1), axis=0)
    row_norm.to_csv(OUT_DIR / "transition_matrix_last1.csv")

    print("=== last_action row-normalized transition matrix (rounded) ===")
    print(row_norm.round(3).to_string())

    print("\n=== per-row max cell (best single-action prediction from last_action alone) ===")
    row_max = row_norm.max(axis=1)
    row_argmax = row_norm.idxmax(axis=1)
    summary = pd.DataFrame({"best_next": row_argmax, "prob": row_max, "n_rows": ct.sum(axis=1)})
    print(summary.round(4).to_string())

    # last2 conditional: for each (last2, last1) pair with enough rows, does purity
    # rise meaningfully above the last1-only max?
    df["last2_last1"] = list(zip(df["last2_action"], df["last_action"]))
    g = df.groupby(["last2_action", "last_action"])
    recs = []
    for (l2, l1), sub in g:
        if len(sub) < 30:
            continue
        vc = sub["action"].value_counts()
        purity = vc.iloc[0] / len(sub)
        base_purity = row_max.get(l1, 0)
        recs.append({
            "last2_action": l2, "last_action": l1, "n_rows": len(sub),
            "n_sessions": sub["session_key"].nunique(),
            "top_action": vc.index[0], "purity": purity,
            "last1_only_purity": base_purity, "lift": purity - base_purity,
        })
    lift_df = pd.DataFrame(recs).sort_values("lift", ascending=False)
    lift_df.to_csv(OUT_DIR / "last2_conditional_lift.csv", index=False)

    print("\n=== Top 20 (last2, last1) cells by purity lift over last1-only (n_rows>=30) ===")
    print(lift_df.head(20).round(4).to_string(index=False))

    print("\n=== Distribution of lift ===")
    print(lift_df["lift"].describe())


if __name__ == "__main__":
    main()
