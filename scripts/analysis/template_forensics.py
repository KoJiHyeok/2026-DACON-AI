"""
Item 2: current_prompt template forensics.

1. Exact-duplicate current_prompt strings: distribution of duplicate group
   sizes, and per-group label purity.
2. Normalized templates: replace numbers, paths/filenames, and quoted spans
   with placeholders, then repeat the duplicate-group purity analysis on the
   normalized text. If the simulator drew prompts from a fixed template bank
   parameterized by entity substitution, this should collapse many distinct
   current_prompt strings into a modest number of templates with high purity.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_frame

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
OUT_DIR.mkdir(exist_ok=True)

PATH_RE = re.compile(r"[\w\-]+(?:/[\w\-.]+)+|\b[\w\-]+\.[a-zA-Z]{1,6}\b")
NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
QUOTED_RE = re.compile(r"'[^']{1,60}'|\"[^\"]{1,60}\"")
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}(?:[A-Z][a-z0-9_]*){1,}\b")  # camelCase-ish idents


def normalize_prompt(text: str) -> str:
    t = text
    t = QUOTED_RE.sub("<QUOTED>", t)
    t = PATH_RE.sub("<PATH>", t)
    t = NUM_RE.sub("<NUM>", t)
    t = IDENT_RE.sub("<IDENT>", t)
    return t


def group_purity(df: pd.DataFrame, key_col: str, min_rows: int = 2) -> pd.DataFrame:
    g = df.groupby(key_col)
    recs = []
    for key, sub in g:
        if len(sub) < min_rows:
            continue
        vc = sub["action"].value_counts()
        recs.append({
            "key": key,
            "n_rows": len(sub),
            "n_sessions": sub["session_key"].nunique(),
            "top_action": vc.index[0],
            "purity": vc.iloc[0] / len(sub),
        })
    return pd.DataFrame(recs).sort_values("n_rows", ascending=False)


def main():
    df = load_frame()
    total = len(df)

    # --- Exact duplicates ---
    dup_counts = df["current_prompt"].value_counts()
    n_unique = len(dup_counts)
    n_dup_gt1 = (dup_counts > 1).sum()
    rows_in_dup_groups = dup_counts[dup_counts > 1].sum()
    print(f"Total rows: {total}, unique current_prompt strings: {n_unique}")
    print(f"Exact-dup groups (size>1): {n_dup_gt1}, rows covered by them: {rows_in_dup_groups} "
          f"({rows_in_dup_groups/total:.4f})")

    exact_groups = group_purity(df, "current_prompt", min_rows=2)
    exact_groups.to_csv(OUT_DIR / "exact_dup_groups.csv", index=False)

    for thresh in (0.95, 0.99, 1.0):
        sel = exact_groups[exact_groups["purity"] >= thresh]
        cov = sel["n_rows"].sum() / total
        print(f"  exact-dup purity>={thresh}: {len(sel)} groups, coverage {cov:.4f} "
              f"({sel['n_rows'].sum()} rows)")

    print("\nDup group size distribution (size>1 groups):")
    print(dup_counts[dup_counts > 1].describe())

    # --- Normalized templates ---
    df["prompt_template"] = df["current_prompt"].map(normalize_prompt)
    tmpl_counts = df["prompt_template"].value_counts()
    n_templates = len(tmpl_counts)
    print(f"\nNormalized templates: {n_templates} unique (from {n_unique} unique raw prompts)")

    tmpl_groups = group_purity(df, "prompt_template", min_rows=2)
    tmpl_groups.to_csv(OUT_DIR / "template_groups.csv", index=False)

    rows_in_tmpl_groups = tmpl_counts[tmpl_counts > 1].sum()
    print(f"Templates with size>1: {(tmpl_counts>1).sum()}, rows covered: {rows_in_tmpl_groups} "
          f"({rows_in_tmpl_groups/total:.4f})")

    for thresh in (0.95, 0.99, 1.0):
        sel = tmpl_groups[tmpl_groups["purity"] >= thresh]
        cov = sel["n_rows"].sum() / total
        n_sess = df[df["prompt_template"].isin(sel["key"])]["session_key"].nunique()
        print(f"  template purity>={thresh}: {len(sel)} templates, coverage {cov:.4f} "
              f"({sel['n_rows'].sum()} rows, {n_sess} unique sessions)")

    # top templates by row count with their purity / top action
    top_templates = tmpl_groups.head(30)
    print("\nTop 30 templates by frequency:")
    with pd.option_context("display.max_colwidth", 80):
        print(top_templates[["key", "n_rows", "n_sessions", "top_action", "purity"]].to_string(index=False))

    # action distribution *within* single dominant template (sanity: is it
    # actually the SAME template text driving many different actions, i.e.
    # not template-determined at all)
    biggest = tmpl_counts.index[0]
    sub = df[df["prompt_template"] == biggest]
    print(f"\nBiggest template (n={len(sub)}): {biggest!r}")
    print(sub["action"].value_counts())

    meta = {
        "total_rows": total,
        "n_unique_prompts": int(n_unique),
        "n_exact_dup_groups": int(n_dup_gt1),
        "n_templates": int(n_templates),
    }
    with open(OUT_DIR / "template_forensics_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
