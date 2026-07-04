"""
Item 4: Discriminative signals for the 4 exploration classes
(read_file, grep_search, list_directory, glob_pattern) -- the team's known
per-class weak spot.

For each candidate signal (regex on current_prompt, last action's args keys,
session_meta fields), report:
  - P(class | signal present) restricted to rows whose true label is one of
    the 4 exploration classes (so we isolate the discrimination problem the
    team actually has, rather than explore-vs-everything).
  - Precision / recall of "signal -> predicted class" as a simple one-off rule
    within the exploration-only subset.
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

EXPLORE = ["read_file", "grep_search", "list_directory", "glob_pattern"]

# Candidate prompt-level regex signals.
SIGNALS = {
    "has_glob_star": r"\*\.\w+|\*\*|\{[\w,]+\}",
    "has_quoted_term": r"['\"][^'\"]{1,40}['\"]",
    "mentions_extension_list": r"\.\w{1,5}(?:,|\s+(?:and|or|,))\s*\.\w{1,5}",
    "mentions_file_singular": r"\bfile\b|파일(?!들)",
    "mentions_files_plural": r"\bfiles\b|파일들",
    "mentions_directory": r"\bdirector(y|ies)\b|\bfolder\b|디렉토리|폴더",
    "mentions_search_verb": r"\bsearch\b|\bgrep\b|\bfind\b|찾아|검색",
    "mentions_list_verb": r"\blist\b|목록|나열",
    "mentions_open_show": r"\bopen\b|\bshow\b|\bcat\b|열어|보여|읽어",
    "has_explicit_path": r"[\w\-]+/[\w\-./]+",
    "has_bare_filename": r"\b[\w\-]+\.(py|js|ts|tsx|jsx|json|yaml|yml|md|toml|txt|go|rs|java|rb|css|html|cfg|ini|sql|c|cpp|h)\b",
    "mentions_pattern_word": r"\bpattern\b|패턴",
    "mentions_definition": r"defin(e|ition)|어디.*(정의|사용)|where.*(defined|used)",
    "mentions_all_of_type": r"\ball\b.*\bfiles\b|모든\s*파일|전체\s*파일",
}


def signal_report(df: pd.DataFrame, col: str, name: str) -> list[dict]:
    sub = df[df["action"].isin(EXPLORE)]
    mask = sub[col]
    recs = []
    if mask.sum() < 20:
        return recs
    on = sub[mask]
    dist = on["action"].value_counts(normalize=True)
    top = dist.index[0]
    recs.append({
        "signal": name,
        "n_rows_on": int(mask.sum()),
        "coverage_of_explore": mask.sum() / len(sub),
        "top_class_given_on": top,
        "precision_on": dist.iloc[0],
        "recall_of_top_class": (on["action"] == top).sum() / (sub["action"] == top).sum(),
    })
    return recs


def main():
    df = load_frame()
    cp = df["current_prompt"].fillna("")

    for name, pattern in SIGNALS.items():
        df[name] = cp.str.contains(pattern, regex=True, case=False)

    # history-based signals: last args sig / last action itself already covered
    # in transition_analysis; add "last action had a 'pattern' arg key" etc.
    df["last_args_had_pattern"] = df["last_args"].str.contains('"pattern"', regex=False, na=False)
    df["last_args_had_path"] = df["last_args"].str.contains('"path"', regex=False, na=False)
    df["last_args_had_scope"] = df["last_args"].str.contains('"scope"', regex=False, na=False)
    df["n_open_files_gt0"] = df["n_open_files"] > 0

    all_signal_cols = list(SIGNALS.keys()) + [
        "last_args_had_pattern", "last_args_had_path", "last_args_had_scope", "n_open_files_gt0",
    ]

    records = []
    for col in all_signal_cols:
        records.extend(signal_report(df, col, col))
    rep = pd.DataFrame(records).sort_values("precision_on", ascending=False)
    rep.to_csv(OUT_DIR / "exploration_signal_report.csv", index=False)
    print("=== Signal precision within exploration-only subset (n_explore_total={}) ===".format(
        (df["action"].isin(EXPLORE)).sum()))
    print(rep.round(4).to_string(index=False))

    # baseline: exploration class prior (no signal) for comparison
    base = df[df["action"].isin(EXPLORE)]["action"].value_counts(normalize=True)
    print("\n=== Baseline class prior within exploration subset ===")
    print(base.round(4))

    # Pairwise: for each pair of explore classes, find the signal with the
    # biggest precision gap between the two (most discriminative single signal)
    print("\n=== Best discriminating signal per class-pair ===")
    pair_records = []
    for i, a in enumerate(EXPLORE):
        for b in EXPLORE[i + 1:]:
            sub = df[df["action"].isin([a, b])]
            best = None
            for col in all_signal_cols:
                on = sub[sub[col]]
                if len(on) < 15:
                    continue
                pa = (on["action"] == a).mean()
                gap = abs(pa - 0.5)
                if best is None or gap > best[1]:
                    best = (col, gap, len(on), pa)
            if best:
                pair_records.append({
                    "class_a": a, "class_b": b, "best_signal": best[0],
                    "n_rows_on": best[2], "P(a|signal)": round(best[3], 3),
                    "gap_from_50_50": round(best[1], 3),
                })
    pair_df = pd.DataFrame(pair_records)
    print(pair_df.to_string(index=False))
    pair_df.to_csv(OUT_DIR / "exploration_pairwise_signals.csv", index=False)


if __name__ == "__main__":
    main()
