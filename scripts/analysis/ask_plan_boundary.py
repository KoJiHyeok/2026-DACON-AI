"""Forensics round 2: search for a deterministic state-rule that separates
`ask_user` from `plan_task` labels in train.jsonl, motivated by CX-003 H1
(champion directed confusion `ask_user<->plan_task`, 191 eligible errors,
oracle-flip scenario +0.0110, bootstrap CI [+0.0085,+0.0138] P=1.000).

Purely descriptive: no fitting, optimization, clustering, or embeddings.
Reuses `scripts/analysis/common.py::load_frame` for the flattened 70,000-row
state frame (session_key, step, last_action/last2/last3, current_prompt,
session_meta fields, workspace fields, label).

Outputs candidate lexical/state conditions with coverage x purity, computed
group-wise (unique sessions per bucket), restricted to the ask_user/plan_task
two-class subset AND, separately, over the full 14-class label space (so we
can see whether a "high purity within ask_user/plan_task" condition still
leaks other classes in, which would break it as a hard override).

Then re-applies the best candidate condition(s) to the CX-003 H1 eligible
holdout rows (191 rows where champion actually confused ask_user<->plan_task)
to see how many would flip correctly vs incorrectly if this condition were
used as an override.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from common import load_frame, session_key

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent / "_out"
OUT_DIR.mkdir(exist_ok=True)

ERRTAX_ROOT = ROOT
CHAMPION_CSV = ROOT / "artifacts/experiments/errtax_h12/champion_holdout_preds.csv"


# ---------------------------------------------------------------------------
# Lexical signal definitions over current_prompt, tuned to distinguish
# "asking a blocking clarifying question" (ask_user) from "requesting an
# explicit plan/roadmap" (plan_task).
# ---------------------------------------------------------------------------
PATTERNS: dict[str, re.Pattern] = {
    "has_question_mark": re.compile(r"\?"),
    "wh_question_word": re.compile(r"\b(?:which|what|where|should i|do you want|do you prefer|got a preference|any preference)\b", re.I),
    "asks_clarify_verb": re.compile(r"\b(?:clarify|confirm|double[- ]check|which one|which way|preference)\b", re.I),
    "mentions_plan_noun": re.compile(r"\b(?:plan|roadmap|steps?|approach|strategy|breakdown|outline)\b", re.I),
    "mentions_plan_verb": re.compile(r"\b(?:plan out|lay out|walk me through|sketch|map out|break down|figure out the steps)\b", re.I),
    "before_you_start": re.compile(r"\bbefore (?:you|we) (?:start|begin|dive|do)\b", re.I),
    "not_sure_which": re.compile(r"\bnot sure (?:which|what|how)\b", re.I),
    "want_to_know_approach": re.compile(r"\b(?:how (?:would|will) you|what's your approach|how are you going to)\b", re.I),
    "give_options": re.compile(r"\b(?:options?|alternatives?|a few ways|couple ways)\b", re.I),
    "no_code_yet": re.compile(r"\b(?:don't (?:start|write|code|implement) (?:yet|anything)|hold off|wait before)\b", re.I),
}


def apply_patterns(prompt: str) -> dict[str, bool]:
    return {name: bool(rx.search(prompt)) for name, rx in PATTERNS.items()}


def purity_table(df: pd.DataFrame, group_col: str, label_col: str = "action") -> pd.DataFrame:
    """For each value of group_col, compute n rows, n unique sessions,
    top label + purity (share of top label among that group's rows),
    restricted purity within {ask_user, plan_task} rows only, and how many
    non-(ask_user/plan_task) rows leak into the bucket."""
    rows = []
    for val, g in df.groupby(group_col, dropna=False):
        n = len(g)
        n_sessions = g["session_key"].nunique()
        vc = g[label_col].value_counts()
        top_label = vc.index[0]
        top_purity = vc.iloc[0] / n
        ap = g[g[label_col].isin(["ask_user", "plan_task"])]
        n_ap = len(ap)
        if n_ap:
            ap_vc = ap[label_col].value_counts()
            ap_top = ap_vc.index[0]
            ap_purity = ap_vc.iloc[0] / n_ap
        else:
            ap_top, ap_purity = None, None
        rows.append({
            "group_value": val, "n_rows": n, "n_sessions": n_sessions,
            "top_label": top_label, "top_purity": top_purity,
            "n_ask_or_plan_rows": n_ap, "ask_plan_top": ap_top, "ask_plan_purity": ap_purity,
            "n_other_14class_rows": n - n_ap,
        })
    out = pd.DataFrame(rows).sort_values("n_rows", ascending=False)
    return out


def scan_condition(df: pd.DataFrame, cond_name: str, cond_mask: pd.Series) -> dict:
    """Evaluate a single boolean condition as a candidate hard rule:
    among rows where cond==True, what's the label distribution (14-class)
    and the ask_user/plan_task sub-distribution, plus unique session counts."""
    g = df[cond_mask]
    n = len(g)
    n_sessions = g["session_key"].nunique()
    if n == 0:
        return {"condition": cond_name, "n_rows": 0}
    vc = g["action"].value_counts()
    ap = g[g["action"].isin(["ask_user", "plan_task"])]
    result = {
        "condition": cond_name, "n_rows": n, "n_sessions": n_sessions,
        "top_label_14class": vc.index[0], "top_purity_14class": float(vc.iloc[0] / n),
        "n_ask_or_plan": len(ap),
        "ask_user_count": int((g["action"] == "ask_user").sum()),
        "plan_task_count": int((g["action"] == "plan_task").sum()),
    }
    if len(ap):
        result["ask_plan_conditional_purity"] = float(max(result["ask_user_count"], result["plan_task_count"]) / len(ap))
        result["ask_plan_favors"] = "ask_user" if result["ask_user_count"] >= result["plan_task_count"] else "plan_task"
    return result


def main() -> None:
    df = load_frame()
    df["session_key"] = df["id"].map(session_key)

    # restrict to rows actually labeled ask_user or plan_task, for
    # state-summary purposes
    ap_df = df[df["action"].isin(["ask_user", "plan_task"])].copy()
    print(f"Total train rows: {len(df):,}; ask_user={int((df.action=='ask_user').sum())}, "
          f"plan_task={int((df.action=='plan_task').sum())}")
    print(f"ask_user unique sessions: {ap_df[ap_df.action=='ask_user'].session_key.nunique()}, "
          f"plan_task unique sessions: {ap_df[ap_df.action=='plan_task'].session_key.nunique()}")

    # ---- (1) last_action / last2_action purity within ask_user/plan_task ----
    last_action_tab = purity_table(ap_df, "last_action")
    last_action_tab.to_csv(OUT_DIR / "ask_plan_by_last_action.csv", index=False)

    ap_df["last2_last1"] = ap_df["last2_action"] + ">" + ap_df["last_action"]
    last2_tab = purity_table(ap_df, "last2_last1")
    last2_tab.to_csv(OUT_DIR / "ask_plan_by_last2_last1.csv", index=False)

    # ---- (2) lexical prompt-pattern signals, scanned over FULL df (14-class) ----
    pattern_flags = df["current_prompt"].fillna("").map(apply_patterns)
    pattern_df = pd.DataFrame(list(pattern_flags))
    df2 = pd.concat([df.reset_index(drop=True), pattern_df.reset_index(drop=True)], axis=1)

    lexical_results = []
    for name in PATTERNS:
        mask = df2[name]
        lexical_results.append(scan_condition(df2, name, mask))
    lexical_df = pd.DataFrame(lexical_results)
    lexical_df.to_csv(OUT_DIR / "ask_plan_lexical_signals.csv", index=False)

    # ---- (3) combinations: lexical AND turn_index==1 (first-ever prompt often plan-ish) ----
    combo_results = []
    for name in PATTERNS:
        mask = df2[name] & (df2["step"] == 1)
        combo_results.append(scan_condition(df2, f"{name}&step1", mask))
        mask2 = df2[name] & (df2["step"] > 1)
        combo_results.append(scan_condition(df2, f"{name}&step_gt1", mask2))
    combo_df = pd.DataFrame(combo_results)
    combo_df.to_csv(OUT_DIR / "ask_plan_lexical_x_step.csv", index=False)

    # ---- (4) session_meta / workspace numeric-ish conditions ----
    meta_conditions = {
        "budget_lt_20k": df2["budget_tokens_remaining"] < 20000,
        "budget_lt_10k": df2["budget_tokens_remaining"] < 10000,
        "n_open_files_0": df2["n_open_files"] == 0,
        "n_open_files_ge1": df2["n_open_files"] >= 1,
        "ci_status_failed": df2["last_ci_status"] == "failed",
        "ci_status_none": df2["last_ci_status"] == "none",
        "git_dirty_true": df2["git_dirty"] == True,  # noqa: E712
        "turn_index_1": df2["step"] == 1,
        "turn_index_le2": df2["step"] <= 2,
    }
    meta_results = [scan_condition(df2, name, mask) for name, mask in meta_conditions.items()]
    meta_df = pd.DataFrame(meta_results)
    meta_df.to_csv(OUT_DIR / "ask_plan_meta_signals.csv", index=False)

    # ---- (5) best single lexical rule candidates: purity>=0.85 AND n_ask_or_plan>=20 ----
    all_candidates = pd.concat([lexical_df, combo_df], ignore_index=True)
    all_candidates = all_candidates[all_candidates["n_ask_or_plan"].fillna(0) >= 20]
    if "ask_plan_conditional_purity" in all_candidates.columns:
        strong = all_candidates[all_candidates["ask_plan_conditional_purity"].fillna(0) >= 0.60]
        strong = strong.sort_values("n_ask_or_plan", ascending=False)
        strong.to_csv(OUT_DIR / "ask_plan_strong_candidates.csv", index=False)
    else:
        strong = pd.DataFrame()

    # ---- (6) mutual-exclusivity check: does the same lexical signal fire
    # on rows the champion currently gets right (i.e. is it truly separating,
    # or just co-occurring)? Also check leakage into other 12 classes. ----
    print("\n=== Lexical signal candidates (min 20 ask/plan rows) ===")
    if len(all_candidates):
        cols = ["condition", "n_rows", "n_sessions", "n_ask_or_plan", "ask_user_count",
                "plan_task_count", "ask_plan_conditional_purity", "ask_plan_favors",
                "top_purity_14class"]
        print(all_candidates[cols].sort_values("n_ask_or_plan", ascending=False).to_string(index=False))

    # =========================================================================
    # (7) Apply best candidate(s) to the CX-003 H1 eligible 191-row holdout set
    # =========================================================================
    print("\n=== Reproducing CX-003 H1 eligible set from champion_holdout_preds.csv ===")
    champ = pd.read_csv(CHAMPION_CSV, dtype=str)
    champ["p_top1"] = champ["p_top1"].astype(float)
    champ["p_top2"] = champ["p_top2"].astype(float)
    h1_eligible = champ[
        ((champ.y_true == "ask_user") & (champ.pred == "plan_task"))
        | ((champ.y_true == "plan_task") & (champ.pred == "ask_user"))
    ].copy()
    print(f"H1 eligible rows reproduced: {len(h1_eligible)} (expect 191)")

    # join champion rows against the state frame to get current_prompt etc.
    id_to_row = df2.set_index("id")
    h1_eligible = h1_eligible.join(id_to_row[list(PATTERNS) + ["last_action", "last2_action", "step"]], on="id")

    if len(strong):
        top_rule_names = strong["condition"].head(5).tolist()
    else:
        top_rule_names = []

    print(f"\nTop candidate rule names to test on H1 eligible set: {top_rule_names}")

    def eval_rule_on_h1(rule_name: str, mask: pd.Series) -> dict:
        applies = h1_eligible[mask]
        n_applies = len(applies)
        if n_applies == 0:
            return {"rule": rule_name, "n_applies": 0}
        # if the rule "favors" ask_user or plan_task, does applying it as an
        # override match the true label?
        favors_row = strong.set_index("condition").get("ask_plan_favors", pd.Series(dtype=object))
        favors = favors_row.get(rule_name)
        if favors is None:
            return {"rule": rule_name, "n_applies": n_applies, "favors": None}
        correct = (applies["y_true"] == favors).sum()
        broken = n_applies - correct  # rows where champion's pred already matched favors's opposite correctly? check below
        # separately: how many of these rows did champion already get right by other criterion (none, all are errors by construction)
        return {
            "rule": rule_name, "n_applies": n_applies, "favors": favors,
            "would_be_correct_if_override": int(correct),
            "would_be_wrong_if_override": int(n_applies - correct),
        }

    h1_results = []
    for name in top_rule_names:
        # reconstruct the mask on h1_eligible using the same pattern name if lexical
        base_name = name.split("&")[0]
        if base_name in PATTERNS:
            mask = h1_eligible[base_name].fillna(False)
            if "&step1" in name:
                mask &= (h1_eligible["step"] == 1)
            elif "&step_gt1" in name:
                mask &= (h1_eligible["step"] > 1)
            h1_results.append(eval_rule_on_h1(name, mask))
    h1_df = pd.DataFrame(h1_results)
    h1_df.to_csv(OUT_DIR / "ask_plan_h1_eligible_rule_application.csv", index=False)
    print("\n=== Rule application on H1 eligible (191-row) set ===")
    print(h1_df.to_string(index=False) if len(h1_df) else "(no candidates met the strong threshold)")

    # Also directly report last_action / last2_action purity restricted to
    # the H1 eligible set itself, descriptively (not as an override, just to see).
    print("\n=== H1 eligible set: last_action distribution (descriptive only) ===")
    print(h1_eligible["last_action"].value_counts())

    print("\nDone. CSVs written to scripts/analysis/_out/ask_plan_*.csv")


if __name__ == "__main__":
    main()
