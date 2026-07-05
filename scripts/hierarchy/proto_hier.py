"""Prototype R4: flat 14-way baseline vs explore hierarchy.

Runs 5-fold StratifiedGroupKFold by session id and compares:
  A) flat 14-way LinearSVC(C=0.1, class_weight="balanced")
  B1) stage-1 family gate; if predicted explore, replace with explore branch
  B2) same gate, but non-explore rows are constrained to the predicted family

The script is deliberately submission-independent. It is a local CV probe for
the R1 forensics lead that last2/last1 is useful only after an explore gate.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.base import clone
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
FALLBACK_DATA_ROOT = Path(r"C:\dev\2026-AI-DACON\data")
DEFAULT_OUT_DIR = ROOT / "scripts" / "hierarchy" / "_out"

ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]
EXPLORE = {"read_file", "grep_search", "list_directory", "glob_pattern"}
MUTATE = {"edit_file", "write_file", "apply_patch"}
VALIDATE = {"run_bash", "run_tests", "lint_or_typecheck"}
COORDINATE = {"ask_user", "plan_task", "web_search", "respond_only"}
FAMILIES = {
    "explore": EXPLORE,
    "mutate": MUTATE,
    "validate": VALIDATE,
    "coordinate": COORDINATE,
}

STEP_RE = re.compile(r"-step_\d+$")
GLOB_RE = re.compile(r"\*\*|\*\.\w+|/\*|\*/")
EXT_RE = re.compile(r"\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
PATH_RE = re.compile(r"[\w\-/]+/[\w\-./]+|[\w\-]+\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
GREP_RE = re.compile(r"grep|search|find|찾|훑|검색|where\b|locate|어디", re.I)
LIST_RE = re.compile(r"\blist\b|\bls\b|디렉토리|폴더|목록|안에 뭐|무슨 파일|what.s in", re.I)
READ_RE = re.compile(r"open|열어|보여|봐줘|\bshow\b|\bread\b|읽어|내용|뭐라고", re.I)


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def action_family(action: str) -> str:
    for family, members in FAMILIES.items():
        if action in members:
            return family
    raise ValueError(f"unknown action: {action}")


def _s(value: object) -> str:
    return value if isinstance(value, str) else ("" if value is None else str(value))


def _dominant_lang(workspace: dict) -> str:
    mix = (workspace or {}).get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return max(mix.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)[0]


def _row(sample: dict) -> dict:
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    history = sample.get("history") or []
    prompt = _s(sample.get("current_prompt"))

    user_texts: list[str] = []
    action_names: list[str] = []
    last_user = ""
    for item in history:
        if not isinstance(item, dict):
            continue
        if item.get("role") == "assistant_action":
            name = _s(item.get("name"))
            if name:
                action_names.append(name)
        elif item.get("role") == "user":
            last_user = _s(item.get("content"))
            user_texts.append(last_user)

    counts = Counter(action_names)
    last_action = action_names[-1] if action_names else "none"
    last2_action = action_names[-2] if len(action_names) >= 2 else "none"
    last3_action = action_names[-3] if len(action_names) >= 3 else "none"

    open_files = workspace.get("open_files") or []
    if not isinstance(open_files, list):
        open_files = []
    low_prompt = prompt.lower()
    file_in_open = 0
    for open_file in open_files:
        open_file_s = _s(open_file)
        base = open_file_s.split("/")[-1].lower()
        if open_file_s and (
            open_file_s.lower() in low_prompt or (len(base) > 2 and base in low_prompt)
        ):
            file_in_open = 1
            break

    row = {
        "id": _s(sample.get("id")),
        "session_id": session_id(_s(sample.get("id"))),
        "current_prompt": prompt,
        "history_text": " ".join(user_texts),
        "last_user_msg": last_user,
        "hist_action_seq": " ".join(action_names) if action_names else "none",
        "last_action": last_action,
        "last2_action": last2_action,
        "last3_action": last3_action,
        "last2_bigram": f"{last2_action}_{last_action}",
        "user_tier": _s(meta.get("user_tier")) or "none",
        "language_pref": _s(meta.get("language_pref")) or "none",
        "last_ci_status": _s(workspace.get("last_ci_status")) or "none",
        "dominant_ws_lang": _dominant_lang(workspace),
        "n_history": len(history),
        "n_action_turns": len(action_names),
        "turn_index": float(meta.get("turn_index") or 0),
        "elapsed_log": np.log1p(float(meta.get("elapsed_session_sec") or 0)),
        "budget_log": np.log1p(float(meta.get("budget_tokens_remaining") or 0)),
        "loc_log": np.log1p(float(workspace.get("loc") or 0)),
        "n_open_files": len(open_files),
        "git_dirty": 1 if workspace.get("git_dirty") else 0,
        "has_glob": 1 if GLOB_RE.search(prompt) else 0,
        "has_ext": 1 if EXT_RE.search(prompt) else 0,
        "has_grep": 1 if GREP_RE.search(prompt) else 0,
        "has_list_word": 1 if LIST_RE.search(prompt) else 0,
        "has_read_word": 1 if READ_RE.search(prompt) else 0,
        "has_path": 1 if PATH_RE.search(prompt) else 0,
        "has_quote": 1 if any(ch in prompt for ch in ("'", '"', "`")) else 0,
        "n_slash": prompt.count("/"),
        "n_star": prompt.count("*") + prompt.count("?"),
        "has_regex_meta": 1 if re.search(r"[\^$|\\]|\\s|\\d|\[.*\]", prompt) else 0,
        "file_in_open": file_in_open,
    }
    for action in ACTIONS:
        row[f"act_{action}"] = counts.get(action, 0)
    return row


def resolve_data_root(data_root: str | None) -> Path:
    if data_root:
        return Path(data_root)
    local = ROOT / "data"
    if (local / "train.jsonl").exists():
        return local
    return FALLBACK_DATA_ROOT


def load_training_frame(data_root: Path, max_rows: int | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    train_jsonl = data_root / "train.jsonl"
    labels_csv = data_root / "train_labels.csv"
    if not train_jsonl.exists() or not labels_csv.exists():
        raise FileNotFoundError(f"missing train files under {data_root}")

    labels = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            labels[row["id"]] = row["action"]

    rows = []
    y = []
    with train_jsonl.open("r", encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            if max_rows is not None and i >= max_rows:
                break
            if not line.strip():
                continue
            sample = json.loads(line)
            rows.append(_row(sample))
            y.append(labels[sample["id"]])
    return pd.DataFrame(rows), np.asarray(y)


def build_preprocessor() -> ColumnTransformer:
    cat_cols = [
        "user_tier",
        "language_pref",
        "last_ci_status",
        "dominant_ws_lang",
        "last_action",
        "last2_action",
        "last3_action",
        "last2_bigram",
    ]
    num_cols = [
        "n_history",
        "n_action_turns",
        "turn_index",
        "elapsed_log",
        "budget_log",
        "loc_log",
        "n_open_files",
        "git_dirty",
        "has_glob",
        "has_ext",
        "has_grep",
        "has_list_word",
        "has_read_word",
        "has_path",
        "has_quote",
        "n_slash",
        "n_star",
        "has_regex_meta",
        "file_in_open",
        *[f"act_{action}" for action in ACTIONS],
    ]
    return ColumnTransformer(
        [
            (
                "prompt_word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=40_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
                "current_prompt",
            ),
            (
                "prompt_char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=3,
                    max_features=25_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
                "current_prompt",
            ),
            (
                "history_word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=3,
                    max_features=15_000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
                "history_text",
            ),
            (
                "action_seq",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 3),
                    min_df=3,
                    max_features=5_000,
                    lowercase=False,
                ),
                "hist_action_seq",
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cat_cols),
            ("num", StandardScaler(with_mean=False), num_cols),
        ],
        sparse_threshold=0.3,
    )


def build_classifier() -> LinearSVC:
    return LinearSVC(
        C=0.1,
        class_weight="balanced",
        max_iter=1000,
        tol=1e-3,
        dual="auto",
        random_state=42,
    )


def build_pipeline() -> Pipeline:
    return Pipeline([("features", build_preprocessor()), ("clf", build_classifier())])


def _decision_frame(pipe: Pipeline, x_valid: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    scores = pipe.decision_function(x_valid)
    classes = list(pipe.named_steps["clf"].classes_)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores, classes


def _decision_matrix(clf: LinearSVC, x_valid) -> tuple[np.ndarray, list[str]]:
    scores = clf.decision_function(x_valid)
    classes = list(clf.classes_)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores, classes


def _top_in_family(scores: np.ndarray, classes: list[str], family: str) -> str:
    allowed = FAMILIES[family]
    indexes = [i for i, cls in enumerate(classes) if cls in allowed]
    if not indexes:
        raise ValueError(f"no class indexes for family={family}")
    best_local = indexes[int(np.argmax(scores[indexes]))]
    return classes[best_local]


def run_cv(
    df: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    seed: int,
    out_dir: Path,
    only_folds: set[int] | None = None,
) -> None:
    groups = df["session_id"].to_numpy()
    y_family = np.asarray([action_family(action) for action in y])
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_rows = []
    per_class_rows = []
    split_rows = []
    oof_rows = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(df, y, groups), start=1):
        if only_folds and fold not in only_folds:
            continue
        x_train = df.iloc[train_idx].reset_index(drop=True)
        x_valid = df.iloc[valid_idx].reset_index(drop=True)
        y_train = y[train_idx]
        y_valid = y[valid_idx]
        y_family_train = y_family[train_idx]

        preprocessor = build_preprocessor()
        x_train_matrix = preprocessor.fit_transform(x_train)
        x_valid_matrix = preprocessor.transform(x_valid)

        flat = build_classifier()
        flat.fit(x_train_matrix, y_train)
        flat_pred = flat.predict(x_valid_matrix)
        flat_scores, flat_classes = _decision_matrix(flat, x_valid_matrix)

        family_clf = clone(build_classifier())
        family_clf.fit(x_train_matrix, y_family_train)
        family_pred = family_clf.predict(x_valid_matrix)

        explore_mask_train = np.isin(y_train, list(EXPLORE))
        explore_clf = clone(build_classifier())
        explore_clf.fit(x_train_matrix[explore_mask_train], y_train[explore_mask_train])
        explore_pred_all = explore_clf.predict(x_valid_matrix)

        override_pred = flat_pred.copy()
        gate_explore = family_pred == "explore"
        override_pred[gate_explore] = explore_pred_all[gate_explore]

        route_pred = np.empty_like(flat_pred)
        for i, family in enumerate(family_pred):
            if family == "explore":
                route_pred[i] = explore_pred_all[i]
            else:
                route_pred[i] = _top_in_family(flat_scores[i], flat_classes, family)

        preds = {
            "flat_14way": flat_pred,
            "hier_explore_override": override_pred,
            "hier_family_route": route_pred,
        }
        for model_name, pred in preds.items():
            fold_rows.append(
                {
                    "fold": fold,
                    "model": model_name,
                    "macro_f1": f1_score(y_valid, pred, labels=ACTIONS, average="macro", zero_division=0),
                    "explore_macro_f1": f1_score(
                        y_valid,
                        pred,
                        labels=sorted(EXPLORE),
                        average="macro",
                        zero_division=0,
                    ),
                }
            )
            class_f1 = f1_score(y_valid, pred, labels=ACTIONS, average=None, zero_division=0)
            for action, value in zip(ACTIONS, class_f1):
                per_class_rows.append(
                    {
                        "fold": fold,
                        "model": model_name,
                        "action": action,
                        "f1": value,
                    }
                )

        split_rows.append(
            {
                "fold": fold,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "n_train_groups": len(set(groups[train_idx])),
                "n_valid_groups": len(set(groups[valid_idx])),
                "valid_explore_rate": float(np.mean(np.isin(y_valid, list(EXPLORE)))),
                "stage1_family_macro_f1": f1_score(
                    y_family[valid_idx],
                    family_pred,
                    labels=list(FAMILIES),
                    average="macro",
                    zero_division=0,
                ),
                "stage1_explore_precision": _precision_for_label(y_family[valid_idx], family_pred, "explore"),
                "stage1_explore_recall": _recall_for_label(y_family[valid_idx], family_pred, "explore"),
            }
        )
        for row_id, truth, flat_p, override_p, route_p, fam_p in zip(
            x_valid["id"],
            y_valid,
            flat_pred,
            override_pred,
            route_pred,
            family_pred,
        ):
            oof_rows.append(
                {
                    "fold": fold,
                    "id": row_id,
                    "truth": truth,
                    "family_pred": fam_p,
                    "flat_14way": flat_p,
                    "hier_explore_override": override_p,
                    "hier_family_route": route_p,
                }
            )

        print(
            f"fold {fold}: flat={fold_rows[-3]['macro_f1']:.5f} "
            f"override={fold_rows[-2]['macro_f1']:.5f} route={fold_rows[-1]['macro_f1']:.5f} "
            f"stage1={split_rows[-1]['stage1_family_macro_f1']:.5f}",
            flush=True,
        )
        _write_outputs(out_dir, fold_rows, per_class_rows, split_rows, oof_rows, partial=True)

    _write_outputs(out_dir, fold_rows, per_class_rows, split_rows, oof_rows, partial=False)


def _write_outputs(
    out_dir: Path,
    fold_rows: list[dict],
    per_class_rows: list[dict],
    split_rows: list[dict],
    oof_rows: list[dict],
    partial: bool,
) -> None:
    suffix = "_partial" if partial else ""
    pd.DataFrame(fold_rows).to_csv(out_dir / "proto_hier_fold_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(out_dir / "proto_hier_per_class_f1.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out_dir / "proto_hier_split_stats.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(out_dir / "proto_hier_oof_predictions.csv", index=False)
    if partial:
        pd.DataFrame(fold_rows).to_csv(out_dir / f"proto_hier_fold_metrics{suffix}.csv", index=False)
        pd.DataFrame(per_class_rows).to_csv(out_dir / f"proto_hier_per_class_f1{suffix}.csv", index=False)
        pd.DataFrame(split_rows).to_csv(out_dir / f"proto_hier_split_stats{suffix}.csv", index=False)

    summary = (
        pd.DataFrame(fold_rows)
        .groupby("model", as_index=False)
        .agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            explore_macro_f1_mean=("explore_macro_f1", "mean"),
            explore_macro_f1_std=("explore_macro_f1", "std"),
        )
    )
    summary.to_csv(out_dir / "proto_hier_summary.csv", index=False)
    if not partial:
        print(summary.to_string(index=False))


def _precision_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    pred_mask = y_pred == label
    if not np.any(pred_mask):
        return 0.0
    return float(np.mean(y_true[pred_mask] == label))


def _recall_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    true_mask = y_true == label
    if not np.any(true_mask):
        return 0.0
    return float(np.mean(y_pred[true_mask] == label))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None, help="Directory containing train.jsonl and train_labels.csv")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Debug only; keeps first N rows")
    parser.add_argument("--only-folds", default=None, help="Comma-separated fold numbers to run, e.g. 1,3")
    args = parser.parse_args()

    data_root = resolve_data_root(args.data_root)
    print(f"data_root={data_root}")
    df, y = load_training_frame(data_root, max_rows=args.max_rows)
    print(f"loaded rows={len(df):,} groups={df['session_id'].nunique():,}")
    only_folds = {int(x) for x in args.only_folds.split(",")} if args.only_folds else None
    run_cv(df, y, n_splits=args.folds, seed=args.seed, out_dir=Path(args.out_dir), only_folds=only_folds)


if __name__ == "__main__":
    main()
