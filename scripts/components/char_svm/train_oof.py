# -*- coding: utf-8 -*-
"""Char n-gram LinearSVC OOF experiment for night task1.

This script is intentionally self-contained because the experiment is a
candidate fourth component, not a change to the production linear pipeline.
It reads training data and existing league/OFF probabilities from the main
repo, but writes all outputs under the current task workspace.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[3]
MAIN_ROOT = Path(r"C:\dev\2026-AI-DACON")

DEFAULT_DATA_DIR = MAIN_ROOT / "data"
DEFAULT_HOLDOUT_BASE = MAIN_ROOT / "context/night/2026-07-05/holdout_base.npz"
DEFAULT_OOF_DIR = MAIN_ROOT / "artifacts/oof/oof_rebuild_2026_07_04"
DEFAULT_OUT_DIR = ROOT / "night_out/task1"
DEFAULT_CONTEXT_DIR = ROOT / "context/night/2026-07-06"

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
EXPLORE4 = ["read_file", "grep_search", "list_directory", "glob_pattern"]
STEP_RE = re.compile(r"-step_\d+$")
BASELINE_EXPECTED = 0.71726


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def read_labels(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = {str(r["id"]): str(r["action"]) for r in csv.DictReader(f)}
    bad = sorted(set(rows.values()) - set(ACTIONS))
    if bad:
        raise ValueError(f"unknown labels in train_labels.csv: {bad}")
    return rows


def clean_value(value: Any, max_chars: int = 4000) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + " ... " + text[-half:]
    return text


def bucket_number(value: Any, edges: tuple[float, ...]) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(x):
        return "missing"
    for edge in edges:
        if x <= edge:
            return f"le_{int(edge)}"
    return f"gt_{int(edges[-1])}"


def extract_action_sequence(history: Any) -> list[str]:
    seq: list[str] = []
    if not isinstance(history, list):
        return seq
    for item in history:
        if not isinstance(item, dict) or item.get("role") != "assistant_action":
            continue
        name = item.get("name") or item.get("action") or item.get("tool")
        if name:
            seq.append(str(name))
    return seq


def serialize_char_view(sample: dict[str, Any]) -> str:
    """Sample -> text for the independent char component.

    The view keeps the current prompt verbatim, but makes history compact:
    only assistant action tokens and their order are exposed. Session/workspace
    metadata is rendered as key-value strings so char n-grams can learn file
    extensions, path fragments, language tags, and numeric buckets.
    """
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    actions = extract_action_sequence(history)

    action_tokens = " ".join(f"act:{name}" for name in actions) or "act:none"
    recent_tokens = " ".join(f"recent_act:{name}" for name in actions[-8:]) or "recent_act:none"
    pair_tokens = " ".join(
        f"pair:{left}>{right}" for left, right in zip(actions[-8:], actions[-7:])
    ) or "pair:none"

    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    open_ext = []
    for item in open_files[:12]:
        text = str(item)
        open_ext.append(text.rsplit(".", 1)[-1].lower() if "." in text else "none")

    language_mix = workspace.get("language_mix") if isinstance(workspace.get("language_mix"), dict) else {}
    if language_mix:
        top_lang = max(
            language_mix.items(),
            key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else -1,
        )[0]
    else:
        top_lang = "none"

    meta_parts = [
        f"user_tier={clean_value(meta.get('user_tier'), 80) or 'none'}",
        f"language_pref={clean_value(meta.get('language_pref'), 80) or 'none'}",
        f"turn_index={clean_value(meta.get('turn_index'), 80) or 'missing'}",
        f"turn_bin={bucket_number(meta.get('turn_index'), (0, 1, 2, 4, 8, 16, 32))}",
        f"elapsed_sec={clean_value(meta.get('elapsed_session_sec'), 80) or 'missing'}",
        f"elapsed_bin={bucket_number(meta.get('elapsed_session_sec'), (30, 60, 120, 300, 600, 1200))}",
        f"budget={clean_value(meta.get('budget_tokens_remaining'), 80) or 'missing'}",
        f"budget_bin={bucket_number(meta.get('budget_tokens_remaining'), (512, 1024, 2048, 4096, 8192, 32768, 131072))}",
        f"workspace_loc={clean_value(workspace.get('loc'), 80) or 'missing'}",
        f"loc_bin={bucket_number(workspace.get('loc'), (100, 1000, 5000, 20000, 100000))}",
        f"git_dirty={int(bool(workspace.get('git_dirty')))}",
        f"last_ci_status={clean_value(workspace.get('last_ci_status'), 80) or 'none'}",
        f"top_lang={clean_value(top_lang, 80)}",
        "language_mix=" + clean_value(language_mix, 500),
        "open_files=" + " ".join(clean_value(x, 240) for x in open_files[:12]),
        "open_ext=" + " ".join(open_ext),
    ]

    return "\n".join(
        [
            "[CURRENT_PROMPT]",
            clean_value(sample.get("current_prompt"), max_chars=0),
            "[HISTORY_ACTIONS]",
            action_tokens,
            recent_tokens,
            pair_tokens,
            f"history_len={len(history)} action_count={len(actions)} last_action={actions[-1] if actions else 'none'}",
            "[SESSION_META]",
            " ".join(meta_parts),
        ]
    )


def load_training(data_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, list[str]]:
    samples = read_jsonl(data_dir / "train.jsonl")
    labels = read_labels(data_dir / "train_labels.csv")
    ids = np.array([str(s.get("id")) for s in samples], dtype=object)
    missing = [sample_id for sample_id in ids if sample_id not in labels]
    if missing:
        raise ValueError(f"{len(missing)} training ids missing labels, e.g. {missing[:3]}")
    y = np.array([labels[str(sample_id)] for sample_id in ids], dtype=object)
    groups = np.array([session_id(str(sample_id)) for sample_id in ids], dtype=object)
    texts = [serialize_char_view(s) for s in samples]
    if len(samples) != 70000:
        print(f"[warn] expected 70000 train rows, got {len(samples)}")
    return samples, ids, y, groups, texts


def make_splits(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    for fold, (tr_idx, va_idx) in enumerate(splits):
        overlap = set(groups[tr_idx]) & set(groups[va_idx])
        if overlap:
            raise RuntimeError(f"fold {fold} session leakage, e.g. {list(overlap)[:3]}")
    return splits


def build_vectorizer(args: argparse.Namespace) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(args.ngram_min, args.ngram_max),
        max_features=args.max_features,
        sublinear_tf=True,
        lowercase=True,
        dtype=np.float32,
    )


def build_classifier(args: argparse.Namespace) -> LinearSVC:
    class_weight = None if args.class_weight == "none" else args.class_weight
    return LinearSVC(
        C=args.c,
        class_weight=class_weight,
        max_iter=args.max_iter,
        tol=args.tol,
        dual=True,
        random_state=args.seed,
    )


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64)
    scores -= scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / exp.sum(axis=1, keepdims=True)


def decision_probs(clf: LinearSVC, x_val: Any, dst_actions: list[str]) -> np.ndarray:
    scores = clf.decision_function(x_val)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    src_classes = [str(c) for c in clf.classes_]
    p_src = softmax(scores)
    out = np.zeros((p_src.shape[0], len(dst_actions)), dtype=np.float32)
    for src_i, label in enumerate(src_classes):
        if label in dst_actions:
            out[:, dst_actions.index(label)] = p_src[:, src_i].astype(np.float32)
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise RuntimeError("empty probability row after class alignment")
    out /= row_sum
    return out


def labels_from_probs(probs: np.ndarray, actions: list[str]) -> np.ndarray:
    return np.array(actions, dtype=object)[np.asarray(probs).argmax(axis=1)]


def f1_macro(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str]) -> dict[str, float]:
    values = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {label: float(value) for label, value in zip(labels, values)}


def meta_json(args: argparse.Namespace, extra: dict[str, Any] | None = None) -> str:
    payload: dict[str, Any] = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "vectorizer": {
            "analyzer": "char_wb",
            "ngram_range": [args.ngram_min, args.ngram_max],
            "max_features": args.max_features,
            "sublinear_tf": True,
            "lowercase": True,
            "dtype": "float32",
        },
        "classifier": {
            "type": "LinearSVC",
            "C": args.c,
            "class_weight": None if args.class_weight == "none" else args.class_weight,
            "max_iter": args.max_iter,
            "tol": args.tol,
            "dual": True,
            "seed": args.seed,
        },
        "n_folds": args.n_folds,
        "data_dir": str(args.data_dir),
    }
    if extra:
        payload.update(extra)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def fold_npz_path(out_dir: Path, fold: int) -> Path:
    return out_dir / f"oof_fold{fold}.npz"


def fold_meta_path(out_dir: Path, fold: int) -> Path:
    return out_dir / f"oof_fold{fold}_meta.json"


def update_progress(context_dir: Path, out_dir: Path, n_folds: int, next_step: str) -> None:
    context_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# PROGRESS-task1",
        "",
        f"- last_update: {datetime.now().isoformat(timespec='seconds')}",
        "- worktree: `C:\\dev\\night\\2026-07-06\\task1`",
        "- data/oof source: `C:\\dev\\2026-AI-DACON` (read-only)",
        "",
        "## Checklist",
        "",
        "- [x] script scaffolded: `scripts/components/char_svm/train_oof.py`",
    ]
    for fold in range(n_folds):
        npz_path = fold_npz_path(out_dir, fold)
        meta_path = fold_meta_path(out_dir, fold)
        mark = "x" if npz_path.exists() and meta_path.exists() else " "
        suffix = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                suffix = f" - macro_f1={meta.get('macro_f1', float('nan')):.6f}"
            except Exception:
                suffix = " - meta parse failed"
        lines.append(f"- [{mark}] fold {fold} OOF saved: `{npz_path.as_posix()}`{suffix}")
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            verdict = summary.get("league", {}).get("verdict", "unknown")
            lines.append(f"- [x] league add-test evaluated: verdict={verdict}")
        except Exception:
            lines.append("- [x] league add-test evaluated: summary parse failed")
    else:
        lines.append("- [ ] league add-test evaluated")
    lines.extend(["", f"## Next resume point", "", next_step, ""])
    (context_dir / "PROGRESS-task1.md").write_text("\n".join(lines), encoding="utf-8")


def fit_fold(
    fold: int,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    ids: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    texts: list[str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = fold_npz_path(out_dir, fold)
    meta_path = fold_meta_path(out_dir, fold)
    if npz_path.exists() and meta_path.exists() and not args.force:
        print(f"[fold {fold}] exists, skipping ({npz_path})")
        return json.loads(meta_path.read_text(encoding="utf-8"))

    t0 = time.time()
    print(f"[fold {fold}] vectorize train={len(train_idx)} valid={len(valid_idx)}")
    vectorizer = build_vectorizer(args)
    train_texts = [texts[i] for i in train_idx]
    valid_texts = [texts[i] for i in valid_idx]
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(valid_texts)
    print(f"[fold {fold}] x_train={x_train.shape} nnz={x_train.nnz}")

    clf = build_classifier(args)
    clf.fit(x_train, y[train_idx])
    probs = decision_probs(clf, x_valid, ACTIONS)
    pred = labels_from_probs(probs, ACTIONS)
    macro = f1_macro(y[valid_idx], pred, ACTIONS)
    per_class = per_class_f1(y[valid_idx], pred, ACTIONS)

    elapsed = time.time() - t0
    meta = {
        "fold": fold,
        "n_train": int(len(train_idx)),
        "n_valid": int(len(valid_idx)),
        "n_train_groups": int(len(set(groups[train_idx]))),
        "n_valid_groups": int(len(set(groups[valid_idx]))),
        "n_features": int(x_train.shape[1]),
        "nnz_train": int(x_train.nnz),
        "macro_f1": macro,
        "explore4_macro_f1": f1_macro(y[valid_idx], pred, EXPLORE4),
        "per_class_f1": per_class,
        "elapsed_sec": elapsed,
        "classes_seen": [str(c) for c in clf.classes_],
        "config": json.loads(meta_json(args)),
    }
    np.savez_compressed(
        npz_path,
        ids=ids[valid_idx].astype(str),
        valid_idx=valid_idx.astype(np.int64),
        y_true=y[valid_idx].astype(str),
        groups=groups[valid_idx].astype(str),
        probs=probs.astype(np.float32),
        actions=np.array(ACTIONS, dtype=object),
        fold=np.array(fold, dtype=np.int16),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False)),
    )
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    next_fold = fold + 1
    if next_fold < args.n_folds:
        next_step = (
            f"Run fold {next_fold}: `C:\\dev\\2026-AI-DACON\\.venv\\Scripts\\python.exe "
            f"scripts/components/char_svm/train_oof.py --fold {next_fold}`"
        )
    else:
        next_step = (
            "Assemble/evaluate: `C:\\dev\\2026-AI-DACON\\.venv\\Scripts\\python.exe "
            "scripts/components/char_svm/train_oof.py --evaluate --train-full`"
        )
    update_progress(args.context_dir, args.out_dir, args.n_folds, next_step)
    print(f"[fold {fold}] macro_f1={macro:.6f} explore4={meta['explore4_macro_f1']:.6f} elapsed={elapsed:.1f}s")
    print(f"[fold {fold}] saved {npz_path}")
    return meta


def assemble_oof(
    ids: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir = args.out_dir
    probs = np.zeros((len(ids), len(ACTIONS)), dtype=np.float32)
    fold_assign = np.full(len(ids), -1, dtype=np.int16)
    fold_metrics = []
    for fold in range(args.n_folds):
        path = fold_npz_path(out_dir, fold)
        if not path.exists():
            raise FileNotFoundError(f"missing fold output: {path}")
        d = np.load(path, allow_pickle=True)
        valid_idx = d["valid_idx"].astype(np.int64)
        if np.any(fold_assign[valid_idx] != -1):
            raise RuntimeError(f"duplicate OOF rows in fold {fold}")
        probs[valid_idx] = d["probs"].astype(np.float32)
        fold_assign[valid_idx] = fold
        fold_metrics.append(json.loads(fold_meta_path(out_dir, fold).read_text(encoding="utf-8")))
    missing = np.where(fold_assign < 0)[0]
    if len(missing):
        raise RuntimeError(f"missing OOF rows: {len(missing)}, e.g. {missing[:5].tolist()}")

    pred = labels_from_probs(probs, ACTIONS)
    per_class = per_class_f1(y, pred, ACTIONS)
    summary = {
        "n_rows": int(len(ids)),
        "n_groups": int(len(set(groups))),
        "n_folds": args.n_folds,
        "oof_macro_f1": f1_macro(y, pred, ACTIONS),
        "oof_explore4_macro_f1": f1_macro(y, pred, EXPLORE4),
        "per_class_f1": per_class,
        "folds": fold_metrics,
    }
    np.savez_compressed(
        out_dir / "char_oof.npz",
        ids=ids.astype(str),
        probs=probs.astype(np.float32),
        y_true=y.astype(str),
        groups=groups.astype(str),
        fold=fold_assign,
        actions=np.array(ACTIONS, dtype=object),
        meta_json=np.array(json.dumps(summary, ensure_ascii=False)),
    )
    np.save(out_dir / "char_oof_probs.npy", probs.astype(np.float32))
    (out_dir / "char_oof_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[oof] macro_f1={summary['oof_macro_f1']:.6f} explore4={summary['oof_explore4_macro_f1']:.6f}")
    return summary


def align_columns(probs: np.ndarray, src_actions: list[str], dst_actions: list[str]) -> np.ndarray:
    missing = sorted(set(dst_actions) - set(src_actions))
    if missing:
        raise ValueError(f"missing classes: {missing}")
    return np.asarray(probs)[:, [src_actions.index(a) for a in dst_actions]]


def score_probs(probs: np.ndarray, y_true: np.ndarray, actions: list[str]) -> float:
    pred = labels_from_probs(probs, actions)
    return f1_macro(y_true, pred, actions)


def load_league_components(char_probs: np.ndarray, ids: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    enc = np.load(args.holdout_base, allow_pickle=True)
    eids = np.array([str(x) for x in enc["ids"]], dtype=object)
    eprobs = enc["probs"].astype(np.float64)
    y = np.array([str(x) for x in enc["y_true"]], dtype=object)
    acts = [str(a) for a in enc["actions"]]

    classes = json.loads((args.oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((args.oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(a) for a in acts]
    idx = {str(r): i for i, r in enumerate(row_ids)}
    missing_oof = [sample_id for sample_id in eids if sample_id not in idx]
    if missing_oof:
        raise RuntimeError(f"{len(missing_oof)} holdout ids missing from OOF row_ids")
    rows = [idx[str(sample_id)] for sample_id in eids]
    lin = np.load(args.oof_dir / "linear_probs.npy")[:, col][rows].astype(np.float64)
    stk = np.load(args.oof_dir / "stacker_probs.npy")[:, col][rows].astype(np.float64)

    char_idx = {str(sample_id): i for i, sample_id in enumerate(ids)}
    missing_char = [sample_id for sample_id in eids if sample_id not in char_idx]
    if missing_char:
        raise RuntimeError(f"{len(missing_char)} holdout ids missing from char OOF")
    char_rows = [char_idx[str(sample_id)] for sample_id in eids]
    char = align_columns(char_probs[char_rows], ACTIONS, acts).astype(np.float64)

    return {
        "ids": eids,
        "actions": acts,
        "y_true": y,
        "linear": lin,
        "stacker": stk,
        "encoder": eprobs,
        "char": char,
    }


def evaluate_league(oof_summary: dict[str, Any], char_probs: np.ndarray, ids: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    item = load_league_components(char_probs, ids, args)
    y = item["y_true"]
    acts = item["actions"]
    lin = item["linear"]
    stk = item["stacker"]
    enc = item["encoder"]
    char = item["char"]

    baseline_probs = (lin + stk + 2.0 * enc) / 4.0
    baseline = score_probs(baseline_probs, y, acts)
    if abs(baseline - BASELINE_EXPECTED) > args.baseline_tol:
        raise RuntimeError(
            f"league baseline mismatch: got {baseline:.6f}, expected {BASELINE_EXPECTED:.5f}"
        )

    rows = []
    for w4 in args.w4:
        probs = (lin + stk + 2.0 * enc + float(w4) * char) / (4.0 + float(w4))
        macro = score_probs(probs, y, acts)
        rows.append({"w4": float(w4), "macro_f1": macro, "delta": macro - baseline})

    char_pred = labels_from_probs(char, acts)
    lin_pred = labels_from_probs(lin, acts)
    diversity = {
        "holdout_char_vs_linear_disagreement": float(np.mean(char_pred != lin_pred)),
        "holdout_char_solo_macro_f1": score_probs(char, y, acts),
        "holdout_linear_solo_macro_f1": score_probs(lin, y, acts),
        "holdout_stacker_solo_macro_f1": score_probs(stk, y, acts),
        "holdout_encoder_solo_macro_f1": score_probs(enc, y, acts),
    }
    best = max(rows, key=lambda r: r["delta"])
    verdict = "PASS" if best["delta"] >= 0.002 else "FAIL"
    league = {
        "baseline_macro_f1": baseline,
        "baseline_expected": BASELINE_EXPECTED,
        "w4_grid": rows,
        "best": best,
        "verdict": verdict,
        "pass_rule": "any w4 delta >= +0.002 vs baseline",
        "diversity": diversity,
        "holdout_rows": int(len(y)),
        "holdout_actions": acts,
    }
    summary = {"oof": oof_summary, "league": league}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[league] baseline={baseline:.6f} expected={BASELINE_EXPECTED:.5f}")
    for row in rows:
        print(f"[league] w4={row['w4']:g} macro_f1={row['macro_f1']:.6f} delta={row['delta']:+.6f}")
    print(f"[league] verdict={verdict} best_w4={best['w4']:g} best_delta={best['delta']:+.6f}")
    return summary


def full_train(ids: np.ndarray, y: np.ndarray, texts: list[str], args: argparse.Namespace) -> dict[str, Any]:
    out_path = args.out_dir / "char_svm_full.pkl"
    meta_path = args.out_dir / "char_svm_full_meta.json"
    if out_path.exists() and meta_path.exists() and not args.force:
        print(f"[full] exists, skipping ({out_path})")
        return json.loads(meta_path.read_text(encoding="utf-8"))
    t0 = time.time()
    print(f"[full] vectorize rows={len(ids)}")
    vectorizer = build_vectorizer(args)
    x_all = vectorizer.fit_transform(texts)
    print(f"[full] x_all={x_all.shape} nnz={x_all.nnz}")
    clf = build_classifier(args)
    clf.fit(x_all, y)
    artifact = {
        "vectorizer": vectorizer,
        "clf": clf,
        "actions": ACTIONS,
        "serialize": "scripts/components/char_svm/train_oof.py::serialize_char_view",
        "config": json.loads(meta_json(args, {"fit_rows": int(len(ids)), "n_features": int(x_all.shape[1])})),
    }
    joblib.dump(artifact, out_path, compress=3)
    meta = {
        "path": str(out_path),
        "n_rows": int(len(ids)),
        "n_features": int(x_all.shape[1]),
        "nnz": int(x_all.nnz),
        "classes_seen": [str(c) for c in clf.classes_],
        "elapsed_sec": time.time() - t0,
        "config": artifact["config"],
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[full] saved {out_path} elapsed={meta['elapsed_sec']:.1f}s")
    return meta


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--holdout-base", type=Path, default=DEFAULT_HOLDOUT_BASE)
    parser.add_argument("--oof-dir", type=Path, default=DEFAULT_OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--context-dir", type=Path, default=DEFAULT_CONTEXT_DIR)
    parser.add_argument("--fold", type=int, default=None, help="run one fold; omit with --all-folds")
    parser.add_argument("--all-folds", action="store_true")
    parser.add_argument("--evaluate", action="store_true", help="assemble OOF and run league add-test")
    parser.add_argument("--train-full", action="store_true", help="fit full 70k artifact")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ngram-min", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=300_000)
    parser.add_argument("--c", type=float, default=0.1)
    parser.add_argument("--class-weight", choices=["balanced", "none"], default="balanced")
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--baseline-tol", type=float, default=2e-4)
    parser.add_argument("--w4", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    args = parser.parse_args()
    if args.fold is None and not args.all_folds and not args.evaluate and not args.train_full:
        parser.error("choose --fold K, --all-folds, --evaluate, or --train-full")
    if args.fold is not None and not (0 <= args.fold < args.n_folds):
        parser.error(f"--fold must be in [0,{args.n_folds - 1}]")
    return args


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[load] data_dir={args.data_dir}")
    _, ids, y, groups, texts = load_training(args.data_dir)
    print(f"[load] rows={len(ids)} groups={len(set(groups))} classes={len(set(y))}")
    splits = make_splits(y, groups, args.n_folds, args.seed)
    update_progress(args.context_dir, args.out_dir, args.n_folds, "Loaded data; next fold/eval command is in the task prompt.")

    folds_to_run: list[int] = []
    if args.fold is not None:
        folds_to_run = [args.fold]
    elif args.all_folds:
        folds_to_run = list(range(args.n_folds))
    for fold in folds_to_run:
        train_idx, valid_idx = splits[fold]
        fit_fold(fold, train_idx, valid_idx, ids, y, groups, texts, args)

    oof_summary: dict[str, Any] | None = None
    if args.evaluate:
        oof_summary = assemble_oof(ids, y, groups, args)
        char_probs = np.load(args.out_dir / "char_oof.npz", allow_pickle=True)["probs"].astype(np.float32)
        summary = evaluate_league(oof_summary, char_probs, ids, args)
        next_step = "Write `context/night/2026-07-06/task1_report.md` and `task1.DONE`; commit final state."
        if summary["league"]["verdict"] == "PASS" and not (args.out_dir / "char_svm_full.pkl").exists():
            next_step = (
                "PASS: run full train if not done: `C:\\dev\\2026-AI-DACON\\.venv\\Scripts\\python.exe "
                "scripts/components/char_svm/train_oof.py --train-full`"
            )
        update_progress(args.context_dir, args.out_dir, args.n_folds, next_step)

    if args.train_full:
        full_train(ids, y, texts, args)
        if oof_summary is None and (args.out_dir / "char_oof_summary.json").exists():
            oof_summary = json.loads((args.out_dir / "char_oof_summary.json").read_text(encoding="utf-8"))
        update_progress(
            args.context_dir,
            args.out_dir,
            args.n_folds,
            "Full artifact saved; write report/DONE and commit final state.",
        )


if __name__ == "__main__":
    main()
