# -*- coding: utf-8 -*-
"""Probe whether a sess_au-only linear specialist can beat the fixed 3-way blend."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
HOLDOUT_BASE = Path("context/night/2026-07-05/holdout_base.npz")
OUT_JSON = Path("context/night/2026-07-06/task3_probe_au_linear.json")
EXPECTED_3WAY = 0.71726
DEFAULT_ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def is_au(sample_id: str) -> bool:
    return str(sample_id).startswith("sess_au")


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def load_train(data_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with (data_dir / "train_labels.csv").open(encoding="utf-8") as f:
        label_map = {row["id"]: row["action"] for row in csv.DictReader(f)}
    ids = np.array([str(sample["id"]) for sample in samples], dtype=object)
    y = np.array([label_map[str(sample["id"])] for sample in samples], dtype=object)
    groups = np.array([session_id(str(sample["id"])) for sample in samples], dtype=object)
    return samples, ids, y, groups


def compact_json(value: Any, limit: int = 240) -> str:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def serialize(sample: dict[str, Any]) -> str:
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    parts = [
        "id_prefix=sess_au" if is_au(str(sample.get("id", ""))) else "id_prefix=sess_sim",
        f"user_tier={meta.get('user_tier', '')}",
        f"language_pref={meta.get('language_pref', '')}",
        f"turn_index={meta.get('turn_index', '')}",
        f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}",
        f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}",
        f"git_dirty={workspace.get('git_dirty', '')}",
        f"last_ci_status={workspace.get('last_ci_status', '')}",
        f"loc={workspace.get('loc', '')}",
        "language_mix=" + compact_json(workspace.get("language_mix") or {}),
        "open_files=" + " ".join(str(x) for x in (workspace.get("open_files") or [])),
        "current_prompt=" + str(sample.get("current_prompt") or ""),
    ]
    for item in sample.get("history") or []:
        role = str(item.get("role", ""))
        if role == "assistant_action":
            parts.append(
                "history_action="
                + str(item.get("name", ""))
                + " args="
                + compact_json(item.get("args") or {}, 180)
                + " result="
                + str(item.get("result_summary", ""))
            )
        else:
            parts.append(f"history_{role}=" + str(item.get("content", "")))
    return "\n".join(parts)


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_probs(
    probs: np.ndarray,
    src_classes: Sequence[str],
    dst_classes: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    out = np.full((probs.shape[0], len(dst_classes)), fill_value, dtype=np.float64)
    for dst_i, cls in enumerate(dst_classes):
        if cls in src:
            out[:, dst_i] = probs[:, src.index(cls)]
    row_sum = out.sum(axis=1, keepdims=True)
    missing_rows = row_sum.ravel() <= 0
    if missing_rows.any():
        out[missing_rows, :] = 1.0 / len(dst_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def build_features(train_texts: list[str], valid_texts: list[str]):
    union = FeatureUnion([
        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                min_df=1,
                max_features=80_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                min_df=1,
                max_features=120_000,
                sublinear_tf=True,
                strip_accents="unicode",
            ),
        ),
    ])
    x_train = union.fit_transform(train_texts)
    x_valid = union.transform(valid_texts)
    if not sparse.issparse(x_train) or not sparse.issparse(x_valid):
        raise TypeError("TF-IDF feature union should return sparse matrices")
    return x_train, x_valid


def preds_from_probs(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    return np.array(list(actions), dtype=object)[np.asarray(probs).argmax(axis=1)]


def macro_f1_from_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> float:
    return float(f1_score(y_true, preds_from_probs(probs, actions), average="macro", zero_division=0))


def per_class_f1(y_true: np.ndarray, pred: np.ndarray, actions: Sequence[str]) -> list[dict[str, Any]]:
    per = f1_score(y_true, pred, labels=list(actions), average=None, zero_division=0)
    return [
        {
            "class": str(cls),
            "f1": float(score),
            "support": int((y_true == cls).sum()),
            "pred_count": int((pred == cls).sum()),
        }
        for cls, score in zip(actions, per)
    ]


def load_holdout_join(holdout_base: Path, oof_dir: Path) -> dict[str, Any]:
    enc = np.load(holdout_base, allow_pickle=True)
    ids = np.array([str(x) for x in enc["ids"]], dtype=object)
    enc_probs = np.asarray(enc["probs"], dtype=np.float64)
    y_true = np.array([str(x) for x in enc["y_true"]], dtype=object)
    actions = [str(a) for a in enc["actions"]]

    classes = json.loads((oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(a) for a in actions]
    idx = {str(row_id): i for i, row_id in enumerate(row_ids)}
    rows = np.array([idx[str(sample_id)] for sample_id in ids], dtype=np.int64)
    lin = np.load(oof_dir / "linear_probs.npy")[:, col][rows]
    stk = np.load(oof_dir / "stacker_probs.npy")[:, col][rows]
    blend = (lin + stk + 2.0 * enc_probs) / 4.0
    score = macro_f1_from_probs(blend, y_true, actions)
    if abs(score - EXPECTED_3WAY) > 5e-5:
        raise AssertionError(f"3-way join mismatch: got {score:.8f}, expected {EXPECTED_3WAY}")
    return {"ids": ids, "y_true": y_true, "actions": actions, "blend": blend, "score": score}


def run_au_oof(
    samples: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int,
    seed: int,
    c_value: float,
) -> dict[str, Any]:
    au_idx = np.array([i for i, sample_id in enumerate(ids) if is_au(str(sample_id))], dtype=np.int64)
    au_ids = ids[au_idx]
    au_y = y[au_idx]
    au_groups = groups[au_idx]
    texts = [serialize(samples[int(i)]) for i in au_idx]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(au_idx), len(DEFAULT_ACTIONS)), dtype=np.float64)
    fold_rows = []

    for fold, (train_pos, valid_pos) in enumerate(splitter.split(np.zeros(len(au_y)), au_y, groups=au_groups), 1):
        overlap = set(au_groups[train_pos]) & set(au_groups[valid_pos])
        if overlap:
            raise AssertionError(f"fold {fold} group leakage: {len(overlap)} groups")
        train_texts = [texts[int(i)] for i in train_pos]
        valid_texts = [texts[int(i)] for i in valid_pos]
        x_train, x_valid = build_features(train_texts, valid_texts)
        clf = LinearSVC(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed)
        clf.fit(x_train, au_y[train_pos])
        probs = softmax(clf.decision_function(x_valid))
        aligned = align_probs(probs, [str(c) for c in clf.classes_], DEFAULT_ACTIONS)
        oof[valid_pos] = aligned
        pred = preds_from_probs(aligned, DEFAULT_ACTIONS)
        fold_rows.append({
            "fold": fold,
            "rows": int(len(valid_pos)),
            "groups": int(len(set(au_groups[valid_pos]))),
            "macro_f1": float(f1_score(au_y[valid_pos], pred, average="macro", zero_division=0)),
            "classes_in_train": [str(c) for c in clf.classes_],
        })
        print(
            f"[fold {fold}] rows={len(valid_pos)} groups={len(set(au_groups[valid_pos]))} "
            f"macro_f1={fold_rows[-1]['macro_f1']:.6f}"
        )

    oof_pred = preds_from_probs(oof, DEFAULT_ACTIONS)
    return {
        "au_ids": au_ids,
        "au_y": au_y,
        "au_groups": au_groups,
        "oof_probs": oof,
        "folds": fold_rows,
        "oof_macro_f1": float(f1_score(au_y, oof_pred, average="macro", zero_division=0)),
        "oof_per_class": per_class_f1(au_y, oof_pred, DEFAULT_ACTIONS),
    }


def evaluate_routing(probe: dict[str, Any], holdout: dict[str, Any]) -> dict[str, Any]:
    au_prob_by_id = {str(sample_id): probe["oof_probs"][i] for i, sample_id in enumerate(probe["au_ids"])}
    ids = holdout["ids"]
    y_true = holdout["y_true"]
    actions = holdout["actions"]
    blend = holdout["blend"]
    au_mask = np.array([is_au(str(sample_id)) for sample_id in ids], dtype=bool)
    missing = [str(sample_id) for sample_id in ids[au_mask] if str(sample_id) not in au_prob_by_id]
    if missing:
        raise ValueError(f"holdout AU ids missing from probe OOF: {missing[:5]}")

    probe_probs = np.vstack([au_prob_by_id[str(sample_id)] for sample_id in ids[au_mask]])
    probe_probs = align_probs(probe_probs, DEFAULT_ACTIONS, actions)
    blend_pred = preds_from_probs(blend, actions)
    probe_pred_au = preds_from_probs(probe_probs, actions)

    hybrid_pred = blend_pred.copy()
    hybrid_pred[au_mask] = probe_pred_au

    au_y = y_true[au_mask]
    blend_pred_au = blend_pred[au_mask]
    all_blend = float(f1_score(y_true, blend_pred, average="macro", zero_division=0))
    all_hybrid = float(f1_score(y_true, hybrid_pred, average="macro", zero_division=0))
    au_blend = float(f1_score(au_y, blend_pred_au, average="macro", zero_division=0))
    au_probe = float(f1_score(au_y, probe_pred_au, average="macro", zero_division=0))

    return {
        "holdout_rows": int(len(ids)),
        "holdout_au_rows": int(au_mask.sum()),
        "holdout_au_share": float(au_mask.mean()),
        "blend_all_macro_f1": all_blend,
        "hybrid_all_macro_f1": all_hybrid,
        "hybrid_delta": float(all_hybrid - all_blend),
        "blend_au_macro_f1": au_blend,
        "probe_au_macro_f1_on_holdout_au_oof": au_probe,
        "probe_minus_blend_au": float(au_probe - au_blend),
        "blend_au_per_class": per_class_f1(au_y, blend_pred_au, actions),
        "probe_au_per_class": per_class_f1(au_y, probe_pred_au, actions),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--holdout-base", type=Path, default=HOLDOUT_BASE)
    parser.add_argument("--out-json", type=Path, default=OUT_JSON)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples, ids, y, groups = load_train(args.data_dir)
    probe = run_au_oof(samples, ids, y, groups, args.n_splits, args.seed, args.c)
    holdout = load_holdout_join(args.holdout_base, args.oof_dir)
    routing = evaluate_routing(probe, holdout)
    result = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "oof_dir": str(args.oof_dir),
            "holdout_base": str(args.holdout_base),
            "n_splits": args.n_splits,
            "seed": args.seed,
            "c": args.c,
            "model": "FeatureUnion(word 1-2 + char_wb 3-5) + LinearSVC(class_weight=balanced)",
        },
        "au_oof": {
            "rows": int(len(probe["au_ids"])),
            "sessions": int(len(set(probe["au_groups"]))),
            "folds": probe["folds"],
            "macro_f1": probe["oof_macro_f1"],
            "per_class": probe["oof_per_class"],
        },
        "routing_eval": routing,
        "decision_rule": {
            "pass_threshold_hybrid_delta": 0.002,
            "passes": bool(routing["hybrid_delta"] >= 0.002),
        },
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[au-oof] macro_f1={probe['oof_macro_f1']:.6f} rows={len(probe['au_ids'])}")
    print(
        f"[routing] blend_au={routing['blend_au_macro_f1']:.6f} "
        f"probe_au={routing['probe_au_macro_f1_on_holdout_au_oof']:.6f}"
    )
    print(
        f"[routing] blend_all={routing['blend_all_macro_f1']:.6f} "
        f"hybrid_all={routing['hybrid_all_macro_f1']:.6f} "
        f"delta={routing['hybrid_delta']:+.6f}"
    )
    print(f"[save] {args.out_json}")


if __name__ == "__main__":
    main()
