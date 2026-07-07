# -*- coding: utf-8 -*-
"""Shared helpers for task2 teammate-configuration evaluation."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

sys.dont_write_bytecode = True

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
sys.path.insert(0, str(SUBMIT_DIR))
import au_route  # noqa: E402


DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
HOLDOUT_BASE = ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"
MBERT_HOLDOUT = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz")
OUT_DIR = ROOT / "context" / "night" / "2026-07-07"
AU_CACHE = OUT_DIR / "mate_au_char_c1_holdout.npz"

EXPECTED_3WAY = 0.71726
EXPECTED_4WAY = 0.72255
SANITY_TOL = 5e-4
SOFT_AU_ALPHA = 0.9
EPS = 1e-12

ACTION_CLASSES = [
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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--holdout-base", type=Path, default=HOLDOUT_BASE)
    parser.add_argument("--mbert-holdout", type=Path, default=MBERT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--au-cache", type=Path, default=AU_CACHE)
    parser.add_argument("--alpha", type=float, default=SOFT_AU_ALPHA)
    parser.add_argument("--refresh-au", action="store_true")


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: list[str] = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {path}")


def predict_labels(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray([str(a) for a in actions], dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1_labels(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    values = []
    for label in labels:
        truth = y_true_arr == str(label)
        pred = y_pred_arr == str(label)
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        denom = 2 * tp + fp + fn
        values.append(0.0 if denom == 0 else (2.0 * tp) / denom)
    return float(np.mean(values))


def macro_f1_probs(probs: np.ndarray, y_true: Sequence[str], actions: Sequence[str]) -> float:
    return macro_f1_labels(y_true, predict_labels(probs, actions), actions)


def per_class_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> list[dict[str, Any]]:
    y_true_arr = np.asarray(y_true, dtype=object)
    y_pred_arr = np.asarray(y_pred, dtype=object)
    rows: list[dict[str, Any]] = []
    for label in labels:
        label = str(label)
        truth = y_true_arr == label
        pred = y_pred_arr == label
        tp = int(np.sum(truth & pred))
        fp = int(np.sum(~truth & pred))
        fn = int(np.sum(truth & ~pred))
        denom = 2 * tp + fp + fn
        rows.append(
            {
                "class": label,
                "support": int(np.sum(truth)),
                "pred_count": int(np.sum(pred)),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": 0.0 if denom == 0 else (2.0 * tp) / denom,
            }
        )
    return rows


def align_probs(probs: np.ndarray, src_actions: Sequence[str], dst_actions: Sequence[str]) -> np.ndarray:
    src = [str(x) for x in src_actions]
    missing = [str(x) for x in dst_actions if str(x) not in src]
    if missing:
        raise ValueError(f"missing actions in source probabilities: {missing}")
    idx = [src.index(str(x)) for x in dst_actions]
    return np.asarray(probs, dtype=np.float64)[:, idx]


def reorder_npz_to_reference(path: Path, ref_ids: np.ndarray, ref_y: np.ndarray, dst_actions: Sequence[str]) -> np.ndarray:
    item = np.load(path, allow_pickle=True)
    ids = np.asarray([str(x) for x in item["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in item["y_true"]], dtype=object)
    actions = [str(x) for x in item["actions"]]
    pos = {str(sample_id): i for i, sample_id in enumerate(ids)}
    missing = [str(sample_id) for sample_id in ref_ids if str(sample_id) not in pos]
    extra = [str(sample_id) for sample_id in ids if str(sample_id) not in set(ref_ids)]
    if missing or extra:
        raise ValueError(f"{path} id mismatch: missing={len(missing)} extra={len(extra)}")
    order = np.asarray([pos[str(sample_id)] for sample_id in ref_ids], dtype=np.int64)
    if not np.array_equal(y_true[order], ref_y):
        raise ValueError(f"{path} y_true mismatch after id alignment")
    return align_probs(np.asarray(item["probs"], dtype=np.float64)[order], actions, dst_actions)


def load_components(
    *,
    holdout_base: Path = HOLDOUT_BASE,
    oof_dir: Path = OOF_DIR,
    mbert_holdout: Path = MBERT_HOLDOUT,
) -> dict[str, Any]:
    enc = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in enc["y_true"]], dtype=object)
    actions = [str(x) for x in enc["actions"]]
    e5 = np.asarray(enc["probs"], dtype=np.float64)

    classes = [str(x) for x in read_json(oof_dir / "classes.json")]
    row_ids = [str(x) for x in read_json(oof_dir / "row_ids.json")]
    col = [classes.index(action) for action in actions]
    row_index = {sample_id: i for i, sample_id in enumerate(row_ids)}
    missing = [sample_id for sample_id in ids if sample_id not in row_index]
    if missing:
        raise ValueError(f"OOF row_ids missing holdout rows: {len(missing)}")
    rows = np.asarray([row_index[str(sample_id)] for sample_id in ids], dtype=np.int64)
    linear = np.load(oof_dir / "linear_probs.npy")[:, col][rows].astype(np.float64)
    stacker = np.load(oof_dir / "stacker_probs.npy")[:, col][rows].astype(np.float64)
    mbert = reorder_npz_to_reference(mbert_holdout, ids, y_true, actions)

    blend3 = (linear + stacker + 2.0 * e5) / 4.0
    blend4 = (linear + stacker + 1.2 * e5 + 0.8 * mbert) / 4.0
    score3 = macro_f1_probs(blend3, y_true, actions)
    score4 = macro_f1_probs(blend4, y_true, actions)
    if abs(score3 - EXPECTED_3WAY) > SANITY_TOL:
        raise AssertionError(f"3-way sanity failed: got {score3:.8f}, expected {EXPECTED_3WAY:.5f}")
    if abs(score4 - EXPECTED_4WAY) > SANITY_TOL:
        raise AssertionError(f"4-way sanity failed: got {score4:.8f}, expected {EXPECTED_4WAY:.5f}")

    return {
        "ids": ids,
        "y_true": y_true,
        "actions": actions,
        "au_mask": np.asarray([au_route.is_au(str(sample_id)) for sample_id in ids], dtype=bool),
        "components": {
            "linear": linear,
            "stacker": stacker,
            "e5": e5,
            "mbert": mbert,
            "blend3": blend3,
            "blend4": blend4,
        },
        "sanity": {
            "blend3_macro_f1": score3,
            "blend4_macro_f1": score4,
            "expected_blend3": EXPECTED_3WAY,
            "expected_blend4": EXPECTED_4WAY,
            "tolerance": SANITY_TOL,
        },
        "paths": {
            "holdout_base": str(holdout_base),
            "mbert_holdout": str(mbert_holdout),
            "oof_dir": str(oof_dir),
        },
    }


def load_train(data_dir: Path) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    labels: dict[str, str] = {}
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            labels[str(row["id"])] = str(row["action"])
    ids = np.asarray([str(sample["id"]) for sample in samples], dtype=object)
    y = np.asarray([labels[str(sample["id"])] for sample in samples], dtype=object)
    return samples, ids, y


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit_au_char_c1(
    *,
    data_dir: Path,
    holdout: dict[str, Any],
    cache_path: Path,
    refresh: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    actions = [str(x) for x in holdout["actions"]]
    au_ids = np.asarray(holdout["ids"][holdout["au_mask"]], dtype=object)
    if cache_path.exists() and not refresh:
        cached = np.load(cache_path, allow_pickle=True)
        cached_ids = np.asarray([str(x) for x in cached["ids"]], dtype=object)
        cached_actions = [str(x) for x in cached["actions"]]
        if np.array_equal(cached_ids, au_ids) and cached_actions == actions:
            return {
                "ids": cached_ids,
                "probs": np.asarray(cached["probs"], dtype=np.float64),
                "actions": cached_actions,
                "meta": json.loads(str(np.asarray(cached["meta_json"]).item())),
                "from_cache": True,
            }
        print(f"[cache] ignoring stale AU cache: {cache_path}")

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    started = time.time()
    samples, ids, y = load_train(data_dir)
    holdout_id_set = set(str(x) for x in holdout["ids"])
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    train_idx = np.asarray(
        [
            i
            for i, sample_id in enumerate(ids)
            if str(sample_id) not in holdout_id_set and au_route.is_au(str(sample_id))
        ],
        dtype=np.int64,
    )
    if any(str(sample_id) in holdout_id_set for sample_id in ids[train_idx]):
        raise AssertionError("holdout id leaked into AU train")
    eval_samples = [sample_by_id[str(sample_id)] for sample_id in au_ids]
    train_samples = [samples[int(i)] for i in train_idx]
    train_texts = [au_route.serialize(sample) for sample in train_samples]
    eval_texts = [au_route.serialize(sample) for sample in eval_samples]

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vectorizer.fit_transform(train_texts)
    x_eval = vectorizer.transform(eval_texts)
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed)
    clf.fit(x_train, y[train_idx])
    probs = softmax(clf.decision_function(x_eval))
    probs = align_probs(probs, [str(c) for c in clf.classes_], actions)
    meta = {
        "model": "char_wb_3_5_linear_svc_c1_balanced",
        "train_rows": int(len(train_idx)),
        "train_sessions": int(len({session_id(str(x)) for x in ids[train_idx]})),
        "holdout_au_rows": int(len(au_ids)),
        "holdout_au_sessions": int(len({session_id(str(x)) for x in au_ids})),
        "n_features": int(x_train.shape[1]),
        "classes": [str(c) for c in clf.classes_],
        "elapsed_sec": round(time.time() - started, 3),
        "seed": seed,
        "holdout_excluded": True,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        ids=au_ids,
        probs=probs,
        actions=np.asarray(actions, dtype=object),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False), dtype=object),
    )
    print(f"[save] {cache_path}")
    return {"ids": au_ids, "probs": probs, "actions": actions, "meta": meta, "from_cache": False}


def apply_soft_au(
    base_probs: np.ndarray,
    *,
    holdout: dict[str, Any],
    au_probs: np.ndarray,
    alpha: float,
) -> np.ndarray:
    out = np.asarray(base_probs, dtype=np.float64).copy()
    mask = np.asarray(holdout["au_mask"], dtype=bool)
    out[mask] = float(alpha) * np.asarray(au_probs, dtype=np.float64) + (1.0 - float(alpha)) * out[mask]
    return out


def score_variant(
    name: str,
    probs: np.ndarray,
    *,
    holdout: dict[str, Any],
    reference_score: float | None = None,
) -> dict[str, Any]:
    y_true = holdout["y_true"]
    actions = holdout["actions"]
    pred = predict_labels(probs, actions)
    mask = np.asarray(holdout["au_mask"], dtype=bool)
    score = macro_f1_labels(y_true, pred, actions)
    row: dict[str, Any] = {
        "variant": name,
        "macro_f1": score,
        "au_macro_f1": macro_f1_labels(y_true[mask], pred[mask], actions),
        "sim_macro_f1": macro_f1_labels(y_true[~mask], pred[~mask], actions),
        "changed_vs_blend4": int(np.sum(pred != predict_labels(holdout["components"]["blend4"], actions))),
    }
    if reference_score is not None:
        row["delta_vs_reference"] = score - float(reference_score)
    return row


def per_class_variant_rows(
    name: str,
    probs: np.ndarray,
    *,
    holdout: dict[str, Any],
    baseline_pred: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    actions = holdout["actions"]
    pred = predict_labels(probs, actions)
    rows = []
    base_lookup = None
    if baseline_pred is not None:
        base_lookup = {row["class"]: row for row in per_class_f1(holdout["y_true"], baseline_pred, actions)}
    for row in per_class_f1(holdout["y_true"], pred, actions):
        out = {"variant": name, **row}
        if base_lookup is not None:
            base = base_lookup[row["class"]]
            out["baseline_f1"] = base["f1"]
            out["delta_vs_baseline_f1"] = row["f1"] - base["f1"]
        rows.append(out)
    return rows


def apply_log_bias(probs: np.ndarray, actions: Sequence[str], bias: dict[str, float]) -> np.ndarray:
    offsets = np.zeros(len(actions), dtype=np.float64)
    for cls, value in bias.items():
        if cls not in actions:
            raise ValueError(f"unknown bias class: {cls}")
        offsets[list(actions).index(cls)] = float(value)
    return np.log(np.clip(probs, EPS, None)) + offsets.reshape(1, -1)


def labels_from_log_bias(probs: np.ndarray, actions: Sequence[str], bias: dict[str, float]) -> np.ndarray:
    labels = np.asarray([str(a) for a in actions], dtype=object)
    return labels[apply_log_bias(probs, actions, bias).argmax(axis=1)]


def scaled_bias(scale: float) -> dict[str, float]:
    return {
        "read_file": 0.1 * float(scale),
        "grep_search": -0.1 * float(scale),
        "list_directory": -0.18 * float(scale),
    }


def format_float(value: float) -> str:
    return f"{float(value):.9f}"


def print_score_table(rows: Iterable[dict[str, Any]]) -> None:
    print("variant,macro_f1,delta_vs_reference,au_macro_f1,sim_macro_f1,changed_vs_blend4")
    for row in rows:
        delta = row.get("delta_vs_reference")
        delta_text = "" if delta is None else f"{float(delta):+.9f}"
        print(
            "{variant},{macro},{delta},{au},{sim},{changed}".format(
                variant=row["variant"],
                macro=format_float(row["macro_f1"]),
                delta=delta_text,
                au=format_float(row["au_macro_f1"]),
                sim=format_float(row["sim_macro_f1"]),
                changed=row["changed_vs_blend4"],
            )
        )
