# -*- coding: utf-8 -*-
"""Shared utilities for the task1 4-way local league.

The joins in this file intentionally mirror context/night/2026-07-07/task1.md.
All mutable outputs stay under night_out/league4 inside this worktree.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
sys.path.insert(0, str(SUBMIT_DIR))
import au_route  # noqa: E402


DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
HOLDOUT_BASE = ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"
MBERT_HOLDOUT = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz")
OUT_DIR = ROOT / "night_out" / "league4"

EXPECTED_3WAY = 0.71726
EXPECTED_4WAY = 0.72255
DEFAULT_ALPHA = 0.9
BASE_E5_WEIGHT = 1.2
BASE_MBERT_WEIGHT = 0.8


@dataclass(frozen=True)
class LeagueData:
    ids: np.ndarray
    y_true: np.ndarray
    actions: list[str]
    lin: np.ndarray
    stk: np.ndarray
    e5: np.ndarray
    mbert: np.ndarray
    samples_by_id: dict[str, dict[str, Any]]
    train_samples: list[dict[str, Any]]
    train_ids: np.ndarray
    train_y: np.ndarray
    train_groups: np.ndarray
    au_mask: np.ndarray
    non_au_mask: np.ndarray


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_train(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as f:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    ids = np.asarray([str(s["id"]) for s in samples], dtype=object)
    y = np.asarray([labels[str(s["id"])] for s in samples], dtype=object)
    groups = np.asarray([session_id(str(s["id"])) for s in samples], dtype=object)
    return samples, ids, y, groups


def align_npz_probs(path: Path, ids: Sequence[str], y_true: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    z = np.load(path, allow_pickle=True)
    src_ids = [str(x) for x in z["ids"]]
    src_actions = [str(x) for x in z["actions"]]
    src_y = np.asarray([str(x) for x in z["y_true"]], dtype=object)
    row_index = {sample_id: i for i, sample_id in enumerate(src_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in ids], dtype=np.int64)
    if not np.array_equal(src_y[rows], y_true):
        raise AssertionError(f"{path} y_true does not match holdout_base rows")
    col = [src_actions.index(str(action)) for action in actions]
    return np.asarray(z["probs"], dtype=np.float64)[rows][:, col]


def load_oof_probs(
    oof_dir: Path,
    ids: Sequence[str],
    actions: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    classes = [str(x) for x in load_json(oof_dir / "classes.json")]
    row_ids = [str(x) for x in load_json(oof_dir / "row_ids.json")]
    col = [classes.index(str(action)) for action in actions]
    row_index = {sample_id: i for i, sample_id in enumerate(row_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in ids], dtype=np.int64)
    lin = np.load(oof_dir / "linear_probs.npy")[:, col][rows].astype(np.float64)
    stk = np.load(oof_dir / "stacker_probs.npy")[:, col][rows].astype(np.float64)
    return lin, stk


def predict_from_probs(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray([str(action) for action in actions], dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1_labels(y_true: np.ndarray, pred: np.ndarray, actions: Sequence[str]) -> float:
    return float(f1_score(y_true, pred, labels=[str(a) for a in actions], average="macro", zero_division=0))


def macro_f1_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> float:
    return macro_f1_labels(y_true, predict_from_probs(probs, actions), actions)


def assert_row_probs(name: str, probs: np.ndarray, n_rows: int, n_classes: int) -> None:
    if probs.shape != (n_rows, n_classes):
        raise AssertionError(f"{name} shape {probs.shape} != {(n_rows, n_classes)}")
    row_sum = probs.sum(axis=1)
    if not np.all(np.isfinite(probs)):
        raise AssertionError(f"{name} contains non-finite values")
    if not np.allclose(row_sum, 1.0, atol=1e-5):
        raise AssertionError(f"{name} row sums drift: min={row_sum.min()} max={row_sum.max()}")


def four_way_blend(data: LeagueData, e5_weight: float = BASE_E5_WEIGHT, mbert_weight: float = BASE_MBERT_WEIGHT) -> np.ndarray:
    return (data.lin + data.stk + e5_weight * data.e5 + mbert_weight * data.mbert) / (
        2.0 + e5_weight + mbert_weight
    )


def three_way_blend(data: LeagueData) -> np.ndarray:
    return (data.lin + data.stk + 2.0 * data.e5) / 4.0


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def align_probs(
    probs: np.ndarray,
    src_classes: Sequence[str],
    dst_classes: Sequence[str],
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    out = np.full((probs.shape[0], len(dst_classes)), fill_value, dtype=np.float64)
    for dst_i, cls in enumerate(dst_classes):
        if str(cls) in src:
            out[:, dst_i] = probs[:, src.index(str(cls))]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.ravel() <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(dst_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def load_league_data(
    holdout_base: Path = HOLDOUT_BASE,
    mbert_holdout: Path = MBERT_HOLDOUT,
    oof_dir: Path = OOF_DIR,
    data_dir: Path = DATA_DIR,
) -> LeagueData:
    enc = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in enc["y_true"]], dtype=object)
    actions = [str(x) for x in enc["actions"]]
    e5 = np.asarray(enc["probs"], dtype=np.float64)
    lin, stk = load_oof_probs(oof_dir, ids, actions)
    mbert = align_npz_probs(mbert_holdout, ids, y_true, actions)
    for name, probs in (("linear", lin), ("stacker", stk), ("e5", e5), ("mbert", mbert)):
        assert_row_probs(name, probs, len(ids), len(actions))

    train_samples, train_ids, train_y, train_groups = load_train(data_dir)
    samples_by_id = {str(s["id"]): s for s in train_samples}
    missing = [str(sample_id) for sample_id in ids if str(sample_id) not in samples_by_id]
    if missing:
        raise AssertionError(f"holdout ids missing from train.jsonl: {missing[:3]} ({len(missing)} total)")

    au_mask = np.asarray([au_route.is_au(str(sample_id)) for sample_id in ids], dtype=bool)
    data = LeagueData(
        ids=ids,
        y_true=y_true,
        actions=actions,
        lin=lin,
        stk=stk,
        e5=e5,
        mbert=mbert,
        samples_by_id=samples_by_id,
        train_samples=train_samples,
        train_ids=train_ids,
        train_y=train_y,
        train_groups=train_groups,
        au_mask=au_mask,
        non_au_mask=~au_mask,
    )
    assert_sanity_scores(data)
    return data


def assert_sanity_scores(data: LeagueData) -> None:
    s3 = macro_f1_probs(three_way_blend(data), data.y_true, data.actions)
    s4 = macro_f1_probs(four_way_blend(data), data.y_true, data.actions)
    if abs(s3 - EXPECTED_3WAY) > 5e-4:
        raise AssertionError(f"3-way sanity mismatch: {s3:.10f} != {EXPECTED_3WAY:.10f}")
    if abs(s4 - EXPECTED_4WAY) > 5e-4:
        raise AssertionError(f"4-way sanity mismatch: {s4:.10f} != {EXPECTED_4WAY:.10f}")


def train_or_load_au_probs(data: LeagueData, out_dir: Path = OUT_DIR, force: bool = False) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "au_charwb_C1_holdout_probs.npz"
    meta_path = out_dir / "au_charwb_C1_holdout_meta.json"
    holdout_au_ids = data.ids[data.au_mask]
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True)
        cached_ids = np.asarray([str(x) for x in z["ids"]], dtype=object)
        cached_actions = [str(x) for x in z["actions"]]
        if np.array_equal(cached_ids, holdout_au_ids) and cached_actions == data.actions:
            meta = load_json(meta_path) if meta_path.exists() else {}
            return {
                "probs": np.asarray(z["probs"], dtype=np.float64),
                "ids": cached_ids,
                "actions": cached_actions,
                "meta": meta,
                "cache_hit": True,
            }

    started = time.time()
    holdout_id_set = set(str(x) for x in data.ids)
    train_idx = np.asarray(
        [
            i
            for i, sample_id in enumerate(data.train_ids)
            if str(sample_id) not in holdout_id_set and au_route.is_au(str(sample_id))
        ],
        dtype=np.int64,
    )
    if any(str(sample_id) in holdout_id_set for sample_id in data.train_ids[train_idx]):
        raise AssertionError("holdout id leaked into AU train")
    train_samples = [data.train_samples[int(i)] for i in train_idx]
    eval_samples = [data.samples_by_id[str(sample_id)] for sample_id in holdout_au_ids]
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    x_train = vec.fit_transform([au_route.serialize(s) for s in train_samples])
    x_eval = vec.transform([au_route.serialize(s) for s in eval_samples])
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, random_state=42)
    clf.fit(x_train, data.train_y[train_idx])
    probs = align_probs(softmax(clf.decision_function(x_eval)), [str(c) for c in clf.classes_], data.actions)
    np.savez_compressed(
        cache_path,
        ids=holdout_au_ids,
        probs=probs,
        actions=np.asarray(data.actions, dtype=object),
    )
    meta = {
        "feature": "TfidfVectorizer(char_wb, ngram_range=(3,5), max_features=120000)",
        "model": "LinearSVC(C=1.0, class_weight=balanced, max_iter=5000, random_state=42)",
        "train_protocol": "nonholdout sess_au rows only",
        "train_rows": int(len(train_idx)),
        "train_sessions": int(len(set(data.train_groups[train_idx]))),
        "holdout_au_rows": int(len(holdout_au_ids)),
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x_train.shape[1]),
        "elapsed_sec": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"probs": probs, "ids": holdout_au_ids, "actions": data.actions, "meta": meta, "cache_hit": False}


def apply_soft_au(
    data: LeagueData,
    blend: np.ndarray,
    au_probs: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    out = np.asarray(blend, dtype=np.float64).copy()
    out[data.au_mask] = alpha * au_probs + (1.0 - alpha) * out[data.au_mask]
    return out


def score_bundle(data: LeagueData, probs: np.ndarray, prefix: str = "") -> dict[str, float]:
    return {
        f"{prefix}macro_f1": macro_f1_probs(probs, data.y_true, data.actions),
        f"{prefix}au_macro_f1": macro_f1_probs(probs[data.au_mask], data.y_true[data.au_mask], data.actions),
        f"{prefix}non_au_macro_f1": macro_f1_probs(
            probs[data.non_au_mask], data.y_true[data.non_au_mask], data.actions
        ),
    }


def half_scores(data: LeagueData, probs: np.ndarray, seed: int = 42) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(data.ids))
    half = len(perm) // 2
    rows = {}
    for name, idx in (("half1", perm[:half]), ("half2", perm[half:])):
        rows[f"{name}_macro_f1"] = macro_f1_probs(probs[idx], data.y_true[idx], data.actions)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {path}")
