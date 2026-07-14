# -*- coding: utf-8 -*-
"""Shared, deployment-compatible helpers for the CX-C AU linear sweep."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
MAIN_ROOT = Path(os.environ.get("DACON_MAIN_ROOT", r"C:\dev\2026-AI-DACON"))
DATA_DIR = MAIN_ROOT / "data"
HOLDOUT_NPZ = MAIN_ROOT / "colab_out" / "qwen_i2ep_h85.npz"
OOF_DIR = MAIN_ROOT / "artifacts" / "oof" / "oof_rebuild_2026_07_04"
CURRENT_AU_MODEL = MAIN_ROOT / "submit" / "model" / "au_linear" / "model.pkl"
CANDIDATE_DIR = ROOT / "scripts" / "cx_au2" / "au_linear_candidate"

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

_STEP_RE = re.compile(r"-step_\d+$")


@dataclass(frozen=True)
class Variant:
    name: str
    feature_kind: str
    c: float
    class_weight: str | None = "balanced"


# Predeclared before reading holdout labels. Keep this list at five or fewer.
VARIANTS = (
    Variant("baseline_char_C1", "char", 1.0),
    Variant("char_C0.5", "char", 0.5),
    Variant("word_char_C0.25", "word_char", 0.25),
    Variant("word_char_C0.5", "word_char", 0.5),
    Variant("word_char_C1", "word_char", 1.0),
)


def session_id(sample_id: str) -> str:
    return _STEP_RE.sub("", str(sample_id))


def is_au(sample_id: str) -> bool:
    return str(sample_id).startswith("sess_au")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def load_au_route():
    path = MAIN_ROOT / "submit" / "au_route.py"
    if not path.exists():
        path = ROOT / "submit" / "au_route.py"
    spec = importlib.util.spec_from_file_location("cx_au2_au_route", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import au_route from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_train(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as f:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    ids = np.asarray([str(sample["id"]) for sample in samples], dtype=object)
    y = np.asarray([labels[str(sample_id)] for sample_id in ids], dtype=object)
    groups = np.asarray([session_id(str(sample_id)) for sample_id in ids], dtype=object)
    return samples, ids, y, groups


def load_holdout(path: Path = HOLDOUT_NPZ) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as z:
        required = {"ids", "probs", "y_true", "actions"}
        missing = required - set(z.files)
        if missing:
            raise AssertionError(f"{path} missing arrays: {sorted(missing)}")
        ids = np.asarray([str(x) for x in z["ids"]], dtype=object)
        y_true = np.asarray([str(x) for x in z["y_true"]], dtype=object)
        actions = [str(x) for x in z["actions"]]
        probs = np.asarray(z["probs"], dtype=np.float64)
    if len(set(ids.tolist())) != len(ids):
        raise AssertionError("holdout ids contain duplicates")
    assert_probabilities("qwen", probs, len(ids), len(actions))
    return {"ids": ids, "y_true": y_true, "actions": actions, "probs": probs}


def assert_holdout_excluded(train_ids: Iterable[str], holdout_ids: Iterable[str]) -> None:
    overlap = set(map(str, train_ids)) & set(map(str, holdout_ids))
    if overlap:
        example = sorted(overlap)[:3]
        raise AssertionError(f"holdout id leaked into AU train: {example} ({len(overlap)} total)")


def assert_group_disjoint(train_groups: Iterable[str], valid_groups: Iterable[str], label: str) -> None:
    overlap = set(map(str, train_groups)) & set(map(str, valid_groups))
    if overlap:
        raise AssertionError(f"{label} group leakage: {len(overlap)} overlapping sessions")


def select_nonholdout_au(
    samples: Sequence[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    holdout_ids: Sequence[str],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    holdout_set = set(map(str, holdout_ids))
    idx = np.asarray(
        [i for i, sample_id in enumerate(ids) if is_au(str(sample_id)) and str(sample_id) not in holdout_set],
        dtype=np.int64,
    )
    selected_ids = np.asarray(ids[idx], dtype=object)
    selected_groups = np.asarray(groups[idx], dtype=object)
    assert_holdout_excluded(selected_ids, holdout_ids)
    holdout_au_groups = [session_id(str(x)) for x in holdout_ids if is_au(str(x))]
    assert_group_disjoint(selected_groups, holdout_au_groups, "frozen holdout")
    return [samples[int(i)] for i in idx], selected_ids, np.asarray(y[idx], dtype=object), selected_groups


def make_vectorizer(feature_kind: str):
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    if feature_kind == "char":
        return char
    if feature_kind == "word_char":
        word = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=80_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(f"unknown feature kind: {feature_kind}")


def make_classifier(variant: Variant, seed: int = 42) -> LinearSVC:
    return LinearSVC(
        C=variant.c,
        class_weight=variant.class_weight,
        max_iter=5000,
        random_state=seed,
    )


def fit_artifact(
    samples: Sequence[dict[str, Any]],
    y: Sequence[str],
    variant: Variant,
    seed: int = 42,
) -> dict[str, Any]:
    au_route = load_au_route()
    texts = [au_route.serialize(sample) for sample in samples]
    union = make_vectorizer(variant.feature_kind)
    x = union.fit_transform(texts)
    clf = make_classifier(variant, seed)
    clf.fit(x, np.asarray(y, dtype=object))
    return {"union": union, "clf": clf}


def dump_artifact(path: Path, artifact: dict[str, Any]) -> None:
    if set(artifact) != {"union", "clf"}:
        raise AssertionError(f"artifact keys must be union/clf, got {sorted(artifact)}")
    if not hasattr(artifact["union"], "transform") or not hasattr(artifact["clf"], "decision_function"):
        raise AssertionError("artifact is incompatible with au_route.predict_proba")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, path, compress=3)
    print(f"[save] {path}")


def load_artifact(path: Path) -> dict[str, Any]:
    artifact = joblib.load(path)
    if not isinstance(artifact, dict) or set(artifact) != {"union", "clf"}:
        raise AssertionError(f"invalid AU artifact at {path}")
    return artifact


def align_probabilities(
    probs: np.ndarray,
    src_actions: Sequence[str],
    dst_actions: Sequence[str],
) -> np.ndarray:
    src = [str(x) for x in src_actions]
    out = np.zeros((len(probs), len(dst_actions)), dtype=np.float64)
    for j, action in enumerate(dst_actions):
        if str(action) in src:
            out[:, j] = probs[:, src.index(str(action))]
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise AssertionError("probability alignment produced an empty row")
    return out / row_sum


def predict_artifact(
    artifact: dict[str, Any],
    samples: Sequence[dict[str, Any]],
    actions: Sequence[str] = ACTIONS,
) -> np.ndarray:
    au_route = load_au_route()
    probs, classes = au_route.predict_proba(artifact, samples)
    return align_probabilities(np.asarray(probs, dtype=np.float64), classes, actions)


def assert_probabilities(name: str, probs: np.ndarray, n_rows: int, n_classes: int) -> None:
    if probs.shape != (n_rows, n_classes):
        raise AssertionError(f"{name} shape {probs.shape} != {(n_rows, n_classes)}")
    if not np.all(np.isfinite(probs)):
        raise AssertionError(f"{name} has non-finite values")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-6):
        raise AssertionError(f"{name} rows do not sum to one")


def variant_payload(variant: Variant) -> dict[str, Any]:
    return asdict(variant)


def variant_from_payload(payload: dict[str, Any]) -> Variant:
    return Variant(
        name=str(payload["name"]),
        feature_kind=str(payload["feature_kind"]),
        c=float(payload["c"]),
        class_weight=payload.get("class_weight"),
    )
