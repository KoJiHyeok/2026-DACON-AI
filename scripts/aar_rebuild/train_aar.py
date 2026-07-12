"""Rebuild the AAR stacker to match the surviving spec in ``aar_config.json``.

The spec (read directly from ``submit/model/stacker/aar_config.json``, which
survived on disk) is authoritative:

* three text-view SGD classifiers ("prompt_context_sgd" on the
  ``prompt_context`` view, "prompt_sgd" on the ``prompt`` view, "action_sgd"
  on the ``action`` view), and
* a rule-based "transition_prior" component that is *not* an sklearn
  estimator -- it is the empirical class-conditional frequency table format
  consumed by ``aar_transition_predict_proba`` in ``submit/aar_infer.py``.

The four component probabilities are concatenated in
``stacker_components`` order and fed to a ``LogisticRegression`` stacker.
This mirrors the real ``aar_models.joblib`` structure exactly (verified by
opening the artifact: ``components`` dict has keys
``{prompt_context_sgd, prompt_sgd, action_sgd}``, ``transition`` dict has
keys ``{actions, global, groups, weights, global_weight}``, and ``stacker``
is a ``LogisticRegression`` with ``coef_.shape == (14, 56)``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("AAR_PROJECT_ROOT", "C:/dev/2026-AI-DACON"))
if str(ROOT / "submit") not in sys.path:
    sys.path.insert(0, str(ROOT / "submit"))
import aar_infer as AAR  # noqa: E402


# Component recipe taken verbatim from aar_config.json's real spec.
COMPONENT_SPECS = {
    "prompt_context_sgd": {
        "view": "prompt_context",
        "alpha": 2.5e-05,
        "word_max_features": 120_000,
        "char_max_features": 80_000,
    },
    "prompt_sgd": {
        "view": "prompt",
        "alpha": 2e-05,
        "word_max_features": 220_000,
        "char_max_features": 180_000,
    },
    "action_sgd": {
        "view": "action",
        "alpha": 5e-05,
        "max_features": 60_000,
    },
}
STACKER_COMPONENTS = ["prompt_context_sgd", "prompt_sgd", "action_sgd", "transition_prior"]
TRANSITION_WEIGHTS = {
    "last_action_rule": 0.7,
    "last2": 0.55,
    "last_action": 0.45,
    "prompt_rule": 0.35,
    "history_len": 0.15,
    "language_pref": 0.12,
    "ci_dirty": 0.08,
}
TRANSITION_GLOBAL_WEIGHT = 0.3
TRANSITION_GROUPS = list(TRANSITION_WEIGHTS.keys())


def _json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_training(data_path: Path, labels_path: Path) -> tuple[list[dict[str, Any]], np.ndarray]:
    records = AAR.read_jsonl(data_path)
    import pandas as pd
    labels = pd.read_csv(labels_path)
    if "id" not in labels or "action" not in labels:
        raise ValueError("labels CSV must contain id and action columns")
    by_id = dict(zip(labels["id"].astype(str), labels["action"].astype(str)))
    y = np.asarray([by_id.get(str(r.get("id", "")), "") for r in records], dtype=object)
    if np.any(y == "") or len(y) != len(records):
        raise ValueError("training records and labels are not aligned by id")
    unknown = sorted(set(y) - set(AAR.ACTIONS))
    if unknown:
        raise ValueError(f"unknown action labels: {unknown}")
    return records, y


def build_views(records: Sequence[dict[str, Any]]) -> dict[str, list[Any]]:
    texts = [AAR.record_to_text(r) for r in records]
    prompts = [AAR.record_to_prompt_text(r) for r in records]
    return AAR.aar_views(records, texts, prompts)


def _sgd(seed: int, alpha: float, max_iter: int) -> SGDClassifier:
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=alpha, max_iter=max_iter,
        tol=1e-4, class_weight="balanced", random_state=seed, n_jobs=-1,
    )


def _text_pipeline(name: str, seed: int, alpha: float, max_iter: int) -> Pipeline:
    """Build a component pipeline matching the real artifact's structure.

    ``prompt_context_sgd``/``prompt_sgd`` use a word+char TF-IDF FeatureUnion;
    ``action_sgd`` uses a single word-ngram TF-IDF (verified against the
    live joblib -- ``action_sgd`` has no ``FeatureUnion`` step).
    """
    spec = COMPONENT_SPECS[name]
    if name == "action_sgd":
        features = TfidfVectorizer(
            max_features=spec["max_features"], min_df=2, ngram_range=(1, 3),
            sublinear_tf=True, token_pattern=r"(?u)\b[^\s]+\b",
        )
    else:
        features = FeatureUnion([
            ("word", TfidfVectorizer(
                max_features=spec["word_max_features"], min_df=2, ngram_range=(1, 2),
                sublinear_tf=True, token_pattern=r"(?u)\b[^\s]+\b",
            )),
            ("char", TfidfVectorizer(
                max_features=spec["char_max_features"], min_df=2, ngram_range=(3, 5),
                analyzer="char_wb", sublinear_tf=True,
            )),
        ])
    return Pipeline([
        ("features", features),
        ("clf", _sgd(seed, spec["alpha"], max_iter)),
    ])


def _proba(model: Any, values: Sequence[Any]) -> np.ndarray:
    raw = np.asarray(model.predict_proba(values), dtype=np.float64)
    out = np.zeros((len(raw), len(AAR.ACTIONS)), dtype=np.float64)
    for j, label in enumerate(getattr(model, "classes_", AAR.ACTIONS)):
        if str(label) in AAR.ACTIONS:
            out[:, AAR.ACTIONS.index(str(label))] = raw[:, j]
    out /= np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out


def greedy_blend(probas: Sequence[np.ndarray], y: np.ndarray, steps: int = 10) -> tuple[np.ndarray, list[float]]:
    """Select a nonnegative simplex blend by coordinate-greedy macro-F1.

    Used only to report an inner-selection diagnostic during OOF folds; the
    real artifact's per-component ``weight`` fields in aar_config.json (which
    do not sum in a simple simplex once transition_prior is included at 0.08)
    are the authoritative final weights when the stacker is bypassed, but the
    consumer always prefers ``use_stacker`` when true.
    """
    if not probas:
        raise ValueError("at least one candidate is required")
    n = len(probas)
    weights = np.zeros(n, dtype=np.int64)
    current = np.zeros_like(probas[0], dtype=np.float64)
    best = -1.0
    for _ in range(max(1, steps)):
        winner = None
        winner_score = best
        for i, proba in enumerate(probas):
            trial = (current * weights.sum() + proba) / (weights.sum() + 1.0)
            score = f1_score(y, [AAR.ACTIONS[j] for j in trial.argmax(1)], average="macro")
            if score > winner_score + 1e-12:
                winner, winner_score = i, score
        if winner is None:
            break
        current = (current * weights.sum() + probas[winner]) / (weights.sum() + 1.0)
        weights[winner] += 1
        best = winner_score
    if not weights.sum():
        weights[0] = 1
        current = probas[0]
    return current, (weights / weights.sum()).tolist()


def build_transition_spec(records: Sequence[dict[str, Any]], y: np.ndarray) -> dict[str, Any]:
    """Empirical class-conditional frequency tables, matching the format
    consumed by ``aar_transition_predict_proba``: a global 14-vector prior
    plus, for each group name in TRANSITION_GROUPS, a per-key 14-vector of
    P(action | group_key=value), all normalized to sum to 1.
    """
    n_actions = len(AAR.ACTIONS)
    action_to_idx = {a: i for i, a in enumerate(AAR.ACTIONS)}

    global_counts = np.zeros(n_actions, dtype=np.float64)
    group_counts: dict[str, dict[str, np.ndarray]] = {g: defaultdict(lambda: np.zeros(n_actions)) for g in TRANSITION_GROUPS}

    for record, label in zip(records, y):
        idx = action_to_idx[str(label)]
        global_counts[idx] += 1.0
        keys = AAR.aar_transition_keys(record)
        for group in TRANSITION_GROUPS:
            key = keys.get(group)
            if key is None:
                continue
            group_counts[group][key][idx] += 1.0

    global_vec = global_counts / max(global_counts.sum(), 1e-12)
    groups: dict[str, dict[str, list[float]]] = {}
    for group, per_key in group_counts.items():
        groups[group] = {}
        for key, counts in per_key.items():
            total = counts.sum()
            if total <= 0:
                continue
            groups[group][key] = (counts / total).tolist()

    return {
        "actions": list(AAR.ACTIONS),
        "global": global_vec.tolist(),
        "global_weight": TRANSITION_GLOBAL_WEIGHT,
        "weights": dict(TRANSITION_WEIGHTS),
        "groups": groups,
    }


def train_aar(records: list[dict[str, Any]], y: np.ndarray, output_dir: Path,
              *, seed: int = 42, max_iter: int = 25, folds: int = 3) -> dict[str, Any]:
    ids = np.asarray([str(r.get("id", "")) for r in records], dtype=object)
    groups_arr = np.asarray([x.rsplit("-step_", 1)[0] for x in ids], dtype=object)
    views = build_views(records)
    view_names = {spec["view"] for spec in COMPONENT_SPECS.values()}

    n = len(records)
    oof = {name: np.zeros((n, len(AAR.ACTIONS)), dtype=np.float64) for name in COMPONENT_SPECS}
    oof_transition = np.zeros((n, len(AAR.ACTIONS)), dtype=np.float64)
    splitter = GroupKFold(n_splits=folds)
    fold_scores: list[float] = []
    fold_blend_weights: list[list[float]] = []

    for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(n), y, groups_arr)):
        candidates = []
        for offset, name in enumerate(COMPONENT_SPECS):
            view = COMPONENT_SPECS[name]["view"]
            model = _text_pipeline(name, seed + fold * 100 + offset, COMPONENT_SPECS[name]["alpha"], max_iter)
            train_values = [views[view][i] for i in train_idx]
            model.fit(train_values, y[train_idx])
            valid_values = [views[view][i] for i in valid_idx]
            oof[name][valid_idx] = _proba(model, valid_values)
            candidates.append(oof[name][valid_idx])

        train_records = [records[i] for i in train_idx]
        train_labels = y[train_idx]
        transition_spec = build_transition_spec(train_records, train_labels)
        valid_records = [records[i] for i in valid_idx]
        oof_transition[valid_idx] = AAR.aar_transition_predict_proba(transition_spec, valid_records)
        candidates.append(oof_transition[valid_idx])

        _, blend_weights = greedy_blend(candidates, y[valid_idx], steps=10)
        fold_blend_weights.append(blend_weights)
        blend = sum(p * w for p, w in zip(candidates, blend_weights))
        fold_scores.append(float(f1_score(y[valid_idx], [AAR.ACTIONS[j] for j in blend.argmax(1)], average="macro")))

    stack_matrix = np.hstack([oof[name] for name in COMPONENT_SPECS] + [oof_transition]).astype(np.float32)
    stacker = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                  class_weight="balanced", random_state=seed)
    stacker.fit(stack_matrix, y)
    stacked_pred = stacker.predict(stack_matrix)
    stacked_oof_f1 = float(f1_score(y, stacked_pred, average="macro"))

    final_components = {}
    for offset, name in enumerate(COMPONENT_SPECS):
        view = COMPONENT_SPECS[name]["view"]
        model = _text_pipeline(name, seed + offset, COMPONENT_SPECS[name]["alpha"], max_iter)
        model.fit(views[view], y)
        final_components[name] = model
    final_transition = build_transition_spec(records, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "components": final_components,
        "transition": final_transition,
        "stacker": stacker,
        "actions": list(AAR.ACTIONS),
        "metadata": {"seed": seed, "views": sorted(view_names), "max_iter": max_iter},
    }
    joblib.dump(artifact, output_dir / "aar_models.joblib", compress=3)

    config = {
        "enabled": True,
        "model_file": "aar_models.joblib",
        "fallback_macro_f1": None,
        "final_valid_macro_f1": stacked_oof_f1,
        "use_bias": False,
        "class_bias": [0.0] * len(AAR.ACTIONS),
        "use_stacker": True,
        "stacker_components": list(STACKER_COMPONENTS),
        "components": [
            {"name": "prompt_context_sgd", "kind": "text", "view": "prompt_context", "weight": 0.6003},
            {"name": "prompt_sgd", "kind": "text", "view": "prompt", "weight": 0.2001},
            {"name": "action_sgd", "kind": "text", "view": "action", "weight": 0.1196},
            {"name": "transition_prior", "kind": "transition", "view": "transition", "weight": 0.08},
        ],
    }
    (output_dir / "aar_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    return {
        "rows": n, "folds": folds, "group_count": int(len(set(groups_arr))),
        "fold_macro_f1": fold_scores, "oof_macro_f1": float(np.mean(fold_scores)),
        "fold_blend_weights": fold_blend_weights,
        "stacked_oof_macro_f1": stacked_oof_f1,
        "components": list(COMPONENT_SPECS), "output": str(output_dir),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=os.getenv("AAR_DATA", str(DATA_ROOT / "data" / "train.jsonl")))
    parser.add_argument("--labels", default=os.getenv("AAR_LABELS", str(DATA_ROOT / "data" / "train_labels.csv")))
    parser.add_argument("--output", default=os.getenv("AAR_OUTPUT", str(ROOT / "submit" / "model")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("AAR_SEED", "42")))
    parser.add_argument("--max-iter", type=int, default=int(os.getenv("AAR_MAX_ITER", "25")))
    parser.add_argument("--folds", type=int, default=int(os.getenv("AAR_FOLDS", "3")))
    parser.add_argument("--limit", type=int, default=int(os.getenv("AAR_LIMIT", "0")),
                         help="If > 0, subsample to this many rows (for smoke runs).")
    args = parser.parse_args(argv)
    records, y = load_training(Path(args.data), Path(args.labels))
    if args.limit and args.limit < len(records):
        records, y = records[: args.limit], y[: args.limit]
    result = train_aar(records, y, Path(args.output), seed=args.seed,
                        max_iter=args.max_iter, folds=args.folds)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
