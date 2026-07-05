# -*- coding: utf-8 -*-
"""Collect 85/15 grouped-holdout probabilities for local blend search.

The split contract matches ``colab/holdout_eval.py``:
StratifiedGroupKFold, seed=42, groups = ``id.rsplit("-step_", 1)[0]``.

Outputs are NPZ files with:
  ids      : holdout sample ids
  probs    : N x 14 probabilities
  y_true   : action labels
  actions  : probability column labels
  meta_json: JSON metadata string

Linear is refit on the 85% training split. Stacker uses the existing AAR
artifacts as inference-only probabilities, because this bundle does not include
the source needed to retrain the stacker on the same 85% split.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold


DEFAULT_DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
DEFAULT_BOOST_DIR = Path(r"C:\dev\dacon-agent-action-api-boost")
DEFAULT_OUT_DIR = Path("context/night/2026-07-05")
DEFAULT_ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
]


def import_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_cols(probs: np.ndarray, src_labels: Sequence[str], dst_labels: Sequence[str]) -> np.ndarray:
    src = [str(x) for x in src_labels]
    missing = sorted(set(dst_labels) - set(src))
    if missing:
        raise ValueError(f"missing probability labels: {missing}")
    idx = [src.index(a) for a in dst_labels]
    return np.asarray(probs, dtype=np.float64)[:, idx]


def load_train(data_dir: Path):
    train_jsonl = data_dir / "train.jsonl"
    labels_csv = data_dir / "train_labels.csv"
    if not train_jsonl.exists():
        raise FileNotFoundError(train_jsonl)
    if not labels_csv.exists():
        raise FileNotFoundError(labels_csv)

    samples: list[dict[str, Any]] = []
    with train_jsonl.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with labels_csv.open(encoding="utf-8") as f:
        labels = {r["id"]: r["action"] for r in csv.DictReader(f)}

    ids = np.array([str(s["id"]) for s in samples])
    y = np.array([labels[str(s["id"])] for s in samples])
    groups = np.array([session_id(i) for i in ids])
    return samples, ids, y, groups


def make_holdout_split(y: np.ndarray, groups: np.ndarray, seed: int, valid_frac: float):
    n_splits = max(2, int(round(1.0 / valid_frac)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
    overlap = set(groups[tr_idx]) & set(groups[va_idx])
    if overlap:
        raise AssertionError(f"session leakage: {len(overlap)} overlapping groups")
    return tr_idx, va_idx, n_splits


def summarize_probs(name: str, probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> dict[str, Any]:
    preds = np.array(actions)[np.asarray(probs).argmax(axis=1)]
    per = f1_score(y_true, preds, average=None, labels=list(actions), zero_division=0)
    return {
        "name": name,
        "macro_f1": float(f1_score(y_true, preds, average="macro", zero_division=0)),
        "per_class_f1": {a: float(v) for a, v in zip(actions, per)},
    }


def save_npz(path: Path, ids: np.ndarray, probs: np.ndarray, y_true: np.ndarray,
             actions: Sequence[str], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ids=ids.astype(str),
        probs=np.asarray(probs, dtype=np.float64),
        y_true=y_true.astype(str),
        actions=np.array(list(actions), dtype=object),
        meta_json=np.array(json.dumps(meta, ensure_ascii=False, sort_keys=True), dtype=object),
    )
    print(f"[save] {path} ids={ids.shape} probs={probs.shape}")


def collect_linear(args, samples, ids, y, tr_idx, va_idx):
    features_path = args.boost_dir / "linear_pipeline" / "features.py"
    F = import_from_path("boost_linear_features", features_path)
    fs = F.FEATURE_SETS[args.linear_feature_set]
    df = F.build_dataframe(samples)
    pipe = F.build_pipeline(
        fs,
        clf=args.linear_clf,
        C=args.linear_c,
        alpha=args.linear_alpha,
        max_iter=args.linear_max_iter,
    )
    print(
        f"[linear] fit feature_set={args.linear_feature_set} clf={args.linear_clf} "
        f"C={args.linear_c:g} alpha={args.linear_alpha:g} train={len(tr_idx)}"
    )
    pipe.fit(df.iloc[tr_idx], y[tr_idx])
    classes = [str(c) for c in pipe.named_steps["clf"].classes_]
    if hasattr(pipe, "decision_function"):
        probs = softmax(pipe.decision_function(df.iloc[va_idx]))
    elif hasattr(pipe, "predict_proba"):
        probs = np.asarray(pipe.predict_proba(df.iloc[va_idx]), dtype=np.float64)
    else:
        preds = [str(p) for p in pipe.predict(df.iloc[va_idx])]
        probs = np.zeros((len(preds), len(classes)), dtype=np.float64)
        c2i = {c: i for i, c in enumerate(classes)}
        for i, pred in enumerate(preds):
            probs[i, c2i[pred]] = 1.0
    probs = align_cols(probs, classes, DEFAULT_ACTIONS)
    meta = {
        "component": "linear",
        "source": str(features_path),
        "feature_set": args.linear_feature_set,
        "clf": args.linear_clf,
        "C": args.linear_c,
        "alpha": args.linear_alpha,
        "honest_holdout": True,
        "note": "model was fit only on StratifiedGroupKFold train split",
    }
    summary = summarize_probs("linear", probs, y[va_idx], DEFAULT_ACTIONS)
    meta.update(summary)
    save_npz(args.out_dir / args.linear_out, ids[va_idx], probs, y[va_idx], DEFAULT_ACTIONS, meta)
    print("[linear] macro_f1={:.6f}".format(summary["macro_f1"]))
    return summary


def collect_stacker(args, samples, ids, y, va_idx):
    aar_path = args.boost_dir / "ensemble" / "aar_infer.py"
    AAR = import_from_path("boost_aar_infer", aar_path)
    stacker_dir = args.stacker_dir or (args.boost_dir / "model")
    config = AAR.load_config(stacker_dir / "aar_config.json")
    if not config.get("enabled"):
        raise ValueError(f"AAR stacker is disabled in {stacker_dir / 'aar_config.json'}")
    artifact = joblib.load(stacker_dir / str(config.get("model_file", "aar_models.joblib")))

    holdout_samples = [samples[int(i)] for i in va_idx]
    texts = [AAR.record_to_text(r) for r in holdout_samples]
    prompt_texts = [AAR.record_to_prompt_text(r) for r in holdout_samples]
    views = AAR.aar_views(holdout_samples, texts, prompt_texts)
    comp: dict[str, np.ndarray] = {}
    for component in config.get("components", []):
        name = str(component.get("name"))
        kind = str(component.get("kind"))
        view = str(component.get("view"))
        if kind == "transition":
            comp[name] = AAR.aar_transition_predict_proba(artifact["transition"], holdout_samples)
        else:
            model = artifact.get("components", {}).get(name)
            if model is None:
                raise ValueError(f"AAR component missing: {name}")
            comp[name] = AAR.predict_proba_aligned(model, views[view])
    if config.get("use_stacker"):
        names = [str(x) for x in config.get("stacker_components", [])]
        matrix = np.hstack([comp[n] for n in names]).astype(np.float32)
        probs = AAR.predict_proba_aligned(artifact["stacker"], matrix)
    else:
        probs = AAR.weighted_average(
            [(comp[str(c["name"])], float(c.get("weight", 0.0))) for c in config["components"]]
        )
    if config.get("use_bias"):
        probs = AAR.aar_apply_bias(probs, config.get("class_bias", [0.0] * len(DEFAULT_ACTIONS)))
    probs = align_cols(probs, list(AAR.ACTIONS), DEFAULT_ACTIONS)
    meta = {
        "component": "stacker",
        "source": str(stacker_dir),
        "config": str(stacker_dir / "aar_config.json"),
        "honest_holdout": False,
        "note": (
            "existing artifact was used for holdout inference; source retrain code for "
            "85% split was not available, so this score is leakage-prone and should not "
            "be treated as an unbiased holdout estimate"
        ),
    }
    summary = summarize_probs("stacker_artifact", probs, y[va_idx], DEFAULT_ACTIONS)
    meta.update(summary)
    save_npz(args.out_dir / args.stacker_out, ids[va_idx], probs, y[va_idx], DEFAULT_ACTIONS, meta)
    print("[stacker] macro_f1={:.6f} (artifact-only; leakage-prone)".format(summary["macro_f1"]))
    return summary


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--boost-dir", type=Path, default=DEFAULT_BOOST_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--component", choices=["linear", "stacker", "both"], default="both")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--linear-feature-set", default="E_+seq")
    p.add_argument("--linear-clf", choices=["svc", "sgd", "logreg"], default="svc")
    p.add_argument("--linear-c", type=float, default=1.0)
    p.add_argument("--linear-alpha", type=float, default=1e-4)
    p.add_argument("--linear-max-iter", type=int, default=1000)
    p.add_argument("--linear-out", default="holdout_linear.npz")
    p.add_argument("--stacker-dir", type=Path, default=None)
    p.add_argument("--stacker-out", default="holdout_stacker.npz")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    samples, ids, y, groups = load_train(args.data_dir)
    tr_idx, va_idx, n_splits = make_holdout_split(y, groups, args.seed, args.valid_frac)
    split_meta = {
        "seed": args.seed,
        "valid_frac_requested": args.valid_frac,
        "n_splits": n_splits,
        "n_rows": int(len(y)),
        "n_train": int(len(tr_idx)),
        "n_holdout": int(len(va_idx)),
        "actual_holdout_frac": float(len(va_idx) / len(y)),
        "n_groups": int(len(set(groups))),
        "n_train_groups": int(len(set(groups[tr_idx]))),
        "n_holdout_groups": int(len(set(groups[va_idx]))),
        "group_overlap": int(len(set(groups[tr_idx]) & set(groups[va_idx]))),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "holdout_split_meta.json").write_text(
        json.dumps(split_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[split] " + json.dumps(split_meta, ensure_ascii=False, sort_keys=True))

    summaries = []
    if args.component in {"linear", "both"}:
        summaries.append(collect_linear(args, samples, ids, y, tr_idx, va_idx))
    if args.component in {"stacker", "both"}:
        summaries.append(collect_stacker(args, samples, ids, y, va_idx))

    print(json.dumps({"split": split_meta, "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
