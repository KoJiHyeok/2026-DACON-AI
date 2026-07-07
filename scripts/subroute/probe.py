# -*- coding: utf-8 -*-
"""Specialist probes for weak subroute candidates from `sweep.py`."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from common import (
    DATA_DIR,
    DEFAULT_ALPHAS,
    HOLDOUT_BASE,
    MBERT_HOLDOUT,
    OOF_DIR,
    OUT_DIR,
    SCREEN_MIN_HOLDOUT,
    SCREEN_MIN_TRAIN,
    SCREEN_WEAK_DELTA,
    SOFT_AU_ALPHA,
    align_probs,
    build_context,
    candidate_mask,
    is_au,
    load_league,
    load_train,
    macro_f1_labels,
    macro_f1_probs,
    predict_labels,
    save_csv,
    save_json,
    softmax,
)


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
if str(SUBMIT_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMIT_DIR))
import au_route  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--holdout-base", type=Path, default=HOLDOUT_BASE)
    parser.add_argument("--mbert-holdout", type=Path, default=MBERT_HOLDOUT)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--sweep-json", type=Path, default=OUT_DIR / "sweep.json")
    parser.add_argument("--min-holdout", type=int, default=SCREEN_MIN_HOLDOUT)
    parser.add_argument("--min-train", type=int, default=SCREEN_MIN_TRAIN)
    parser.add_argument("--weak-delta", type=float, default=SCREEN_WEAK_DELTA)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(DEFAULT_ALPHAS))
    parser.add_argument("--max-candidates", type=int, default=0, help="0 means all screened candidates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=120_000,
        sublinear_tf=True,
        strip_accents="unicode",
    )


def fit_predict_specialist(
    *,
    train_samples: Sequence[dict[str, Any]],
    train_y: np.ndarray,
    eval_samples: Sequence[dict[str, Any]],
    actions: Sequence[str],
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    if len(set(str(y) for y in train_y)) < 2:
        raise ValueError("LinearSVC specialist requires at least two classes")
    vec = make_vectorizer()
    train_texts = [au_route.serialize(sample) for sample in train_samples]
    eval_texts = [au_route.serialize(sample) for sample in eval_samples]
    x_train = vec.fit_transform(train_texts)
    x_eval = vec.transform(eval_texts)
    if not sparse.issparse(x_train) or not sparse.issparse(x_eval):
        raise TypeError("TF-IDF output must be sparse")
    clf = LinearSVC(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed)
    clf.fit(x_train, train_y)
    probs = softmax(clf.decision_function(x_eval))
    probs = align_probs(probs, [str(c) for c in clf.classes_], actions)
    return {
        "probs": probs,
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x_train.shape[1]),
    }


def load_sweep_candidates(path: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        summary = json.load(f)
    rows = []
    for row in summary.get("rows", []):
        delta = float(row.get("delta_vs_overall_4way", math.nan))
        if (
            int(row.get("holdout_rows", 0)) >= args.min_holdout
            and int(row.get("train_nonholdout_rows", 0)) >= args.min_train
            and not math.isnan(delta)
            and delta <= args.weak_delta
        ):
            rows.append(row)
    rows = sorted(rows, key=lambda row: (float(row["delta_vs_overall_4way"]), -int(row["holdout_rows"])))
    if args.max_candidates and args.max_candidates > 0:
        rows = rows[: args.max_candidates]
    return rows


def build_soft_au_baseline(
    *,
    samples: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    ctx: Any,
    league: dict[str, Any],
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    au_train_idx = np.asarray(
        [
            i
            for i, sample_id in enumerate(ids)
            if ctx.nonholdout_mask[i] and is_au(str(sample_id))
        ],
        dtype=np.int64,
    )
    au_holdout_mask = np.asarray([is_au(str(sample_id)) for sample_id in league["ids"]], dtype=bool)
    au_eval_samples = [ctx.holdout_samples[int(i)] for i in np.where(au_holdout_mask)[0]]
    result = fit_predict_specialist(
        train_samples=[samples[int(i)] for i in au_train_idx],
        train_y=y[au_train_idx],
        eval_samples=au_eval_samples,
        actions=league["actions"],
        c_value=c_value,
        seed=seed,
    )
    soft_probs = league["blend4"].copy()
    soft_probs[au_holdout_mask] = (
        SOFT_AU_ALPHA * result["probs"] + (1.0 - SOFT_AU_ALPHA) * league["blend4"][au_holdout_mask]
    )
    return {
        "probs": soft_probs,
        "score": macro_f1_probs(soft_probs, league["y_true"], league["actions"]),
        "au_train_rows": int(len(au_train_idx)),
        "au_holdout_rows": int(au_holdout_mask.sum()),
        "au_n_features": int(result["n_features"]),
        "au_classes": result["classes"],
    }


def evaluate_candidate(
    *,
    row: dict[str, Any],
    samples: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    ctx: Any,
    league: dict[str, Any],
    soft_au: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    spec = row["spec"]
    train_mask = candidate_mask(spec, ctx.flats) & ctx.nonholdout_mask & ctx.train_non_au_mask
    holdout_mask = candidate_mask(spec, ctx.holdout_flats) & ctx.holdout_non_au_mask
    train_idx = np.where(train_mask)[0]
    holdout_idx = np.where(holdout_mask)[0]
    if any(str(ids[int(i)]) in ctx.holdout_id_set for i in train_idx):
        raise AssertionError(f"holdout leakage in train for {row['name']}")

    started = time.time()
    result = fit_predict_specialist(
        train_samples=[samples[int(i)] for i in train_idx],
        train_y=y[train_idx],
        eval_samples=[ctx.holdout_samples[int(i)] for i in holdout_idx],
        actions=league["actions"],
        c_value=args.c,
        seed=args.seed,
    )
    spec_probs = result["probs"]
    y_group = league["y_true"][holdout_mask]
    blend_group_probs = league["blend4"][holdout_mask]
    blend_group_pred = predict_labels(blend_group_probs, league["actions"])
    specialist_pred = predict_labels(spec_probs, league["actions"])
    blend_group_f1 = macro_f1_labels(y_group, blend_group_pred)
    specialist_group_f1 = macro_f1_labels(y_group, specialist_pred)

    route_rows: list[dict[str, Any]] = []
    for alpha in args.alphas:
        mixed = alpha * spec_probs + (1.0 - alpha) * blend_group_probs
        mixed_pred = predict_labels(mixed, league["actions"])
        hybrid = league["blend4"].copy()
        hybrid[holdout_mask] = mixed
        hybrid_soft_au = soft_au["probs"].copy()
        hybrid_soft_au[holdout_mask] = mixed
        score4 = macro_f1_probs(hybrid, league["y_true"], league["actions"])
        score_soft_au = macro_f1_probs(hybrid_soft_au, league["y_true"], league["actions"])
        group_f1 = macro_f1_labels(y_group, mixed_pred)
        route_rows.append(
            {
                "candidate": row["name"],
                "alpha": float(alpha),
                "group_macro_f1": group_f1,
                "group_delta_vs_blend": group_f1 - blend_group_f1,
                "league_macro_f1_vs_4way": score4,
                "delta_vs_4way": score4 - float(league["blend4_score"]),
                "league_macro_f1_vs_soft_au": score_soft_au,
                "delta_vs_soft_au": score_soft_au - float(soft_au["score"]),
                "changed_group_vs_blend": int(np.sum(mixed_pred != blend_group_pred)),
            }
        )
    best = max(route_rows, key=lambda item: (float(item["delta_vs_soft_au"]), float(item["group_delta_vs_blend"])))
    specialist_margin = specialist_group_f1 - blend_group_f1
    if specialist_margin < 0.02:
        decision = "discard_info_limited"
    elif float(best["delta_vs_soft_au"]) >= 0.005:
        decision = "lb_gate_candidate"
    elif float(best["delta_vs_soft_au"]) >= 0.002:
        decision = "report_only"
    else:
        decision = "discard"
    return {
        "candidate": row["name"],
        "family": row["family"],
        "spec": spec,
        "screen": {
            "holdout_rows": int(row["holdout_rows"]),
            "train_nonholdout_rows": int(row["train_nonholdout_rows"]),
            "blend4_group_macro_f1_from_sweep": float(row["blend4_group_macro_f1"]),
            "delta_vs_overall_4way": float(row["delta_vs_overall_4way"]),
        },
        "fit": {
            "train_rows": int(len(train_idx)),
            "holdout_rows": int(len(holdout_idx)),
            "train_classes": sorted(str(c) for c in set(y[train_idx])),
            "n_features": int(result["n_features"]),
            "classes": result["classes"],
            "elapsed_sec": round(time.time() - started, 3),
        },
        "group_eval": {
            "blend4_macro_f1": blend_group_f1,
            "specialist_hard_macro_f1": specialist_group_f1,
            "specialist_minus_blend": specialist_margin,
            "passes_specialist_margin_0.02": bool(specialist_margin >= 0.02),
        },
        "route_rows": route_rows,
        "best": {**best, "decision": decision},
    }


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_sweep_candidates(args.sweep_json, args)
    print(f"[load] screened candidates={len(candidates)} from {args.sweep_json}")

    samples, ids, y, groups = load_train(args.data_dir)
    league = load_league(args.holdout_base, args.oof_dir, args.mbert_holdout)
    ctx = build_context(samples, ids, y, groups, league["ids"])
    summary_path = args.out_dir / "probe.json"

    summary: dict[str, Any] = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "oof_dir": str(args.oof_dir),
            "holdout_base": str(args.holdout_base),
            "mbert_holdout": str(args.mbert_holdout),
            "sweep_json": str(args.sweep_json),
            "alphas": [float(a) for a in args.alphas],
            "specialist": "char_wb(3-5) max_features=120000 + LinearSVC(C=1.0, class_weight=balanced)",
            "seed": args.seed,
        },
        "screen": {
            "weak_delta_vs_overall_4way_lte": args.weak_delta,
            "min_holdout_rows": args.min_holdout,
            "min_train_nonholdout_rows": args.min_train,
            "specialist_margin_gate": 0.02,
            "lb_gate_candidate_delta": 0.005,
            "report_only_delta_min": 0.002,
        },
        "baseline": {
            "blend3_macro_f1": float(league["blend3_score"]),
            "blend4_macro_f1": float(league["blend4_score"]),
        },
        "candidate_order": [row["name"] for row in candidates],
        "results": [],
        "best": None,
    }
    if not candidates:
        save_json(summary_path, summary)
        save_csv(args.out_dir / "probe_route_rows.csv", [])
        print("[done] no screened candidates")
        return

    print("[fit] current soft-AU baseline on sess_au only")
    soft_au = build_soft_au_baseline(
        samples=samples,
        ids=ids,
        y=y,
        ctx=ctx,
        league=league,
        c_value=args.c,
        seed=args.seed,
    )
    summary["baseline"].update(
        {
            "soft_au_alpha": SOFT_AU_ALPHA,
            "soft_au_macro_f1": float(soft_au["score"]),
            "soft_au_delta_vs_4way": float(soft_au["score"] - league["blend4_score"]),
            "soft_au_train_rows": soft_au["au_train_rows"],
            "soft_au_holdout_rows": soft_au["au_holdout_rows"],
            "soft_au_n_features": soft_au["au_n_features"],
        }
    )
    save_json(summary_path, summary)

    for row in candidates:
        print(
            "[probe] {name} holdout={holdout} train={train} weak_delta={delta:+.6f}".format(
                name=row["name"],
                holdout=int(row["holdout_rows"]),
                train=int(row["train_nonholdout_rows"]),
                delta=float(row["delta_vs_overall_4way"]),
            )
        )
        result = evaluate_candidate(
            row=row,
            samples=samples,
            ids=ids,
            y=y,
            ctx=ctx,
            league=league,
            soft_au=soft_au,
            args=args,
        )
        summary["results"].append(result)
        route_rows = [route for item in summary["results"] for route in item["route_rows"]]
        summary["best"] = max(route_rows, key=lambda item: (float(item["delta_vs_soft_au"]), float(item["group_delta_vs_blend"])))
        summary["elapsed_sec"] = round(time.time() - t0, 3)
        save_json(summary_path, summary)
        save_csv(args.out_dir / "probe_route_rows.csv", route_rows)
        print(
            "[result] {name} hard_margin={margin:+.6f} best_alpha={alpha:g} "
            "delta_soft_au={delta:+.6f} decision={decision}".format(
                name=result["candidate"],
                margin=float(result["group_eval"]["specialist_minus_blend"]),
                alpha=float(result["best"]["alpha"]),
                delta=float(result["best"]["delta_vs_soft_au"]),
                decision=result["best"]["decision"],
            )
        )
    save_json(summary_path, summary)
    print(f"[done] elapsed={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
