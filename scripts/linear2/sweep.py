# -*- coding: utf-8 -*-
"""Sweep linear replacement candidates on the saved 2026-07-04 folds."""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.svm import LinearSVC

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.linear2 import common as C
else:
    from . import common as C


DEFAULT_VARIANTS: list[dict[str, Any]] = [
    {"variant": "char_3-5_mf120k_C1", "feature_kind": "char", "ngram": [3, 5], "max_features": 120_000, "C": 1.0},
    {"variant": "char_3-5_mf120k_C0.5", "feature_kind": "char", "ngram": [3, 5], "max_features": 120_000, "C": 0.5},
    {"variant": "char_3-5_mf120k_C2", "feature_kind": "char", "ngram": [3, 5], "max_features": 120_000, "C": 2.0},
    {"variant": "char_2-5_mf120k_C1", "feature_kind": "char", "ngram": [2, 5], "max_features": 120_000, "C": 1.0},
    {"variant": "char_2-4_mf120k_C1", "feature_kind": "char", "ngram": [2, 4], "max_features": 120_000, "C": 1.0},
    {"variant": "union_3-5_mf120k_C1", "feature_kind": "word_char", "ngram": [3, 5], "max_features": 120_000, "C": 1.0},
    {"variant": "union_2-5_mf120k_C1", "feature_kind": "word_char", "ngram": [2, 5], "max_features": 120_000, "C": 1.0},
    {"variant": "union_2-4_mf120k_C1", "feature_kind": "word_char", "ngram": [2, 4], "max_features": 120_000, "C": 1.0},
    {"variant": "char_2-5_mf120k_C0.5", "feature_kind": "char", "ngram": [2, 5], "max_features": 120_000, "C": 0.5},
    {"variant": "char_3-5_mf200k_C1", "feature_kind": "char", "ngram": [3, 5], "max_features": 200_000, "C": 1.0},
    {"variant": "char_3-5_mf300k_C1", "feature_kind": "char", "ngram": [3, 5], "max_features": 300_000, "C": 1.0},
    {"variant": "union_3-5_mf200k_C1", "feature_kind": "word_char", "ngram": [3, 5], "max_features": 200_000, "C": 1.0},
]


def decision_probs(clf: LinearSVC, x_valid: Any) -> np.ndarray:
    scores = clf.decision_function(x_valid)
    return C.align_probs(C.softmax(scores), [str(c) for c in clf.classes_], C.ACTIONS).astype(np.float32)


def variant_out_dir(out_dir: Path, variant: str) -> Path:
    return out_dir / variant


def fold_paths(out_dir: Path, variant: str, fold_no: int) -> tuple[Path, Path]:
    base = variant_out_dir(out_dir, variant)
    return base / f"fold{fold_no}_probs.npy", base / f"fold{fold_no}_meta.json"


def run_fold(
    *,
    spec: dict[str, Any],
    fold: dict[str, Any],
    texts: list[str],
    y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    fold_no = int(fold["fold"])
    probs_path, meta_path = fold_paths(args.out_dir, str(spec["variant"]), fold_no)
    probs_path.parent.mkdir(parents=True, exist_ok=True)
    if probs_path.exists() and meta_path.exists() and not args.force:
        print(f"[{spec['variant']} fold {fold_no}] cache hit")
        return C.read_json(meta_path)

    tr_idx = fold["train_idx"]
    va_idx = fold["valid_idx"]
    ngram_min, ngram_max = [int(x) for x in spec["ngram"]]
    t0 = time.time()
    vectorizer = C.make_vectorizer(
        feature_kind=str(spec["feature_kind"]),
        ngram_min=ngram_min,
        ngram_max=ngram_max,
        max_features=int(spec["max_features"]),
        word_max_features=args.word_max_features,
    )
    train_texts = [texts[int(i)] for i in tr_idx]
    valid_texts = [texts[int(i)] for i in va_idx]
    print(f"[{spec['variant']} fold {fold_no}] vectorize train={len(tr_idx)} valid={len(va_idx)}")
    x_train = vectorizer.fit_transform(train_texts)
    x_valid = vectorizer.transform(valid_texts)
    clf = LinearSVC(
        C=float(spec["C"]),
        class_weight="balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        tol=args.tol,
        dual=True,
    )
    print(f"[{spec['variant']} fold {fold_no}] fit x_train={x_train.shape} nnz={getattr(x_train, 'nnz', -1)}")
    clf.fit(x_train, y[tr_idx])
    probs = decision_probs(clf, x_valid)
    macro = C.macro_f1_probs(probs, y[va_idx], C.ACTIONS)
    np.save(probs_path, probs.astype(np.float32))
    meta = {
        "variant": spec["variant"],
        "fold": fold_no,
        "train_rows": int(len(tr_idx)),
        "valid_rows": int(len(va_idx)),
        "macro_f1": macro,
        "per_class_f1": C.per_class_f1(probs, y[va_idx], C.ACTIONS),
        "n_features": int(x_train.shape[1]),
        "nnz_train": int(getattr(x_train, "nnz", -1)),
        "classes_seen": [str(c) for c in clf.classes_],
        "elapsed_sec": round(time.time() - t0, 3),
        "spec": spec,
        "serializer": args.serializer,
        "solver": {
            "class": "LinearSVC",
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "seed": int(args.seed),
            "dual": True,
            "class_weight": "balanced",
        },
    }
    C.write_json(meta_path, meta)
    print(f"[{spec['variant']} fold {fold_no}] macro_f1={macro:.6f} elapsed={meta['elapsed_sec']:.1f}s")
    return meta


def assemble_variant(
    *,
    spec: dict[str, Any],
    folds: list[dict[str, Any]],
    ids: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out = variant_out_dir(args.out_dir, str(spec["variant"]))
    probs = np.zeros((len(ids), len(C.ACTIONS)), dtype=np.float32)
    fold_assign = np.full(len(ids), -1, dtype=np.int16)
    fold_meta = []
    for fold in folds:
        fold_no = int(fold["fold"])
        probs_path, meta_path = fold_paths(args.out_dir, str(spec["variant"]), fold_no)
        if not probs_path.exists():
            raise FileNotFoundError(probs_path)
        valid_idx = fold["valid_idx"]
        probs[valid_idx] = np.load(probs_path).astype(np.float32)
        fold_assign[valid_idx] = fold_no
        fold_meta.append(C.read_json(meta_path))
    missing = np.where(fold_assign < 0)[0]
    if len(missing):
        raise AssertionError(f"{spec['variant']} missing OOF rows: {len(missing)}")
    oof_macro = C.macro_f1_probs(probs, y, C.ACTIONS)
    league = C.evaluate_lin_replacement(lin_probs_all=probs, row_ids=[str(x) for x in ids])
    delta = float(league["delta_vs_baseline_soft_au"])
    if delta >= 0.005:
        decision = "LB_candidate"
    elif delta >= 0.002:
        decision = "report_only"
    else:
        decision = "discard"
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "variant": spec["variant"],
        "spec": spec,
        "serializer": args.serializer,
        "n_rows": int(len(ids)),
        "oof_macro_f1": oof_macro,
        "oof_per_class_f1": C.per_class_f1(probs, y, C.ACTIONS),
        "folds": fold_meta,
        "league": league,
        "league_macro_f1": float(league["league_macro_f1"]),
        "delta_vs_baseline_soft_au": delta,
        "decision": decision,
        "baseline_soft_au_macro_f1": float(league["baseline_soft_au_macro_f1"]),
        "solver": {
            "class": "LinearSVC",
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "seed": int(args.seed),
            "dual": True,
            "class_weight": "balanced",
        },
    }
    np.save(out / "oof_probs.npy", probs.astype(np.float32))
    np.save(out / "fold_assign.npy", fold_assign)
    C.write_json(out / "summary.json", summary)
    print(
        "[variant] {variant} oof={oof:.6f} league={league_score:.6f} delta={delta:+.6f} decision={decision}".format(
            variant=spec["variant"],
            oof=oof_macro,
            league_score=summary["league_macro_f1"],
            delta=delta,
            decision=decision,
        )
    )
    return summary


def load_completed_rows(out_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(out_dir.glob("*/summary.json")):
        try:
            summary = C.read_json(path)
        except Exception:
            continue
        if "variant" not in summary:
            continue
        rows.append(
            {
                "variant": summary["variant"],
                "feature_kind": summary.get("spec", {}).get("feature_kind"),
                "ngram": "-".join(str(x) for x in summary.get("spec", {}).get("ngram", [])),
                "max_features": summary.get("spec", {}).get("max_features"),
                "C": summary.get("spec", {}).get("C"),
                "oof_macro_f1": summary.get("oof_macro_f1"),
                "league_macro_f1": summary.get("league_macro_f1"),
                "delta_vs_baseline_soft_au": summary.get("delta_vs_baseline_soft_au"),
                "decision": summary.get("decision"),
                "half1_macro_f1": summary.get("league", {}).get("half_scores", {}).get("half1_macro_f1"),
                "half2_macro_f1": summary.get("league", {}).get("half_scores", {}).get("half2_macro_f1"),
                "lin_argmax_disagreement_vs_reference": summary.get("league", {}).get(
                    "lin_argmax_disagreement_vs_reference"
                ),
            }
        )
    rows.sort(key=lambda row: float(row.get("delta_vs_baseline_soft_au") or -999), reverse=True)
    return rows


def select_variants(args: argparse.Namespace) -> list[dict[str, Any]]:
    variants = DEFAULT_VARIANTS
    if args.variant:
        wanted = set(args.variant)
        variants = [spec for spec in variants if str(spec["variant"]) in wanted]
        missing = sorted(wanted - {str(spec["variant"]) for spec in variants})
        if missing:
            raise ValueError(f"unknown variants: {missing}")
    if args.max_variants:
        variants = variants[: args.max_variants]
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=C.DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=C.OOF_DIR)
    parser.add_argument("--out-dir", type=Path, default=C.OUT_DIR)
    parser.add_argument("--context-dir", type=Path, default=C.CONTEXT_DIR)
    parser.add_argument("--variant", action="append", help="variant name from --plan; can repeat")
    parser.add_argument("--max-variants", type=int, default=0)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--word-max-features", type=int, default=80_000)
    parser.add_argument("--serializer", choices=["compact", "au_route"], default="compact")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    variants = select_variants(args)
    if args.plan:
        print(json.dumps(variants, ensure_ascii=False, indent=2))
        return

    print("[load] train and saved folds")
    samples, ids, y, groups = C.load_train(args.data_dir)
    folds = C.load_saved_folds(args.oof_dir, groups=groups)
    print(f"[serialize] {args.serializer} over train rows")
    texts = C.serialize_samples(samples, mode=args.serializer)
    completed = load_completed_rows(args.out_dir)
    C.write_progress(
        path=args.context_dir / "PROGRESS-task2.md",
        rows=completed,
        next_step="Run `C:\\dev\\2026-AI-DACON\\.venv\\Scripts\\python.exe scripts\\linear2\\sweep.py --max-variants 8`.",
        note="- Scripts scaffolded; starting/resuming sweep.",
    )

    for spec in variants:
        summary_path = variant_out_dir(args.out_dir, str(spec["variant"])) / "summary.json"
        if summary_path.exists() and not args.force:
            print(f"[variant] cache hit {spec['variant']}")
        else:
            for fold in folds:
                run_fold(spec=spec, fold=fold, texts=texts, y=y, args=args)
            assemble_variant(spec=spec, folds=folds, ids=ids, y=y, args=args)
        completed = load_completed_rows(args.out_dir)
        C.write_csv(args.out_dir / "sweep_results.csv", completed)
        C.write_json(
            args.out_dir / "sweep_summary.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "baseline_soft_au_macro_f1": C.BASELINE_SOFT_AU,
                "completed": completed,
                "completed_count": len(completed),
                "planned_order": [str(item["variant"]) for item in DEFAULT_VARIANTS],
            },
        )
        C.write_progress(
            path=args.context_dir / "PROGRESS-task2.md",
            rows=completed,
            next_step=(
                "Continue `sweep.py --max-variants 8` until at least 8 variants complete; "
                "then write `report_linear2.md` and `task2.DONE`."
            ),
        )


if __name__ == "__main__":
    main()
