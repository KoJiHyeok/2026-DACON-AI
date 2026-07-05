# -*- coding: utf-8 -*-
"""Task4 AU routing grid with a strict no-holdout-training protocol.

The script evaluates variants that only change predictions for sess_au rows.
All rows present in holdout_base.npz are excluded from model training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import joblib
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
sys.path.insert(0, str(SUBMIT_DIR))
import au_route  # noqa: E402


DATA_DIR = Path(r"C:\dev\2026-AI-DACON\data")
OOF_DIR = Path(r"C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04")
HOLDOUT_BASE = Path(r"C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz")
OUT_DIR = ROOT / "night_out" / "task4"
EXPECTED_3WAY = 0.7172592175
PASS_BASELINE = 0.7266130476
PASS_THRESHOLD = 0.002
ALPHAS = (0.5, 0.7, 0.9, 1.0)
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
        labels = {row["id"]: row["action"] for row in csv.DictReader(f)}
    ids = np.asarray([str(s["id"]) for s in samples], dtype=object)
    y = np.asarray([labels[str(s["id"])] for s in samples], dtype=object)
    groups = np.asarray([session_id(str(s["id"])) for s in samples], dtype=object)
    return samples, ids, y, groups


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_from_probs(probs: np.ndarray, actions: Sequence[str]) -> np.ndarray:
    labels = np.asarray([str(a) for a in actions], dtype=object)
    return labels[np.asarray(probs).argmax(axis=1)]


def macro_f1(pred: np.ndarray, y_true: np.ndarray) -> float:
    return float(f1_score(y_true, pred, average="macro", zero_division=0))


def macro_f1_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str]) -> float:
    return macro_f1(predict_from_probs(probs, actions), y_true)


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


def load_holdout(holdout_base: Path, oof_dir: Path) -> dict[str, Any]:
    enc = np.load(holdout_base, allow_pickle=True)
    ids = np.asarray([str(x) for x in enc["ids"]], dtype=object)
    y_true = np.asarray([str(x) for x in enc["y_true"]], dtype=object)
    actions = [str(x) for x in enc["actions"]]
    enc_probs = np.asarray(enc["probs"], dtype=np.float64)

    classes = json.loads((oof_dir / "classes.json").read_text(encoding="utf-8"))
    row_ids = json.loads((oof_dir / "row_ids.json").read_text(encoding="utf-8"))
    col = [classes.index(a) for a in actions]
    row_index = {str(row_id): i for i, row_id in enumerate(row_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in ids], dtype=np.int64)
    lin = np.load(oof_dir / "linear_probs.npy")[:, col][rows]
    stk = np.load(oof_dir / "stacker_probs.npy")[:, col][rows]
    blend = (lin + stk + 2.0 * enc_probs) / 4.0
    score = macro_f1_probs(blend, y_true, actions)
    if abs(score - EXPECTED_3WAY) > 5e-8:
        raise AssertionError(f"3-way join mismatch: {score:.10f} != {EXPECTED_3WAY:.10f}")
    return {
        "ids": ids,
        "y_true": y_true,
        "actions": actions,
        "blend": blend,
        "blend_pred": predict_from_probs(blend, actions),
        "score": score,
    }


def make_vectorizer(kind: str):
    if kind == "word_char":
        return FeatureUnion(
            [
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
            ]
        )
    if kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=1,
            max_features=120_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
    if kind == "word":
        return TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            max_features=80_000,
            sublinear_tf=True,
            strip_accents="unicode",
        )
    raise ValueError(f"unknown feature kind: {kind}")


def fit_predict(
    train_samples: Sequence[dict[str, Any]],
    train_y: np.ndarray,
    eval_samples: Sequence[dict[str, Any]],
    *,
    feature_kind: str,
    c_value: float,
    sample_weight: np.ndarray | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    vec = make_vectorizer(feature_kind)
    train_texts = [au_route.serialize(s) for s in train_samples]
    eval_texts = [au_route.serialize(s) for s in eval_samples]
    x_train = vec.fit_transform(train_texts)
    x_eval = vec.transform(eval_texts)
    if not sparse.issparse(x_train) or not sparse.issparse(x_eval):
        raise TypeError("TF-IDF output must be sparse")
    clf = LinearSVC(C=c_value, class_weight="balanced", max_iter=5000, random_state=seed)
    kwargs = {}
    if sample_weight is not None:
        kwargs["sample_weight"] = sample_weight
    clf.fit(x_train, train_y, **kwargs)
    probs = softmax(clf.decision_function(x_eval))
    probs = align_probs(probs, [str(c) for c in clf.classes_], ACTIONS)
    return {
        "probs": probs,
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x_train.shape[1]),
    }


def run_au_cv(
    train_samples: Sequence[dict[str, Any]],
    train_y: np.ndarray,
    train_groups: np.ndarray,
    *,
    feature_kind: str,
    c_value: float,
    seed: int,
    n_splits: int,
) -> dict[str, Any]:
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.zeros((len(train_samples), len(ACTIONS)), dtype=np.float64)
    folds = []
    for fold, (tr, va) in enumerate(
        splitter.split(np.zeros(len(train_y)), train_y, groups=train_groups),
        1,
    ):
        overlap = set(train_groups[tr]) & set(train_groups[va])
        if overlap:
            raise AssertionError(f"fold {fold} group leakage: {len(overlap)} groups")
        result = fit_predict(
            [train_samples[int(i)] for i in tr],
            train_y[tr],
            [train_samples[int(i)] for i in va],
            feature_kind=feature_kind,
            c_value=c_value,
            seed=seed,
        )
        oof[va] = result["probs"]
        pred = predict_from_probs(result["probs"], ACTIONS)
        folds.append(
            {
                "fold": fold,
                "rows": int(len(va)),
                "groups": int(len(set(train_groups[va]))),
                "macro_f1": macro_f1(pred, train_y[va]),
                "n_features": result["n_features"],
            }
        )
    pred = predict_from_probs(oof, ACTIONS)
    return {"macro_f1": macro_f1(pred, train_y), "folds": folds}


def evaluate_route(
    *,
    variant: str,
    au_probs: np.ndarray,
    holdout: dict[str, Any],
    au_mask: np.ndarray,
    alphas: Iterable[float],
) -> list[dict[str, Any]]:
    y_true = holdout["y_true"]
    actions = holdout["actions"]
    blend = holdout["blend"]
    blend_pred = holdout["blend_pred"]
    au_probs = align_probs(au_probs, ACTIONS, actions)
    rows = []
    for alpha in alphas:
        mixed = alpha * au_probs + (1.0 - alpha) * blend[au_mask]
        mixed_pred_au = predict_from_probs(mixed, actions)
        hybrid_pred = blend_pred.copy()
        hybrid_pred[au_mask] = mixed_pred_au
        score = macro_f1(hybrid_pred, y_true)
        au_score = macro_f1(mixed_pred_au, y_true[au_mask])
        rows.append(
            {
                "variant": variant,
                "alpha": float(alpha),
                "league_macro_f1": score,
                "delta_vs_3way": score - float(holdout["score"]),
                "delta_vs_task3_hard": score - PASS_BASELINE,
                "au_macro_f1": au_score,
                "changed_au_vs_blend": int(np.sum(mixed_pred_au != blend_pred[au_mask])),
            }
        )
    return rows


def alpha_key(alpha: float) -> str:
    return f"{float(alpha):g}"


def per_class_rows(
    *,
    name: str,
    pred_au: np.ndarray,
    blend_pred_au: np.ndarray,
    y_au: np.ndarray,
    actions: Sequence[str],
) -> list[dict[str, Any]]:
    f1_blend = f1_score(y_au, blend_pred_au, labels=list(actions), average=None, zero_division=0)
    f1_var = f1_score(y_au, pred_au, labels=list(actions), average=None, zero_division=0)
    rows = []
    for cls, old, new in zip(actions, f1_blend, f1_var):
        cls = str(cls)
        support = int(np.sum(y_au == cls))
        rows.append(
            {
                "variant": name,
                "class": cls,
                "support": support,
                "sparse_n_le_14": bool(support <= 14),
                "blend_f1": float(old),
                "variant_f1": float(new),
                "delta_f1": float(new - old),
                "blend_pred_count": int(np.sum(blend_pred_au == cls)),
                "variant_pred_count": int(np.sum(pred_au == cls)),
            }
        )
    return rows


def best_row(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return max(rows, key=lambda r: (float(r["league_macro_f1"]), float(r.get("au_macro_f1", -1.0))))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--oof-dir", type=Path, default=OOF_DIR)
    parser.add_argument("--holdout-base", type=Path, default=HOLDOUT_BASE)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument(
        "--mode",
        choices=("soft", "au-grid", "sim-weight", "all"),
        default="all",
        help="Subset of experiments to run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    t0 = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("[load] train and holdout")
    samples, ids, y, groups = load_train(args.data_dir)
    holdout = load_holdout(args.holdout_base, args.oof_dir)
    holdout_id_set = set(str(x) for x in holdout["ids"])
    sample_by_id = {str(sample["id"]): sample for sample in samples}

    au_mask = np.asarray([au_route.is_au(str(sample_id)) for sample_id in holdout["ids"]], dtype=bool)
    holdout_au_ids = [str(x) for x in holdout["ids"][au_mask]]
    holdout_au_samples = [sample_by_id[sample_id] for sample_id in holdout_au_ids]
    y_au = holdout["y_true"][au_mask]
    blend_pred_au = holdout["blend_pred"][au_mask]

    nonholdout = np.asarray([str(sample_id) not in holdout_id_set for sample_id in ids], dtype=bool)
    au_train_idx = np.asarray(
        [i for i, sample_id in enumerate(ids) if nonholdout[i] and au_route.is_au(str(sample_id))],
        dtype=np.int64,
    )
    all_train_idx = np.asarray([i for i in range(len(ids)) if nonholdout[i]], dtype=np.int64)
    if any(str(sample_id) in holdout_id_set for sample_id in ids[au_train_idx]):
        raise AssertionError("holdout id leaked into AU train")
    if any(str(sample_id) in holdout_id_set for sample_id in ids[all_train_idx]):
        raise AssertionError("holdout id leaked into all-data train")

    summary: dict[str, Any] = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "oof_dir": str(args.oof_dir),
            "holdout_base": str(args.holdout_base),
            "seed": args.seed,
            "n_splits": args.n_splits,
            "alphas": list(ALPHAS),
            "pass_baseline_task3_hard": PASS_BASELINE,
            "pass_threshold": PASS_THRESHOLD,
        },
        "split": {
            "train_rows_total": int(len(ids)),
            "holdout_rows": int(len(holdout["ids"])),
            "holdout_au_rows": int(au_mask.sum()),
            "nonholdout_rows": int(nonholdout.sum()),
            "nonholdout_au_rows": int(len(au_train_idx)),
            "nonholdout_au_sessions": int(len(set(groups[au_train_idx]))),
            "holdout_au_sessions": int(len(set(session_id(x) for x in holdout_au_ids))),
        },
        "baseline": {
            "blend_3way_macro_f1": float(holdout["score"]),
            "task3_hard_route_macro_f1": PASS_BASELINE,
            "pass_line_macro_f1": PASS_BASELINE + PASS_THRESHOLD,
            "blend_au_macro_f1": macro_f1(blend_pred_au, y_au),
        },
        "runs": [],
        "best": None,
        "per_class_best_vs_blend": [],
    }

    print(
        "[split] holdout rows={holdout_rows} au={holdout_au_rows} "
        "train_nonholdout={nonholdout_rows} au_train={nonholdout_au_rows}".format(**summary["split"])
    )
    print(f"[join] 3-way blend={holdout['score']:.10f}")

    route_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    per_class_candidates: dict[str, list[dict[str, Any]]] = {}

    def record_model(
        *,
        variant: str,
        train_samples: Sequence[dict[str, Any]],
        train_y: np.ndarray,
        eval_samples: Sequence[dict[str, Any]],
        feature_kind: str,
        c_value: float,
        sample_weight: np.ndarray | None = None,
        cv: dict[str, Any] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        started = time.time()
        result = fit_predict(
            train_samples,
            train_y,
            eval_samples,
            feature_kind=feature_kind,
            c_value=c_value,
            sample_weight=sample_weight,
            seed=args.seed,
        )
        rows = evaluate_route(
            variant=variant,
            au_probs=result["probs"],
            holdout=holdout,
            au_mask=au_mask,
            alphas=ALPHAS,
        )
        route_rows.extend(rows)
        aligned_probs = align_probs(result["probs"], ACTIONS, holdout["actions"])
        for route in rows:
            alpha = float(route["alpha"])
            mixed = alpha * aligned_probs + (1.0 - alpha) * holdout["blend"][au_mask]
            pred_au = predict_from_probs(mixed, holdout["actions"])
            key = f"{variant}|alpha={alpha_key(alpha)}"
            per_class_candidates[key] = per_class_rows(
                name=key,
                pred_au=pred_au,
                blend_pred_au=blend_pred_au,
                y_au=y_au,
                actions=holdout["actions"],
            )
        run = {
            "variant": variant,
            "feature_kind": feature_kind,
            "c": c_value,
            "train_rows": int(len(train_samples)),
            "train_au_rows": int(sum(au_route.is_au(s.get("id", "")) for s in train_samples)),
            "n_features": result["n_features"],
            "classes": result["classes"],
            "cv": cv,
            "routes": rows,
            "best_route": best_row(rows),
            "elapsed_sec": round(time.time() - started, 3),
        }
        if extra:
            run.update(extra)
        run_rows.append(run)
        print(
            "[run] {variant} best={score:.6f} alpha={alpha:g} au={au:.6f} "
            "delta_task3={delta:+.6f} elapsed={elapsed:.1f}s".format(
                variant=variant,
                score=run["best_route"]["league_macro_f1"],
                alpha=run["best_route"]["alpha"],
                au=run["best_route"]["au_macro_f1"],
                delta=run["best_route"]["delta_vs_task3_hard"],
                elapsed=run["elapsed_sec"],
            )
        )

    if args.mode in {"soft", "all"}:
        print("[axis] soft alpha grid for current AU model")
        feature_kind = "word_char"
        c_value = 0.5
        cv = run_au_cv(
            [samples[int(i)] for i in au_train_idx],
            y[au_train_idx],
            groups[au_train_idx],
            feature_kind=feature_kind,
            c_value=c_value,
            seed=args.seed,
            n_splits=args.n_splits,
        )
        record_model(
            variant="au_only_word_char_C0.5",
            train_samples=[samples[int(i)] for i in au_train_idx],
            train_y=y[au_train_idx],
            eval_samples=holdout_au_samples,
            feature_kind=feature_kind,
            c_value=c_value,
            cv=cv,
        )
        summary["runs"] = run_rows
        summary["route_rows"] = route_rows
        save_json(args.out_dir / "soft_alpha.json", summary)

    if args.mode in {"au-grid", "all"}:
        print("[axis] AU-only C/features grid")
        for feature_kind in ("word_char", "char", "word"):
            for c_value in (0.25, 0.5, 1.0):
                variant = f"au_only_{feature_kind}_C{c_value:g}"
                if any(r["variant"] == variant for r in run_rows):
                    continue
                cv = run_au_cv(
                    [samples[int(i)] for i in au_train_idx],
                    y[au_train_idx],
                    groups[au_train_idx],
                    feature_kind=feature_kind,
                    c_value=c_value,
                    seed=args.seed,
                    n_splits=args.n_splits,
                )
                record_model(
                    variant=variant,
                    train_samples=[samples[int(i)] for i in au_train_idx],
                    train_y=y[au_train_idx],
                    eval_samples=holdout_au_samples,
                    feature_kind=feature_kind,
                    c_value=c_value,
                    cv=cv,
                )
                summary["runs"] = run_rows
                summary["route_rows"] = route_rows
                save_json(args.out_dir / "au_grid_partial.json", summary)

    if args.mode in {"sim-weight", "all"}:
        print("[axis] all-nonholdout training with AU sample weights")
        all_train_samples = [samples[int(i)] for i in all_train_idx]
        all_train_y = y[all_train_idx]
        base_weight = np.ones(len(all_train_idx), dtype=np.float64)
        is_au_train = np.asarray(
            [au_route.is_au(samples[int(i)].get("id", "")) for i in all_train_idx],
            dtype=bool,
        )
        for au_weight in (1.0, 5.0, 10.0):
            weights = base_weight.copy()
            weights[is_au_train] = au_weight
            record_model(
                variant=f"all_nonholdout_word_char_C0.5_auWeight{au_weight:g}",
                train_samples=all_train_samples,
                train_y=all_train_y,
                eval_samples=holdout_au_samples,
                feature_kind="word_char",
                c_value=0.5,
                sample_weight=weights,
                cv=None,
                extra={"au_sample_weight": au_weight},
            )
            summary["runs"] = run_rows
            summary["route_rows"] = route_rows
            save_json(args.out_dir / "sim_weight_partial.json", summary)

    if route_rows:
        top = best_row(route_rows)
        pc_key = f"{top['variant']}|alpha={alpha_key(float(top['alpha']))}"
        summary["per_class_best_vs_blend"] = per_class_candidates.get(pc_key, [])
        summary["best"] = {
            **top,
            "passes_requested_gate": bool(top["delta_vs_task3_hard"] >= PASS_THRESHOLD),
            "decision_line": PASS_BASELINE + PASS_THRESHOLD,
        }

    summary["runs"] = run_rows
    summary["route_rows"] = route_rows
    summary["elapsed_sec"] = round(time.time() - t0, 3)
    save_json(args.out_dir / "summary.json", summary)
    print(
        "[done] best={variant} alpha={alpha:g} score={score:.6f} "
        "delta_task3={delta:+.6f} pass={passes}".format(
            variant=summary["best"]["variant"] if summary["best"] else "none",
            alpha=summary["best"]["alpha"] if summary["best"] else math.nan,
            score=summary["best"]["league_macro_f1"] if summary["best"] else math.nan,
            delta=summary["best"]["delta_vs_task3_hard"] if summary["best"] else math.nan,
            passes=summary["best"]["passes_requested_gate"] if summary["best"] else False,
        )
    )


if __name__ == "__main__":
    main()
