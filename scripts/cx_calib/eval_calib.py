"""Evaluate fold-honest Qwen calibration on the current 3x-Qwen surface."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

try:
    from .fit_calib import (
        DEFAULT_COMMON,
        DEFAULT_QWEN,
        apply_calibration,
        build_group_folds,
        load_calibration_json,
        load_external_common,
        sessions_sha256,
        sha256_file,
        session_id,
    )
except ImportError:  # direct script execution
    from fit_calib import (  # type: ignore
        DEFAULT_COMMON,
        DEFAULT_QWEN,
        apply_calibration,
        build_group_folds,
        load_calibration_json,
        load_external_common,
        sessions_sha256,
        sha256_file,
        session_id,
    )


DEFAULT_EXTERNAL_ROOT = Path(r"C:\dev\2026-AI-DACON")
DEFAULT_AU_PROBS = DEFAULT_EXTERNAL_ROOT / "night_out" / "league4" / "au_charwb_C1_holdout_probs.npz"
DEFAULT_SUMMARY = Path(__file__).with_name("fit_summary.json")
DEFAULT_CANDIDATE = Path(__file__).with_name("calib_candidate.json")
DEFAULT_OUTPUT = Path(__file__).with_name("eval_results.json")
ALPHA = 0.85
QWEN_WEIGHT = 3.0
GATE_DELTA = 0.005
REPORT_DELTA = 0.002


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, actions: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=[str(a) for a in actions], average="macro", zero_division=0))


def _confusion(y_idx: np.ndarray, pred_idx: np.ndarray, n_classes: int, weights=None) -> np.ndarray:
    flat = y_idx * n_classes + pred_idx
    return np.bincount(flat, weights=weights, minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def _macro_f1_confusion(confusion: np.ndarray) -> float:
    true_positive = np.diag(confusion)
    denominator = confusion.sum(axis=1) + confusion.sum(axis=0)
    per_class = np.divide(
        2.0 * true_positive,
        denominator,
        out=np.zeros_like(true_positive, dtype=np.float64),
        where=denominator != 0,
    )
    return float(per_class.mean())


def session_uniform_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sessions: np.ndarray,
    actions: Sequence[str],
) -> float:
    counts: dict[str, int] = defaultdict(int)
    for group in sessions:
        counts[str(group)] += 1
    weights = np.asarray([1.0 / counts[str(group)] for group in sessions], dtype=np.float64)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=[str(a) for a in actions],
            average="macro",
            zero_division=0,
            sample_weight=weights,
        )
    )


def five_metric_judgment(
    y_true: np.ndarray,
    sessions: np.ndarray,
    pred_base: np.ndarray,
    pred_candidate: np.ndarray,
    actions: Sequence[str],
    seed: int = 42,
    mc_repeats: int = 200,
    bootstrap_repeats: int = 1000,
) -> dict:
    action_index = {str(action): i for i, action in enumerate(actions)}
    y_idx = np.asarray([action_index[str(value)] for value in y_true], dtype=np.int64)
    base_idx = np.asarray([action_index[str(value)] for value in pred_base], dtype=np.int64)
    candidate_idx = np.asarray([action_index[str(value)] for value in pred_candidate], dtype=np.int64)
    n_classes = len(actions)
    base_row = _macro_f1_confusion(_confusion(y_idx, base_idx, n_classes))
    candidate_row = _macro_f1_confusion(_confusion(y_idx, candidate_idx, n_classes))

    session_counts: dict[str, int] = defaultdict(int)
    for group in sessions:
        session_counts[str(group)] += 1
    session_weights = np.asarray([1.0 / session_counts[str(group)] for group in sessions], dtype=np.float64)
    base_session = _macro_f1_confusion(_confusion(y_idx, base_idx, n_classes, weights=session_weights))
    candidate_session = _macro_f1_confusion(_confusion(y_idx, candidate_idx, n_classes, weights=session_weights))

    rows_by_session: dict[str, list[int]] = defaultdict(list)
    for row, group in enumerate(sessions):
        rows_by_session[str(group)].append(row)
    groups = list(rows_by_session.values())
    rng = np.random.default_rng(seed)
    mc_delta = []
    for _ in range(mc_repeats):
        idx = np.asarray([rows[int(rng.integers(len(rows)))] for rows in groups], dtype=np.int64)
        mc_delta.append(
            _macro_f1_confusion(_confusion(y_idx[idx], candidate_idx[idx], n_classes))
            - _macro_f1_confusion(_confusion(y_idx[idx], base_idx[idx], n_classes))
        )
    mc = np.asarray(mc_delta, dtype=np.float64)

    unique_sessions = list(rows_by_session)
    base_session_confusion = np.stack(
        [_confusion(y_idx[rows_by_session[group]], base_idx[rows_by_session[group]], n_classes) for group in unique_sessions]
    )
    candidate_session_confusion = np.stack(
        [
            _confusion(y_idx[rows_by_session[group]], candidate_idx[rows_by_session[group]], n_classes)
            for group in unique_sessions
        ]
    )
    bootstrap_delta = []
    for _ in range(bootstrap_repeats):
        picked = rng.choice(len(unique_sessions), size=len(unique_sessions), replace=True)
        bootstrap_delta.append(
            _macro_f1_confusion(candidate_session_confusion[picked].sum(axis=0))
            - _macro_f1_confusion(base_session_confusion[picked].sum(axis=0))
        )
    bootstrap = np.asarray(bootstrap_delta, dtype=np.float64)
    ci_lo, ci_hi = np.percentile(bootstrap, [2.5, 97.5])

    permutation = np.random.RandomState(seed).permutation(len(y_true))
    midpoint = len(permutation) // 2
    half1, half2 = permutation[:midpoint], permutation[midpoint:]

    return {
        "1_row_macro_f1": {
            "baseline": base_row,
            "candidate": candidate_row,
            "delta": candidate_row - base_row,
        },
        "2_session_uniform_macro_f1": {
            "baseline": base_session,
            "candidate": candidate_session,
            "delta": candidate_session - base_session,
        },
        "3_mc200_delta": {
            "mean": float(mc.mean()),
            "std": float(mc.std()),
            "min": float(mc.min()),
            "max": float(mc.max()),
            "repeats": mc_repeats,
        },
        "4_paired_session_bootstrap": {
            "ci_lo": float(ci_lo),
            "ci_hi": float(ci_hi),
            "p_delta_gt_0": float((bootstrap > 0).mean()),
            "repeats": bootstrap_repeats,
        },
        "5_half_split": {
            "half1_delta": _macro_f1_confusion(_confusion(y_idx[half1], candidate_idx[half1], n_classes))
            - _macro_f1_confusion(_confusion(y_idx[half1], base_idx[half1], n_classes)),
            "half2_delta": _macro_f1_confusion(_confusion(y_idx[half2], candidate_idx[half2], n_classes))
            - _macro_f1_confusion(_confusion(y_idx[half2], base_idx[half2], n_classes)),
        },
    }


def load_au_probs(path: Path, data) -> np.ndarray:
    archive = np.load(path, allow_pickle=True)
    source_ids = [str(x) for x in archive["ids"]]
    source_actions = [str(x) for x in archive["actions"]]
    source_probs = np.asarray(archive["probs"], dtype=np.float64)
    target_ids = [str(x) for x in data.ids[data.au_mask]]
    row_index = {sample_id: i for i, sample_id in enumerate(source_ids)}
    missing = [sample_id for sample_id in target_ids if sample_id not in row_index]
    if missing:
        raise AssertionError(f"AU probability cache is missing {len(missing)} holdout rows")
    columns = [source_actions.index(str(action)) for action in data.actions]
    rows = np.asarray([row_index[sample_id] for sample_id in target_ids], dtype=np.int64)
    aligned = source_probs[rows][:, columns]
    if aligned.shape != (int(data.au_mask.sum()), len(data.actions)):
        raise AssertionError("aligned AU probability shape mismatch")
    return aligned


def reconstruct_oof_calibration(
    probs: np.ndarray,
    ids: Sequence[str],
    actions: Sequence[str],
    summary: dict,
) -> tuple[np.ndarray, list[dict]]:
    if [str(a) for a in summary["actions"]] != [str(a) for a in actions]:
        raise AssertionError("fit summary action order differs from league action order")
    folds = build_group_folds(ids, int(summary["n_splits"]))
    if len(folds) != len(summary["folds"]):
        raise AssertionError("fit summary fold count mismatch")
    groups = np.asarray([session_id(str(x)) for x in ids], dtype=object)
    out = np.empty_like(probs, dtype=np.float64)
    audit = []
    for fold_number, (_, valid_idx) in enumerate(folds):
        fold = summary["folds"][fold_number]
        if int(fold["fold"]) != fold_number:
            raise AssertionError("fit summary folds are out of order")
        actual_hash = sessions_sha256(groups[valid_idx])
        if actual_hash != fold["validation_sessions_sha256"]:
            raise AssertionError(f"fold {fold_number} validation-session hash mismatch")
        calibration_bias = np.asarray([float(fold["class_bias"][str(a)]) for a in actions], dtype=np.float64)
        out[valid_idx] = apply_calibration(probs[valid_idx], float(fold["temperature"]), calibration_bias)
        audit.append({"fold": fold_number, "n_valid_rows": int(len(valid_idx)), "validation_sessions_sha256": actual_hash})
    return out, audit


def recommendation(metrics: dict) -> dict[str, str]:
    row_delta = metrics["1_row_macro_f1"]["delta"]
    mc_mean = metrics["3_mc200_delta"]["mean"]
    ci_lo = metrics["4_paired_session_bootstrap"]["ci_lo"]
    if row_delta >= GATE_DELTA and mc_mean > 0 and ci_lo > 0:
        return {"decision": "ADOPT", "reason": "row delta >= +0.005, MC mean > 0, and bootstrap CI lower bound > 0"}
    if row_delta >= REPORT_DELTA:
        return {"decision": "REPORT_ONLY", "reason": "row delta is +0.002 or higher but the strict LB gate is incomplete"}
    return {"decision": "REJECT", "reason": "fold-honest row delta is below +0.002"}


def run(args: argparse.Namespace) -> dict:
    common = load_external_common(args.common)
    data = common.load_league_data()
    qwen = common.align_npz_probs(args.qwen, data.ids, data.y_true, data.actions)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    if sha256_file(args.qwen) != summary["inputs"]["qwen_sha256"]:
        raise AssertionError("Qwen NPZ hash differs from the fit input")
    calibrated_qwen, fold_audit = reconstruct_oof_calibration(qwen, data.ids, data.actions, summary)

    full_candidate = load_calibration_json(args.candidate, data.actions)
    full_summary = summary["full_refit"]
    if not np.isclose(full_candidate.temperature, float(full_summary["temperature"]), atol=1e-12):
        raise AssertionError("candidate JSON temperature differs from fit summary")
    summary_bias = np.asarray([float(full_summary["class_bias"][str(a)]) for a in data.actions])
    if not np.allclose(full_candidate.class_bias, summary_bias, atol=1e-12):
        raise AssertionError("candidate JSON class bias differs from fit summary")

    baseline_blend = (data.lin + data.stk + QWEN_WEIGHT * qwen) / (2.0 + QWEN_WEIGHT)
    candidate_blend = (data.lin + data.stk + QWEN_WEIGHT * calibrated_qwen) / (2.0 + QWEN_WEIGHT)
    au_probs = load_au_probs(args.au_probs, data)
    baseline_final = common.apply_soft_au(data, baseline_blend, au_probs, alpha=args.alpha)
    candidate_final = common.apply_soft_au(data, candidate_blend, au_probs, alpha=args.alpha)
    action_array = np.asarray(data.actions, dtype=object)
    pred_base = action_array[baseline_final.argmax(axis=1)]
    pred_candidate = action_array[candidate_final.argmax(axis=1)]
    sessions = np.asarray([session_id(str(x)) for x in data.ids], dtype=object)
    metrics = five_metric_judgment(
        data.y_true,
        sessions,
        pred_base,
        pred_candidate,
        data.actions,
        seed=args.seed,
        mc_repeats=args.mc_repeats,
        bootstrap_repeats=args.bootstrap_repeats,
    )

    fold_surface = []
    for fold_number, (_, valid_idx) in enumerate(build_group_folds(data.ids, summary["n_splits"])):
        base_score = macro_f1(data.y_true[valid_idx], pred_base[valid_idx], data.actions)
        candidate_score = macro_f1(data.y_true[valid_idx], pred_candidate[valid_idx], data.actions)
        fold_surface.append(
            {"fold": fold_number, "baseline": base_score, "candidate": candidate_score, "delta": candidate_score - base_score}
        )

    result = {
        "schema_version": 1,
        "evaluation_contract": "fold-honest OOF Qwen calibration; full refit excluded from all recommendation metrics",
        "baseline_recipe": f"(linear + stacker + {QWEN_WEIGHT:g}*qwen)/5 then soft-AU alpha={args.alpha:g}",
        "candidate_recipe": "same surface with each row's Qwen probabilities calibrated by its held-out fold parameters",
        "seed": args.seed,
        "inputs": {
            "qwen_npz": str(args.qwen),
            "qwen_sha256": sha256_file(args.qwen),
            "fit_summary": str(args.summary),
            "fit_summary_sha256": sha256_file(args.summary),
            "candidate_json": str(args.candidate),
            "candidate_json_sha256": sha256_file(args.candidate),
            "au_probs": str(args.au_probs),
            "au_probs_sha256": sha256_file(args.au_probs),
        },
        "fold_audit": fold_audit,
        "fold_surface_scores": fold_surface,
        "metrics": metrics,
        "recommendation": recommendation(metrics),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    row = metrics["1_row_macro_f1"]
    session = metrics["2_session_uniform_macro_f1"]
    mc = metrics["3_mc200_delta"]
    boot = metrics["4_paired_session_bootstrap"]
    half = metrics["5_half_split"]
    print(f"row macro-F1      {row['baseline']:.6f} -> {row['candidate']:.6f} ({row['delta']:+.6f})")
    print(f"session-uniform   {session['baseline']:.6f} -> {session['candidate']:.6f} ({session['delta']:+.6f})")
    print(f"MC{args.mc_repeats} delta      {mc['mean']:+.6f} +/- {mc['std']:.6f}")
    print(f"bootstrap 95% CI  [{boot['ci_lo']:+.6f}, {boot['ci_hi']:+.6f}], P(delta>0)={boot['p_delta_gt_0']:.3f}")
    print(f"half split        {half['half1_delta']:+.6f} / {half['half2_delta']:+.6f}")
    print(f"decision          {result['recommendation']['decision']}: {result['recommendation']['reason']}")
    print(f"wrote {args.output}")
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, default=DEFAULT_QWEN)
    parser.add_argument("--common", type=Path, default=DEFAULT_COMMON)
    parser.add_argument("--au-probs", type=Path, default=DEFAULT_AU_PROBS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mc-repeats", type=int, default=200)
    parser.add_argument("--bootstrap-repeats", type=int, default=1000)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())
