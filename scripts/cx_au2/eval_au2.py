# -*- coding: utf-8 -*-
"""Evaluate the OOF-selected AU candidate on the frozen w3/alpha=.85 surface."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score

if __package__:
    from . import common
else:
    import common


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=common.DATA_DIR)
    parser.add_argument("--holdout-npz", type=Path, default=common.HOLDOUT_NPZ)
    parser.add_argument("--oof-dir", type=Path, default=common.OOF_DIR)
    parser.add_argument("--candidate-dir", type=Path, default=common.CANDIDATE_DIR)
    parser.add_argument("--alpha", type=float, default=0.85)
    parser.add_argument("--qwen-weight", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap", type=int, default=2000)
    return parser.parse_args()


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, actions: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=list(actions), average="macro", zero_division=0))


def session_uniform_f1(
    y_true: np.ndarray, y_pred: np.ndarray, sessions: np.ndarray, actions: Sequence[str]
) -> float:
    counts: dict[str, int] = defaultdict(int)
    for session in sessions:
        counts[str(session)] += 1
    weights = np.asarray([1.0 / counts[str(session)] for session in sessions], dtype=np.float64)
    return float(
        f1_score(
            y_true,
            y_pred,
            labels=list(actions),
            average="macro",
            zero_division=0,
            sample_weight=weights,
        )
    )


def five_metric_judgment(
    y_true: np.ndarray,
    sessions: np.ndarray,
    baseline_pred: np.ndarray,
    candidate_pred: np.ndarray,
    actions: Sequence[str],
    seed: int = 42,
    bootstrap: int = 2000,
) -> dict:
    row_base = macro_f1(y_true, baseline_pred, actions)
    row_cand = macro_f1(y_true, candidate_pred, actions)
    sess_base = session_uniform_f1(y_true, baseline_pred, sessions, actions)
    sess_cand = session_uniform_f1(y_true, candidate_pred, sessions, actions)
    session_rows: dict[str, list[int]] = defaultdict(list)
    for i, session in enumerate(sessions):
        session_rows[str(session)].append(i)

    rng = np.random.default_rng(seed)
    groups = list(session_rows.values())
    mc = []
    for _ in range(200):
        idx = np.asarray([rows[int(rng.integers(len(rows)))] for rows in groups], dtype=np.int64)
        mc.append(
            macro_f1(y_true[idx], candidate_pred[idx], actions)
            - macro_f1(y_true[idx], baseline_pred[idx], actions)
        )
    mc_arr = np.asarray(mc, dtype=np.float64)

    session_names = list(session_rows)
    boot = []
    for _ in range(bootstrap):
        picks = rng.choice(len(session_names), size=len(session_names), replace=True)
        idx = np.concatenate([session_rows[session_names[int(k)]] for k in picks])
        boot.append(
            macro_f1(y_true[idx], candidate_pred[idx], actions)
            - macro_f1(y_true[idx], baseline_pred[idx], actions)
        )
    boot_arr = np.asarray(boot, dtype=np.float64)
    ci_lo, ci_hi = np.percentile(boot_arr, [2.5, 97.5])

    perm = np.random.RandomState(seed).permutation(len(y_true))
    half = len(perm) // 2
    half_deltas = []
    for idx in (perm[:half], perm[half:]):
        half_deltas.append(
            macro_f1(y_true[idx], candidate_pred[idx], actions)
            - macro_f1(y_true[idx], baseline_pred[idx], actions)
        )
    result = {
        "1_row_macro_f1": {"baseline": row_base, "candidate": row_cand, "delta": row_cand - row_base},
        "2_session_uniform_macro_f1": {
            "baseline": sess_base,
            "candidate": sess_cand,
            "delta": sess_cand - sess_base,
        },
        "3_one_row_per_session_mc200": {
            "mean_delta": float(mc_arr.mean()),
            "std_delta": float(mc_arr.std()),
            "min_delta": float(mc_arr.min()),
            "max_delta": float(mc_arr.max()),
        },
        f"4_paired_session_bootstrap{bootstrap}": {
            "ci95_low": float(ci_lo),
            "ci95_high": float(ci_hi),
            "p_delta_gt_0": float(np.mean(boot_arr > 0)),
        },
        "5_deterministic_halves": {"half1_delta": float(half_deltas[0]), "half2_delta": float(half_deltas[1])},
    }
    result["all_five_positive"] = bool(
        result["1_row_macro_f1"]["delta"] > 0
        and result["2_session_uniform_macro_f1"]["delta"] > 0
        and result["3_one_row_per_session_mc200"]["mean_delta"] > 0
        and result[f"4_paired_session_bootstrap{bootstrap}"]["ci95_low"] > 0
        and min(half_deltas) > 0
    )
    return result


def load_oof_components(oof_dir: Path, ids: Sequence[str], actions: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    src_actions = [str(x) for x in common.read_json(oof_dir / "classes.json")]
    row_ids = [str(x) for x in common.read_json(oof_dir / "row_ids.json")]
    if len(set(row_ids)) != len(row_ids):
        raise AssertionError("OOF row_ids contain duplicates")
    row_index = {sample_id: i for i, sample_id in enumerate(row_ids)}
    missing = [str(sample_id) for sample_id in ids if str(sample_id) not in row_index]
    if missing:
        raise AssertionError(f"holdout ids absent from OOF components: {missing[:3]}")
    rows = np.asarray([row_index[str(sample_id)] for sample_id in ids], dtype=np.int64)
    lin = np.asarray(np.load(oof_dir / "linear_probs.npy"), dtype=np.float64)[rows]
    stk = np.asarray(np.load(oof_dir / "stacker_probs.npy"), dtype=np.float64)[rows]
    return (
        common.align_probabilities(lin, src_actions, actions),
        common.align_probabilities(stk, src_actions, actions),
    )


def main() -> None:
    args = parse_args()
    summary_path = args.candidate_dir / "train_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"run train_au2.py first: {summary_path}")
    train_summary = common.read_json(summary_path)
    selected = common.variant_from_payload(train_summary["selected"]["variant"])

    holdout = common.load_holdout(args.holdout_npz)
    samples, ids, y, groups = common.load_train(args.data_dir)
    train_samples, train_ids, train_y, train_groups = common.select_nonholdout_au(
        samples, ids, y, groups, holdout["ids"]
    )
    sample_by_id = {str(sample["id"]): sample for sample in samples}
    holdout_au_rows = np.asarray(
        [i for i, sample_id in enumerate(holdout["ids"]) if common.is_au(str(sample_id))], dtype=np.int64
    )
    holdout_au_samples = [sample_by_id[str(holdout["ids"][i])] for i in holdout_au_rows]

    baseline_variant = next(v for v in common.VARIANTS if v.name == "baseline_char_C1")
    baseline_artifact = common.fit_artifact(train_samples, train_y, baseline_variant, args.seed)
    baseline_au = common.predict_artifact(baseline_artifact, holdout_au_samples, holdout["actions"])

    candidate_path = args.candidate_dir / "model.pkl"
    if train_summary["selected_beats_baseline"]:
        if not candidate_path.exists():
            raise FileNotFoundError(f"selected winner artifact missing: {candidate_path}")
        candidate_artifact = common.load_artifact(candidate_path)
        candidate_source = "saved OOF winner artifact"
    elif selected.name == baseline_variant.name:
        candidate_artifact = baseline_artifact
        candidate_source = "selected baseline; reused honest baseline refit"
    else:
        candidate_artifact = common.fit_artifact(train_samples, train_y, selected, args.seed)
        candidate_source = "refit selected baseline (no improved candidate persisted)"
    candidate_au = common.predict_artifact(candidate_artifact, holdout_au_samples, holdout["actions"])

    lin, stk = load_oof_components(args.oof_dir, holdout["ids"], holdout["actions"])
    qwen = holdout["probs"]
    blend = (lin + stk + args.qwen_weight * qwen) / (2.0 + args.qwen_weight)
    common.assert_probabilities("new surface", blend, len(holdout["ids"]), len(holdout["actions"]))
    baseline_final = blend.copy()
    candidate_final = blend.copy()
    baseline_final[holdout_au_rows] = args.alpha * baseline_au + (1.0 - args.alpha) * blend[holdout_au_rows]
    candidate_final[holdout_au_rows] = args.alpha * candidate_au + (1.0 - args.alpha) * blend[holdout_au_rows]
    action_array = np.asarray(holdout["actions"], dtype=object)
    baseline_pred = action_array[baseline_final.argmax(axis=1)]
    candidate_pred = action_array[candidate_final.argmax(axis=1)]
    sessions = np.asarray([common.session_id(str(x)) for x in holdout["ids"]], dtype=object)
    metrics = five_metric_judgment(
        holdout["y_true"], sessions, baseline_pred, candidate_pred, holdout["actions"], args.seed, args.bootstrap
    )

    baseline_au_pred = action_array[baseline_au.argmax(axis=1)]
    candidate_au_pred = action_array[candidate_au.argmax(axis=1)]
    au_y = holdout["y_true"][holdout_au_rows]
    payload = {
        "surface": {
            "formula": f"(linear + stacker + {args.qwen_weight:g}*qwen) / {2.0 + args.qwen_weight:g}",
            "soft_au_alpha": args.alpha,
            "baseline_au": "deployed char-C1 recipe honestly refit on nonholdout sess_au rows",
            "candidate_au": selected.name,
            "candidate_source": candidate_source,
        },
        "selection": {
            "selected_variant": selected.name,
            "oof_macro_f1": train_summary["selected"]["oof_macro_f1"],
            "oof_delta_vs_baseline": train_summary["selected_oof_delta_vs_baseline"],
            "selected_beats_baseline": train_summary["selected_beats_baseline"],
        },
        "rows": {
            "holdout": int(len(holdout["ids"])),
            "holdout_sessions": int(len(set(sessions.tolist()))),
            "holdout_au": int(len(holdout_au_rows)),
            "holdout_au_sessions": int(len(set(sessions[holdout_au_rows].tolist()))),
            "nonholdout_au_train": int(len(train_ids)),
            "nonholdout_au_sessions": int(len(set(train_groups.tolist()))),
        },
        "au_specialist_macro_f1": {
            "baseline": macro_f1(au_y, baseline_au_pred, holdout["actions"]),
            "candidate": macro_f1(au_y, candidate_au_pred, holdout["actions"]),
        },
        "five_metrics": metrics,
        "recommendation": "candidate_for_claude_review" if metrics["all_five_positive"] else "do_not_swap",
        "inputs": {
            "qwen_npz": str(args.holdout_npz),
            "qwen_npz_sha256": common.sha256(args.holdout_npz),
            "oof_dir": str(args.oof_dir),
            "candidate_model_sha256": common.sha256(candidate_path) if candidate_path.exists() else None,
        },
    }
    common.write_json(args.candidate_dir / "eval_summary.json", payload)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"[done] recommendation={payload['recommendation']}")


if __name__ == "__main__":
    main()
