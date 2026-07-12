# -*- coding: utf-8 -*-
"""D-014 Lane A phase 2 — frozen-shadow 5-component parity stacker: judgment.

Pipeline:
  1. Build meta-training features (fold_map rows minus holdout ids, ~59.5k) via
     shadow_features.build_meta_train_features. Assert zero id overlap with holdout.
  2. Diagnostic meta-CV: 5-fold cross-fit by fold_map fold within meta-train rows.
  3. Fit final meta model on all meta-train rows (MaxAbsScaler + LogisticRegression).
  4. Build holdout features from the 85%-trained surfaces via
     shadow_features.build_holdout_features, predict, and run the 5-metric judgment
     against the champion baseline (scripts/league4/common.py + probe_c_args_lite.py
     recipe: e5 slot swapped to hist12 npz, mbert stays hist6 default).
  5. Write artifacts/experiments/shadow_stack/verdict.json with promotion_eligible=false.
  6. Re-run the whole eval a second time and hash-compare both verdicts (minus
     timestamps) to prove determinism.

No promotion decision is made here — that is the orchestrator's call.
"""
from __future__ import annotations

import hashlib
import importlib.metadata as _ilm
import importlib.util as _ilu
import json
import platform
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import MaxAbsScaler

import common
import shadow_features as sf


def _load_league_common():
    """Import scripts/league4/common.py under a distinct module name.

    Both scripts/oof5/common.py and scripts/league4/common.py are named
    `common`; a plain `import common` after sys.path insertion would just
    return the already-cached oof5 module. Load league4's by explicit path.
    """
    path = Path(r"C:\dev\2026-AI-DACON\scripts\league4\common.py")
    spec = _ilu.spec_from_file_location("league4_common", path)
    module = _ilu.module_from_spec(spec)
    sys.modules["league4_common"] = module
    spec.loader.exec_module(module)
    return module


league_common = _load_league_common()

ROOT = common.ROOT
OUT_DIR = ROOT / "artifacts" / "experiments" / "shadow_stack"
ACTIONS = common.ACTIONS
CX002_WATCH_CLASSES = ["read_file", "grep_search", "ask_user"]


def session_id(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0))


def per_class_f1(y_true, y_pred) -> dict:
    scores = f1_score(y_true, y_pred, labels=ACTIONS, average=None, zero_division=0)
    return {a: float(s) for a, s in zip(ACTIONS, scores)}


def session_uniform_f1(y_true, y_pred, sess) -> float:
    cnt = defaultdict(int)
    for s in sess:
        cnt[s] += 1
    w = np.array([1.0 / cnt[s] for s in sess])
    return float(f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0, sample_weight=w))


def fit_meta_model(X: np.ndarray, y: np.ndarray) -> tuple[MaxAbsScaler, LogisticRegression]:
    scaler = MaxAbsScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
    clf.fit(Xs, y)
    return scaler, clf


def diagnostic_meta_cv(X: np.ndarray, y: np.ndarray, ids: np.ndarray, fold_map: dict) -> dict:
    folds = np.asarray([fold_map[str(i)] for i in ids], dtype=np.int64)
    uniq_folds = sorted(set(folds.tolist()))
    per_fold = {}
    all_true, all_pred = [], []
    for f in uniq_folds:
        va = folds == f
        tr = ~va
        scaler, clf = fit_meta_model(X[tr], y[tr])
        pred = clf.predict(scaler.transform(X[va]))
        score = macro_f1(y[va], pred)
        per_fold[str(f)] = {"n_rows": int(va.sum()), "macro_f1": score}
        all_true.append(y[va])
        all_pred.append(pred)
    pooled_true = np.concatenate(all_true)
    pooled_pred = np.concatenate(all_pred)
    pooled = macro_f1(pooled_true, pooled_pred)
    return {"per_fold": per_fold, "pooled_macro_f1": pooled, "n_folds": len(uniq_folds)}


def champion_baseline(data: league_common.LeagueData, au: dict) -> tuple[np.ndarray, float]:
    """Reproduce the deployed champion holdout blend (0.75601 recipe):
    four_way_blend + soft-AU, with e5 swapped to the hist12 npz (mbert stays
    at common.py's default hist6 npz — mBERT hist12 was rejected, exp #42).
    This matches scripts/league4/probe_c_args_lite.py's baseline construction.
    """
    e5_h12 = league_common.align_npz_probs(
        Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz"),
        data.ids, data.y_true, data.actions,
    )
    d = replace(data, e5=np.asarray(e5_h12, dtype=np.float64))
    blend = league_common.four_way_blend(d)
    final = league_common.apply_soft_au(d, blend, au["probs"], league_common.DEFAULT_ALPHA)
    score = league_common.macro_f1_probs(final, d.y_true, d.actions)
    pred = league_common.predict_from_probs(final, d.actions)
    return pred, score


def five_metric_judgment(y_true, sess, p_base, p_cand, seed: int = 42) -> dict:
    out = {}
    f_base, f_cand = macro_f1(y_true, p_base), macro_f1(y_true, p_cand)
    out["1_row_macro_f1"] = {"baseline": f_base, "candidate": f_cand, "delta": f_cand - f_base}

    su_base = session_uniform_f1(y_true, p_base, sess)
    su_cand = session_uniform_f1(y_true, p_cand, sess)
    out["2_session_uniform_macro_f1"] = {"baseline": su_base, "candidate": su_cand, "delta": su_cand - su_base}

    rng = np.random.default_rng(seed)
    sess_rows = defaultdict(list)
    for i, s in enumerate(sess):
        sess_rows[s].append(i)
    groups = list(sess_rows.values())
    mc = []
    for _ in range(200):
        idx = np.array([g[rng.integers(len(g))] for g in groups])
        mc.append(macro_f1(y_true[idx], p_cand[idx]) - macro_f1(y_true[idx], p_base[idx]))
    mc = np.array(mc)
    out["3_mc200_delta"] = {"mean": float(mc.mean()), "std": float(mc.std()), "min": float(mc.min()), "max": float(mc.max())}

    uniq = list(sess_rows.keys())
    boot = []
    for _ in range(1000):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([sess_rows[uniq[k]] for k in pick])
        boot.append(macro_f1(y_true[idx], p_cand[idx]) - macro_f1(y_true[idx], p_base[idx]))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    out["4_paired_bootstrap_1000"] = {
        "ci_lo": float(lo), "ci_hi": float(hi), "p_delta_gt_0": float((boot > 0).mean()),
    }

    perm = np.random.RandomState(seed).permutation(len(y_true))
    half = len(perm) // 2
    h1, h2 = perm[:half], perm[half:]
    d1 = macro_f1(y_true[h1], p_cand[h1]) - macro_f1(y_true[h1], p_base[h1])
    d2 = macro_f1(y_true[h2], p_cand[h2]) - macro_f1(y_true[h2], p_base[h2])
    out["5_half_split"] = {"half1_delta": d1, "half2_delta": d2}

    return out


def package_versions() -> dict:
    versions = {"python": platform.python_version()}
    for pkg in ("numpy", "scikit-learn", "scipy"):
        try:
            versions[pkg] = _ilm.version(pkg)
        except _ilm.PackageNotFoundError:
            versions[pkg] = "not-installed"
    return versions


def run_once() -> dict:
    fold_map = common.load_fold_map()
    hb = np.load(sf.HOLDOUT_BASE_NPZ, allow_pickle=True)
    holdout_ids = np.asarray([str(x) for x in hb["ids"]], dtype=object)
    holdout_id_set = set(holdout_ids.tolist())

    all_ids = list(fold_map.keys())
    meta_train_ids = [i for i in all_ids if i not in holdout_id_set]
    overlap = set(meta_train_ids) & holdout_id_set
    if overlap:
        raise AssertionError(f"leak: {len(overlap)} meta-train ids overlap holdout")

    X_train, y_train, train_meta = sf.build_meta_train_features(meta_train_ids)
    X_hold, y_hold, hold_meta = sf.build_holdout_features(holdout_ids)

    if not np.array_equal(y_hold, np.asarray([str(x) for x in hb["y_true"]], dtype=object)):
        raise AssertionError("holdout y_true built from feature sources does not match holdout_base.npz y_true")

    diag = diagnostic_meta_cv(X_train, y_train, np.asarray(meta_train_ids, dtype=object), fold_map)

    scaler, clf = fit_meta_model(X_train, y_train)
    stacker_pred = clf.predict(scaler.transform(X_hold))
    stacker_pred = np.asarray(stacker_pred, dtype=object)

    # determinism proof requires re-fit to be bit-identical too — fit twice here.
    scaler2, clf2 = fit_meta_model(X_train, y_train)
    stacker_pred2 = np.asarray(clf2.predict(scaler2.transform(X_hold)), dtype=object)
    refit_identical = bool(np.array_equal(stacker_pred, stacker_pred2))

    data = league_common.load_league_data()
    au = league_common.train_or_load_au_probs(data)
    if not au.get("cache_hit"):
        raise AssertionError("AU probs cache miss — expected cache_hit=True per spec")
    baseline_pred, baseline_score = champion_baseline(data, au)

    if not np.array_equal(np.asarray(data.ids, dtype=object), holdout_ids):
        raise AssertionError("league4 holdout id order does not match holdout_base.npz order used for stacker features")

    sess = np.array([session_id(str(i)) for i in holdout_ids])
    judgment = five_metric_judgment(y_hold, sess, baseline_pred, stacker_pred)

    base_per_class = per_class_f1(y_hold, baseline_pred)
    cand_per_class = per_class_f1(y_hold, stacker_pred)
    per_class_delta = {a: cand_per_class[a] - base_per_class[a] for a in ACTIONS}
    cx002_regressed = {a: per_class_delta[a] for a in CX002_WATCH_CLASSES}

    hashes = {
        "meta_train_source_hashes": train_meta["source_hashes"],
        "holdout_source_hashes": hold_meta["source_hashes"],
        "fold_map_sha256": common.FOLD_MAP_SHA256,
    }

    verdict = {
        "task": "D-014 Lane A frozen-shadow 5-component parity stacker",
        "promotion_eligible": False,
        "n_rows": {
            "meta_train": int(len(meta_train_ids)),
            "meta_train_au_scope": train_meta["n_au_scope_rows"],
            "holdout": int(len(holdout_ids)),
            "holdout_au_scope": hold_meta["n_au_scope_rows"],
        },
        "leak_check": {"meta_train_holdout_overlap": len(overlap)},
        "diagnostic_meta_cv": diag,
        "baseline": {
            "recipe": "four_way_blend(e5=hist12 npz, mbert=hist6 default) + soft_au(alpha=0.9), per scripts/league4/probe_c_args_lite.py",
            "macro_f1": baseline_score,
        },
        "five_metric_judgment": judgment,
        "per_class_f1_delta": per_class_delta,
        "cx002_watch_classes_delta": cx002_regressed,
        "determinism": {"refit_prediction_identical": refit_identical},
        "input_hashes": hashes,
        "package_versions": package_versions(),
        "deviation_notes": [
            hold_meta["deviation_note"],
        ],
        "meta_model": {
            "scaler": "MaxAbsScaler",
            "model": "LogisticRegression(C=1.0, max_iter=3000, random_state=42)",
            "n_features": 76,
        },
    }
    return verdict


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    v1 = run_once()
    v1["_run_seconds"] = round(time.time() - t0, 2)
    v1["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    t0 = time.time()
    v2 = run_once()
    v2["_run_seconds"] = round(time.time() - t0, 2)
    v2["_generated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    def strip_ts(v: dict) -> dict:
        return {k: val for k, val in v.items() if k not in ("_run_seconds", "_generated_at")}

    h1 = hashlib.sha256(json.dumps(strip_ts(v1), sort_keys=True).encode("utf-8")).hexdigest()
    h2 = hashlib.sha256(json.dumps(strip_ts(v2), sort_keys=True).encode("utf-8")).hexdigest()
    determinism_ok = h1 == h2

    verdict = v1
    verdict["determinism"]["two_run_verdict_hash_1"] = h1
    verdict["determinism"]["two_run_verdict_hash_2"] = h2
    verdict["determinism"]["two_run_byte_identical_minus_timestamps"] = determinism_ok

    out_path = OUT_DIR / "verdict.json"
    out_path.write_text(json.dumps(verdict, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[save] {out_path}")

    print("=" * 70)
    print(f"baseline macro_f1     : {verdict['baseline']['macro_f1']:.5f}")
    print(f"1 row delta           : {verdict['five_metric_judgment']['1_row_macro_f1']['delta']:+.5f}")
    print(f"2 session-uniform delta: {verdict['five_metric_judgment']['2_session_uniform_macro_f1']['delta']:+.5f}")
    mc = verdict["five_metric_judgment"]["3_mc200_delta"]
    print(f"3 MC200 delta         : {mc['mean']:+.5f} +/- {mc['std']:.5f}")
    boot = verdict["five_metric_judgment"]["4_paired_bootstrap_1000"]
    print(f"4 bootstrap 95% CI    : [{boot['ci_lo']:+.5f}, {boot['ci_hi']:+.5f}]  P(delta>0)={boot['p_delta_gt_0']:.3f}")
    half = verdict["five_metric_judgment"]["5_half_split"]
    print(f"5 half-split          : half1 {half['half1_delta']:+.5f} / half2 {half['half2_delta']:+.5f}")
    print(f"diagnostic meta-CV pooled: {verdict['diagnostic_meta_cv']['pooled_macro_f1']:.5f}")
    print(f"determinism (two runs identical): {determinism_ok}")
    print("promotion_eligible: False (per spec — orchestrator decides)")
    print("[DONE]")


if __name__ == "__main__":
    main()
