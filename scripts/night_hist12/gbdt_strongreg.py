# -*- coding: utf-8 -*-
"""GBDT 메타 스태커 프로토타입 (config 'strongreg') — exp night_hist12 / A.

가설: hist12 블렌드의 선형결합(4-way+soft-AU) 천장을 sklearn
HistGradientBoostingClassifier 메타러너로 넘을 수 있는가.

데이터: common.load_league_data() -> hist12 e5 스왑 (colab_out/holdout_e5_h12.npz).
입력 피처: 4성분(linear/stacker/e5-h12/mbert) OOF확률(14*4=56열) + 성분별
  max/top2margin/entropy(4*3=12열) + 구조피처 7열
  (turn_index, budget_bucket, loc_bucket, git_dirty, n_open_files, hist_len, au_flag).
모델: HistGradientBoostingClassifier(max_leaf_nodes=15, min_samples_leaf=200,
  l2_regularization=1.0, learning_rate=0.05, max_iter=300, early_stopping=True),
  sample_weight=balanced (클래스 빈도 역가중).
검증: nested StratifiedGroupKFold(n_splits=5, group=session_id) — fold별로 메타러너를
  학습셋에서 학습, valid 행만 예측 -> OOF 메타확률(같은 행 학습/평가 금지).
  AU 라우팅(soft-AU)은 메타확률에도 동일 적용 후 최종 4-way 대체 macro-F1 비교.

판정: 메타 OOF 4-way대체 macro-F1 vs hist12 baseline(자기 계산, 4-way+softAU).
  게이트 +0.005 이상 승격후보, +0.002~0.005 보고, 미만 폐기. half1/half2 반반 안정성도 기록.

출력: night_out/night_hist12/gbdt_strongreg.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "league4"))
import common  # noqa: E402

ROOT = Path(r"C:\dev\2026-AI-DACON")
E5_H12 = ROOT / "colab_out" / "holdout_e5_h12.npz"
OUT_DIR = ROOT / "night_out" / "night_hist12"
OUT_JSON = OUT_DIR / "gbdt_strongreg.json"
LEAGUE_OUT = ROOT / "night_out" / "league4"

GATE = 0.005
REPORT = 0.002
N_SPLITS = 5
SEED = 42


def entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top2_margin(p: np.ndarray) -> np.ndarray:
    sorted_p = np.sort(p, axis=1)
    return sorted_p[:, -1] - sorted_p[:, -2]


def component_stats(p: np.ndarray) -> np.ndarray:
    return np.stack([p.max(axis=1), top2_margin(p), entropy(p)], axis=1)


def bucketize(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.digitize(values, edges).astype(np.float64)


def build_structural_features(data: common.LeagueData) -> np.ndarray:
    n = len(data.ids)
    turn_index = np.zeros(n)
    budget_bucket = np.zeros(n)
    loc_bucket = np.zeros(n)
    git_dirty = np.zeros(n)
    n_open_files = np.zeros(n)
    hist_len = np.zeros(n)
    au_flag = data.au_mask.astype(np.float64)

    budget_edges = [1000, 5000, 20000, 60000, 120000]
    loc_edges = [500, 2000, 8000, 20000, 50000]

    for i, sample_id in enumerate(data.ids):
        sample = data.samples_by_id[str(sample_id)]
        meta = sample.get("session_meta") or {}
        workspace = meta.get("workspace") or {}
        turn_index[i] = float(meta.get("turn_index") or 0)
        budget = float(meta.get("budget_tokens_remaining") or 0)
        loc = float(workspace.get("loc") or 0)
        git_dirty[i] = 1.0 if workspace.get("git_dirty") else 0.0
        n_open_files[i] = float(len(workspace.get("open_files") or []))
        hist_len[i] = float(len(sample.get("history") or []))
        budget_bucket[i] = budget
        loc_bucket[i] = loc

    budget_bucket = bucketize(budget_bucket, budget_edges)
    loc_bucket = bucketize(loc_bucket, loc_edges)

    return np.stack(
        [turn_index, budget_bucket, loc_bucket, git_dirty, n_open_files, hist_len, au_flag],
        axis=1,
    )


def build_feature_matrix(data: common.LeagueData) -> tuple[np.ndarray, list[str]]:
    comps = [("linear", data.lin), ("stacker", data.stk), ("e5_h12", data.e5), ("mbert", data.mbert)]
    blocks = []
    names: list[str] = []
    for cname, probs in comps:
        blocks.append(probs)
        names.extend([f"{cname}__{a}" for a in data.actions])
    for cname, probs in comps:
        stats = component_stats(probs)
        blocks.append(stats)
        names.extend([f"{cname}__max", f"{cname}__top2margin", f"{cname}__entropy"])
    struct = build_structural_features(data)
    blocks.append(struct)
    names.extend(
        ["turn_index", "budget_bucket", "loc_bucket", "git_dirty", "n_open_files", "hist_len", "au_flag"]
    )
    X = np.concatenate(blocks, axis=1).astype(np.float64)
    return X, names


def make_model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_leaf_nodes=15,
        min_samples_leaf=200,
        l2_regularization=1.0,
        learning_rate=0.05,
        max_iter=300,
        early_stopping=True,
        random_state=SEED,
    )


def nested_oof_meta_probs(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray, classes: list[str]
) -> np.ndarray:
    n = len(y)
    n_classes = len(classes)
    oof = np.zeros((n, n_classes), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)

    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y, groups)):
        assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx])), "session leakage across fold"
        sw = compute_sample_weight(class_weight="balanced", y=y[train_idx])
        model = make_model()
        model.fit(X[train_idx], y[train_idx], sample_weight=sw)
        proba = model.predict_proba(X[valid_idx])
        model_classes = [str(c) for c in model.classes_]
        aligned = common.align_probs(proba, model_classes, classes)
        oof[valid_idx] = aligned
        filled[valid_idx] = True
        print(f"  fold {fold_i}: train={len(train_idx)} valid={len(valid_idx)}")

    if not filled.all():
        raise AssertionError(f"{(~filled).sum()} rows never appeared in a validation fold")
    row_sum = oof.sum(axis=1)
    if not np.allclose(row_sum, 1.0, atol=1e-5):
        raise AssertionError("meta OOF probs do not sum to 1")
    return oof


def main() -> None:
    started = time.time()
    print("=" * 60)
    print("[load] league data (hist12 e5 swap)")
    data = common.load_league_data()
    h12 = common.align_npz_probs(E5_H12, data.ids, data.y_true, data.actions)
    data = replace(data, e5=h12)
    common.assert_row_probs("e5(h12 swap)", data.e5, len(data.ids), len(data.actions))

    # The 9969-row holdout matrix from load_league_data() is already honest out-of-sample
    # (lin/stk are OOF, e5/mbert are held-out-only probs) -- no additional holdout exclusion
    # is needed here (leak_excluded_holdout=true). The only leakage risk in THIS script is the
    # meta-learner itself, which is guarded by the nested StratifiedGroupKFold below (fold-disjoint
    # session groups, asserted per-fold).
    au = common.train_or_load_au_probs(data, LEAGUE_OUT, force=False)

    print("[baseline] hist12 4-way+soft-AU (self-computed)")
    base_blend = common.four_way_blend(data)
    base_final = common.apply_soft_au(data, base_blend, au["probs"], common.DEFAULT_ALPHA)
    base_bundle = common.score_bundle(data, base_final, prefix="baseline_")
    base_half = common.half_scores(data, base_final)
    print(f"  baseline macro_f1={base_bundle['baseline_macro_f1']:.5f}")

    print("[features] building meta-learner feature matrix")
    X, feat_names = build_feature_matrix(data)
    groups = np.asarray([common.session_id(str(sid)) for sid in data.ids], dtype=object)
    print(f"  X shape={X.shape} n_groups={len(set(groups))}")

    print("[nested CV] StratifiedGroupKFold meta OOF (session-grouped)")
    meta_oof = nested_oof_meta_probs(X, data.y_true, groups, data.actions)

    meta_solo_f1 = common.macro_f1_probs(meta_oof, data.y_true, data.actions)
    print(f"  meta OOF solo macro_f1={meta_solo_f1:.5f}")

    print("[route] apply soft-AU on top of meta OOF probs (fair comparison)")
    meta_final = common.apply_soft_au(data, meta_oof, au["probs"], common.DEFAULT_ALPHA)
    meta_bundle = common.score_bundle(data, meta_final, prefix="meta_")
    meta_half = common.half_scores(data, meta_final)
    print(f"  meta 4-way-replace + soft-AU macro_f1={meta_bundle['meta_macro_f1']:.5f}")

    delta = meta_bundle["meta_macro_f1"] - base_bundle["baseline_macro_f1"]
    half1_delta = meta_half["half1_macro_f1"] - base_half["half1_macro_f1"]
    half2_delta = meta_half["half2_macro_f1"] - base_half["half2_macro_f1"]

    if delta >= GATE:
        gate = "promote"
    elif delta >= REPORT:
        gate = "report"
    else:
        gate = "discard"

    print("=" * 60)
    print(f"delta = {delta:+.5f}  half1={half1_delta:+.5f}  half2={half2_delta:+.5f}  gate={gate}")

    result = {
        "name": "gbdt_strongreg",
        "config": "strongreg",
        "hypothesis": (
            "HistGradientBoostingClassifier meta-stacker over 4-component OOF probs + "
            "structural features can beat the linear 4-way+soft-AU blend ceiling on hist12."
        ),
        "own_baseline": base_bundle["baseline_macro_f1"],
        "meta_solo_macro_f1": meta_solo_f1,
        "meta_4way_replace_macro_f1": meta_bundle["meta_macro_f1"],
        "meta_au_macro_f1": meta_bundle["meta_au_macro_f1"],
        "meta_non_au_macro_f1": meta_bundle["meta_non_au_macro_f1"],
        "baseline_au_macro_f1": base_bundle["baseline_au_macro_f1"],
        "baseline_non_au_macro_f1": base_bundle["baseline_non_au_macro_f1"],
        "delta": delta,
        "half1_delta": half1_delta,
        "half2_delta": half2_delta,
        "half1_baseline": base_half["half1_macro_f1"],
        "half1_meta": meta_half["half1_macro_f1"],
        "half2_baseline": base_half["half2_macro_f1"],
        "half2_meta": meta_half["half2_macro_f1"],
        "gate": gate,
        "gate_thresholds": {"promote": GATE, "report": REPORT},
        "leak_excluded_holdout": True,
        "n_rows": int(len(data.ids)),
        "n_features": int(X.shape[1]),
        "feature_names": feat_names,
        "model_params": {
            "max_leaf_nodes": 15,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
            "learning_rate": 0.05,
            "max_iter": 300,
            "early_stopping": True,
            "sample_weight": "balanced",
        },
        "cv": {
            "scheme": "StratifiedGroupKFold",
            "n_splits": N_SPLITS,
            "group": "session_id (session prefix)",
            "seed": SEED,
            "nested": True,
        },
        "elapsed_sec": round(time.time() - started, 2),
        "script_path": str(Path(__file__).resolve()),
    }
    common.write_json(OUT_JSON, result)
    print("[DONE]")


if __name__ == "__main__":
    main()
