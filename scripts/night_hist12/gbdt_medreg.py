# -*- coding: utf-8 -*-
"""GBDT 메타 스태커 프로토타입 (config 'medreg') — exp #A / hist12 리그.

목표: hist12 블렌드(선형결합) 천장을 sklearn HistGradientBoostingClassifier
메타러너로 넘을 수 있는지 로컬 nested-CV로 판정.

데이터: common.load_league_data() → e5 슬롯을 holdout_e5_h12.npz로 스왑 (honest 9969행).
피처: 4성분(linear/stacker/e5-h12/mbert) OOF확률(14*4=56열)
      + 성분별 [max, top2margin, entropy] (4*3=12열)
      + 구조피처(session_meta/workspace, 추론시 결정적인 것만).
모델: HistGradientBoostingClassifier(max_leaf_nodes=31, min_samples_leaf=100,
      l2_regularization=0.3, learning_rate=0.05, max_iter=400, early_stopping=True),
      sample_weight=balanced(클래스 빈도 역가중).
검증: nested StratifiedGroupKFold(n_splits=5, group=session_id) — outer fold의 valid는
      해당 fold 학습 모델의 예측만 사용 (같은 행 학습·평가 절대 금지) → 5-fold를 합쳐 OOF 메타확률
      9969행 전체 구성. AU 라우팅은 메타확률에도 apply_soft_au 동일 적용.
판정: 메타 OOF macro-F1(4-way 대체, +soft-AU) vs hist12 baseline(0.75601).
       델타, half1/half2 반반 안정성.

사용: C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\night_hist12\gbdt_medreg.py
출력: night_out\night_hist12\gbdt_medreg.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "league4"))
import common  # noqa: E402

ROOT = Path(r"C:\dev\2026-AI-DACON")
E5_H12 = ROOT / "colab_out" / "holdout_e5_h12.npz"
OUT_DIR = ROOT / "night_out" / "night_hist12"
LEAGUE_OUT = ROOT / "night_out" / "league4"
OUT_JSON = OUT_DIR / "gbdt_medreg.json"

HIST12_BASELINE = 0.75601
GATE = 0.005
REPORT = 0.002
N_SPLITS = 5
SEED = 42
CONFIG_NAME = "medreg"

COMPONENT_NAMES = ["linear", "stacker", "e5_h12", "mbert"]


def component_stats(probs: np.ndarray) -> np.ndarray:
    """[max, top2margin, entropy] per row for one component's prob matrix."""
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    sorted_p = np.sort(p, axis=1)
    top1 = sorted_p[:, -1]
    top2 = sorted_p[:, -2]
    margin = top1 - top2
    entropy = -(p * np.log(p)).sum(axis=1)
    return np.column_stack([top1, margin, entropy])


def structural_features(ids: np.ndarray, samples_by_id: dict[str, dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    """Deterministic-at-inference structural features from session_meta/workspace.

    Only fields that are directly present in the input sample at inference time
    (no leakage from future turns / labels).
    """
    rows = []
    for sample_id in ids:
        s = samples_by_id[str(sample_id)]
        meta = s.get("session_meta", {}) or {}
        ws = meta.get("workspace", {}) or {}
        lang_mix = ws.get("language_mix", {}) or {}
        history = s.get("history", []) or []

        user_tier = str(meta.get("user_tier", "unknown"))
        language_pref = str(meta.get("language_pref", "unknown"))
        budget_tokens_remaining = float(meta.get("budget_tokens_remaining", 0) or 0)
        turn_index = float(meta.get("turn_index", 0) or 0)
        elapsed_session_sec = float(meta.get("elapsed_session_sec", 0) or 0)

        loc = float(ws.get("loc", 0) or 0)
        git_dirty = 1.0 if ws.get("git_dirty") else 0.0
        n_open_files = float(len(ws.get("open_files", []) or []))
        last_ci_status = str(ws.get("last_ci_status", "unknown"))

        py_frac = float(lang_mix.get("py", 0.0) or 0.0)
        n_langs = float(len(lang_mix))
        max_lang_frac = float(max(lang_mix.values())) if lang_mix else 0.0

        n_history = float(len(history))
        last_action = ""
        for turn in reversed(history):
            if turn.get("role") == "assistant_action":
                last_action = str(turn.get("name", ""))
                break
        current_prompt_len = float(len(str(s.get("current_prompt", "") or "")))

        rows.append(
            {
                "user_tier": user_tier,
                "language_pref": language_pref,
                "budget_tokens_remaining": budget_tokens_remaining,
                "turn_index": turn_index,
                "elapsed_session_sec": elapsed_session_sec,
                "loc": loc,
                "git_dirty": git_dirty,
                "n_open_files": n_open_files,
                "last_ci_status": last_ci_status,
                "py_frac": py_frac,
                "n_langs": n_langs,
                "max_lang_frac": max_lang_frac,
                "n_history": n_history,
                "last_action": last_action,
                "current_prompt_len": current_prompt_len,
            }
        )

    cat_cols = ["user_tier", "language_pref", "last_ci_status", "last_action"]
    num_cols = [
        "budget_tokens_remaining",
        "turn_index",
        "elapsed_session_sec",
        "loc",
        "git_dirty",
        "n_open_files",
        "py_frac",
        "n_langs",
        "max_lang_frac",
        "n_history",
        "current_prompt_len",
    ]

    # simple frequency-independent category encoding: stable label -> int code
    # (fit vocab once over the full 9969-row set; ids/order are fixed, no OOF label leakage
    #  since these are raw input categorical strings, not derived from y)
    cat_arrays = []
    cat_names = []
    for col in cat_cols:
        values = [r[col] for r in rows]
        vocab = sorted(set(values))
        vocab_index = {v: i for i, v in enumerate(vocab)}
        codes = np.asarray([vocab_index[v] for v in values], dtype=np.float64)
        cat_arrays.append(codes)
        cat_names.append(f"cat_{col}")

    num_arrays = [np.asarray([r[col] for r in rows], dtype=np.float64) for col in num_cols]
    num_names = [f"num_{col}" for col in num_cols]

    feat = np.column_stack(cat_arrays + num_arrays)
    names = cat_names + num_names
    return feat, names


def build_features(data: common.LeagueData) -> tuple[np.ndarray, list[str]]:
    comps = {"linear": data.lin, "stacker": data.stk, "e5_h12": data.e5, "mbert": data.mbert}
    prob_blocks = []
    prob_names = []
    for name, probs in comps.items():
        prob_blocks.append(probs)
        prob_names.extend(f"{name}_p_{a}" for a in data.actions)

    stat_blocks = []
    stat_names = []
    for name, probs in comps.items():
        stat_blocks.append(component_stats(probs))
        stat_names.extend([f"{name}_max", f"{name}_top2margin", f"{name}_entropy"])

    struct_feat, struct_names = structural_features(data.ids, data.samples_by_id)

    X = np.column_stack(prob_blocks + stat_blocks + [struct_feat])
    names = prob_names + stat_names + struct_names
    return X, names


def nested_oof_meta_probs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    actions: list[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> np.ndarray:
    n = X.shape[0]
    n_classes = len(actions)
    action_index = {a: i for i, a in enumerate(actions)}
    y_idx = np.asarray([action_index[str(v)] for v in y], dtype=np.int64)

    oof = np.zeros((n, n_classes), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)

    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_meta = []
    for fold_i, (train_idx, valid_idx) in enumerate(skf.split(X, y_idx, groups)):
        # leakage guard: fold train/valid must not share session groups
        train_groups = set(groups[train_idx])
        valid_groups = set(groups[valid_idx])
        overlap = train_groups & valid_groups
        if overlap:
            raise AssertionError(f"fold {fold_i}: session group leakage between train/valid ({len(overlap)} groups)")

        sw = compute_sample_weight(class_weight="balanced", y=y_idx[train_idx])
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=31,
            min_samples_leaf=100,
            l2_regularization=0.3,
            learning_rate=0.05,
            max_iter=400,
            early_stopping=True,
            random_state=seed,
        )
        clf.fit(X[train_idx], y_idx[train_idx], sample_weight=sw)

        proba = clf.predict_proba(X[valid_idx])
        # HGB only learns classes present in y_idx[train_idx]; align to full action space
        proba_full = np.zeros((len(valid_idx), n_classes), dtype=np.float64)
        for local_i, cls in enumerate(clf.classes_):
            proba_full[:, int(cls)] = proba[:, local_i]
        row_sum = proba_full.sum(axis=1, keepdims=True)
        row_sum[row_sum <= 0] = 1.0
        proba_full = proba_full / row_sum

        oof[valid_idx] = proba_full
        filled[valid_idx] = True
        fold_meta.append(
            {
                "fold": fold_i,
                "n_train": int(len(train_idx)),
                "n_valid": int(len(valid_idx)),
                "n_iter_": int(getattr(clf, "n_iter_", -1)),
                "train_sessions": int(len(train_groups)),
                "valid_sessions": int(len(valid_groups)),
            }
        )

    if not filled.all():
        missing = int((~filled).sum())
        raise AssertionError(f"nested CV did not cover all rows: {missing} missing")

    return oof, fold_meta


def half_delta(y_true, actions, final_a, final_b, seed=42):
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(y_true))
    half = len(perm) // 2
    acts = np.array(actions)
    out = {}
    for name, idx in (("half1", perm[:half]), ("half2", perm[half:])):
        fa = f1_score(y_true[idx], acts[final_a[idx].argmax(1)], average="macro", zero_division=0)
        fb = f1_score(y_true[idx], acts[final_b[idx].argmax(1)], average="macro", zero_division=0)
        out[name] = fb - fa
    return out


def main() -> None:
    started = time.time()
    print("=" * 60)
    print(f"[config] {CONFIG_NAME}: HGB meta-stacker over 4-way OOF probs + stats + structural feats")

    # 1) load league (hist12 swap)
    data = common.load_league_data()
    h12 = common.align_npz_probs(E5_H12, data.ids, data.y_true, data.actions)
    data = replace(data, e5=h12)

    # leakage guard: holdout ids must not be in train set used elsewhere (honest OOF already)
    train_id_set = set(str(x) for x in data.train_ids)
    holdout_id_set = set(str(x) for x in data.ids)
    # sanity: holdout ids ARE expected to be a subset of train.jsonl ids (labels are known),
    # but this league treats them as an out-of-sample honest holdout (per task instructions:
    # leak_excluded_holdout=true, i.e. no *additional* holdout carve-out is required here).
    print(f"[data] n_rows={len(data.ids)} n_actions={len(data.actions)}")

    # baseline: linear blend of the same hist12 4 components (self-computed reference)
    au = common.train_or_load_au_probs(data, LEAGUE_OUT, force=False)
    base_blend = common.four_way_blend(data)
    base_final = common.apply_soft_au(data, base_blend, au["probs"], common.DEFAULT_ALPHA)
    base_f1 = common.macro_f1_probs(base_final, data.y_true, data.actions)
    print(f"[self baseline] hist12 4-way+softAU = {base_f1:.5f} (task-stated {HIST12_BASELINE:.5f})")

    # 2) features
    X, feat_names = build_features(data)
    print(f"[features] X.shape={X.shape} n_features={len(feat_names)}")

    groups = np.asarray([common.session_id(str(x)) for x in data.ids], dtype=object)

    # 3) nested OOF meta-probs
    oof_meta, fold_meta = nested_oof_meta_probs(X, data.y_true, groups, data.actions)

    # honest OOF sanity: no row should be predicted "with itself in train" -- guaranteed by
    # nested_oof_meta_probs raising on group overlap per fold.

    meta_solo_f1 = common.macro_f1_probs(oof_meta, data.y_true, data.actions)
    meta_final = common.apply_soft_au(data, oof_meta, au["probs"], common.DEFAULT_ALPHA)
    meta_f1 = common.macro_f1_probs(meta_final, data.y_true, data.actions)

    delta_vs_self_baseline = meta_f1 - base_f1
    delta_vs_task_baseline = meta_f1 - HIST12_BASELINE

    print(f"[meta] solo macro-F1 (no AU route) = {meta_solo_f1:.5f}")
    print(f"[meta] +soft-AU macro-F1           = {meta_f1:.5f}")
    print(f"[delta] meta - self baseline        = {delta_vs_self_baseline:+.5f}")
    print(f"[delta] meta - task-stated baseline = {delta_vs_task_baseline:+.5f}")

    hd = half_delta(data.y_true, data.actions, base_final, meta_final)
    print(f"[half] half1={hd['half1']:+.5f} half2={hd['half2']:+.5f}")

    if delta_vs_self_baseline >= GATE:
        gate = "promote"
    elif delta_vs_self_baseline >= REPORT:
        gate = "report"
    else:
        gate = "discard"
    print(f"[gate] {gate} (>= +{GATE} promote, >= +{REPORT} report, else discard)")

    elapsed = time.time() - started
    print(f"[time] elapsed={elapsed:.1f}s")

    result = {
        "name": "gbdt_medreg",
        "config": CONFIG_NAME,
        "own_baseline_self_computed": round(base_f1, 5),
        "own_baseline_task_stated": HIST12_BASELINE,
        "meta_solo_macro_f1_no_au": round(meta_solo_f1, 5),
        "meta_macro_f1_with_soft_au": round(meta_f1, 5),
        "delta_vs_self_baseline": round(delta_vs_self_baseline, 5),
        "delta_vs_task_baseline": round(delta_vs_task_baseline, 5),
        "half1_delta": round(hd["half1"], 5),
        "half2_delta": round(hd["half2"], 5),
        "gate": gate,
        "n_rows": int(len(data.ids)),
        "n_features": int(X.shape[1]),
        "feature_names": feat_names,
        "model_params": {
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 0.3,
            "learning_rate": 0.05,
            "max_iter": 400,
            "early_stopping": True,
            "sample_weight": "balanced",
        },
        "cv": {
            "scheme": "nested StratifiedGroupKFold(n_splits=5, group=session_id)",
            "seed": SEED,
            "folds": fold_meta,
        },
        "leak_excluded_holdout": True,
        "elapsed_sec": round(elapsed, 1),
        "script": str(Path(__file__).resolve()),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {OUT_JSON}")
    print("[DONE]")


if __name__ == "__main__":
    main()
