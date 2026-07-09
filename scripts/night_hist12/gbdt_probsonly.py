# -*- coding: utf-8 -*-
"""GBDT 메타 스태커 프로토타입 (config 'probsonly') — hist12 리그, nested CV.

가설: hist12 블렌드(선형결합, 고정 alpha/weight)의 천장을 sklearn
HistGradientBoostingClassifier 메타러너로 넘을 수 있는가?

입력 피처 = 4성분(linear/stacker/e5-h12/mbert) OOF확률 56열만 (구조피처 없음 — 대조군).
검증 = nested StratifiedGroupKFold(5, group=session_id) — 메타러너는 fold train에서
학습하고 fold valid에서 예측(OOF 메타확률)하여 같은 행 학습/평가를 피한다.
AU 라우팅(soft-AU, alpha=0.9)은 메타확률에도 동일 적용해 4-way 대체품으로 비교.

사용: .venv python scripts/night_hist12/gbdt_probsonly.py
출력: night_out/night_hist12/gbdt_probsonly.json
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

ROOT = Path(r"C:\dev\2026-AI-DACON")
sys.path.insert(0, str(ROOT / "scripts" / "league4"))
import common  # noqa: E402

E5_H12 = ROOT / "colab_out" / "holdout_e5_h12.npz"
OUT_DIR = ROOT / "night_out" / "league4"
RESULT_PATH = ROOT / "night_out" / "night_hist12" / "gbdt_probsonly.json"
GATE = 0.005
REPORT = 0.002
N_SPLITS = 5
SEED = 42


def build_feature_matrix(data: common.LeagueData) -> np.ndarray:
    """56열 = [lin(14), stk(14), e5-h12(14), mbert(14)] — 구조피처 없음 (대조군)."""
    return np.concatenate([data.lin, data.stk, data.e5, data.mbert], axis=1).astype(np.float64)


def nested_oof_meta_probs(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    classes: list[str],
    n_splits: int = N_SPLITS,
    seed: int = SEED,
) -> tuple[np.ndarray, dict]:
    """fold별 HGB 학습 → fold valid에 대한 OOF 메타확률(정렬은 data.actions 순서)."""
    n = X.shape[0]
    n_classes = len(classes)
    oof = np.zeros((n, n_classes), dtype=np.float64)
    filled = np.zeros(n, dtype=bool)
    y_enc_map = {c: i for i, c in enumerate(classes)}
    y_enc = np.asarray([y_enc_map[str(v)] for v in y], dtype=np.int64)

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_meta = []
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(sgkf.split(X, y_enc, groups)):
        # train/valid 세션 그룹 분리 확인 (누수 차단)
        tr_groups = set(groups[tr_idx])
        va_groups = set(groups[va_idx])
        if tr_groups & va_groups:
            raise AssertionError(f"fold {fold_i}: train/valid 세션 그룹 겹침 (누수)")

        sw = compute_sample_weight("balanced", y_enc[tr_idx])
        clf = HistGradientBoostingClassifier(
            max_leaf_nodes=15,
            min_samples_leaf=200,
            l2_regularization=1.0,
            random_state=seed,
        )
        clf.fit(X[tr_idx], y_enc[tr_idx], sample_weight=sw)

        # clf.classes_는 fold train에 존재하는 클래스만 포함할 수 있음 → 전체 14클래스 열로 정렬
        proba = clf.predict_proba(X[va_idx])
        fold_classes = [classes[int(c)] for c in clf.classes_]
        aligned = common.align_probs(proba, fold_classes, classes)

        oof[va_idx] = aligned
        filled[va_idx] = True
        fold_meta.append(
            {
                "fold": fold_i,
                "train_rows": int(len(tr_idx)),
                "valid_rows": int(len(va_idx)),
                "train_sessions": int(len(tr_groups)),
                "valid_sessions": int(len(va_groups)),
                "fold_classes_seen": len(fold_classes),
            }
        )

    if not filled.all():
        missing = int((~filled).sum())
        raise AssertionError(f"OOF 메타확률 미충족 행 {missing}개 (fold 커버리지 문제)")

    meta = {
        "n_splits": n_splits,
        "seed": seed,
        "elapsed_sec": round(time.time() - t0, 3),
        "folds": fold_meta,
    }
    return oof, meta


def main() -> None:
    print("=" * 60)
    print("[load] league data (honest 9969) + hist12 e5 스왑")
    data = common.load_league_data()  # sanity assert 내장 (기본 e5)
    h12 = common.align_npz_probs(E5_H12, data.ids, data.y_true, data.actions)
    data = replace(data, e5=h12)  # hist12 e5로 교체 — 이후 모든 계산은 hist12 리그 기준

    # hist12 baseline (4-way + soft-AU) 자기 계산
    au = common.train_or_load_au_probs(data, OUT_DIR, force=False)
    base_blend = common.four_way_blend(data)
    base_final = common.apply_soft_au(data, base_blend, au["probs"], common.DEFAULT_ALPHA)
    baseline_f1 = common.macro_f1_probs(base_final, data.y_true, data.actions)
    print(f"[baseline] hist12 4-way+softAU macro-F1 = {baseline_f1:.5f}")

    # 56열 피처 (구조피처 없음 — 대조군)
    X = build_feature_matrix(data)
    print(f"[features] X shape = {X.shape} (4성분 x 14클래스 OOF확률만)")

    groups = np.asarray([common.session_id(str(i)) for i in data.ids], dtype=object)

    print("[nested CV] StratifiedGroupKFold(n_splits=5, group=session_id) — fold train 학습 / fold valid 예측")
    meta_oof, cv_meta = nested_oof_meta_probs(X, data.y_true, groups, data.actions)

    # 메타확률 자체(soft-AU 적용 전)의 solo 스코어
    meta_solo_f1 = common.macro_f1_probs(meta_oof, data.y_true, data.actions)

    # soft-AU 동일 적용 (공정 비교: 4-way 대체품으로서 메타확률)
    meta_final = common.apply_soft_au(data, meta_oof, au["probs"], common.DEFAULT_ALPHA)
    meta_f1 = common.macro_f1_probs(meta_final, data.y_true, data.actions)

    delta = meta_f1 - baseline_f1
    print(f"[gbdt meta] solo(soft-AU 전) macro-F1 = {meta_solo_f1:.5f}")
    print(f"[gbdt meta] 4-way대체 + soft-AU macro-F1 = {meta_f1:.5f}")
    print(f"[delta] gbdt_meta - hist12_baseline = {delta:+.5f}")

    # 반반 안정성
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data.ids))
    half = len(perm) // 2
    h1_idx, h2_idx = perm[:half], perm[half:]

    def half_delta(idx: np.ndarray) -> tuple[float, float, float]:
        base_h = common.macro_f1_probs(base_final[idx], data.y_true[idx], data.actions)
        meta_h = common.macro_f1_probs(meta_final[idx], data.y_true[idx], data.actions)
        return meta_h - base_h, base_h, meta_h

    half1_delta, half1_base, half1_meta = half_delta(h1_idx)
    half2_delta, half2_base, half2_meta = half_delta(h2_idx)
    print(f"[half1] base={half1_base:.5f} meta={half1_meta:.5f} delta={half1_delta:+.5f}")
    print(f"[half2] base={half2_base:.5f} meta={half2_meta:.5f} delta={half2_delta:+.5f}")

    sign_agree = (half1_delta >= 0) == (half2_delta >= 0) == (delta >= 0)
    if delta >= GATE:
        gate = "promote"
    elif delta >= REPORT:
        gate = "report"
    else:
        gate = "discard"
    if gate in ("promote", "report") and not sign_agree:
        gate_note = "half1/half2 부호 불일치 — 신기루 의심 (report로 강등)"
        if gate == "promote":
            gate = "report"
    else:
        gate_note = "half1/half2 부호 일치" if sign_agree else "half 부호 불일치(미해당, delta<report)"

    result = {
        "name": "gbdt_probsonly",
        "own_baseline": round(baseline_f1, 5),
        "delta": round(delta, 5),
        "half1_delta": round(half1_delta, 5),
        "half2_delta": round(half2_delta, 5),
        "gate": gate,
        "gate_note": gate_note,
        "leak_excluded_holdout": True,
        "meta_solo_f1": round(meta_solo_f1, 5),
        "meta_final_f1": round(meta_f1, 5),
        "baseline_final_f1": round(baseline_f1, 5),
        "model": {
            "estimator": "HistGradientBoostingClassifier",
            "max_leaf_nodes": 15,
            "min_samples_leaf": 200,
            "l2_regularization": 1.0,
            "sample_weight": "balanced",
        },
        "cv": {
            "scheme": "StratifiedGroupKFold(n_splits=5, group=session_id)",
            "seed": SEED,
            **cv_meta,
        },
        "features": {
            "description": "4성분 OOF확률 56열만 (linear/stacker/e5-h12/mbert x 14클래스) — 구조피처 없음(대조군)",
            "n_features": int(X.shape[1]),
        },
        "au_routing": {
            "method": "apply_soft_au",
            "alpha": common.DEFAULT_ALPHA,
            "note": "au_probs는 기존 캐시(char_wb TF-IDF+LinearSVC) 재사용, 메타확률에도 동일 적용",
        },
        "rows": int(len(data.ids)),
        "n_actions": len(data.actions),
        "e5_slot": "hist12 (colab_out/holdout_e5_h12.npz)",
        "script": "scripts/night_hist12/gbdt_probsonly.py",
    }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {RESULT_PATH}")
    print("=" * 60)
    print(f"판정: {gate}  (delta={delta:+.5f}, gate_note={gate_note})")


if __name__ == "__main__":
    main()
