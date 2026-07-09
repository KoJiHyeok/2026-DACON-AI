# -*- coding: utf-8 -*-
"""exp #34 hist12 e5가 탐색계열 혼동을 얼마나 녹이는가 — 배포(hist6) vs hist12 혼동쌍 델타.

C/B 정찰(hist6 기준)의 결론이 #34 배포로 이미 해소되는지 판정.
동일 4-way+soft-AU 표면에서 e5 슬롯만 hist6대조↔hist12 스왑하여 상위 오분류쌍 count 비교.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common as C

E5_H6 = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h6.npz")
E5_H12 = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz")


def final_pred(data, au):
    blend = C.four_way_blend(data)
    final = C.apply_soft_au(data, blend, au["probs"], C.DEFAULT_ALPHA)
    return C.predict_from_probs(final, data.actions), final


def confusion_counts(y, pred, actions):
    a_idx = {a: i for i, a in enumerate(actions)}
    K = len(actions)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[a_idx[t], a_idx[p]] += 1
    return cm


def main():
    data = C.load_league_data()
    au = C.train_or_load_au_probs(data)
    actions = data.actions
    y = np.asarray(data.y_true)

    h6 = C.align_npz_probs(E5_H6, data.ids, data.y_true, actions)
    h12 = C.align_npz_probs(E5_H12, data.ids, data.y_true, actions)

    d6 = replace(data, e5=h6)
    d12 = replace(data, e5=h12)
    pred6, final6 = final_pred(d6, au)
    pred12, final12 = final_pred(d12, au)

    f6 = f1_score(y, pred6, labels=actions, average="macro", zero_division=0)
    f12 = f1_score(y, pred12, labels=actions, average="macro", zero_division=0)
    print(f"[블렌드 macroF1] hist6대조={f6:.5f}  hist12={f12:.5f}  델타={f12-f6:+.5f}")

    cm6 = confusion_counts(y, pred6, actions)
    cm12 = confusion_counts(y, pred12, actions)

    # 관심 오분류쌍 (hist6 recon의 상위)
    pairs = [
        ("grep_search", "read_file"),
        ("grep_search", "list_directory"),
        ("glob_pattern", "list_directory"),
        ("read_file", "grep_search"),
        ("read_file", "list_directory"),
        ("glob_pattern", "read_file"),
        ("list_directory", "read_file"),
        ("edit_file", "apply_patch"),
        ("ask_user", "plan_task"),
        ("lint_or_typecheck", "run_tests"),
    ]
    a_idx = {a: i for i, a in enumerate(actions)}
    print(f"\n{'오분류쌍':38s} {'hist6':>6s} {'hist12':>7s} {'델타':>6s}")
    total6 = total12 = 0
    for tr, pr in pairs:
        i, j = a_idx[tr], a_idx[pr]
        c6, c12 = int(cm6[i, j]), int(cm12[i, j])
        total6 += c6
        total12 += c12
        print(f"  {tr+' -> '+pr:36s} {c6:6d} {c12:7d} {c12-c6:+6d}")
    print(f"  {'(위 쌍 합계)':36s} {total6:6d} {total12:7d} {total12-total6:+6d}")

    # 탐색계열 per-class F1 델타
    print(f"\n{'클래스':18s} {'hist6 F1':>9s} {'hist12 F1':>10s} {'델타':>7s}")
    for a in ["list_directory", "read_file", "grep_search", "glob_pattern",
              "lint_or_typecheck", "ask_user", "edit_file", "apply_patch"]:
        m = y == a
        f1a = f1_score((y == a), (pred6 == a), zero_division=0)
        f1b = f1_score((y == a), (pred12 == a), zero_division=0)
        print(f"  {a:18s} {f1a:9.4f} {f1b:10.4f} {f1b-f1a:+7.4f}")


if __name__ == "__main__":
    main()
