# -*- coding: utf-8 -*-
"""Type F 오분류쌍 원시행 표본 — 판별피처 존재 여부 육안 판정.

혼동쌍별로 (true==A & blend_pred==B) 행에서 current_prompt + 직전 history action을
표본 출력. features.py에 넣을 리터럴 신호가 보이는지(=features 갭) vs 진짜 모호(=라벨 노이즈) 판정.
"""
from __future__ import annotations

import numpy as np

import common as C


def last_hist_action(sample):
    hist = sample.get("history") or []
    for turn in reversed(hist):
        a = turn.get("assistant_action")
        if isinstance(a, dict) and a.get("name"):
            return a["name"]
    return "<none>"


def main():
    data = C.load_league_data()
    blend = C.four_way_blend(data)
    au = C.train_or_load_au_probs(data)
    final = C.apply_soft_au(data, blend, au["probs"], alpha=C.DEFAULT_ALPHA)
    pred = C.predict_from_probs(final, data.actions)
    y = np.asarray(data.y_true)

    pairs = [
        ("grep_search", "read_file"),
        ("grep_search", "list_directory"),
        ("glob_pattern", "list_directory"),
        ("read_file", "grep_search"),
        ("edit_file", "apply_patch"),
    ]
    rng = np.random.default_rng(42)
    for true_a, pred_b in pairs:
        mask = (y == true_a) & (pred == pred_b)
        idx = np.where(mask)[0]
        pick = rng.choice(idx, size=min(6, len(idx)), replace=False)
        print(f"\n{'='*90}\n### {true_a}  ->(blend)  {pred_b}   (n={len(idx)})")
        for i in sorted(pick):
            sid = str(data.ids[i])
            s = data.samples_by_id[sid]
            cp = str(s.get("current_prompt", "")).replace("\n", " ")
            print(f"  last={last_hist_action(s):16s} | {cp[:150]}")


if __name__ == "__main__":
    main()
