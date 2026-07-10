# -*- coding: utf-8 -*-
"""Bet A 리그 판정: e5 hist12 maxlen 384→512 (D-010 후속, exp #34 잔림 8.5% 회수).

격리 설계: 동일 레시피(6ep/b16/holdout85/hist12)에서 MAXLEN만 다름.
  대조군 = colab_out/holdout_e5_h12.npz        (384, 배포판과 동일 레시피)
  후보   = colab_out/holdout_e5_h12_len512.npz (512, Colab에서 리네임 다운로드)
판정 델타 = league(len512) − league(len384), 4-way+soft-AU (mBERT 슬롯 현행 프록시).

사용: cd scripts/league4 && ..\..\.venv\Scripts\python.exe probe_a_maxlen512.py
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common

BASE_NPZ = Path(os.environ.get("E5_H12_384", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz"))
CAND_NPZ = Path(os.environ.get("E5_H12_512", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12_len512.npz"))
GATE, REPORT = 0.005, 0.002

for p in (BASE_NPZ, CAND_NPZ):
    if not p.exists():
        raise FileNotFoundError(f"npz 없음: {p} — Colab 다운로드/리네임 확인")

data = common.load_league_data()
au = common.train_or_load_au_probs(data)


def scored(e5_probs, tag):
    d = replace(data, e5=np.asarray(e5_probs, dtype=np.float64))
    final = common.apply_soft_au(d, common.four_way_blend(d), au["probs"], common.DEFAULT_ALPHA)
    solo = common.macro_f1_probs(e5_probs, data.y_true, data.actions)
    f1 = common.macro_f1_probs(final, data.y_true, data.actions)
    print(f"  [{tag}] e5 solo={solo:.5f} | 4-way+softAU={f1:.5f}")
    return f1, final


print("=" * 60)
h384 = common.align_npz_probs(BASE_NPZ, data.ids, data.y_true, data.actions)
h512 = common.align_npz_probs(CAND_NPZ, data.ids, data.y_true, data.actions)
f384, fin384 = scored(h384, "len384 대조 (배포 레시피)")
f512, fin512 = scored(h512, "len512 후보")

d_iso = f512 - f384
print("\n" + "=" * 60)
print(f"격리 델타 (len512 − len384) = {d_iso:+.5f}")
verdict = "게이트 통과 → T4 시간 실측 후 승격 후보" if d_iso >= GATE else (
    "보고 (문턱 미달)" if d_iso >= REPORT else "미달 → 폐기")
print(f"판정: {verdict}  (게이트 +{GATE})")
print("⚠️ 게이트 통과여도 배포 전 T4 추론시간 실측 필수 — 512는 인코더당 ~1.7배 추정")

if d_iso >= REPORT:
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data.ids))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    print("\n[반반 안정성] len512 vs len384:")
    acts = np.array(data.actions)
    for name, h in [("half1", h1), ("half2", h2)]:
        a = f1_score(data.y_true[h], acts[fin384[h].argmax(1)], average="macro")
        b = f1_score(data.y_true[h], acts[fin512[h].argmax(1)], average="macro")
        print(f"  {name}: {b - a:+.5f}")

print("\n[DONE]")
