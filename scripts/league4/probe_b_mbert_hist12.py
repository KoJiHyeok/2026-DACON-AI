# -*- coding: utf-8 -*-
"""Bet B 리그 판정: mBERT hist6→hist12 (D-010 후속, 남은 hist6 성분 정합).

격리 설계: mBERT 동일 레시피(bert-base-multilingual-cased, 2ep/유효배치16/384/holdout85)에서
MAXHIST만 다름. e5 슬롯은 hist12(배포판 프록시)로 고정 — 새 baseline 위에서 판정.
  대조군 = colab_out/holdout_mbert.npz       (2ep hist6, exp #27 프록시)
  후보   = colab_out/holdout_mbert_h12.npz   (2ep hist12, Colab에서 리네임 다운로드)
판정 델타 = league(e5h12 + mbert_h12) − league(e5h12 + mbert_h6).

⚠️ #29 교훈: mBERT 단독 상승 ≠ 블록 기여 — 블록 그리드(비율 재탐색)까지 봐야 함.
   여기선 현행 비율 [1.2, 0.8] 고정 델타 + 비율 그리드 스캔 둘 다 출력.

사용: cd scripts/league4 && ..\..\.venv\Scripts\python.exe probe_b_mbert_hist12.py
"""
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common

E5_H12 = Path(os.environ.get("E5_H12", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz"))
CAND_NPZ = Path(os.environ.get("MBERT_H12", r"C:\dev\2026-AI-DACON\colab_out\holdout_mbert_h12.npz"))
GATE, REPORT = 0.005, 0.002

for p in (E5_H12, CAND_NPZ):
    if not p.exists():
        raise FileNotFoundError(f"npz 없음: {p} — Colab 다운로드/리네임 확인")

data0 = common.load_league_data()          # mbert = 2ep hist6 프록시 (대조군)
au = common.train_or_load_au_probs(data0)
h12 = common.align_npz_probs(E5_H12, data0.ids, data0.y_true, data0.actions)
data = replace(data0, e5=h12)              # e5 슬롯 = hist12 고정 (배포 정합)
mb_h12 = common.align_npz_probs(CAND_NPZ, data.ids, data.y_true, data.actions)


def scored(d, tag, e5_w=common.BASE_E5_WEIGHT, mb_w=common.BASE_MBERT_WEIGHT):
    blend = common.four_way_blend(d, e5_w, mb_w)
    final = common.apply_soft_au(d, blend, au["probs"], common.DEFAULT_ALPHA)
    f1 = common.macro_f1_probs(final, data.y_true, data.actions)
    print(f"  [{tag}] 4-way+softAU={f1:.5f} (블록 {e5_w}/{mb_w})")
    return f1, final


print("=" * 60)
mb_solo6 = common.macro_f1_probs(data.mbert, data.y_true, data.actions)
mb_solo12 = common.macro_f1_probs(mb_h12, data.y_true, data.actions)
print(f"mBERT solo: hist6={mb_solo6:.5f} → hist12={mb_solo12:.5f} ({mb_solo12-mb_solo6:+.5f})")

f_base, fin_base = scored(data, "대조 (e5h12 + mbert_h6)")
d_cand = replace(data, mbert=mb_h12)
f_cand, fin_cand = scored(d_cand, "후보 (e5h12 + mbert_h12)")

d_iso = f_cand - f_base
print("\n" + "=" * 60)
print(f"격리 델타 (mbert h12 − h6, 블록 [1.2,0.8] 고정) = {d_iso:+.5f}")

# 블록 비율 재탐색 — hist12 mBERT가 e5와 상관이 달라지면 최적 비율 이동 가능 (#29 방식)
print("\n[블록 비율 그리드] 총가중 2.0 고정, e5_w 스캔:")
best = (None, -1)
for e5_w in (1.5, 1.4, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8):
    mb_w = 2.0 - e5_w
    f1, _ = scored(d_cand, f"e5={e5_w:.1f}", e5_w, mb_w)
    if f1 > best[1]:
        best = ((e5_w, mb_w), f1)
print(f"  best: 블록 {best[0]} → {best[1]:.5f} (대조 대비 {best[1]-f_base:+.5f})")

verdict = "게이트 통과 → mBERT full-train 재학습 후보" if max(d_iso, best[1] - f_base) >= GATE else (
    "보고 (문턱 미달)" if max(d_iso, best[1] - f_base) >= REPORT else "미달 → 폐기")
print(f"\n판정: {verdict}  (게이트 +{GATE})")
print("⚠️ 비율 그리드 최적이 e5 지분을 낮추는 쪽이면 enc-지분 신기루 경계 — LB 게이트로만 승격")

if max(d_iso, best[1] - f_base) >= REPORT:
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data.ids))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    print("\n[반반 안정성] 후보(고정 비율) vs 대조:")
    acts = np.array(data.actions)
    for name, h in [("half1", h1), ("half2", h2)]:
        a = f1_score(data.y_true[h], acts[fin_base[h].argmax(1)], average="macro")
        b = f1_score(data.y_true[h], acts[fin_cand[h].argmax(1)], average="macro")
        print(f"  {name}: {b - a:+.5f}")

print("\n[DONE]")
