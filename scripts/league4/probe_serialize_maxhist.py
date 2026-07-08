# -*- coding: utf-8 -*-
"""serialize max_hist 재심 리그 판정 (exp #34 / D-010).

e5 슬롯에 hist6 대조군·hist12 후보를 각각 스왑해 4-way+soft-AU 점수를 재계산한다.
판정 델타 = hist12 − hist6대조 (serialize 효과 격리, 프록시 출처 무관 self-contained).

입력 npz (colab에서 다운로드):
  ENV E5_H6  기본 colab_out/holdout_e5_h6.npz   (ENC_MAXHIST=6 대조군)
  ENV E5_H12 기본 colab_out/holdout_e5_h12.npz  (ENC_MAXHIST=12 후보)

사용: .venv-merge python probe_serialize_maxhist.py
"""
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

E5_H6 = Path(os.environ.get("E5_H6", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h6.npz"))
E5_H12 = Path(os.environ.get("E5_H12", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz"))
GATE = 0.005
REPORT = 0.002
OUT = Path(r"C:\dev\2026-AI-DACON\night_out\league4")

for p in (E5_H6, E5_H12):
    if not p.exists():
        raise FileNotFoundError(f"npz 없음: {p} — Colab 다운로드 확인")

# 1) 리그 로드 (원본 e5 — sanity 통과) + AU 확률(리그 e5와 무관, 1회)
data = common.load_league_data()   # 기본 인자 = 배포 e5/2ep mBERT, sanity 검증 포함
au = common.train_or_load_au_probs(data, OUT, force=False)


def scored(e5_probs, tag):
    d = replace(data, e5=np.asarray(e5_probs, dtype=np.float64))
    blend = common.four_way_blend(d)                                  # [lin,stk,1.2*e5,0.8*mbert]
    final = common.apply_soft_au(d, blend, au["probs"], common.DEFAULT_ALPHA)
    solo = common.macro_f1_probs(e5_probs, data.y_true, data.actions)
    f1 = common.macro_f1_probs(final, data.y_true, data.actions)
    print(f"  [{tag}] e5 solo={solo:.5f} | 4-way+softAU={f1:.5f}")
    return f1, final


print("=" * 60)
# 0) 배포 e5 기준선 재현 (0.73877 근처여야 리그 정합)
base_blend = common.four_way_blend(data)
base_final = common.apply_soft_au(data, base_blend, au["probs"], common.DEFAULT_ALPHA)
base_f1 = common.macro_f1_probs(base_final, data.y_true, data.actions)
print(f"[기준선] 배포 e5(프록시) 4-way+softAU = {base_f1:.5f}  (기대 0.73877)")

# 2) hist6 대조군 / hist12 후보 스왑
h6 = common.align_npz_probs(E5_H6, data.ids, data.y_true, data.actions)
h12 = common.align_npz_probs(E5_H12, data.ids, data.y_true, data.actions)
print("\n[스왑 판정]")
f6, final6 = scored(h6, "hist6 대조")
f12, final12 = scored(h12, "hist12 후보")

# 3) 델타
d_iso = f12 - f6           # serialize 효과 격리 (동일 레시피)
d_prac = f12 - base_f1     # 배포 e5 대비 실용 델타
print("\n" + "=" * 60)
print(f"격리 델타 (hist12 − hist6대조) = {d_iso:+.5f}")
print(f"실용 델타 (hist12 − 배포 e5)   = {d_prac:+.5f}")
print(f"대조군 정합 (hist6대조 − 배포)  = {f6-base_f1:+.5f}  (0 근처면 레시피 일치)")
verdict = "게이트 통과 → 승격 후보" if d_iso >= GATE else (
    "보고 (문턱 미달)" if d_iso >= REPORT else "미달 → 폐기")
print(f"판정: {verdict}  (게이트 +{GATE})")

# 4) 반반 안정성 (격리 델타가 보고 이상일 때만)
if d_iso >= REPORT:
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(data.ids))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    print("\n[반반 안정성] hist12 vs hist6대조:")
    for name, h in [("half1", h1), ("half2", h2)]:
        y = data.y_true[h]
        acts = np.array(data.actions)
        a = f1_score(y, acts[final6[h].argmax(1)], average="macro")
        b = f1_score(y, acts[final12[h].argmax(1)], average="macro")
        print(f"  {name}: {b-a:+.5f}")

print("\n[DONE]")
