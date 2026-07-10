# -*- coding: utf-8 -*-
"""D-011 리그 판정: e5 hist12 + last-action args-lite (제3자 SOL 감사 P1-B).

대조군 = colab_out/holdout_e5_h12.npz          (배포 hist12, ENC_ARGSLITE=0과 바이트 동일 계약)
후보   = colab_out/holdout_e5_h12_args1.npz    (hist12 + lastargs, Colab ENC_ARGSLITE=1)

판정 지표 (감사 P0 반영 — row 단독 판정 금지):
  ① row Macro-F1 델타            ② 세션 균등 가중 Macro-F1 델타
  ③ 세션당 1행 Monte Carlo 200회 델타 평균±표준편차
  ④ paired session bootstrap 2,000회 95% CI
  ⑤ 반반 안정성
게이트: ①≥+0.005 AND ④ CI 하한>0 AND ③ 평균>0 (D-011).
"""
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common

BASE_NPZ = Path(os.environ.get("E5_H12", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz"))
CAND_NPZ = Path(os.environ.get("E5_H12_ARGS", r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12_args1.npz"))
GATE = 0.005

for p in (BASE_NPZ, CAND_NPZ):
    if not p.exists():
        raise FileNotFoundError(f"npz 없음: {p} — Colab 다운로드 확인")

data = common.load_league_data()
au = common.train_or_load_au_probs(data)
actions = data.actions
acts = np.array(actions)
y = np.asarray(data.y_true)
sess = np.array([common.session_id(str(i)) for i in data.ids])


def final_pred(e5_probs):
    d = replace(data, e5=np.asarray(e5_probs, dtype=np.float64))
    final = common.apply_soft_au(d, common.four_way_blend(d), au["probs"], common.DEFAULT_ALPHA)
    return acts[final.argmax(1)]


def mf1(yy, pp):
    return float(f1_score(yy, pp, labels=actions, average="macro", zero_division=0))


def session_uniform_f1(yy, pp, ss):
    # 각 행 가중 = 1/세션행수 → sample_weight로 세션 균등
    cnt = defaultdict(int)
    for s_ in ss:
        cnt[s_] += 1
    w = np.array([1.0 / cnt[s_] for s_ in ss])
    return float(f1_score(yy, pp, labels=actions, average="macro", zero_division=0, sample_weight=w))


h12 = common.align_npz_probs(BASE_NPZ, data.ids, data.y_true, actions)
cand = common.align_npz_probs(CAND_NPZ, data.ids, data.y_true, actions)
p_base = final_pred(h12)
p_cand = final_pred(cand)

f_base, f_cand = mf1(y, p_base), mf1(y, p_cand)
d_row = f_cand - f_base
su_base = session_uniform_f1(y, p_base, sess)
su_cand = session_uniform_f1(y, p_cand, sess)

print("=" * 60)
print(f"① row Macro-F1        : {f_base:.5f} → {f_cand:.5f}  델타 {d_row:+.5f}")
print(f"② 세션균등 Macro-F1   : {su_base:.5f} → {su_cand:.5f}  델타 {su_cand - su_base:+.5f}")

# ③ 세션당 1행 MC
rng = np.random.default_rng(42)
sess_rows = defaultdict(list)
for i, s_ in enumerate(sess):
    sess_rows[s_].append(i)
groups = list(sess_rows.values())
mc = []
for _ in range(200):
    idx = np.array([g[rng.integers(len(g))] for g in groups])
    mc.append(mf1(y[idx], p_cand[idx]) - mf1(y[idx], p_base[idx]))
mc = np.array(mc)
print(f"③ 세션당1행 MC(200)   : 델타 {mc.mean():+.5f} ± {mc.std():.5f}  [min {mc.min():+.4f}, max {mc.max():+.4f}]")

# ④ paired session bootstrap
uniq = list(sess_rows.keys())
boot = []
for _ in range(2000):
    pick = rng.choice(len(uniq), size=len(uniq), replace=True)
    idx = np.concatenate([sess_rows[uniq[k]] for k in pick])
    boot.append(mf1(y[idx], p_cand[idx]) - mf1(y[idx], p_base[idx]))
boot = np.array(boot)
lo, hi = np.percentile(boot, [2.5, 97.5])
print(f"④ paired bootstrap 95%: [{lo:+.5f}, {hi:+.5f}]  P(Δ>0)={float((boot > 0).mean()):.3f}")

# ⑤ 반반
perm = np.random.RandomState(42).permutation(len(y))
h1, h2 = perm[:len(perm)//2], perm[len(perm)//2:]
d1 = mf1(y[h1], p_cand[h1]) - mf1(y[h1], p_base[h1])
d2 = mf1(y[h2], p_cand[h2]) - mf1(y[h2], p_base[h2])
print(f"⑤ 반반                : half1 {d1:+.5f} / half2 {d2:+.5f}")

ok = d_row >= GATE and lo > 0 and mc.mean() > 0
print("\n판정:", "게이트 통과 → full-train 배포 후보 (LB 게이트)" if ok else
      ("보고 (일부 조건 미달)" if d_row >= 0.002 else "미달 → 폐기"))
print("[DONE]")
