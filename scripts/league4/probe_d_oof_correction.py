# -*- coding: utf-8 -*-
"""probe D: OOF 기반 국소 보정 (exp #46 후속 — 동료 입력 없이 우리 자산만).

가설: #46에서 스태커는 전역 대체로는 −0.0054 열세지만 list_directory/lint/web_search를
  실제로 고쳤다(+367/+84/+29행). 보정 모델의 고신뢰 탐색계열 예측만 champion 표면 위에
  국소 override하면 손해(read_file/ask_user 붕괴) 없이 이득만 취할 수 있는가.

대조군 = probe_c와 동일한 champion 표면: e5 슬롯 hist12 스왑 4-way + soft-AU α=0.9.
보정 모델 = LogisticRegression(45열: e5 OOF 14 + mBERT OOF 14 + linear OOF 14 + 엔트로피 3)
  train = 비홀드아웃 60,031행 (홀드아웃 세션 그룹 완전 제외 — assert)
  infer = 홀드아웃 9,969행 (OOF 확률 = 그 행을 학습하지 않은 fold 모델 출력 — 정직)

변형:
  (a) full        — corrector로 전체 교체 (대조군, #46 재현 기대: 실패)
  (b) explore-τ   — corrector argmax ∈ EXPLORE & 신뢰도 ≥ τ & 비-AU 행만 override
                    EXPLORE = {list_directory, lint_or_typecheck, web_search, glob_pattern}
                    (#46 양(+) 클래스만; grep_search는 −177행이라 제외)
판정: 변형별 5지표 (①row ≥ +0.005 & ④CI 하한>0 & ③MC>0 게이트 — D-011과 동일).
"""
import os
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import common

ROOT = Path(r"C:\dev\2026-AI-DACON")
BASE_NPZ = Path(os.environ.get("E5_H12", str(ROOT / "colab_out" / "holdout_e5_h12.npz")))
E5_OOF_DIR = ROOT / "artifacts" / "experiments" / "oof_h12"
MBERT_OOF_DIR = ROOT / "artifacts" / "experiments" / "oof_mbert_h6"
GATE = 0.005
EXPLORE = ("list_directory", "lint_or_typecheck", "web_search", "glob_pattern")
TAUS = (0.5, 0.6, 0.7)

data = common.load_league_data()
au = common.train_or_load_au_probs(data)
actions = data.actions
acts = np.array(actions, dtype=object)
y = np.asarray(data.y_true)
sess = np.array([common.session_id(str(i)) for i in data.ids])


def load_fold_oof(oof_dir: Path, pattern: str) -> tuple[dict, list]:
    """fold npz 5개 → id→prob행 dict (actions 순서는 리그 순서로 정렬)."""
    by_id = {}
    for k in range(5):
        z = np.load(oof_dir / pattern.format(k), allow_pickle=True)
        src_actions = [str(x) for x in z["actions"]]
        col = [src_actions.index(str(a)) for a in actions]
        probs = np.asarray(z["probs"], dtype=np.float64)[:, col]
        for i, sid in enumerate(z["ids"]):
            sid = str(sid)
            assert sid not in by_id, f"중복 id {sid} ({oof_dir.name})"
            by_id[sid] = probs[i]
    assert len(by_id) == 70000, f"{oof_dir.name} 합계 {len(by_id)} != 70000"
    return by_id


def entropy(p: np.ndarray) -> np.ndarray:
    return -(p * np.log(np.clip(p, 1e-12, None))).sum(axis=1, keepdims=True)


print("[load] e5 / mBERT fold OOF ...")
e5_oof = load_fold_oof(E5_OOF_DIR, "oof_fold{}.npz")
mb_oof = load_fold_oof(MBERT_OOF_DIR, "oof_mbert_fold{}.npz")

# linear OOF: 리그 canonical 디렉토리에서 전체 70k 정렬
all_ids = [str(i) for i in data.train_ids]
lin_all, _ = common.load_oof_probs(common.OOF_DIR, all_ids, actions)
lin_by_id = {sid: lin_all[i] for i, sid in enumerate(all_ids)}


def feat(ids_seq) -> np.ndarray:
    e = np.stack([e5_oof[str(i)] for i in ids_seq])
    m = np.stack([mb_oof[str(i)] for i in ids_seq])
    l = np.stack([lin_by_id[str(i)] for i in ids_seq])
    return np.hstack([e, m, l, entropy(e), entropy(m), entropy(l)])


# ----- 보정 모델 학습 (홀드아웃 세션 완전 제외) -----
hold_ids = set(str(i) for i in data.ids)
hold_sess = set(sess)
tr_idx = [i for i, sid in enumerate(all_ids) if sid not in hold_ids]
tr_sess = {common.session_id(all_ids[i]) for i in tr_idx}
assert not (tr_sess & hold_sess), f"홀드아웃 세션 {len(tr_sess & hold_sess)}개가 train에 누수"
X_tr = feat([all_ids[i] for i in tr_idx])
y_tr = np.asarray([str(data.train_y[i]) for i in tr_idx])
print(f"[train] corrector rows={len(tr_idx)} feats={X_tr.shape[1]}")
clf = LogisticRegression(C=1.0, max_iter=3000, random_state=42)
clf.fit(X_tr, y_tr)

X_ho = feat(data.ids)
corr_probs = clf.predict_proba(X_ho)
cls_order = [list(clf.classes_).index(str(a)) for a in actions]
corr_probs = corr_probs[:, cls_order]
corr_pred = acts[corr_probs.argmax(1)]
corr_conf = corr_probs.max(1)

# ----- champion 표면 (probe_c와 동일) -----
h12 = common.align_npz_probs(BASE_NPZ, data.ids, data.y_true, actions)
d_champ = replace(data, e5=h12)
final = common.apply_soft_au(d_champ, common.four_way_blend(d_champ), au["probs"], common.DEFAULT_ALPHA)
p_base = acts[final.argmax(1)]


def mf1(yy, pp):
    return float(f1_score(yy, pp, labels=actions, average="macro", zero_division=0))


def session_uniform_f1(yy, pp, ss):
    cnt = defaultdict(int)
    for s_ in ss:
        cnt[s_] += 1
    w = np.array([1.0 / cnt[s_] for s_ in ss])
    return float(f1_score(yy, pp, labels=actions, average="macro", zero_division=0, sample_weight=w))


sess_rows = defaultdict(list)
for i, s_ in enumerate(sess):
    sess_rows[s_].append(i)
groups = list(sess_rows.values())
uniq = list(sess_rows.keys())
f_base = mf1(y, p_base)
su_base = session_uniform_f1(y, p_base, sess)


def judge(name: str, p_cand: np.ndarray, n_override: int):
    rng = np.random.default_rng(42)
    f_cand = mf1(y, p_cand)
    d_row = f_cand - f_base
    su_cand = session_uniform_f1(y, p_cand, sess)
    mc = []
    for _ in range(200):
        idx = np.array([g[rng.integers(len(g))] for g in groups])
        mc.append(mf1(y[idx], p_cand[idx]) - mf1(y[idx], p_base[idx]))
    mc = np.array(mc)
    boot = []
    for _ in range(2000):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        idx = np.concatenate([sess_rows[uniq[k]] for k in pick])
        boot.append(mf1(y[idx], p_cand[idx]) - mf1(y[idx], p_base[idx]))
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    perm = np.random.RandomState(42).permutation(len(y))
    h1, h2 = perm[:len(perm) // 2], perm[len(perm) // 2:]
    d1 = mf1(y[h1], p_cand[h1]) - mf1(y[h1], p_base[h1])
    d2 = mf1(y[h2], p_cand[h2]) - mf1(y[h2], p_base[h2])
    flips = int((p_cand != p_base).sum())
    fixed = int(((p_cand == y) & (p_base != y)).sum())
    broke = int(((p_cand != y) & (p_base == y)).sum())
    print("=" * 60)
    print(f"[{name}] override={n_override} flips={flips} (고침 {fixed} / 망침 {broke})")
    print(f"① row Macro-F1        : {f_base:.5f} → {f_cand:.5f}  델타 {d_row:+.5f}")
    print(f"② 세션균등 Macro-F1   : {su_base:.5f} → {su_cand:.5f}  델타 {su_cand - su_base:+.5f}")
    print(f"③ 세션당1행 MC(200)   : 델타 {mc.mean():+.5f} ± {mc.std():.5f}")
    print(f"④ paired bootstrap 95%: [{lo:+.5f}, {hi:+.5f}]  P(Δ>0)={float((boot > 0).mean()):.3f}")
    print(f"⑤ 반반                : half1 {d1:+.5f} / half2 {d2:+.5f}")
    ok = d_row >= GATE and lo > 0 and mc.mean() > 0
    print("판정:", "게이트 통과" if ok else ("보고 (문턱 0.002 이상)" if d_row >= 0.002 else "미달"))
    return d_row


# (a) 전체 교체 — #46 재현 대조군
judge("a: full replace", corr_pred.copy(), len(y))

# (b) explore-τ 국소 override (비-AU 행 한정)
explore_mask = np.isin(corr_pred, EXPLORE)
for tau in TAUS:
    mask = explore_mask & (corr_conf >= tau) & data.non_au_mask
    p_cand = p_base.copy()
    p_cand[mask] = corr_pred[mask]
    judge(f"b: explore tau={tau}", p_cand, int(mask.sum()))

print("[DONE]")
