# -*- coding: utf-8 -*-
"""C(오류-상보성) + B(정규화 혼동행렬) 정찰 — lecture-gap-analysis 07-09.

무제출 정찰. honest 9969행 리그 행렬(linear/stacker OOF + e5/mBERT holdout + soft-AU)을
그대로 재사용하여:
  C: 성분 pairwise 오류상관(disagreement/double-fault/Yule Q), oracle 상한, 복구가능 오류
  B: 최고 블렌드 argmax 14x14 정규화 혼동행렬, 최대 오분류쌍, 탐색계열 교차확인,
     각 오분류쌍에서 성분별 정답률(A GBDT 입력 설계 근거)

출력: night_out/league4/diag_{complementarity,confusion}.json  (+ stdout 요약)
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

import common as C

OUT_DIR = C.OUT_DIR


def argmax_labels(probs, actions):
    return C.predict_from_probs(probs, actions)


def build_final_blend(data):
    """LB 0.7480 재현 구성: 4-way [1,1,1.2,0.8] + soft-AU α=0.9."""
    blend = C.four_way_blend(data)  # e5=1.2, mbert=0.8 기본
    au = C.train_or_load_au_probs(data)
    final = C.apply_soft_au(data, blend, au["probs"], alpha=C.DEFAULT_ALPHA)
    return final, au


def complementarity(data, comp_probs: dict[str, np.ndarray]):
    y = data.y_true
    actions = data.actions
    names = list(comp_probs.keys())
    preds = {n: argmax_labels(p, actions) for n, p in comp_probs.items()}
    correct = {n: (preds[n] == y) for n in names}

    per_comp = {}
    for n in names:
        per_comp[n] = {
            "acc": float(correct[n].mean()),
            "macro_f1": float(f1_score(y, preds[n], labels=actions, average="macro", zero_division=0)),
        }

    pairs = []
    for a, b in itertools.combinations(names, 2):
        ca, cb = correct[a], correct[b]
        # 2x2 correctness table
        n11 = int(np.sum(ca & cb))     # both right
        n00 = int(np.sum(~ca & ~cb))   # both wrong (double-fault)
        n10 = int(np.sum(ca & ~cb))
        n01 = int(np.sum(~ca & cb))
        N = len(y)
        disagreement_pred = float(np.mean(preds[a] != preds[b]))
        double_fault = n00 / N
        # Yule's Q on correctness (낮을수록 오류 독립 = 상보적)
        denom = (n11 * n00 + n01 * n10)
        q = float((n11 * n00 - n01 * n10) / denom) if denom else 0.0
        pairs.append({
            "pair": f"{a}|{b}",
            "pred_disagreement": round(disagreement_pred, 4),
            "double_fault": round(double_fault, 4),
            "both_right": round(n11 / N, 4),
            "only_a_right": round(n10 / N, 4),
            "only_b_right": round(n01 / N, 4),
            "yule_q_correctness": round(q, 4),
        })

    # oracle 상한: 각 행에서 하나라도 맞으면 맞은 것으로 처리 가능한 최대
    stacked_correct = np.vstack([correct[n] for n in names])  # (n_comp, N)
    any_right = stacked_correct.any(axis=0)
    oracle_acc = float(any_right.mean())
    # oracle macro-F1: 각 행에서 맞은 성분이 있으면 그 성분의 pred, 없으면 최빈 오답(=blend pred 대체)
    return {
        "components": names,
        "per_component": per_comp,
        "pairs": sorted(pairs, key=lambda r: r["double_fault"]),
        "oracle_at_least_one_right_acc": round(oracle_acc, 4),
        "n_rows": int(len(y)),
    }


def recoverable_errors(data, final_probs, comp_probs: dict[str, np.ndarray]):
    """블렌드는 틀렸지만 성분 중 하나 이상은 맞은 행 = 복구가능 오류."""
    y = data.y_true
    actions = data.actions
    final_pred = argmax_labels(final_probs, actions)
    final_wrong = final_pred != y
    names = list(comp_probs.keys())
    preds = {n: argmax_labels(p, actions) for n, p in comp_probs.items()}
    any_comp_right = np.zeros(len(y), dtype=bool)
    for n in names:
        any_comp_right |= (preds[n] == y)
    recoverable = final_wrong & any_comp_right
    return {
        "final_macro_f1": round(float(f1_score(y, final_pred, labels=actions, average="macro", zero_division=0)), 5),
        "final_wrong_rate": round(float(final_wrong.mean()), 4),
        "recoverable_rate": round(float(recoverable.mean()), 4),
        "recoverable_share_of_errors": round(float(recoverable.sum() / max(final_wrong.sum(), 1)), 4),
        "n_final_wrong": int(final_wrong.sum()),
        "n_recoverable": int(recoverable.sum()),
    }


def confusion(data, final_probs, comp_probs: dict[str, np.ndarray], top_k=12):
    y = data.y_true
    actions = data.actions
    a_idx = {a: i for i, a in enumerate(actions)}
    pred = argmax_labels(final_probs, actions)
    K = len(actions)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[a_idx[t], a_idx[p]] += 1
    support = cm.sum(axis=1)
    # row-normalized (recall view)
    cm_norm = cm / np.maximum(support[:, None], 1)

    # per-class P/R/F1
    per_class = {}
    for i, a in enumerate(actions):
        tp = cm[i, i]
        fn = support[i] - tp
        fp = cm[:, i].sum() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        per_class[a] = {
            "support": int(support[i]),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1": round(float(f1), 4),
        }

    # 최대 off-diagonal 오분류쌍 (count 기준 + normalized 기준)
    off = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            if cm[i, j] > 0:
                off.append({
                    "true": actions[i],
                    "pred": actions[j],
                    "count": int(cm[i, j]),
                    "row_frac": round(float(cm_norm[i, j]), 4),
                })
    off_by_count = sorted(off, key=lambda r: r["count"], reverse=True)[:top_k]

    # 각 상위 오분류쌍에서 성분별 정답률 (그 true 클래스 & 블렌드가 그 pred로 틀린 행)
    comp_names = list(comp_probs.keys())
    comp_preds = {n: argmax_labels(p, actions) for n, p in comp_probs.items()}
    for r in off_by_count:
        i, j = a_idx[r["true"]], a_idx[r["pred"]]
        mask = (np.asarray(y) == r["true"]) & (pred == r["pred"])
        r["n_confused_rows"] = int(mask.sum())
        comp_recovery = {}
        for n in comp_names:
            # 이 혼동 행들에서 성분이 true를 맞추는 비율
            comp_recovery[n] = round(float(np.mean(comp_preds[n][mask] == r["true"])) if mask.any() else 0.0, 3)
        r["component_recovers_true"] = comp_recovery

    return {
        "actions": actions,
        "per_class": per_class,
        "top_confusion_pairs": off_by_count,
        "confusion_matrix_counts": cm.tolist(),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = C.load_league_data()  # sanity assert 내장 (3-way/4-way 일치)
    print(f"[load] rows={len(data.ids)} actions={len(data.actions)} au={int(data.au_mask.sum())}")

    comp_probs = {
        "linear": data.lin,
        "stacker": data.stk,
        "e5": data.e5,
        "mbert": data.mbert,
    }
    final, au = build_final_blend(data)
    print(f"[au] cache_hit={au.get('cache_hit')} rows={len(au['ids'])}")

    comp_res = complementarity(data, comp_probs)
    rec = recoverable_errors(data, final, comp_probs)
    comp_res["recoverable"] = rec
    conf_res = confusion(data, final, comp_probs)

    C.write_json(OUT_DIR / "diag_complementarity.json", comp_res)
    C.write_json(OUT_DIR / "diag_confusion.json", conf_res)

    # ---- stdout 요약 ----
    print("\n=== C. 성분 성능 ===")
    for n, v in comp_res["per_component"].items():
        print(f"  {n:8s} acc={v['acc']:.4f} macroF1={v['macro_f1']:.4f}")
    print(f"  oracle(≥1 맞음) acc = {comp_res['oracle_at_least_one_right_acc']:.4f}")
    print("\n=== C. pairwise 오류상관 (double_fault 오름차순 = 상보적 우선) ===")
    print(f"  {'pair':20s} {'disagr':>7s} {'dblflt':>7s} {'onlyA':>6s} {'onlyB':>6s} {'yuleQ':>7s}")
    for p in comp_res["pairs"]:
        print(f"  {p['pair']:20s} {p['pred_disagreement']:7.4f} {p['double_fault']:7.4f} "
              f"{p['only_a_right']:6.4f} {p['only_b_right']:6.4f} {p['yule_q_correctness']:7.4f}")
    print(f"\n  final macroF1={rec['final_macro_f1']:.5f}  wrong={rec['n_final_wrong']} "
          f"복구가능={rec['n_recoverable']}({rec['recoverable_share_of_errors']*100:.1f}% of errors)")

    print("\n=== B. per-class F1 (약한 순) ===")
    for a, v in sorted(conf_res["per_class"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"  {a:16s} sup={v['support']:4d} P={v['precision']:.3f} R={v['recall']:.3f} F1={v['f1']:.3f}")
    print("\n=== B. 최대 오분류쌍 (count) + 성분 복구율 ===")
    for r in conf_res["top_confusion_pairs"]:
        cr = r["component_recovers_true"]
        cr_s = " ".join(f"{k}={v:.2f}" for k, v in cr.items())
        print(f"  {r['true']:14s} -> {r['pred']:14s} n={r['count']:3d} ({r['row_frac']*100:4.1f}% of true)  [{cr_s}]")


if __name__ == "__main__":
    main()
