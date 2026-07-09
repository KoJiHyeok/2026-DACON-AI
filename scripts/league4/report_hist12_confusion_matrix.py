# -*- coding: utf-8 -*-
"""hist12 4-way+soft-AU 블렌드 argmax 정식 혼동행렬 산출물.

- 14x14 정규화(행=recall) 혼동행렬
- per-class P/R/F1/support
- 최대 off-diagonal 오분류쌍 top15 (count + row_frac)
- hist12 e5 성분 단독 복구율(= e5 argmax가 정답 맞춘 비율, hist6 대비)
- hist6(현행 배포) vs hist12 오분류쌍/클래스 F1 델타 비교

산출물: context/reports/confusion_hist12_2026-07-09.md
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support

import common as C

E5_H6 = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h6.npz")
E5_H12 = Path(r"C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz")
OUT_MD = Path(r"C:\dev\2026-AI-DACON\context\reports\confusion_hist12_2026-07-09.md")


def final_pred(data, au):
    blend = C.four_way_blend(data)
    final = C.apply_soft_au(data, blend, au["probs"], C.DEFAULT_ALPHA)
    return C.predict_from_probs(final, data.actions), final


def confusion_matrix(y, pred, actions):
    a_idx = {a: i for i, a in enumerate(actions)}
    K = len(actions)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        cm[a_idx[t], a_idx[p]] += 1
    return cm


def fmt_pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def main():
    data = C.load_league_data()
    au = C.train_or_load_au_probs(data)
    actions = data.actions
    y = np.asarray(data.y_true)
    K = len(actions)

    h6 = C.align_npz_probs(E5_H6, data.ids, data.y_true, actions)
    h12 = C.align_npz_probs(E5_H12, data.ids, data.y_true, actions)

    d6 = replace(data, e5=h6)
    d12 = replace(data, e5=h12)
    pred6, final6 = final_pred(d6, au)
    pred12, final12 = final_pred(d12, au)

    f6 = C.macro_f1_labels(y, pred6, actions)
    f12 = C.macro_f1_labels(y, pred12, actions)

    cm6 = confusion_matrix(y, pred6, actions)
    cm12 = confusion_matrix(y, pred12, actions)

    # normalized (row = recall) confusion matrix for hist12
    row_sum12 = cm12.sum(axis=1, keepdims=True).astype(np.float64)
    row_sum12[row_sum12 == 0] = 1.0
    cm12_norm = cm12 / row_sum12

    # per-class P/R/F1/support for hist12
    p12, r12, f1_12, sup12 = precision_recall_fscore_support(
        y, pred12, labels=actions, average=None, zero_division=0
    )
    p6, r6, f1_6, sup6 = precision_recall_fscore_support(
        y, pred6, labels=actions, average=None, zero_division=0
    )

    # top15 off-diagonal misclass pairs by count (hist12)
    a_idx = {a: i for i, a in enumerate(actions)}
    off_pairs = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm12[i, j])
            if c > 0:
                row_frac = c / max(1, int(cm12[i].sum()))
                off_pairs.append((c, row_frac, actions[i], actions[j]))
    off_pairs.sort(key=lambda t: (-t[0], -t[1]))
    top15 = off_pairs[:15]

    # e5-only recovery rate (hist12 vs hist6): does e5 argmax alone match y_true?
    e5_pred6 = C.predict_from_probs(h6, actions)
    e5_pred12 = C.predict_from_probs(h12, actions)
    e5_acc6 = float((e5_pred6 == y).mean())
    e5_acc12 = float((e5_pred12 == y).mean())
    e5_f1_6 = C.macro_f1_labels(y, e5_pred6, actions)
    e5_f1_12 = C.macro_f1_labels(y, e5_pred12, actions)

    # per-class comparison hist6 vs hist12 (weakest classes by hist12 F1)
    class_rows = []
    for i, a in enumerate(actions):
        class_rows.append({
            "action": a,
            "support": int(sup12[i]),
            "p6": p6[i], "r6": r6[i], "f1_6": f1_6[i],
            "p12": p12[i], "r12": r12[i], "f1_12": f1_12[i],
            "delta_f1": f1_12[i] - f1_6[i],
        })
    weakest = sorted(class_rows, key=lambda r: r["f1_12"])[:5]

    # pair-level delta hist6 -> hist12 for the union of top pairs from both
    off_pairs6 = []
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            c = int(cm6[i, j])
            if c > 0:
                off_pairs6.append((c, actions[i], actions[j]))
    off_pairs6.sort(key=lambda t: -t[0])
    top_pair_keys = set((tr, pr) for _, tr, pr in off_pairs6[:15]) | set((tr, pr) for _, _, tr, pr in top15)
    pair_delta_rows = []
    for tr, pr in top_pair_keys:
        i, j = a_idx[tr], a_idx[pr]
        c6, c12 = int(cm6[i, j]), int(cm12[i, j])
        pair_delta_rows.append((tr, pr, c6, c12, c12 - c6))
    pair_delta_rows.sort(key=lambda t: t[3])  # most reduced first (most negative delta)

    # ---------------- write markdown ----------------
    lines = []
    lines.append("# hist12 혼동행렬 정식 산출물 (2026-07-09)")
    lines.append("")
    lines.append("리그 프레임: `scripts/league4/common.py` honest 9969행 (linear/stacker OOF + e5/mBERT holdout + AU 라우팅), "
                  "4-way+soft-AU 블렌드(alpha=0.9), e5 슬롯만 hist6↔hist12 스왑.")
    lines.append("")
    lines.append(f"- hist6(현행 배포) 블렌드 macro-F1 = **{f6:.5f}**")
    lines.append(f"- hist12 블렌드 macro-F1 = **{f12:.5f}** (델타 {f12-f6:+.5f})")
    lines.append(f"- baseline 참고값(4-way+soft-AU) = 0.75601")
    lines.append("")
    lines.append("## e5 성분 단독 복구율 (hist6 vs hist12)")
    lines.append("")
    lines.append("| 지표 | hist6 e5 | hist12 e5 | 델타 |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| e5 argmax accuracy | {fmt_pct(e5_acc6)} | {fmt_pct(e5_acc12)} | {fmt_pct(e5_acc12-e5_acc6)} |")
    lines.append(f"| e5 argmax macro-F1 | {e5_f1_6:.4f} | {e5_f1_12:.4f} | {e5_f1_12-e5_f1_6:+.4f} |")
    lines.append("")
    lines.append("## per-class P/R/F1/support (hist12 블렌드)")
    lines.append("")
    lines.append("| action | support | precision | recall | F1 | F1(hist6) | 델타F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in sorted(class_rows, key=lambda r: -r["support"]):
        lines.append(
            f"| {row['action']} | {row['support']} | {row['p12']:.3f} | {row['r12']:.3f} | "
            f"{row['f1_12']:.4f} | {row['f1_6']:.4f} | {row['delta_f1']:+.4f} |"
        )
    lines.append("")
    lines.append("## 가장 약한 클래스 top5 (hist12 F1 기준)")
    lines.append("")
    for row in weakest:
        lines.append(f"- **{row['action']}**: F1={row['f1_12']:.4f} (support={row['support']}, "
                      f"hist6 대비 {row['delta_f1']:+.4f})")
    lines.append("")
    lines.append("## 최대 오분류쌍 top15 (hist12, count 기준)")
    lines.append("")
    lines.append("| true → pred | count | row_frac(정답 행 내 비율) |")
    lines.append("|---|---:|---:|")
    for c, row_frac, tr, pr in top15:
        lines.append(f"| {tr} → {pr} | {c} | {fmt_pct(row_frac)} |")
    lines.append("")
    lines.append("## hist6 → hist12 오분류쌍 델타 (개선/악화 상위, count 기준)")
    lines.append("")
    lines.append("| true → pred | hist6 count | hist12 count | 델타 |")
    lines.append("|---|---:|---:|---:|")
    for tr, pr, c6, c12, d in pair_delta_rows:
        lines.append(f"| {tr} → {pr} | {c6} | {c12} | {d:+d} |")
    lines.append("")
    lines.append("## 14x14 정규화 혼동행렬 (행=recall, hist12)")
    lines.append("")
    header = "| true\\pred | " + " | ".join(actions) + " |"
    sep = "|---|" + "|".join(["---:"] * K) + "|"
    lines.append(header)
    lines.append(sep)
    for i, a in enumerate(actions):
        row_cells = " | ".join(f"{cm12_norm[i,j]*100:.0f}%" if cm12_norm[i, j] > 0 else "" for j in range(K))
        lines.append(f"| **{a}** | {row_cells} |")
    lines.append("")
    lines.append("## 요약: hist6 대비 hist12에서 무엇이 바뀌었나")
    lines.append("")
    improved = [r for r in pair_delta_rows if r[4] < 0]
    worsened = [r for r in pair_delta_rows if r[4] > 0]
    lines.append(f"- 오분류 감소 쌍: {len(improved)}개 (합계 델타 {sum(r[4] for r in improved):+d}건)")
    lines.append(f"- 오분류 증가 쌍: {len(worsened)}개 (합계 델타 {sum(r[4] for r in worsened):+d}건)")
    if improved:
        top_improved = improved[:3]
        lines.append("- 가장 크게 준 쌍: " + ", ".join(f"{tr}→{pr}({d:+d})" for tr, pr, _, _, d in top_improved))
    if worsened:
        top_worsened = sorted(worsened, key=lambda r: -r[4])[:3]
        lines.append("- 가장 크게 는 쌍: " + ", ".join(f"{tr}→{pr}({d:+d})" for tr, pr, _, _, d in top_worsened))
    class_delta_sorted = sorted(class_rows, key=lambda r: -r["delta_f1"])
    up = [r for r in class_delta_sorted if r["delta_f1"] > 0.001]
    down = [r for r in class_delta_sorted if r["delta_f1"] < -0.001]
    lines.append(f"- F1 상승 클래스({len(up)}개): " + ", ".join(f"{r['action']}({r['delta_f1']:+.4f})" for r in up) if up else "- F1 상승 클래스: 없음(임계 0.001)")
    lines.append(f"- F1 하락 클래스({len(down)}개): " + ", ".join(f"{r['action']}({r['delta_f1']:+.4f})" for r in down) if down else "- F1 하락 클래스: 없음(임계 0.001)")
    lines.append("")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {OUT_MD}")

    # console summary for structured return
    print(f"\nmacro-F1 hist6={f6:.5f} hist12={f12:.5f} delta={f12-f6:+.5f}")
    print("weakest classes:", [r["action"] for r in weakest])
    print("top pairs (hist12):", [(tr, pr, c) for c, _, tr, pr in top15[:5]])


if __name__ == "__main__":
    main()
