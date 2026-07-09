# -*- coding: utf-8 -*-
"""라우터 확장 정찰 — 추론 가능 필드 슬라이스 전수 스캔 (AU 패턴 후속).

AU 라우팅(+0.014)이 통한 조건: ① 추론 시 결정적 식별 ② 슬라이스에서 블렌드 체계적 약함
③ 특화가 크게 이김. exp #28의 id-prefix 스캔(23그룹, 소진)과 달리 여기선
session_meta/workspace 필드 슬라이스를 전수로 훑어 '스코어 차 역추적' — 각 슬라이스에서
blend F1 vs best-single-component F1 vs oracle 갭을 support 가중으로 랭크한다.

낮은 blend F1 × 큰 support × (성분 or oracle 여유 큼) = AU 같은 특화 후보.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

import common as C


def blend_final(data):
    blend = C.four_way_blend(data)
    au = C.train_or_load_au_probs(data)
    return C.apply_soft_au(data, blend, au["probs"], C.DEFAULT_ALPHA)


def bucket_num(v, edges):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"<{e}"
    return f">={edges[-1]}"


def primary_lang(ws):
    lm = ws.get("language_mix")
    if isinstance(lm, dict) and lm:
        return max(lm.items(), key=lambda kv: kv[1])[0]
    return "na"


def slice_values(sample):
    """추론 시 결정적으로 뽑히는 슬라이스 키 → 값 딕셔너리."""
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    hist = sample.get("history") or []
    return {
        "user_tier": str(sm.get("user_tier", "na")),
        "language_pref": str(sm.get("language_pref", "na")),
        "primary_lang": primary_lang(ws),
        "last_ci_status": str(ws.get("last_ci_status", "na")),
        "git_dirty": str(bool(ws.get("git_dirty"))),
        "turn_bucket": bucket_num(sm.get("turn_index"), (1, 4, 9)),
        "hist_bucket": ("0" if len(hist) == 0 else "1-3" if len(hist) <= 3 else "4-6" if len(hist) <= 6 else "7-12"),
        "n_open_files": ("0" if not ws.get("open_files") else str(min(len(ws.get("open_files")), 4))),
        "budget_bucket": bucket_num(sm.get("budget_tokens_remaining"), (10000, 50000, 100000)),
        "loc_bucket": bucket_num(ws.get("loc"), (1000, 10000, 50000)),
    }


def mf1(y, pred, actions):
    return float(f1_score(y, pred, labels=actions, average="macro", zero_division=0))


def main():
    data = C.load_league_data()
    actions = data.actions
    y = np.asarray(data.y_true)
    final = blend_final(data)
    blend_pred = C.predict_from_probs(final, actions)

    comp_probs = {"linear": data.lin, "stacker": data.stk, "e5": data.e5, "mbert": data.mbert}
    comp_pred = {n: C.predict_from_probs(p, actions) for n, p in comp_probs.items()}
    correct = {n: (comp_pred[n] == y) for n in comp_probs}
    any_right = np.any(np.vstack([correct[n] for n in comp_probs]), axis=0)

    # 각 행의 슬라이스 값 미리 계산
    rows_slices = [slice_values(data.samples_by_id[str(i)]) for i in data.ids]
    keys = list(rows_slices[0].keys())

    overall = mf1(y, blend_pred, actions)
    print(f"[전체] blend macroF1={overall:.5f}  oracle acc={any_right.mean():.4f}  n={len(y)}\n")

    candidates = []
    for key in keys:
        vals = sorted(set(rs[key] for rs in rows_slices))
        for v in vals:
            mask = np.array([rs[key] == v for rs in rows_slices], dtype=bool)
            n = int(mask.sum())
            if n < 150:   # support 너무 작으면 라우팅 무의미 (노이즈)
                continue
            ys = y[mask]
            b = mf1(ys, blend_pred[mask], actions)
            # 슬라이스에서 각 성분 단독 F1 (특화 대신 성분 라우팅 상한)
            comp_f1 = {n_: mf1(ys, comp_pred[n_][mask], actions) for n_ in comp_probs}
            best_comp = max(comp_f1.items(), key=lambda kv: kv[1])
            oracle_acc = float(any_right[mask].mean())
            blend_acc = float((blend_pred[mask] == ys).mean())
            candidates.append({
                "slice": f"{key}={v}",
                "n": n,
                "blend_f1": round(b, 4),
                "best_comp": best_comp[0],
                "best_comp_f1": round(best_comp[1], 4),
                "comp_gain": round(best_comp[1] - b, 4),          # 성분 라우팅 여유
                "blend_acc": round(blend_acc, 4),
                "oracle_acc": round(oracle_acc, 4),
                "oracle_gap": round(oracle_acc - blend_acc, 4),   # 특화 상한 여유
                "impact": round((oracle_acc - blend_acc) * n, 1),  # 특화로 딸 수 있는 총 정답질량
            })

    print("=== 블렌드가 약한 슬라이스 (blend_f1 오름차순, support>=150) ===")
    print(f"{'slice':32s} {'n':>5s} {'bF1':>6s} {'bAcc':>6s} {'oracle':>7s} {'gap':>6s} {'impact':>7s} bestComp")
    for c in sorted(candidates, key=lambda r: r["blend_f1"])[:18]:
        print(f"{c['slice']:32s} {c['n']:5d} {c['blend_f1']:6.3f} {c['blend_acc']:6.3f} "
              f"{c['oracle_acc']:7.3f} {c['oracle_gap']:6.3f} {c['impact']:7.1f}  "
              f"{c['best_comp']}({c['best_comp_f1']:.3f} {c['comp_gain']:+.3f})")

    print("\n=== 성분 라우팅 여유 큰 슬라이스 (comp_gain 내림차순) — 단일 성분 스왑이 블렌드 이기는 곳 ===")
    print(f"{'slice':32s} {'n':>5s} {'bF1':>6s} {'best':>10s} {'compF1':>7s} {'gain':>6s}")
    for c in sorted(candidates, key=lambda r: r["comp_gain"], reverse=True)[:12]:
        print(f"{c['slice']:32s} {c['n']:5d} {c['blend_f1']:6.3f} {c['best_comp']:>10s} "
              f"{c['best_comp_f1']:7.3f} {c['comp_gain']:+6.3f}")

    print("\n=== 특화 상한 impact 큰 슬라이스 (oracle_gap*n) — AU 같은 전용학습 후보 ===")
    print(f"{'slice':32s} {'n':>5s} {'bAcc':>6s} {'oracle':>7s} {'gap':>6s} {'impact':>7s}")
    for c in sorted(candidates, key=lambda r: r["impact"], reverse=True)[:12]:
        print(f"{c['slice']:32s} {c['n']:5d} {c['blend_acc']:6.3f} {c['oracle_acc']:7.3f} "
              f"{c['oracle_gap']:6.3f} {c['impact']:7.1f}")


if __name__ == "__main__":
    main()
