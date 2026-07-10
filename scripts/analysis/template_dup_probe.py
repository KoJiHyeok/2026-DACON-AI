# -*- coding: utf-8 -*-
"""템플릿(current_prompt) 완전일치 라우팅 카드 판정 — G4 (deep_research_gap_check_2026-07-10.md).

가설: train에 current_prompt가 완전 일치하는 템플릿 그룹이 존재한다. 홀드아웃 행의
current_prompt가 (홀드아웃 세션을 제외한) train측 템플릿과 완전 일치하면 그 템플릿의
train측 다수 라벨로 챔피언 예측을 오버라이드하는 규칙이 챔피언 블렌드를 능가하는지 측정.

챔피언 재현: scripts/league4/common.py의 4-way 블렌드 로직을 그대로 재사용한다.
배포판 구성(exp #35 승격, exp #37로 재확인된 기록값 0.756006)은:
  - e5 슬롯 = colab_out/holdout_e5_h12.npz (hist12, league4/common.py의 HOLDOUT_BASE는
    구식 hist6 e5라서 그대로 쓰면 안 됨 — probe_b_mbert_hist12.py / diag_hist12_confusion_delta.py
    가 쓰는 것과 동일하게 dataclasses.replace(data, e5=h12)로 교체한다)
  - mBERT 슬롯 = colab_out/holdout_mbert.npz (hist6 그대로 — mbert_h12는 Bet B 후보일 뿐
    아직 배포되지 않았고 실제 npz 파일도 저장소에 없음, exp #37 FAIL/유지)
  - 블록 가중 [e5=1.2, mbert=0.8], soft-AU alpha=0.9 (common.DEFAULT_ALPHA)
블렌드/soft-AU 계산 자체는 재발명하지 않고 scripts/league4/common.py의
four_way_blend / apply_soft_au / train_or_load_au_probs / macro_f1_probs를 그대로 호출한다.

세션 그룹키: id에서 정규식 `-step_\d+$` 제거 (league4.common.session_id와 동일 로직).

이 스크립트는 npz/데이터 파일을 읽기만 하고, submit/ 이하는 건드리지 않는다.
출력은 전부 콘솔 print + scratchpad(또는 --out-dir)의 CSV/JSON.

실행:
    cd scripts/analysis
    ..\..\.venv\Scripts\python.exe template_dup_probe.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[2]
LEAGUE4_DIR = ROOT / "scripts" / "league4"
sys.path.insert(0, str(LEAGUE4_DIR))
import common as league  # noqa: E402  (scripts/league4/common.py — champion blend logic)

DATA_DIR = ROOT / "data"
TRAIN_JSONL = DATA_DIR / "train.jsonl"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
E5_H12 = ROOT / "colab_out" / "holdout_e5_h12.npz"

DEFAULT_OUT_DIR = ROOT / "scripts" / "analysis" / "_out"
SCRATCH_DIR = Path(
    r"C:\Users\wlgur\AppData\Local\Temp\claude\C--dev-2026-AI-DACON"
    r"\8281d6d2-8149-40b5-a895-bee094e9df88\scratchpad"
)

ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern", "edit_file",
    "write_file", "apply_patch", "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

STEP_RE = re.compile(r"-step_\d+$")

GRIDS = {
    "A_purity0.90_n5": dict(min_purity=0.90, min_n=5),
    "B_purity0.95_n10": dict(min_purity=0.95, min_n=10),
    "C_purity0.99_n3": dict(min_purity=0.99, min_n=3),
}

N_MC = 50
MC_SEED0 = 42
OVERRIDE_GATE = 0.005


def session_id(sample_id: str) -> str:
    return STEP_RE.sub("", str(sample_id))


def load_all_prompts(path: Path = TRAIN_JSONL) -> pd.DataFrame:
    """id, current_prompt (trimmed only) for all 70,000 train rows."""
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.append({
                "id": str(obj["id"]),
                "current_prompt": str(obj.get("current_prompt") or "").strip(),
            })
    df = pd.DataFrame(rows)
    df["session_key"] = df["id"].map(session_id)
    return df


def load_labels(path: Path = TRAIN_LABELS) -> dict[str, str]:
    labels = pd.read_csv(path)
    return {str(r.id): str(r.action) for r in labels.itertuples()}


def build_champion_probs() -> tuple[league.LeagueData, np.ndarray]:
    """Reproduce the deployed champion blend (e5=hist12, mbert=hist6, [1.2,0.8], soft-AU 0.9).

    Returns (league_data with e5 swapped to hist12, final soft-AU blended probs).
    """
    data0 = league.load_league_data()  # sanity-checks 3-way/4-way vs EXPECTED_* internally
    au = league.train_or_load_au_probs(data0)
    h12 = league.align_npz_probs(E5_H12, data0.ids, data0.y_true, data0.actions)
    data = replace(data0, e5=h12)
    blend = league.four_way_blend(data, league.BASE_E5_WEIGHT, league.BASE_MBERT_WEIGHT)
    final = league.apply_soft_au(data, blend, au["probs"], league.DEFAULT_ALPHA)
    return data, final


def group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per current_prompt group: n_rows, n_sessions, majority label, purity."""
    recs = []
    for key, sub in df.groupby("current_prompt"):
        vc = sub["action"].value_counts()
        recs.append({
            "current_prompt": key,
            "n_rows": len(sub),
            "n_sessions": sub["session_key"].nunique(),
            "majority_label": vc.index[0],
            "purity": float(vc.iloc[0] / len(sub)),
        })
    return pd.DataFrame(recs)


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=ACTIONS, average="macro", zero_division=0))


def error_breakdown(y_true: np.ndarray, y_pred: np.ndarray, top_n: int = 6) -> list[dict[str, Any]]:
    wrong = y_true != y_pred
    pairs = Counter(zip(y_true[wrong].tolist(), y_pred[wrong].tolist()))
    rows = []
    for (t, p), c in pairs.most_common(top_n):
        rows.append({"true": t, "pred": p, "count": int(c)})
    return rows


def run_override_simulation(
    holdout_pred: np.ndarray,
    holdout_override_label: np.ndarray,  # "" if no override
    y_true: np.ndarray,
    session_keys: np.ndarray,
    n_mc: int = N_MC,
    seed0: int = MC_SEED0,
) -> dict[str, Any]:
    has_override = holdout_override_label != ""
    overridden_pred = np.where(has_override, holdout_override_label, holdout_pred)

    row_baseline = macro_f1(y_true, holdout_pred)
    row_override = macro_f1(y_true, overridden_pred)
    row_delta = row_override - row_baseline

    # Monte Carlo: one row per session, seeds 42..42+n_mc-1
    unique_sessions = np.unique(session_keys)
    sess_to_rows: dict[str, np.ndarray] = {
        s: np.where(session_keys == s)[0] for s in unique_sessions
    }
    mc_deltas = []
    for i in range(n_mc):
        rng = np.random.RandomState(seed0 + i)
        sampled_idx = np.array([
            rng.choice(sess_to_rows[s]) for s in unique_sessions
        ])
        yt = y_true[sampled_idx]
        base_p = holdout_pred[sampled_idx]
        ovr_p = overridden_pred[sampled_idx]
        mc_deltas.append(macro_f1(yt, ovr_p) - macro_f1(yt, base_p))
    mc_deltas = np.asarray(mc_deltas, dtype=np.float64)

    return {
        "n_override_rows": int(has_override.sum()),
        "row_baseline_macro_f1": row_baseline,
        "row_override_macro_f1": row_override,
        "row_delta": row_delta,
        "mc_mean_delta": float(mc_deltas.mean()),
        "mc_std_delta": float(mc_deltas.std(ddof=1)),
        "mc_min_delta": float(mc_deltas.min()),
        "mc_max_delta": float(mc_deltas.max()),
        "mc_n": n_mc,
    }


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ---- 1. Champion reproduction on the fixed holdout (9,969 rows / 1,350 sessions) ----
    print("=" * 70)
    print("[1] 챔피언 블렌드 재현 (e5=hist12, mbert=hist6, [1.2,0.8], soft-AU a=0.9)")
    data, champ_probs = build_champion_probs()
    champ_pred = league.predict_from_probs(champ_probs, data.actions)
    y_true = np.asarray(data.y_true, dtype=object)
    champ_macro_f1 = macro_f1(y_true, champ_pred)
    record_value = 0.756006
    print(f"  홀드아웃 행수={len(data.ids)}, 세션수={len(set(data.train_groups)) if False else '-'}")
    print(f"  champion macro_f1 = {champ_macro_f1:.6f}  (기록값 {record_value:.6f}, "
          f"delta={champ_macro_f1 - record_value:+.6f})")

    holdout_sessions = sorted(set(session_id(str(x)) for x in data.ids))
    print(f"  홀드아웃 세션수 = {len(holdout_sessions)} (예상 1350)")

    # ---- 2. Load all train rows + normalized (trim-only) current_prompt ----
    print("\n" + "=" * 70)
    print("[2] train.jsonl 전량 로드 + current_prompt 정규화(트림만)")
    df_all = load_all_prompts()
    labels = load_labels()
    df_all["action"] = df_all["id"].map(labels)
    missing_label = df_all["action"].isna().sum()
    if missing_label:
        raise AssertionError(f"{missing_label} rows missing labels")
    print(f"  전체 행수={len(df_all)}, 유니크 current_prompt={df_all['current_prompt'].nunique()}")

    holdout_session_set = set(holdout_sessions)
    df_train_side = df_all[~df_all["session_key"].isin(holdout_session_set)].copy()
    df_holdout_side = df_all[df_all["id"].isin(set(str(x) for x in data.ids))].copy()
    print(f"  오버라이드 소스(train측, 홀드아웃 세션 제외) 행수={len(df_train_side)}")
    print(f"  홀드아웃 측 매칭 대상 행수={len(df_holdout_side)} (기대 9969)")
    if len(df_holdout_side) != len(data.ids):
        raise AssertionError("holdout id join mismatch")

    # sanity: no holdout session leaked into override source
    leaked = set(df_train_side["session_key"]) & holdout_session_set
    if leaked:
        raise AssertionError(f"leak: {len(leaked)} holdout sessions present in override source")

    # ---- 3. Group train-side rows by exact current_prompt ----
    print("\n" + "=" * 70)
    print("[3] train측(홀드아웃 세션 제외) current_prompt 완전일치 그룹핑")
    groups = group_stats(df_train_side)
    groups = groups.set_index("current_prompt")
    print(f"  train측 유니크 current_prompt 그룹수={len(groups)}")

    # align holdout rows to (data.ids order) for probs/pred consistency
    holdout_id_order = [str(x) for x in data.ids]
    df_holdout_side = df_holdout_side.set_index("id").loc[holdout_id_order].reset_index()
    holdout_prompt = df_holdout_side["current_prompt"].to_numpy()
    holdout_session_keys = df_holdout_side["session_key"].to_numpy()

    matched_mask = np.asarray([p in groups.index for p in holdout_prompt], dtype=bool)
    coverage_pct = 100.0 * matched_mask.sum() / len(matched_mask)
    print(f"  템플릿 커버리지(홀드아웃 행 중 train측 완전일치 그룹 존재 비율) = {coverage_pct:.2f}% "
          f"({matched_mask.sum()}/{len(matched_mask)})")

    # ---- 4a. Coverage + template vs non-template champion error breakdown ----
    print("\n" + "=" * 70)
    print("[4a] 템플릿 행 vs 비템플릿 행 챔피언 오류율")
    tmpl_mask = matched_mask
    non_tmpl_mask = ~matched_mask

    def acc_err(mask: np.ndarray) -> dict[str, Any]:
        yt = y_true[mask]
        yp = champ_pred[mask]
        acc = float((yt == yp).mean()) if mask.sum() else float("nan")
        return {
            "n_rows": int(mask.sum()),
            "accuracy": acc,
            "error_rate": 1.0 - acc if mask.sum() else float("nan"),
            "macro_f1": macro_f1(yt, yp) if mask.sum() else float("nan"),
            "top_errors": error_breakdown(yt, yp),
        }

    tmpl_stats = acc_err(tmpl_mask)
    non_tmpl_stats = acc_err(non_tmpl_mask)
    print(f"  템플릿 행:    n={tmpl_stats['n_rows']:5d}  acc={tmpl_stats['accuracy']:.4f}  "
          f"err={tmpl_stats['error_rate']:.4f}  macro_f1={tmpl_stats['macro_f1']:.4f}")
    print(f"    상위 오류쌍: {tmpl_stats['top_errors']}")
    print(f"  비템플릿 행:  n={non_tmpl_stats['n_rows']:5d}  acc={non_tmpl_stats['accuracy']:.4f}  "
          f"err={non_tmpl_stats['error_rate']:.4f}  macro_f1={non_tmpl_stats['macro_f1']:.4f}")
    print(f"    상위 오류쌍: {non_tmpl_stats['top_errors']}")

    # ---- 4b/4c/4d. Grid-based majority-label accuracy + override simulation ----
    print("\n" + "=" * 70)
    print("[4b-d] 그리드별 다수라벨 정확도 + 오버라이드 시뮬레이션")

    grid_rows = []
    override_rows = []
    for grid_name, cond in GRIDS.items():
        min_purity, min_n = cond["min_purity"], cond["min_n"]
        eligible_groups = groups[(groups["purity"] >= min_purity) & (groups["n_rows"] >= min_n)]
        eligible_prompt_set = set(eligible_groups.index)

        elig_mask = np.asarray([p in eligible_prompt_set for p in holdout_prompt], dtype=bool)
        n_elig = int(elig_mask.sum())
        elig_cov_pct = 100.0 * n_elig / len(elig_mask)

        # majority-label-as-prediction accuracy, restricted to eligible rows
        if n_elig:
            maj_labels = np.asarray(
                [groups.loc[p, "majority_label"] if p in eligible_prompt_set else "" for p in holdout_prompt],
                dtype=object,
            )
            maj_acc = float((y_true[elig_mask] == maj_labels[elig_mask]).mean())
        else:
            maj_labels = np.asarray([""] * len(holdout_prompt), dtype=object)
            maj_acc = float("nan")

        grid_rows.append({
            "grid": grid_name,
            "min_purity": min_purity,
            "min_n": min_n,
            "n_eligible_groups": int(len(eligible_groups)),
            "n_holdout_rows_covered": n_elig,
            "coverage_pct": elig_cov_pct,
            "majority_label_accuracy": maj_acc,
        })

        override_label = np.where(elig_mask, maj_labels, "")
        sim = run_override_simulation(
            holdout_pred=champ_pred,
            holdout_override_label=override_label,
            y_true=y_true,
            session_keys=holdout_session_keys,
        )
        sim["grid"] = grid_name
        override_rows.append(sim)

        print(f"  [{grid_name}] purity>={min_purity} n>={min_n}: 적용 그룹수={len(eligible_groups)}, "
              f"커버리지={elig_cov_pct:.2f}% ({n_elig}행), 다수라벨정확도={maj_acc:.4f}")
        print(f"      row baseline={sim['row_baseline_macro_f1']:.6f} override={sim['row_override_macro_f1']:.6f} "
              f"delta={sim['row_delta']:+.6f}  | MC mean={sim['mc_mean_delta']:+.6f} "
              f"std={sim['mc_std_delta']:.6f} range=[{sim['mc_min_delta']:+.6f},{sim['mc_max_delta']:+.6f}]")

    best_row = max(override_rows, key=lambda r: r["row_delta"])
    best_mc = max(override_rows, key=lambda r: r["mc_mean_delta"])
    verdict_delta = max(best_row["row_delta"], best_mc["mc_mean_delta"])
    verdict = "후속 검토" if verdict_delta >= OVERRIDE_GATE else "폐기"
    print("\n" + "=" * 70)
    print(f"[판정] 게이트 +{OVERRIDE_GATE} 기준 최고 그리드 delta(row)={best_row['row_delta']:+.6f} "
          f"({best_row['grid']}), delta(MC mean)={best_mc['mc_mean_delta']:+.6f} ({best_mc['grid']}) "
          f"=> {verdict}")

    # ---- write outputs ----
    for out_dir in (args.out_dir, SCRATCH_DIR):
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(grid_rows).to_csv(out_dir / "template_dup_probe_grids.csv", index=False)
            pd.DataFrame(override_rows).to_csv(out_dir / "template_dup_probe_overrides.csv", index=False)
        except OSError:
            pass

    result = {
        "champion": {
            "macro_f1": champ_macro_f1,
            "record_value": record_value,
            "delta_vs_record": champ_macro_f1 - record_value,
            "n_holdout_rows": int(len(data.ids)),
            "n_holdout_sessions": len(holdout_sessions),
        },
        "coverage_pct": coverage_pct,
        "template_stats": tmpl_stats,
        "non_template_stats": non_tmpl_stats,
        "grids": grid_rows,
        "overrides": override_rows,
        "verdict": {
            "gate": OVERRIDE_GATE,
            "best_row_grid": best_row["grid"],
            "best_row_delta": best_row["row_delta"],
            "best_mc_grid": best_mc["grid"],
            "best_mc_mean_delta": best_mc["mc_mean_delta"],
            "decision": verdict,
        },
        "elapsed_sec": round(time.time() - t0, 3),
    }
    out_json = args.out_dir / "template_dup_probe_result.json"
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {out_json}")
    print(f"[done] elapsed={result['elapsed_sec']:.3f}s")


if __name__ == "__main__":
    main()
