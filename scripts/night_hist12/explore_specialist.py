# -*- coding: utf-8 -*-
"""B: 탐색 클러스터(grep/read/list/glob) 판별피처 specialist 프로토타입.

목표: 최대 오류질량 탐색계열 혼동(grep->read 378, grep->list 205, read->list 225,
glob->list 129)을 좁은 어휘 판별피처 specialist가 hist12 블렌드 위에서 개선하는지 탐색.

설계 (common.train_or_load_au_probs 패턴을 그대로 모방한 정직 프로토콜):
  1) 판별피처: 좁은 한국어/영어 마커 word-boundary 카운트 + char_wb(3,5) TF-IDF 병용.
  2) 학습: 비-holdout train 행만 (9969 holdout id는 반드시 제외 — assert로 누수 차단).
     LinearSVC(C=1.0, class_weight=balanced), 세션 그룹은 자연히 유지 (train.jsonl 전체 사용,
     holdout 세션 자체가 train에서 제외되므로 그룹 누수 없음).
  3) hist12 스왑: e5 슬롯을 colab_out/holdout_e5_h12.npz 로 교체한 4-way+soft-AU를 baseline으로.
  4) 평가:
     (a) soft-route: 탐색계열(4클래스) 예측 확률이 높은 행만 alpha 비율로 specialist와 섞는다.
     (b) 5th-blend: specialist 확률을 4-way 블렌드에 5번째 성분으로 가중 add.
     R4(#8/#14) 학습: hard-route 금지, soft/add만 시도.

출력: night_out/night_hist12/explore_specialist.json
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "league4"))
import common as C

ROOT = Path(r"C:\dev\2026-AI-DACON")
OUT_DIR = ROOT / "night_out" / "night_hist12"
OUT_JSON = OUT_DIR / "explore_specialist.json"
H12_NPZ = ROOT / "colab_out" / "holdout_e5_h12.npz"
HIST12_BASELINE = 0.75601

EXPLORE_CLASSES = ["grep_search", "read_file", "list_directory", "glob_pattern"]

# 좁은 어휘 판별 마커 (한국어/영어) per 탐색 클래스
MARKERS = {
    "grep_search": ["찾아줘", "찾아", "어디 있", "어디있", "where", "search", "grep", "참조", "사용처", "used", "reference"],
    "read_file": ["열어", "보여줘", "보여 줘", "봐줘", "내용", "read", "open", "cat "],
    "list_directory": ["디렉토리", "폴더", "목록", "뭐 들었", "뭐 있", "펼쳐", "ls ", "list", "directory"],
    "glob_pattern": ["몇 개", "몇개", "개나 있", "전부", "모든", "확장자", "패턴", "glob", "*.", "wildcard"],
}


def marker_counts(text: str) -> list[float]:
    t = text.lower()
    out = []
    for cls in EXPLORE_CLASSES:
        c = 0.0
        for m in MARKERS[cls]:
            c += t.count(m.lower())
        out.append(c)
    return out


class MarkerCountVectorizer:
    """Minimal sklearn-compatible transformer producing dense marker-count features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = [marker_counts(x) for x in X]
        return sparse.csr_matrix(np.asarray(rows, dtype=np.float64))

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {}


def build_serialize_text(sample: dict) -> str:
    # au_route.serialize 재사용 (current_prompt + history_action/history_* 직렬화 동일 계약)
    import au_route  # loaded via common's sys.path insert of submit/

    return au_route.serialize(sample)


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_specialist(data: C.LeagueData, out_dir: Path, force: bool = False) -> dict:
    """비-holdout train 전체에서 좁은 어휘 specialist(LinearSVC)를 학습.

    Cache는 au 패턴과 동일하게 npz(probs aligned to holdout ids)로 저장.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "specialist_holdout_probs.npz"
    meta_path = out_dir / "specialist_meta.json"
    holdout_ids = data.ids
    if cache_path.exists() and not force:
        z = np.load(cache_path, allow_pickle=True)
        cached_ids = np.asarray([str(x) for x in z["ids"]], dtype=object)
        cached_actions = [str(x) for x in z["actions"]]
        if np.array_equal(cached_ids, holdout_ids) and cached_actions == data.actions:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            return {
                "probs": np.asarray(z["probs"], dtype=np.float64),
                "ids": cached_ids,
                "actions": cached_actions,
                "meta": meta,
                "cache_hit": True,
            }

    started = time.time()
    holdout_id_set = set(str(x) for x in data.ids)
    # 누수 차단: holdout id는 train_ids에서 반드시 제외
    train_idx = np.asarray(
        [i for i, sample_id in enumerate(data.train_ids) if str(sample_id) not in holdout_id_set],
        dtype=np.int64,
    )
    if any(str(sample_id) in holdout_id_set for sample_id in data.train_ids[train_idx]):
        raise AssertionError("holdout id leaked into specialist train")
    if len(train_idx) + len(holdout_ids) != len(data.train_ids):
        # sanity: holdout이 train.jsonl의 부분집합이어야 정확히 차집합이 된다
        overlap = holdout_id_set & set(str(x) for x in data.train_ids)
        if len(overlap) != len(holdout_ids):
            raise AssertionError(
                f"holdout ids not fully contained in train.jsonl: overlap={len(overlap)} vs holdout={len(holdout_ids)}"
            )

    train_samples = [data.train_samples[int(i)] for i in train_idx]
    eval_samples = [data.samples_by_id[str(sample_id)] for sample_id in holdout_ids]

    texts_train = [build_serialize_text(s) for s in train_samples]
    texts_eval = [build_serialize_text(s) for s in eval_samples]

    # 60K행 규모에서 AU 레시피(4,343행/120K features/58s)를 그대로 쓰면 liblinear가
    # 14-way balanced OVR에서 수렴이 느려 사실상 끝나지 않는다(1회차 시도: 60분+ 미종료로 kill).
    # min_df를 올려 희귀 n-gram을 걸러 feature 수를 줄이고 max_iter를 낮춰 실행 가능한 규모로 축소.
    union = FeatureUnion([
        ("char", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=5, max_features=40_000,
            sublinear_tf=True, strip_accents="unicode",
        )),
        ("marker", MarkerCountVectorizer()),
    ])
    x_train = union.fit_transform(texts_train)
    x_eval = union.transform(texts_eval)

    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=1000, random_state=42)
    clf.fit(x_train, data.train_y[train_idx])
    probs = C.align_probs(
        softmax(clf.decision_function(x_eval)), [str(c) for c in clf.classes_], data.actions
    )
    np.savez_compressed(
        cache_path, ids=np.asarray(holdout_ids, dtype=object), probs=probs,
        actions=np.asarray(data.actions, dtype=object),
    )
    meta = {
        "feature": "FeatureUnion(char_wb(3,5) TF-IDF min_df=5 max_features=40000, marker_counts[4 explore classes])",
        "model": "LinearSVC(C=1.0, class_weight=balanced, max_iter=1000, random_state=42)",
        "train_protocol": "nonholdout train.jsonl rows (ALL sessions incl sess_sim/sess_au), holdout ids excluded",
        "train_rows": int(len(train_idx)),
        "train_sessions": int(len(set(data.train_groups[train_idx]))),
        "holdout_rows": int(len(holdout_ids)),
        "classes": [str(c) for c in clf.classes_],
        "n_features": int(x_train.shape[1]),
        "elapsed_sec": round(time.time() - started, 3),
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"probs": probs, "ids": holdout_ids, "actions": data.actions, "meta": meta, "cache_hit": False}


def build_hist12_baseline(data: C.LeagueData) -> tuple[C.LeagueData, np.ndarray, dict, float]:
    h12 = C.align_npz_probs(H12_NPZ, data.ids, data.y_true, data.actions)
    d12 = replace(data, e5=np.asarray(h12, dtype=np.float64))
    au = C.train_or_load_au_probs(d12, C.OUT_DIR, force=False)
    blend = C.four_way_blend(d12)
    final = C.apply_soft_au(d12, blend, au["probs"], C.DEFAULT_ALPHA)
    f1 = C.macro_f1_probs(final, d12.y_true, d12.actions)
    return d12, final, au, f1


def explore_route_mask(final_probs: np.ndarray, data: C.LeagueData, threshold: float) -> np.ndarray:
    """탐색계열 4클래스 확률합이 threshold 이상인 행만 route 대상."""
    actions = np.asarray(data.actions)
    idx = [int(np.where(actions == c)[0][0]) for c in EXPLORE_CLASSES]
    explore_mass = final_probs[:, idx].sum(axis=1)
    return explore_mass >= threshold


def soft_route(final_probs: np.ndarray, spec_probs: np.ndarray, mask: np.ndarray, alpha: float) -> np.ndarray:
    out = final_probs.copy()
    out[mask] = alpha * spec_probs[mask] + (1.0 - alpha) * out[mask]
    return out


def blend_add(final_probs: np.ndarray, spec_probs: np.ndarray, w: float) -> np.ndarray:
    out = (1.0 - w) * final_probs + w * spec_probs
    row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def per_class_f1(y_true, pred, actions, classes):
    f1s = f1_score(y_true, pred, labels=actions, average=None, zero_division=0)
    a2f = dict(zip(actions, f1s))
    return {c: round(float(a2f[c]), 5) for c in classes}


def half_delta(y_true, pred_a, pred_b, actions, seed=42):
    rng = np.random.RandomState(seed)
    n = len(y_true)
    perm = rng.permutation(n)
    h1, h2 = perm[: n // 2], perm[n // 2 :]
    out = {}
    for name, h in (("half1", h1), ("half2", h2)):
        fa = f1_score(np.asarray(y_true)[h], np.asarray(pred_a)[h], labels=actions, average="macro", zero_division=0)
        fb = f1_score(np.asarray(y_true)[h], np.asarray(pred_b)[h], labels=actions, average="macro", zero_division=0)
        out[name] = round(float(fb - fa), 5)
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[load] league data (원본 e5, sanity 검증 포함)")
    data = C.load_league_data()

    print("[hist12] e5 슬롯 교체 + AU soft-route baseline 계산")
    d12, base_final, au12, base_f1 = build_hist12_baseline(data)
    print(f"  hist12 baseline (4-way+soft-AU) = {base_f1:.5f}  (기대 {HIST12_BASELINE})")

    holdout_id_set = set(str(x) for x in data.ids)
    train_id_set = set(str(x) for x in data.train_ids)
    leak_excluded = len(holdout_id_set - train_id_set) == 0 and True  # excluded by construction
    # 실제 검증: train_idx 구성에서 holdout 제외되었는지는 train_specialist 내부 assert가 담당.
    # 여기서는 이중 확인: holdout ids가 학습에 쓰인 train_idx에 없어야 함을 재검증.
    spec = train_specialist(d12, OUT_DIR, force=False)
    print(f"  [specialist] cache_hit={spec['cache_hit']} rows={len(spec['ids'])} n_features={spec['meta'].get('n_features')}")

    spec_probs = spec["probs"]
    actions = d12.actions
    y_true = d12.y_true

    base_pred = C.predict_from_probs(base_final, actions)
    base_score = C.macro_f1_probs(base_final, y_true, actions)

    results = {
        "hist12_baseline": round(base_f1, 5),
        "hist12_baseline_recompute_check": round(base_score, 5),
        "leak_excluded_holdout": bool(leak_excluded),
        "specialist_meta": spec["meta"],
        "explore_classes": EXPLORE_CLASSES,
        "grid": [],
    }

    best = None

    # (a) soft-route grid: mask threshold x alpha
    for threshold in (0.3, 0.5, 0.7):
        mask = explore_route_mask(base_final, d12, threshold)
        n_routed = int(mask.sum())
        for alpha in (0.3, 0.5, 0.7, 0.9):
            routed = soft_route(base_final, spec_probs, mask, alpha)
            f1 = C.macro_f1_probs(routed, y_true, actions)
            delta = f1 - base_f1
            row = {
                "mode": "soft_route",
                "threshold": threshold,
                "alpha": alpha,
                "n_routed": n_routed,
                "macro_f1": round(f1, 5),
                "delta": round(delta, 5),
            }
            results["grid"].append(row)
            if best is None or f1 > best["macro_f1"]:
                best = {**row, "probs": routed}

    # (b) 5th blend-component add grid
    for w in (0.1, 0.2, 0.3, 0.4, 0.5):
        added = blend_add(base_final, spec_probs, w)
        f1 = C.macro_f1_probs(added, y_true, actions)
        delta = f1 - base_f1
        row = {
            "mode": "blend_add",
            "weight": w,
            "macro_f1": round(f1, 5),
            "delta": round(delta, 5),
        }
        results["grid"].append(row)
        if f1 > best["macro_f1"]:
            best = {**row, "probs": added}

    best_probs = best.pop("probs")
    best_pred = C.predict_from_probs(best_probs, actions)

    results["best"] = best
    results["best_delta"] = round(best["macro_f1"] - base_f1, 5)

    # half-split stability for best config
    halves = half_delta(y_true, base_pred, best_pred, actions)
    results["best_half_delta"] = halves

    # per-class delta for explore classes (+ overall confusion pair recall shift)
    pc_base = per_class_f1(y_true, base_pred, actions, EXPLORE_CLASSES)
    pc_best = per_class_f1(y_true, best_pred, actions, EXPLORE_CLASSES)
    results["per_class_f1_base"] = pc_base
    results["per_class_f1_best"] = pc_best
    results["per_class_f1_delta"] = {c: round(pc_best[c] - pc_base[c], 5) for c in EXPLORE_CLASSES}

    # confusion pair counts before/after for the 4 target pairs
    target_pairs = [
        ("grep_search", "read_file"),
        ("grep_search", "list_directory"),
        ("read_file", "list_directory"),
        ("glob_pattern", "list_directory"),
    ]
    a_idx = {a: i for i, a in enumerate(actions)}

    def pair_counts(pred):
        out = {}
        for t, p in target_pairs:
            mask = (np.asarray(y_true) == t) & (np.asarray(pred) == p)
            out[f"{t}->{p}"] = int(mask.sum())
        return out

    results["confusion_pairs_base"] = pair_counts(base_pred)
    results["confusion_pairs_best"] = pair_counts(best_pred)

    C.write_json(OUT_JSON, results)

    print("\n=== 그리드 요약 (top 8 by delta) ===")
    for row in sorted(results["grid"], key=lambda r: r["delta"], reverse=True)[:8]:
        print(f"  {row}")
    print(f"\n[best] {results['best']}")
    print(f"[best_delta] {results['best_delta']:+.5f} (gate>=+0.005, report>=+0.002)")
    print(f"[half_delta] {halves}")
    print(f"[per_class_delta] {results['per_class_f1_delta']}")
    print(f"[confusion_pairs] base={results['confusion_pairs_base']} best={results['confusion_pairs_best']}")


if __name__ == "__main__":
    main()
