# -*- coding: utf-8 -*-
"""serialize_v2 vs serialize_v3 로컬 프록시 A/B (GPU 게이트 결정용).

목적: e5 파인튜닝(수 시간, Colab GPU)을 태우기 전에, 텍스트 직렬화 변경이 방향적으로
유의미한 개선인지 로컬 CPU에서 값싸게 검증한다. 실제 인코더가 아니라
TF-IDF(word+char) + LinearSVC 프록시로 macro-F1 랭킹만 본다 — 절대 수치가 아니라
v2 대비 v3(및 ablation)의 상대적 순위가 신호다.

데이터 실사 근거(2026-07-05, data/train.jsonl 70,000행 전수):
  - history 길이 분포: 12(21,586) > 0(9,000) > 2(8,797) > 4(8,446) > 6(8,001)
    > 8(7,526) > 10(6,644) — v2의 max_hist=6은 표본 상당수(12턴 보유 21.6k행)의
    최근 6턴을 이미 버리고 있었다.
  - assistant_action.args 키 분포(전부 dict, 값 길이 짧음 — p99 대부분 <75자,
    ask_user 질문만 최대 209자): read_file/list_directory/write_file/edit_file
    → path, grep_search → pattern+scope, glob_pattern → pattern, edit_file →
    target_symbol, run_bash → cmd, run_tests/lint_or_typecheck → target,
    apply_patch → n_files, ask_user → question, plan_task → goal, web_search
    → query. v2는 이 args를 전부 버리고 result_summary만 쓴다 — explore
    4클래스(read_file/grep_search/list_directory/glob_pattern)는 path/pattern
    없이는 서로 구분하기 어려운 액션 쌍이 많다(예: read_file vs list_directory
    모두 "ok; ..." 류 요약).
  - workspace.language_mix: {"py":0.82,"yaml":0.1,...} 형태 dict, 상위 언어
    분포 py(27,897) > rs(7,881) > java(7,196) > vue/tsx/ts/go/yaml 순.
  - session_meta.elapsed_session_sec: 30~1530초, mean 511 / median 498.
  - current_prompt: p50=56자 p99=169자 — v2의 800자 캡은 이미 여유(정보 손실 없음).

serialize_v3 스펙 (v2 diff):
  1. max_hist 6 → 12 (history 최대 길이와 일치, 정보 손실 축소)
  2. assistant_action 턴에 args 핵심 값 추가: "act:{name} {abbr}:{value[:60]} ... r:{summary}"
     (액션별 축약 키는 ARG_ABBR 참고, 값은 60자 truncate — 실측 p99 대비 넉넉)
  3. 맨 끝(최저 우선순위)에 "toplang={language_mix 최댓값 키} elapsed={버킷}" 한 줄 추가
     — 잘림이 나면 여기부터 잘리도록 텍스트 맨 뒤에 배치(v2의 reversed 최신-우선 원칙과
     같은 이유: 자를 거면 덜 중요한 것부터).
  헤더/open/query 라인은 v2와 동일(정보 손실 없어 바꿀 이유 없음).

잘림률 실측 — ⚠️ 2026-07-05 정정(독립 리뷰 FAIL, 최초 수치는 틀렸음):
  최초 설계 시 "문자수/4 ≈ 토큰수"로 근사했으나 실제 e5 토크나이저
  (`artifacts/enc_v2_s42/model/tokenizer.json`, `tokenizers` 패키지로 오프라인 로드해
  no_truncation()+no_padding()으로 저장된 384-cap 설정을 해제하고 측정)로 70,000행
  전수 재측정한 결과 chars/token ≈ 2.43(4가 아니라 4의 약 61% — 근사가 토큰수를
  약 1.64배 과소평가했었다). 실측치:
    max_len=384: v2 잘림 0/70000(0.00%), v3(max_hist=12) 잘림 20,511/70000(29.30%),
      v3 중 history=12턴 보유 행(전체의 30.8%, v3가 노리는 핵심 타깃)만 보면 82.88% 잘림,
      v3_hist6(=v2+args+lang+elapsed, max_hist=6) 잘림 29/70000(0.04%, 무시 가능)
    max_len=512: v3 잘림 1,309/70000(1.87%), v3_hist6 잘림 0/70000(0.00%)
  즉 v3(전체, max_hist=12)를 384로 돌리면 개선 대상 표본(history=12턴)의 83%가
  잘려 max_hist 확장 효과가 대부분 무효화된다 — v3(전체)를 쓸 경우 ENC_MAX_LEN=512가
  사실상 필수(모델 max_position_embeddings=514라 512가 물리적 상한). v3_hist6은
  384에서도 잘림이 무시 가능해 그대로 384를 써도 안전하다. 상세는
  colab/encoder_v3_repro.py 모듈 docstring의 "ENC_SERIALIZE 스위치" 절 참고.

실행:
  "C:/dev/2026-AI-DACON/.venv/Scripts/python.exe" scripts/encoding/serialize_ab.py
  환경변수:
    AB_FOLDS=5          StratifiedGroupKFold fold 수(단축 필요 시 3)
    AB_MAX_FEATURES=30000  word/char 벡터라이저 각각의 max_features
    AB_VARIANTS=v2,v3,v3_no_args   콤마 구분 variant 부분집합(미설정 시 ALL_VARIANTS 전체)
    AB_DATA_DIR=./data
    AB_OUT_DIR=./scripts/encoding/_out
"""
import json
import os
import re
import time
from collections import defaultdict

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import LinearSVC

DATA_DIR = os.environ.get("AB_DATA_DIR", "./data")
OUT_DIR = os.environ.get("AB_OUT_DIR", "./scripts/encoding/_out")
N_FOLDS = int(os.environ.get("AB_FOLDS", "5"))
MAX_FEATURES = int(os.environ.get("AB_MAX_FEATURES", "30000"))
SEED = 42

EXPLORE4 = ["read_file", "grep_search", "list_directory", "glob_pattern"]

_STEP_RE = re.compile(r"-step_\d+$")


def _bucket(v, edges=(1000, 10000, 50000, 100000)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


def _bucket_elapsed(v, edges=(120, 300, 600, 900, 1200)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


# ===== serialize_v2 (submit/script.py:108-132 verbatim — 현재 챔피언 계약) =====
def serialize_v2(s, max_hist=6):
    sm = s.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    parts = []
    parts.append(
        f"tier={sm.get('user_tier', '?')} lang={sm.get('language_pref', '?')} "
        f"turn={sm.get('turn_index', '?')} ci={ws.get('last_ci_status', '?')} "
        f"dirty={int(bool(ws.get('git_dirty')))} "
        f"budget={_bucket(sm.get('budget_tokens_remaining'))} loc={_bucket(ws.get('loc'))}"
    )
    open_files = ws.get("open_files") or []
    if open_files:
        parts.append("open: " + " ".join(str(x) for x in open_files[:5]))
    parts.append("query: " + str(s.get("current_prompt") or "")[:800])
    hist = s.get("history") or []
    items = []
    for h in reversed(hist[-max_hist:]):
        if h.get("role") == "assistant_action":
            items.append(f"act:{h.get('name')} r:{str(h.get('result_summary') or '')[:120]}")
        else:
            items.append(f"user: {str(h.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    return "\n".join(parts)


# ===== serialize_v3 후보 (액션별 args 축약 키) =====
ARG_ABBR = {
    "read_file": [("path", "p")],
    "grep_search": [("pattern", "pat"), ("scope", "scope")],
    "list_directory": [("path", "p")],
    "glob_pattern": [("pattern", "pat")],
    "edit_file": [("path", "p"), ("target_symbol", "sym")],
    "write_file": [("path", "p")],
    "apply_patch": [("n_files", "n")],
    "run_bash": [("cmd", "cmd")],
    "run_tests": [("target", "t")],
    "lint_or_typecheck": [("target", "t")],
    "ask_user": [("question", "q")],
    "plan_task": [("goal", "g")],
    "web_search": [("query", "q")],
}


def _args_str(name, args):
    if not isinstance(args, dict):
        return ""
    out = []
    for key, abbr in ARG_ABBR.get(name, []):
        if key in args:
            out.append(f"{abbr}:{str(args[key])[:60]}")
    return " ".join(out)


def serialize_v3_variant(s, max_hist=12, include_args=True, include_lang=True,
                          include_elapsed=True):
    """v3 및 ablation 공용 빌더. 기본 인자 = full v3."""
    sm = s.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    parts = []
    parts.append(
        f"tier={sm.get('user_tier', '?')} lang={sm.get('language_pref', '?')} "
        f"turn={sm.get('turn_index', '?')} ci={ws.get('last_ci_status', '?')} "
        f"dirty={int(bool(ws.get('git_dirty')))} "
        f"budget={_bucket(sm.get('budget_tokens_remaining'))} loc={_bucket(ws.get('loc'))}"
    )
    open_files = ws.get("open_files") or []
    if open_files:
        parts.append("open: " + " ".join(str(x) for x in open_files[:5]))
    parts.append("query: " + str(s.get("current_prompt") or "")[:800])
    hist = s.get("history") or []
    items = []
    for h in reversed(hist[-max_hist:]):
        if h.get("role") == "assistant_action":
            name = h.get("name")
            r = str(h.get("result_summary") or "")[:120]
            a = _args_str(name, h.get("args")) if include_args else ""
            items.append(f"act:{name} {a} r:{r}" if a else f"act:{name} r:{r}")
        else:
            items.append(f"user: {str(h.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    tail = []
    if include_lang:
        lm = ws.get("language_mix") or {}
        toplang = max(lm.items(), key=lambda kv: kv[1])[0] if lm else "?"
        tail.append(f"toplang={toplang}")
    if include_elapsed:
        tail.append(f"elapsed={_bucket_elapsed(sm.get('elapsed_session_sec'))}")
    if tail:
        parts.append(" ".join(tail))
    return "\n".join(parts)


ALL_VARIANTS = {
    "v2": lambda s: serialize_v2(s, max_hist=6),
    "v3": lambda s: serialize_v3_variant(s),
    "v3_no_args": lambda s: serialize_v3_variant(s, include_args=False),
    "v3_hist6": lambda s: serialize_v3_variant(s, max_hist=6),
    "v3_no_lang": lambda s: serialize_v3_variant(s, include_lang=False),
    "v3_no_elapsed": lambda s: serialize_v3_variant(s, include_elapsed=False),
}

_variant_filter = os.environ.get("AB_VARIANTS", "").strip()
if _variant_filter:
    _names = [v.strip() for v in _variant_filter.split(",") if v.strip()]
    VARIANTS = {name: ALL_VARIANTS[name] for name in _names}
else:
    VARIANTS = ALL_VARIANTS


def load_samples():
    import csv
    samples = []
    with open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with open(os.path.join(DATA_DIR, "train_labels.csv"), encoding="utf-8") as f:
        lab = {r["id"]: r["action"] for r in csv.DictReader(f)}
    y = np.array([lab[s["id"]] for s in samples])
    groups = np.array([_STEP_RE.sub("", s["id"]) for s in samples])
    return samples, y, groups


def vectorize_fit_transform(train_texts, val_texts):
    word_vec = TfidfVectorizer(ngram_range=(1, 2), max_features=MAX_FEATURES,
                                sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                max_features=MAX_FEATURES, sublinear_tf=True)
    xw_tr = word_vec.fit_transform(train_texts)
    xc_tr = char_vec.fit_transform(train_texts)
    xw_va = word_vec.transform(val_texts)
    xc_va = char_vec.transform(val_texts)
    return hstack([xw_tr, xc_tr]).tocsr(), hstack([xw_va, xc_va]).tocsr()


def run_variant(name, texts, y, groups, splits):
    fold_rows = []
    oof_pred = np.empty(len(y), dtype=object)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        t0 = time.time()
        x_tr, x_va = vectorize_fit_transform([texts[i] for i in tr_idx],
                                              [texts[i] for i in va_idx])
        clf = LinearSVC(C=0.1, class_weight="balanced", max_iter=5000)
        clf.fit(x_tr, y[tr_idx])
        pred = clf.predict(x_va)
        oof_pred[va_idx] = pred
        macro_f1 = f1_score(y[va_idx], pred, average="macro")
        per_class = dict(zip(clf.classes_, f1_score(y[va_idx], pred, average=None,
                                                      labels=clf.classes_)))
        row = {"variant": name, "fold": fold, "n_train": len(tr_idx), "n_val": len(va_idx),
               "macro_f1": macro_f1, "elapsed_sec": time.time() - t0}
        for c in EXPLORE4:
            row[f"f1_{c}"] = per_class.get(c, float("nan"))
        fold_rows.append(row)
        print(f"  [{name}] fold{fold} macro_f1={macro_f1:.4f} "
              f"explore4={[round(row[f'f1_{c}'], 3) for c in EXPLORE4]} "
              f"({row['elapsed_sec']:.0f}s)")
    overall_macro = f1_score(y, oof_pred, average="macro")
    overall_explore = f1_score(y, oof_pred, average=None, labels=EXPLORE4)
    return fold_rows, overall_macro, dict(zip(EXPLORE4, overall_explore))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[load] {DATA_DIR}/train.jsonl + train_labels.csv")
    samples, y, groups = load_samples()
    print(f"  n={len(samples)} classes={sorted(set(y))}")

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    splits = list(sgkf.split(np.zeros(len(y)), y, groups))
    # 세션 누수 방어 검증
    for fold, (tr_idx, va_idx) in enumerate(splits):
        tr_g, va_g = set(groups[tr_idx]), set(groups[va_idx])
        assert not (tr_g & va_g), f"fold {fold} 세션 누수: {list(tr_g & va_g)[:3]}"
    print(f"[cv] StratifiedGroupKFold n_splits={N_FOLDS} seed={SEED} — 누수 검증 통과")
    print(f"[variants] {list(VARIANTS.keys())} (AB_VARIANTS={_variant_filter or '(all)'})")

    # v2 대비 v3 계열 텍스트 길이 배율 실측(TF-IDF 희석 우려 정량화용, 70,000행 전체)
    len_v2 = np.array([len(serialize_v2(s, max_hist=6)) for s in samples])
    len_v3 = np.array([len(serialize_v3_variant(s)) for s in samples])
    len_v3_no_args = np.array([len(serialize_v3_variant(s, include_args=False)) for s in samples])
    print(f"[textlen] v2 mean={len_v2.mean():.0f} v3 mean={len_v3.mean():.0f} "
          f"(ratio v3/v2={len_v3.mean() / len_v2.mean():.3f}) "
          f"v3_no_args mean={len_v3_no_args.mean():.0f} "
          f"(ratio v3_no_args/v2={len_v3_no_args.mean() / len_v2.mean():.3f})")

    all_fold_rows = []
    summary_rows = []
    for name, fn in VARIANTS.items():
        print(f"[variant] {name} — 직렬화 중…")
        texts = [fn(s) for s in samples]
        t0 = time.time()
        fold_rows, overall_macro, overall_explore = run_variant(name, texts, y, groups, splits)
        all_fold_rows.extend(fold_rows)
        macro_f1s = [r["macro_f1"] for r in fold_rows]
        summary = {
            "variant": name,
            "mean_macro_f1": float(np.mean(macro_f1s)),
            "std_macro_f1": float(np.std(macro_f1s)),
            "oof_macro_f1": overall_macro,
            "elapsed_sec_total": time.time() - t0,
        }
        for c in EXPLORE4:
            summary[f"oof_f1_{c}"] = overall_explore[c]
            summary[f"mean_f1_{c}"] = float(np.mean([r[f"f1_{c}"] for r in fold_rows]))
        summary_rows.append(summary)
        print(f"[variant] {name} DONE mean_macro_f1={summary['mean_macro_f1']:.4f} "
              f"oof_macro_f1={overall_macro:.4f} ({summary['elapsed_sec_total']:.0f}s)")

    import csv as csvmod
    fold_path = os.path.join(OUT_DIR, "serialize_ab_folds.csv")
    with open(fold_path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=list(all_fold_rows[0].keys()))
        w.writeheader()
        w.writerows(all_fold_rows)
    summary_path = os.path.join(OUT_DIR, "serialize_ab_summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csvmod.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(f"[DONE] {fold_path}")
    print(f"[DONE] {summary_path}")

    print("\n=== SUMMARY (oof macro-F1 순 정렬) ===")
    for row in sorted(summary_rows, key=lambda r: -r["oof_macro_f1"]):
        explore_str = ", ".join(f"{c}:{row['oof_f1_' + c]:.3f}" for c in EXPLORE4)
        print(f"  {row['variant']:16s} oof_macro_f1={row['oof_macro_f1']:.4f} "
              f"mean_macro_f1={row['mean_macro_f1']:.4f}±{row['std_macro_f1']:.4f} "
              f"explore4_oof=[{explore_str}]")


if __name__ == "__main__":
    main()
