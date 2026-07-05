"""공용 피처 모듈 — train 과 추론(script.py)이 반드시 같이 import.

핵심 계약: `build_dataframe(samples)` 가 만드는 DataFrame 의 컬럼은 train/추론에서
항상 동일해야 한다. 학습된 sklearn 파이프라인(ColumnTransformer+분류기)은 컬럼 '이름'으로
피처를 고르므로, 이 함수만 같으면 추론이 학습과 정확히 일치한다.
(커스텀 로직은 pickle 에 안 들어가고 이 파일 코드로 산다 → 오프라인 안전.)
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC

# ---- 14개 행동 클래스 ----
ACTIONS = [
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch",
    "run_bash", "run_tests", "lint_or_typecheck",
    "ask_user", "plan_task", "web_search", "respond_only",
]

_STEP_RE = re.compile(r"-step_\d+$")
_GLOB_RE = re.compile(r"\*\*|\*\.\w+|/\*|\*/")
_EXT_RE = re.compile(r"\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
_PATH_RE = re.compile(r"[\w\-/]+/[\w\-./]+|[\w\-]+\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
_GREP_WORDS = re.compile(
    r"grep|search|find|찾|훑|검색|where\b|locate|어디", re.IGNORECASE)


def session_id(sample_id):
    return _STEP_RE.sub("", str(sample_id))


def load_jsonl(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _s(x):
    """안전 문자열."""
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _dominant_lang(ws):
    mix = (ws or {}).get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return max(mix.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)[0]


def _row(sample):
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    hist = sample.get("history") or []
    cp = _s(sample.get("current_prompt"))

    # history 파싱
    user_texts, action_names, last_action = [], [], "none"
    for h in hist:
        if not isinstance(h, dict):
            continue
        role = h.get("role")
        if role == "assistant_action":
            nm = _s(h.get("name"))
            action_names.append(nm)
            last_action = nm or last_action  # 마지막 action 갱신
        else:
            user_texts.append(_s(h.get("content")))
    acnt = Counter(action_names)

    open_files = ws.get("open_files") or []
    if not isinstance(open_files, list):
        open_files = []
    # 발화가 언급한 파일이 열린 파일 목록에 있나
    file_in_open = 0
    low_cp = cp.lower()
    for of in open_files:
        of = _s(of)
        base = of.split("/")[-1].lower()
        if of and (of.lower() in low_cp or (len(base) > 2 and base in low_cp)):
            file_in_open = 1
            break

    row = {
        "cp": cp,
        "hist_text": " ".join(user_texts),
        # 범주형
        "user_tier": _s(sm.get("user_tier")) or "none",
        "language_pref": _s(sm.get("language_pref")) or "none",
        "last_ci_status": _s(ws.get("last_ci_status")) or "none",
        "dominant_ws_lang": _dominant_lang(ws),
        "last_action": last_action or "none",
        # 수치 (history)
        "n_history": len(hist),
        # 수치 (session_meta)
        "turn_index": float(sm.get("turn_index") or 0),
        "elapsed_log": np.log1p(float(sm.get("elapsed_session_sec") or 0)),
        "budget_log": np.log1p(float(sm.get("budget_tokens_remaining") or 0)),
        "loc_log": np.log1p(float(ws.get("loc") or 0)),
        "n_open_files": len(open_files),
        "git_dirty": 1 if ws.get("git_dirty") else 0,
        # 플래그 (탐색 계열 신호)
        "has_glob": 1 if _GLOB_RE.search(cp) else 0,
        "has_ext": 1 if _EXT_RE.search(cp) else 0,
        "has_grep": 1 if _GREP_WORDS.search(cp) else 0,
        "path_in_prompt": 1 if _PATH_RE.search(cp) else 0,
        "file_in_open": file_in_open,
    }
    # history 내 action 카운트 14개
    for a in ACTIONS:
        row[f"act_{a}"] = acnt.get(a, 0)

    # --- E 피처셋: action 시퀀스 + 구문(regex) 신호 ---
    # 직전 행동들의 '순서'를 텍스트로 (n-gram 벡터라이저가 마르코프/멀티스텝 패턴 학습)
    row["hist_action_seq"] = " ".join(action_names) if action_names else "none"
    # 마지막 두 행동 결합 (bigram 전이)
    if len(action_names) >= 2:
        row["last2_action"] = "_".join(action_names[-2:])
    else:
        row["last2_action"] = last_action
    # 구문 신호 (탐색 계열 분기: glob/grep/read/list)
    row["n_slash"] = cp.count("/")
    row["n_star"] = cp.count("*") + cp.count("?")
    row["has_regex_meta"] = 1 if re.search(r"[\^$|\\]|\\s|\\d|\[.*\]", cp) else 0
    row["has_list_word"] = 1 if re.search(
        r"\blist\b|\bls\b|디렉토리|폴더|목록|안에 뭐|무슨 파일|what.s in", cp, re.I) else 0
    row["has_read_word"] = 1 if re.search(
        r"open|열어|보여|봐줘|\bshow\b|\bread\b|읽어|내용|뭐라고", cp, re.I) else 0
    row["has_quote"] = 1 if ('"' in cp or "'" in cp or "`" in cp) else 0
    return row


def build_dataframe(samples):
    return pd.DataFrame([_row(s) for s in samples])


# ---- 컬럼 그룹 ----
CAT_META = ["user_tier", "language_pref", "last_ci_status", "dominant_ws_lang"]
CAT_HIST = ["last_action"]
NUM_HIST = ["n_history"] + [f"act_{a}" for a in ACTIONS]
NUM_META = ["turn_index", "elapsed_log", "budget_log", "loc_log",
            "n_open_files", "git_dirty", "has_glob", "has_ext",
            "has_grep", "path_in_prompt", "file_in_open"]
# E 피처셋 추가분
CAT_SEQ = ["last2_action"]
NUM_SEQ = ["n_slash", "n_star", "has_regex_meta",
           "has_list_word", "has_read_word", "has_quote"]
_ACTSEQ = dict(analyzer="word", ngram_range=(1, 3), min_df=3,
               max_features=5_000, lowercase=False)

# 벡터라이저 설정 (train/추론 공유)
_WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2,
             max_features=40_000, sublinear_tf=True, lowercase=True)
_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
             max_features=25_000, sublinear_tf=True, lowercase=True)
_HIST_WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=3,
                  max_features=15_000, sublinear_tf=True, lowercase=True)

# ablation 피처셋: 한 칸씩 누적
FEATURE_SETS = {
    "A_word":        {"cp_word"},
    "B_word_char":   {"cp_word", "cp_char"},
    "C_+history":    {"cp_word", "cp_char", "hist", "act", "last_action"},
    "D_+meta":       {"cp_word", "cp_char", "hist", "act", "last_action", "meta"},
    "E_+seq":        {"cp_word", "cp_char", "hist", "act", "last_action",
                      "meta", "seq"},
}


def build_column_transformer(fs):
    """fs = FEATURE_SETS 의 값(set). 해당 피처만 켠 ColumnTransformer."""
    parts = []
    if "cp_word" in fs:
        parts.append(("cp_word", TfidfVectorizer(**_WORD), "cp"))
    if "cp_char" in fs:
        parts.append(("cp_char", TfidfVectorizer(**_CHAR), "cp"))
    if "hist" in fs:
        parts.append(("hist_word", TfidfVectorizer(**_HIST_WORD), "hist_text"))
    if "seq" in fs:
        parts.append(("actseq", TfidfVectorizer(**_ACTSEQ), "hist_action_seq"))
    cats = []
    if "last_action" in fs:
        cats += CAT_HIST
    if "meta" in fs:
        cats += CAT_META
    if "seq" in fs:
        cats += CAT_SEQ
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           min_frequency=20), cats))
    nums = []
    if "act" in fs:
        nums += NUM_HIST
    if "meta" in fs:
        nums += NUM_META
    if "seq" in fs:
        nums += NUM_SEQ
    if nums:
        parts.append(("num", StandardScaler(with_mean=False), nums))
    return ColumnTransformer(parts, sparse_threshold=0.3)


def build_clf(kind="sgd", C=1.0, max_iter=1000, alpha=3e-5):
    """kind: 'sgd'(SGDClassifier, 매우 빠름·대용량 희소 표준) |
    'svc'(LinearSVC, 강하나 느림) | 'logreg'(확률, 가장 느림)."""
    if kind == "logreg":
        return LogisticRegression(solver="saga", max_iter=max_iter, C=C,
                                  class_weight="balanced", tol=1e-3)
    if kind == "svc":
        return LinearSVC(C=C, class_weight="balanced", max_iter=max_iter,
                         tol=1e-3, dual=True)
    # sgd (기본) — 선형 SVM(hinge)을 SGD로. 확률적이라 매우 빠름.
    return SGDClassifier(loss="hinge", penalty="l2", alpha=alpha,
                         class_weight="balanced", max_iter=40, tol=1e-3,
                         random_state=42, n_jobs=-1, early_stopping=False)


def build_pipeline(fs, clf="sgd", C=1.0, max_iter=1000, alpha=3e-5):
    return Pipeline([
        ("feat", build_column_transformer(fs)),
        ("clf", build_clf(clf, C=C, max_iter=max_iter, alpha=alpha)),
    ])
