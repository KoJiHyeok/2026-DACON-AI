# -*- coding: utf-8 -*-
"""2-way offline inference: linear champion + AAR stacker.

Package layout:
    script.py
    requirements.txt
    model/
      linear/model.pkl
      stacker/aar_config.json
      stacker/aar_models.joblib

The blend is uniform by default. Optionally set ENS_WEIGHTS="lin,stk" or add
model/weights.json as [1, 1] / {"weights": [1, 1]}.
"""
from __future__ import annotations

import csv
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC


HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get("ENS_DATA", "./data"))
OUT = Path(os.environ.get("ENS_OUT", "./output"))
MODEL = HERE / "model"
LINEAR_PKL = Path(os.environ.get("ENS_LINEAR_PKL", MODEL / "linear" / "model.pkl"))
STACKER_DIR = Path(os.environ.get("ENS_STACKER_DIR", MODEL / "stacker"))

ACTIONS = [
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
]

_STEP_RE = re.compile(r"-step_\d+$")
_GLOB_RE = re.compile(r"\*\*|\*\.\w+|/\*|\*/")
_EXT_RE = re.compile(r"\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
_PATH_RE = re.compile(r"[\w\-/]+/[\w\-./]+|[\w\-]+\.[a-zA-Z][a-zA-Z0-9]{1,4}\b")
_GREP_WORDS = re.compile(r"grep|search|find|찾|훑|검색|where\b|locate|어디", re.IGNORECASE)


def _s(x: Any) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))


def _dominant_lang(ws: Dict[str, Any]) -> str:
    mix = (ws or {}).get("language_mix") or {}
    if not isinstance(mix, dict) or not mix:
        return "none"
    return max(mix.items(), key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else 0)[0]


def _linear_row(sample: Dict[str, Any]) -> Dict[str, Any]:
    sm = sample.get("session_meta") or {}
    ws = sm.get("workspace") or {}
    hist = sample.get("history") or []
    cp = _s(sample.get("current_prompt"))

    user_texts: List[str] = []
    action_names: List[str] = []
    last_action = "none"
    for h in hist:
        if not isinstance(h, dict):
            continue
        if h.get("role") == "assistant_action":
            name = _s(h.get("name"))
            action_names.append(name)
            last_action = name or last_action
        else:
            user_texts.append(_s(h.get("content")))
    acnt = Counter(action_names)

    open_files = ws.get("open_files") or []
    if not isinstance(open_files, list):
        open_files = []
    low_cp = cp.lower()
    file_in_open = 0
    for open_file in open_files:
        open_file = _s(open_file)
        base = open_file.split("/")[-1].lower()
        if open_file and (open_file.lower() in low_cp or (len(base) > 2 and base in low_cp)):
            file_in_open = 1
            break

    row: Dict[str, Any] = {
        "cp": cp,
        "hist_text": " ".join(user_texts),
        "user_tier": _s(sm.get("user_tier")) or "none",
        "language_pref": _s(sm.get("language_pref")) or "none",
        "last_ci_status": _s(ws.get("last_ci_status")) or "none",
        "dominant_ws_lang": _dominant_lang(ws),
        "last_action": last_action or "none",
        "n_history": len(hist),
        "turn_index": float(sm.get("turn_index") or 0),
        "elapsed_log": np.log1p(float(sm.get("elapsed_session_sec") or 0)),
        "budget_log": np.log1p(float(sm.get("budget_tokens_remaining") or 0)),
        "loc_log": np.log1p(float(ws.get("loc") or 0)),
        "n_open_files": len(open_files),
        "git_dirty": 1 if ws.get("git_dirty") else 0,
        "has_glob": 1 if _GLOB_RE.search(cp) else 0,
        "has_ext": 1 if _EXT_RE.search(cp) else 0,
        "has_grep": 1 if _GREP_WORDS.search(cp) else 0,
        "path_in_prompt": 1 if _PATH_RE.search(cp) else 0,
        "file_in_open": file_in_open,
    }
    for action in ACTIONS:
        row[f"act_{action}"] = acnt.get(action, 0)
    row["hist_action_seq"] = " ".join(action_names) if action_names else "none"
    row["last2_action"] = "_".join(action_names[-2:]) if len(action_names) >= 2 else last_action
    row["n_slash"] = cp.count("/")
    row["n_star"] = cp.count("*") + cp.count("?")
    row["has_regex_meta"] = 1 if re.search(r"[\^$|\\]|\\s|\\d|\[.*\]", cp) else 0
    row["has_list_word"] = 1 if re.search(
        r"\blist\b|\bls\b|디렉토리|폴더|목록|안에 뭐|무슨 파일|what.s in", cp, re.I
    ) else 0
    row["has_read_word"] = 1 if re.search(
        r"open|열어|보여|봐줘|\bshow\b|\bread\b|읽어|내용|뭐라고", cp, re.I
    ) else 0
    row["has_quote"] = 1 if ('"' in cp or "'" in cp or "`" in cp) else 0
    return row


def build_linear_dataframe(samples: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([_linear_row(s) for s in samples])


CAT_META = ["user_tier", "language_pref", "last_ci_status", "dominant_ws_lang"]
CAT_HIST = ["last_action"]
NUM_HIST = ["n_history"] + [f"act_{a}" for a in ACTIONS]
NUM_META = [
    "turn_index",
    "elapsed_log",
    "budget_log",
    "loc_log",
    "n_open_files",
    "git_dirty",
    "has_glob",
    "has_ext",
    "has_grep",
    "path_in_prompt",
    "file_in_open",
]
CAT_SEQ = ["last2_action"]
NUM_SEQ = ["n_slash", "n_star", "has_regex_meta", "has_list_word", "has_read_word", "has_quote"]
_ACTSEQ = dict(analyzer="word", ngram_range=(1, 3), min_df=3, max_features=5000, lowercase=False)
_WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True, lowercase=True)
_CHAR = dict(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=25000, sublinear_tf=True, lowercase=True)
_HIST_WORD = dict(analyzer="word", ngram_range=(1, 2), min_df=3, max_features=15000, sublinear_tf=True, lowercase=True)
FEATURE_SETS = {
    "A_word": {"cp_word"},
    "B_word_char": {"cp_word", "cp_char"},
    "C_+history": {"cp_word", "cp_char", "hist", "act", "last_action"},
    "D_+meta": {"cp_word", "cp_char", "hist", "act", "last_action", "meta"},
    "E_+seq": {"cp_word", "cp_char", "hist", "act", "last_action", "meta", "seq"},
}


def build_column_transformer(fs: set[str]) -> ColumnTransformer:
    parts: List[Tuple[str, Any, Any]] = []
    if "cp_word" in fs:
        parts.append(("cp_word", TfidfVectorizer(**_WORD), "cp"))
    if "cp_char" in fs:
        parts.append(("cp_char", TfidfVectorizer(**_CHAR), "cp"))
    if "hist" in fs:
        parts.append(("hist_word", TfidfVectorizer(**_HIST_WORD), "hist_text"))
    if "seq" in fs:
        parts.append(("actseq", TfidfVectorizer(**_ACTSEQ), "hist_action_seq"))
    cats: List[str] = []
    if "last_action" in fs:
        cats += CAT_HIST
    if "meta" in fs:
        cats += CAT_META
    if "seq" in fs:
        cats += CAT_SEQ
    if cats:
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), cats))
    nums: List[str] = []
    if "act" in fs:
        nums += NUM_HIST
    if "meta" in fs:
        nums += NUM_META
    if "seq" in fs:
        nums += NUM_SEQ
    if nums:
        parts.append(("num", StandardScaler(with_mean=False), nums))
    return ColumnTransformer(parts, sparse_threshold=0.3)


def build_clf(kind: str = "sgd", C: float = 1.0, max_iter: int = 1000, alpha: float = 3e-5) -> Any:
    if kind == "logreg":
        return LogisticRegression(solver="saga", max_iter=max_iter, C=C, class_weight="balanced", tol=1e-3)
    if kind == "svc":
        return LinearSVC(C=C, class_weight="balanced", max_iter=max_iter, tol=1e-3, dual=True)
    return SGDClassifier(
        loss="hinge",
        penalty="l2",
        alpha=alpha,
        class_weight="balanced",
        max_iter=40,
        tol=1e-3,
        random_state=42,
        n_jobs=-1,
        early_stopping=False,
    )


def build_pipeline(fs: set[str], clf: str = "sgd", C: float = 1.0, max_iter: int = 1000, alpha: float = 3e-5) -> Pipeline:
    return Pipeline([("feat", build_column_transformer(fs)), ("clf", build_clf(clf, C=C, max_iter=max_iter, alpha=alpha))])


def _clean_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    else:
        text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[: max_chars // 2] + " ... " + text[-max_chars // 2 :]
    return text


def _flatten_meta(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for key, value in sorted(obj.items(), key=lambda kv: str(kv[0])):
            name = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                items.extend(_flatten_meta(value, name))
            elif isinstance(value, list):
                if len(value) <= 8:
                    items.append((name, "|".join(_clean_text(x, 80) for x in value)))
                else:
                    first = "|".join(_clean_text(x, 60) for x in value[:5])
                    items.append((name, f"list_len={len(value)} first={first}"))
            else:
                items.append((name, _clean_text(value, 120)))
    return items


def extract_action_sequence(history: Iterable[Dict[str, Any]] | None) -> List[str]:
    seq: List[str] = []
    for item in history or []:
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            name = item.get("name") or item.get("action") or item.get("tool")
            if name:
                seq.append(str(name))
    return seq


def keyword_tokens(prompt: str) -> str:
    lower = prompt.lower()
    pairs = {
        "kw_test": ["test", "pytest", "unittest", "unit test", "테스트", "검증", "확인"],
        "kw_lint": ["lint", "typecheck", "type check", "mypy", "ruff", "eslint", "타입", "린트"],
        "kw_search": ["search", "grep", "find", "찾", "검색", "어디", "정의", "참조", "reference"],
        "kw_file": ["file", "파일", "열어", "읽", "수정", "패치", "저장", "open", "read"],
        "kw_dir": ["directory", "folder", "폴더", "디렉토리", "목록", "구조", "tree", "list"],
        "kw_web": ["web", "internet", "latest", "최신", "검색해서", "웹", "사이트", "뉴스", "lookup"],
        "kw_ask": ["?", "어떻게", "뭐", "무엇", "확인해", "which", "what", "how", "clarify"],
        "kw_run": ["run", "execute", "bash", "shell", "terminal", "실행", "터미널", "명령어", "커맨드"],
        "kw_plan": ["plan", "설계", "구조", "계획", "로드맵", "단계", "아키텍처", "approach"],
    }
    return " ".join(token for token, words in pairs.items() if any(word in lower for word in words))


def serialize_session_meta(meta: Dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return "[SESSION_META] none"
    return "[SESSION_META] " + " ; ".join(f"{k}={v}" for k, v in _flatten_meta(meta))


def serialize_history(history: List[Dict[str, Any]] | None, max_turns: int = 12) -> str:
    if not isinstance(history, list) or not history:
        return "[HISTORY] empty"
    recent = history[-max_turns:]
    lines = ["[HISTORY_RECENT]"]
    for idx, item in enumerate(recent):
        if not isinstance(item, dict):
            lines.append(f"turn_{idx}: {_clean_text(item, 500)}")
            continue
        role = _clean_text(item.get("role", "unknown"), 50)
        if role == "assistant_action":
            name = _clean_text(item.get("name") or item.get("action") or item.get("tool") or "unknown", 80)
            args = _clean_text(item.get("args", ""), 500)
            result = _clean_text(item.get("result_summary", item.get("result", "")), 500)
            lines.append(f"turn_{idx}: role=assistant_action action={name} args={args} result={result}")
        else:
            lines.append(f"turn_{idx}: role={role} content={_clean_text(item.get('content', ''), 800)}")
    seq = extract_action_sequence(history)
    if seq:
        lines.append("[ACTION_SEQUENCE] " + " > ".join(seq[-16:]))
        lines.append("[LAST_ACTION] " + seq[-1])
    return "\n".join(lines)


def record_to_text(record: Dict[str, Any]) -> str:
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    feature_line = "[DERIVED] " + " ".join(
        [
            f"history_len={len(history) if isinstance(history, list) else 0}",
            f"last_action={last_action}",
            keyword_tokens(prompt),
        ]
    )
    return "\n".join(
        [
            f"[ID] {_clean_text(record.get('id', ''), 120)}",
            "[CURRENT_PROMPT] " + prompt,
            feature_line,
            serialize_session_meta(record.get("session_meta") if isinstance(record.get("session_meta"), dict) else None),
            serialize_history(history if isinstance(history, list) else None),
        ]
    )


def record_to_prompt_text(record: Dict[str, Any]) -> str:
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    return "\n".join(
        [
            "[CURRENT_PROMPT] " + prompt,
            "[DERIVED] "
            + " ".join(
                [
                    f"history_len={len(history) if isinstance(history, list) else 0}",
                    f"last_action={last_action}",
                    keyword_tokens(prompt),
                ]
            ),
        ]
    )


def aar_last_user_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    if not isinstance(history, list):
        return ""
    for item in reversed(history):
        if isinstance(item, dict) and item.get("role") == "user":
            return _clean_text(item.get("content", ""), 1000)
    return ""


def aar_workspace(record: Dict[str, Any]) -> Dict[str, Any]:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        return {}
    workspace = meta.get("workspace")
    return workspace if isinstance(workspace, dict) else {}


def aar_bucket_number(value: Any, bounds: Sequence[float]) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    for bound in bounds:
        if x <= bound:
            return f"le_{int(bound)}"
    return f"gt_{int(bounds[-1])}"


def aar_keyword_flags(text: str) -> Dict[str, int]:
    lower = text.lower()
    groups = {
        "read_file": ("read", "open", "show", "cat ", "view", "inspect file", "file content"),
        "grep_search": ("grep", "rg ", "search", "find", "reference", "defined", "where is"),
        "list_directory": ("ls", "tree", "folder", "directory", "list files", "structure"),
        "glob_pattern": ("glob", "*.py", "*.js", "*.ts", "*.json", "*.csv", "*.md", "all files", "pattern"),
        "edit_file": ("edit", "fix", "change", "update", "modify", "replace", "refactor"),
        "write_file": ("write file", "create file", "new file", "save as", "generate file"),
        "apply_patch": ("patch", "diff", "apply_patch", "apply patch"),
        "run_bash": ("run ", "execute", "bash", "shell", "terminal", "command", "npm ", "pip ", "python "),
        "run_tests": ("test", "pytest", "unittest", "coverage", "spec", "happy path"),
        "lint_or_typecheck": ("lint", "typecheck", "type check", "mypy", "ruff", "eslint", "tsc "),
        "ask_user": ("?", "which", "choose", "clarify", "confirm", "what do you", "should i"),
        "plan_task": ("plan", "roadmap", "approach", "strategy", "steps", "architecture"),
        "web_search": ("web", "internet", "latest", "today", "news", "lookup", "browse", "search online"),
        "respond_only": ("explain", "tell me", "answer", "summarize", "describe", "why"),
    }
    return {name: int(any(token in lower for token in tokens)) for name, tokens in groups.items()}


def aar_metadata_features(record: Dict[str, Any]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    workspace = aar_workspace(record)
    history = record.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    prompt = _clean_text(record.get("current_prompt", ""), 2500)

    for name in ("user_tier", "language_pref"):
        features[f"{name}={_clean_text(meta.get(name), 40)}"] = 1.0
    for name in ("git_dirty", "last_ci_status"):
        features[f"workspace.{name}={_clean_text(workspace.get(name), 40)}"] = 1.0

    budget = meta.get("budget_tokens_remaining")
    turn = meta.get("turn_index")
    elapsed = meta.get("elapsed_session_sec")
    loc = workspace.get("loc")
    open_files = workspace.get("open_files")
    open_count = len(open_files) if isinstance(open_files, list) else 0

    features[f"budget_bin={aar_bucket_number(budget, [256, 512, 1024, 2048, 4096, 8192, 16384])}"] = 1.0
    features[f"turn_bin={aar_bucket_number(turn, [0, 1, 2, 4, 8, 16, 32])}"] = 1.0
    features[f"elapsed_bin={aar_bucket_number(elapsed, [30, 60, 120, 300, 600, 1200, 2400])}"] = 1.0
    features[f"loc_bin={aar_bucket_number(loc, [100, 1000, 5000, 20000, 100000])}"] = 1.0
    features[f"history_len={history_len}"] = 1.0
    features[f"action_count={len(seq)}"] = 1.0
    features[f"open_count={open_count}"] = 1.0
    features[f"last_action={last_action}"] = 1.0
    if len(seq) >= 2:
        features[f"last2={seq[-2]}>{seq[-1]}"] = 1.0

    language_mix = workspace.get("language_mix")
    if isinstance(language_mix, dict):
        for key, value in language_mix.items():
            try:
                features[f"langmix={key}"] = float(value)
            except (TypeError, ValueError):
                continue

    if isinstance(open_files, list):
        for path in open_files[:8]:
            suffix = str(path).rsplit(".", 1)[-1].lower() if "." in str(path) else "none"
            features[f"open_ext={suffix}"] = 1.0

    features["num_budget_log"] = math.log1p(float(budget or 0.0)) / 12.0
    features["num_turn_log"] = math.log1p(float(turn or 0.0)) / 4.0
    features["num_elapsed_log"] = math.log1p(float(elapsed or 0.0)) / 8.0
    features["num_loc_log"] = math.log1p(float(loc or 0.0)) / 12.0
    features["num_prompt_len"] = min(len(prompt), 2000) / 2000.0
    features["num_prompt_qmark"] = float("?" in prompt)
    features["num_history_len"] = history_len / 12.0
    return features


def aar_meta_text(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    parts: List[str] = []
    for key, value in _flatten_meta(meta):
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = None
        if numeric is not None:
            parts.append(f"{key}_bin={aar_bucket_number(value, [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768])}")
        else:
            parts.append(f"{key}={_clean_text(value, 120)}")
    seq = extract_action_sequence(record.get("history") if isinstance(record.get("history"), list) else [])
    if seq:
        parts.append("last_action=" + seq[-1])
    return "[META] " + " ; ".join(parts)


def aar_action_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    parts = [f"hist_len={len(history) if isinstance(history, list) else 0}"]
    if not seq:
        parts.append("last_action=none")
    for action in seq[-12:]:
        parts.append(f"act_{action}")
    if seq:
        parts.append(f"last_action={seq[-1]}")
    for left, right in zip(seq[-12:], seq[-11:]):
        parts.append(f"pair_{left}>{right}")
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    parts.extend(f"kw_{name}" for name, value in aar_keyword_flags(prompt).items() if value)
    return " ".join(parts)


def aar_prompt_context_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    action_line = "[ACTIONS] " + " > ".join(seq[-8:]) if seq else "[ACTIONS] none"
    return "\n".join(
        [
            "[CURRENT_PROMPT] " + _clean_text(record.get("current_prompt", ""), 2500),
            "[LAST_USER] " + aar_last_user_text(record),
            action_line,
            aar_meta_text(record),
            aar_action_text(record),
        ]
    )


def aar_transition_keys(record: Dict[str, Any]) -> Dict[str, str]:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    workspace = aar_workspace(record)
    history = record.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    flags = aar_keyword_flags(prompt)
    active_flags = [name for name, value in flags.items() if value]
    first_flag = active_flags[0] if active_flags else "none"
    return {
        "last_action": last_action,
        "last2": f"{seq[-2]}>{seq[-1]}" if len(seq) >= 2 else "none",
        "history_len": str(history_len),
        "language_pref": _clean_text(meta.get("language_pref", "none"), 40),
        "ci_dirty": f"{workspace.get('last_ci_status', 'none')}|{workspace.get('git_dirty', 'none')}",
        "prompt_rule": first_flag,
        "last_action_rule": f"{last_action}|{first_flag}",
    }


def aar_transition_predict_proba(spec: Dict[str, Any], records: Sequence[Dict[str, Any]]) -> np.ndarray:
    global_vec = np.asarray(spec["global"], dtype=np.float32)
    weights = spec.get("weights", {})
    groups = spec.get("groups", {})
    out = np.zeros((len(records), len(ACTIONS)), dtype=np.float32)
    for row_idx, record in enumerate(records):
        keys = aar_transition_keys(record)
        total = global_vec * float(spec.get("global_weight", 0.3))
        weight_sum = float(spec.get("global_weight", 0.3))
        for group, weight in weights.items():
            key = keys.get(group)
            values = groups.get(group, {}).get(key)
            if values is None:
                continue
            total += np.asarray(values, dtype=np.float32) * float(weight)
            weight_sum += float(weight)
        out[row_idx] = total / max(weight_sum, 1e-6)
    return out


def aar_views(records: Sequence[Dict[str, Any]], texts: Sequence[str], prompt_texts: Sequence[str]) -> Dict[str, List[Any]]:
    return {
        "full": list(texts),
        "prompt": list(prompt_texts),
        "prompt_context": [aar_prompt_context_text(r) for r in records],
        "history": [],
        "action": [],
        "meta_text": [],
        "meta_dict": [aar_metadata_features(r) for r in records],
        "rule_dict": [],
    }


def _model_classes(model: object) -> Sequence[str] | None:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(x) for x in classes]
    named_steps = getattr(model, "named_steps", None)
    if isinstance(named_steps, dict):
        clf = named_steps.get("clf")
        classes = getattr(clf, "classes_", None)
        if classes is not None:
            return [str(x) for x in classes]
    return None


def predict_proba_aligned(model: object, texts: Sequence[Any]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(texts), dtype=np.float32)
        classes = _model_classes(model)
    else:
        preds = [str(x) for x in model.predict(texts)]
        raw = np.zeros((len(preds), len(ACTIONS)), dtype=np.float32)
        action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
        for row_idx, pred in enumerate(preds):
            if pred in action_to_idx:
                raw[row_idx, action_to_idx[pred]] = 1.0
        classes = list(ACTIONS)

    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    aligned = np.zeros((raw.shape[0], len(ACTIONS)), dtype=np.float32)
    for src_idx, label in enumerate(classes or ACTIONS):
        dst_idx = action_to_idx.get(str(label))
        if dst_idx is not None and src_idx < raw.shape[1]:
            aligned[:, dst_idx] = raw[:, src_idx]
    return aligned


def softmax(scores: np.ndarray, temp: float = 1.0) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64) / max(temp, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_cols(probs: np.ndarray, src_labels: Sequence[str]) -> np.ndarray:
    idx = [list(src_labels).index(action) for action in ACTIONS]
    return np.asarray(probs, dtype=np.float64)[:, idx]


def linear_probs(samples: Sequence[Dict[str, Any]]) -> np.ndarray:
    obj = joblib.load(LINEAR_PKL)
    pipe = obj["pipe"] if isinstance(obj, dict) else obj
    bias_map = obj.get("class_bias") if isinstance(obj, dict) else None
    df = build_linear_dataframe(samples)
    classes = list(pipe.named_steps["clf"].classes_)
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(df)
        if bias_map:
            bias = np.array([float(bias_map.get(str(c), 0.0)) for c in classes])
            scores = scores + bias.reshape(1, -1)
        probs = softmax(scores)
    elif hasattr(pipe, "predict_proba"):
        probs = np.asarray(pipe.predict_proba(df), dtype=np.float64)
    else:
        preds = [str(x) for x in pipe.predict(df)]
        probs = np.zeros((len(preds), len(classes)))
        class_to_idx = {c: i for i, c in enumerate(classes)}
        for row_idx, pred in enumerate(preds):
            if pred in class_to_idx:
                probs[row_idx, class_to_idx[pred]] = 1.0
    return align_cols(probs, classes)


def stacker_probs(samples: Sequence[Dict[str, Any]]) -> np.ndarray:
    with (STACKER_DIR / "aar_config.json").open(encoding="utf-8") as f:
        config = json.load(f)
    if not config.get("enabled"):
        raise ValueError("AAR config disabled")
    artifact = joblib.load(STACKER_DIR / str(config.get("model_file", "aar_models.joblib")))
    texts = [record_to_text(r) for r in samples]
    prompt_texts = [record_to_prompt_text(r) for r in samples]
    views = aar_views(samples, texts, prompt_texts)
    component_probas: Dict[str, np.ndarray] = {}
    for component in config.get("components", []):
        name = str(component.get("name"))
        kind = str(component.get("kind"))
        view = str(component.get("view"))
        if kind == "transition":
            component_probas[name] = aar_transition_predict_proba(artifact["transition"], samples)
        else:
            model = artifact.get("components", {}).get(name)
            if model is None:
                raise ValueError(f"AAR component missing: {name}")
            component_probas[name] = predict_proba_aligned(model, views[view])
    if config.get("use_stacker"):
        names = [str(x) for x in config.get("stacker_components", [])]
        matrix = np.hstack([component_probas[n] for n in names]).astype(np.float32)
        probs = predict_proba_aligned(artifact["stacker"], matrix)
    else:
        total = None
        weight_sum = 0.0
        for component in config.get("components", []):
            weight = float(component.get("weight", 0.0))
            if weight <= 0:
                continue
            arr = component_probas[str(component["name"])]
            total = arr * weight if total is None else total + arr * weight
            weight_sum += weight
        if total is None or weight_sum <= 0:
            raise ValueError("AAR has no positive-weight components")
        probs = total / weight_sum
    return np.asarray(probs, dtype=np.float64)


def _validate_weights(parts: Sequence[Any], source: str) -> Tuple[float, float]:
    if len(parts) != 2:
        raise ValueError(f"{source} must contain 2 weights: linear,stacker")
    try:
        weights = tuple(float(p) for p in parts)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{source} weight parse failed: {parts!r}") from exc
    if any(not math.isfinite(v) for v in weights) or any(v < 0 for v in weights) or sum(weights) <= 0:
        raise ValueError(f"{source} weights must be finite, non-negative, and sum > 0: {weights!r}")
    return weights[0], weights[1]


def parse_weights() -> Tuple[float, float]:
    raw = os.environ.get("ENS_WEIGHTS", "").strip()
    if raw:
        return _validate_weights([p.strip() for p in raw.split(",")], "ENS_WEIGHTS")
    path = MODEL / "weights.json"
    if not path.exists():
        return 1.0, 1.0
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("weights")
    if not isinstance(data, list):
        raise ValueError(f"weights.json must be a list or an object with weights: {path}")
    return _validate_weights(data, "weights.json")


def load_test() -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    with (DATA / "test.jsonl").open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def write_submission(samples: Sequence[Dict[str, Any]], preds: Sequence[str]) -> None:
    id_to_pred = {str(sample["id"]): str(pred) for sample, pred in zip(samples, preds)}
    with (DATA / "sample_submission.csv").open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if fields != ["id", "action"]:
        raise ValueError(f"sample_submission columns must be ['id', 'action']; got {fields}")
    missing = 0
    for row in rows:
        pred = id_to_pred.get(str(row["id"]))
        if pred is None:
            missing += 1
        else:
            row["action"] = pred
    if missing:
        raise ValueError(f"predictions missing for {missing} sample_submission ids")
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "submission.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    start = time.time()
    samples = load_test()
    print(f"[1/4] test rows={len(samples)}")
    print("[2/4] linear")
    linear = linear_probs(samples)
    print("[3/4] stacker")
    stacker = stacker_probs(samples)
    for name, probs in [("linear", linear), ("stacker", stacker)]:
        if probs.shape != (len(samples), len(ACTIONS)):
            raise ValueError(f"{name} probability shape mismatch: {probs.shape}")
    wl, ws = parse_weights()
    blend = (wl * linear + ws * stacker) / (wl + ws)
    preds = np.array(ACTIONS)[blend.argmax(axis=1)]
    bad = sorted(set(map(str, preds)) - set(ACTIONS))
    if bad:
        raise ValueError(f"invalid labels: {bad}")
    print(f"[4/4] write submission (weights linear={wl:g}, stacker={ws:g})")
    write_submission(samples, preds)
    print(f"[DONE] {OUT / 'submission.csv'} rows={len(preds)} elapsed={time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
