from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd


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


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
            if not isinstance(obj, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_no}")
            rows.append(obj)
    return rows


def find_first_existing(candidates: Iterable[str | Path]) -> Path:
    for cand in candidates:
        p = Path(cand)
        if p.exists():
            return p
    raise FileNotFoundError("None of these paths exist: " + ", ".join(map(str, candidates)))


def load_test() -> List[Dict[str, Any]]:
    test_jsonl = find_first_existing([
        "data/test.jsonl",
        "open/test.jsonl",
        "open/data/test.jsonl",
        "./test.jsonl",
    ])
    return read_jsonl(test_jsonl)


def _clean_text(value: Any, max_chars: int = 1200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            s = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            s = str(value)
    else:
        s = str(value)
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) > max_chars:
        return s[: max_chars // 2] + " ... " + s[-max_chars // 2 :]
    return s


def _flatten_meta(obj: Any, prefix: str = "") -> List[Tuple[str, str]]:
    items: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in sorted(obj.items(), key=lambda kv: str(kv[0])):
            key = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(v, dict):
                items.extend(_flatten_meta(v, key))
            elif isinstance(v, list):
                if len(v) <= 8:
                    items.append((key, "|".join(_clean_text(x, 80) for x in v)))
                else:
                    first = "|".join(_clean_text(x, 60) for x in v[:5])
                    items.append((key, f"list_len={len(v)} first={first}"))
            else:
                items.append((key, _clean_text(v, 120)))
    return items


def serialize_session_meta(meta: Dict[str, Any] | None) -> str:
    if not isinstance(meta, dict):
        return "[SESSION_META] none"
    parts = [f"{k}={v}" for k, v in _flatten_meta(meta)]
    return "[SESSION_META] " + " ; ".join(parts)


def extract_action_sequence(history: Iterable[Dict[str, Any]]) -> List[str]:
    seq: List[str] = []
    for h in history or []:
        if isinstance(h, dict) and h.get("role") == "assistant_action":
            name = h.get("name") or h.get("action") or h.get("tool")
            if name:
                seq.append(str(name))
    return seq


def serialize_history(history: List[Dict[str, Any]] | None, max_turns: int = 12) -> str:
    if not isinstance(history, list) or not history:
        return "[HISTORY] empty"
    recent = history[-max_turns:]
    lines: List[str] = ["[HISTORY_RECENT]"]
    for i, h in enumerate(recent):
        if not isinstance(h, dict):
            lines.append(f"turn_{i}: {_clean_text(h, 500)}")
            continue
        role = _clean_text(h.get("role", "unknown"), 50)
        if role == "assistant_action":
            name = _clean_text(h.get("name") or h.get("action") or h.get("tool") or "unknown", 80)
            args = _clean_text(h.get("args", ""), 500)
            result = _clean_text(h.get("result_summary", h.get("result", "")), 500)
            lines.append(f"turn_{i}: role=assistant_action action={name} args={args} result={result}")
        else:
            content = _clean_text(h.get("content", h), 800)
            lines.append(f"turn_{i}: role={role} content={content}")
    seq = extract_action_sequence(history)
    if seq:
        lines.append("[ACTION_SEQUENCE] " + " > ".join(seq[-16:]))
        lines.append("[LAST_ACTION] " + seq[-1])
    return "\n".join(lines)


def keyword_tokens(prompt: str) -> str:
    p = prompt.lower()
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
    tokens: List[str] = []
    for token, words in pairs.items():
        if any(w in p for w in words):
            tokens.append(token)
    return " ".join(tokens)


def record_to_text(record: Dict[str, Any]) -> str:
    current_prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    meta = record.get("session_meta")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    feature_line = "[DERIVED] " + " ".join([
        f"history_len={len(history) if isinstance(history, list) else 0}",
        f"last_action={last_action}",
        keyword_tokens(current_prompt),
    ])
    return "\n".join([
        f"[ID] {_clean_text(record.get('id', ''), 120)}",
        "[CURRENT_PROMPT] " + current_prompt,
        feature_line,
        serialize_session_meta(meta if isinstance(meta, dict) else None),
        serialize_history(history if isinstance(history, list) else None),
    ])


def record_to_prompt_text(record: Dict[str, Any]) -> str:
    current_prompt = _clean_text(record.get("current_prompt", ""), 2500)
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    last_action = seq[-1] if seq else "none"
    return "\n".join([
        "[CURRENT_PROMPT] " + current_prompt,
        "[DERIVED] " + " ".join([
            f"history_len={len(history) if isinstance(history, list) else 0}",
            f"last_action={last_action}",
            keyword_tokens(current_prompt),
        ]),
    ])


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


def predict_proba_aligned(model: object, texts: Sequence[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        raw = np.asarray(model.predict_proba(texts), dtype=np.float32)
        classes = _model_classes(model)
    else:
        preds = [str(x) for x in model.predict(texts)]
        raw = np.zeros((len(preds), len(ACTIONS)), dtype=np.float32)
        action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
        for i, pred in enumerate(preds):
            if pred in action_to_idx:
                raw[i, action_to_idx[pred]] = 1.0
        classes = list(ACTIONS)

    action_to_idx = {a: i for i, a in enumerate(ACTIONS)}
    aligned = np.zeros((raw.shape[0], len(ACTIONS)), dtype=np.float32)
    for src_idx, label in enumerate(classes or ACTIONS):
        dst_idx = action_to_idx.get(str(label))
        if dst_idx is not None and src_idx < raw.shape[1]:
            aligned[:, dst_idx] = raw[:, src_idx]
    return aligned


def weighted_average(parts: Iterable[Tuple[np.ndarray, float]]) -> np.ndarray:
    total: np.ndarray | None = None
    weight_sum = 0.0
    for proba, weight in parts:
        if weight <= 0:
            continue
        arr = np.asarray(proba, dtype=np.float32)
        total = arr * weight if total is None else total + arr * weight
        weight_sum += weight
    if total is None or weight_sum <= 0:
        raise ValueError("At least one positive-weight probability matrix is required.")
    return total / weight_sum


def labels_from_proba(proba: np.ndarray) -> List[str]:
    indices = np.asarray(proba).argmax(axis=1)
    return [ACTIONS[int(i)] for i in indices]


def load_config(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


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
        "glob_pattern": ("glob", "*.py", "*.js", "*.ts", "*.json", "all files", "pattern"),
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


def aar_rule_features(record: Dict[str, Any]) -> Dict[str, float]:
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    last_user = aar_last_user_text(record)
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    joined = prompt + "\n" + last_user
    flags = aar_keyword_flags(joined)
    lower = joined.lower()
    features: Dict[str, float] = {f"rule_{key}": float(value) for key, value in flags.items()}
    features["has_code_fence"] = float("```" in joined)
    features["has_file_path"] = float(bool(re.search(r"[\w./\\-]+\.(py|js|ts|tsx|json|csv|md|yml|yaml|txt)", lower)))
    features["has_shell_op"] = float(any(x in lower for x in ("&&", "||", "npm ", "pip ", "pytest", "python ")))
    features["has_latest_word"] = float(any(x in lower for x in ("latest", "today", "current", "recent")))
    features["starts_question"] = float(lower.strip().startswith(("what", "why", "how", "which", "can ", "should ")))
    if seq:
        features[f"after_{seq[-1]}"] = 1.0
    return features


def aar_history_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    prompt = _clean_text(record.get("current_prompt", ""), 2500)
    if not isinstance(history, list) or not history:
        return "[HISTORY] empty\n[CURRENT_PROMPT] " + prompt
    lines = ["[CURRENT_PROMPT] " + prompt, "[HISTORY_FOCUSED]"]
    for idx, item in enumerate(history[-10:]):
        if not isinstance(item, dict):
            lines.append(f"turn_{idx} raw={_clean_text(item, 400)}")
            continue
        role = _clean_text(item.get("role", "unknown"), 40)
        if role == "assistant_action":
            name = _clean_text(item.get("name") or item.get("action") or item.get("tool") or "unknown", 80)
            result = _clean_text(item.get("result_summary", item.get("result", "")), 500)
            lines.append(f"assistant_action={name} result={result}")
        else:
            lines.append(f"{role}={_clean_text(item.get('content', ''), 700)}")
    return "\n".join(lines)


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


def aar_meta_text(record: Dict[str, Any]) -> str:
    meta = record.get("session_meta")
    if not isinstance(meta, dict):
        meta = {}
    parts: List[str] = []
    for key, value in _flatten_meta(meta):
        if isinstance(value, (int, float)):
            parts.append(f"{key}_bin={aar_bucket_number(value, [0, 1, 2, 4, 8, 16, 32, 64, 128, 512, 2048, 8192, 32768])}")
        else:
            parts.append(f"{key}={_clean_text(value, 120)}")
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    if seq:
        parts.append("last_action=" + seq[-1])
    return "[META] " + " ; ".join(parts)


def aar_prompt_context_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    seq = extract_action_sequence(history if isinstance(history, list) else [])
    action_line = "[ACTIONS] " + " > ".join(seq[-8:]) if seq else "[ACTIONS] none"
    return "\n".join([
        "[CURRENT_PROMPT] " + _clean_text(record.get("current_prompt", ""), 2500),
        "[LAST_USER] " + aar_last_user_text(record),
        action_line,
        aar_meta_text(record),
        aar_action_text(record),
    ])


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
        "history": [aar_history_text(r) for r in records],
        "action": [aar_action_text(r) for r in records],
        "meta_text": [aar_meta_text(r) for r in records],
        "meta_dict": [aar_metadata_features(r) for r in records],
        "rule_dict": [aar_rule_features(r) for r in records],
    }


def aar_apply_bias(proba: np.ndarray, bias: Sequence[float]) -> np.ndarray:
    logits = np.log(np.clip(proba, 1e-9, 1.0)) + np.asarray(bias, dtype=np.float32)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_aar(
    records: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
    config: Dict[str, Any],
) -> List[str]:
    model_file = str(config.get("model_file", "aar_models.joblib"))
    artifact = joblib.load(Path("model") / model_file)
    views = aar_views(records, texts, prompt_texts)
    component_probas: Dict[str, np.ndarray] = {}
    for component in config.get("components", []):
        name = str(component.get("name"))
        kind = str(component.get("kind"))
        view = str(component.get("view"))
        if kind == "transition":
            transition = artifact.get("transition")
            if not isinstance(transition, dict):
                raise ValueError("AAR transition component is missing.")
            component_probas[name] = aar_transition_predict_proba(transition, records)
        else:
            model = artifact.get("components", {}).get(name)
            if model is None:
                raise ValueError(f"AAR component is missing: {name}")
            component_probas[name] = predict_proba_aligned(model, views[view])

    if config.get("use_stacker"):
        stacker = artifact.get("stacker")
        if stacker is None:
            raise ValueError("AAR stacker is missing.")
        names = [str(x) for x in config.get("stacker_components", [])]
        matrix = np.hstack([component_probas[name] for name in names]).astype(np.float32)
        probas = predict_proba_aligned(stacker, matrix)
    else:
        parts = []
        for component in config.get("components", []):
            name = str(component.get("name"))
            parts.append((component_probas[name], float(component.get("weight", 0.0))))
        probas = weighted_average(parts)

    if config.get("use_bias"):
        probas = aar_apply_bias(probas, config.get("class_bias", [0.0] * len(ACTIONS)))
    return labels_from_proba(probas)


def try_aar_predictions(
    records: Sequence[Dict[str, Any]],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
) -> List[str] | None:
    config = load_config("model/aar_config.json")
    if not config.get("enabled"):
        return None
    try:
        return predict_aar(records, texts, prompt_texts, config)
    except Exception as exc:
        print(f"[WARN] AAR disabled at inference: {exc}")
        return None


def sample_submission_path() -> Path | None:
    for cand in [
        Path("data/sample_submission.csv"),
        Path("open/data/sample_submission.csv"),
        Path("sample_submission.csv"),
    ]:
        if cand.exists():
            return cand
    return None


def write_submission(ids: List[str], preds: List[str], output_path: Path) -> None:
    if len(ids) != len(preds):
        raise ValueError(f"ids/preds length mismatch: ids={len(ids)} preds={len(preds)}")

    sample_path = sample_submission_path()
    if sample_path is None:
        sub = pd.DataFrame({"id": ids, "action": preds})
    else:
        sub = pd.read_csv(sample_path)
        if not {"id", "action"}.issubset(sub.columns):
            raise ValueError(f"sample_submission.csv must contain id, action columns: {sample_path}")
        pred_map = dict(zip(ids, preds))
        if len(pred_map) != len(ids):
            raise ValueError("Duplicate ids detected in test.jsonl")
        sub["id"] = sub["id"].astype(str)
        missing_ids = sorted(set(sub["id"]) - set(pred_map))
        extra_ids = sorted(set(pred_map) - set(sub["id"]))
        if missing_ids or extra_ids:
            raise ValueError(
                "sample_submission ids do not match test ids: "
                f"missing={len(missing_ids)} extra={len(extra_ids)}"
            )
        sub["action"] = sub["id"].map(pred_map)
    sub.to_csv(output_path, index=False)
    print(f"saved {output_path} rows={len(sub)}")


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path("model/model.joblib")
    prompt_model_path = Path("model/prompt_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError("model/model.joblib not found.")

    records = load_test()
    ids = [str(r.get("id", f"test_{i:06d}")) for i, r in enumerate(records)]
    texts = [record_to_text(r) for r in records]
    prompt_texts = [record_to_prompt_text(r) for r in records]

    final_preds = try_aar_predictions(records, texts, prompt_texts)
    if final_preds is None:
        config = load_config("model/ensemble_config.json")
        model = joblib.load(model_path)
        parts = [(predict_proba_aligned(model, texts), float(config.get("tfidf_weight", 1.0)))]

        if prompt_model_path.exists():
            prompt_model = joblib.load(prompt_model_path)
            parts.append((
                predict_proba_aligned(prompt_model, prompt_texts),
                float(config.get("prompt_weight", 0.45)),
            ))

        probas = weighted_average(parts)
        final_preds = labels_from_proba(probas)

    bad = sorted(set(final_preds) - set(ACTIONS))
    if bad:
        raise ValueError(f"Invalid predicted labels: {bad}")

    write_submission(ids, final_preds, output_dir / "submission.csv")


if __name__ == "__main__":
    main()
