# -*- coding: utf-8 -*-
"""Shared helpers for the 2026-07-08 linear replacement lane."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.pipeline import FeatureUnion


ROOT = Path(__file__).resolve().parents[2]
SUBMIT_DIR = ROOT / "submit"
MAIN_ROOT = Path(r"C:\dev\2026-AI-DACON")
DATA_DIR = MAIN_ROOT / "data"
OOF_DIR = MAIN_ROOT / "artifacts" / "oof" / "oof_rebuild_2026_07_04"
OUT_DIR = ROOT / "night_out" / "linear2"
CONTEXT_DIR = ROOT / "context" / "night" / "2026-07-08"

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

BASELINE_SOFT_AU = 0.73877

_STEP_RE = re.compile(r"-step_\d+$")


def session_id(sample_id: str) -> str:
    return _STEP_RE.sub("", str(sample_id))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {path}")


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[save] {path}")


def load_train(data_dir: Path = DATA_DIR) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    samples: list[dict[str, Any]] = []
    with (data_dir / "train.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with (data_dir / "train_labels.csv").open(newline="", encoding="utf-8") as f:
        labels = {str(row["id"]): str(row["action"]) for row in csv.DictReader(f)}
    ids = np.asarray([str(sample["id"]) for sample in samples], dtype=object)
    y = np.asarray([labels[str(sample_id)] for sample_id in ids], dtype=object)
    groups = np.asarray([session_id(str(sample_id)) for sample_id in ids], dtype=object)
    return samples, ids, y, groups


def load_saved_folds(
    oof_dir: Path = OOF_DIR,
    groups: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    payload = read_json(oof_dir / "fold_indices.json")
    folds: list[dict[str, Any]] = []
    for item in payload["folds"]:
        fold = int(item["fold"])
        train_idx = np.asarray(item["train_idx"], dtype=np.int64)
        valid_idx = np.asarray(item.get("valid_idx") or item.get("val_idx") or item.get("test_idx"), dtype=np.int64)
        npy_path = oof_dir / f"fold{fold}_valid_idx.npy"
        if npy_path.exists():
            valid_npy = np.load(npy_path).astype(np.int64)
            if not np.array_equal(valid_idx, valid_npy):
                raise AssertionError(f"fold {fold} JSON valid_idx differs from {npy_path}")
        if groups is not None:
            overlap = set(groups[train_idx]) & set(groups[valid_idx])
            if overlap:
                raise AssertionError(f"fold {fold} group leakage: {len(overlap)} overlapping sessions")
        folds.append({"fold": fold, "train_idx": train_idx, "valid_idx": valid_idx})
    seen = np.concatenate([fold["valid_idx"] for fold in folds])
    if len(seen) != len(set(int(x) for x in seen)):
        raise AssertionError("fold valid_idx coverage contains duplicate rows")
    return folds


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def align_probs(
    probs: np.ndarray,
    src_classes: Sequence[str],
    dst_classes: Sequence[str] = ACTIONS,
    fill_value: float = 0.0,
) -> np.ndarray:
    src = [str(c) for c in src_classes]
    out = np.full((probs.shape[0], len(dst_classes)), fill_value, dtype=np.float64)
    for dst_i, label in enumerate(dst_classes):
        if str(label) in src:
            out[:, dst_i] = probs[:, src.index(str(label))]
    row_sum = out.sum(axis=1, keepdims=True)
    missing = row_sum.ravel() <= 0
    if missing.any():
        out[missing, :] = 1.0 / len(dst_classes)
        row_sum = out.sum(axis=1, keepdims=True)
    return out / row_sum


def labels_from_probs(probs: np.ndarray, actions: Sequence[str] = ACTIONS) -> np.ndarray:
    return np.asarray([str(action) for action in actions], dtype=object)[np.asarray(probs).argmax(axis=1)]


def macro_f1_probs(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str] = ACTIONS) -> float:
    return float(f1_score(y_true, labels_from_probs(probs, actions), labels=list(actions), average="macro", zero_division=0))


def per_class_f1(probs: np.ndarray, y_true: np.ndarray, actions: Sequence[str] = ACTIONS) -> dict[str, float]:
    values = f1_score(y_true, labels_from_probs(probs, actions), labels=list(actions), average=None, zero_division=0)
    return {str(label): float(value) for label, value in zip(actions, values)}


def load_reference_oof(
    oof_dir: Path = OOF_DIR,
) -> tuple[np.ndarray, list[str], list[str], np.ndarray]:
    classes = [str(c) for c in read_json(oof_dir / "classes.json")]
    row_ids = [str(x) for x in read_json(oof_dir / "row_ids.json")]
    y_true = np.asarray([str(x) for x in read_json(oof_dir / "y_true.json")], dtype=object)
    probs = np.load(oof_dir / "linear_probs.npy").astype(np.float64)
    probs = align_probs(probs, classes, ACTIONS)
    return probs, classes, row_ids, y_true


def import_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_au_route_module():
    return import_module_from_path("linear2_au_route", SUBMIT_DIR / "au_route.py")


def compact_value(value: Any, max_chars: int = 4000) -> str:
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
    if max_chars and len(text) > max_chars:
        half = max_chars // 2
        return text[:half] + " ... " + text[-half:]
    return text


def bucket_number(value: Any, edges: Sequence[float]) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "missing"
    if not math.isfinite(x):
        return "missing"
    for edge in edges:
        if x <= edge:
            return f"le_{int(edge)}"
    return f"gt_{int(edges[-1])}"


def extract_action_sequence(history: Any) -> list[str]:
    seq: list[str] = []
    if not isinstance(history, list):
        return seq
    for item in history:
        if not isinstance(item, dict) or item.get("role") != "assistant_action":
            continue
        name = item.get("name") or item.get("action") or item.get("tool")
        if name:
            seq.append(str(name))
    return seq


def serialize_compact(sample: dict[str, Any]) -> str:
    """Compact all-data text view from the discarded task1 char component."""
    meta = sample.get("session_meta") if isinstance(sample.get("session_meta"), dict) else {}
    workspace = meta.get("workspace") if isinstance(meta.get("workspace"), dict) else {}
    history = sample.get("history") if isinstance(sample.get("history"), list) else []
    actions = extract_action_sequence(history)

    action_tokens = " ".join(f"act:{name}" for name in actions) or "act:none"
    recent_tokens = " ".join(f"recent_act:{name}" for name in actions[-8:]) or "recent_act:none"
    pair_tokens = " ".join(f"pair:{left}>{right}" for left, right in zip(actions[-8:], actions[-7:])) or "pair:none"

    open_files = workspace.get("open_files") if isinstance(workspace.get("open_files"), list) else []
    open_ext = []
    for item in open_files[:12]:
        text = str(item)
        open_ext.append(text.rsplit(".", 1)[-1].lower() if "." in text else "none")

    language_mix = workspace.get("language_mix") if isinstance(workspace.get("language_mix"), dict) else {}
    if language_mix:
        top_lang = max(
            language_mix.items(),
            key=lambda kv: kv[1] if isinstance(kv[1], (int, float)) else -1,
        )[0]
    else:
        top_lang = "none"

    meta_parts = [
        f"id_prefix={'sess_au' if str(sample.get('id', '')).startswith('sess_au') else 'sess_sim'}",
        f"user_tier={compact_value(meta.get('user_tier'), 80) or 'none'}",
        f"language_pref={compact_value(meta.get('language_pref'), 80) or 'none'}",
        f"turn_index={compact_value(meta.get('turn_index'), 80) or 'missing'}",
        f"turn_bin={bucket_number(meta.get('turn_index'), (0, 1, 2, 4, 8, 16, 32))}",
        f"elapsed_sec={compact_value(meta.get('elapsed_session_sec'), 80) or 'missing'}",
        f"elapsed_bin={bucket_number(meta.get('elapsed_session_sec'), (30, 60, 120, 300, 600, 1200))}",
        f"budget={compact_value(meta.get('budget_tokens_remaining'), 80) or 'missing'}",
        f"budget_bin={bucket_number(meta.get('budget_tokens_remaining'), (512, 1024, 2048, 4096, 8192, 32768, 131072))}",
        f"workspace_loc={compact_value(workspace.get('loc'), 80) or 'missing'}",
        f"loc_bin={bucket_number(workspace.get('loc'), (100, 1000, 5000, 20000, 100000))}",
        f"git_dirty={int(bool(workspace.get('git_dirty')))}",
        f"last_ci_status={compact_value(workspace.get('last_ci_status'), 80) or 'none'}",
        f"top_lang={compact_value(top_lang, 80)}",
        "language_mix=" + compact_value(language_mix, 500),
        "open_files=" + " ".join(compact_value(x, 240) for x in open_files[:12]),
        "open_ext=" + " ".join(open_ext),
    ]

    return "\n".join(
        [
            "[CURRENT_PROMPT]",
            compact_value(sample.get("current_prompt"), max_chars=0),
            "[HISTORY_ACTIONS]",
            action_tokens,
            recent_tokens,
            pair_tokens,
            f"history_len={len(history)} action_count={len(actions)} last_action={actions[-1] if actions else 'none'}",
            "[SESSION_META]",
            " ".join(meta_parts),
        ]
    )


def serialize_samples(samples: Sequence[dict[str, Any]], mode: str = "compact") -> list[str]:
    if mode == "compact":
        return [serialize_compact(sample) for sample in samples]
    if mode != "au_route":
        raise ValueError(f"unknown serializer: {mode}")
    au_route = load_au_route_module()
    return [au_route.serialize(sample) for sample in samples]


def make_vectorizer(
    *,
    feature_kind: str,
    ngram_min: int,
    ngram_max: int,
    max_features: int,
    word_max_features: int = 80_000,
):
    if feature_kind == "char":
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(ngram_min, ngram_max),
            min_df=1,
            max_features=max_features,
            sublinear_tf=True,
            strip_accents="unicode",
            dtype=np.float32,
        )
    if feature_kind == "word_char":
        return FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        max_features=word_max_features,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        dtype=np.float32,
                    ),
                ),
                (
                    "char",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        ngram_range=(ngram_min, ngram_max),
                        min_df=1,
                        max_features=max_features,
                        sublinear_tf=True,
                        strip_accents="unicode",
                        dtype=np.float32,
                    ),
                ),
            ]
        )
    raise ValueError(f"unknown feature_kind: {feature_kind}")


def load_league4_common():
    return import_module_from_path("linear2_league4_common", ROOT / "scripts" / "league4" / "common.py")


def evaluate_lin_replacement(
    *,
    lin_probs_all: np.ndarray,
    row_ids: Sequence[str],
    summary_prefix: str = "",
) -> dict[str, Any]:
    league = load_league4_common()
    data = league.load_league_data()
    row_index = {str(sample_id): i for i, sample_id in enumerate(row_ids)}
    rows = np.asarray([row_index[str(sample_id)] for sample_id in data.ids], dtype=np.int64)
    candidate_lin = np.asarray(lin_probs_all, dtype=np.float64)[rows]
    candidate_lin = league.align_probs(candidate_lin, ACTIONS, data.actions)
    data_candidate = replace(data, lin=candidate_lin)

    au = league.train_or_load_au_probs(data)
    baseline_soft = league.apply_soft_au(data, league.four_way_blend(data), au["probs"])
    candidate_soft = league.apply_soft_au(data_candidate, league.four_way_blend(data_candidate), au["probs"])
    baseline_score = league.macro_f1_probs(baseline_soft, data.y_true, data.actions)
    candidate_score = league.macro_f1_probs(candidate_soft, data.y_true, data.actions)
    return {
        f"{summary_prefix}baseline_soft_au_macro_f1": float(baseline_score),
        f"{summary_prefix}league_macro_f1": float(candidate_score),
        f"{summary_prefix}delta_vs_baseline_soft_au": float(candidate_score - baseline_score),
        f"{summary_prefix}half_scores": league.half_scores(data_candidate, candidate_soft),
        f"{summary_prefix}candidate_lin_solo_holdout_macro_f1": league.macro_f1_probs(
            candidate_lin, data.y_true, data.actions
        ),
        f"{summary_prefix}reference_lin_solo_holdout_macro_f1": league.macro_f1_probs(
            data.lin, data.y_true, data.actions
        ),
        f"{summary_prefix}lin_argmax_disagreement_vs_reference": float(
            np.mean(league.predict_from_probs(candidate_lin, data.actions) != league.predict_from_probs(data.lin, data.actions))
        ),
        f"{summary_prefix}holdout_rows": int(len(data.ids)),
        f"{summary_prefix}soft_au_cache_hit": bool(au.get("cache_hit", False)),
    }


def write_progress(
    *,
    path: Path,
    rows: Sequence[dict[str, Any]],
    next_step: str,
    note: str = "",
) -> None:
    lines = [
        "# PROGRESS-task2",
        "",
        "- worktree: `C:\\dev\\night\\2026-07-08\\task2`",
        "- lane: linear replacement candidates; outputs under `night_out/linear2/`",
        "- fold contract: `C:\\dev\\2026-AI-DACON\\artifacts\\oof\\oof_rebuild_2026_07_04\\fold_indices.json`",
        "",
        "## Status",
        "",
    ]
    if note:
        lines.extend([note, ""])
    if rows:
        lines.append("| variant | oof_f1 | league | delta | decision |")
        lines.append("|---|---:|---:|---:|---|")
        for row in rows:
            lines.append(
                "| {variant} | {oof:.6f} | {league:.6f} | {delta:+.6f} | {decision} |".format(
                    variant=row.get("variant", ""),
                    oof=float(row.get("oof_macro_f1", float("nan"))),
                    league=float(row.get("league_macro_f1", float("nan"))),
                    delta=float(row.get("delta_vs_baseline_soft_au", float("nan"))),
                    decision=row.get("decision", ""),
                )
            )
    else:
        lines.append("- No sweep variant completed yet.")
    lines.extend(["", "## Next resume point", "", next_step, ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[save] {path}")
