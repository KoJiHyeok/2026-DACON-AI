from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import joblib
import numpy as np
import torch
from scipy import sparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.au_route import is_au, predict_proba as au_predict_proba
from src.constants import ACTIONS
from src.sprint080_features import feature_dicts
from src.sprint080_serialization import serialize_record


ENCODER_DIR_STR = "model/enc_v2_s42_pro"
ENCODER_DIR = Path(ENCODER_DIR_STR)
MBERT_DIR_STR = "model/mbert_full"  # merge080: mBERT full-train fp16 (id2label 알파벳순 — 이름으로 재정렬)
ATTACK_MODEL_PATH = Path("model/attack_model.joblib")
ATTACK_CONFIG_PATH = Path("model/attack_config.json")
AU_MODEL_PATH = Path("model/au_route/model.pkl")
MAX_LEN = int(os.environ.get("ENCODER_MAX_LEN", "384"))
DEFAULT_BATCH_SIZE = 64


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {exc}") from exc
    return rows


def find_first_existing(candidates: Iterable[str | Path]) -> Path:
    for cand in candidates:
        path = Path(cand)
        if path.exists():
            return path
    raise FileNotFoundError("None of these paths exist: " + ", ".join(str(x) for x in candidates))


def load_test_records() -> List[Dict[str, Any]]:
    return read_jsonl(find_first_existing(["data/test.jsonl", "open/data/test.jsonl", "open/test.jsonl", "test.jsonl"]))


def load_sample_submission() -> List[str]:
    sample_path = find_first_existing(
        ["data/sample_submission.csv", "open/data/sample_submission.csv", "open/sample_submission.csv", "sample_submission.csv"]
    )
    with sample_path.open("r", encoding="utf-8", newline="") as f:
        return [str(row["id"]) for row in csv.DictReader(f)]


def validate_action_order(model_dir: Path) -> None:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError("model/enc_v2_s42_pro/config.json is missing")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    id2label = config.get("id2label")
    if not isinstance(id2label, dict):
        raise RuntimeError("config.json id2label is missing")
    labels = [str(id2label.get(str(i), id2label.get(i))) for i in range(len(id2label))]
    if labels != ACTIONS:
        raise RuntimeError(f"label order mismatch: expected={ACTIONS} got={labels}")


def validate_mbert(config: Dict[str, Any]) -> None:
    """mbert.mix>0이면 시작 시점에 fail-fast (joblib 로드 ~1분 이후에 죽지 않도록)."""
    mbert_cfg = config.get("mbert") or {}
    mix = float(os.environ.get("MBERT_MIX", mbert_cfg.get("mix", 0.0)))
    if mix <= 0.0:
        return
    mbert_dir = Path(str(mbert_cfg.get("dir", MBERT_DIR_STR)))
    if not mbert_dir.exists():
        raise FileNotFoundError(f"{mbert_dir} is missing (mbert.mix={mix})")
    if not any((mbert_dir / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError(f"{mbert_dir} weight file is missing")
    config_path = mbert_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} is missing")
    id2label = json.loads(config_path.read_text(encoding="utf-8")).get("id2label") or {}
    labels = sorted(str(v) for v in id2label.values())
    if labels != sorted(ACTIONS):
        raise RuntimeError(f"mbert label set mismatch: {labels}")


def validate_files() -> None:
    if not ENCODER_DIR.exists():
        raise FileNotFoundError("model/enc_v2_s42_pro folder is missing")
    if not ATTACK_MODEL_PATH.exists():
        raise FileNotFoundError("model/attack_model.joblib is missing")
    if not ATTACK_CONFIG_PATH.exists():
        raise FileNotFoundError("model/attack_config.json is missing")
    if not AU_MODEL_PATH.exists():
        raise FileNotFoundError("model/au_route/model.pkl is missing")
    if not any((ENCODER_DIR / name).exists() for name in ("model.safetensors", "pytorch_model.bin")):
        raise FileNotFoundError("encoder weight file is missing")
    if not any((ENCODER_DIR / name).exists() for name in ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model", "tokenizer_config.json")):
        raise FileNotFoundError("encoder tokenizer files are missing")
    validate_action_order(ENCODER_DIR)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def batch_size_candidates() -> List[int]:
    preferred = int(os.environ.get("ENCODER_BATCH_SIZE", str(DEFAULT_BATCH_SIZE)))
    candidates = [preferred]
    for fallback in (64, 32, 16):
        if fallback < preferred and fallback not in candidates:
            candidates.append(fallback)
        elif "ENCODER_BATCH_SIZE" not in os.environ and fallback not in candidates:
            candidates.append(fallback)
    return candidates


def _is_cuda_oom(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or "cuda oom" in str(exc).lower()


def encoder_predict(records: List[Dict[str, Any]], encoder_dir: str = ENCODER_DIR_STR, align_by_label: bool = False) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(encoder_dir, local_files_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.half().to(device)
    else:
        model = model.float().to(device)
    model.eval()
    texts = [serialize_record(record, "e5_compatible") for record in records]
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    for batch_size in batch_size_candidates():
        out = np.zeros((len(texts), len(ACTIONS)), dtype=np.float64)
        print(f"encoder_runtime device={device}")
        print(f"encoder_runtime gpu_name={gpu_name}")
        print(f"encoder_runtime model_dtype={next(model.parameters()).dtype}")
        print(f"encoder_runtime actual_model_device={next(model.parameters()).device}")
        print(f"encoder_runtime batch_size={batch_size}")
        print(f"encoder_runtime max_length={MAX_LEN}")
        print(f"encoder_runtime num_test_rows={len(texts)}")
        print(f"encoder_runtime estimated_batches={math.ceil(len(texts) / max(batch_size, 1))}")
        started = time.time()
        try:
            with torch.inference_mode():
                for start in range(0, len(texts), batch_size):
                    batch = texts[start : start + batch_size]
                    enc = tokenizer(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
                    enc = {k: v.to(device) for k, v in enc.items()}
                    logits = model(**enc).logits.float().cpu().numpy()
                    out[start : start + len(batch)] = softmax(logits)
            elapsed = time.time() - started
            print(f"encoder_runtime actual_batch_size={batch_size}")
            print(f"encoder_runtime elapsed_sec={elapsed:.3f}")
            if align_by_label:
                id2label = model.config.id2label
                labels = [str(id2label.get(i, id2label.get(str(i)))) for i in range(len(id2label))]
                if sorted(labels) != sorted(ACTIONS):
                    raise RuntimeError(f"{encoder_dir} label set mismatch: {labels}")
                out = out[:, [labels.index(a) for a in ACTIONS]]
            return out
        except RuntimeError as exc:
            if device == "cuda" and _is_cuda_oom(exc):
                print(f"WARNING: CUDA OOM at batch_size={batch_size}; retrying smaller batch")
                torch.cuda.empty_cache()
                continue
            raise
    raise RuntimeError("CUDA OOM after batch size fallbacks 64/32/16")


def sparse_predict(model: Any, records: List[Dict[str, Any]]) -> np.ndarray:
    texts = [serialize_record(record, "e5_compatible") for record in records]
    raw = model.predict_proba(texts)
    out = np.zeros((len(records), len(ACTIONS)), dtype=np.float64)
    classes = getattr(model.named_steps.get("sgdclassifier"), "classes_", None)
    if classes is None:
        classes = np.arange(len(ACTIONS))
    for src_idx, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / len(ACTIONS)), where=row_sum > 0)


def top2_margin(proba: np.ndarray) -> np.ndarray:
    part = np.partition(proba, -2, axis=1)
    return part[:, -1] - part[:, -2]


def entropy(proba: np.ndarray) -> np.ndarray:
    return -(proba * np.log(proba + 1e-12)).sum(axis=1)


def aligned_model_proba(model: Any, x: Any) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((x.shape[0], len(ACTIONS)), dtype=np.float64)
    for src_idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, src_idx]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, row_sum, out=np.full_like(out, 1.0 / len(ACTIONS)), where=row_sum > 0)


def stacker_predict(payload: Dict[str, Any], records: List[Dict[str, Any]], baseline: np.ndarray, e5: np.ndarray) -> np.ndarray:
    vec = payload["dict_vectorizer"]
    struct = vec.transform(feature_dicts(records))
    numeric = np.hstack(
        [
            baseline,
            e5,
            baseline.max(axis=1, keepdims=True),
            e5.max(axis=1, keepdims=True),
            top2_margin(baseline).reshape(-1, 1),
            top2_margin(e5).reshape(-1, 1),
            entropy(baseline).reshape(-1, 1),
            entropy(e5).reshape(-1, 1),
        ]
    )
    x = sparse.hstack([sparse.csr_matrix(numeric), struct], format="csr")
    return aligned_model_proba(payload["stacker"], x)


def apply_fix(base: np.ndarray, cfg: Dict[str, Any]) -> np.ndarray:
    score = np.log(base + 1e-12)
    for action, key in [("read_file", "read_bias"), ("grep_search", "grep_bias"), ("list_directory", "list_bias")]:
        if key in cfg:
            score[:, ACTIONS.index(action)] += float(cfg[key])
    score = score - score.max(axis=1, keepdims=True)
    exp = np.exp(score)
    return exp / exp.sum(axis=1, keepdims=True)


def apply_au_route(records: List[Dict[str, Any]], proba: np.ndarray, alpha: float) -> np.ndarray:
    au_idx = [i for i, record in enumerate(records) if is_au(record.get("id", ""))]
    if not au_idx:
        print("au_route rows=0")
        return proba
    artifact = joblib.load(AU_MODEL_PATH)
    au_proba, _ = au_predict_proba(artifact, [records[i] for i in au_idx])
    out = proba.copy()
    changed = 0
    for k, i in enumerate(au_idx):
        mixed = alpha * au_proba[k] + (1.0 - alpha) * proba[i]
        mixed = mixed / max(float(mixed.sum()), 1e-12)
        if int(mixed.argmax()) != int(proba[i].argmax()):
            changed += 1
        out[i] = mixed
    print(f"au_route alpha={alpha:g} rows={len(au_idx)} changed={changed}")
    return out


def write_submission(records: List[Dict[str, Any]], proba: np.ndarray) -> Path:
    sample_ids = load_sample_submission()
    record_by_id = {str(record.get("id")): idx for idx, record in enumerate(records)}
    if len(sample_ids) != len(records):
        raise RuntimeError(f"sample_submission row count mismatch: sample={len(sample_ids)} test={len(records)}")
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "submission.csv"
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "action"])
        for rid in sample_ids:
            idx = record_by_id.get(rid)
            if idx is None:
                raise RuntimeError(f"sample id missing in test: {rid}")
            writer.writerow([rid, ACTIONS[int(np.argmax(proba[idx]))]])
    return output_path


def main() -> None:
    validate_files()
    config = json.loads(ATTACK_CONFIG_PATH.read_text(encoding="utf-8"))
    validate_mbert(config)
    records = load_test_records()
    payload = joblib.load(ATTACK_MODEL_PATH)
    if payload.get("actions") != ACTIONS:
        raise RuntimeError("attack_model action order mismatch")

    e5 = encoder_predict(records)
    baseline = sparse_predict(payload["sparse_model"], records)
    stacker = stacker_predict(payload, records, baseline, e5)
    final = apply_fix(stacker, config.get("fix", {}))
    mbert_cfg = config.get("mbert") or {}
    mix = float(os.environ.get("MBERT_MIX", mbert_cfg.get("mix", 0.0)))
    if mix > 0.0:
        mbert_dir = str(mbert_cfg.get("dir", MBERT_DIR_STR))
        if not Path(mbert_dir).exists():
            raise FileNotFoundError(f"{mbert_dir} is missing (mbert.mix={mix})")
        mbert = encoder_predict(records, mbert_dir, align_by_label=True)
        final = (1.0 - mix) * final + mix * mbert
        final = final / final.sum(axis=1, keepdims=True)
        print(f"mbert_mix w={mix:g} dir={mbert_dir}")
    else:
        print("mbert_mix off")
    alpha = float(config.get("au_alpha", os.environ.get("ENS_AU_ALPHA", "0.8")))
    final = apply_au_route(records, final, alpha)
    out = write_submission(records, final)
    print(f"saved {out} rows={len(records)}")


if __name__ == "__main__":
    main()

