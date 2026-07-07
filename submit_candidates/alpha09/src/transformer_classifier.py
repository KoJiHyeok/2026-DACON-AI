from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.metrics import f1_score

from .constants import ACTIONS, ACTION_TO_ID


def _require_stack() -> dict[str, Any]:
    try:
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            get_linear_schedule_with_warmup,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Transformer training/inference requires torch and transformers. "
            "Install them locally or run on the DACON environment that provides them."
        ) from exc
    return {
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def _set_seed(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_metadata(model_dir: Path) -> dict[str, Any]:
    metadata_path = model_dir / "transformer_metadata.json"
    if not metadata_path.exists():
        return {}
    with metadata_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def train_transformer_classifier(
    train_texts: Sequence[str],
    train_labels: Sequence[str],
    valid_texts: Sequence[str],
    valid_labels: Sequence[str],
    model_dir: str | Path,
    *,
    model_name: str = "xlm-roberta-base",
    max_length: int = 384,
    batch_size: int = 8,
    eval_batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.06,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
    fp16: bool = False,
    save_fp16: bool = True,
) -> dict[str, Any]:
    stack = _require_stack()
    torch = stack["torch"]
    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForSequenceClassification = stack["AutoModelForSequenceClassification"]
    get_linear_schedule_with_warmup = stack["get_linear_schedule_with_warmup"]

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _set_seed(torch, seed)

    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(ACTIONS),
        id2label={i: label for i, label in enumerate(ACTIONS)},
        label2id={label: i for i, label in enumerate(ACTIONS)},
    )

    class TextDataset(torch.utils.data.Dataset):
        def __init__(self, texts: Sequence[str], labels: Sequence[str]) -> None:
            self.texts = list(texts)
            self.labels = [ACTION_TO_ID[str(label)] for label in labels]

        def __len__(self) -> int:
            return len(self.texts)

        def __getitem__(self, idx: int) -> tuple[str, int]:
            return self.texts[idx], self.labels[idx]

    def collate(batch: Sequence[tuple[str, int]]) -> dict[str, Any]:
        texts, labels = zip(*batch)
        encoded = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(labels, dtype=torch.long)
        return encoded

    train_loader = torch.utils.data.DataLoader(
        TextDataset(train_texts, train_labels),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    grad_accum = max(1, gradient_accumulation_steps)
    update_steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, update_steps_per_epoch * epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    use_amp = bool(fp16 and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    best_macro = -1.0

    print(
        "[INFO] transformer "
        f"model={model_name} device={device} train={len(train_texts)} valid={len(valid_texts)}"
    )
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(**batch)
                loss = outputs.loss / grad_accum
            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * grad_accum

            is_update_step = step % grad_accum == 0 or step == len(train_loader)
            if is_update_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        valid_proba = transformer_predict_proba_from_loaded(
            model,
            tokenizer,
            valid_texts,
            max_length=max_length,
            batch_size=eval_batch_size,
            torch_module=torch,
            device=device,
        )
        pred_labels = [ACTIONS[int(i)] for i in valid_proba.argmax(axis=1)]
        macro = f1_score(valid_labels, pred_labels, average="macro")
        avg_loss = running_loss / max(1, len(train_loader))
        print(f"[VALID][transformer] epoch={epoch} loss={avg_loss:.6f} macro_f1={macro:.6f}")

        if macro >= best_macro:
            best_macro = float(macro)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            with (model_dir / "transformer_metadata.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "actions": ACTIONS,
                        "model_name": model_name,
                        "max_length": max_length,
                        "valid_macro_f1": best_macro,
                        "epochs": epochs,
                        "learning_rate": learning_rate,
                        "batch_size": batch_size,
                        "eval_batch_size": eval_batch_size,
                        "seed": seed,
                        "saved_dtype": "float32",
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"[INFO] saved best transformer: {model_dir}")

    if save_fp16:
        best_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        best_model.half()
        best_model.save_pretrained(model_dir)
        metadata = _read_metadata(model_dir)
        metadata["saved_dtype"] = "float16"
        with (model_dir / "transformer_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        del best_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[INFO] converted transformer weights to fp16 for submission: {model_dir}")

    return {
        "path": str(model_dir),
        "model_name": model_name,
        "max_length": max_length,
        "valid_macro_f1": best_macro,
        "saved_dtype": "float16" if save_fp16 else "float32",
    }


def transformer_predict_proba(
    model_dir: str | Path,
    texts: Sequence[str],
    *,
    max_length: int | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    stack = _require_stack()
    torch = stack["torch"]
    AutoTokenizer = stack["AutoTokenizer"]
    AutoModelForSequenceClassification = stack["AutoModelForSequenceClassification"]

    model_dir = Path(model_dir)
    metadata = _read_metadata(model_dir)
    if max_length is None:
        max_length = int(metadata.get("max_length", 384))

    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, use_fast=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch_dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        torch_dtype=torch_dtype,
    )
    model.to(device)

    return transformer_predict_proba_from_loaded(
        model,
        tokenizer,
        texts,
        max_length=max_length,
        batch_size=batch_size,
        torch_module=torch,
        device=device,
    )


def transformer_predict_proba_from_loaded(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    batch_size: int,
    torch_module: Any,
    device: Any,
) -> np.ndarray:
    torch = torch_module
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            chunks.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())

    raw = np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, len(ACTIONS)), dtype=np.float32)
    return _align_transformer_proba(raw, getattr(model, "config", None))


def _align_transformer_proba(raw: np.ndarray, config: Any) -> np.ndarray:
    if raw.shape[1] == len(ACTIONS) and config is None:
        return raw

    id2label = getattr(config, "id2label", {}) if config is not None else {}
    action_to_idx = {label: i for i, label in enumerate(ACTIONS)}
    aligned = np.zeros((raw.shape[0], len(ACTIONS)), dtype=np.float32)
    for src_idx in range(raw.shape[1]):
        label = id2label.get(src_idx, id2label.get(str(src_idx), None))
        if label is None and src_idx < len(ACTIONS):
            label = ACTIONS[src_idx]
        dst_idx = action_to_idx.get(str(label))
        if dst_idx is not None:
            aligned[:, dst_idx] = raw[:, src_idx]
    return aligned
