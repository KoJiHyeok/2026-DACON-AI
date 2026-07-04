# -*- coding: utf-8 -*-
"""세션-프리픽스 StratifiedGroup 85/15 holdout: 인코더 확률을 npz로 저장 (blend 탐색용).

목적: linear/stacker 확률과 로컬 CPU에서 합쳐 blend weight 그리드 탐색을 하려면 encoder의
holdout 확률이 필요하다 — 이 스크립트는 그걸 LB 제출 없이 정직하게 뽑아준다. base/small
어느 모델이든 --model로 지정해서 돌릴 수 있다.

split 방식(팀 리포 train_tscar.py의 기존 홀드아웃 패턴과 동일한 원칙을 따름):
    n_splits = round(1 / valid_frac)  (valid_frac=0.15 → n_splits=7, 홀드아웃 ≈14.3%)
    StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=SEED)의 첫 fold를
    holdout으로 사용. group = 세션 프리픽스(정규식 `-step_\\d+$` 제거), stratify = y(action) —
    라벨 분포를 보존하면서 같은 세션의 step들이 train/holdout 양쪽에 섞이지 않게 한다
    (CLAUDE.md 실험 규칙: "세션 프리픽스 기준 GroupKFold" 와 동일 원칙).
    참고: dacon-agent-action-api-boost/train_tscar.py:493-501 이 같은 패턴(
    `n_holdout_splits = max(2, round(1/valid_size))` → `StratifiedGroupKFold`)을 이미 사용.

serialize() 계약: encoder_v2_s42_repro.py / encoder_small_repro.py / ensemble/script_3way.py
와 코드 동일(AST 비교 완료). 절대 수정 금지.

⚠️ 이 스크립트의 학습 설정(EPOCHS/BATCH)은 최종 제출용 재현 스크립트
(encoder_v2_s42_repro.py, encoder_small_repro.py)와 **별개의 보수적 기본값**이다 — 목적이
"holdout 확률로 blend weight를 탐색하는 것"이지 제출용 인코더를 만드는 게 아니므로,
--epochs/--batch/--lr로 자유롭게 바꿔 빠른 probe를 돌려도 무방하다. 결과 npz를 제출에
쓰지 말 것(85%로만 학습됐고 label_smoothing도 없음 — 스펙이 다르다).

사용 (Colab, T4):
    python holdout_eval.py --model base  --out /content/drive/MyDrive/dacon2026/holdout_base.npz
    python holdout_eval.py --model small --out /content/drive/MyDrive/dacon2026/holdout_small.npz
    python holdout_eval.py --model intfloat/multilingual-e5-large --out holdout_large.npz  # 커스텀도 가능

산출(npz, 다운로드해서 로컬 CPU에서 사용):
    ids     : (n_holdout,) str  — sample id
    probs   : (n_holdout, 14) float64 — ACTIONS 순서(알파벳순, 아래 ACTIONS 참고) 확률
    y_true  : (n_holdout,) str  — 정답 action 이름
사용 예(로컬):
    d = np.load("holdout_base.npz", allow_pickle=True)
    ids, probs, y_true = d["ids"], d["probs"], d["y_true"]
    macro_f1 = f1_score(y_true, np.array(ACTIONS)[probs.argmax(1)], average="macro")
"""
import argparse
import csv
import json
import os
import random
import re

import numpy as np
import torch
import torch.nn.functional as TF
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

MODEL_ALIASES = {
    "base": "intfloat/multilingual-e5-base",
    "small": "intfloat/multilingual-e5-small",
}
# base/small 기본 배치 크기 — encoder_v2_s42_repro.py(base, batch16) /
# encoder_small_repro.py(small, batch32)와 동일한 메모리 프로파일을 따름.
DEFAULT_BATCH = {"base": 16, "small": 32}

MAX_LEN = 384
ACTIONS = ["apply_patch", "ask_user", "edit_file", "glob_pattern", "grep_search",
           "lint_or_typecheck", "list_directory", "plan_task", "read_file",
           "respond_only", "run_bash", "run_tests", "web_search", "write_file"]
LABEL2ID = {a: i for i, a in enumerate(ACTIONS)}

_STEP_RE = re.compile(r"-step_\d+$")


def _bucket(v, edges=(1000, 10000, 50000, 100000)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


def serialize(s, max_hist=6):
    """encoder_v2_s42_repro.py / encoder_small_repro.py / ensemble/script_3way.py와
    byte-identical(AST 비교 완료). train·추론 계약 — 수정 금지."""
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


def session_id(sample_id):
    return _STEP_RE.sub("", str(sample_id))


def load_data(data_dir):
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    train_labels = os.path.join(data_dir, "train_labels.csv")
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(f"train.jsonl 없음: {train_jsonl}")
    if not os.path.exists(train_labels):
        raise FileNotFoundError(f"train_labels.csv 없음: {train_labels}")

    samples = []
    with open(train_jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with open(train_labels, encoding="utf-8") as f:
        lab = {r["id"]: r["action"] for r in csv.DictReader(f)}
    texts = [serialize(s) for s in samples]
    y = np.array([LABEL2ID[lab[s["id"]]] for s in samples])
    ids = np.array([s["id"] for s in samples])
    groups = np.array([session_id(i) for i in ids])
    return texts, y, ids, groups


def check_drive_root(root):
    """멀티 계정 병렬 Colab 대비 — 경로가 안 보이면 원인 진단을 담아 즉시 실패.

    보조 계정 세션에서 drive.mount는 '그 계정'의 MyDrive만 노출한다. 메인 계정이
    소유한 공유 폴더는 보조 계정에선 '공유 문서함(Shared with me)'에 있어 마운트
    경로에 나타나지 않는다 — 2026-07-05 실제로 겪은 오류의 원인.

    함정 2 (유령 폴더): 과거 실패 실행의 os.makedirs가 보조 계정 MyDrive에 같은 이름의
    '빈' 폴더를 만들어 둘 수 있다 — 그러면 경로는 존재하는데 train.jsonl만 없다.
    """
    if os.path.isdir(root):
        if os.path.exists(os.path.join(root, "train.jsonl")):
            return
        listing = ", ".join(sorted(os.listdir(root))[:20]) or "(비어 있음)"
        raise FileNotFoundError(
            f"{root} 는 존재하지만 train.jsonl이 없습니다. 내용: [{listing}]\n"
            "→ 흔한 원인: 이전 실패 실행이 이 계정 MyDrive에 같은 이름의 '유령 폴더'를\n"
            "  만들어 둔 경우입니다 (데이터가 든 진짜 폴더는 공유 문서함의 메인 계정 소유본).\n"
            "  해결 순서: 1) 이 계정 Drive 웹에서 내 드라이브의 이 (빈) 폴더 삭제\n"
            "  2) 공유 문서함의 원본 폴더 우클릭 → 정리 → 바로가기 추가 → 내 드라이브\n"
            "  3) Colab에서 drive.flush_and_unmount() 후 drive.mount 재실행")
    my = "/content/drive/MyDrive"
    if not os.path.isdir(my):
        raise FileNotFoundError(
            "Google Drive가 마운트되지 않았습니다 — drive.mount 셀을 먼저 실행하세요.")
    listing = ", ".join(sorted(os.listdir(my))[:20])
    raise FileNotFoundError(
        f"--data-dir 없음: {root}\n"
        f"현재 마운트된 MyDrive 최상위: [{listing}]\n"
        "→ 목록에 폴더가 없다면 십중팔구 '보조 계정' 세션입니다. 공유받은 폴더는\n"
        "  공유 문서함에 있어 /MyDrive 마운트에 보이지 않습니다. 해결 (둘 중 하나):\n"
        "  (A) 이 계정 Drive 웹에서 해당 폴더 우클릭 → 정리 → '바로가기 추가' → 내 드라이브\n"
        "      (계정당 1회 설정, 이후 경로 그대로 동작)\n"
        "  (B) drive.mount 인증 팝업에서 폴더를 '소유한' 계정을 선택해 마운트")


def make_holdout_split(y, groups, seed, valid_frac):
    n_splits = max(2, int(round(1.0 / valid_frac)))
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(skf.split(np.zeros(len(y)), y, groups=groups))
    assert not (set(groups[tr_idx]) & set(groups[va_idx])), "session leakage"
    print(f"[split] StratifiedGroupKFold n_splits={n_splits} "
          f"(목표 holdout={valid_frac:.0%}, 실제={len(va_idx) / len(y):.1%}) "
          f"train={len(tr_idx)} holdout={len(va_idx)}")
    return tr_idx, va_idx


class DS(torch.utils.data.Dataset):
    def __init__(self, enc, labels):
        self.enc, self.labels = enc, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        return item


class WeightedTrainer(Trainer):
    """class_weight=balanced CE (holdout probe 용 — 제출 인코더와 레시피가 다를 수 있음)."""
    class_w = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_w.to(outputs.logits.device)
        loss = TF.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(pred):
    preds = pred.predictions.argmax(-1)
    return {"macro_f1": f1_score(pred.label_ids, preds, average="macro", zero_division=0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="'base', 'small', 또는 임의의 HF 모델 id/로컬 경로")
    ap.add_argument("--data-dir", default="/content/drive/MyDrive/dacon2026",
                    help="train.jsonl / train_labels.csv 위치 (기본: Colab Drive 경로)")
    ap.add_argument("--out", required=True, help="출력 npz 경로")
    ap.add_argument("--valid-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=None,
                    help="기본: base=16, small=32 (모델 별칭 기준, 커스텀 모델은 16)")
    ap.add_argument("--eval-batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    model_name = MODEL_ALIASES.get(args.model, args.model)
    batch = args.batch if args.batch is not None else DEFAULT_BATCH.get(args.model, 16)
    seed = args.seed
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if not torch.cuda.is_available():
        print("[경고] CUDA GPU가 감지되지 않았습니다 (Colab 런타임 유형을 T4 GPU로 설정하세요).")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    check_drive_root(args.data_dir)
    texts, y, ids, groups = load_data(args.data_dir)
    print(f"loaded {len(y)} rows, {len(set(groups))} sessions | model={model_name} seed={seed}")
    tr, va = make_holdout_split(y, groups, seed, args.valid_frac)

    tok = AutoTokenizer.from_pretrained(model_name)
    enc_tr = tok([texts[i] for i in tr], truncation=True, max_length=MAX_LEN, padding="max_length")
    enc_va = tok([texts[i] for i in va], truncation=True, max_length=MAX_LEN, padding="max_length")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(ACTIONS),
        id2label={i: a for a, i in LABEL2ID.items()}, label2id=LABEL2ID)

    counts = np.bincount(y[tr], minlength=len(ACTIONS))
    class_w = torch.tensor(len(y[tr]) / (len(ACTIONS) * np.maximum(counts, 1)),
                           dtype=torch.float32)

    training_args = TrainingArguments(
        output_dir="/tmp/holdout_eval_ckpt",
        num_train_epochs=args.epochs, learning_rate=args.lr,
        per_device_train_batch_size=batch, per_device_eval_batch_size=args.eval_batch,
        warmup_ratio=0.06, weight_decay=0.01, fp16=True,
        eval_strategy="epoch", save_strategy="no",   # 이 npz는 확률만 필요 — 체크포인트 보관 안 함
        logging_steps=100, seed=seed, report_to="none",
    )
    trainer = WeightedTrainer(model=model, args=training_args,
                              train_dataset=DS(enc_tr, y[tr]), eval_dataset=DS(enc_va, y[va]),
                              compute_metrics=compute_metrics)
    trainer.class_w = class_w
    trainer.train()

    out = trainer.predict(DS(enc_va, y[va]))
    probs = torch.softmax(torch.tensor(out.predictions), dim=-1).numpy().astype(np.float64)
    preds_idx = probs.argmax(-1)
    macro = f1_score(y[va], preds_idx, average="macro", zero_division=0)
    per = f1_score(y[va], preds_idx, average=None, labels=list(range(len(ACTIONS))), zero_division=0)
    print(json.dumps({
        "model": model_name, "seed": seed, "epochs": args.epochs, "batch": batch,
        "n_train": len(tr), "n_holdout": len(va), "holdout_macro_f1": float(macro),
        "per_class_f1": {a: round(float(v), 4) for a, v in zip(ACTIONS, per)},
    }, indent=2, ensure_ascii=False))

    ids_va = ids[va]
    y_true_va = np.array(ACTIONS)[y[va]]
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez(out_path, ids=ids_va, probs=probs, y_true=y_true_va,
             actions=np.array(ACTIONS))
    print(f"saved holdout probs to {out_path} "
          f"(ids={ids_va.shape}, probs={probs.shape}, y_true={y_true_va.shape})")


if __name__ == "__main__":
    main()
