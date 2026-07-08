# -*- coding: utf-8 -*-
"""e5-base 85% split 재학습 — serialize `max_hist` 확장 재심 (exp #34 / D-010).

목적: exp #15에서 폐기된 "serialize 히스토리 확장"을 **인코더 한정 재심**한다.
      encoder_v2_s42_repro.py의 레시피(6ep/b16/lr2e-5/ls0.1/fp16/balanced)를 **그대로**
      유지하고 `max_hist`만 env(ENC_MAXHIST)로 파라미터화한다. 85% split(holdout_base.npz
      ids 제외)로 학습하고 holdout 9,969행 확률 npz를 저장 → 로컬 리그가 e5 슬롯에 스왑.

      **두 번 돌린다(멀티계정 병렬):** ENC_MAXHIST=6(대조군) / ENC_MAXHIST=12(후보).
      리그에서 `hist12 − hist6대조`로 serialize 효과를 격리한다(프록시 출처 무관 self-contained).

⚠️ 하이퍼파라미터(EPOCHS/BATCH/LR/LABEL_SMOOTH/MAX_LEN/balanced)는 encoder_v2와 동일 —
   변경 금지. max_hist 이외 어떤 것도 바꾸면 hist6 대조군이 baseline e5와 달라져 판정이 오염됨.

env:
  ENC_MAXHIST   기본 6 (현행 계약) — 재심은 12로 실행
  ENC_SEED      기본 42 (w112 성분 시드)
  ENC_DATA_DIR  기본 ./data          (train.jsonl + train_labels.csv)
  ENC_HOLDOUT_NPZ 기본 ./holdout_base.npz — valid 행 = 이 npz의 ids (split 재구현 금지)
  ENC_OUT       기본 ./enc_e5_h{MAXHIST}   (Drive 경로 지정 권장 — 세션 끊김 대비 ckpt 저장)

완료 신호: `[npz] holdout_e5_h{N}.npz rows=... macro-F1=...` + `[DONE]`.
"""
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as TF
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)


def _env(k, d):
    v = os.environ.get(k, "").strip()
    return v if v else d


# ===== 설정 =====
MAX_HIST = int(_env("ENC_MAXHIST", "6"))          # ★ 재심 대상 (6=대조 / 12=후보)
SEED = int(_env("ENC_SEED", "42"))
DATA_DIR = _env("ENC_DATA_DIR", "./data")
HOLDOUT_NPZ = _env("ENC_HOLDOUT_NPZ", "./holdout_base.npz")
OUT = _env("ENC_OUT", f"./enc_e5_h{MAX_HIST}")

# ===== 하이퍼파라미터 — encoder_v2_s42_repro.py와 동일, 변경 금지 =====
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LEN = int(_env("ENC_MAXLEN", "384"))   # 기본 384 = 계약. ENC_MAXLEN=512는 명시적 프로브(Bet A) — @384와 직접 비교로 maxlen 격리
EPOCHS = 6
BATCH = 16
LR = 2e-5
LABEL_SMOOTH = 0.1
# ====================================================================

ACTIONS = ["apply_patch", "ask_user", "edit_file", "glob_pattern", "grep_search",
           "lint_or_typecheck", "list_directory", "plan_task", "read_file",
           "respond_only", "run_bash", "run_tests", "web_search", "write_file"]
LABEL2ID = {a: i for i, a in enumerate(ACTIONS)}

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)


def _bucket(v, edges=(1000, 10000, 50000, 100000)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


def serialize(s, max_hist=6):
    """submit/script.py serialize()와 byte-identical (max_hist만 인자화).
    train·추론 계약 — char-cap(query800/rsum120/user200/open5)은 절대 유지."""
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
    for h in reversed(hist[-max_hist:]):   # 최신 우선 → 잘려도 최근 맥락 보존
        if h.get("role") == "assistant_action":
            items.append(f"act:{h.get('name')} r:{str(h.get('result_summary') or '')[:120]}")
        else:
            items.append(f"user: {str(h.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    return "\n".join(parts)


def load_samples():
    tj = os.path.join(DATA_DIR, "train.jsonl")
    tl = os.path.join(DATA_DIR, "train_labels.csv")
    assert os.path.exists(tj), f"train.jsonl 없음: {tj} — 데이터 준비 셀 먼저"
    assert os.path.exists(tl), f"train_labels.csv 없음: {tl}"
    samples = []
    with open(tj, encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    with open(tl, newline="", encoding="utf-8-sig") as f:
        lab = {r["id"]: r["action"] for r in csv.DictReader(f)}
    samples = [s for s in samples if s["id"] in lab]
    return samples, lab


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
    class_w = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_w.to(outputs.logits.device)
        loss = TF.cross_entropy(outputs.logits, labels, weight=w,
                                label_smoothing=LABEL_SMOOTH)
        return (loss, outputs) if return_outputs else loss


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[경고] GPU 미감지 — 런타임 유형 T4 GPU로 설정 (CPU는 비현실적)")
    print(f"[cfg] max_hist={MAX_HIST} seed={SEED} epochs={EPOCHS} batch={BATCH} "
          f"lr={LR} ls={LABEL_SMOOTH} maxlen={MAX_LEN} out={OUT}")

    samples, lab = load_samples()
    assert len(ACTIONS) == 14

    # ----- split: holdout ids는 npz에서 직접 (재구현 금지) -----
    assert os.path.exists(HOLDOUT_NPZ), f"holdout npz 없음: {HOLDOUT_NPZ} — 배치 셀 먼저"
    hz = np.load(HOLDOUT_NPZ, allow_pickle=True)
    hold = set(str(x) for x in hz["ids"])
    ids_all = set(s["id"] for s in samples)
    n_missing = len(hold - ids_all)
    assert n_missing == 0, f"holdout ids {n_missing}/{len(hold)}개가 train에 없음 — 스테일 npz"

    tr = [s for s in samples if s["id"] not in hold]
    va = [s for s in samples if s["id"] in hold]
    print(f"[split] train={len(tr)} valid={len(va)} (holdout ids={len(hold)})")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tr_texts = [serialize(s, MAX_HIST) for s in tr]
    tr_y = np.array([LABEL2ID[lab[s["id"]]] for s in tr])
    enc = tok(tr_texts, truncation=True, max_length=MAX_LEN, padding="max_length")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ACTIONS),
        id2label={i: a for a, i in LABEL2ID.items()}, label2id=LABEL2ID).to(device)

    counts = np.bincount(tr_y, minlength=len(ACTIONS))
    class_w = torch.tensor(len(tr_y) / (len(ACTIONS) * np.maximum(counts, 1)),
                           dtype=torch.float32)

    args = TrainingArguments(
        output_dir=os.path.join(OUT, "ckpt"),
        num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH,
        warmup_ratio=0.06, weight_decay=0.01, fp16=True,
        save_strategy="epoch", save_total_limit=1,
        logging_steps=200, seed=SEED, report_to="none",
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=DS(enc, tr_y))
    trainer.class_w = class_w

    # 세션 끊김 재개: ckpt 폴더에 checkpoint-* 있으면 이어서
    ckpt_root = os.path.join(OUT, "ckpt")
    has_ckpt = os.path.isdir(ckpt_root) and any(
        d.startswith("checkpoint-") for d in os.listdir(ckpt_root)) if os.path.isdir(ckpt_root) else False
    trainer.train(resume_from_checkpoint=has_ckpt)

    # ----- holdout 확률 (fp32, 잘림 max_hist 동일) -----
    model.eval()
    probs = np.zeros((len(va), len(ACTIONS)), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(va), BATCH * 4):
            chunk = va[i:i + BATCH * 4]
            e = tok([serialize(s, MAX_HIST) for s in chunk], truncation=True,
                    max_length=MAX_LEN, padding=True, return_tensors="pt").to(device)
            logits = model(**e).logits.float().cpu().numpy()
            z = logits - logits.max(1, keepdims=True)
            ex = np.exp(z); probs[i:i + len(chunk)] = ex / ex.sum(1, keepdims=True)

    from sklearn.metrics import f1_score
    y_true = np.array([lab[s["id"]] for s in va], dtype=object)
    pred = np.array(ACTIONS)[probs.argmax(1)]
    f1 = f1_score([str(y) for y in y_true], pred, average="macro")

    os.makedirs(OUT, exist_ok=True)
    npz_path = os.path.join(OUT, f"holdout_e5_h{MAX_HIST}.npz")
    np.savez(npz_path,
             ids=np.array([s["id"] for s in va], dtype=object),
             probs=probs, y_true=y_true, actions=np.array(ACTIONS, dtype=object))
    with open(os.path.join(OUT, f"run_h{MAX_HIST}.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "max_hist": MAX_HIST, "seed": SEED,
                   "epochs": EPOCHS, "max_len": MAX_LEN, "n_train": len(tr),
                   "n_holdout": len(va), "solo_macro_f1": round(float(f1), 5)}, f, indent=2)
    print(f"[npz] holdout_e5_h{MAX_HIST}.npz rows={len(va)} macro-F1={f1:.5f}")
    print(f"      (참고 baseline e5 프록시 solo=0.70509 — hist6 대조군이 이 근처면 레시피 일치)")
    print("[DONE]")


main()
