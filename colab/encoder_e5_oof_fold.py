# -*- coding: utf-8 -*-
"""e5 hist12 group 5-fold OOF 생성 — fold 하나를 학습·예측하고 종료 (P1-C 선행, coordination 실행순서 #3).

목적: hist12-aware stacker 재학습(Codex `scripts/stacker_h12/**` 소비)용 정직한 OOF 확률.
      배포 챔피언과 동일한 serialize·레시피(hist12, 6ep/b16/lr2e-5/ls0.1/384, class_weight
      balanced, SESSW=none)를 encoder_e5_holdout85_maxhist에서 **import로 재사용**한다 —
      직렬화 사본 금지(불일치 = 조용한 점수 하락).

실행 (fold당 프로세스 1개 — 래퍼가 fold 순회 + ckpt 즉시 삭제):
  export ENC_MAXHIST=12 ENC_ARGSLITE=0 ENC_SESSW=none ENC_DATA_DIR=~/data ENC_OUT=~/out/oof_h12
  OOF_FOLD=0 python colab/encoder_e5_oof_fold.py   # → $ENC_OUT/oof_fold0.npz + fold_map.csv

fold 배정: StratifiedGroupKFold(5, shuffle, seed=42), group=세션 프리픽스(-step_ 제거),
  stratify=action. 최초 실행이 $ENC_OUT/fold_map.csv를 생성하고 이후 fold는 그 파일을
  재사용한다(재현·sparse OOF 정렬용 단일 소스). 이미 있으면 재생성하지 않는다.

디스크 정책: 학습 ckpt는 $ENC_OUT/fold{k}/ckpt 아래에만 생기고, 래퍼가 npz 저장 확인 후
  fold{k} 디렉토리를 통째로 삭제한다. 서버에는 fold_map.csv + oof_fold*.npz + 로그만 남긴다.

완료 신호: `[npz] oof_fold{k}.npz rows=... macro-F1=...` + `[DONE]`.
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encoder_e5_holdout85_maxhist as base  # serialize·레시피 단일 소스

FOLD = int(os.environ.get("OOF_FOLD", "-1"))
N_FOLDS = int(os.environ.get("OOF_NFOLDS", "5"))
FOLD_SEED = int(os.environ.get("OOF_SEED", "42"))
MAP_ONLY = os.environ.get("OOF_MAP_ONLY", "0") == "1"  # 1=fold_map만 생성 후 종료 (병렬 러너 시작 전 1회 — 동시 생성 레이스 방지)
OUT = base.OUT  # ENC_OUT 그대로 — fold 산출물이 한 곳에 모임
FOLD_MAP = os.path.join(OUT, "fold_map.csv")

assert MAP_ONLY or 0 <= FOLD < N_FOLDS, f"OOF_FOLD 필요 (0~{N_FOLDS-1}): {FOLD}"
assert base.MODE != "full", "OOF는 holdout85 모드 개념이 아님 — ENC_MODE 기본값 유지"
assert base.SESSW == "none" and base.ARGS_LITE == 0 and base.MAX_HIST == 12, \
    f"배포 계약과 다름: sessw={base.SESSW} args_lite={base.ARGS_LITE} hist={base.MAX_HIST}"


def build_or_load_fold_map(samples, lab):
    if os.path.exists(FOLD_MAP):
        with open(FOLD_MAP, newline="", encoding="utf-8") as f:
            m = {r["id"]: int(r["fold"]) for r in csv.DictReader(f)}
        missing = [s["id"] for s in samples if s["id"] not in m]
        assert not missing, f"fold_map에 없는 id {len(missing)}개 — 스테일 맵, 삭제 후 재생성 필요"
        stale = set(m) - {s["id"] for s in samples}
        assert not stale, f"fold_map에만 있는 id {len(stale)}개 — 데이터 불일치, 맵 삭제 후 재생성 필요"
        print(f"[foldmap] 기존 재사용: {FOLD_MAP} ({len(m)}행, 양방향 일치)")
        return m
    from sklearn.model_selection import StratifiedGroupKFold
    ids = [s["id"] for s in samples]
    y = [lab[i] for i in ids]
    groups = [i.rsplit("-step_", 1)[0] for i in ids]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=FOLD_SEED)
    m = {}
    for k, (_, va_idx) in enumerate(sgkf.split(ids, y, groups)):
        for j in va_idx:
            m[ids[j]] = k
    os.makedirs(OUT, exist_ok=True)
    tmp = FOLD_MAP + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "fold"])
        for i in ids:
            w.writerow([i, m[i]])
    os.replace(tmp, FOLD_MAP)  # 원자적 치환 — 부분 쓰기 상태가 관측되지 않게
    # 그룹 무결성: 한 세션이 두 fold에 갈라지면 누수
    sess_fold = {}
    for i in ids:
        g = i.rsplit("-step_", 1)[0]
        assert sess_fold.setdefault(g, m[i]) == m[i], f"세션 {g}이 여러 fold에 존재"
    print(f"[foldmap] 신규 생성: {FOLD_MAP} (seed={FOLD_SEED}, {N_FOLDS}fold, {len(m)}행, "
          f"{len(sess_fold)}세션, 그룹 무결성 OK)")
    return m


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, lab = base.load_samples()
    fold_of = build_or_load_fold_map(samples, lab)
    if MAP_ONLY:
        print("[DONE] map-only")
        return

    tr = [s for s in samples if fold_of[s["id"]] != FOLD]
    va = [s for s in samples if fold_of[s["id"]] == FOLD]
    print(f"[cfg] OOF fold={FOLD}/{N_FOLDS} train={len(tr)} valid={len(va)} "
          f"(hist={base.MAX_HIST} seed={base.SEED} recipe=base)")

    tok = AutoTokenizer.from_pretrained(base.MODEL_NAME)
    tr_texts = [base.serialize(s, base.MAX_HIST, base.ARGS_LITE) for s in tr]
    tr_y = np.array([base.LABEL2ID[lab[s["id"]]] for s in tr])
    enc = tok(tr_texts, truncation=True, max_length=base.MAX_LEN, padding="max_length")

    model = AutoModelForSequenceClassification.from_pretrained(
        base.MODEL_NAME, num_labels=len(base.ACTIONS),
        id2label={i: a for a, i in base.LABEL2ID.items()},
        label2id=base.LABEL2ID).to(device)

    counts = np.bincount(tr_y, minlength=len(base.ACTIONS))
    class_w = torch.tensor(len(tr_y) / (len(base.ACTIONS) * np.maximum(counts, 1)),
                           dtype=torch.float32)

    ckpt_dir = os.path.join(OUT, f"fold{FOLD}", "ckpt")
    args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=base.EPOCHS, learning_rate=base.LR,
        per_device_train_batch_size=base.BATCH,
        warmup_ratio=0.06, weight_decay=0.01, fp16=True,
        save_strategy="epoch", save_total_limit=1,
        logging_steps=200, seed=base.SEED, report_to="none",
        remove_unused_columns=False,
    )
    base.WeightedTrainer._sw_checked = False
    trainer = base.WeightedTrainer(model=model, args=args,
                                   train_dataset=base.DS(enc, tr_y, None))
    trainer.class_w = class_w
    trainer.sessw_mode = "none"

    has_ckpt = os.path.isdir(ckpt_dir) and any(
        d.startswith("checkpoint-") for d in os.listdir(ckpt_dir))
    trainer.train(resume_from_checkpoint=has_ckpt)

    model.eval()
    probs = np.zeros((len(va), len(base.ACTIONS)), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(va), base.BATCH * 4):
            chunk = va[i:i + base.BATCH * 4]
            e = tok([base.serialize(s, base.MAX_HIST, base.ARGS_LITE) for s in chunk],
                    truncation=True, max_length=base.MAX_LEN, padding=True,
                    return_tensors="pt").to(device)
            logits = model(**e).logits.float().cpu().numpy()
            z = logits - logits.max(1, keepdims=True)
            ex = np.exp(z)
            probs[i:i + len(chunk)] = ex / ex.sum(1, keepdims=True)

    from sklearn.metrics import f1_score
    y_true = np.array([lab[s["id"]] for s in va], dtype=object)
    pred = np.array(base.ACTIONS)[probs.argmax(1)]
    f1 = f1_score([str(y) for y in y_true], pred, average="macro")

    npz_path = os.path.join(OUT, f"oof_fold{FOLD}.npz")
    np.savez(npz_path,
             ids=np.array([s["id"] for s in va], dtype=object),
             probs=probs, y_true=y_true,
             actions=np.array(base.ACTIONS, dtype=object),
             fold=np.array([FOLD] * len(va)))
    with open(os.path.join(OUT, f"run_oof_fold{FOLD}.json"), "w", encoding="utf-8") as f:
        json.dump({"fold": FOLD, "n_folds": N_FOLDS, "fold_seed": FOLD_SEED,
                   "model": base.MODEL_NAME, "max_hist": base.MAX_HIST,
                   "seed": base.SEED, "epochs": base.EPOCHS, "max_len": base.MAX_LEN,
                   "n_train": len(tr), "n_valid": len(va),
                   "fold_macro_f1": round(float(f1), 5)}, f, indent=2)
    print(f"[npz] oof_fold{FOLD}.npz rows={len(va)} macro-F1={f1:.5f}")
    print("[DONE]")


if __name__ == "__main__":
    main()
