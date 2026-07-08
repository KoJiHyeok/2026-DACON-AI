# -*- coding: utf-8 -*-
"""e5-base v2 s42 인코더 재학습 (Colab T4) — w112 앙상블 인코더 성분 재현.

원본: dacon-agent-action-api-boost/colab/encoder_v2_full.py (팀 리포, 읽기 전용).
이 파일은 그 스크립트의 하이퍼파라미터·로직을 그대로 유지한 채 Colab에서 바로 실행
가능하도록 경로 상수를 정리하고 seed를 환경변수로 파라미터화한 사본이다.
하이퍼파라미터(EPOCHS/BATCH/LR/LABEL_SMOOTH/MAX_LEN)는 절대 바꾸지 말 것 — 원본
러닝커브 실측(v1 피크 ep6)에 기반한 고정값이며, 바꾸면 w112 앙상블 성분과 다른
모델이 나온다.

serialize() 계약: ensemble/script_3way.py의 serialize()와 (docstring/주석을 제외한)
코드가 문자 단위로 동일함을 확인함 — 이 계약이 조금이라도 어긋나면 학습·추론 텍스트가
달라져 조용한 오답이 된다. 절대 수정 금지.

사용법 요약 (자세한 단계는 colab/README_colab.md 참고):
  1. Google Drive에 dacon2026/ 폴더를 만들고 train.jsonl, train_labels.csv 업로드.
  2. Colab 런타임을 T4 GPU로 설정.
  3. 셀 1(의존성 설치) 실행 → 셀 2(Drive 마운트) 실행 → 셀 3(이 스크립트 본문) 실행.
  4. 완료 후 OUT_DIR(Drive) 아래 model/ + run.json 생성 확인.
  5. to_fp16.py로 fp32 → fp16 변환 후 다운로드 (용량 절반, ~1.1GB → ~550MB).

시드 앙상블(원본 계획: seed 42 → 7 → 2026)을 여러 번 돌리려면 ENC_SEED 환경변수만
바꿔서 재실행하면 된다(코드 수정 불필요). 기본값은 42(w112 성분과 동일).
"""

# %% 셀 1: 의존성 설치 (Colab 셀에서 실행 — 이 파일 자체를 %run 하면 이 줄은 주석이라 무시됨)
# !pip install -q "transformers>=4.51" accelerate

# %% 셀 2: Google Drive 마운트 (Colab 셀에서 실행)
# from google.colab import drive
# drive.mount('/content/drive')

# %% 셀 3: 학습 본체 (아래 전체를 그대로 실행)
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as TF
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          Trainer, TrainingArguments)

# ===== 경로/시드 설정 — Colab에서 필요하면 이 블록만 조정 =====
DRIVE_ROOT = "/content/drive/MyDrive/dacon2026"      # train.jsonl, train_labels.csv 위치
DATA_DIR = DRIVE_ROOT
SEED = int(os.environ.get("ENC_SEED", "42"))          # 시드 앙상블: 42(기본) → 7 → 2026
MAX_HIST = int(os.environ.get("ENC_MAXHIST", "6"))    # serialize 히스토리 창 (계약 기본 6, hist12 배포용=12) — 정의 불변, 호출만
OUT_DIR = os.path.join(DRIVE_ROOT, f"enc_v2_s{SEED}_h{MAX_HIST}")  # 산출물(model/, run.json) — max_hist별 분리
# ================================================================

# ===== 하이퍼파라미터 — 원본(encoder_v2_full.py)과 동일, 변경 금지 =====
MODEL_NAME = "intfloat/multilingual-e5-base"
MAX_LEN = 384
EPOCHS = 6                                       # v1 러닝커브 피크(ep6) 고정
BATCH = 16
LR = 2e-5
LABEL_SMOOTH = 0.1
# ======================================================================

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
    """v1과 byte-identical — 제출 script.py(ensemble/script_3way.py)와의 train·추론 계약.
    수정 금지. (2026-AI-DACON 팀 검증: script_3way.py의 serialize()와 코드 동일 확인됨.)"""
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
            "Google Drive가 마운트되지 않았습니다 — 셀 2(drive.mount)를 먼저 실행하세요.")
    listing = ", ".join(sorted(os.listdir(my))[:20])
    raise FileNotFoundError(
        f"DRIVE_ROOT 없음: {root}\n"
        f"현재 마운트된 MyDrive 최상위: [{listing}]\n"
        "→ 목록에 폴더가 없다면 십중팔구 '보조 계정' 세션입니다. 공유받은 폴더는\n"
        "  공유 문서함에 있어 /MyDrive 마운트에 보이지 않습니다. 해결 (둘 중 하나):\n"
        "  (A) 이 계정 Drive 웹에서 해당 폴더 우클릭 → 정리 → '바로가기 추가' → 내 드라이브\n"
        "      (계정당 1회 설정, 이후 이 스크립트 경로 그대로 동작)\n"
        "  (B) drive.mount 인증 팝업에서 폴더를 '소유한' 계정을 선택해 마운트")


def load_data():
    train_jsonl = os.path.join(DATA_DIR, "train.jsonl")
    train_labels = os.path.join(DATA_DIR, "train_labels.csv")
    if not os.path.exists(train_jsonl):
        raise FileNotFoundError(
            f"train.jsonl 없음: {train_jsonl}\n"
            f"DATA_DIR({DATA_DIR})에 train.jsonl / train_labels.csv를 업로드했는지, "
            "Drive가 마운트됐는지 확인하세요."
        )
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
    texts = [serialize(s, MAX_HIST) for s in samples]
    y = np.array([LABEL2ID[lab[s["id"]]] for s in samples])
    return texts, y


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
    """class_weight=balanced + label smoothing CE."""
    class_w = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_w.to(outputs.logits.device)
        loss = TF.cross_entropy(outputs.logits, labels, weight=w,
                                label_smoothing=LABEL_SMOOTH)
        return (loss, outputs) if return_outputs else loss


def main():
    if not torch.cuda.is_available():
        print("[경고] CUDA GPU가 감지되지 않았습니다. Colab 런타임 유형을 "
              "'T4 GPU'로 설정했는지 확인하세요 (CPU로는 비현실적으로 오래 걸림).")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    check_drive_root(DRIVE_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    texts, y = load_data()
    print(f"loaded {len(y)} rows (FULL train, no holdout) | seed={SEED} | out_dir={OUT_DIR}")
    assert len(ACTIONS) == 14, f"클래스 수가 14가 아님: {len(ACTIONS)}"

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    enc = tok(texts, truncation=True, max_length=MAX_LEN, padding="max_length")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(ACTIONS),
        id2label={i: a for a, i in LABEL2ID.items()}, label2id=LABEL2ID)

    counts = np.bincount(y, minlength=len(ACTIONS))
    class_w = torch.tensor(len(y) / (len(ACTIONS) * np.maximum(counts, 1)),
                           dtype=torch.float32)

    args = TrainingArguments(
        output_dir=os.path.join(OUT_DIR, "ckpt"),
        num_train_epochs=EPOCHS, learning_rate=LR,
        per_device_train_batch_size=BATCH,
        warmup_ratio=0.06, weight_decay=0.01, fp16=True,
        save_strategy="epoch", save_total_limit=1,   # 세션 끊김 대비 최신 1개만 유지
        logging_steps=200, seed=SEED, report_to="none",
    )
    trainer = WeightedTrainer(model=model, args=args, train_dataset=DS(enc, y))
    trainer.class_w = class_w
    trainer.train()

    trainer.save_model(os.path.join(OUT_DIR, "model"))
    tok.save_pretrained(os.path.join(OUT_DIR, "model"))
    with open(os.path.join(OUT_DIR, "run.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "seed": SEED, "epochs": EPOCHS,
                   "max_len": MAX_LEN, "label_smoothing": LABEL_SMOOTH,
                   "n_train": int(len(y)), "full_train": True}, f, indent=2)
    print(f"saved to {OUT_DIR}")


# %% 셀 4: 실행
if __name__ == "__main__":
    main()
