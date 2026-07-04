# -*- coding: utf-8 -*-
"""e5-small 4번째 앙상블 성분 재학습 (Colab T4) — full-70k, no holdout.

원본: dacon-agent-action-api-boost/colab/encoder_finetune.py (팀 리포, 읽기 전용,
intfloat/multilingual-e5-small, 15% group holdout으로 valid_macro_f1 측정하던 스크립트).
이 파일은 그 스크립트의 핵심 하이퍼파라미터(MAX_LEN/EPOCHS/BATCH/LR)를 그대로 두고,
`colab/encoder_v2_full.py`가 e5-base에 적용한 것과 같은 종류의 변경 — **holdout 제거 +
전량(70k) 학습**만 적용한 사본이다. label_smoothing은 추가하지 않았다(v1 e5-small
스크립트에는 원래 없었고, 이번 지시에도 "full-70k로 조정"만 명시됐을 뿐 label smoothing을
넣으라는 지시는 없었음 — 불필요한 하이퍼파라미터 변경을 피하기 위한 보수적 선택).

목적: 팀의 w112 3-way 앙상블(linear/stacker/encoder-base)에 **4번째 성분**을 더한다.
`ensemble/script_3way.py`의 `encoder_dirs()`는 `model/encoder`, `model/encoder_2`, …
디렉터리를 이름순으로 모아 확률을 uniform 평균하도록 이미 구현되어 있으므로, e5-small을
`model/encoder_2`로 넣으면 코드 수정 없이 4-way(사실상 2-encoder 평균 포함 3-way)가 된다.
e5-small은 e5-base와 아키텍처·파라미터 규모가 달라(XLM-R small vs base) 오류 패턴이
달라질 가능성이 있고, 그 이질성이 앙상블에 기여하길 기대하는 것이 이 성분의 존재 이유다.

⚠️ 주의: `ensemble/soup_encoders.py`의 가중치 평균(model soup) 방식과 이 스크립트를
혼동하지 말 것. 팀 보고 기준 seed-soup은 LB 0.697로 **폐기 확정**되었다 — 이 스크립트가
만드는 것은 soup(가중치 평균)이 아니라 **확률을 평균할 별도의 완전한 모델**이다. soup는
"같은 레시피 다른 seed" 전제인데 e5-small은 e5-base와 레시피(모델 크기)부터 다르므로
애초에 soup 대상이 아니다.

serialize() 계약: colab/encoder_v2_s42_repro.py 및 ensemble/script_3way.py와 코드
동일(AST 비교 완료, docstring/주석만 다름) — base 인코더와 완전히 같은 텍스트 직렬화를
쓴다. 절대 수정 금지.

zip 예산 계산 (1024MB 한도, 팀 보고 기준 수치):
    현재 w112 3-way 구성                    ≈ 634 MB  (linear + stacker + encoder-base fp16)
  + e5-small fp16 (118M 파라미터 × 2byte)   ≈ 235 MB  (추정, 팀 colab/README.md 기준)
  ------------------------------------------------------------
  = 4-way 합계                              ≈ 869 MB  → 한도 대비 여유 ≈ 155 MB

사용법 요약 (자세한 단계는 colab/README_colab.md의 job2 참고):
  1. Google Drive에 dacon2026/ 폴더(train.jsonl, train_labels.csv 이미 있다고 가정 —
     encoder_v2_s42_repro.py와 데이터 공유).
  2. Colab 런타임을 T4 GPU로 설정.
  3. 셀 1(의존성 설치) → 셀 2(Drive 마운트, encoder_v2_s42_repro.py와 동일) → 셀 3(이 스크립트).
  4. fp32 산출물을 to_fp16.py로 변환 후 다운로드.

시드는 ENC_SEED 환경변수로 바꿀 수 있으나(기본 42), e5-small은 seed 앙상블이 아니라
"이종 크기 모델 1개"가 목적이므로 기본값(42) 그대로 한 번만 돌리면 된다.
"""

# %% 셀 1: 의존성 설치 (Colab 셀에서 실행)
# !pip install -q "transformers>=4.51" accelerate

# %% 셀 2: Google Drive 마운트 (Colab 셀에서 실행, encoder_v2_s42_repro.py와 공유)
# from google.colab import drive
# drive.mount('/content/drive')

# %% 셀 3: 학습 본체
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
DRIVE_ROOT = "/content/drive/MyDrive/dacon2026"       # train.jsonl, train_labels.csv 위치
DATA_DIR = DRIVE_ROOT
SEED = int(os.environ.get("ENC_SEED", "42"))
OUT_DIR = os.path.join(DRIVE_ROOT, f"enc_small_s{SEED}")   # 산출물(model/, run.json) 위치
# ================================================================

# ===== 하이퍼파라미터 — 원본(encoder_finetune.py)과 동일, 변경 금지 =====
MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LEN = 384
EPOCHS = 3                                       # v1 e5-small holdout 실측 설정 그대로 유지
BATCH = 32
LR = 2e-5
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
    """encoder_v2_s42_repro.py / ensemble/script_3way.py와 byte-identical(AST 비교 완료).
    train·추론 계약 — 수정 금지."""
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
    texts = [serialize(s) for s in samples]
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
    """class_weight=balanced CE (label smoothing 없음 — v1 e5-small 원본과 동일)."""
    class_w = None

    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        w = self.class_w.to(outputs.logits.device)
        loss = TF.cross_entropy(outputs.logits, labels, weight=w)
        return (loss, outputs) if return_outputs else loss


def main():
    if not torch.cuda.is_available():
        print("[경고] CUDA GPU가 감지되지 않았습니다. Colab 런타임 유형을 "
              "'T4 GPU'로 설정했는지 확인하세요 (CPU로는 비현실적으로 오래 걸림).")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

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
                   "max_len": MAX_LEN, "label_smoothing": None,
                   "n_train": int(len(y)), "full_train": True}, f, indent=2)
    print(f"saved to {OUT_DIR}")


# %% 셀 4: 실행
if __name__ == "__main__":
    main()
