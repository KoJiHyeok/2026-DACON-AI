# -*- coding: utf-8 -*-
"""e5-base v3 직렬화 인코더 학습 (Colab T4) — serialize_v3/v3_hist6 계약 정본.

원본: `colab/encoder_v2_s42_repro.py`(챔피언 v2, solo LB ~0.701)를 베이스로 하이퍼
파라미터는 그대로 유지하고 `serialize()`만 교체한 사본이다. `ENC_SERIALIZE` 환경변수로
v3(전체, max_hist=12+args+lang+elapsed) 또는 v3_hist6(=v2+args+lang+elapsed, max_hist=6)
중 하나를 선택한다(아래 "ENC_SERIALIZE 스위치" 절 참고). v2/v3 성능은 서로 다른 텍스트
계약이므로 **비교 불가 — v3(또는 v3_hist6)는 v2를 대체하는 새 성분 후보**다(합칠 경우
model/encoder_2로 앙상블에 추가하는 방식은 v2와 동일 패턴 사용 가능).

⚠️ 이 파일의 `serialize()`가 v3 학습·추론 계약의 **정본(source of truth)**이다.
제출용 script.py에 v3 인코더를 편입할 때는 이 함수를 문자 단위로 복사해서 써야
한다 — 한 글자라도 다르면 학습 때 본 입력과 추론 때 보는 입력이 달라져 조용한
오답이 된다(ai-2026/CLAUDE.md 7절 4항과 같은 원리, v2의 serialize() 계약과 동일 원칙).

## serialize_v2 → v3 diff (2026-07-05, data/train.jsonl 70,000행 실사 근거)

로컬 프록시 A/B(`scripts/encoding/serialize_ab.py`, TF-IDF+LinearSVC,
StratifiedGroupKFold 5-fold) 결과와 판정은 이 스크립트가 아니라 그 스크립트의
산출물(`scripts/encoding/_out/serialize_ab_summary.csv`)과 팀 보고를 따를 것 —
아래 diff 자체는 실측 데이터 근거(필드 실사)로 설계했지만 GPU 투입 여부의 최종
판정은 A/B 결과가 우선한다.

1. **max_hist 6 → 12**: history 길이 분포 실측 결과 12턴 보유 행이 21,586건
   (30.8%)으로 최빈값 — v2는 이 표본들의 최근 6턴을 통째로 버리고 있었다.
2. **assistant_action 턴에 args 핵심 값 추가**: v2는 `result_summary`만 쓰고
   `args`(경로/패턴/명령 등)를 전부 버린다 — explore 4클래스(read_file/
   grep_search/list_directory/glob_pattern)처럼 result_summary만으론 구분이
   어려운 액션 쌍에 특히 치명적. 액션별 축약 키(ARG_ABBR)로 값을 60자 truncate해
   추가한다(실측 args 값 길이 p99 대부분 <75자 — 60자면 충분, ask_user 질문만
   최대 209자라 일부 잘릴 수 있으나 저빈도 클래스라 감내).
3. **language_mix 최상위 언어 1개 추가**: `toplang={최댓값 키}` — dict 전체를
   넣기엔 토큰 낭비, 상위 1개만으로도 스택 신호 충분하다고 판단(실측 최빈:
   py 27,897 > rs 7,881 > java 7,196 > vue/tsx/ts/go/yaml).
4. **elapsed_session_sec 버킷 추가**: 실측 30~1530초(중앙값 498초)를
   (120,300,600,900,1200) 경계로 버킷화.
5. 위 3·4번(우선순위 최저)은 텍스트 맨 끝에 배치 — 토크나이저가 뒤쪽부터
   자르므로(`truncation=True`), 잘림이 나도 헤더/query/최근 history(핵심 정보)는
   보존되고 language_mix/elapsed부터 사라지게 설계했다.

헤더/open/query 라인은 v2와 동일(정보 손실 없어 바꿀 이유 없음, current_prompt
p99=169자로 800자 캡은 원래도 여유였음).

## MAX_LEN 근거 — ⚠️ 2026-07-05 정정 (독립 리뷰 FAIL, D-0xx급 정정)

**최초 설계 시 "문자수/4 ≈ 토큰수" 근사를 썼는데 이게 틀렸다.** 실제 e5 토크나이저
(`artifacts/enc_v2_s42/model/tokenizer.json`, `tokenizers` 패키지로 오프라인 로드,
`no_truncation()`+`no_padding()`로 저장된 384-cap 설정을 해제하고 측정)로 70,000행
전수 재측정한 결과 **chars/token ≈ 2.43** (4가 아니라 4의 약 61% — 근사가 토큰수를
약 1.64배 과소평가하고 있었다):

| 직렬화 | mean tok | max_len=384 잘림 | max_len=512 잘림 |
|---|---:|---:|---:|
| v2(max_hist=6, args 없음) | 173 | 0.00% | 0.00% |
| **v3(max_hist=12, args+lang+elapsed)** | 277 | **29.30%** | 1.87% |
| v3, history=12턴 보유 행만(전체의 30.8%, v3가 노리는 핵심 타깃) | 428 | **82.88%** | 5.93% |
| v3_hist6(=v2+args+lang+elapsed, max_hist=6) | 204 | 0.04% | 0.00% |

즉 **v3(full, max_hist=12)를 384로 돌리면 v3가 개선하려는 바로 그 표본(12턴
history)의 83%가 잘려서 max_hist 확장 효과가 대부분 무효화**된다. v3_hist6은
384에서도 잘림이 무시 가능한 수준(0.04%)이라 그대로 384를 써도 안전하다.

모델(`intfloat/multilingual-e5-base`, XLM-RoBERTa-base)의 `max_position_embeddings`는
514이므로 **512가 MAX_LEN의 물리적 상한**이다(그 이상은 모델이 애초에 못 받음).

결론: `ENC_SERIALIZE=v3`(전체)를 쓰면 `ENC_MAX_LEN=512`가 사실상 필수, `ENC_SERIALIZE=v3_hist6`이면
384로도 충분하다 — 아래 `ENC_SERIALIZE` 스위치와 연동해 기본 MAX_LEN을 모드별로 자동
설정하되 env로 언제든 덮어쓸 수 있게 했다. 512는 384 대비 padding="max_length"라
시퀀스 길이가 균일하게 33% 늘어 Colab 학습시간이 2.5h→약 3.3h로 늘 것으로 추정된다.

## ENC_SERIALIZE 스위치 (v3 전체 vs v3_hist6)

로컬 TF-IDF+LinearSVC 프록시(`scripts/encoding/serialize_ab.py`)에서 v3(전체)가
v2보다 낮게 나오고, args 없는 v3_no_args가 args 있는 v3보다 오히려 높게 나온
1차 신호가 있었다(2026-07-05, 3-fold 70,000행: v2 oof=0.5017, v3 oof=0.4704,
v3_no_args fold0-1 평균=0.4737) — 다만 TF-IDF는 텍스트 길이 희석에 민감하고
인코더(attention)는 이 효과가 다를 수 있어 프록시만으로 단정하지 않는다.
`ENC_SERIALIZE=v3|v3_hist6` 환경변수로 두 계약 중 하나를 선택해 학습할 수 있게
했다 — **기본값은 프록시 최종 결과를 팀이 확인한 뒤에 확정**한다(현재 기본값은
잠정치이니 실행 전에 반드시 이 값을 확인할 것).

하이퍼파라미터(EPOCHS/BATCH/LR/LABEL_SMOOTH)는 v2와 동일하게 유지 — v2 러닝커브
실측(피크 ep6)에 기반한 값이며, 이번 변경은 텍스트 계약(serialize)에 한정한다.

사용법 요약 (자세한 단계는 colab/README_colab.md 참고, v2와 셀 1/2 공유):
  1. Google Drive에 dacon2026/ 폴더(train.jsonl, train_labels.csv — v2와 공유).
  2. Colab 런타임을 T4 GPU로 설정.
  3. 셀 1(의존성 설치) → 셀 2(Drive 마운트) → 셀 3(이 스크립트 본문) 실행.
  4. 완료 후 OUT_DIR(Drive) 아래 model/ + run.json 생성 확인.
  5. to_fp16.py로 fp32 → fp16 변환 후 다운로드.

시드 앙상블이 필요하면 ENC_SEED 환경변수만 바꿔서 재실행(코드 수정 불필요).
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

# v3(전체, max_hist=12+args+lang+elapsed) vs v3_hist6(=v2+args+lang+elapsed, max_hist=6).
# 실측 잘림률·프록시 결과는 상단 docstring "ENC_SERIALIZE 스위치" 절 참고.
# ⚠️ 기본값 "v3"는 잠정치 — 팀 프록시(v3_no_args/v3_hist6) 최종 결과로 확정 예정.
SERIALIZE_MODE = os.environ.get("ENC_SERIALIZE", "v3").strip().lower()
if SERIALIZE_MODE not in ("v3", "v3_hist6"):
    raise ValueError(f"ENC_SERIALIZE는 'v3' 또는 'v3_hist6'여야 함: {SERIALIZE_MODE!r}")

OUT_SUBDIR = os.environ.get("ENC_OUT_SUBDIR", f"enc_{SERIALIZE_MODE}_s{SEED}")
OUT_DIR = os.path.join(DRIVE_ROOT, OUT_SUBDIR)         # 산출물(model/, run.json) 위치
# ================================================================

# ===== 하이퍼파라미터 — v2(encoder_v2_s42_repro.py)와 동일, 변경 금지 =====
MODEL_NAME = "intfloat/multilingual-e5-base"
# 잘림률 실측(2026-07-05, 실제 e5 토크나이저 기준 — chars/4 근사는 틀렸음, 상단 docstring 참고):
# v3(max_hist=12) 384에서 29.30% 잘림(history=12턴 행만 보면 82.88%) vs 512에서 1.87%.
# v3_hist6(max_hist=6)은 384에서도 0.04%로 무시 가능. 모드별 기본값을 다르게 두되 env로 덮어쓸 수 있다.
# 모델 max_position_embeddings=514 — 512가 물리적 상한(그 이상 불가).
_DEFAULT_MAX_LEN = {"v3": 512, "v3_hist6": 384}[SERIALIZE_MODE]
MAX_LEN = int(os.environ.get("ENC_MAX_LEN", str(_DEFAULT_MAX_LEN)))
assert MAX_LEN <= 512, f"multilingual-e5-base max_position_embeddings=514 — MAX_LEN>512 불가: {MAX_LEN}"
EPOCHS = 6                                       # v1/v2 러닝커브 피크(ep6) 고정
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


def _bucket_elapsed(v, edges=(120, 300, 600, 900, 1200)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


# 액션별 args 핵심 키 → 축약명 (data/train.jsonl 70,000행 실사로 확정된 스키마)
ARG_ABBR = {
    "read_file": [("path", "p")],
    "grep_search": [("pattern", "pat"), ("scope", "scope")],
    "list_directory": [("path", "p")],
    "glob_pattern": [("pattern", "pat")],
    "edit_file": [("path", "p"), ("target_symbol", "sym")],
    "write_file": [("path", "p")],
    "apply_patch": [("n_files", "n")],
    "run_bash": [("cmd", "cmd")],
    "run_tests": [("target", "t")],
    "lint_or_typecheck": [("target", "t")],
    "ask_user": [("question", "q")],
    "plan_task": [("goal", "g")],
    "web_search": [("query", "q")],
}


def _args_str(name, args):
    if not isinstance(args, dict):
        return ""
    out = []
    for key, abbr in ARG_ABBR.get(name, []):
        if key in args:
            out.append(f"{abbr}:{str(args[key])[:60]}")
    return " ".join(out)


def serialize(s, max_hist=None):
    """⚠️ v3/v3_hist6 학습·추론 계약의 정본. max_hist=None이면 SERIALIZE_MODE에 따라
    12(v3 전체) 또는 6(v3_hist6=v2+args+lang+elapsed)으로 정해진다. v2(colab/
    encoder_v2_s42_repro.py)와의 diff는 상단 모듈 docstring 참고. 이 함수를 수정하면
    반드시 이 학습 스크립트를 다시 돌려야 하고, 제출 script.py에도 문자 단위로
    동일하게 반영해야 한다."""
    if max_hist is None:
        max_hist = 12 if SERIALIZE_MODE == "v3" else 6
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
    for h in reversed(hist[-max_hist:]):  # 최신 우선 → 긴 입력이 잘려도 최근 맥락 보존
        if h.get("role") == "assistant_action":
            name = h.get("name")
            r = str(h.get("result_summary") or "")[:120]
            a = _args_str(name, h.get("args"))
            items.append(f"act:{name} {a} r:{r}" if a else f"act:{name} r:{r}")
        else:
            items.append(f"user: {str(h.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    tail = []
    lm = ws.get("language_mix") or {}
    if lm:
        tail.append(f"toplang={max(lm.items(), key=lambda kv: kv[1])[0]}")
    tail.append(f"elapsed={_bucket_elapsed(sm.get('elapsed_session_sec'))}")
    parts.append(" ".join(tail))
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
    """class_weight=balanced + label smoothing CE (v2와 동일)."""
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
              "'T4 GPU'로 설정했는지 확인하세요 (CPU로는 비현실적으로 오래 걸립니다).")
    else:
        print(f"[GPU] {torch.cuda.get_device_name(0)}")

    check_drive_root(DRIVE_ROOT)
    os.makedirs(OUT_DIR, exist_ok=True)
    texts, y = load_data()
    print(f"loaded {len(y)} rows (FULL train, no holdout) | seed={SEED} | "
          f"serialize={SERIALIZE_MODE} | max_len={MAX_LEN} | out_dir={OUT_DIR}")
    assert len(ACTIONS) == 14, f"클래스 수가 14가 아님: {len(ACTIONS)}"
    print(f"serialize[{SERIALIZE_MODE}] sample[0]:\n" + texts[0][:400])

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
                   "n_train": int(len(y)), "full_train": True,
                   "serialize_version": SERIALIZE_MODE}, f, indent=2)
    print(f"saved to {OUT_DIR}")


# %% 셀 4: 실행
if __name__ == "__main__":
    main()
