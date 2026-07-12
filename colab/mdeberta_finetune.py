# -*- coding: utf-8 -*-
"""mdeberta-v3-base 파인튜닝 (DACON 236694 — 이질 인코더 성분 후보).

Colab 준비 셀 (이 스크립트 셀보다 먼저):
  !pip install -q "transformers>=4.51" accelerate sentencepiece

Colab 셀에 통짜 붙여넣기 실행이 기본 (required argparse 금지 규약).
모든 설정은 env 폴백:
  MDEB_MODEL        기본 microsoft/mdeberta-v3-base (스모크는 로컬 e5 경로 지정)
  MDEB_DATA_DIR     기본 ./data          (train.jsonl + train_labels.csv)
  MDEB_OUT          기본 ./mdeb_out
  MDEB_MODE         holdout85 | full     (기본 holdout85)
  MDEB_HOLDOUT_NPZ  기본 ./holdout_base.npz — holdout85 모드에서 valid 행 = 이 npz의 ids
                    (job3와 fold 불일치 사고 재발 방지: split 로직 재구현 금지, ids 직접 사용)
  MDEB_SEED=42  MDEB_EPOCHS=2  MDEB_BATCH=8  MDEB_ACCUM=2  MDEB_MAXLEN=384  MDEB_LR=2e-5
  MDEB_RESUME=1     ckpt에서 재개 (Colab 끊김 대비 — 에폭 중간 스텝 단위로도 재개됨)
  MDEB_CKPT_STEPS=2000  중간 체크포인트 간격 (dataloader step, 0이면 끔)
  MDEB_GRAD_CKPT=1  gradient checkpointing (OOM 시 켤 것 — 속도↓ 메모리↓)
  SMOKE=1           200행·1epoch·maxlen 64 (CPU 파이프라인 검증)
  MDEB_FOLD=0~4     ★ OOF 모드 (P1-C mBERT OOF): 해당 fold를 valid로 학습·예측 후
                    oof_mbert_fold{k}.npz + run json 저장, fold ckpt 삭제, 모델 저장 생략
  MDEB_FOLD_MAP     OOF 모드 필수 — 서버 생성 fold_map.csv 경로 (Colab 재생성 금지,
                    e5 OOF(encoder_e5_oof_fold.py)와 fold 정합 단일 소스)

완료 신호(OOF): `[npz] oof_mbert_fold{k}.npz rows=... macro-F1=...` + `[DONE]`.

시간 주의: T4는 fp32 텐서코어가 없어 fp16 대비 3~5배 느림 — 첫 200 step의 elapsed 로그로
총 소요를 재추정하고, 2에폭이 세션 안에 안 끝날 것 같으면 MDEB_EPOCHS=1로 먼저 완주할 것.

⚠️ T4 함정 (변경 금지):
  - DeBERTa-v3는 fp16 **학습**에서 NaN/overflow 빈발 (disentangled attention 스케일 이슈,
    T4는 bf16 미지원) → 학습은 fp32 고정. 저장 시에만 fp16 사본 생성 (추론은 fp16 안전).
  - loss가 non-finite면 즉시 중단하고 마지막 정상 ckpt 보존.

크기 주의: mdeberta-base fp16 ≈ 550MB (vocab 250k). e5-base(573MB)와 동시 탑재 시 zip 1GB
초과 — 이 모델은 e5-base **대체** 후보로 평가하거나 vocab 축소 후 병용 (수치 보고 후 결정).
"""
import json
import math
import os
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # fork worker 경고/지연 예방

# ---------- env config ----------
def _env(k, d):
    v = os.environ.get(k, "").strip()
    return v if v else d

MODEL = _env("MDEB_MODEL", "microsoft/mdeberta-v3-base")
DATA_DIR = _env("MDEB_DATA_DIR", "./data")
OUT = _env("MDEB_OUT", "./mdeb_out")
MODE = _env("MDEB_MODE", "holdout85")
HOLDOUT_NPZ = _env("MDEB_HOLDOUT_NPZ", "./holdout_base.npz")
SEED = int(_env("MDEB_SEED", "42"))
SMOKE = _env("SMOKE", "0") == "1"
EPOCHS = 1 if SMOKE else int(_env("MDEB_EPOCHS", "2"))
BATCH = int(_env("MDEB_BATCH", "8"))
ACCUM = int(_env("MDEB_ACCUM", "2"))
MAX_LEN = 64 if SMOKE else int(_env("MDEB_MAXLEN", "384"))
MAX_HIST = int(_env("MDEB_MAXHIST", "6"))   # serialize 히스토리 창 (계약 기본 6, hist12 프로브 시 12) — 정의는 불변, 호출만 파라미터화
FOLD = int(_env("MDEB_FOLD", "-1"))         # ★ OOF 모드: 0~4면 해당 fold를 valid로 학습·예측 (P1-C mBERT OOF, fold_map은 서버 생성본만)
FOLD_MAP = _env("MDEB_FOLD_MAP", "")        # 서버 oof_h12/fold_map.csv 경로 — Colab에서 재생성 금지 (e5 OOF와 fold 정합 단일 소스)
LR = float(_env("MDEB_LR", "2e-5"))
RESUME = _env("MDEB_RESUME", "0") == "1"
CKPT_STEPS = int(_env("MDEB_CKPT_STEPS", "2000"))
GRAD_CKPT = _env("MDEB_GRAD_CKPT", "0") == "1"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------- serialize(): submit/script.py verbatim (train·추론 계약 — 변경 금지) ----------
def _bucket(v, edges=(1000, 10000, 50000, 100000)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


def serialize(s, max_hist=6):
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
# ---------- serialize 끝 ----------


def load_train():
    samples = []
    with open(os.path.join(DATA_DIR, "train.jsonl"), encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    labels = {}
    import csv
    with open(os.path.join(DATA_DIR, "train_labels.csv"), newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            labels[row["id"]] = row["action"]
    samples = [s for s in samples if s["id"] in labels]
    return samples, labels


class TextDS(Dataset):
    def __init__(self, texts, ys):
        self.texts, self.ys = texts, ys

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        return self.texts[i], self.ys[i]


def main():
    from sklearn.metrics import f1_score
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    os.makedirs(OUT, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cfg] model={MODEL} mode={MODE} epochs={EPOCHS} batch={BATCH}x{ACCUM} "
          f"maxlen={MAX_LEN} lr={LR} seed={SEED} device={device} smoke={SMOKE}")

    samples, labels = load_train()
    if SMOKE:
        samples = samples[:200]
    ACTIONS = sorted(set(labels.values()))
    assert len(ACTIONS) == 14, f"클래스 수 {len(ACTIONS)} != 14"
    a2i = {a: i for i, a in enumerate(ACTIONS)}
    print(f"[data] rows={len(samples)} classes={len(ACTIONS)} (알파벳순)")

    # ----- split: holdout ids는 npz에서 직접 (재구현 금지) -----
    ids = [s["id"] for s in samples]
    if FOLD >= 0:
        # OOF 모드: valid = fold_map의 해당 fold 행. 맵은 서버(StratifiedGroupKFold s42) 생성본만 —
        # sklearn 버전차로 fold가 갈라지면 e5 OOF와 불일치하므로 여기서 절대 재생성하지 않는다.
        assert not SMOKE, "OOF 모드는 SMOKE와 병용 금지"
        assert MODE == "holdout85", "OOF는 full 모드 개념이 아님 — MDEB_MODE 기본값 유지"
        assert FOLD_MAP and os.path.exists(FOLD_MAP), \
            f"MDEB_FOLD_MAP 없음: {FOLD_MAP!r} — 서버 ~/out/oof_h12/fold_map.csv를 Drive에 올릴 것"
        import csv
        with open(FOLD_MAP, newline="", encoding="utf-8-sig") as f:
            fold_of = {r["id"]: int(r["fold"]) for r in csv.DictReader(f)}
        missing = [i for i in ids if i not in fold_of]
        stale = set(fold_of) - set(ids)
        assert not missing and not stale, \
            f"fold_map 양방향 불일치: train에만 {len(missing)}개 / map에만 {len(stale)}개 — 스테일 맵"
        hold = {i for i in ids if fold_of[i] == FOLD}
        assert hold, f"fold {FOLD} 행이 없음 (map의 fold 값 확인)"
        print(f"[oof] fold={FOLD} map={FOLD_MAP} valid={len(hold)}")
    elif MODE == "holdout85" and not SMOKE:
        hz = np.load(HOLDOUT_NPZ, allow_pickle=True)
        hold = set(str(x) for x in hz["ids"])
        n_missing = len(hold - set(ids))
        assert n_missing == 0, (
            f"holdout npz ids {n_missing}/{len(hold)}개가 train에 없음 — 스테일/오경로 npz. "
            f"파일이 context/night/2026-07-05/holdout_base.npz 최신본인지 확인")
    elif MODE == "holdout85":  # SMOKE: 세션 해시로 15% 대충
        import hashlib
        sess = lambda i: i.rsplit("-step", 1)[0]
        hold = {i for i in ids if int(hashlib.md5(sess(i).encode()).hexdigest(), 16) % 100 < 15}
    else:
        hold = set()

    tr = [s for s in samples if s["id"] not in hold]
    va = [s for s in samples if s["id"] in hold]
    print(f"[split] train={len(tr)} valid={len(va)}")

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=len(ACTIONS),
        id2label={i: a for a, i in a2i.items()}, label2id=a2i,
        ignore_mismatched_sizes=True,
    ).float().to(device)                                  # 학습 fp32 고정 (T4 fp16 NaN 함정)
    if tok.pad_token is None:                              # 디코더 LM(Qwen 등)은 pad 미정의 — eos로 폴백 (기존 인코더 모델 무영향)
        tok.pad_token = tok.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tok.pad_token_id
    if GRAD_CKPT:
        model.gradient_checkpointing_enable()
        print("[cfg] gradient checkpointing on")

    def collate(batch):
        texts = [serialize(s, MAX_HIST) for s, _ in batch]
        ys = torch.tensor([y for _, y in batch], dtype=torch.long)
        enc = tok(texts, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        return enc, ys

    pairs = [(s, a2i[labels[s["id"]]]) for s in tr]

    def make_loader(ep):
        """에폭별 시드 고정 셔플 — 중간 재개 시 같은 순서를 재현해 skip할 수 있게."""
        g = torch.Generator(); g.manual_seed(SEED * 1000 + ep)
        return DataLoader(pairs, batch_size=BATCH, shuffle=True, collate_fn=collate,
                          num_workers=2 if device == "cuda" else 0, generator=g)

    steps_total = math.ceil(len(tr) / (BATCH * ACCUM)) * EPOCHS
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda st: min(1.0, st / max(1, int(steps_total * 0.06)))
        * max(0.0, 1 - st / max(1, steps_total)))

    # OOF는 fold별 ckpt 분리 — 같은 OUT에서 fold 간 재개가 섞이면 조용한 오염
    ckpt_path = os.path.join(OUT, "ckpt.pt" if FOLD < 0 else f"ckpt_fold{FOLD}.pt")

    def save_ckpt(ep, step):
        """step=0 은 'epoch ep 완료', step>0 은 'epoch ep 의 step 까지 진행'."""
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": ep, "step": step}, ckpt_path)

    start_ep, start_step = 0, 0
    if RESUME and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        if int(ck.get("step", 0)) > 0:
            start_ep, start_step = ck["epoch"], int(ck["step"])
            print(f"[resume] epoch {start_ep} step {start_step}부터 재개 (동일 셔플 skip)")
        else:
            start_ep = ck["epoch"] + 1
            print(f"[resume] epoch {start_ep}부터 재개")

    @torch.no_grad()
    def evaluate():
        model.eval()
        probs = np.zeros((len(va), len(ACTIONS)))
        for i in range(0, len(va), BATCH * 4):
            chunk = va[i:i + BATCH * 4]
            enc = tok([serialize(s, MAX_HIST) for s in chunk], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float().cpu().numpy()
            z = logits - logits.max(1, keepdims=True)
            e = np.exp(z); probs[i:i + len(chunk)] = e / e.sum(1, keepdims=True)
        y = [a2i[labels[s["id"]]] for s in va]
        f1 = f1_score(y, probs.argmax(1), average="macro")
        model.train()
        return f1, probs

    model.train()
    for ep in range(start_ep, EPOCHS):
        dl = make_loader(ep)
        skip = start_step if ep == start_ep else 0
        t0, run, seen = time.time(), 0.0, 0
        opt.zero_grad()
        for step, (enc, ys) in enumerate(dl):
            if step < skip:                                # 재개: 같은 셔플 순서로 소비만
                continue
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc, labels=ys.to(device))
            loss = out.loss / ACCUM
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss @ep{ep} step{step} — 마지막 ckpt 사용")
            loss.backward()
            run += loss.item() * ACCUM; seen += 1
            if (step + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad()
            if (step + 1) % 200 == 0:
                print(f"  ep{ep} step {step+1}/{len(dl)} loss={run/max(seen,1):.4f} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)
            if CKPT_STEPS and (step + 1) % CKPT_STEPS == 0:
                save_ckpt(ep, step + 1)
                print(f"  [ckpt] ep{ep} step {step+1} 저장", flush=True)
        save_ckpt(ep, 0)
        if va:
            f1, _ = evaluate()
            print(f"[ep{ep}] valid macro-F1 = {f1:.5f} ({time.time()-t0:.0f}s)")
        else:
            print(f"[ep{ep}] 완료 ({time.time()-t0:.0f}s, full 모드 — valid 없음)")

    # ----- holdout 확률 npz — 반드시 fp16 변환 **전** (half()는 in-place라 되돌려도 열화됨) -----
    if va and FOLD >= 0:
        # OOF npz — 필드·dtype은 encoder_e5_oof_fold.py의 oof_fold{k}.npz와 동일 (스태커 소비 정합)
        f1, probs = evaluate()
        np.savez(os.path.join(OUT, f"oof_mbert_fold{FOLD}.npz"),
                 ids=np.array([s["id"] for s in va], dtype=object),
                 probs=probs.astype(np.float64),
                 y_true=np.array([labels[s["id"]] for s in va], dtype=object),
                 actions=np.array(ACTIONS, dtype=object),
                 fold=np.array([FOLD] * len(va)))
        import hashlib
        with open(FOLD_MAP, "rb") as f:
            map_sha = hashlib.sha256(f.read()).hexdigest()
        with open(os.path.join(OUT, f"run_oof_mbert_fold{FOLD}.json"), "w", encoding="utf-8") as f:
            json.dump({"fold": FOLD, "fold_map": os.path.basename(FOLD_MAP),
                       "fold_map_sha256": map_sha, "model": MODEL,
                       "max_hist": MAX_HIST, "seed": SEED, "epochs": EPOCHS, "max_len": MAX_LEN,
                       "batch": BATCH, "accum": ACCUM, "lr": LR,
                       "n_train": len(tr), "n_valid": len(va),
                       "fold_macro_f1": round(float(f1), 5)}, f, indent=2)
        print(f"[npz] oof_mbert_fold{FOLD}.npz rows={len(va)} macro-F1={f1:.5f}")
        try:
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)  # npz 저장 확인 후에만 — Drive 용량 (fold ckpt ≈ 2GB×5)
                print(f"[clean] {os.path.basename(ckpt_path)} 삭제")
        except OSError as e:  # Drive 잠금 등 — npz는 이미 저장됨, [DONE]을 막지 않는다
            print(f"[warn] ckpt 삭제 실패({e}) — 수동 삭제 필요: {ckpt_path}")
        print("[DONE]")
        return  # fold 모델 저장 없음 (스태커는 확률만 소비)
    if va:
        f1, probs = evaluate()
        np.savez(os.path.join(OUT, "holdout_mdeb.npz"),
                 ids=np.array([s["id"] for s in va]), probs=probs,
                 y_true=np.array([labels[s["id"]] for s in va]),
                 actions=np.array(ACTIONS))
        print(f"[npz] holdout_mdeb.npz rows={len(va)} final macro-F1={f1:.5f}")

    # ----- 저장: fp32 먼저, fp16 사본은 맨 마지막 (이후 이 모델로 평가 금지) -----
    d32 = os.path.join(OUT, "model_fp32"); d16 = os.path.join(OUT, "model_fp16")
    model.save_pretrained(d32); tok.save_pretrained(d32)
    model.half().save_pretrained(d16); tok.save_pretrained(d16)
    size16 = sum(os.path.getsize(os.path.join(r, f))
                 for r, _, fs in os.walk(d16) for f in fs) / 1e6
    print(f"[save] fp16 사본 {size16:.0f}MB — e5-base(573MB)와 합산 1GB 초과 여부 확인할 것")
    print("[DONE]")


if __name__ == "__main__":  # Colab 통짜 붙여넣기(__main__)에서는 기존과 동일 실행, import 재사용 시에만 무실행
    main()
