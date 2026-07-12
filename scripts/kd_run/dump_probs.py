# -*- coding: utf-8 -*-
"""KD-RUN (D-014 Lane B fast path) — teacher softmax dump.

목적: 학습된 e5 인코더(체크포인트 dir / model_fp32 / model_fp16 / submit/model/encoder)를
      로드해 지정한 행 집합(train85 | full | holdout)에 대해 softmax 확률을 npz로 덤프한다.
      이 npz가 train_student.py의 teacher 입력이 된다 (여러 개를 평균하면 앙상블 teacher).

챔피언 계약 재사용 (colab/encoder_e5_holdout85_maxhist.py, 읽기 전용 — 수정 금지):
  - serialize(s, max_hist, args_lite), load_samples(), ACTIONS, LABEL2ID 를 그대로 import.
  - softmax 수치: logits.float() → row-max subtract → exp → normalize (챔피언과 동일).
  - DATA_DIR은 그 모듈이 import 시점에 env(ENC_DATA_DIR)를 한 번만 읽으므로, import 이전에
    os.environ["ENC_DATA_DIR"]을 세팅한다.

env:
  KD_MODEL_DIR   필수 (assert, argparse required 사용 안 함) — 모델 dir (HF repo id도 허용)
  KD_ROWS        train85(기본) | full | holdout
  KD_HOLDOUT_NPZ 기본 ./data/holdout_base.npz
  KD_DATA_DIR    기본 ./data
  KD_OUT         기본 ./kd_dump.npz
  KD_MAXHIST     기본 12 (배포 모델 hist12 계약 — exp #48)
  KD_MAXLEN      기본 384
  KD_BATCH       기본 64 (추론 — gradient 없음, 학습 batch보다 크게)
  KD_LIMIT       기본 0 (off) — >0이면 행 목록을 앞에서부터 N개로 자름 (smoke 전용)

완료 신호: `[DONE] npz=... rows=... shape=...`
"""
import importlib.util
import os
import sys
import time

import numpy as np
import torch


def _env(k, d):
    v = os.environ.get(k, "").strip()
    return v if v else d


def _load_champion_module():
    """colab/encoder_e5_holdout85_maxhist.py 를 cwd 무관하게 spec_from_file_location 으로 로드.

    주의: 이 모듈은 import 시점에 ENC_DATA_DIR 등 env를 한 번만 읽어 DATA_DIR 등 module-level
    상수를 고정한다 (top-level `_env()` 호출). 그래서 반드시 import *이전에* 원하는 env를
    세팅해야 한다 — 이 함수를 호출하기 전에 os.environ["ENC_DATA_DIR"]를 지정해 둘 것.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    champion_path = os.path.normpath(os.path.join(here, "..", "..", "colab",
                                                    "encoder_e5_holdout85_maxhist.py"))
    assert os.path.exists(champion_path), f"champion script not found: {champion_path}"
    spec = importlib.util.spec_from_file_location("encoder_e5_holdout85_maxhist", champion_path)
    mod = importlib.util.module_from_spec(spec)
    # __main__ guard가 있으므로 exec_module은 함수/상수 정의만 실행하고 main()은 돌리지 않는다.
    spec.loader.exec_module(mod)
    return mod


def main():
    kd_model_dir = os.environ.get("KD_MODEL_DIR", "").strip()
    assert kd_model_dir, (
        "KD_MODEL_DIR 미설정 — 모델 dir(checkpoint-*, model_fp32, model_fp16, "
        "submit/model/encoder) 또는 HF repo id를 지정하세요."
    )
    rows_mode = _env("KD_ROWS", "train85")
    assert rows_mode in ("train85", "full", "holdout"), f"KD_ROWS 값 오류: {rows_mode}"
    holdout_npz = _env("KD_HOLDOUT_NPZ", "./data/holdout_base.npz")
    data_dir = _env("KD_DATA_DIR", "./data")
    out_path = _env("KD_OUT", "./kd_dump.npz")
    max_hist = int(_env("KD_MAXHIST", "12"))
    max_len = int(_env("KD_MAXLEN", "384"))
    batch = int(_env("KD_BATCH", "64"))
    limit = int(_env("KD_LIMIT", "0"))

    # DATA_DIR은 챔피언 모듈 import 시점에 한 번만 읽힌다 — import 전에 세팅.
    os.environ["ENC_DATA_DIR"] = data_dir
    champ = _load_champion_module()

    # sidecar run_h*.json 이 있으면 max_hist 참고 정보로 출력 (기본값을 덮어쓰지는 않음 — 명시적
    # KD_MAXHIST가 우선. 스펙: "확인만 하고 default는 12로 유지").
    sidecar_hint = None
    model_parent = os.path.dirname(os.path.normpath(kd_model_dir)) if os.path.isdir(kd_model_dir) else None
    for probe_dir in filter(None, [kd_model_dir, model_parent]):
        if probe_dir and os.path.isdir(probe_dir):
            for fn in os.listdir(probe_dir):
                if fn.startswith("run_h") and fn.endswith(".json"):
                    sidecar_hint = os.path.join(probe_dir, fn)
                    break
        if sidecar_hint:
            break
    if sidecar_hint:
        print(f"[hint] sidecar run json found: {sidecar_hint} (KD_MAXHIST default/override still governs)")

    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "") else "cpu"
    print(f"[cfg] model_dir={kd_model_dir} rows={rows_mode} max_hist={max_hist} max_len={max_len} "
          f"batch={batch} limit={limit} device={device} out={out_path}")

    samples, lab = champ.load_samples()
    assert len(champ.ACTIONS) == 14
    n_actions = len(champ.ACTIONS)

    if rows_mode == "full":
        rows = samples
    else:
        assert os.path.exists(holdout_npz), f"holdout npz 없음: {holdout_npz}"
        hz = np.load(holdout_npz, allow_pickle=True)
        hold = set(str(x) for x in hz["ids"])
        if rows_mode == "train85":
            rows = [s for s in samples if s["id"] not in hold]
        else:  # holdout
            rows = [s for s in samples if s["id"] in hold]
    print(f"[split] rows_mode={rows_mode} n_rows={len(rows)}")

    if limit > 0:
        rows = rows[:limit]
        print(f"[limit] truncated to first {len(rows)} rows (KD_LIMIT={limit}, smoke-test only)")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # ⚠️ Trainer checkpoint-* 디렉토리에는 토크나이저 파일이 없다. transformers 5.13은 이때
    # 실패하는 대신 vocab_size=5짜리 깡통 XLMRobertaTokenizer를 조용히 생성한다 (2026-07-12 실측:
    # 전 입력이 <unk>로 토큰화 → 덤프 in-sample acc 6% 참사). KD_TOK_DIR로 명시하거나
    # 모델 디렉토리 로드 결과가 깡통이면 베이스 모델 토크나이저로 폴백하고, vocab 하한을 강제한다.
    kd_tok_dir = os.getenv("KD_TOK_DIR", "")
    if kd_tok_dir:
        tok = AutoTokenizer.from_pretrained(kd_tok_dir)
    else:
        tok = AutoTokenizer.from_pretrained(kd_model_dir)
        if tok.vocab_size < 100_000:
            print(f"[tok] model_dir tokenizer degenerate (vocab={tok.vocab_size}) — falling back to {champ.MODEL_NAME}")
            tok = AutoTokenizer.from_pretrained(champ.MODEL_NAME)
    assert tok.vocab_size >= 100_000, f"degenerate tokenizer: vocab_size={tok.vocab_size}"
    # num_labels/id2label/label2id 명시: 이미 파인튜닝된 체크포인트(checkpoint-*, model_fp32/fp16)는
    # config에 14-way 헤드가 저장돼 있어 이 인자가 사실상 no-op 검증이지만, 스모크 테스트처럼
    # 파인튜닝 전 원본 HF 모델(예: intfloat/multilingual-e5-base)을 가리킬 경우 분류 헤드가
    # 없으므로 랜덤 초기화될 때 shape을 14로 강제해야 한다 (기본 num_labels=2로 깨짐 방지).
    model = AutoModelForSequenceClassification.from_pretrained(
        kd_model_dir, num_labels=n_actions,
        id2label={i: a for a, i in champ.LABEL2ID.items()}, label2id=champ.LABEL2ID,
        ignore_mismatched_sizes=True)
    model.to(device)
    model.eval()

    ids_out = []
    probs_out = np.zeros((len(rows), n_actions), dtype=np.float32)

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            texts = [champ.serialize(s, max_hist, False) for s in chunk]
            enc = tok(texts, truncation=True, max_length=max_len, padding=True,
                      return_tensors="pt").to(device)
            logits = model(**enc).logits.float().cpu().numpy()
            z = logits - logits.max(1, keepdims=True)
            ex = np.exp(z)
            p = ex / ex.sum(1, keepdims=True)
            # ids와 probs를 같은 루프에서 같은 순서로 append — id/probs 순서 불일치 방지.
            for j, s in enumerate(chunk):
                ids_out.append(s["id"])
            probs_out[i:i + len(chunk)] = p.astype(np.float32)
            if (i // batch) % 10 == 0 or i + batch >= len(rows):
                elapsed = time.time() - t0
                print(f"[progress] {min(i + batch, len(rows))}/{len(rows)} rows ({elapsed:.1f}s)")

    ids_arr = np.array(ids_out, dtype=object)
    assert len(ids_arr) == probs_out.shape[0], "id/probs row count mismatch — dump 로직 점검 필요"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    np.savez(out_path, ids=ids_arr, probs=probs_out,
              actions=np.array(champ.ACTIONS, dtype=object))

    print(f"[DONE] npz={out_path} rows={len(ids_arr)} shape={probs_out.shape}")


if __name__ == "__main__":
    main()
