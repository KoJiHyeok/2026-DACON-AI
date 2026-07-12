# -*- coding: utf-8 -*-
"""KD-RUN (D-014 Lane B fast path) — e5-base KD student trainer.

챔피언 레시피(colab/encoder_e5_holdout85_maxhist.py, 읽기 전용)를 그대로 재사용하고
teacher soft-label(하나 이상의 dump_probs.py npz를 평균)로 KD 항을 더한다.

Loss:
  total = (1-λ)*CE_classweighted(y, label_smoothing=0.1)
        + λ * T^2 * KL( log_softmax(z_student/T) ‖ normalize(teacher_probs**(1/T)) )

λ=0이면 수치적으로 WeightedTrainer.compute_loss(sw=None 분기)와 동일해야 한다 — --selftest로 검증.

env:
  KD_TEACHER_NPZ   콤마 구분 dump npz 경로 목록 (필수 — selftest 모드에서는 생략 가능)
  KD_MODE          holdout85(기본) | full
  KD_LAMBDA        기본 0.5
  KD_T             기본 2.0 (temperature)
  KD_HOLDOUT_NPZ   기본 ./data/holdout_base.npz
  KD_DATA_DIR      기본 ./data
  KD_OUT           기본 ./kd_student_out
  KD_MAXHIST       기본 12
  KD_MAXLEN        기본 384 (챔피언 기본값 — 변경 금지, smoke 목적 외 override 금지)
  KD_BATCH         기본 16 (학습 배치 — 챔피언과 동일)
  KD_SEED          기본 42
  KD_LIMIT         기본 0 (off) — smoke test 전용, 학습 표본을 앞 N개로 자름
  KD_EPOCHS        미설정시 챔피언 기본 6 (smoke test 전용 override)
  KD_SELFTEST      1이면 --selftest와 동일 (CLI 플래그도 지원)

완료 신호: `[DONE]`
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as TF


def _env(k, d):
    v = os.environ.get(k, "").strip()
    return v if v else d


def _load_champion_module(data_dir):
    import importlib.util
    here = os.path.dirname(os.path.abspath(__file__))
    champion_path = os.path.normpath(os.path.join(here, "..", "..", "colab",
                                                    "encoder_e5_holdout85_maxhist.py"))
    assert os.path.exists(champion_path), f"champion script not found: {champion_path}"
    # DATA_DIR은 챔피언 모듈 import 시점에 한 번만 읽힌다 — import 전에 세팅.
    os.environ["ENC_DATA_DIR"] = data_dir
    spec = importlib.util.spec_from_file_location("encoder_e5_holdout85_maxhist", champion_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_teacher_npzs(paths):
    """콤마 구분 npz 경로 목록을 로드, ids가 순서까지 동일한지 검증 후 probs를 평균."""
    assert paths, "KD_TEACHER_NPZ 비어있음"
    loaded = []
    for p in paths:
        p = p.strip()
        assert os.path.exists(p), f"teacher npz 없음: {p}"
        z = np.load(p, allow_pickle=True)
        ids = np.array([str(x) for x in z["ids"]], dtype=object)
        probs = np.asarray(z["probs"], dtype=np.float64)
        loaded.append((p, ids, probs))

    ref_path, ref_ids, _ = loaded[0]
    mismatches = []
    for p, ids, _ in loaded[1:]:
        if len(ids) != len(ref_ids):
            mismatches.append(f"{p}: len={len(ids)} != {ref_path} len={len(ref_ids)}")
            continue
        diff_idx = np.nonzero(ids != ref_ids)[0]
        if len(diff_idx) > 0:
            first = diff_idx[0]
            mismatches.append(
                f"{p}: id mismatch vs {ref_path} at index {first} "
                f"({ids[first]!r} != {ref_ids[first]!r}), {len(diff_idx)} total mismatches"
            )
    assert not mismatches, "teacher npz ids 불일치 (순서까지 동일해야 함):\n" + "\n".join(mismatches)

    avg_probs = np.mean(np.stack([p for _, _, p in loaded], axis=0), axis=0)
    return ref_ids, avg_probs.astype(np.float32), [p for p, _, _ in loaded]


class KDDataset(torch.utils.data.Dataset):
    """enc(tokenizer 출력) + labels + teacher_probs(row-aligned) 를 담는 Dataset.

    teacher_probs를 item dict에 "teacher_probs" 키로 넣어 HF Trainer collator를 통해
    compute_loss의 inputs로 흘려보낸다. remove_unused_columns=False가 필수
    (기본 True면 collator가 모델 forward 시그니처에 없는 컬럼을 제거해버림 — 챔피언 스크립트의
    sw 컬럼과 동일한 함정, TrainingArguments에서 반드시 False로 설정).
    """

    def __init__(self, enc, labels, teacher_probs):
        self.enc = enc
        self.labels = labels
        self.teacher_probs = teacher_probs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(int(self.labels[i]))
        item["teacher_probs"] = torch.tensor(self.teacher_probs[i], dtype=torch.float32)
        return item


def make_kd_trainer_cls(base_cls, kd_lambda, kd_t, label_smooth):
    """base_cls(챔피언 WeightedTrainer)를 상속해 compute_loss에 KD 항을 추가한 클래스를 생성.

    kd_lambda=0이면 teacher_probs를 pop만 하고 순수 CE(부모와 동일 코드 경로,
    sw=None 분기)를 반환 — --selftest에서 이 조건을 수치로 검증한다.
    """

    class KDTrainer(base_cls):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            teacher_probs = inputs.pop("teacher_probs", None)
            labels = inputs.pop("labels")
            sw = inputs.pop("sw", None)
            assert sw is None, "KD student는 session-weight(sw) 미사용 경로 — sw 컬럼이 들어오면 설계 오류"

            outputs = model(**inputs)
            w = self.class_w.to(outputs.logits.device)
            ce = TF.cross_entropy(outputs.logits, labels, weight=w,
                                  label_smoothing=label_smooth)

            if kd_lambda <= 0.0 or teacher_probs is None:
                loss = ce
            else:
                logits = outputs.logits
                t = kd_t
                # teacher_T = normalize(teacher_probs ** (1/T)) — row sum 1로 재정규화
                tp = teacher_probs.to(logits.device).clamp_min(1e-12)
                tp_t = tp.pow(1.0 / t)
                tp_t = tp_t / tp_t.sum(dim=-1, keepdim=True)

                log_p_student = TF.log_softmax(logits / t, dim=-1)
                kd = TF.kl_div(log_p_student, tp_t, reduction="batchmean", log_target=False)
                kd = kd * (t ** 2)

                loss = (1.0 - kd_lambda) * ce + kd_lambda * kd

            return (loss, outputs) if return_outputs else loss

    return KDTrainer


def build_dataset(champ, samples, lab, ids_order, teacher_probs_by_id, max_hist, max_len, tok):
    """samples를 ids_order와 동일한 순서로 정렬하고, 그 순서로 tokenize + teacher_probs 정렬."""
    id_to_sample = {s["id"]: s for s in samples}
    missing = [i for i in ids_order if i not in id_to_sample]
    assert not missing, f"teacher npz ids 중 samples에 없는 것 {len(missing)}개 (예: {missing[:3]})"

    ordered = [id_to_sample[i] for i in ids_order]
    texts = [champ.serialize(s, max_hist, False) for s in ordered]
    y = np.array([champ.LABEL2ID[lab[s["id"]]] for s in ordered])
    enc = tok(texts, truncation=True, max_length=max_len, padding="max_length")
    tp = np.stack([teacher_probs_by_id[i] for i in ids_order], axis=0).astype(np.float32)
    return enc, y, tp, ordered


def run_selftest(champ, args):
    """λ=0일 때 KD-augmented compute_loss == 챔피언 WeightedTrainer.compute_loss(sw=None) 임을
    실제 미니배치로 수치 검증. 가장 중요한 검증 — 이게 깨지면 KD 파이프라인 전체가 무효."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print("[selftest] building tiny real batch from data...")
    samples, lab = champ.load_samples()
    small = samples[:8]
    tok = AutoTokenizer.from_pretrained(champ.MODEL_NAME)
    texts = [champ.serialize(s, args["max_hist"], False) for s in small]
    y = np.array([champ.LABEL2ID[lab[s["id"]]] for s in small])
    enc = tok(texts, truncation=True, max_length=args["max_len"], padding="max_length",
              return_tensors="pt")

    model = AutoModelForSequenceClassification.from_pretrained(
        champ.MODEL_NAME, num_labels=len(champ.ACTIONS),
        id2label={i: a for a, i in champ.LABEL2ID.items()}, label2id=champ.LABEL2ID)
    model.eval()

    counts = np.bincount(y, minlength=len(champ.ACTIONS))
    class_w = torch.tensor(len(y) / (len(champ.ACTIONS) * np.maximum(counts, 1)), dtype=torch.float32)

    labels_t = torch.tensor(y)
    dummy_teacher = torch.full((len(small), len(champ.ACTIONS)), 1.0 / len(champ.ACTIONS))

    with torch.no_grad():
        outputs = model(**enc)

    # (a) 챔피언 WeightedTrainer.compute_loss 경로 그대로 재현 (sw=None 분기)
    loss_champion = TF.cross_entropy(outputs.logits, labels_t, weight=class_w,
                                     label_smoothing=champ.LABEL_SMOOTH)

    # (b) KD 서브클래스, kd_lambda=0
    KDTrainerCls = make_kd_trainer_cls(champ.WeightedTrainer, kd_lambda=0.0, kd_t=args["kd_t"],
                                       label_smooth=champ.LABEL_SMOOTH)

    class _Shim:
        class_w = None
    shim = _Shim()
    shim.class_w = class_w

    inputs = {k: v.clone() for k, v in enc.items()}
    inputs["labels"] = labels_t.clone()
    inputs["teacher_probs"] = dummy_teacher.clone()

    # compute_loss는 인스턴스 메서드이므로 unbound 함수로 호출 (self=shim)
    with torch.no_grad():
        loss_kd = KDTrainerCls.compute_loss(shim, model, inputs, return_outputs=False)

    diff = abs(float(loss_champion.item()) - float(loss_kd.item()))
    passed = diff < 1e-5
    print(f"[selftest] loss_champion(plain weighted CE) = {loss_champion.item():.8f}")
    print(f"[selftest] loss_kd(lambda=0)                 = {loss_kd.item():.8f}")
    print(f"[selftest] abs_diff = {diff:.3e} (tol=1e-5)")
    print(f"[selftest] {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    selftest = "--selftest" in sys.argv or _env("KD_SELFTEST", "0") == "1"

    data_dir = _env("KD_DATA_DIR", "./data")
    champ = _load_champion_module(data_dir)

    max_hist = int(_env("KD_MAXHIST", "12"))
    max_len = int(_env("KD_MAXLEN", str(champ.MAX_LEN)))
    kd_t = float(_env("KD_T", "2.0"))

    if selftest:
        ok = run_selftest(champ, {"max_hist": max_hist, "max_len": max_len, "kd_t": kd_t})
        sys.exit(0 if ok else 1)

    kd_teacher_npz = _env("KD_TEACHER_NPZ", "")
    assert kd_teacher_npz, "KD_TEACHER_NPZ 미설정 (콤마 구분 dump_probs.py 출력 npz 경로 목록 필요)"
    teacher_paths = [p.strip() for p in kd_teacher_npz.split(",") if p.strip()]

    mode = _env("KD_MODE", "holdout85")
    assert mode in ("holdout85", "full"), f"KD_MODE 값 오류: {mode}"
    kd_lambda = float(_env("KD_LAMBDA", "0.5"))
    holdout_npz = _env("KD_HOLDOUT_NPZ", "./data/holdout_base.npz")
    out_dir = _env("KD_OUT", "./kd_student_out")
    batch = int(_env("KD_BATCH", "16"))
    seed = int(_env("KD_SEED", "42"))
    limit = int(_env("KD_LIMIT", "0"))
    epochs = int(_env("KD_EPOCHS", str(champ.EPOCHS)))

    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    print(f"[cfg] mode={mode} lambda={kd_lambda} T={kd_t} max_hist={max_hist} max_len={max_len} "
          f"batch={batch} epochs={epochs} seed={seed} limit={limit} out={out_dir}")
    print(f"[cfg] teacher_npz={teacher_paths}")

    teacher_ids, teacher_probs, teacher_paths_ok = load_teacher_npzs(teacher_paths)
    teacher_hashes = {p: _sha256(p) for p in teacher_paths_ok}
    print(f"[teacher] n_teachers={len(teacher_paths_ok)} n_rows={len(teacher_ids)}")
    for p, h in teacher_hashes.items():
        print(f"[teacher] {p} sha256={h}")

    samples, lab = champ.load_samples()
    assert len(champ.ACTIONS) == 14

    # KD_LIMIT>0(smoke-test 전용)일 때는 teacher npz도 KD_LIMIT으로 잘려 있는 게 정상 흐름이므로
    # "teacher가 train85/70k 전량을 덮어야 한다"는 엄격한 커버리지 검증을 건너뛴다. 실전
    # (KD_LIMIT=0)에서는 계속 전량 일치를 강제한다 — 여기서 완화하면 안 됨(행 정렬 안전장치의 핵심).
    if mode == "full":
        sample_ids = set(s["id"] for s in samples)
        if limit == 0:
            assert set(teacher_ids) >= sample_ids, (
                "KD_MODE=full 인데 teacher npz ids가 70k 전량을 덮지 않음 — teacher 덤프를 KD_ROWS=full로 재생성 필요"
            )
        else:
            assert set(teacher_ids) <= sample_ids, "teacher npz에 samples에 없는 id가 섞여 있음"
        train_ids_order = [i for i in teacher_ids if i in sample_ids]
    else:
        assert os.path.exists(holdout_npz), f"holdout npz 없음: {holdout_npz}"
        hz = np.load(holdout_npz, allow_pickle=True)
        hold = set(str(x) for x in hz["ids"])
        sample_ids = set(s["id"] for s in samples)
        train_ids_set = sample_ids - hold
        teacher_ids_set = set(teacher_ids)
        if limit == 0:
            assert teacher_ids_set == train_ids_set, (
                f"teacher npz ids != train85 ids (holdout 제외 non-holdout 집합) — "
                f"teacher에만 있는 것 {len(teacher_ids_set - train_ids_set)}개, "
                f"train85에만 있는 것 {len(train_ids_set - teacher_ids_set)}개. "
                f"teacher 덤프가 KD_ROWS=train85로 만들어졌는지, 같은 holdout npz를 썼는지 확인 필요."
            )
        else:
            assert teacher_ids_set <= train_ids_set, (
                f"teacher npz ids 중 train85(non-holdout)에 없는 것 {len(teacher_ids_set - train_ids_set)}개 "
                f"— holdout 행이 섞였거나 다른 split 기준 npz일 가능성 (KD_LIMIT smoke 모드에서도 이 부분집합 "
                f"관계는 유지되어야 함)"
            )
        train_ids_order = list(teacher_ids)  # teacher npz의 행 순서를 기준으로 학습 데이터셋을 구성

    if limit > 0:
        train_ids_order = train_ids_order[:limit]
        print(f"[limit] truncated to first {len(train_ids_order)} train rows (KD_LIMIT={limit}, smoke-only)")

    teacher_probs_by_id = {i: teacher_probs[idx] for idx, i in enumerate(teacher_ids)}

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments

    tok = AutoTokenizer.from_pretrained(champ.MODEL_NAME)
    enc, tr_y, tr_teacher, ordered_samples = build_dataset(
        champ, samples, lab, train_ids_order, teacher_probs_by_id, max_hist, max_len, tok)
    print(f"[data] n_train={len(tr_y)}")

    device = "cuda" if (torch.cuda.is_available() and os.environ.get("CUDA_VISIBLE_DEVICES", "x") != "") else "cpu"
    model = AutoModelForSequenceClassification.from_pretrained(
        champ.MODEL_NAME, num_labels=len(champ.ACTIONS),
        id2label={i: a for a, i in champ.LABEL2ID.items()}, label2id=champ.LABEL2ID).to(device)

    counts = np.bincount(tr_y, minlength=len(champ.ACTIONS))
    class_w = torch.tensor(len(tr_y) / (len(champ.ACTIONS) * np.maximum(counts, 1)), dtype=torch.float32)

    ckpt_dir = os.path.join(out_dir, "ckpt")
    training_args = TrainingArguments(
        output_dir=ckpt_dir,
        num_train_epochs=epochs, learning_rate=champ.LR,
        per_device_train_batch_size=batch,
        warmup_ratio=0.06, weight_decay=0.01, fp16=(device == "cuda"),
        save_strategy="epoch", save_total_limit=1,
        logging_steps=max(1, min(200, len(tr_y) // max(1, batch))),
        seed=seed, report_to="none",
        remove_unused_columns=False,  # teacher_probs 컬럼이 collator에서 제거되지 않도록 필수
    )

    KDTrainerCls = make_kd_trainer_cls(champ.WeightedTrainer, kd_lambda=kd_lambda, kd_t=kd_t,
                                       label_smooth=champ.LABEL_SMOOTH)
    train_dataset = KDDataset(enc, tr_y, tr_teacher)
    trainer = KDTrainerCls(model=model, args=training_args, train_dataset=train_dataset)
    trainer.class_w = class_w

    t0 = time.time()
    train_result = trainer.train()
    train_seconds = time.time() - t0
    print(f"[train] done in {train_seconds:.1f}s, final loss={train_result.training_loss:.6f}")

    os.makedirs(out_dir, exist_ok=True)

    run_meta = {
        "kd_mode": mode,
        "kd_lambda": kd_lambda,
        "kd_t": kd_t,
        "max_hist": max_hist,
        "max_len": max_len,
        "batch": batch,
        "epochs": epochs,
        "seed": seed,
        "limit": limit,
        "n_train": len(tr_y),
        "teacher_npz": teacher_paths_ok,
        "teacher_npz_sha256": teacher_hashes,
        "training_loss": float(train_result.training_loss),
        "train_seconds": round(train_seconds, 1),
        "torch_version": torch.__version__,
    }
    try:
        import transformers
        run_meta["transformers_version"] = transformers.__version__
    except Exception:
        run_meta["transformers_version"] = None
    try:
        import accelerate
        run_meta["accelerate_version"] = accelerate.__version__
    except Exception:
        run_meta["accelerate_version"] = None

    if mode == "full":
        d32 = os.path.join(out_dir, "model_fp32")
        d16 = os.path.join(out_dir, "model_fp16")
        model.save_pretrained(d32); tok.save_pretrained(d32)
        model.half().save_pretrained(d16); tok.save_pretrained(d16)  # half()는 in-place — fp32 저장 이후에만 호출
        size16 = sum(os.path.getsize(os.path.join(r, f))
                     for r, _, fs in os.walk(d16) for f in fs) / 1e6
        run_meta["fp16_MB"] = round(size16, 1)
        with open(os.path.join(out_dir, "run_kd_full.json"), "w", encoding="utf-8") as f:
            json.dump(run_meta, f, indent=2, ensure_ascii=False)
        print(f"[save] full-train fp16 {size16:.0f}MB")
        print("[DONE]")
        return

    # holdout85: holdout 행에 대해 추론 + macro-F1 (챔피언과 동일 소프트맥스 수치)
    hz = np.load(holdout_npz, allow_pickle=True)
    hold = set(str(x) for x in hz["ids"])
    va = [s for s in samples if s["id"] in hold]

    model.eval()
    n_actions = len(champ.ACTIONS)
    probs = np.zeros((len(va), n_actions), dtype=np.float64)
    infer_batch = batch * 4
    with torch.no_grad():
        for i in range(0, len(va), infer_batch):
            chunk = va[i:i + infer_batch]
            e = tok([champ.serialize(s, max_hist, False) for s in chunk], truncation=True,
                    max_length=max_len, padding=True, return_tensors="pt").to(device)
            logits = model(**e).logits.float().cpu().numpy()
            z = logits - logits.max(1, keepdims=True)
            ex = np.exp(z)
            probs[i:i + len(chunk)] = ex / ex.sum(1, keepdims=True)

    from sklearn.metrics import f1_score
    y_true = np.array([lab[s["id"]] for s in va], dtype=object)
    pred = np.array(champ.ACTIONS)[probs.argmax(1)]
    f1 = f1_score([str(y) for y in y_true], pred, average="macro")

    npz_path = os.path.join(out_dir, "holdout_kd.npz")
    np.savez(npz_path,
             ids=np.array([s["id"] for s in va], dtype=object),
             probs=probs.astype(np.float32), y_true=y_true,
             actions=np.array(champ.ACTIONS, dtype=object))

    run_meta["n_holdout"] = len(va)
    run_meta["solo_macro_f1"] = round(float(f1), 5)
    with open(os.path.join(out_dir, "run_kd_holdout85.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2, ensure_ascii=False)

    print(f"[npz] {npz_path} rows={len(va)} macro-F1={f1:.5f}")
    print("[DONE]")


if __name__ == "__main__":
    main()
