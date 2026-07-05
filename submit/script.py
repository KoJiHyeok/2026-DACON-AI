# -*- coding: utf-8 -*-
"""3-way 이종 앙상블 추론 (linear × stacker × encoder block).

현재 기준: v2 s42 3-way 1,1,1.5가 팀 최고 LB 0.7200(uniform 0.7190).
      이종 성분(linear 0.6732 / stacker 0.6708 / encoder 단독 0.701)의 평균이 유효 경로.

세 성분(각각 14-class 확률, 클래스명으로 정렬 후 평균/가중평균 → argmax):
  1) linear  = 우리 챔피언(submit/model/model.pkl): decision_function(+class_bias) → softmax
     · features.py = submit/features.py (그 모델과 짝. src/features.py 아님 — 짝 틀리면 조용한 오답)
  2) stacker = work2 AAR(model/stacker/aar_config.json + aar_models.joblib): predict_aar 확률 경로
     · aar_infer.py = work2/script.py **verbatim 벤더**(자기완결, base 패키지만 import). 아티팩트는 읽기만.
  3) encoder block = model/encoder[, encoder_2...] 확률 uniform 평균. serialize() 학습 계약 verbatim.

제약(상위 CLAUDE.md 4절): 오프라인(원격 0, HF_OFFLINE) · 상대경로 · UTF-8 · ≤1GB · 30k 추론 ≤10분.
  크기(현재 v2): linear 8MiB + stacker 92MiB + base fp16 encoder 547MiB ≈ 647MiB.
  4-way(base fp16 + small fp16) 추정 ≈ 888MiB로 1GiB 아래.
  시간: encoder 배치추론(T4 fp16) 수십초 + stacker/linear TF-IDF ~수분 → 여유.

선택 제어:
  · model/weights.json 또는 ENS_WEIGHTS: 3슬롯(lin,stk,encoder-block) 가중.
  · model/bucket_weights.json 또는 ENS_BUCKET_WEIGHTS: 버킷별 3슬롯 가중(기본 off).
  · model/calib.json: encoder block에 softmax(log(p)/T + bias) 적용.
  · model/encoder_2...: encoder block 내부 uniform 평균.

경로(패키징 기본 = ./model/{linear,stacker,encoder[,encoder_2…]}). 인코더가 2개 이상이면
확률을 uniform 평균(seed·이종 인코더 앙상블). 스모크는 아래 env로 실아티팩트 지정(복사 회피):
  ENS_LINEAR_PKL · ENS_STACKER_DIR · ENS_ENCODER_DIR(콤마 구분 복수 가능) · ENS_DATA · ENS_OUT
"""
import csv
import json
import math
import os
import re
import sys
import time
from collections import Counter

os.environ.setdefault("HF_HUB_OFFLINE", "1")        # 추론 중 인터넷 없음(대회 계약)
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# ⚠️ torch 를 numpy/sklearn 보다 먼저 import. Windows 로컬에서 torch c10.dll 이 numpy/MKL(OpenMP)
#   뒤에 로드되면 WinError 1114(DLL init fail) 발생(실측). 서버(Linux)는 무관하나 torch-first 로
#   고정하면 양쪽 다 안전. 3-way 는 encoder 때문에 torch 가 항상 필요하므로 top-level import.
import torch  # noqa: E402,F401
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import features as F     # linear (= submit/features.py 벤더 — 챔피언과 짝)
import aar_infer as AAR  # stacker (= work2/script.py verbatim 벤더)

DATA = os.environ.get("ENS_DATA", "./data")
MODEL = "./model"
OUT = os.environ.get("ENS_OUT", "./output")
LINEAR_PKL = os.environ.get("ENS_LINEAR_PKL", os.path.join(MODEL, "linear", "model.pkl"))
STACKER_DIR = os.environ.get("ENS_STACKER_DIR", os.path.join(MODEL, "stacker"))
MAX_LEN = 384
BATCH = 64


def encoder_dirs():
    """인코더 성분 디렉토리 목록. ENS_ENCODER_DIR(콤마 구분 가능)이 우선, 기본은
    model/encoder + model/encoder_* (이름순). 2개 이상이면 확률을 uniform 평균
    (seed 앙상블·이종 인코더용) — 1개면 기존 단일 인코더와 동일 동작."""
    raw = os.environ.get("ENS_ENCODER_DIR", "").strip()
    if raw:
        dirs = [p.strip() for p in raw.split(",") if p.strip()]
    else:
        dirs = []
        if os.path.isdir(MODEL):
            names = sorted(d for d in os.listdir(MODEL)
                           if d == "encoder" or d.startswith("encoder_"))
            dirs = [os.path.join(MODEL, d) for d in names
                    if os.path.isdir(os.path.join(MODEL, d))]
    if not dirs:
        raise FileNotFoundError("encoder model dir 없음 (model/encoder* 또는 ENS_ENCODER_DIR)")
    return dirs

# 세 성분 정렬 기준 = 우리 ACTIONS 순서. stacker(AAR)도 같은 순서여야 함(방어적 assert).
ACTIONS = list(F.ACTIONS)
assert ACTIONS == list(AAR.ACTIONS), "linear/stacker ACTIONS 순서 불일치 — 정렬 기준 깨짐"


def softmax(z, temp=1.0):
    z = np.asarray(z, dtype=np.float64) / max(temp, 1e-6)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def align_cols(probs, src_labels):
    """probs(열=src_labels 순) → ACTIONS 순 재배치. 14개 전부 존재 가정."""
    idx = [list(src_labels).index(a) for a in ACTIONS]
    return np.asarray(probs)[:, idx]


# ===== encoder serialize(): colab/encoder_finetune.py 학습본 verbatim (train·추론 계약) =====
def _bucket(v, edges=(1000, 10000, 50000, 100000)):
    if v is None:
        return "na"
    for i, e in enumerate(edges):
        if v < e:
            return f"b{i}"
    return f"b{len(edges)}"


def serialize(s, max_hist=6):
    """샘플 1개 → 인코더 입력 텍스트. ⚠️ 제출 script.py에서 이 함수를 그대로 재사용해야 함
    (train·추론 직렬화 동일 = 계약, ai-2026/CLAUDE.md 7절 4항과 같은 원리)."""
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
            items.append(f"act:{h.get('name')} r:{str(h.get('result_summary') or '')[:120]}")
        else:
            items.append(f"user: {str(h.get('content') or '')[:200]}")
    parts.append("recent: " + " | ".join(items))
    return "\n".join(parts)
# ===== serialize 복사 끝 =====


def linear_probs(samples):
    """우리 챔피언 확률 = softmax(decision_function + class_bias). ACTIONS 순 반환."""
    obj = joblib.load(LINEAR_PKL)
    pipe = obj["pipe"] if isinstance(obj, dict) else obj
    bias_map = obj.get("class_bias") if isinstance(obj, dict) else None
    df = F.build_dataframe(samples)
    classes = list(pipe.named_steps["clf"].classes_)
    if hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(df)
        if bias_map:                                       # 챔피언 0.6753 = bias 포함
            bias = np.array([float(bias_map.get(str(c), 0.0)) for c in classes])
            scores = scores + bias.reshape(1, -1)
        p = softmax(scores)
    elif hasattr(pipe, "predict_proba"):
        p = np.asarray(pipe.predict_proba(df), dtype=np.float64)
    else:                                                  # 최후 방어: one-hot
        preds = [str(x) for x in pipe.predict(df)]
        p = np.zeros((len(preds), len(classes)))
        ci = {c: i for i, c in enumerate(classes)}
        for i, pr in enumerate(preds):
            if pr in ci:
                p[i, ci[pr]] = 1.0
    return align_cols(p, classes)


def stacker_probs(samples):
    """work2 AAR 스태커 확률. predict_aar(work2/script.py)의 **확률 경로**를 그대로 재현
    (단 model_dir 만 STACKER_DIR 로 파라미터화). AAR.ACTIONS 순 = ACTIONS 순."""
    config = AAR.load_config(os.path.join(STACKER_DIR, "aar_config.json"))
    if not config.get("enabled"):
        raise ValueError("AAR config disabled")
    artifact = joblib.load(os.path.join(STACKER_DIR, str(config.get("model_file", "aar_models.joblib"))))
    texts = [AAR.record_to_text(r) for r in samples]
    prompt_texts = [AAR.record_to_prompt_text(r) for r in samples]
    views = AAR.aar_views(samples, texts, prompt_texts)
    comp = {}
    for c in config.get("components", []):
        name, kind, view = str(c.get("name")), str(c.get("kind")), str(c.get("view"))
        if kind == "transition":
            comp[name] = AAR.aar_transition_predict_proba(artifact["transition"], samples)
        else:
            model = artifact.get("components", {}).get(name)
            if model is None:
                raise ValueError(f"AAR component missing: {name}")
            comp[name] = AAR.predict_proba_aligned(model, views[view])
    if config.get("use_stacker"):
        names = [str(x) for x in config.get("stacker_components", [])]
        matrix = np.hstack([comp[n] for n in names]).astype(np.float32)
        probas = AAR.predict_proba_aligned(artifact["stacker"], matrix)
    else:
        probas = AAR.weighted_average(
            [(comp[str(c["name"])], float(c.get("weight", 0.0))) for c in config["components"]])
    if config.get("use_bias"):
        probas = AAR.aar_apply_bias(probas, config.get("class_bias", [0.0] * len(ACTIONS)))
    return np.asarray(probas, dtype=np.float64)   # 이미 AAR.ACTIONS(=ACTIONS) 순


def _one_encoder_probs(samples, enc_dir):
    """파인튜닝 인코더 1개 확률 = softmax(logits). 오프라인 로컬 로드 + 배치추론. ACTIONS 순 반환."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(enc_dir)
    model = AutoModelForSequenceClassification.from_pretrained(enc_dir, torch_dtype="auto")
    model = model.half().to(device) if device == "cuda" else model.float()
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    enc_labels = [id2label[i] for i in range(len(id2label))]

    texts = [serialize(s) for s in samples]
    out = np.zeros((len(texts), len(enc_labels)), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(texts), BATCH):
            enc = tok(texts[i:i + BATCH], truncation=True, max_length=MAX_LEN,
                      padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits.float().cpu().numpy()
            out[i:i + BATCH] = softmax(logits, temp=1.0)
    del model
    if device == "cuda":
        torch.cuda.empty_cache()          # 다음 인코더 로드 전 VRAM 반납 (T4 16GB에 base 2개 동시 방지)
    return align_cols(out, enc_labels)


def load_calib():
    """인코더 캘리브레이션(model/calib.json, 세션 B calib_v1): p ← softmax(log(p)/T + bias).
    파일 없으면 None = 기존 동작 그대로. holdout 내부 5-fold 정직 검증 +0.005 (ECE 0.080→0.024)."""
    path = os.path.join(MODEL, "calib.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        c = json.load(f)
    t = float(c.get("temperature", 1.0))
    bias_map = c.get("class_bias") or {}
    bias = np.array([float(bias_map.get(a, 0.0)) for a in ACTIONS], dtype=np.float64)
    return t, bias


def encoder_probs(samples):
    """인코더 블록 확률 = 성분 인코더들의 uniform 평균 (1개면 그대로) + 옵션 캘리브레이션."""
    dirs = encoder_dirs()
    acc = None
    for d in dirs:
        p = _one_encoder_probs(samples, d)
        acc = p if acc is None else acc + p
        print(f"      encoder[{os.path.basename(os.path.normpath(d))}] 완료")
    p = acc / len(dirs)
    calib = load_calib()
    if calib is not None:
        t, bias = calib
        p = softmax(np.log(np.clip(p, 1e-12, None)) / t + bias.reshape(1, -1))
        print(f"      encoder calib 적용 (T={t:g})")
    return p


def _validate_weights(parts, source):
    if len(parts) != 3:
        raise ValueError(f"{source} 는 'lin,stk,enc' 3개 가중치여야 함: {parts!r}")
    try:
        w = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        raise ValueError(f"{source} 숫자 파싱 실패: {parts!r}")
    if any(not math.isfinite(v) for v in w) or any(v < 0 for v in w) or sum(w) <= 0:
        raise ValueError(f"{source} 는 유한 숫자·음수 불가·합>0 이어야 함: {w}")
    return w


def parse_weights():
    """가중치 우선순위: ENS_WEIGHTS env > model/weights.json > None(uniform).
    순서 = blend 순서(lin,stk,enc-block). 미설정이면 main 의 기존 uniform 식이 그대로 실행된다.
    weights.json 은 [1,1,1.5] 또는 {"weights":[1,1,1.5]} 형식을 허용한다."""
    raw = os.environ.get("ENS_WEIGHTS", "").strip()
    if raw:
        return _validate_weights([p.strip() for p in raw.split(",")], "ENS_WEIGHTS")

    path = os.path.join(MODEL, "weights.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("weights")
    if not isinstance(data, list):
        raise ValueError(f"weights.json 형식은 배열 또는 weights 배열 객체여야 함: {path}")
    return _validate_weights(data, "weights.json")


def _last_action(sample):
    hist = sample.get("history") or []
    if not isinstance(hist, list):
        return "none"
    for item in reversed(hist):
        if isinstance(item, dict) and item.get("role") == "assistant_action":
            return str(item.get("name") or item.get("action") or item.get("tool") or "unknown")
    return "none"


def _bucket_history(sample):
    hist = sample.get("history") or []
    return "history_empty" if not isinstance(hist, list) or len(hist) == 0 else "history_present"


def _bucket_last_action_family4(sample):
    action = _last_action(sample)
    if action == "none":
        return "none"
    if action in {"read_file", "grep_search", "list_directory", "glob_pattern"}:
        return "explore"
    if action in {"edit_file", "write_file", "apply_patch", "run_bash", "run_tests", "lint_or_typecheck"}:
        return "mutate_validate"
    return "coordinate"


def _turn_index(sample):
    meta = sample.get("session_meta") or {}
    value = meta.get("turn_index")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return int(value)
    return -1


def _bucket_turn_index_bin(sample):
    turn = _turn_index(sample)
    if turn <= 1:
        return "turn_01"
    if turn <= 3:
        return "turn_02_03"
    if turn <= 6:
        return "turn_04_06"
    return "turn_07_plus"


def bucket_key(sample, scheme):
    if scheme == "history_presence":
        return _bucket_history(sample)
    if scheme == "last_action_family4":
        return _bucket_last_action_family4(sample)
    if scheme == "turn_index_bin":
        return _bucket_turn_index_bin(sample)
    if scheme == "history_x_last_family4":
        return _bucket_history(sample) + "|" + _bucket_last_action_family4(sample)
    raise ValueError(f"bucket_weights.json unknown scheme: {scheme!r}")


def parse_bucket_weights(fallback_weights):
    """model/bucket_weights.json schema:
    {
      "scheme": "history_presence",
      "default_weights": [1,1,1.5],
      "buckets": {"history_empty": [0.75,0.5,2], "history_present": [0.75,1,0.75]}
    }
    Missing file = None, so the legacy blend path is byte-identical.
    """
    raw = os.environ.get("ENS_BUCKET_WEIGHTS", "").strip()
    path = raw or os.path.join(MODEL, "bucket_weights.json")
    if not os.path.exists(path):
        if raw:
            raise FileNotFoundError(f"ENS_BUCKET_WEIGHTS 파일 없음: {path}")
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"bucket_weights.json 형식은 객체여야 함: {path}")
    scheme = str(data.get("scheme") or "")
    if not scheme:
        raise ValueError(f"bucket_weights.json scheme 누락: {path}")
    raw_buckets = data.get("buckets")
    if not isinstance(raw_buckets, dict) or not raw_buckets:
        raise ValueError(f"bucket_weights.json buckets 객체 누락: {path}")
    default_raw = data.get("default_weights", fallback_weights or (1.0, 1.0, 1.0))
    default_weights = _validate_weights(default_raw, "bucket_weights.default_weights")
    buckets = {
        str(name): _validate_weights(value, f"bucket_weights.buckets[{name!r}]")
        for name, value in raw_buckets.items()
    }
    return {"path": path, "scheme": scheme, "default": default_weights, "buckets": buckets}


def bucket_weighted_blend(samples, lin, stk, enc, cfg):
    out = np.empty_like(lin, dtype=np.float64)
    counts = Counter()
    missing = Counter()
    for i, sample in enumerate(samples):
        key = bucket_key(sample, cfg["scheme"])
        weights = cfg["buckets"].get(key)
        if weights is None:
            weights = cfg["default"]
            missing[key] += 1
        counts[key] += 1
        wl, wstk, we = weights
        out[i] = (wl * lin[i] + wstk * stk[i] + we * enc[i]) / (wl + wstk + we)
    print(f"[BLEND] bucket_weights scheme={cfg['scheme']} file={cfg['path']}")
    print(f"        bucket counts: {dict(counts.most_common())}")
    if missing:
        print(f"        default fallback buckets: {dict(missing.most_common())}")
    return out


def load_test():
    samples = []
    with open(os.path.join(DATA, "test.jsonl"), encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


_SIB_STEP_RE = re.compile(r"-step_(\d+)$")


def sibling_label_recovery(samples, preds):
    """세션 형제 행 라벨 복원 (D-008) — step k의 정답은 같은 세션 step k+g 행
    history의 뒤에서 g번째 assistant_action.name (train 실측 gap1~6 231,664쌍 100.00%).

    test가 세션당 1행이면 형제가 없어 아무것도 하지 않는다(모델 예측 유지) — 하방 위험 0.
    history가 잘려 g번째가 없거나 name이 14클래스 밖이어도 폴백. ENS_RECOVER=0으로 끌 수 있다.
    """
    if os.environ.get("ENS_RECOVER", "1").strip().lower() in ("0", "false", "no"):
        print("[RECOVER] off (ENS_RECOVER=0)")
        return preds
    by_sess = {}
    for i, s in enumerate(samples):
        sid = str(s.get("id", ""))
        m = _SIB_STEP_RE.search(sid)
        if m:
            by_sess.setdefault(sid[:m.start()], []).append((int(m.group(1)), i))
    n_over = n_diff = 0
    action_set = set(ACTIONS)
    for rows in by_sess.values():
        rows.sort()
        for a in range(len(rows)):
            sa, ia = rows[a]
            for b in range(a + 1, len(rows)):      # 가까운 형제부터 (gap↑ = history 잘림 위험↑)
                g = rows[b][0] - sa
                acts = [t.get("name") for t in (samples[rows[b][1]].get("history") or [])
                        if isinstance(t, dict) and t.get("role") == "assistant_action"]
                if g <= len(acts) and acts[-g] in action_set:
                    if preds[ia] != acts[-g]:
                        n_diff += 1
                    preds[ia] = acts[-g]
                    n_over += 1
                    break
    print(f"[RECOVER] sibling overrides: {n_over}/{len(samples)} (model 예측과 다른 값 {n_diff})")
    return preds


def main():
    t0 = time.time()
    samples = load_test()
    print(f"[1/5] test rows={len(samples)}")

    print("[2/5] linear…");  lin = linear_probs(samples)
    print("[3/5] stacker…"); stk = stacker_probs(samples)
    print("[4/5] encoder…"); enc = encoder_probs(samples)
    for name, p in [("linear", lin), ("stacker", stk), ("encoder", enc)]:
        assert p.shape == (len(samples), len(ACTIONS)), f"{name} shape {p.shape}"

    weights = parse_weights()
    bucket_weights = parse_bucket_weights(weights)
    if bucket_weights is not None:
        blend = bucket_weighted_blend(samples, lin, stk, enc, bucket_weights)
    elif weights is None:
        blend = (lin + stk + enc) / 3.0                   # uniform 평균 (미설정 = 기존과 byte-identical)
    else:
        wl, wstk, we = weights
        blend = (wl * lin + wstk * stk + we * enc) / (wl + wstk + we)
        print(f"[BLEND] weighted lin={wl:g} stk={wstk:g} enc={we:g} "
              f"(정규화합={wl + wstk + we:g})")
    preds = [str(p) for p in np.array(ACTIONS)[blend.argmax(1)]]
    preds = sibling_label_recovery(samples, preds)
    id2pred = {s["id"]: p for s, p in zip(samples, preds)}

    print("[5/5] submission.csv (sample_submission id 순서 유지)…")
    os.makedirs(OUT, exist_ok=True)
    n_missing = 0
    with open(os.path.join(DATA, "sample_submission.csv"), newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or fields[:2] != ["id", "action"]:
        raise ValueError(f"sample_submission 컬럼 (id,action) 아님: {fields}")
    for row in rows:
        p = id2pred.get(row["id"])
        if p is None:
            n_missing += 1
        else:
            row["action"] = p
    if n_missing:
        print(f"  경고: 예측 없는 id {n_missing}건")
    with open(os.path.join(OUT, "submission.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    bad = sorted(set(id2pred.values()) - set(ACTIONS))
    if bad:
        raise ValueError(f"잘못된 라벨: {bad}")
    print(f"[DONE] {OUT}/submission.csv rows={len(rows)} elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
