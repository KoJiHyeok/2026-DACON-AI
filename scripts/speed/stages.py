import os
import time
from pathlib import Path

import numpy as np

from .common import MODEL, env_int, load, mod, setup


def _timed(times, name, fn):
    started = time.perf_counter()
    value = fn()
    times[name] = time.perf_counter() - started
    return value


def _pipeline(m, samples):
    times = {}
    _timed(times, "feature_serialization", lambda: (
        m.F.build_dataframe(samples),
        [m.serialize(s, max_hist=m._encoder_max_hist(d))
         for d in m.encoder_dirs() for s in samples],
    ))
    lin = _timed(times, "linear", lambda: m.linear_probs(samples))
    stk = _timed(times, "aar_stacker", lambda: m.stacker_probs(samples))
    # enc 는 원본 encoder_probs() 를 그대로 한 번 호출한 결과다(가중 평균 enc_block_weights() +
    # calib load_calib() 포함) — 하네스가 uniform 평균으로 복제하던 것을 없애 원본과의 편차
    # 원천을 제거한다. 성분별 개별 시간은 _one_encoder_probs 를 감싸서 그 한 번의 실행 안에서
    # 잡는다(재실행 없음 → 이중 계측으로 인한 시간 왜곡 없음).
    real_one_encoder = m._one_encoder_probs

    def _timed_one_encoder(samples_, enc_dir):
        name = os.path.basename(os.path.normpath(enc_dir))
        started = time.perf_counter()
        value = real_one_encoder(samples_, enc_dir)
        times[name] = time.perf_counter() - started
        return value

    m._one_encoder_probs = _timed_one_encoder
    try:
        enc = _timed(times, "encoder_block", lambda: m.encoder_probs(samples))
    finally:
        m._one_encoder_probs = real_one_encoder

    def make_blend():
        weights = m.parse_weights()
        bucket = m.parse_bucket_weights(weights)
        if bucket is not None:
            return m.bucket_weighted_blend(samples, lin, stk, enc, bucket)
        if weights is None:
            return (lin + stk + enc) / 3.0
        wl, ws, we = weights
        return (wl * lin + ws * stk + we * enc) / (wl + ws + we)

    blend = _timed(times, "blend", make_blend)
    blend = _timed(times, "au_routing",
                   lambda: m.au_route_blend(samples, blend.copy()))

    def postprocess():
        preds = [str(p) for p in np.array(m.ACTIONS)[blend.argmax(1)]]
        return m.sibling_label_recovery(samples, preds)

    preds = _timed(times, "postprocess", postprocess)
    return preds, times


def run(n=None, device=None, model=None, repeats=None):
    n = env_int("SPEED_ROWS", 300) if n is None else int(n)
    device = os.environ.get("SPEED_DEVICE", "cpu").lower() if device is None else device.lower()
    repeats = env_int("SPEED_REPEATS", 3) if repeats is None else int(repeats)
    if n < 1 or repeats < 1:
        raise ValueError("rows and repeats must be positive")
    model = Path(model or os.environ.get("SPEED_MODEL_DIR", str(MODEL)))
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    m = mod()
    setup(m, model, device)
    started = time.perf_counter()
    samples = load(n=n)
    load_seconds = time.perf_counter() - started
    runs = []
    predictions = None
    for _ in range(repeats):
        predictions, times = _pipeline(m, samples)
        runs.append(times)
    names = sorted({k for item in runs for k in item})
    median = {k: float(np.median([item[k] for item in runs])) for k in names}
    return {
        "rows": len(samples), "device": device, "model_dir": str(model),
        "repeats": repeats, "data_load_seconds": load_seconds,
        "stage_seconds_median": median, "stage_seconds_runs": runs,
        "total_instrumented_seconds": float(sum(median.values())),
        "ids": [str(s.get("id", "")) for s in samples],
        "predictions": predictions,
    }
