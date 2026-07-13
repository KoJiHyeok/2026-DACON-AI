from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts.aar_speed import fast_aar
from submit import aar_infer


DATA_PATH = Path("C:/dev/2026-AI-DACON/data/train.jsonl")
MODEL_DIR = Path("C:/dev/2026-AI-DACON/submit/model/stacker")


@pytest.fixture(scope="module")
def real_inputs():
    if not DATA_PATH.exists() or not (MODEL_DIR / "aar_models.joblib").exists():
        pytest.skip("read-only AAR speed-gate inputs are not available")
    records = fast_aar.sample_evenly(DATA_PATH, 300)
    config = json.loads((MODEL_DIR / "aar_config.json").read_text(encoding="utf-8"))
    artifact = joblib.load(MODEL_DIR / config["model_file"])
    texts = [aar_infer.record_to_text(record) for record in records]
    prompts = [aar_infer.record_to_prompt_text(record) for record in records]
    return records, texts, prompts, config, artifact


def test_real_300_row_probability_and_argmax_equivalence(real_inputs, monkeypatch) -> None:
    records, texts, prompts, config, artifact = real_inputs
    reference = fast_aar.reference_predict_proba(
        records, texts, prompts, config, artifact
    )
    actual = fast_aar.fast_predict_proba(records, texts, prompts, config, artifact)

    assert actual.shape == (300, len(aar_infer.ACTIONS))
    assert actual.dtype == reference.dtype
    assert float(np.max(np.abs(reference - actual))) <= 1e-9
    assert np.array_equal(reference.argmax(axis=1), actual.argmax(axis=1))

    # The public label API must match the unmodified vendor predict_aar path.
    monkeypatch.setattr(aar_infer.joblib, "load", lambda _path: artifact)
    vendor_labels = aar_infer.predict_aar(records, texts, prompts, config)
    assert fast_aar.predict_aar(records, texts, prompts, config, artifact) == vendor_labels


def test_probability_rows_preserve_input_order(real_inputs) -> None:
    records, texts, prompts, config, artifact = real_inputs
    count = 17
    forward = fast_aar.fast_predict_proba(
        records[:count], texts[:count], prompts[:count], config, artifact
    )
    reverse = fast_aar.fast_predict_proba(
        list(reversed(records[:count])),
        list(reversed(texts[:count])),
        list(reversed(prompts[:count])),
        config,
        artifact,
    )
    np.testing.assert_array_equal(forward, reverse[::-1])


def test_length_contract_rejects_misaligned_inputs(real_inputs) -> None:
    records, texts, prompts, config, artifact = real_inputs
    with pytest.raises(ValueError, match="equal length"):
        fast_aar.fast_predict_proba(records[:2], texts[:1], prompts[:2], config, artifact)

