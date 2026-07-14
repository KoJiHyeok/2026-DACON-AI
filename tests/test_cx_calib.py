import json

import numpy as np

from scripts.cx_calib.fit_calib import (
    Calibration,
    apply_calibration,
    build_group_folds,
    load_calibration_json,
    session_id,
    write_calibration_json,
)


def test_group_folds_keep_complete_sessions_together():
    ids = np.asarray([f"session_{session:02d}-step_{step:02d}" for session in range(20) for step in range(3)])
    groups = np.asarray([session_id(sample_id) for sample_id in ids])
    seen = np.zeros(len(ids), dtype=np.int8)

    for train_idx, valid_idx in build_group_folds(ids, n_splits=5):
        assert set(groups[train_idx]).isdisjoint(set(groups[valid_idx]))
        seen[valid_idx] += 1

    np.testing.assert_array_equal(seen, np.ones(len(ids), dtype=np.int8))


def test_apply_calibration_matches_submit_formula():
    probs = np.asarray([[0.7, 0.2, 0.1], [1e-20, 0.4, 0.6]], dtype=np.float64)
    temperature = 1.37
    bias = np.asarray([0.2, -0.1, -0.1], dtype=np.float64)
    logits = np.log(np.clip(probs, 1e-12, None)) / temperature + bias.reshape(1, -1)
    logits -= logits.max(axis=1, keepdims=True)
    expected = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)

    actual = apply_calibration(probs, temperature, bias)

    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual.sum(axis=1), 1.0, rtol=0, atol=1e-14)


def test_calibration_json_round_trip_uses_submit_keys(tmp_path):
    actions = ["apply_patch", "ask_user", "run_tests"]
    calibration = Calibration(temperature=0.875, class_bias=np.asarray([0.25, -0.10, -0.15]))
    path = tmp_path / "calib.json"

    write_calibration_json(path, calibration, actions)
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = load_calibration_json(path, actions)

    assert set(payload) == {"temperature", "class_bias"}
    assert list(payload["class_bias"]) == actions
    assert loaded.temperature == calibration.temperature
    np.testing.assert_allclose(loaded.class_bias, calibration.class_bias)
