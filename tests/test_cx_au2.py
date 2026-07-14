from __future__ import annotations

import joblib
import numpy as np
import pytest

from scripts.cx_au2 import common


def _sample(sample_id: str, prompt: str) -> dict:
    return {
        "id": sample_id,
        "current_prompt": prompt,
        "history": [],
        "session_meta": {"turn_index": 0, "workspace": {}},
    }


def test_holdout_leak_assert_rejects_overlap() -> None:
    with pytest.raises(AssertionError, match="holdout id leaked"):
        common.assert_holdout_excluded(
            ["sess_au_train-step_00", "sess_au_leak-step_01"],
            ["sess_au_leak-step_01"],
        )


def test_candidate_artifact_roundtrip_through_au_predict_proba(tmp_path) -> None:
    samples = [
        _sample("sess_au_a-step_00", "read package json"),
        _sample("sess_au_b-step_00", "open source file"),
        _sample("sess_au_c-step_00", "run unit tests"),
        _sample("sess_au_d-step_00", "execute pytest"),
    ]
    labels = np.asarray(["read_file", "read_file", "run_tests", "run_tests"], dtype=object)
    variant = common.Variant("test_word_char", "word_char", 0.5)
    artifact = common.fit_artifact(samples, labels, variant)
    model_path = tmp_path / "model.pkl"
    common.dump_artifact(model_path, artifact)

    loaded = joblib.load(model_path)
    au_route = common.load_au_route()
    probs, classes = au_route.predict_proba(loaded, samples[:2])

    assert set(loaded) == {"union", "clf"}
    assert probs.shape == (2, 2)
    assert set(classes) == {"read_file", "run_tests"}
    np.testing.assert_allclose(probs.sum(axis=1), 1.0)


def test_group_leak_assert_rejects_shared_session() -> None:
    with pytest.raises(AssertionError, match="group leakage"):
        common.assert_group_disjoint(["sess_au_a", "sess_au_b"], ["sess_au_b"], "fold 0")
