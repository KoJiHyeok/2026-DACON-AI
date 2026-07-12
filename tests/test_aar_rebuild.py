from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from scripts.aar_rebuild.train_aar import (
    COMPONENT_SPECS,
    STACKER_COMPONENTS,
    TRANSITION_WEIGHTS,
    build_transition_spec,
    train_aar,
)
from submit import aar_infer


AAR_CONFIG_PATH = Path("C:/dev/2026-AI-DACON/submit/model/stacker/aar_config.json")


def _records() -> tuple[list[dict], np.ndarray]:
    actions = [
        "read_file", "grep_search", "list_directory", "glob_pattern",
        "edit_file", "write_file", "apply_patch", "run_bash",
        "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
        "web_search", "respond_only",
    ]
    rows = []
    labels = []
    for group in range(3):
        for idx, action in enumerate(actions):
            rows.append({"id": f"session_{group}-step_{idx:02d}",
                         "current_prompt": f"{action} file_{idx}.py",
                         "history": [], "session_meta": {"turn_index": idx, "workspace": {}}})
            labels.append(action)
    return rows, np.asarray(labels, dtype=object)


def test_component_spec_matches_surviving_aar_config() -> None:
    """The recovered aar_config.json is the ground truth: 3 named text SGD
    components plus a transition_prior, in this exact stacker order."""
    assert AAR_CONFIG_PATH.exists(), "surviving aar_config.json must exist on disk"
    config = json.loads(AAR_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["use_stacker"] is True
    assert config["stacker_components"] == STACKER_COMPONENTS
    real_names = {c["name"] for c in config["components"]}
    assert real_names == set(COMPONENT_SPECS) | {"transition_prior"}
    real_views = {c["name"]: c["view"] for c in config["components"]}
    for name, spec in COMPONENT_SPECS.items():
        assert real_views[name] == spec["view"]


def test_rebuild_writes_consumer_schema_and_is_deterministic(tmp_path: Path) -> None:
    records, labels = _records()
    first = train_aar(records, labels, tmp_path / "one", max_iter=3)
    second = train_aar(records, labels, tmp_path / "two", max_iter=3)
    assert first["folds"] == 3
    assert len(first["fold_macro_f1"]) == 3
    assert first["oof_macro_f1"] == second["oof_macro_f1"]
    assert first["stacked_oof_macro_f1"] == second["stacked_oof_macro_f1"]

    config = json.loads((tmp_path / "one" / "aar_config.json").read_text())
    artifact = joblib.load(tmp_path / "one" / "aar_models.joblib")
    assert config["use_stacker"] is True
    assert config["stacker_components"] == STACKER_COMPONENTS
    assert set(artifact) >= {"components", "transition", "stacker", "actions"}
    assert set(artifact["components"]) == set(COMPONENT_SPECS)
    assert isinstance(artifact["transition"], dict)
    assert set(artifact["transition"]) >= {"actions", "global", "groups", "weights", "global_weight"}
    assert artifact["transition"]["weights"] == TRANSITION_WEIGHTS


def test_rebuild_artifact_predicts_probability_rows_via_real_consumer(tmp_path: Path, monkeypatch) -> None:
    """Loads the rebuilt artifact through the unmodified aar_infer.predict_aar
    consumer path -- not a hand-rolled reimplementation of the stacking math."""
    records, labels = _records()
    train_aar(records, labels, tmp_path / "model", max_iter=3)
    monkeypatch.chdir(tmp_path)
    config = json.loads((tmp_path / "model" / "aar_config.json").read_text())
    texts = [aar_infer.record_to_text(r) for r in records]
    prompts = [aar_infer.record_to_prompt_text(r) for r in records]
    preds = aar_infer.predict_aar(records, texts, prompts, config)
    assert len(preds) == len(records)
    assert set(preds) <= set(aar_infer.ACTIONS)


def test_transition_spec_produces_normalized_distributions() -> None:
    records, labels = _records()
    spec = build_transition_spec(records, labels)
    assert spec["weights"] == TRANSITION_WEIGHTS
    global_vec = np.asarray(spec["global"])
    assert global_vec.shape == (14,)
    np.testing.assert_allclose(global_vec.sum(), 1.0, atol=1e-9)
    for group_name, table in spec["groups"].items():
        for key, vec in table.items():
            np.testing.assert_allclose(np.sum(vec), 1.0, atol=1e-9)


def test_submit_aar_infer_load_smoke(tmp_path: Path, monkeypatch) -> None:
    records, labels = _records()
    train_aar(records, labels, tmp_path / "model", max_iter=3)
    monkeypatch.chdir(tmp_path)
    predictions = aar_infer.predict_aar(
        records[:2],
        [aar_infer.record_to_text(r) for r in records[:2]],
        [aar_infer.record_to_prompt_text(r) for r in records[:2]],
        json.loads((tmp_path / "model" / "aar_config.json").read_text()),
    )
    assert len(predictions) == 2
    assert set(predictions) <= set(aar_infer.ACTIONS)
