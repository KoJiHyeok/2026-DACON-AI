from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

from scripts.aar_rebuild.train_aar import train_aar
from submit import aar_infer


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


def test_rebuild_writes_consumer_schema_and_is_deterministic(tmp_path: Path) -> None:
    records, labels = _records()
    first = train_aar(records, labels, tmp_path / "one", max_iter=3)
    second = train_aar(records, labels, tmp_path / "two", max_iter=3)
    assert first["folds"] == 3
    assert len(first["fold_macro_f1"]) == 3
    assert first["oof_macro_f1"] == second["oof_macro_f1"]
    config = json.loads((tmp_path / "one" / "aar_config.json").read_text())
    artifact = joblib.load(tmp_path / "one" / "aar_models.joblib")
    assert config["use_stacker"] is True
    assert set(artifact) >= {"components", "stacker", "actions"}
    assert set(artifact["components"]) == {"sgd_full", "sgd_prompt_context", "sgd_history", "sgd_action"}


def test_rebuild_artifact_predicts_probability_rows(tmp_path: Path) -> None:
    records, labels = _records()
    train_aar(records, labels, tmp_path, max_iter=3)
    artifact = joblib.load(tmp_path / "aar_models.joblib")
    views = {"full": ["x"] * 2, "prompt_context": ["x"] * 2,
             "history": ["x"] * 2, "action": ["x"] * 2}
    parts = []
    for name in ("sgd_full", "sgd_prompt_context", "sgd_history", "sgd_action"):
        view = name.removeprefix("sgd_")
        parts.append(artifact["components"][name].predict_proba(views[view]))
    pred = artifact["stacker"].predict_proba(np.hstack(parts))
    assert pred.shape == (2, 14)
    np.testing.assert_allclose(pred.sum(axis=1), 1.0, atol=1e-6)


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
