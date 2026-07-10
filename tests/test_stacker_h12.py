from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from scripts.stacker_h12.common import (
    ACTIONS,
    align_bundle,
    align_probs,
    load_hist12_oof,
    load_legacy_oof,
)
from scripts.stacker_h12.features import (
    build_teammate_matrix,
    numeric_feature_names,
    teammate_numeric_features,
)
from scripts.stacker_h12.train import train_stacker


def _one_hot(labels: list[str], actions: tuple[str, ...]) -> np.ndarray:
    out = np.zeros((len(labels), len(actions)), dtype=np.float64)
    for row, label in enumerate(labels):
        out[row, actions.index(label)] = 1.0
    return out


def _write_fold(
    directory: Path,
    fold: int,
    ids: list[str],
    labels: list[str],
    *,
    actions: tuple[str, ...] = tuple(reversed(ACTIONS)),
    probs: np.ndarray | None = None,
) -> None:
    if probs is None:
        probs = _one_hot(labels, actions)
    np.savez(
        directory / f"oof_fold{fold}.npz",
        ids=np.asarray(ids, dtype=object),
        probs=probs,
        y_true=np.asarray(labels, dtype=object),
        actions=np.asarray(actions, dtype=object),
        fold=np.full(len(ids), fold, dtype=np.int64),
    )


def _write_map(directory: Path, rows: list[tuple[str, object]]) -> None:
    with (directory / "fold_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "fold"])
        writer.writerows(rows)


def _valid_hist12(directory: Path) -> tuple[list[str], list[str]]:
    directory.mkdir()
    ids = ["sess_a-step_00", "sess_b-step_00"]
    labels = [ACTIONS[0], ACTIONS[1]]
    _write_fold(directory, 0, [ids[0]], [labels[0]])
    _write_fold(directory, 1, [ids[1]], [labels[1]])
    _write_map(directory, [(ids[1], 1), (ids[0], 0)])
    return ids, labels


def test_hist12_loader_aligns_actions_and_reference_order(tmp_path: Path) -> None:
    ids, labels = _valid_hist12(tmp_path / "oof")
    bundle = load_hist12_oof(
        tmp_path / "oof",
        expected_folds=(0, 1),
        reference_ids=[ids[1], ids[0]],
        reference_y=[labels[1], labels[0]],
    )
    assert bundle.ids.tolist() == [ids[1], ids[0]]
    assert bundle.actions == ACTIONS
    assert bundle.probs.argmax(axis=1).tolist() == [1, 0]
    assert bundle.folds.tolist() == [1, 0]


def test_align_probs_rejects_missing_and_duplicate_actions() -> None:
    probs = np.full((1, len(ACTIONS)), 1.0 / len(ACTIONS))
    with pytest.raises(ValueError, match="action set mismatch"):
        align_probs(probs[:, :-1], ACTIONS[:-1], ACTIONS)
    duplicate = list(ACTIONS)
    duplicate[-1] = duplicate[0]
    with pytest.raises(ValueError, match="duplicate"):
        align_probs(probs, duplicate, ACTIONS)


@pytest.mark.parametrize("kind", ["nan", "negative", "row_sum"])
def test_hist12_loader_rejects_invalid_probabilities(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "oof"
    ids, labels = _valid_hist12(directory)
    probs = _one_hot([labels[0]], tuple(reversed(ACTIONS)))
    if kind == "nan":
        probs[0, 0] = np.nan
    elif kind == "negative":
        probs[0, 0] = -0.1
    else:
        probs[0] *= 0.5
    _write_fold(directory, 0, [ids[0]], [labels[0]], probs=probs)
    with pytest.raises(ValueError):
        load_hist12_oof(directory, expected_folds=(0, 1))


def test_hist12_loader_rejects_duplicate_ids_across_folds(tmp_path: Path) -> None:
    directory = tmp_path / "oof"
    directory.mkdir()
    sample_id = "sess_a-step_00"
    _write_fold(directory, 0, [sample_id], [ACTIONS[0]])
    _write_fold(directory, 1, [sample_id], [ACTIONS[0]])
    _write_map(directory, [(sample_id, 0)])
    with pytest.raises(ValueError, match="duplicate IDs across folds"):
        load_hist12_oof(directory, expected_folds=(0, 1))


@pytest.mark.parametrize("kind", ["missing", "extra", "mismatch"])
def test_hist12_loader_rejects_fold_map_mismatch(tmp_path: Path, kind: str) -> None:
    directory = tmp_path / "oof"
    ids, _ = _valid_hist12(directory)
    if kind == "missing":
        rows = [(ids[0], 0)]
    elif kind == "extra":
        rows = [(ids[0], 0), (ids[1], 1), ("sess_extra-step_00", 0)]
    else:
        rows = [(ids[0], 1), (ids[1], 1)]
    _write_map(directory, rows)
    with pytest.raises(ValueError):
        load_hist12_oof(directory, expected_folds=(0, 1))


def test_hist12_loader_rejects_session_split_across_folds(tmp_path: Path) -> None:
    directory = tmp_path / "oof"
    directory.mkdir()
    first = "sess_shared-step_00"
    second = "sess_shared-step_01"
    _write_fold(directory, 0, [first], [ACTIONS[0]])
    _write_fold(directory, 1, [second], [ACTIONS[1]])
    _write_map(directory, [(first, 0), (second, 1)])
    with pytest.raises(ValueError, match="split across folds"):
        load_hist12_oof(directory, expected_folds=(0, 1))


def test_hist12_loader_rejects_reference_label_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "oof"
    ids, labels = _valid_hist12(directory)
    with pytest.raises(ValueError, match="y_true mismatch"):
        load_hist12_oof(
            directory,
            expected_folds=(0, 1),
            reference_ids=ids,
            reference_y=[labels[1], labels[0]],
        )


def test_legacy_loader_aligns_ids_and_actions_without_fold_contract(tmp_path: Path) -> None:
    directory = tmp_path / "legacy"
    directory.mkdir()
    ids = ["sess_b-step_00", "sess_a-step_00"]
    labels = [ACTIONS[1], ACTIONS[0]]
    source_actions = tuple(reversed(ACTIONS))
    (directory / "row_ids.json").write_text(json.dumps(ids), encoding="utf-8")
    (directory / "classes.json").write_text(json.dumps(source_actions), encoding="utf-8")
    (directory / "y_true.json").write_text(json.dumps(labels), encoding="utf-8")
    np.save(directory / "linear_probs.npy", _one_hot(labels, source_actions))

    bundle, components = load_legacy_oof(directory, components=("linear",))
    aligned = align_bundle(bundle, list(reversed(ids)), reference_y=list(reversed(labels)))
    assert bundle.folds is None
    assert aligned.ids.tolist() == list(reversed(ids))
    assert components["linear"].argmax(axis=1).tolist() == [1, 0]


def _record(sample_id: str, action: str, index: int) -> dict[str, object]:
    return {
        "id": sample_id,
        "current_prompt": f"read src/file_{index}.py and run pytest",
        "history": [
            {"role": "user", "content": "check"},
            {
                "role": "assistant_action",
                "name": action,
                "args": {"path": f"src/file_{index}.py"},
                "result_summary": "ok",
            },
        ],
        "session_meta": {
            "workspace": {
                "open_files": [f"src/file_{index}.py"],
                "git_dirty": bool(index % 2),
                "loc": index,
                "last_ci_status": "pass",
                "language_mix": {"python": 1.0},
            }
        },
    }


def test_teammate_feature_order_and_structured_matrix_are_deterministic() -> None:
    baseline = np.full((2, len(ACTIONS)), 0.01 / (len(ACTIONS) - 1))
    e5 = baseline.copy()
    baseline[0, 0] = 0.99
    baseline[1, 1] = 0.99
    e5[0, 1] = 0.99
    e5[0, 0] = 0.01 / (len(ACTIONS) - 1)
    e5[1, 2] = 0.99
    e5[1, 1] = 0.01 / (len(ACTIONS) - 1)
    records = [_record("a-step_00", ACTIONS[0], 0), _record("b-step_00", ACTIONS[1], 1)]

    numeric = teammate_numeric_features(baseline, e5)
    assert numeric.shape == (2, 34)
    assert len(numeric_feature_names()) == 34
    np.testing.assert_allclose(numeric[:, :14], baseline)
    np.testing.assert_allclose(numeric[:, 14:28], e5)
    np.testing.assert_allclose(numeric[:, 28], baseline.max(axis=1))
    np.testing.assert_allclose(numeric[:, 29], e5.max(axis=1))

    first, vectorizer = build_teammate_matrix(
        baseline,
        e5,
        records,
        fit_vectorizer=True,
    )
    second, _ = build_teammate_matrix(
        baseline,
        e5,
        records,
        vectorizer=vectorizer,
    )
    assert first.shape[1] > 34
    np.testing.assert_allclose(first.toarray(), second.toarray())


def test_training_is_deterministic_and_never_promotion_eligible(tmp_path: Path) -> None:
    ids: list[str] = []
    labels: list[str] = []
    folds: list[int] = []
    records: list[dict[str, object]] = []
    baseline_rows: list[np.ndarray] = []
    e5_rows: list[np.ndarray] = []
    for fold in range(3):
        for class_index, action in enumerate(ACTIONS):
            sample_id = f"sess_{fold}_{class_index}-step_00"
            ids.append(sample_id)
            labels.append(action)
            folds.append(fold)
            records.append(_record(sample_id, action, fold * len(ACTIONS) + class_index))
            baseline = np.full(len(ACTIONS), 0.2 / (len(ACTIONS) - 1))
            encoder = np.full(len(ACTIONS), 0.3 / (len(ACTIONS) - 1))
            baseline[class_index] = 0.8
            encoder[class_index] = 0.7
            baseline_rows.append(baseline)
            e5_rows.append(encoder)

    kwargs = {
        "ids": np.asarray(ids, dtype=object),
        "y_true": np.asarray(labels, dtype=object),
        "folds": np.asarray(folds, dtype=np.int64),
        "baseline": np.asarray(baseline_rows),
        "e5": np.asarray(e5_rows),
        "records": records,
        "baseline_origin": "legacy_linear_proxy",
        "c_value": 0.1,
        "max_iter": 300,
        "seed": 42,
    }
    first = train_stacker(output_dir=tmp_path / "run1", **kwargs)
    second = train_stacker(output_dir=tmp_path / "run2", **kwargs)

    assert first["promotion_eligible"] is False
    assert first["teammate_parity"] is False
    assert first["final_fit_scored_in_sample"] is False
    assert first["diagnostic_meta_cv"]["name"] == "diagnostic_meta_cv"
    assert first["diagnostic_meta_cv"]["promotion_eligible"] is False
    assert (
        first["diagnostic_meta_cv"]["macro_f1"]
        == second["diagnostic_meta_cv"]["macro_f1"]
    )
    payload = joblib.load(tmp_path / "run1" / "stacker_h12.joblib")
    assert set(payload) == {"dict_vectorizer", "stacker", "actions", "metadata"}
    assert payload["metadata"]["teammate_parity"] is False


def test_training_rejects_misaligned_records_and_split_sessions(tmp_path: Path) -> None:
    labels = np.asarray([ACTIONS[0], ACTIONS[1]], dtype=object)
    probabilities = _one_hot(labels.tolist(), ACTIONS)
    base = {
        "ids": np.asarray(["sess_a-step_00", "sess_b-step_00"], dtype=object),
        "y_true": labels,
        "folds": np.asarray([0, 1], dtype=np.int64),
        "baseline": probabilities,
        "e5": probabilities,
        "records": [
            _record("sess_b-step_00", ACTIONS[1], 1),
            _record("sess_a-step_00", ACTIONS[0], 0),
        ],
        "baseline_origin": "legacy_linear_proxy",
        "output_dir": tmp_path / "misaligned",
        "validate_only": True,
    }
    with pytest.raises(ValueError, match="exactly aligned"):
        train_stacker(**base)

    base["ids"] = np.asarray(["sess_shared-step_00", "sess_shared-step_01"], dtype=object)
    base["records"] = [
        _record("sess_shared-step_00", ACTIONS[0], 0),
        _record("sess_shared-step_01", ACTIONS[1], 1),
    ]
    with pytest.raises(ValueError, match="split across folds"):
        train_stacker(**base)


def test_alpha_origin_is_a_claim_not_verified_parity(tmp_path: Path) -> None:
    ids = np.asarray(["sess_a-step_00", "sess_b-step_00"], dtype=object)
    labels = np.asarray([ACTIONS[0], ACTIONS[1]], dtype=object)
    probabilities = _one_hot(labels.tolist(), ACTIONS)
    manifest = train_stacker(
        ids=ids,
        y_true=labels,
        folds=np.asarray([0, 1], dtype=np.int64),
        baseline=probabilities,
        e5=probabilities,
        records=[
            _record(ids[0], ACTIONS[0], 0),
            _record(ids[1], ACTIONS[1], 1),
        ],
        baseline_origin="alpha09_sparse_oof",
        output_dir=tmp_path / "claim",
        validate_only=True,
    )
    assert manifest["teammate_baseline_parity_claimed"] is True
    assert manifest["teammate_baseline_parity_verified"] is False
    assert manifest["teammate_parity"] is False
    assert len(manifest["feature_source_sha256"]) == 64
