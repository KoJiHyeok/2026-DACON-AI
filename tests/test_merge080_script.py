"""Regression tests for submit_candidates/merge080/script.py."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
MERGE080_DIR = ROOT / "submit_candidates" / "merge080"
MERGE080_SCRIPT = MERGE080_DIR / "script.py"
_MISSING = object()


class _InferenceMode:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _torch_stub() -> ModuleType:
    torch = ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    torch.cuda = SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
        get_device_name=lambda index: "stub-gpu",
    )
    torch.inference_mode = _InferenceMode
    return torch


def _transformers_stub() -> ModuleType:
    transformers = ModuleType("transformers")

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("AutoTokenizer should be monkeypatched in this test")

    class _AutoModelForSequenceClassification:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise AssertionError("AutoModelForSequenceClassification should be monkeypatched in this test")

    transformers.AutoTokenizer = _AutoTokenizer
    transformers.AutoModelForSequenceClassification = _AutoModelForSequenceClassification
    return transformers


def _joblib_stub() -> ModuleType:
    joblib = ModuleType("joblib")
    joblib.load = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("joblib.load should be monkeypatched in this test")
    )
    return joblib


def _scipy_stubs() -> dict[str, ModuleType]:
    scipy = ModuleType("scipy")
    sparse = ModuleType("scipy.sparse")
    sparse.csr_matrix = lambda value: value
    sparse.hstack = lambda values, format=None: values
    scipy.sparse = sparse
    return {"scipy": scipy, "scipy.sparse": sparse}


def load_merge080_script():
    """Load merge080/script.py with heavy dependencies stubbed and src imports isolated."""
    stub_modules = {
        "torch": _torch_stub(),
        "transformers": _transformers_stub(),
        "joblib": _joblib_stub(),
        **_scipy_stubs(),
    }
    saved_stubs = {name: sys.modules.get(name, _MISSING) for name in stub_modules}
    saved_src = {name: module for name, module in sys.modules.items() if name == "src" or name.startswith("src.")}
    module_name = "merge080_script_under_test"

    for name in saved_src:
        sys.modules.pop(name, None)
    for name, module in stub_modules.items():
        sys.modules[name] = module
    sys.modules.pop(module_name, None)
    sys.path.insert(0, str(MERGE080_DIR))
    try:
        spec = importlib.util.spec_from_file_location(module_name, MERGE080_SCRIPT)
        loaded_module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[module_name] = loaded_module
        spec.loader.exec_module(loaded_module)
    finally:
        sys.path.remove(str(MERGE080_DIR))
        sys.modules.pop(module_name, None)
        for name in list(sys.modules):
            if name == "src" or name.startswith("src."):
                sys.modules.pop(name, None)
        sys.modules.update(saved_src)
        for name, saved_module in saved_stubs.items():
            if saved_module is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved_module
    return loaded_module


@pytest.fixture()
def merge080_script(monkeypatch):
    monkeypatch.delenv("MBERT_MIX", raising=False)
    return load_merge080_script()


class _FakeTensor:
    def __init__(self, batch_len: int):
        self.batch_len = batch_len

    def to(self, device: str):
        return self


class _FakeLogits:
    def __init__(self, values: np.ndarray):
        self._values = values

    def float(self):
        return self

    def cpu(self):
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeTokenizer:
    def __call__(self, batch, **kwargs):
        return {"input_ids": _FakeTensor(len(batch))}


class _FakeSequenceClassifier:
    def __init__(self, logits: np.ndarray, id2label: dict[int, str]):
        self._logits = np.asarray(logits, dtype=np.float64)
        self._cursor = 0
        self.config = SimpleNamespace(id2label=id2label)

    def half(self):
        return self

    def float(self):
        return self

    def to(self, device: str):
        return self

    def eval(self):
        return None

    def parameters(self):
        return iter([SimpleNamespace(dtype="float32", device="cpu")])

    def __call__(self, **kwargs):
        batch_len = kwargs["input_ids"].batch_len
        start = self._cursor
        self._cursor += batch_len
        return SimpleNamespace(logits=_FakeLogits(self._logits[start : start + batch_len]))


def _normalized_rows(module, reverse: bool = False) -> np.ndarray:
    base = np.arange(1, len(module.ACTIONS) + 1, dtype=np.float64)
    if reverse:
        base = base[::-1]
    rows = np.vstack([base, np.roll(base, 3)])
    return rows / rows.sum(axis=1, keepdims=True)


def _write_attack_config(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "attack_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _patch_main_dependencies(monkeypatch, module, tmp_path: Path, config: dict, final: np.ndarray, mbert: np.ndarray):
    records = [{"id": "row0"}, {"id": "row1"}]
    captured: dict[str, np.ndarray] = {}
    encoder_calls: list[tuple[str, bool]] = []

    monkeypatch.setattr(module, "ATTACK_CONFIG_PATH", _write_attack_config(tmp_path, config))
    monkeypatch.setattr(module, "validate_files", lambda: None)
    monkeypatch.setattr(module, "validate_mbert", lambda cfg: None)
    monkeypatch.setattr(module, "load_test_records", lambda: records)
    monkeypatch.setattr(module.joblib, "load", lambda path: {"actions": module.ACTIONS, "sparse_model": object()})

    def fake_encoder_predict(records_arg, encoder_dir=module.ENCODER_DIR_STR, align_by_label=False):
        encoder_calls.append((str(encoder_dir), align_by_label))
        if align_by_label:
            return mbert.copy()
        return np.full_like(final, 1.0 / final.shape[1])

    def fake_write_submission(records_arg, proba):
        captured["proba"] = proba.copy()
        return Path("output/submission.csv")

    monkeypatch.setattr(module, "encoder_predict", fake_encoder_predict)
    monkeypatch.setattr(module, "sparse_predict", lambda model, records_arg: np.full_like(final, 1.0 / final.shape[1]))
    monkeypatch.setattr(module, "stacker_predict", lambda payload, records_arg, baseline, e5: final.copy())
    monkeypatch.setattr(module, "apply_fix", lambda base, cfg: base.copy())
    monkeypatch.setattr(module, "apply_au_route", lambda records_arg, proba, alpha: proba.copy())
    monkeypatch.setattr(module, "write_submission", fake_write_submission)
    return captured, encoder_calls


def test_load_merge080_script_uses_candidate_action_order(merge080_script):
    module = merge080_script

    assert module.ACTIONS[0] == "read_file"
    assert module.ACTIONS[-1] == "respond_only"
    assert len(module.ACTIONS) == 14


def test_encoder_predict_align_by_label_reorders_alphabetical_id2label(merge080_script, monkeypatch):
    module = merge080_script
    alphabetical_labels = sorted(module.ACTIONS)
    id2label = {i: label for i, label in enumerate(alphabetical_labels)}
    raw_logits = np.vstack(
        [
            np.arange(len(module.ACTIONS), dtype=np.float64),
            np.arange(len(module.ACTIONS), 0, -1, dtype=np.float64),
        ]
    )
    fake_model = _FakeSequenceClassifier(raw_logits, id2label)

    monkeypatch.setattr(module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *args, **kwargs: _FakeTokenizer()))
    monkeypatch.setattr(
        module,
        "AutoModelForSequenceClassification",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: fake_model),
    )

    actual = module.encoder_predict([{"id": "row0"}, {"id": "row1"}], "fake-mbert", align_by_label=True)
    raw_proba = module.softmax(raw_logits)
    expected = raw_proba[:, [alphabetical_labels.index(action) for action in module.ACTIONS]]

    np.testing.assert_allclose(actual, expected)
    assert not np.allclose(actual, raw_proba)


def test_main_mbert_mix_uses_env_override_and_normalizes_formula(merge080_script, monkeypatch, tmp_path):
    module = merge080_script
    final = _normalized_rows(module)
    mbert = _normalized_rows(module, reverse=True)
    mbert_dir = tmp_path / "mbert_full"
    mbert_dir.mkdir()
    config = {"mbert": {"mix": 0.0, "dir": str(mbert_dir)}}
    monkeypatch.setenv("MBERT_MIX", "0.2")
    captured, encoder_calls = _patch_main_dependencies(monkeypatch, module, tmp_path, config, final, mbert)

    module.main()

    expected = 0.8 * final + 0.2 * mbert
    expected = expected / expected.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(captured["proba"], expected)
    np.testing.assert_allclose(captured["proba"].sum(axis=1), np.ones(final.shape[0]))
    assert encoder_calls == [(module.ENCODER_DIR_STR, False), (str(mbert_dir), True)]


def test_main_mbert_mix_zero_skips_mbert_path(merge080_script, monkeypatch, tmp_path):
    module = merge080_script
    final = _normalized_rows(module)
    mbert = _normalized_rows(module, reverse=True)
    config = {"mbert": {"mix": 0.0, "dir": str(tmp_path / "missing_mbert")}}
    captured, encoder_calls = _patch_main_dependencies(monkeypatch, module, tmp_path, config, final, mbert)

    module.main()

    np.testing.assert_allclose(captured["proba"], final)
    assert encoder_calls == [(module.ENCODER_DIR_STR, False)]


def test_validate_mbert_missing_dir_raises_when_mix_positive(merge080_script, tmp_path):
    module = merge080_script

    with pytest.raises(FileNotFoundError, match="mbert.mix=0.2"):
        module.validate_mbert({"mbert": {"mix": 0.2, "dir": str(tmp_path / "missing")}})


def test_validate_mbert_skips_when_mix_zero(merge080_script, tmp_path):
    module = merge080_script

    module.validate_mbert({"mbert": {"mix": 0.0, "dir": str(tmp_path / "missing")}})


def test_validate_mbert_bad_label_set_raises(merge080_script, tmp_path):
    module = merge080_script
    mbert_dir = tmp_path / "mbert_full"
    mbert_dir.mkdir()
    (mbert_dir / "pytorch_model.bin").write_bytes(b"")
    bad_id2label = {str(i): action for i, action in enumerate(module.ACTIONS)}
    bad_id2label["0"] = "not_an_action"
    (mbert_dir / "config.json").write_text(json.dumps({"id2label": bad_id2label}), encoding="utf-8")

    with pytest.raises(RuntimeError, match="mbert label set mismatch"):
        module.validate_mbert({"mbert": {"mix": 0.1, "dir": str(mbert_dir)}})


def test_validate_mbert_env_mix_overrides_config(merge080_script, monkeypatch, tmp_path):
    module = merge080_script
    missing_dir = tmp_path / "missing"

    monkeypatch.setenv("MBERT_MIX", "0")
    module.validate_mbert({"mbert": {"mix": 0.5, "dir": str(missing_dir)}})

    monkeypatch.setenv("MBERT_MIX", "0.3")
    with pytest.raises(FileNotFoundError, match="mbert.mix=0.3"):
        module.validate_mbert({"mbert": {"mix": 0.0, "dir": str(missing_dir)}})
