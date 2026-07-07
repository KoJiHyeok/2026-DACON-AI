"""Regression tests for submit/script.py encoder block weights parsing."""
from __future__ import annotations

import codecs
import importlib.util
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SUBMIT_SCRIPT = ROOT / "submit" / "script.py"


def load_submit_script():
    if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
        torch_stub = ModuleType("torch")
        torch_stub.Tensor = type("Tensor", (), {})
        torch_stub.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
        sys.modules["torch"] = torch_stub
    # script.py의 `import features`는 submit/features.py를 기대한다. 다른 테스트가
    # src/features.py를 같은 이름으로 캐시해 두면 잘못된 모듈이 잡히므로 격리한다.
    saved = {k: sys.modules.pop(k) for k in ("features", "aar_infer") if k in sys.modules}
    sys.path.insert(0, str(SUBMIT_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("submit_script_under_test", SUBMIT_SCRIPT)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SUBMIT_SCRIPT.parent))
        for k in ("features", "aar_infer"):
            sys.modules.pop(k, None)
        sys.modules.update(saved)
    return module


@pytest.fixture(scope="module")
def submit_script():
    return load_submit_script()


@pytest.fixture()
def isolated_model(tmp_path, monkeypatch, submit_script):
    monkeypatch.delenv("ENS_ENC_BLOCK_WEIGHTS", raising=False)
    monkeypatch.setattr(submit_script, "MODEL", str(tmp_path))
    return tmp_path


def test_enc_block_weights_reads_bom_json(isolated_model, submit_script):
    path = isolated_model / "enc_block_weights.json"
    path.write_bytes(codecs.BOM_UTF8 + b'{"weights": [1.2, 0.8]}')

    assert submit_script.enc_block_weights(2) == [1.2, 0.8]


def test_enc_block_weights_length_mismatch_raises(isolated_model, submit_script):
    (isolated_model / "enc_block_weights.json").write_text("[1.0]", encoding="utf-8")

    with pytest.raises(ValueError, match="1.*!=.*2"):
        submit_script.enc_block_weights(2)


def test_enc_block_weights_negative_raises(isolated_model, submit_script):
    (isolated_model / "enc_block_weights.json").write_text("[1.0, -0.1]", encoding="utf-8")

    with pytest.raises(ValueError, match="음수"):
        submit_script.enc_block_weights(2)


def test_enc_block_weights_missing_returns_none(isolated_model, submit_script):
    assert submit_script.enc_block_weights(2) is None


def test_enc_block_weights_env_overrides_file(isolated_model, monkeypatch, submit_script):
    (isolated_model / "enc_block_weights.json").write_text("[1.0]", encoding="utf-8")
    monkeypatch.setenv("ENS_ENC_BLOCK_WEIGHTS", "1.2,0.8")

    assert submit_script.enc_block_weights(2) == [1.2, 0.8]
