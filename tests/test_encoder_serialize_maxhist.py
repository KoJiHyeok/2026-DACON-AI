"""Regression tests for submit/script.py per-encoder serialize max_hist contract.

exp #34 / D-010: e5는 hist12, mBERT는 hist6 — 인코더별 serialize_config.json 으로 분리.
파일 없으면 6(기존 계약) = 무회귀. serialize(max_hist)가 실제로 history 절단 길이를 바꾸는지도 확인.
"""
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
    saved = {k: sys.modules.pop(k) for k in ("features", "aar_infer") if k in sys.modules}
    sys.path.insert(0, str(SUBMIT_SCRIPT.parent))
    try:
        spec = importlib.util.spec_from_file_location("submit_script_maxhist_test", SUBMIT_SCRIPT)
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


def test_missing_config_defaults_to_6(tmp_path, submit_script):
    """serialize_config.json 없으면 6 = 기존 계약(무회귀)."""
    assert submit_script._encoder_max_hist(str(tmp_path)) == 6


def test_reads_max_hist_12(tmp_path, submit_script):
    (tmp_path / "serialize_config.json").write_text('{"max_hist": 12}', encoding="utf-8")
    assert submit_script._encoder_max_hist(str(tmp_path)) == 12


def test_reads_bom_json(tmp_path, submit_script):
    (tmp_path / "serialize_config.json").write_bytes(codecs.BOM_UTF8 + b'{"max_hist": 12}')
    assert submit_script._encoder_max_hist(str(tmp_path)) == 12


def test_malformed_json_falls_back_to_default(tmp_path, submit_script):
    (tmp_path / "serialize_config.json").write_text("{not json", encoding="utf-8")
    assert submit_script._encoder_max_hist(str(tmp_path)) == 6


def test_nonpositive_falls_back_to_default(tmp_path, submit_script):
    (tmp_path / "serialize_config.json").write_text('{"max_hist": 0}', encoding="utf-8")
    assert submit_script._encoder_max_hist(str(tmp_path)) == 6


def test_serialize_maxhist_changes_history_truncation(submit_script):
    """serialize(max_hist=N)이 실제로 최근 N턴만 반영 — 계약의 핵심."""
    hist = []
    for i in range(12):
        hist.append({"role": "user", "content": f"u{i}"})
        hist.append({"role": "assistant_action", "name": f"act{i}", "result_summary": ""})
    sample = {"current_prompt": "q", "history": hist, "session_meta": {}}

    s6 = submit_script.serialize(sample, max_hist=6)
    s12 = submit_script.serialize(sample, max_hist=12)

    # 24개 엔트리: hist[-6:]=u9~u11, hist[-12:]=u6~u11
    assert "u11" in s6 and "u11" in s12           # 최신은 둘 다
    assert "u6" not in s6 and "u6" in s12          # u6~u8은 hist12만
    assert "u5" not in s12                          # u5 이전은 둘 다 잘림
    assert len(s12) > len(s6)
