"""scripts/speed 하네스 스모크 테스트.

DoD(task2.md #4): 하네스가 소행에서 완주 + 등가성 로직이 검증됨을 증명한다.
이 venv(C:\\dev\\2026-AI-DACON\\.venv)에는 torch가 없다 — 모델 로드가 필요한
실측 경로는 torch 부재 시 skip 하고, 컴파일·구조·`encoder_probs()` 위임 검증까지만
이 단계에서 확인한다. 실측(실제 300행 CPU 계측)은 torch가 설치된 환경(서버)에서
별도로 수행한다.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    HAS_REAL_TORCH = True
except ImportError:
    HAS_REAL_TORCH = False


def _stub_torch():
    """submit/script.py 의 top-level `import torch` 를 만족시키는 최소 스텁.
    tests/test_enc_block_weights.py 와 동일한 패턴(실 torch 없이 로직만 검증)."""
    if "torch" in sys.modules:
        return
    stub = ModuleType("torch")
    stub.Tensor = type("Tensor", (), {})
    stub.cuda = SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)
    stub.no_grad = _NullContext
    sys.modules["torch"] = stub


class _NullContext:
    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_x_py_removed():
    """리뷰 지적 (b): 미참조 잔재 x.py 는 더 이상 존재하지 않는다."""
    assert not (ROOT / "scripts" / "speed" / "x.py").exists()


def test_stages_module_compiles_without_torch(monkeypatch):
    """torch 미설치 환경에서도 stages.py 자체는 import(컴파일) 가능해야 한다
    (실제 파이프라인 실행이 아니라 모듈 로드만 확인)."""
    _stub_torch()
    from scripts.speed import stages  # noqa: F401
    assert hasattr(stages, "_pipeline")
    assert hasattr(stages, "run")


def test_pipeline_encoder_block_delegates_to_encoder_probs(monkeypatch):
    """리뷰 지적 (a) 회귀 방지: _pipeline 의 encoder_block 스테이지는 반드시
    m.encoder_probs(samples) 를 호출해야 한다 — enc_block_weights()의 가중 평균과
    load_calib() 를 우회하는 uniform 평균 하드코딩을 재도입하면 이 테스트가 실패한다.
    실제 모델 로드 없이 가짜 모듈 m 으로 호출 계약만 검증한다."""
    _stub_torch()
    from scripts.speed import stages
    import numpy as np

    calls = {"encoder_probs": 0, "one_encoder": 0}

    class FakeModule:
        ACTIONS = ["a", "b"]

        def F(self):
            raise AssertionError("not used in this test")

        def encoder_dirs(self):
            return ["encoder", "encoder_2"]

        def _encoder_max_hist(self, d):
            return 6

        def serialize(self, s, max_hist):
            return "x"

        def linear_probs(self, samples):
            return np.array([[0.5, 0.5]] * len(samples))

        def stacker_probs(self, samples):
            return np.array([[0.5, 0.5]] * len(samples))

        def _one_encoder_probs(self, samples, enc_dir):
            calls["one_encoder"] += 1
            return np.array([[0.5, 0.5]] * len(samples))

        def encoder_probs(self, samples):
            calls["encoder_probs"] += 1
            # 원본 계약: enc_block_weights() 가중 평균 + load_calib() 를 내부에서 적용.
            # 여기서는 위임 여부만 확인하면 되므로 _one_encoder_probs 를 통해 값을 만든다.
            dirs = self.encoder_dirs()
            acc = None
            for d in dirs:
                p = self._one_encoder_probs(samples, d)
                acc = p if acc is None else acc + p
            return acc / len(dirs)

        F = SimpleNamespace(build_dataframe=lambda samples: samples)

        def parse_weights(self):
            return None

        def parse_bucket_weights(self, weights):
            return None

        def au_route_blend(self, samples, blend):
            return blend

        def sibling_label_recovery(self, samples, preds):
            return preds

    m = FakeModule()
    samples = [{"id": "s1"}, {"id": "s2"}]
    preds, times = stages._pipeline(m, samples)

    assert calls["encoder_probs"] == 1, "encoder_block 스테이지는 encoder_probs()를 정확히 1회 호출해야 함"
    assert calls["one_encoder"] == len(m.encoder_dirs()), (
        "이중 실행 없이 encoder_probs() 내부의 _one_encoder_probs 호출만으로 성분별 시간이 잡혀야 함"
    )
    assert "encoder_block" in times
    assert "encoder" in times and "encoder_2" in times, "성분별(encoder, encoder_2) 개별 시간이 기록돼야 함"
    assert len(preds) == len(samples)


def test_run_rejects_non_positive_args():
    _stub_torch()
    from scripts.speed import stages
    with pytest.raises(ValueError):
        stages.run(n=0)
    with pytest.raises(ValueError):
        stages.run(n=5, repeats=0)


@pytest.mark.skipif(not HAS_REAL_TORCH, reason="실측(모델 로드) 경로는 실 torch 설치 환경에서만 실행")
def test_harness_completes_on_small_subset():
    """실제 모델·torch 가 있는 환경(서버)에서: 소행(30) 구간에서 완주 + 등가성 PASS."""
    from scripts.speed.stages import run
    from scripts.speed.equivalence_check import run_check

    result = run(n=30, repeats=1)
    assert result["rows"] == 30
    assert len(result["predictions"]) == 30

    check = run_check(n=30)
    assert check["matched"], check["mismatches"]
