from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.repro_rehearsal import verify


def test_load_json_or_log_uses_final_object(tmp_path: Path) -> None:
    log = tmp_path / "run.log"
    log.write_text(
        'progress {not json}\n{"old": 1}\nDONE\n{"score": 0.7034, "nested": {"last": true}}\n',
        encoding="utf-8",
    )
    assert verify.load_json_or_log(log) == {"score": 0.7034, "nested": {"last": True}}


def test_load_json_or_log_accepts_powershell_utf16(tmp_path: Path) -> None:
    log = tmp_path / "powershell.log"
    log.write_text('warning\n{"stacked_oof_macro_f1": 0.7034}\n', encoding="utf-16")
    assert verify.load_json_or_log(log)["stacked_oof_macro_f1"] == 0.7034


def test_metric_check_accepts_boundary_and_rejects_drift() -> None:
    assert verify.metric_check([0.6984], 0.7034, 0.005)["status"] == "pass"
    assert verify.metric_check([0.69839], 0.7034, 0.005)["status"] == "fail"


def test_verify_manifest_pass_then_detects_corruption(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"stable artifact")
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (tmp_path / "SHA256SUMS").write_text(f"{digest} *artifact.bin\n", encoding="utf-8")

    assert verify.verify_manifest(tmp_path)["status"] == "pass"
    artifact.write_bytes(b"changed")
    assert verify.verify_manifest(tmp_path)["status"] == "fail"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_qwen_contract_accepts_single_encoder_and_fast_paths(tmp_path: Path) -> None:
    encoder = tmp_path / "submit/model/encoder"
    _write_json(
        encoder / "config.json",
        {
            "architectures": ["Qwen2ForSequenceClassification"],
            "model_type": "qwen2",
            "dtype": "float16",
            "num_hidden_layers": 24,
            "id2label": {str(i): f"c{i}" for i in range(14)},
        },
    )
    _write_json(encoder / "serialize_config.json", {"max_hist": 12})
    _write_json(tmp_path / "submit/model/weights.json", {"weights": [1.0, 1.0, 2.0]})
    _write_json(tmp_path / "submit/model/enc_block_weights.json", {"weights": [1.0]})
    (tmp_path / "submit/script.py").write_text(
        'import features as F\nimport fast_aar as FAAR\n'
        'FAAR.fast_predict_proba\norder = sorted([])\nout[idx] = softmax(x)\n'
        'alpha = os.environ.get("ENS_AU_ALPHA", "0.9")\n',
        encoding="utf-8",
    )
    assert verify.verify_qwen_contract(tmp_path)["status"] == "pass"


def test_qwen_contract_rejects_encoder_2(tmp_path: Path) -> None:
    encoder = tmp_path / "submit/model/encoder"
    _write_json(
        encoder / "config.json",
        {
            "architectures": ["Qwen2ForSequenceClassification"],
            "model_type": "qwen2",
            "dtype": "float16",
            "num_hidden_layers": 24,
            "id2label": {str(i): f"c{i}" for i in range(14)},
        },
    )
    _write_json(encoder / "serialize_config.json", {"max_hist": 12})
    _write_json(tmp_path / "submit/model/weights.json", {"weights": [1.0, 1.0, 2.0]})
    _write_json(tmp_path / "submit/model/enc_block_weights.json", {"weights": [1.0]})
    (tmp_path / "submit/model/encoder_2").mkdir()
    (tmp_path / "submit/script.py").write_text(
        'import features as F\nimport fast_aar as FAAR\nFAAR.fast_predict_proba\n'
        'order = sorted([])\nout[idx] = softmax(x)\nENS_AU_ALPHA", "0.9"',
        encoding="utf-8",
    )
    result = verify.verify_qwen_contract(tmp_path)
    assert result["status"] == "fail"
    assert next(row for row in result["checks"] if row["name"] == "encoder_dirs")["status"] == "fail"


def test_verify_deployed_artifact_detects_hash_drift(tmp_path: Path, monkeypatch) -> None:
    target = tmp_path / "submit/model/demo.bin"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"demo")
    monkeypatch.setitem(
        verify.DEPLOYED_ARTIFACTS,
        "demo",
        (("submit/model/demo.bin", 4, hashlib.sha256(b"demo").hexdigest()),),
    )
    assert verify.verify_deployed_artifacts("demo", tmp_path)["status"] == "pass"
    target.write_bytes(b"drift")
    assert verify.verify_deployed_artifacts("demo", tmp_path)["status"] == "fail"


def test_component_set_matches_qwen_champion() -> None:
    assert verify.COMPONENT_NAMES == ("aar", "linear", "qwen-encoder", "au")
    assert "mbert" not in verify.COMPONENT_NAMES


def test_summary_preserves_component_and_gap_statuses() -> None:
    report = {
        "status": "pass",
        "champion": {"submission": 14},
        "external_root": "external",
        "repro_root": "repro",
        "components": [
            {
                "component": "qwen-encoder",
                "status": "pass",
                "checks": {
                    "runtime_contract": {
                        "status": "pass",
                        "checks": [{"status": "pass"}, {"status": "pass"}],
                        "training_evidence": {"status": "documented-only"},
                    }
                },
            }
        ],
    }
    summary = verify.summarize_report(report)
    contract = summary["components"][0]["checks"]["runtime_contract"]
    assert contract["checks_passed"] == 2
    assert contract["training_evidence"] == "documented-only"


def test_argparse_has_no_required_options() -> None:
    args = verify.parse_args([])
    assert args.component == "all"
    assert args.output is None
