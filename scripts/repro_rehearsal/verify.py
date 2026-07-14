"""Verify the deployed #14 champion and surviving reproduction evidence.

The default invocation verifies four active components: AAR, linear,
Qwen encoder, and AU.  Qwen full-training metadata is not available locally,
so that component verifies the immutable deployed bytes and runtime contract;
the report calls the missing training manifest out explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EXTERNAL_ROOT = Path(r"C:\dev\2026-AI-DACON")
DEFAULT_REPRO_ROOT = Path(r"C:\dev\night\2026-07-13\task2\out_repro")
COMPONENT_NAMES = ("aar", "linear", "qwen-encoder", "au")


@dataclass(frozen=True)
class ComponentSpec:
    metric_paths: tuple[str, ...]
    metric_key: str
    expected: float
    tolerance: float
    manifest_dir: str | None = None
    required_repro_files: tuple[str, ...] = ()
    metric_root: str = "external"


SPECS = {
    "aar": ComponentSpec(
        metric_paths=("aar/train.combined.log",),
        metric_key="stacked_oof_macro_f1",
        expected=0.7034,
        tolerance=0.005,
        manifest_dir="artifacts/experiments/oof_aar",
        required_repro_files=("aar/model/aar_models.joblib", "aar/model/aar_config.json"),
        metric_root="repro",
    ),
    "linear": ComponentSpec(
        metric_paths=("linear/baseline_repro/summary.json",),
        metric_key="oof_macro_f1",
        expected=0.663895,
        tolerance=0.005,
        manifest_dir="artifacts/experiments/oof_linear",
        required_repro_files=(
            "linear/baseline_repro/summary.json",
            "linear/baseline_repro/repro_probs.npy",
        ),
        metric_root="repro",
    ),
    "au": ComponentSpec(
        metric_paths=("artifacts/experiments/oof_au/run_oof_au.json",),
        metric_key="pooled_au_subset_macro_f1",
        expected=0.703154,
        tolerance=0.005,
        manifest_dir="artifacts/experiments/oof_au",
    ),
}


# Exact bytes measured in C:\dev\2026-AI-DACON\submit on 2026-07-14.
DEPLOYED_ARTIFACTS = {
    "aar": (
        ("submit/model/stacker/aar_models.joblib", 47_420_632,
         "31b10456d072ce0e7e4a868c1f02fe7451cf90fe786de53104fed3dec1e0ed6d"),
        ("submit/model/stacker/aar_config.json", 1_402,
         "f7eb7d95003bf003cf6cdd68c593547b486bb7a67ae201c644589a70fb362e27"),
        ("submit/fast_aar.py", 17_401,
         "7ccbb898a9e9103ff889c603be94378db04bef264c11dea0a18fa0b6132b2042"),
    ),
    "linear": (
        ("submit/model/linear/model.pkl", 8_354_283,
         "ebc07c26455c3e2e93d4d662abb3ba14876636904f8c69288fd58179c72a8877"),
    ),
    "qwen-encoder": (
        ("submit/model/encoder/model.safetensors", 988_122_712,
         "0e9f798c58b4334861376a2f8372ee0d41d38a88e3fc38854427488e54e5a056"),
    ),
    "au": (
        ("submit/model/au_linear/model.pkl", 17_673_981,
         "bc01eb659eca930bcad238d9210beb6c2c72d11b4cdbb778fc136a0bd98725e0"),
    ),
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_or_log(path: Path) -> dict[str, Any]:
    """Load a JSON object, including the final object in a mixed stdout log."""
    raw = path.read_bytes()
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = raw.decode("utf-16", errors="replace")
    elif raw.count(b"\x00") > max(8, len(raw) // 10):
        text = raw.decode("utf-16-le", errors="replace")
    else:
        text = raw.decode("utf-8-sig", errors="replace")
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    objects: list[tuple[int, int, dict[str, Any]]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            objects.append((index + end, -index, value))
    if not objects:
        raise ValueError(f"no JSON object found in {path}")
    return max(objects, key=lambda row: (row[0], row[1]))[2]


def metric_check(values: Iterable[float], expected: float, tolerance: float) -> dict[str, Any]:
    measured_values = [float(value) for value in values]
    if not measured_values:
        raise ValueError("metric check requires at least one value")
    measured = sum(measured_values) / len(measured_values)
    delta = measured - float(expected)
    within = abs(delta) <= tolerance or math.isclose(
        abs(delta), tolerance, rel_tol=0.0, abs_tol=1e-12
    )
    return {
        "status": "pass" if math.isfinite(measured) and within else "fail",
        "measured": measured,
        "values": measured_values,
        "expected": float(expected),
        "tolerance": float(tolerance),
        "delta": delta,
    }


def parse_sha256sums(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64:
            raise ValueError(f"invalid SHA256SUMS line {line_no}: {raw!r}")
        name = parts[1].lstrip("*")
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError(f"unsafe manifest path on line {line_no}: {name!r}")
        entries.append((parts[0].lower(), name))
    if not entries:
        raise ValueError(f"empty SHA256SUMS: {path}")
    return entries


def verify_manifest(directory: Path) -> dict[str, Any]:
    manifest = directory / "SHA256SUMS"
    result: dict[str, Any] = {"path": str(manifest), "status": "fail", "files": []}
    if not manifest.is_file():
        result["error"] = "manifest missing"
        return result
    try:
        entries = parse_sha256sums(manifest)
    except (OSError, ValueError) as exc:
        result["error"] = str(exc)
        return result
    for expected, name in entries:
        target = directory / name
        row: dict[str, Any] = {"path": str(target), "expected_sha256": expected}
        if not target.is_file():
            row.update(status="fail", error="file missing")
        else:
            actual = sha256_file(target)
            row.update(actual_sha256=actual, status="pass" if actual == expected else "fail")
        result["files"].append(row)
    result["status"] = "pass" if all(row["status"] == "pass" for row in result["files"]) else "fail"
    return result


def verify_deployed_artifacts(name: str, external_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for relative, expected_size, expected_sha in DEPLOYED_ARTIFACTS[name]:
        path = external_root / relative
        row: dict[str, Any] = {
            "path": str(path),
            "expected_bytes": expected_size,
            "expected_sha256": expected_sha,
            "status": "fail",
        }
        if not path.is_file():
            row["error"] = "file missing"
        else:
            actual_size = path.stat().st_size
            actual_sha = sha256_file(path)
            row.update(actual_bytes=actual_size, actual_sha256=actual_sha)
            row["status"] = (
                "pass" if actual_size == expected_size and actual_sha == expected_sha else "fail"
            )
        rows.append(row)
    return {"status": "pass" if all(row["status"] == "pass" for row in rows) else "fail", "files": rows}


def _contract_row(name: str, actual: Any, expected: Any) -> dict[str, Any]:
    return {
        "name": name,
        "actual": actual,
        "expected": expected,
        "status": "pass" if actual == expected else "fail",
    }


def verify_qwen_contract(external_root: Path) -> dict[str, Any]:
    submit = external_root / "submit"
    model = submit / "model"
    encoder = model / "encoder"
    rows: list[dict[str, Any]] = []
    try:
        config = json.loads((encoder / "config.json").read_text(encoding="utf-8-sig"))
        serialize_config = json.loads(
            (encoder / "serialize_config.json").read_text(encoding="utf-8-sig")
        )
        weights = json.loads((model / "weights.json").read_text(encoding="utf-8-sig"))["weights"]
        block_weights = json.loads(
            (model / "enc_block_weights.json").read_text(encoding="utf-8-sig")
        )["weights"]
        script = (submit / "script.py").read_text(encoding="utf-8-sig")
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return {"status": "fail", "error": str(exc), "checks": rows}

    encoder_dirs = sorted(
        path.name for path in model.iterdir()
        if path.is_dir() and (path.name == "encoder" or path.name.startswith("encoder_"))
    )
    rows.extend(
        [
            _contract_row("encoder_dirs", encoder_dirs, ["encoder"]),
            _contract_row("architecture", config.get("architectures"), ["Qwen2ForSequenceClassification"]),
            _contract_row("model_type", config.get("model_type"), "qwen2"),
            _contract_row("dtype", config.get("dtype"), "float16"),
            _contract_row("num_hidden_layers", config.get("num_hidden_layers"), 24),
            _contract_row("label_count", len(config.get("id2label", {})), 14),
            _contract_row("serialize.max_hist", serialize_config.get("max_hist"), 12),
            _contract_row("enc_block_weights", block_weights, [1.0]),
            _contract_row("blend_weights", weights, [1.0, 1.0, 2.0]),
            _contract_row("fast_aar_import", "import fast_aar as FAAR" in script, True),
            _contract_row("fast_aar_call", "FAAR.fast_predict_proba" in script, True),
            _contract_row("length_sorted_batching", "order = sorted" in script, True),
            _contract_row("encoder_order_restore", "out[idx] = softmax" in script, True),
            _contract_row("deployed_features_import", "import features as F" in script, True),
            _contract_row("soft_au_default", 'ENS_AU_ALPHA", "0.9"' in script, True),
        ]
    )
    return {
        "status": "pass" if all(row["status"] == "pass" for row in rows) else "fail",
        "checks": rows,
        "training_evidence": {
            "status": "documented-only",
            "note": "Qwen full mode has no preserved local run JSON or SHA256 manifest; deployed bytes are the immutable anchor.",
        },
    }


def verify_component(name: str, external_root: Path, repro_root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"component": name, "status": "fail", "checks": {}}
    result["checks"]["deployed_artifacts"] = verify_deployed_artifacts(name, external_root)

    if name == "qwen-encoder":
        result["checks"]["runtime_contract"] = verify_qwen_contract(external_root)
    else:
        spec = SPECS[name]
        metric_base = repro_root if spec.metric_root == "repro" else external_root
        values: list[float] = []
        files: list[dict[str, Any]] = []
        error = None
        for relative in spec.metric_paths:
            path = metric_base / relative
            row: dict[str, Any] = {"path": str(path), "key": spec.metric_key}
            try:
                value = float(load_json_or_log(path)[spec.metric_key])
                values.append(value)
                row.update(status="pass", value=value)
            except (OSError, ValueError, KeyError, TypeError) as exc:
                row.update(status="fail", error=str(exc))
                error = str(exc)
            files.append(row)
        metric = (
            metric_check(values, spec.expected, spec.tolerance)
            if error is None
            else {"status": "fail", "expected": spec.expected, "tolerance": spec.tolerance, "error": error}
        )
        metric["files"] = files
        result["checks"]["metric"] = metric

        required = []
        for relative in spec.required_repro_files:
            path = repro_root / relative
            required.append({"path": str(path), "status": "pass" if path.is_file() else "fail"})
        if required:
            result["checks"]["repro_files"] = {
                "status": "pass" if all(row["status"] == "pass" for row in required) else "fail",
                "files": required,
                "note": "fresh 2026-07-13 CPU rehearsal outputs; existence checked",
            }
        if spec.manifest_dir:
            result["checks"]["artifact_manifest"] = verify_manifest(external_root / spec.manifest_dir)

    result["status"] = (
        "pass" if all(check["status"] == "pass" for check in result["checks"].values()) else "fail"
    )
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--component", choices=("all",) + COMPONENT_NAMES, default="all")
    parser.add_argument("--external-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    parser.add_argument("--repro-root", type=Path, default=DEFAULT_REPRO_ROOT)
    parser.add_argument("--output", type=Path, default=None, help="also write the JSON report")
    parser.add_argument("--details", action="store_true", help="print per-file details to stdout")
    return parser.parse_args(argv)


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    """Return a concise, complete status view for human-readable stdout."""
    components = []
    for component in report["components"]:
        checks: dict[str, Any] = {}
        for name, check in component["checks"].items():
            row: dict[str, Any] = {"status": check["status"]}
            if "files" in check:
                row["files_checked"] = len(check["files"])
            if "measured" in check:
                row.update(
                    measured=check["measured"],
                    expected=check["expected"],
                    tolerance=check["tolerance"],
                )
            if "checks" in check:
                row["checks_passed"] = sum(
                    item["status"] == "pass" for item in check["checks"]
                )
                row["checks_total"] = len(check["checks"])
            if "training_evidence" in check:
                row["training_evidence"] = check["training_evidence"]["status"]
            checks[name] = row
        components.append(
            {"component": component["component"], "status": component["status"], "checks": checks}
        )
    return {
        "status": report["status"],
        "champion": report["champion"],
        "external_root": report["external_root"],
        "repro_root": report["repro_root"],
        "components": components,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    selected = COMPONENT_NAMES if args.component == "all" else (args.component,)
    components = [verify_component(name, args.external_root, args.repro_root) for name in selected]
    report = {
        "status": "pass" if all(row["status"] == "pass" for row in components) else "fail",
        "champion": {"submission": 14, "public_lb": 0.77089},
        "external_root": str(args.external_root),
        "repro_root": str(args.repro_root),
        "components": components,
    }
    rendered = json.dumps(report if args.details else summarize_report(report), ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(rendered)
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
