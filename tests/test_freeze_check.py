import hashlib
import json
from decimal import Decimal
from pathlib import Path

from scripts.freeze_check.check import Config, best_submission, main, write_manifest


def test_ledger_parser_selects_numeric_submission_14() -> None:
    ledger = """\
| # | 일시 | LB (public) | 메모 |
|---|---|---|---|
| 11 | now | **0.7623** | old |
| 13 | now | **시간초과 FAIL** | skip text |
| 14 | now | **0.77089** | champion |
"""

    champion = best_submission(ledger)

    assert champion.number == 14
    assert champion.public_lb == Decimal("0.77089")


def test_manifest_smoke_records_every_file(tmp_path: Path) -> None:
    submit_dir = tmp_path / "submit"
    (submit_dir / "model").mkdir(parents=True)
    (submit_dir / "script.py").write_bytes(b"print('ok')\n")
    (submit_dir / "model" / "weights.bin").write_bytes(b"abc")
    destination = tmp_path / "reports" / "manifest.json"

    manifest = write_manifest(submit_dir, destination)

    saved = json.loads(destination.read_text(encoding="utf-8"))
    assert manifest["file_count"] == 2
    assert saved["total_size_bytes"] == len(b"print('ok')\nabc")
    assert [item["path"] for item in saved["files"]] == [
        "model/weights.bin",
        "script.py",
    ]
    assert saved["files"][0]["sha256"] == hashlib.sha256(b"abc").hexdigest()


def test_fail_condition_returns_exit_1(tmp_path: Path) -> None:
    project = tmp_path / "project"
    submit_dir = project / "submit"
    config_dir = submit_dir / "model" / "encoder"
    config_dir.mkdir(parents=True)
    (submit_dir / "script.py").write_text("pass\n", encoding="utf-8")
    (submit_dir / "fast_aar.py").write_text("pass\n", encoding="utf-8")
    (config_dir / "serialize_config.json").write_text(
        '{"max_hist": 6}\n', encoding="utf-8"
    )
    (project / "colab_out" / "mbert_encoder2_backup").mkdir(parents=True)
    ledger = project / "context" / "submissions.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text(
        "| # | LB (public) |\n|---|---|\n| 14 | **0.77089** |\n",
        encoding="utf-8",
    )

    exit_code = main(
        [
            "--project-root",
            str(project),
            "--output-root",
            str(tmp_path / "output"),
            "--submit-dir",
            str(submit_dir),
            "--ledger",
            str(ledger),
            "--git-root",
            str(project),
            "--temp-dir",
            str(tmp_path / "temp"),
            "--disk-root",
            str(tmp_path),
        ]
    )

    assert exit_code == 1
    checklist = (
        tmp_path
        / "output"
        / "context"
        / "reports"
        / "freeze_checklist_2026-07-14.md"
    ).read_text(encoding="utf-8")
    assert "encoder serialize 계약 | **FAIL**" in checklist
    assert "종합 판정: **FAIL**" in checklist
