"""Generate the 2026-07-14 final-submission freeze checklist.

The command has no required arguments.  Optional CLI values override environment
variables, which in turn override the production defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence


DEFAULT_PROJECT_ROOT = Path(r"C:\dev\2026-AI-DACON")
DEFAULT_FREEZE_DATE = "2026-07-14"
DEFAULT_EXPECTED_SUBMISSION = 20  # #19·#20 동점(0.77350, max-draw) — 파서는 마지막 최대 행. 개별 실점수는 제출 페이지만 앎
DEFAULT_EXPECTED_LB = Decimal("0.77350")
SERVER_ROLLBACK_PATHS = (
    "~/models/champ_encoder_s42",
    "~/out/qwen05i_2ep_full",
)
STATUS_ORDER = {"PASS": 0, "WARN": 1, "FAIL": 2}


@dataclass(frozen=True)
class Submission:
    number: int
    public_lb: Decimal
    cells: dict[str, str]


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str
    detail: str

    def __post_init__(self) -> None:
        if self.status not in STATUS_ORDER:
            raise ValueError(f"unknown status: {self.status}")


@dataclass(frozen=True)
class Config:
    project_root: Path
    output_root: Path
    submit_dir: Path
    ledger_path: Path
    git_root: Path
    temp_dir: Path
    disk_root: Path
    freeze_date: str = DEFAULT_FREEZE_DATE
    expected_submission: int = DEFAULT_EXPECTED_SUBMISSION
    expected_lb: Decimal = DEFAULT_EXPECTED_LB
    min_free_gb: float = 10.0

    @property
    def report_dir(self) -> Path:
        return self.output_root / "context" / "reports"

    @property
    def manifest_path(self) -> Path:
        return self.report_dir / f"freeze_manifest_{self.freeze_date}.json"

    @property
    def checklist_path(self) -> Path:
        return self.report_dir / f"freeze_checklist_{self.freeze_date}.md"


def _strip_markdown(value: str) -> str:
    return value.strip().replace("**", "").replace("`", "").strip()


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return []
    return [cell.strip() for cell in stripped[1:-1].split("|")]


def _normalized_header(value: str) -> str:
    return re.sub(r"\s+", " ", _strip_markdown(value)).casefold()


def parse_submissions(text: str) -> list[Submission]:
    """Parse numeric public-LB rows from the submissions Markdown table."""

    lines = text.splitlines()
    header_index = -1
    headers: list[str] = []
    for index, line in enumerate(lines):
        cells = _split_markdown_row(line)
        normalized = [_normalized_header(cell) for cell in cells]
        if "#" in normalized and "lb (public)" in normalized:
            header_index = index
            headers = normalized
            break
    if header_index < 0:
        raise ValueError("submissions table header not found")

    number_column = headers.index("#")
    lb_column = headers.index("lb (public)")
    submissions: list[Submission] = []
    for line in lines[header_index + 1 :]:
        cells = _split_markdown_row(line)
        if not cells:
            if submissions:
                break
            continue
        if len(cells) != len(headers):
            continue
        try:
            number_text = _strip_markdown(cells[number_column])
            if not re.fullmatch(r"\d+", number_text):
                continue
            lb_text = _strip_markdown(cells[lb_column])
            if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", lb_text):
                continue
            public_lb = Decimal(lb_text)
        except (InvalidOperation, ValueError):
            continue
        submissions.append(
            Submission(
                number=int(number_text),
                public_lb=public_lb,
                cells={headers[i]: cells[i].strip() for i in range(len(headers))},
            )
        )
    return submissions


def best_submission(text: str) -> Submission:
    submissions = parse_submissions(text)
    if not submissions:
        raise ValueError("no submission row has a numeric public LB")
    return max(submissions, key=lambda row: (row.public_lb, row.number))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(submit_dir: Path) -> dict[str, object]:
    if not submit_dir.is_dir():
        raise FileNotFoundError(f"submit directory not found: {submit_dir}")

    paths = sorted(
        (path for path in submit_dir.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(submit_dir).as_posix(),
    )
    files: list[dict[str, object]] = []
    total_size = 0
    for path in paths:
        before = path.stat()
        digest = _sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"file changed while hashing: {path}")
        size = after.st_size
        files.append(
            {
                "path": path.relative_to(submit_dir).as_posix(),
                "size_bytes": size,
                "sha256": digest,
            }
        )
        total_size += size

    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "submit_root": str(submit_dir.resolve()),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "files": files,
    }


def write_manifest(submit_dir: Path, destination: Path) -> dict[str, object]:
    manifest = build_manifest(submit_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return manifest


def _result(name: str, status: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status=status, detail=detail)


def check_manifest(config: Config) -> CheckResult:
    try:
        manifest = write_manifest(config.submit_dir, config.manifest_path)
    except Exception as exc:  # checklist must record operational failures
        return _result("submit/ SHA256 매니페스트", "FAIL", str(exc))
    size_mb = int(manifest["total_size_bytes"]) / (1024 * 1024)
    return _result(
        "submit/ SHA256 매니페스트",
        "PASS",
        f"{manifest['file_count']}개 파일, {size_mb:.1f} MiB → {config.manifest_path}",
    )


def check_ledger(config: Config) -> CheckResult:
    try:
        text = config.ledger_path.read_text(encoding="utf-8")
        champion = best_submission(text)
    except Exception as exc:
        return _result("public LB 최고 제출", "FAIL", str(exc))
    expected = (
        champion.number == config.expected_submission
        and champion.public_lb == config.expected_lb
    )
    status = "PASS" if expected else "FAIL"
    detail = (
        f"numeric LB 최고 = #{champion.number} ({champion.public_lb}); "
        f"기대값 = #{config.expected_submission} ({config.expected_lb})"
    )
    return _result("public LB 최고 제출", status, detail)


def check_rollbacks(config: Config) -> list[CheckResult]:
    backup = config.project_root / "colab_out" / "mbert_encoder2_backup"
    fast_aar = config.submit_dir / "fast_aar.py"
    serialize_config = config.submit_dir / "model" / "encoder" / "serialize_config.json"
    results = [
        _result(
            "mBERT rollback 자산",
            "PASS" if backup.exists() else "FAIL",
            f"{'존재' if backup.exists() else '없음'}: {backup}",
        ),
        _result(
            "fast_aar rollback 자산",
            "PASS" if fast_aar.is_file() else "FAIL",
            f"{'존재' if fast_aar.is_file() else '없음'}: {fast_aar}",
        ),
    ]
    try:
        payload = json.loads(serialize_config.read_text(encoding="utf-8"))
        exact = payload == {"max_hist": 12}
        results.append(
            _result(
                "encoder serialize 계약",
                "PASS" if exact else "FAIL",
                f"{serialize_config}: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}; "
                "기대값 = {\"max_hist\": 12}",
            )
        )
    except Exception as exc:
        results.append(_result("encoder serialize 계약", "FAIL", str(exc)))
    results.append(
        _result(
            "서버 rollback 경로",
            "PASS",
            "문서 기재만(접속 시도 안 함): " + ", ".join(SERVER_ROLLBACK_PATHS),
        )
    )
    return results


def check_temp(config: Config) -> CheckResult:
    try:
        leftovers = sorted(config.temp_dir.glob("dacon_submit_*"), key=lambda p: str(p))
    except OSError as exc:
        return _result("TEMP 제출 잔해", "WARN", f"목록 조회 실패: {exc}")
    if not leftovers:
        return _result("TEMP 제출 잔해", "PASS", f"없음: {config.temp_dir}")
    rendered = []
    for path in leftovers:
        kind = "dir" if path.is_dir() else "file"
        size = f", {path.stat().st_size} bytes" if path.is_file() else ""
        rendered.append(f"{path} ({kind}{size})")
    return _result(
        "TEMP 제출 잔해",
        "WARN",
        f"{len(leftovers)}개 발견(삭제하지 않음): " + "; ".join(rendered),
    )


def check_disk(config: Config) -> CheckResult:
    try:
        usage = shutil.disk_usage(config.disk_root)
    except OSError as exc:
        return _result("디스크 여유", "WARN", f"조회 실패: {exc}")
    free_gb = usage.free / (1024**3)
    status = "PASS" if free_gb >= config.min_free_gb else "WARN"
    return _result(
        "디스크 여유",
        status,
        f"{config.disk_root}: {free_gb:.2f} GiB free "
        f"(WARN 기준 {config.min_free_gb:.2f} GiB 미만)",
    )


def check_git_clean(config: Config) -> CheckResult:
    command = [
        "git",
        "-c",
        f"safe.directory={config.git_root.resolve().as_posix()}",
        "-C",
        str(config.git_root),
        "status",
        "--porcelain=v1",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        return _result("git 작업트리", "FAIL", f"git 실행 실패: {exc}")
    if completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        return _result("git 작업트리", "FAIL", f"상태 조회 실패: {message}")
    # 이 도구의 자기 산출물(매니페스트·체크리스트)은 실행할 때마다 재생성되어
    # (generated_at 타임스탬프) 추적 파일이면 스스로 dirty를 만든다 — 판정에서 제외.
    own_outputs = set()
    for p in (config.manifest_path, config.checklist_path):
        try:
            own_outputs.add(p.resolve().relative_to(config.git_root.resolve()).as_posix())
        except ValueError:
            pass
    lines = [
        line
        for line in completed.stdout.splitlines()
        if line.strip() and line[3:].strip().strip('"') not in own_outputs
    ]
    if lines:
        return _result("git 작업트리", "FAIL", "미커밋 변경:\n" + "\n".join(lines))
    detail = f"clean: {config.git_root}"
    if own_outputs:
        detail += " (자기 산출물 제외: " + ", ".join(sorted(own_outputs)) + ")"
    return _result("git 작업트리", "PASS", detail)


def _markdown_text(text: str) -> str:
    return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")


def render_checklist(config: Config, results: Sequence[CheckResult]) -> str:
    counts = {status: sum(r.status == status for r in results) for status in STATUS_ORDER}
    overall = "FAIL" if counts["FAIL"] else "PASS"
    lines = [
        f"# Freeze checklist — {config.freeze_date}",
        "",
        f"- 종합 판정: **{overall}**",
        f"- 집계: PASS {counts['PASS']} / WARN {counts['WARN']} / FAIL {counts['FAIL']}",
        "- 종료 규칙: FAIL이 하나라도 있으면 exit 1; WARN만 있으면 exit 0",
        "",
        "| 항목 | 상태 | 상세 |",
        "|---|---|---|",
    ]
    for result in results:
        lines.append(
            f"| {_markdown_text(result.name)} | **{result.status}** | "
            f"{_markdown_text(result.detail)} |"
        )
    lines.extend(
        [
            "",
            "## 사람 확인 요약",
            "",
            (
                "자동 동결 점검을 통과했습니다. WARN 항목은 삭제·수정하지 않았으므로 "
                "최종 선택 전에 사람이 확인하세요."
                if overall == "PASS"
                else "FAIL 항목이 있습니다. 자산을 자동 수정하지 않았습니다. 최종 선택 전 사람이 판단하세요."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def run_freeze_check(config: Config) -> tuple[int, list[CheckResult]]:
    results: list[CheckResult] = [check_manifest(config), check_ledger(config)]
    results.extend(check_rollbacks(config))
    results.extend([check_temp(config), check_disk(config), check_git_clean(config)])
    config.checklist_path.parent.mkdir(parents=True, exist_ok=True)
    config.checklist_path.write_text(
        render_checklist(config, results), encoding="utf-8", newline="\n"
    )
    return (1 if any(result.status == "FAIL" for result in results) else 0, results)


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default))).expanduser()


def parse_args(argv: Sequence[str] | None = None) -> Config:
    script_root = Path(__file__).resolve().parents[2]
    project_default = _env_path("FREEZE_PROJECT_ROOT", DEFAULT_PROJECT_ROOT)
    output_default = _env_path("FREEZE_OUTPUT_ROOT", script_root)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=output_default,
        help="worktree receiving context/reports (default: this script's worktree)",
    )
    parser.add_argument(
        "--submit-dir",
        type=Path,
        default=_env_path("FREEZE_SUBMIT_DIR", project_default / "submit"),
    )
    parser.add_argument(
        "--ledger",
        type=Path,
        default=_env_path(
            "FREEZE_LEDGER_PATH", project_default / "context" / "submissions.md"
        ),
    )
    parser.add_argument(
        "--git-root",
        type=Path,
        default=_env_path("FREEZE_GIT_ROOT", project_default),
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=_env_path("FREEZE_TEMP_DIR", Path(os.environ.get("TEMP", os.curdir))),
    )
    parser.add_argument(
        "--disk-root",
        type=Path,
        default=_env_path("FREEZE_DISK_ROOT", Path("C:/") if os.name == "nt" else Path("/")),
    )
    parser.add_argument(
        "--freeze-date", default=os.environ.get("FREEZE_DATE", DEFAULT_FREEZE_DATE)
    )
    parser.add_argument(
        "--expected-submission",
        type=int,
        default=int(os.environ.get("FREEZE_EXPECTED_SUBMISSION", DEFAULT_EXPECTED_SUBMISSION)),
    )
    parser.add_argument(
        "--expected-lb",
        type=Decimal,
        default=Decimal(os.environ.get("FREEZE_EXPECTED_LB", str(DEFAULT_EXPECTED_LB))),
    )
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=float(os.environ.get("FREEZE_MIN_FREE_GB", "10")),
    )
    args = parser.parse_args(argv)
    return Config(
        project_root=args.project_root,
        output_root=args.output_root,
        submit_dir=args.submit_dir,
        ledger_path=args.ledger,
        git_root=args.git_root,
        temp_dir=args.temp_dir,
        disk_root=args.disk_root,
        freeze_date=args.freeze_date,
        expected_submission=args.expected_submission,
        expected_lb=args.expected_lb,
        min_free_gb=args.min_free_gb,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        exit_code, results = run_freeze_check(config)
    except Exception as exc:
        print(f"[FAIL] freeze checklist 생성 실패: {exc}", file=sys.stderr)
        return 1
    for result in results:
        print(f"[{result.status}] {result.name}: {result.detail}")
    print(f"[REPORT] manifest: {config.manifest_path}")
    print(f"[REPORT] checklist: {config.checklist_path}")
    print(f"[RESULT] exit {exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
