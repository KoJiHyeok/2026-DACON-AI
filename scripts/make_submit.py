"""제출 게이트 — submit/ 스테이징 → submit.zip. 게이트를 전부 통과해야 zip이 남는다.

게이트 순서 (하나라도 실패하면 zip 미생성/삭제):
    [G1] 단위 테스트 통과 (tests/ — pytest 있으면 pytest, 없으면 직접 실행)
    [G2] git 작업 트리 clean (모든 제출은 커밋에 묶인다 — 본선 코드 검증 7/24 대비)
    [G3] 패키징 (대회 규정 zip 구조: model/ + script.py + requirements.txt)
    [G4] validate_submit.py 12개 검증 (구조·용량·오프라인·시간·출력 형식)
    [G5] context/submissions.md 제출 대장 자동 기록

Usage:
    python scripts/make_submit.py [--cv 0.7xx] [--note "..."]
    python scripts/make_submit.py --allow-dirty   # (개발 중 확인용) G2 생략, 대장 기록 안 함
"""
import argparse
import datetime
import glob
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT_DIR = os.path.join(ROOT, "submit")
ZIP_PATH = os.path.join(SUBMIT_DIR, "submit.zip")
LEDGER = os.path.join(ROOT, "context", "submissions.md")


def fail(msg):
    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    sys.exit(f"[게이트 실패] {msg}")


def gate_tests():
    print("[G1] 단위 테스트...")
    probe = subprocess.run([sys.executable, "-m", "pytest", "--version"],
                           capture_output=True)
    if probe.returncode == 0:
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                             os.path.join(ROOT, "tests")]).returncode
        if rc != 0:
            fail("tests/ 실패 — 고치기 전에는 제출 불가")
    else:
        for test_file in sorted(glob.glob(os.path.join(ROOT, "tests", "test_*.py"))):
            rc = subprocess.run([sys.executable, test_file]).returncode
            if rc != 0:
                fail(f"{os.path.basename(test_file)} 실패 — 고치기 전에는 제출 불가")
    print("  통과")


def gate_git_clean():
    print("[G2] git clean...")
    out = subprocess.run(["git", "-C", ROOT, "status", "--porcelain"],
                         capture_output=True, text=True).stdout.strip()
    if out:
        fail("커밋되지 않은 변경 존재 — 제출은 반드시 커밋에 묶여야 함 (D-005).\n"
             + out)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    print(f"  통과 (commit {commit})")
    return commit


def gate_package():
    print("[G3] 패키징...")
    for f in ("script.py", "requirements.txt"):
        if not os.path.exists(os.path.join(SUBMIT_DIR, f)):
            fail(f"submit/{f} 없음")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in ("script.py", "requirements.txt"):
            zf.write(os.path.join(SUBMIT_DIR, f), f)
        # script.py가 import하는 벤더 모듈도 zip 루트에 포함 (예: features.py, aar_infer.py)
        for extra in sorted(glob.glob(os.path.join(SUBMIT_DIR, "*.py"))):
            if os.path.basename(extra) != "script.py":
                zf.write(extra, os.path.basename(extra))
        for dirpath, _, filenames in os.walk(os.path.join(SUBMIT_DIR, "model")):
            for fname in filenames:
                if fname == ".gitkeep":
                    continue
                full = os.path.join(dirpath, fname)
                arc = os.path.relpath(full, SUBMIT_DIR).replace(os.sep, "/")
                zf.write(full, arc)
    size_mb = os.path.getsize(ZIP_PATH) / 1024 / 1024
    print(f"  통과 ({size_mb:.1f} MB)")
    return size_mb


def gate_validate():
    print("[G4] 대회 기준 검증...")
    rc = subprocess.run([
        sys.executable, os.path.join(ROOT, "scripts", "validate_submit.py"),
        ZIP_PATH, "--data-dir", os.path.join(ROOT, "data"),
    ]).returncode
    if rc != 0:
        fail("검증 실패 — 위 FAIL 항목을 고치기 전에는 제출 불가")


def gate_ledger(commit, size_mb, cv, note):
    print("[G5] 제출 대장 기록...")
    with open(LEDGER, encoding="utf-8") as f:
        n_rows = sum(1 for line in f if line.startswith("|")) - 2  # 헤더 2줄 제외
    now = datetime.datetime.now().strftime("%m-%d %H:%M")
    row = (f"| {n_rows + 1} | {now} | `{commit}` | {size_mb:.1f}MB | PASS "
           f"| {cv} | (제출 후 기입) | {note} |\n")
    with open(LEDGER, "a", encoding="utf-8", newline="\n") as f:
        f.write(row)
    print(f"  기록됨: context/submissions.md #{n_rows + 1} — 제출 후 LB 점수를 채우세요")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cv", default="-", help="이 제출물의 로컬 CV Macro-F1")
    ap.add_argument("--note", default="", help="제출 메모 (무엇이 바뀌었나)")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="개발 중 확인용: git 게이트 생략, 대장 기록 안 함")
    args = ap.parse_args()

    gate_tests()
    commit = "(dirty)" if args.allow_dirty else gate_git_clean()
    size_mb = gate_package()
    gate_validate()
    if args.allow_dirty:
        print("[G5] 생략 (--allow-dirty) — 실제 제출용 zip은 clean 커밋에서 다시 생성할 것")
    else:
        gate_ledger(commit, size_mb, args.cv, args.note)
    print(f"\n모든 게이트 통과: {ZIP_PATH}")


if __name__ == "__main__":
    main()
