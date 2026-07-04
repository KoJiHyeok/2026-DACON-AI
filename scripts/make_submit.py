"""submit/ 스테이징 폴더 → submit.zip 패키징 + 검증 실행.

Usage:
    python scripts/make_submit.py            # 패키징 후 자동 검증
    python scripts/make_submit.py --no-check # 패키징만

패키징 규칙 (대회 규정 구조):
    submit.zip (루트)
    ├── model/            <- submit/model/ 전체 (.gitkeep 제외)
    ├── script.py         <- submit/script.py
    └── requirements.txt  <- submit/requirements.txt
"""
import argparse
import os
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBMIT_DIR = os.path.join(ROOT, "submit")
ZIP_PATH = os.path.join(SUBMIT_DIR, "submit.zip")


def package():
    required = ["script.py", "requirements.txt"]
    for f in required:
        if not os.path.exists(os.path.join(SUBMIT_DIR, f)):
            sys.exit(f"오류: submit/{f} 없음")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)

    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in required:
            zf.write(os.path.join(SUBMIT_DIR, f), f)
        model_dir = os.path.join(SUBMIT_DIR, "model")
        for dirpath, _, filenames in os.walk(model_dir):
            for fname in filenames:
                if fname == ".gitkeep":
                    continue
                full = os.path.join(dirpath, fname)
                arc = os.path.relpath(full, SUBMIT_DIR).replace(os.sep, "/")
                zf.write(full, arc)

    size_mb = os.path.getsize(ZIP_PATH) / 1024 / 1024
    print(f"패키징 완료: {ZIP_PATH} ({size_mb:.1f} MB)")
    return ZIP_PATH


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-check", action="store_true", help="검증 생략")
    args = ap.parse_args()

    zip_path = package()
    if not args.no_check:
        rc = subprocess.run([
            sys.executable,
            os.path.join(ROOT, "scripts", "validate_submit.py"),
            zip_path,
            "--data-dir", os.path.join(ROOT, "data"),
        ]).returncode
        sys.exit(rc)


if __name__ == "__main__":
    main()
