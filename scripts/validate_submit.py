"""제출물 검증 하네스 — 평가 서버의 실행 환경을 로컬에서 시뮬레이션한다.

Usage:
    python scripts/validate_submit.py <submit.zip 경로> [--data-dir data]

대회 기준 검증 항목:
    [1] zip 구조   : 루트에 script.py / requirements.txt / model/ 존재, 용량 <= 1GB
    [2] 의존성     : requirements.txt 모든 패키지 버전 고정(==) 여부
    [3] 오프라인 실행: 샌드박스에 서버 레이아웃 재현(./data 제공) 후 script.py 실행
                     - 네트워크 차단(socket 몽키패치 주입) — 위반 시 즉시 실패
                     - 실행 시간 측정, 10분 제한 (로컬 기준 경고선 8분)
    [4] 출력 형식  : ./output/submission.csv 생성 여부, 컬럼 (id, action),
                     sample_submission.csv 와 id 순서·행 수 일치, action ∈ 14클래스

종료 코드: 전부 통과 0, 하나라도 실패 1.
"""
import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

LIMIT_ZIP_BYTES = 1 * 1024 * 1024 * 1024   # 1GB
LIMIT_INFER_SEC = 600                      # 10분 (대회 규정)
WARN_INFER_SEC = 480                       # 8분 (로컬 경고선 — 서버 환경 차이 버퍼)

ACTION_CLASSES = {
    "read_file", "grep_search", "list_directory", "glob_pattern",
    "edit_file", "write_file", "apply_patch", "run_bash",
    "run_tests", "lint_or_typecheck", "ask_user", "plan_task",
    "web_search", "respond_only",
}

# 샌드박스의 PYTHONPATH에 주입되어 네트워크 연결을 차단한다.
# socket.socket 클래스 자체를 바꾸면 이를 서브클래싱하는 라이브러리(joblib 등)가
# 깨지므로, 클래스는 유지하고 '연결' 동작만 차단한다.
NETBLOCK_SITECUSTOMIZE = '''\
import socket

def _blocked(*args, **kwargs):
    raise RuntimeError("NETWORK BLOCKED: submit script attempted a network call (offline rule)")

socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.socket.sendto = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
'''


class Check:
    def __init__(self):
        self.results = []

    def record(self, name, ok, detail=""):
        self.results.append((name, ok, detail))
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}" + (f" — {detail}" if detail else ""))
        return ok

    @property
    def all_ok(self):
        return all(ok for _, ok, _ in self.results)


def check_zip_structure(zip_path, chk):
    size = os.path.getsize(zip_path)
    chk.record("zip 용량 <= 1GB", size <= LIMIT_ZIP_BYTES, f"{size / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    roots = {n.split("/")[0] for n in names}
    chk.record("루트에 script.py", "script.py" in roots)
    chk.record("루트에 requirements.txt", "requirements.txt" in roots)
    chk.record("루트에 model/", any(n.startswith("model/") for n in names))


def check_requirements_pinned(zip_path, chk):
    with zipfile.ZipFile(zip_path) as zf:
        try:
            lines = zf.read("requirements.txt").decode("utf-8").splitlines()
        except KeyError:
            chk.record("requirements 버전 고정", False, "requirements.txt 없음")
            return
    pkgs = [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]
    unpinned = [p for p in pkgs if "==" not in p]
    chk.record("requirements 버전 고정(==)", not unpinned,
               f"{len(pkgs)}개 중 미고정 {unpinned}" if unpinned else f"{len(pkgs)}개 전부 고정")


def run_sandbox(zip_path, data_dir, chk):
    """서버 레이아웃을 재현한 샌드박스에서 script.py를 오프라인 실행한다."""
    sandbox = tempfile.mkdtemp(prefix="dacon_submit_")
    try:
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(sandbox)

        # 서버가 제공하는 입력 재현
        sb_data = os.path.join(sandbox, "data")
        os.makedirs(sb_data, exist_ok=True)
        for fname in ("test.jsonl", "sample_submission.csv"):
            src = os.path.join(data_dir, fname)
            if not os.path.exists(src):
                chk.record("서버 입력 재현", False, f"{src} 없음 — data/ 압축 해제 필요")
                return None
            shutil.copy2(src, sb_data)

        # 네트워크 차단 주입
        netblock = os.path.join(sandbox, "_netblock")
        os.makedirs(netblock, exist_ok=True)
        with open(os.path.join(netblock, "sitecustomize.py"), "w", encoding="utf-8") as f:
            f.write(NETBLOCK_SITECUSTOMIZE)
        env = dict(os.environ)
        env["PYTHONPATH"] = netblock + os.pathsep + env.get("PYTHONPATH", "")

        t0 = time.monotonic()
        proc = subprocess.run(
            [sys.executable, "script.py"],
            cwd=sandbox, env=env, capture_output=True, text=True,
            timeout=LIMIT_INFER_SEC + 60,
        )
        elapsed = time.monotonic() - t0

        ok = proc.returncode == 0
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
        chk.record("script.py 오프라인 실행", ok, "; ".join(tail) if not ok else f"{elapsed:.1f}초")
        if ok:
            chk.record("추론 시간 <= 10분", elapsed <= LIMIT_INFER_SEC, f"{elapsed:.1f}초")
            if elapsed > WARN_INFER_SEC:
                print(f"  경고: {elapsed:.0f}초 — 서버(T4/3vCPU)가 로컬보다 느릴 수 있음")
        return sandbox if ok else None
    except subprocess.TimeoutExpired:
        chk.record("script.py 오프라인 실행", False, f"{LIMIT_INFER_SEC + 60}초 초과 (강제 종료)")
        return None
    finally:
        # 출력 검증을 위해 성공 시에는 호출부에서 정리
        if not chk.all_ok and os.path.isdir(sandbox):
            shutil.rmtree(sandbox, ignore_errors=True)


def read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def check_output(sandbox, data_dir, chk):
    out_path = os.path.join(sandbox, "output", "submission.csv")
    if not chk.record("output/submission.csv 생성", os.path.exists(out_path)):
        return

    fields, rows = read_csv_rows(out_path)
    chk.record("컬럼 == (id, action)", fields == ["id", "action"], str(fields))

    _, sample_rows = read_csv_rows(os.path.join(data_dir, "sample_submission.csv"))
    sample_ids = [r["id"] for r in sample_rows]
    out_ids = [r["id"] for r in rows]
    chk.record("행 수 일치", len(out_ids) == len(sample_ids),
               f"출력 {len(out_ids)} vs 기준 {len(sample_ids)}")
    chk.record("id 순서 일치", out_ids == sample_ids)

    bad = {r["action"] for r in rows if r["action"] not in ACTION_CLASSES}
    chk.record("action ∈ 14클래스", not bad, f"허용 외 값: {sorted(bad)}" if bad else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zip_path", help="검증할 submit.zip 경로")
    ap.add_argument("--data-dir", default="data",
                    help="test.jsonl / sample_submission.csv 위치 (기본: data)")
    args = ap.parse_args()

    print(f"=== 제출물 검증: {args.zip_path} ===")
    chk = Check()
    check_zip_structure(args.zip_path, chk)
    check_requirements_pinned(args.zip_path, chk)
    sandbox = run_sandbox(args.zip_path, args.data_dir, chk)
    if sandbox:
        try:
            check_output(sandbox, args.data_dir, chk)
        finally:
            shutil.rmtree(sandbox, ignore_errors=True)

    n_fail = sum(1 for _, ok, _ in chk.results if not ok)
    print(f"=== 결과: {len(chk.results)}개 검증, 실패 {n_fail}건 ===")
    sys.exit(0 if chk.all_ok else 1)


if __name__ == "__main__":
    main()
