# -*- coding: utf-8 -*-
"""제출 파이프라인 scale smoke — 히든 규모 추론시간·peak mem·zip 실측.

공개 test.jsonl(5행)은 형식 검증용이라 10분/메모리 예산을 보장하지 못한다(제3자 평가 지적).
이 스크립트는 train.jsonl을 N행으로 복제해 **실제 submit/script.py**를 서브프로세스로 돌리고
벽시계·스크립트 내부 elapsed·peak RSS·(GPU면)peak VRAM·출력 정합·zip 크기를 실측한다.

config 무관: submit/model/ 에 무엇이 있든(hist6 현행 or hist12 스왑 후) 그대로 측정 → 배포 전후 재사용.

실행:
  # 로컬 하네스 검증(소규모, CPU):
  .venv-merge/Scripts/python.exe scripts/scale_smoke.py --n 60
  # T4 실측(Colab, 아래 셀):
  python scripts/scale_smoke.py --n 30000 --python python --submit ./submit

측정 원리: 서버는 T4 GPU. 로컬 CPU 벽시계는 **보수적 상한**(T4가 훨씬 빠름).
따라서 로컬은 "터지지 않는가/출력 정합" 확인용, 시간 판정은 Colab T4에서.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUDGET_SEC = 600  # 서버 추론 예산 10분


def build_scaled_data(src_jsonl: Path, out_dir: Path, n: int, seed: int = 42) -> int:
    """train.jsonl에서 n행 복제(라벨 제거, id 고유화) → test.jsonl + sample_submission.csv."""
    import random

    rng = random.Random(seed)
    base = []
    with src_jsonl.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                base.append(json.loads(line))
    assert base, f"빈 소스: {src_jsonl}"

    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.jsonl"
    sub_path = out_dir / "sample_submission.csv"
    dummy = "read_file"  # sample_submission의 자리표시 라벨 (script가 덮어씀)
    with test_path.open("w", encoding="utf-8") as tf, sub_path.open("w", encoding="utf-8", newline="") as sf:
        sf.write("id,action\n")
        for i in range(n):
            s = dict(rng.choice(base))
            s.pop("action", None)  # 혹시 섞여있으면 라벨 제거 (test엔 없음)
            new_id = f"{s['id']}__smoke{i}"
            s["id"] = new_id
            tf.write(json.dumps(s, ensure_ascii=False) + "\n")
            sf.write(f"{new_id},{dummy}\n")
    return n


def poll_peak(proc: subprocess.Popen, stop: threading.Event, out: dict, interval: float = 0.2):
    """자식 프로세스 트리의 peak RSS(psutil) + peak VRAM(nvidia-smi, 있으면)."""
    try:
        import psutil
    except ImportError:
        out["rss_mb"] = None
        return
    p = psutil.Process(proc.pid)
    peak_rss = 0
    peak_vram = 0
    have_smi = _has_nvidia_smi()
    while not stop.is_set():
        try:
            rss = p.memory_info().rss
            for c in p.children(recursive=True):
                try:
                    rss += c.memory_info().rss
                except Exception:
                    pass
            peak_rss = max(peak_rss, rss)
        except Exception:
            break
        if have_smi:
            peak_vram = max(peak_vram, _vram_used_mb())
        time.sleep(interval)
    out["rss_mb"] = round(peak_rss / 1e6, 1)
    out["vram_mb"] = round(peak_vram, 1) if have_smi else None


def _has_nvidia_smi() -> bool:
    from shutil import which
    return which("nvidia-smi") is not None


def _vram_used_mb() -> float:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5)
        return max(float(x) for x in r.stdout.split() if x.strip())
    except Exception:
        return 0.0


def zip_size_mb(submit_dir: Path) -> float | None:
    z = submit_dir / "submit.zip"
    return round(z.stat().st_size / 1e6, 1) if z.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("SMOKE_N", "30000")))
    ap.add_argument("--src", default=str(ROOT / "data" / "train.jsonl"))
    ap.add_argument("--submit", default=os.environ.get("SMOKE_SUBMIT", str(ROOT / "submit")))
    ap.add_argument("--python", default=os.environ.get("SMOKE_PY", sys.executable))
    ap.add_argument("--sandbox", default=os.environ.get("SMOKE_SANDBOX", str(ROOT / "_smoke_sandbox")))
    args = ap.parse_args()

    submit_dir = Path(args.submit).resolve()
    sandbox = Path(args.sandbox).resolve()
    data_dir = sandbox / "data"
    out_dir = sandbox / "output"
    assert (submit_dir / "script.py").exists(), f"script.py 없음: {submit_dir}"

    print(f"[smoke] N={args.n} submit={submit_dir} python={args.python}")
    print(f"[build] {args.n}행 복제 → {data_dir} …")
    build_scaled_data(Path(args.src), data_dir, args.n)

    env = dict(os.environ)
    env["ENS_DATA"] = str(data_dir)
    env["ENS_OUT"] = str(out_dir)
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    env.setdefault("HF_HUB_OFFLINE", "1")

    stats: dict = {}
    stop = threading.Event()
    t0 = time.time()
    proc = subprocess.Popen([args.python, "script.py"], cwd=str(submit_dir), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    watcher = threading.Thread(target=poll_peak, args=(proc, stop, stats))
    watcher.start()
    internal_elapsed = None
    for line in proc.stdout:
        sys.stdout.write("    " + line)
        if "elapsed=" in line:
            try:
                internal_elapsed = float(line.split("elapsed=")[1].split("s")[0].strip())
            except Exception:
                pass
    proc.wait()
    stop.set()
    watcher.join()
    wall = time.time() - t0

    sub = out_dir / "submission.csv"
    ok_rows = passed = None
    bad = []
    if sub.exists():
        import csv
        with sub.open(encoding="utf-8") as f:
            r = csv.DictReader(f)
            fields = r.fieldnames
            rows = list(r)
        ok_rows = len(rows)
        valid = _valid_actions(submit_dir)
        bad = sorted({row["action"] for row in rows} - valid) if valid else []
        passed = (fields[:2] == ["id", "action"]) and (ok_rows == args.n) and not bad

    print("\n" + "=" * 60)
    print(f"[결과] exit={proc.returncode}")
    print(f"  벽시계          : {wall:.1f}s  (N={args.n})")
    if internal_elapsed is not None:
        print(f"  script elapsed  : {internal_elapsed:.0f}s")
        rate = args.n / max(internal_elapsed, 1e-6)
        print(f"  처리율          : {rate:.0f} rows/s → 10분 예산 상한 ≈ {rate*BUDGET_SEC:,.0f}행")
    print(f"  peak RSS        : {stats.get('rss_mb')} MB")
    print(f"  peak VRAM       : {stats.get('vram_mb')} MB (nvidia-smi)")
    print(f"  출력 행수        : {ok_rows} (기대 {args.n})")
    print(f"  submit.zip      : {zip_size_mb(submit_dir)} MB (한도 1024)")
    if bad:
        print(f"  ⚠️ 잘못된 라벨   : {bad}")
    print(f"  판정            : {'PASS' if passed else 'FAIL'}  "
          f"(exit0·행수정합·라벨유효; 시간판정은 T4 실측 기준)")
    sys.exit(0 if passed and proc.returncode == 0 else 1)


def _valid_actions(submit_dir: Path) -> set[str] | None:
    """features.ACTIONS를 읽어 라벨 유효성 검사(실패시 None=검사생략)."""
    try:
        sys.path.insert(0, str(submit_dir))
        import features as F  # type: ignore
        return set(F.ACTIONS)
    except Exception:
        return None
    finally:
        if str(submit_dir) in sys.path:
            sys.path.remove(str(submit_dir))


if __name__ == "__main__":
    main()
