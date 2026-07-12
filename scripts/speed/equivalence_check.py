"""Compare the harness result with an unmodified submit/script.py subprocess."""
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.speed.common import MODEL, PYTHON, load
    from scripts.speed.stages import _pipeline
else:
    from .common import MODEL, PYTHON, load
    from .stages import _pipeline


def _write_input(samples, directory):
    data = Path(directory) / "test.jsonl"
    sample = Path(directory) / "sample_submission.csv"
    data.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in samples),
                    encoding="utf-8")
    with sample.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "action"])
        writer.writerows([[row["id"], ""] for row in samples])
    return data, sample


def run_check(n=None, device=None, model=None):
    if __package__ in (None, ""):
        from scripts.speed.common import env_int, mod, setup
    else:
        from .common import env_int, mod, setup
    n = env_int("SPEED_ROWS", 300) if n is None else int(n)
    device = (device or os.environ.get("SPEED_DEVICE", "cpu")).lower()
    model = Path(model or os.environ.get("SPEED_MODEL_DIR", str(MODEL)))
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    module = mod()
    setup(module, model, device)
    samples = load(n=n)
    expected, _ = _pipeline(module, samples)
    with tempfile.TemporaryDirectory(prefix="speed-equivalence-") as tmp:
        _write_input(samples, tmp)
        env = os.environ.copy()
        env.update({"ENS_DATA": tmp, "ENS_OUT": tmp,
                    "ENS_LINEAR_PKL": str(model / "linear" / "model.pkl"),
                    "ENS_STACKER_DIR": str(model / "stacker"),
                    "ENS_ENCODER_DIR": ",".join(str(p) for p in sorted(
                        model.glob("encoder*")) if p.is_dir())})
        code = "import script; script.main()"
        proc = subprocess.run([PYTHON, "-c", code], cwd=str(model.parent), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, check=False)
        if proc.returncode:
            raise RuntimeError("reference script.py failed:\n" + proc.stdout[-4000:])
        with (Path(tmp) / "submission.csv").open(newline="", encoding="utf-8") as handle:
            actual = [row["action"] for row in csv.DictReader(handle)]
    mismatches = [(samples[i].get("id", ""), expected[i], actual[i])
                  for i in range(min(len(expected), len(actual)))
                  if expected[i] != actual[i]]
    if len(actual) != len(expected):
        mismatches.append(("<row-count>", str(len(expected)), str(len(actual))))
    return {"rows": len(samples), "matched": len(mismatches) == 0,
            "match_rate": 1.0 if not mismatches else 1 - len(mismatches) / len(expected),
            "mismatches": mismatches}


def main():
    result = run_check()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["matched"] else 1)


if __name__ == "__main__":
    main()
