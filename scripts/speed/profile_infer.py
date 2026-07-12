import json
import os
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from scripts.speed.stages import run
else:
    from .stages import run


def main():
    result = run()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    output = os.environ.get("SPEED_JSON_OUT", "")
    if output:
        Path(output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
