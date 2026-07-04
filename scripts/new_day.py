"""데일리 게이트 — 하루의 시작. 오늘의 daily 로그를 생성하고 INDEX 타임라인에 등록한다.

Usage:
    python scripts/new_day.py --lb1 0.79015 --lb12 0.77585 [--ours 0.7xx]

이미 오늘 파일이 있으면 LB 스냅샷만 갱신 안내 후 종료 (멱등).
"""
import argparse
import datetime
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_DIR = os.path.join(ROOT, "context", "daily")
INDEX_PATH = os.path.join(ROOT, "context", "INDEX.md")
DEADLINE = datetime.date(2026, 7, 15)

TEMPLATE = """# {date} ({weekday}) — D-{dday} (예선 마감 07.15 09:59)

## LB 스냅샷

| 1등 | 12등 (커트라인) | 우리 |
|---|---|---|
| {lb1} | {lb12} | {ours} |

## 오늘의 실험 큐 (아침에 확정)

- [ ]

## 오늘 한 일 / 결과

-

## 배운 것 / 발견

-

## 결정

- (있으면 decisions.md에 항목 추가 후 여기 링크)

## 내일

- [ ]
"""

WEEKDAYS_KO = ["월", "화", "수", "목", "금", "토", "일"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lb1", default="?", help="리더보드 1등 점수")
    ap.add_argument("--lb12", default="?", help="리더보드 12등(커트라인) 점수")
    ap.add_argument("--ours", default="-", help="우리 팀 최고 점수")
    args = ap.parse_args()

    today = datetime.date.today()
    date_str = today.isoformat()
    path = os.path.join(DAILY_DIR, f"{date_str}.md")

    if os.path.exists(path):
        print(f"이미 존재: {path}")
        print("LB 스냅샷이 바뀌었으면 파일에서 직접 갱신하세요.")
        return

    dday = (DEADLINE - today).days
    content = TEMPLATE.format(
        date=date_str,
        weekday=WEEKDAYS_KO[today.weekday()],
        dday=dday,
        lb1=args.lb1,
        lb12=args.lb12,
        ours=args.ours,
    )
    os.makedirs(DAILY_DIR, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    print(f"생성: {path} (D-{dday})")

    # INDEX 타임라인 등록 (중복 방지)
    with open(INDEX_PATH, encoding="utf-8") as f:
        index = f.read()
    line = f"- {date_str} — [daily](daily/{date_str}.md) : "
    if date_str not in index:
        with open(INDEX_PATH, "a", encoding="utf-8", newline="\n") as f:
            f.write(line + "\n")
        print("INDEX.md 타임라인에 등록됨 — 한 줄 요약을 채우세요.")

    if args.lb1 == "?":
        print("경고: LB 스냅샷 없이 시작했습니다. --lb1 --lb12 로 기록하는 것을 권장.")


if __name__ == "__main__":
    main()
