---
name: day-start
description: 하루 시작 의식 — new_day.py로 데일리 로그·LB 스냅샷 생성, 직전 daily의 '내일' 섹션을 오늘 계획으로 이월. Trigger: /day-start
---

# 하루 시작

1. 사용자에게 오늘 LB 수치를 확인한다: 1등 / 12등(커트라인) / 우리 점수.
2. 실행: `python scripts/new_day.py --lb1 <1등> --lb12 <12등> --ours <우리>`
3. 직전 daily의 "내일" 섹션을 읽어 미완료 항목을 오늘 daily의 계획으로 이월한다.
4. 커트라인 갭 변화를 한 줄로 보고한다 — 갭이 벌어지고 있으면 전략 재점검 신호.
