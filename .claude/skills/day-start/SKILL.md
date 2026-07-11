---
name: day-start
description: 하루 시작 의식 — new_day.py로 데일리 로그·LB 스냅샷 생성, 직전 daily의 '내일' 섹션을 오늘 계획으로 이월. Trigger: /day-start
---

# 하루 시작

1. LB 수치를 **WebFetch로 직접 조회**한다 (로그인 불필요, 공개 HTML — 2026-07-10 실측 검증):
   - URL: `https://dacon.io/competitions/official/236694/leaderboard`
   - prompt로 1등 점수·팀명, 12등(커트라인) 점수·팀명을 추출한다.
   - WebFetch는 15분 캐시 — 하루 시작 시점 1회 조회 값을 그대로 쓴다. 검색엔진 캐시 값 금지, 반드시 페이지 직접 조회 값 사용.
   - 조회 실패(사이트 개편·차단 등) 시에만 사용자에게 수치를 묻는다.
2. 우리 점수는 `context/submissions.md`(제출 대장)의 최신 LB 실측 점수를 쓴다. 대장에 없으면 사용자에게 확인.
3. 실행: `python scripts/new_day.py --lb1 <1등> --lb12 <12등> --ours <우리>`
4. 직전 daily의 "내일" 섹션을 읽어 미완료 항목을 오늘 daily의 계획으로 이월한다.
5. 커트라인 갭 변화를 한 줄로 보고한다 — 갭이 벌어지고 있으면 전략 재점검 신호.
