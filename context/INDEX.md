# Context Index — 모든 기록의 진입점

> 원칙: **기록이 없으면 일어나지 않은 것이다.** 모든 리서치·실험·결정·제출은 이 폴더에 남는다.
> 이 기록이 본선(08.11) 발표 자료의 원천이 된다.

## 문서 지도

| 문서 | 내용 | 갱신 주체 |
|---|---|---|
| [decisions.md](decisions.md) | 의사결정 기록 (결정·이유·버린 대안) | 결정이 날 때마다 |
| [experiments.md](experiments.md) | 실험 로그 (가설→변경점→CV→LB) | train.py 자동 + 수기 |
| [research.md](research.md) | 리서치 허브 (큐·피처 백로그), 깊은 조사는 [research/](research/) | 리서치할 때마다 |
| [submissions.md](submissions.md) | 제출 대장 (커밋·검증·점수) | make_submit.py 자동 |
| [daily/](daily/) | 데일리 로그 (LB 스냅샷·실험 큐·결과·내일) | scripts/new_day.py + 수기 |
| [reports/](reports/) | EDA·페이즈 리포트 | 페이즈 완료 시 |

## 하루 리듬 (게이트)

1. **아침**: `/day-start` 스킬 경유 권장 (LB를 WebFetch로 직접 조회) — 수동 실행 시 `python scripts/new_day.py --lb1 <1등> --lb12 <12등> --ours <우리>` → 오늘 daily 생성, 실험 큐 확정
2. **실험마다**: 학습 → CV 결과 experiments.md 기록
3. **제출마다**: `python scripts/make_submit.py` → tests·git clean·12개 검증 통과 시에만 zip + submissions.md 자동 기록
4. **밤**: daily 마감(결과·결정·내일 계획), decisions.md 갱신, **git push**

## 타임라인

- 2026-07-04 — [daily](daily/2026-07-04.md) : 대회 분석, 리포·검증 하네스·기록 시스템 구축
- 2026-07-05 — [daily](daily/2026-07-05.md) : 밤샘 3작업 완주(R4 계층 +0.0246 생존), w112 원본 인코더 발견·fp16 변환, 3-way 재조립 제출 준비
- 2026-07-06 — [daily](daily/2026-07-06.md) : AU 라우팅 발견·2연속 승격 (exp #23~24, LB 0.7331 → **0.7400** 팀 최고 갱신), first-step 라우팅 폐기
- 2026-07-07 — [밤샘 3작업](night/2026-07-07/) : league4 리그 정비·mate 평가·subroute — report_league4/mate_eval/subroute.md
- 2026-07-09 — [밤샘 오답 분석](night/2026-07-09/report_errtax.md) : 구챔피언 오답 택소노미 1차 (이후 CX-003으로 대체)
- 2026-07-10 — [daily](daily/2026-07-10.md) : LB 0.7623 팀 최고 탈환(e5 hist12), 감사 반영 D-011 args-lite 개시 — 새 세션 인계사항 포함
- 2026-07-10 — [제3자 SOL 모델 감사](reports/third_party_sol_model_audit_2026-07-10.md) : 활성 0.7623 모델의 검증 표본단위·고정 홀드아웃·직렬화 누락을 재진단하고, args-lite·hist12-aware stacker 중심의 점수 개선 우선순위 제시
- 2026-07-11 — 판정 집중일 (daily 없음 — experiments #44~49·decisions D-013·coordination 참조) : 도약 3축 전부 폐기(시드 앙상블 #48·H1 규칙 [forensics r2](reports/forensics_r2.md)·klue-large #49), CX-003 이중검증 승격, **D-013 챔피언 0.7623 자동 최종 확정**
- 2026-07-12 — [daily](daily/2026-07-12.md) : 밤샘 3작업 회수(러너 자기치유 실증·커밋은 아침 구조), task3 문서 병합, task1·2 리뷰 FAIL 재작업 — 원본 AAR 아티팩트 실존 발견. 컷 갭 −0.0305로 악화
