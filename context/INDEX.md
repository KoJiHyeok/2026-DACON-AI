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

1. **아침**: `python scripts/new_day.py --lb1 <1등점수> --lb12 <12등점수>` → 오늘 daily 생성, 실험 큐 확정
2. **실험마다**: 학습 → CV 결과 experiments.md 기록
3. **제출마다**: `python scripts/make_submit.py` → tests·git clean·12개 검증 통과 시에만 zip + submissions.md 자동 기록
4. **밤**: daily 마감(결과·결정·내일 계획), decisions.md 갱신, **git push**

## 타임라인

- 2026-07-04 — [daily](daily/2026-07-04.md) : 대회 분석, 리포·검증 하네스·기록 시스템 구축
- 2026-07-05 — [daily](daily/2026-07-05.md) : 밤샘 3작업 완주(R4 계층 +0.0246 생존), w112 원본 인코더 발견·fp16 변환, 3-way 재조립 제출 준비
- 2026-07-06 — [daily](daily/2026-07-06.md) : 
