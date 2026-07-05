# task5 — first-step(hist=0) 서브모집단 라우팅 프로브 (분석 우선, CPU 전용)

## 컨텍스트

DACON 236694, 14클래스 Macro-F1. task3에서 "**결정적 키로 판별되는 서브모집단**에 전용 모델을 라우팅"하는 패턴이 처음으로 생존했다 (AU: id 프리픽스, 리그 +0.0094). 이 패턴의 두 번째 후보가 **first-step 행**(history가 빈 리스트, holdout 9,969행 중 1,293행 ≈ 13%)이다 — 판별이 결정적(파생 신호 아님)이라는 점에서 R4/meta-selector 실패와 다르고 AU와 같다.

⚠️ 단 AU와 결정적으로 다른 점: **first-step은 인코더가 가장 강한 구간**이다 (07-05 버킷 분석에서 hist_0의 최적 가중이 enc 몰빵 [0.75,0.5,2]였음). 전용 linear로 blend를 교체하면 그 행에서 enc 기여가 0이 되므로, 리그의 enc 지분 신기루(#16/#19/#20: 리그는 enc 제거를 과대평가, LB에서 3회 부호 반전)가 **AU 때보다 훨씬 강하게 작동할 가능성**이 크다. 그래서 이 태스크는 분석 우선이고, PASS 문턱도 높다.

## 목표 / 완료 조건 (DoD)

1. **분석**: holdout hist_0 1,293행에서 성분별(linear/stacker/encoder-proxy/blend) macro-F1 — AU처럼 "전 성분 공통 약세"인가, 아니면 enc만 강한가? (이게 갈림길: 전 성분이 hist_0에서 강하면 specialist 여지 없음 → 즉시 FAIL 종결 가능)
2. 분석에서 여지가 보일 때만 **프로브**: hist_0 전용 linear (train의 history=0 행 ~9,000개로 학습, 세션 Group 3-fold, **holdout 행 혼입 금지** — 학습은 비holdout hist_0 행만) → holdout hist_0 라우팅/soft 결합 리그 평가
3. **판정 (높은 문턱)**: 리그 전체 델타 **+0.005 이상**이어야 PASS (enc 제거 신기루 할인 명목 — AU의 +0.002 문턱보다 높음). 미달·애매는 전부 FAIL
4. 리포트 `context/night/2026-07-06/report_task5.md` (파일명 task로 시작 금지) — FAIL이어도 성분별 hist_0 표는 남길 것 (내일 판단 재료)
5. `context/night/2026-07-06/task5.DONE` + 최종 커밋

## 재료 (절대 경로)

- task4.md의 재료 절과 동일 (데이터·리그 npz·OOF·조인 코드·파이썬). 3-way 0.717259 재현 assert 필수
- hist_0 판별: `len(sample.get('history') or []) == 0` (id가 아니라 history 필드 — test에서도 동일하게 판별 가능)
- serialize: `submit/au_route.py`의 것을 기본으로 (hist_0 행은 history 파트가 비므로 사실상 meta+prompt만 남음 — 필요시 first-step 특화 직렬화 변형 1개까지 허용, 반드시 별도 표기)

## 금지

- task4.md의 금지 절과 동일 (워크트리 밖 수정·push·submit/ 수정·네트워크·enc 지분 튜닝·prior 보정 금지)
- 파일 경로 비중첩: `scripts/firststep/` 사용 (scripts/au*, scripts/components는 건드리지 말 것)

## 진행 프로토콜 (재개 대비)

1. `context/night/2026-07-06/PROGRESS-task5.md` 확인 — 있으면 이어서
2. 의미 단위(분석 → 프로브 → 리포트)마다 부분 저장(`night_out/task5/`) + PROGRESS + git commit
3. 끝나면 `task5.DONE` + 최종 커밋

## 작업 내용

1. `scripts/firststep/analyze.py` — holdout hist_0 성분별·클래스별 F1 표 + train hist_0 라벨 분포 (첫 스텝은 plan_task/read_file 쏠림 예상 — 실측).
2. 여지가 있으면 `scripts/firststep/probe_firststep.py` — 격리 규약 학습 → 라우팅/soft 그리드 (α ∈ {0.5, 0.7, 1.0}).
3. 판정 기준 적용해 리포트 + DONE. FAIL이면 사유(전 성분 강세 / 마진 부족 / 신기루 의심)를 명시.
