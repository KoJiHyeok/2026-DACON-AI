# task4 — AU 라우팅 심화: 하드 교체를 넘어서 (CPU 전용)

## 컨텍스트

DACON 236694, 14클래스 Macro-F1. 어젯밤 task3에서 **AU(sess_au) 하드 라우팅**이 리그 +0.00935(오염 격리 재검증 +0.01143)로 생존, 현재 LB 게이트 제출 대기 중이다 (exp #23, 아침 회수 부록 = `context/night/2026-07-06/report_task3.md`). 이 태스크는 그 위를 노린다: **하드 교체가 최선인가?**

현재 구현(제출 스테이징): `submit/au_route.py`(serialize 단일 소스) + `submit/model/au_linear/model.pkl`(AU 5,025행 full-train, FeatureUnion word 1-2 80k + char_wb 3-5 120k, LinearSVC C=0.5 balanced) + `submit/script.py`의 `au_route_override()`(argmax 교체).

## 목표 / 완료 조건 (DoD)

1. 리그에서 아래 변형들을 하드 라우팅 베이스라인(**0.726613**, task3 probe 재현치)과 비교:
   - **soft 결합**: AU 행에서 `α·P_au + (1-α)·P_blend` (α ∈ {0.5, 0.7, 0.9, 1.0}) — P_au는 decision_function softmax
   - **AU 모델 강화**: C ∈ {0.25, 0.5, 1.0} × (word+char 현행 / char만 / word만) 그리드 — AU OOF 기준 (3-fold Group, task3와 동일 규약, **holdout 행 혼입 금지** — 아침 부록에서 지적된 결함 재발 금지: 학습은 비holdout AU 행만, 평가는 holdout AU 682행)
   - **sim 데이터 활용**: 전체 70k 학습 + AU 행 sample_weight 상향(×5, ×10) 모델이 AU 전용 모델을 이기는지
2. **판정**: 최고 변형이 하드 라우팅 대비 **+0.002 이상**이면 PASS(제출 승격 후보), 미달이면 "하드 교체 유지"로 FAIL 처리 — 애매하면 FAIL
3. per-class 진단: AU 682행에서 어느 클래스가 개선/악화되는지 표 (희소 클래스 n≤14는 별도 표기)
4. 리포트 `context/night/2026-07-06/report_task4.md` (파일명 **task로 시작 금지** — 러너 glob 함정)
5. `context/night/2026-07-06/task4.DONE` (판정 요약 3줄) + 최종 커밋

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` + `data\train_labels.csv` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 리그 재료·조인 코드: `C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz` + `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` — 조인 패턴은 메인 리포 `scripts/au/probe_au_linear.py`와 `context/night/2026-07-06/task1.md`(git log로 열람 가능한 이전 버전) 참고. 3-way [1,1,2] = 0.717259 재현 assert 필수
- serialize 단일 소스: `C:\dev\2026-AI-DACON\submit\au_route.py` (읽기 전용 — 변형 실험에서도 이 serialize를 기본으로, 바꾸는 변형은 별도 표기)
- 기존 probe: `scripts/au/probe_au_linear.py`, `scripts/au/common.py` (워크트리에 포함됨)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출물(submit/) 수정 금지 (읽기만), 네트워크 코드 금지
- **enc 지분/blend 가중 튜닝 금지** — SIM 행의 blend는 [1,1,2] 고정. AU 행 내부의 α 결합만 허용 (라우팅 축)
- class prior/bias 사후 보정 금지 (calib_v1 유형)
- ⚠️ 리그 해석 주의: AU 라우팅 계열은 enc 지분 신기루(#16/#19/#20)와 구조 유사 — 리그 델타는 크게 할인해 읽고, 판정 기준(+0.002)을 넘어도 "LB 게이트 필수"를 리포트에 명기

## 진행 프로토콜 (재개 대비)

1. 시작 시 `context/night/2026-07-06/PROGRESS-task4.md` 확인 — 있으면 이어서
2. 변형 그리드 한 축 끝날 때마다 부분 결과 json 저장(`night_out/task4/`) + PROGRESS 갱신 + git commit
3. 끝나면 `task4.DONE` + 최종 커밋

## 작업 내용

1. task3 probe의 조인·평가 코드를 `scripts/au2/`로 복제·정리 (경로 비중첩: scripts/au/는 수정 금지).
2. 완전 격리 규약으로 AU 모델 재학습 파이프라인 구성 (학습=비holdout AU, 평가=holdout AU 682행).
3. 변형 그리드 실행 → 표 → 판정.
4. 리포트 + DONE.
