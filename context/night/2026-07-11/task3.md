# task3 — 본선 전문가심사 대비: 모델 검증·재현성 문서 초안

## 컨텍스트

DACON 236694 본선 산식의 40%가 전문가심사다(데이터분석 10 + 모델검증 10 + 알고리즘 15 + 전달력 5). 우리의 검증 체계(5지표 리그 게이트, 세션 group split, 작성자·검증자 분리, 재시도 금지 테이블)와 실험 여정(#0~#49)은 context/에 흩어져 있다 — 심사 관점으로 재구성한 초안을 만들어두면 본선 진출 시 발표 자료의 뼈대가 되고, 탈락해도 회고 자산이 된다. **report-only 작업이며 코드 산출물이 없다.**

## 목표 / 완료 조건(DoD)

1. `docs/finals/analysis_draft.md` (데이터분석 10점 대응): 데이터 구조(70,000행·14클래스·세션 시뮬레이터 기원), 클래스 불균형과 Macro-F1 함의, 포렌식 라운드(r1·r2)에서 확인된 데이터 성질(결정 규칙 부재, sim/au 이질성, 형제 행 라벨 구조 D-008), AU 서브모집단 발견 스토리(#23~24, +0.021 LB).
2. `docs/finals/validation_draft.md` (모델검증 10점 대응): 세션 프리픽스 group split 원칙(누수 사례와 방어), 5지표 리그 게이트(row/세션균등/MC/bootstrap/반반)의 설계 이유, CV→LB 할인율 실측 테이블, 리그 신기루 사례(#19~20)와 그로부터 만든 규칙, 작성자·검증자 분리(reviewer/tester 이중검증) 운영.
3. `docs/finals/algorithm_draft.md` (알고리즘 15점 대응): 챔피언 아키텍처(4-way blend: linear+AAR stacker+e5-hist12+mBERT, per-encoder serialize, soft-AU α0.9), 각 성분의 기여 근거(exp 번호 인용), 폐기된 대안들이 보여주는 설계 공간 탐색(백본 대형화·시드 앙상블·스태커 대체·OOF 보정 전부 게이트 미달 — 왜 이 구성이 국소 최적인가), T4/10분/1GB 제약 하 설계 결정.
4. 각 문서는 (a) 모든 수치에 근거 인용(exp #, D-00x, 대장 #)을 달고, (b) context에 없는 수치를 창작하지 않으며, (c) 2~4쪽 분량의 개조식 초안으로.
5. **`context/night/2026-07-11/task3.DONE` 생성(3줄 요약 포함)**.

## 재료 (절대 경로 — 전부 워크트리 안에 있음, git 추적 파일)

- `context/experiments.md` (실험 #0~#49 + 재시도 금지 테이블 + 할인율)
- `context/decisions.md` (D-001~D-013), `context/coordination.md` (감사 노트들)
- `context/reports/forensics_r1.md`, `forensics_r2.md`, `context/research.md`
- `context/submissions.md` (제출 대장), `context/handoffs/codex/CX-001~003.md`
- `PLAN.md`, `docs/server_setup.md`
- 파이썬 불필요 (문서 작업)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지.
- **context/의 원본 기록 수정 금지** — docs/finals/ 신규 파일만 생성.
- 수치 창작 금지 — context에서 찾을 수 없는 수치는 `[확인 필요]`로 표기.

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-11/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터.
2. 문서 1편 완성마다 PROGRESS 갱신 후 **git commit**.
3. 전부 끝나면 `task3.DONE` 생성 + 최종 커밋.

## 작업 내용

1. context/ 재료를 통독하고 심사 기준 4축(데이터분석·모델검증·알고리즘·전달력)에 매핑되는 사실 목록을 먼저 뽑는다.
2. analysis → validation → algorithm 순으로 초안 작성 (각 편 완성 시 커밋).
3. 마지막에 세 문서의 교차 참조·인용 일관성을 점검하고 DONE.
