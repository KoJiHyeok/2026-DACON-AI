# task3 — 본선 발표 사료 컴파일 (전문가심사 40% 대비)

## 컨텍스트

DACON 236694. 본선(08.11) 심사 산식 = 예선 Private 50% + 추론속도 10% + **전문가심사 40% (데이터분석 10 · 모델검증 10 · 알고리즘 15 · 전달력 5)**. context/에는 실험 51건, 결정 D-001~014, 데일리 9일, 포렌식·감사 리포트가 쌓여 있지만 발표용으로 구조화돼 있지 않다. 이 작업은 **기록 → 심사 산식 매핑 사료집**을 만든다. 창작 금지 — 모든 수치·주장은 기록 인용으로만.

## 목표 / 완료 조건 (DoD)

1. `docs/presentation/sources.md` — 심사 4항목별 섹션:
   - **데이터분석(10)**: 포렌식 라운드 발견(시뮬레이터 규칙·세션 구조·라벨 생성 규칙 추정), 클래스 분포·세션 통계, hist12 서사(D-010 재심 — "83% 잘림" 기각 근거)
   - **모델검증(10)**: GroupKFold 누수 방지 설계, 5지표 게이트 체계(row/세션균등/MC/bootstrap/반반), 작성자-검증자 분리(reviewer/tester 이중검증 사례), holdout→LB 전이 할인 실측(#34 67%, #51 비전이 — CI가 경고를 맞힌 사례)
   - **알고리즘(15)**: 5성분 이종 앙상블 구조(linear/AAR/e5-hist12/mBERT/soft-AU α0.9, per-encoder serialize), 승격 서사(0.7188→0.7623 — 대장 #1~#11), 폐기 서사(51실험 중 승격 4건 — 시드·백본·스태커·증류가 어떻게 실측으로 종결됐나)
   - **전달력(5)**: 발표 스토리라인 초안(문제→탐색→체계→결과), 그림/표 후보 목록(무엇을 그릴지 + 원천 데이터 위치)
2. **인용 규율**: 모든 수치·주장 옆에 출처 표기 — `(exp #34)`, `(D-013)`, `(대장 #11)`, `(reports/forensics_r2.md)` 형식. 출처 없는 문장 금지.
3. `docs/presentation/key_numbers.md` — 발표에 쓸 핵심 수치 단일표 (LB 궤적, 성분별 solo, 게이트 수치, 실험 통계) — 전부 출처 포함.
4. **자기 검증**: 완성 후 임의 20개 인용을 골라 원본 대조한 스팟체크 결과를 report에 표로 남긴다 (불일치 0건이어야 PASS — 발견 시 수정 후 재체크).
5. `context/night/2026-07-13/report_presentation.md` — 작업 요약, 스팟체크 표, 기록 공백(발표에 필요한데 context/에 없는 것) 목록. **파일명 task 시작 금지.**
6. `context/night/2026-07-13/task3.DONE` 생성 (요약 포함).

## 재료 (전부 워크트리 내 추적 파일 — 읽기 전용)

- `context/experiments.md`, `context/decisions.md`, `context/submissions.md`, `context/daily/**`, `context/reports/**`, `context/coordination.md`, `context/night/**` (기존 리포트들)
- `PLAN.md`, `docs/**`
- 파이썬 (인용 스팟체크 보조용): `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 네트워크 코드 금지
- **canonical context 수정 금지** — `context/experiments.md` 등은 읽기만, 산출물은 `docs/presentation/`과 night 폴더에만
- 수치 창작·보정 금지 — 기록에 없는 수치는 '기록 공백' 목록으로 보낼 것
- task1(`scripts/aar_speed`)·task2(`scripts/repro_rehearsal`, `docs/repro_playbook.md`) 경로 침범 금지

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-13/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터
2. 섹션 단위마다 PROGRESS 갱신 + **git commit** (실패 시 사유 기록)
3. 전부 끝나면 `task3.DONE` + 최종 커밋

## 작업 내용 (단계)

1. PROGRESS 생성 → 기록 전체 통독 (experiments 51행·decisions 14건·daily·reports)
2. 심사 4항목 × 기록 매핑 표 초안 → sources.md 섹션별 집필 (인용 규율 준수)
3. key_numbers.md 집계 → 20건 인용 스팟체크 → 불일치 수정
4. report_presentation.md (기록 공백 목록 포함) → task3.DONE
