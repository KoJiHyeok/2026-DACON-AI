# task3 — forensics R2 후속: 고순도 템플릿 오버라이드 후보 정밀 선별

## 컨텍스트

DACON 236694. `context/reports/forensics_r1.md`의 R2 규칙 — "current_prompt 정규화 템플릿 purity≥0.99 (비-respond_only 2,606행/3.72%, 템플릿당 2~11행)" — 은 **폐기가 아니라 보류**였다: "고빈도·의미 명확한 소수 템플릿만 골라 별도 LB 프로브" 권고. 당시엔 리그가 없어 판정 수단이 부족했지만, 지금은 holdout 리그로 오버라이드의 실효를 LB 소모 없이 실측할 수 있다. 시뮬레이터 생성 데이터라 결정적 템플릿이 존재할 개연성이 있다.

## 목표 / 완료 조건 (DoD)

1. `scripts/tmpl_override/mine.py` — train.jsonl에서 current_prompt 정규화(소문자·공백 정리·숫자/경로/식별자 마스킹 — r1의 정규화 방식을 forensics_r1.md에서 확인해 재현) → 템플릿별 (빈도, 지배 action, purity) 표. **holdout 9,969행을 제외한 60,031행으로만** purity 계산 (누수 방지 — 평가 대상 행으로 규칙을 만들면 안 됨).
2. `scripts/tmpl_override/judge.py` — 선별 기준(비holdout 빈도 ≥ 20 & purity ≥ 0.995 & 지배 action이 현행 blend와 다른 행이 holdout에 실존)을 통과한 템플릿만: holdout에서 매칭 행에 오버라이드 적용 → 4-way+soft-AU 최종 리그 델타. 템플릿별 개별 델타와 전체 합산 델타를 표로.
3. 안전 분석: 오버라이드가 **틀리게 바꾼 행**(fixed vs broken) 카운트 — 동료 fix의 net_gain 91/1913 같은 아슬아슬한 규칙인지, 확실한 규칙인지 구분.
4. `context/night/2026-07-08/report_tmpl_override.md` — 표 + 판정 (게이트 +0.005 = LB 후보 / +0.002~0.005 = 보고 / 미달 = R2 최종 폐기 선언). 후보가 있으면 script.py 통합 방식(추론 시 정규화 매칭 사전) 스케치 포함. (**파일명 task로 시작 금지**)
5. `context/night/2026-07-08/task3.DONE` (한 줄 요약) + 최종 커밋.

## 재료 (절대 경로)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (시스템 파이썬 금지)
- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv` (읽기 전용)
- 선행 분석: `context/reports/forensics_r1.md` (R2 절 — 정규화 방식·purity 계산의 정본)
- 리그 조인: `scripts/league4/common.py` (커밋됨) — sanity 자동 검증 + soft-AU 재사용. 기준선 B4+softAU = **0.73877**
- holdout ids: `context/night/2026-07-05/holdout_base.npz` (커밋됨)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출 금지, submit/ 수정 금지
- **holdout 행을 템플릿 마이닝/purity 계산에 사용 금지** (누수 — 반드시 비holdout 60,031행만)
- threshold/prior/calib 가족(D-009)과 혼동 금지 — 이것은 확률 보정이 아니라 결정적 규칙 오버라이드다. 단 판정 문턱은 동일하게 적용
- sess_au 행은 오버라이드 대상에서 제외 (AU 라우팅과 이중 개입 금지)
- GPU 금지

## 진행 프로토콜 (재개 대비)

1. 시작 시 `context/night/2026-07-08/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(정규화 재현 / 마이닝 표 / judge / 리포트)마다 PROGRESS 갱신 + **git commit**
3. 끝나면 `task3.DONE` + 최종 커밋

## 작업 내용

1. forensics_r1.md의 R2 정규화를 코드로 재현 — r1 수치(2,606행/3.72%, purity 0.99+)와 대략 일치하는지 교차 확인 (전체 train 기준으로 재현 확인 후, 실제 마이닝은 비holdout만).
2. 선별 기준 통과 템플릿 목록 → PROGRESS에 기록.
3. judge: holdout 매칭 행 오버라이드 → 리그 델타 + fixed/broken 분해. blend가 이미 맞히는 행을 바꾸는 템플릿은 즉시 제외.
4. 리포트 작성. 후보 없으면 'R2 최종 폐기'를 명확히 선언 (r1의 보류 상태를 종결하는 것도 성과다).
