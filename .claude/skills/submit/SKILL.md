---
name: submit
description: 게이트 통과 제출 — make_submit.py 경유로 테스트·패키징·12항목 검증·제출 대장 기록까지 한 번에. 수동 zip 금지. Trigger: /submit
---

# 제출 게이트

수동 zip 금지 (D-002, D-005). 항상 `scripts/make_submit.py` 경유.

## 절차

1. **사전**: 로컬 CV 값과 변경 요약(note)을 확보한다. 없으면 사용자에게 묻는다.
2. **git clean 확인**: dirty면 커밋부터 — 제출물은 커밋과 1:1 대응이 원칙.
3. **실행**: `.venv\Scripts\python.exe scripts\make_submit.py --cv <값> --note "<변경 요약>"`
4. **게이트 실패 시** 순서대로 해결하고 재실행: G1 tests → G2 git clean → G3 패키징 → G4 validate_submit(12검증) → G5 대장 기록. 게이트 우회 금지 (`--allow-dirty`는 개발 중 검증 전용 — 실제 제출에 쓰지 않는다).
5. **업로드 (G6)**: 공식 제출 API로 직접 업로드할 수 있다 (2026-07-10 도입).
   - 사전 확인 (제출 소모 없음): `.venv\Scripts\python.exe scripts\dacon_submit.py --check` — 토큰·팀·오늘 남은 횟수·용량 한도 확인.
   - 실제 업로드: `.venv\Scripts\python.exe scripts\dacon_submit.py --file <zip> --memo "<exp# 요약>" --yes`
   - `--yes`를 붙이기 전에 반드시 사용자에게 "이 제출이 무엇을 검증하는가" 한 줄과 함께 업로드 여부를 확인받는다. 자격 정보는 사용자 환경변수 `DACON_TOKEN`/`DACON_TEAM` (리포·대화에 토큰 노출 금지). API 실패 시 폴백 = 웹사이트 수동 업로드.
6. **업로드 후** LB 점수가 나오면: `context/submissions.md` 해당 행에 LB 기입, `context/experiments.md`에 실험 행 추가, CV→LB 할인율 테이블 갱신 여부 판단.

## 주의

- 일 제출 10회 예산 — 제출 전에 "이 제출이 무엇을 검증하는가"를 한 줄로 말할 수 있어야 한다.
- 팀 리포(`dacon-agent-action-api-boost`) push는 LB > 0.7208일 때만.
