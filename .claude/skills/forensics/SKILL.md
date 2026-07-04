---
name: forensics
description: 시뮬레이터 포렌식 라운드 실행 — train.jsonl에서 state→action 결정 규칙을 찾고 context/reports/forensics_rN.md로 기록한다. Trigger: /forensics
---

# 시뮬레이터 포렌식 라운드

train 데이터는 시뮬레이터(`sess_sim_*`) 출력이므로 (상태→행동) 결정 규칙이 존재할 수 있다.
각 라운드는 **가설 → 측정 → 규칙 후보 → 기록**으로 닫는다. 열린 채 끝내지 않는다.

## 절차

1. **라운드 번호 확정**: `context/reports/forensics_r*.md` 최신 번호 +1 = N.
2. **선행 확인**: 직전 라운드 리포트의 "다음 라운드 제안"과 `context/research.md`의 P0 항목을 읽고 이번 라운드 가설을 정한다. 팀 리포(`C:\dev\dacon-agent-action-api-boost`)의 `src/rules.py`·`notes/`와 중복 여부도 확인.
3. **분석 실행**: `scripts/analysis/`의 기존 스크립트(`common.py` 등)를 재사용·확장한다. 새 가설은 새 스크립트 파일로. 실행은 반드시 서버 미러 venv: `.venv\Scripts\python.exe`.
4. **함정 방지 (필수)**:
   - purity 버킷마다 **unique 세션 수**를 함께 산출 — 한 세션이 부풀린 버킷은 일반화되지 않는다.
   - 규칙 후보 평가는 **세션 프리픽스 group-split** 기준 coverage×purity로 (`id.rsplit("-step_", 1)[0]` = `src/features.py:session_id`).
   - respond_only·write_file은 이미 per-class F1 1.0 — 이 클래스 겨냥 규칙은 후순위. 우선 타깃은 탐색 4클래스(read_file/grep_search/list_directory/glob_pattern).
5. **산출**: `context/reports/forensics_rN.md` — 가설 / 방법(스크립트 경로) / 수치(coverage·purity 표) / 규칙 후보(적용 조건·기대 Macro-F1 이득·리스크) / 다음 라운드 제안.
6. **기록 연결**: `context/research.md`에 리포트 링크 추가, 오늘 `context/daily/`에 한 줄 요약.

## 서브에이전트 사용 시

- model은 **Sonnet 5 고정** (CLAUDE.md 규칙).
- 병렬 실행은 가설 단위로 분할하고, 산출 파일 경로가 서로 겹치지 않게 명시 지정한다.
