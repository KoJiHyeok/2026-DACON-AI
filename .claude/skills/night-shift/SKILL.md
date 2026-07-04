---
name: night-shift
description: 밤샘 Codex 작업 준비 — 오늘 상태에서 병렬 밤샘 작업(2~3개)을 선정해 context/night/<date>/task*.md 프롬프트를 생성하고, 중단 재개 러너(scripts/night_shift.ps1) 가동을 안내한다. Trigger: /night-shift
---

# 밤샘 Codex 작업 준비

목표: 사용자가 자는 동안 Codex 멀티 세션이 안전하게 일하고, **끊겨도 스스로 재개**되게 한다.
Codex 토큰·사용량 한도로 세션이 중간에 죽는 것을 전제로 설계한다.

## 절차

1. **상태 수집**: 최신 `context/daily/*.md`, `context/experiments.md`(재시도 금지 테이블 포함), `context/research.md`, 진행 중 산출물을 확인한다.
2. **작업 선정 (2~3개)** — 전부 만족해야 한다:
   - 서로 파일 경로가 겹치지 않는다 (병렬 워크트리에서 아침 병합 시 충돌 없음)
   - GPU·LB 제출 없이 로컬 CPU만으로 완료·검증 가능
   - 아침에 PASS/FAIL 판정 가능한 명시적 완료 조건(DoD) 존재
   - 폐기 목록(재시도 금지 테이블) 위반 없음
3. **프롬프트 생성**: `context/night/<YYYY-MM-DD>/taskN.md` — 아래 필수 섹션을 지킨다.
4. **가동 안내** (사용자가 자기 전에 실행):
   ```powershell
   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\night_shift.ps1 -Register
   ```
   `-Register` = 30분 간격 자기치유 스케줄러 — 러너 자체가 죽거나 Codex 사용량 한도로
   재시도가 소진돼도 다시 살린다 (사용량 한도는 시간이 지나면 풀리므로 재기동이 유효).
   전원 연결 + 절전 꺼짐(`설정 > 전원`)을 사용자에게 상기시킨다.
   중간 확인: `night_shift.ps1 -Status`, 로그: `context/night/<date>/_runner/`.
5. **아침 회수**: `night/<date>/<task>` 브랜치별로 reviewer·tester 서브에이전트 검증
   (작성자·검증자 분리 규칙 — Codex가 작성자다) → 병합 → experiments.md·daily 기록 →
   워크트리 정리(`git worktree remove`).

## taskN.md 필수 섹션

- **컨텍스트** — 대회 한 줄 + 왜 이 작업인지
- **목표 / 완료 조건(DoD)** — 마지막 항목은 반드시: `context/night/<date>/taskN.DONE` 파일 생성(요약 포함). 러너의 완료 판정 기준이다.
- **재료 (절대 경로)** — 워크트리에는 gitignore된 것(data/, .venv, 미커밋 산출물)이 없다:
  - 데이터: `C:\dev\2026-AI-DACON\data\` (읽기 전용)
  - 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)
  - 팀 리포: `C:\dev\dacon-agent-action-api-boost` (읽기 전용 — 필요 파일은 워크트리로 복사)
- **금지** — 워크트리 밖(메인 리포 작업트리·팀 리포) 수정 금지, `git push` 금지, 수동 zip 제출 금지, 제출물에 네트워크 코드 금지, 폐기 목록 재시도 금지
- **진행 프로토콜 (재개 대비 — 핵심)**:
  1. 시작하자마자 `context/night/<date>/PROGRESS-taskN.md` 확인 — 있으면 '다음 재개 지점'부터 이어서
  2. 의미 단위 작업마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit**
  3. 전부 끝나면 `taskN.DONE` 생성 + 최종 커밋
- **작업 내용** — 단계별 지시

## 러너 동작 요약 (scripts/night_shift.ps1)

- task*.md마다 git worktree(`C:\dev\night\<date>\<task>`, 브랜치 `night/<date>/<task>`)를 만들고 `codex exec`를 병렬 실행
- DONE 없이 프로세스가 죽으면 백오프(1→2→4→…→30분) 후 재개: 로그의 세션 ID로 `codex exec resume` 우선, 못 찾으면 PROGRESS 기반 새 세션
- Windows에서 codex 샌드박스 오류 시: `-CodexArgs "--dangerously-bypass-approvals-and-sandbox"`
