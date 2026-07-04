# 2026 DACON — AI·SW중심대학 디지털 경진대회 : AI부문

> **AI Agent 행동(Action) 의사결정 예측 챌린지**
> https://dacon.io/competitions/official/236694

AI 코딩 에이전트 세션의 특정 시점 상태를 보고, 에이전트가 다음에 수행할 행동을 **14개 클래스** 중 하나로 예측하는 분류 문제.

## 문제 정의

각 샘플의 입력은 세 부분으로 구성된다:

| 입력 | 내용 |
|---|---|
| `current_prompt` | 현재(가장 최근) 사용자 발화 — 이 직후의 행동이 예측 대상 |
| `history` | 직전까지의 대화·행동 기록 0~12턴 (`user` ↔ `assistant_action` 교대) |
| `session_meta` | 요금제, 잔여 토큰 예산, 턴 번호, 워크스페이스 상태(언어 비율·LOC·git_dirty·open_files·CI 상태) |

**14개 행동 클래스:** `read_file` `grep_search` `list_directory` `glob_pattern` `edit_file` `write_file` `apply_patch` `run_bash` `run_tests` `lint_or_typecheck` `ask_user` `plan_task` `web_search` `respond_only`

## 평가 및 제출

- **평가 지표:** Macro-F1 (희소 클래스 성능이 점수를 좌우)
- **제출 방식:** 코드 제출 대회 — `submit.zip` = `model/` + `script.py` + `requirements.txt`
- **실행 제약:** T4 GPU(16GB) · 3 vCPU · 12GB RAM · 추론 ≤10분 · 설치 ≤10분 · zip ≤1GB · **오프라인**(인터넷 불가)
- **일정:** 예선 2026.07.01 ~ 07.15 09:59 → 상위 12팀 본선(08.11 발표평가)

## 저장소 구조

```
├── PLAN.md        # 워크플로우·전략·일정 (메인 문서)
├── CLAUDE.md      # 에이전트 공통 콘텍스트
├── context/       # 모든 과정 기록 (INDEX·decisions·experiments·research·submissions·daily·reports)
├── agents/        # 역할별 sub-agent instruction
├── configs/       # 실험 설정 placeholder
├── data/          # 대회 데이터 (git 제외 — DACON에서 직접 다운로드)
├── docs/          # validation.md, agent_workflow.md (운영 방식 문서)
├── src/           # features / train / infer
├── submit/        # 제출 스테이징 (대회 규정 zip 구조)
├── tests/         # 단위 테스트 (피처 불변식)
└── scripts/       # new_day.py(데일리 게이트) · make_submit.py(제출 게이트) · validate_submit.py
```

전체 전략과 실험 계획은 [PLAN.md](PLAN.md) 참고.
