# 2026 DACON AI부문 — 에이전트 콘텍스트

AI 코딩 에이전트의 다음 행동을 14클래스로 분류하는 대회. 전략·일정은 `PLAN.md`, 실험 기록은 `context/experiments.md` 참고.

역할별 병렬 작업은 `agents/` instruction과 `docs/agent_workflow.md`를 따른다.

## 절대 규칙 (제출 제약)

- 평가 서버: **T4 16GB / 3 vCPU / 12GB RAM, 오프라인** (네트워크 호출 코드 금지)
- 서브에이전트를 돌릴 때는 서브에이전트의 model은 Sonnet 5로 고정.
- **리뷰·테스트는 항상 작성자와 다른 에이전트가 수행** — `.claude/agents/`의 `reviewer`(코드·산출물 리뷰), `tester`(실행 검증) 서브에이전트를 사용한다. 작성자(메인 세션 포함)의 자기 검증으로 판정을 끝내지 않는다.
- submit.zip 루트 = `model/` + `script.py` + `requirements.txt`, **≤ 1GB**
- 추론 ≤ 10분, pip 설치 ≤ 10분, Python만 허용
- **서버 실행 규약**: 서버가 `./data/test.jsonl`, `./data/sample_submission.csv` 제공 → script.py가 `./output/submission.csv` 생성 (sample_submission과 같은 id 순서·컬럼)

## 데이터

- `data/train.jsonl` 70,000건 + `data/train_labels.csv` (id, action)
- 샘플 필드: `id`(예: `sess_sim_..._-step_02`), `current_prompt`, `history`(0~12턴, user↔assistant_action 교대), `session_meta`(user_tier, language_pref, budget_tokens_remaining, turn_index, elapsed_session_sec, workspace{language_mix, loc, git_dirty, open_files, last_ci_status})
- 14클래스: read_file, grep_search, list_directory, glob_pattern, edit_file, write_file, apply_patch, run_bash, run_tests, lint_or_typecheck, ask_user, plan_task, web_search, respond_only

## 실험 규칙

- 평가 지표 **Macro-F1** — 희소 클래스가 점수를 좌우
- CV는 반드시 **세션 프리픽스 기준 GroupKFold** (같은 세션이 train/valid에 갈라지면 누수)
- 피처 코드는 `src/features.py` 단일 소스 — 학습·추론이 같은 함수를 import (불일치 금지)
- 실험은 `context/experiments.md`에 기록: 가설 → 변경점 → 로컬 CV → 리더보드

## 기록 시스템 (context/) — 강제 게이트

- 진입점은 `context/INDEX.md`. **기록이 없으면 일어나지 않은 것** — 리서치·실험·결정·제출 전부 context/에 남긴다.
- 하루 시작: `python scripts/new_day.py --lb1 <1등> --lb12 <12등>` (데일리 로그 + LB 스냅샷)
- 제출은 **반드시** `python scripts/make_submit.py` 경유 — G1 tests → G2 git clean → G3 패키징 → G4 12개 검증 → G5 제출 대장 자동 기록. **수동 zip 금지.**
- 의사결정은 `context/decisions.md`에 ADR-lite로 기록하고 D-00x 번호로 인용한다.

## 프로젝트 Skills (`.claude/skills/`)

반복 작업은 스킬로 실행한다 — 절차·함정 방지 규칙이 스킬에 내장되어 있다.

- `/forensics` — 시뮬레이터 포렌식 라운드 (분석 → `context/reports/forensics_rN.md`)
- `/submit` — 게이트 제출 (make_submit.py 경유, 수동 zip 금지)
- `/exp` — 실험 기록 (experiments.md 형식·판정 규칙 강제)
- `/day-start` — 하루 시작 (new_day.py + 전일 계획 이월)
- `/night-shift` — 밤샘 Codex 작업 생성 (`context/night/<date>/task*.md` + 재개 러너 `scripts/night_shift.ps1`)

## 코드 구조

- `src/features.py` 피처 추출 (공용) / `src/train.py` 학습 → 모델 저장 / `src/infer.py` script.py 원형
- `submit/` = 제출물 스테이징 (script.py, requirements.txt, model/)
- `agents/` = EDA, Feature, Modeling, Encoder, Ensemble, Submission, Review 역할별 작업 지시서
- `configs/` = 실험 설정 placeholder
- 재현성: 시드 고정, 패키지 버전 고정 (본선 코드 검증 7/24 대비)
