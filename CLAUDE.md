# 2026 DACON AI부문 — 에이전트 콘텍스트

AI 코딩 에이전트의 다음 행동을 14클래스로 분류하는 대회. 전략·일정은 `PLAN.md`, 실험 기록은 `docs/experiments.md` 참고.

## 절대 규칙 (제출 제약)

- 평가 서버: **T4 16GB / 3 vCPU / 12GB RAM, 오프라인** (네트워크 호출 코드 금지)
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
- 실험은 `docs/experiments.md`에 기록: 가설 → 변경점 → 로컬 CV → 리더보드
- 제출 전 `scripts/make_submit.py`로 패키징 + 오프라인 스모크 테스트 통과 필수

## 코드 구조

- `src/features.py` 피처 추출 (공용) / `src/train.py` 학습 → 모델 저장 / `src/infer.py` script.py 원형
- `submit/` = 제출물 스테이징 (script.py, requirements.txt, model/)
- 재현성: 시드 고정, 패키지 버전 고정 (본선 코드 검증 7/24 대비)
