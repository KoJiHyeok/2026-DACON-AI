# EDA Agent

## Mission

데이터 구조, 분포, 누수 가능성, 강한 피처 후보를 찾아 모델링 agent가 바로 구현 가능한 형태로 정리한다.

## Read First

- `agents/common.md`
- `README.md`
- `PLAN.md`
- `context/research.md`
- `context/experiments.md`

## Required Checks

- Class distribution and rare classes.
- Session count, steps per session, history length distribution.
- Last assistant action -> target transition matrix.
- `current_prompt` lexical cues by action.
- `session_meta` correlations: `last_ci_status`, `git_dirty`, `turn_index`, `budget_tokens_remaining`, `language_pref`, `workspace.language_mix`, `open_files`.
- Train/test sample caveat: local `test.jsonl` may be only a public smoke sample.

## Deliverables

- Append concise findings to `context/research.md`.
- Add concrete feature candidates with priority:
  - P0: implement immediately
  - P1: worth testing
  - P2: only if time remains
- If notebooks are created, put them under `notebooks/` and keep them lightweight.

## Do Not

- Do not report random-split scores as official evidence.
- Do not edit submission packaging unless explicitly asked.
