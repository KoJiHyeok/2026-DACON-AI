# Feature Agent

## Mission

학습과 추론이 공유할 피처를 `src/features.py`에 구현한다. 목표는 feature leakage 없이 빠르게 반복 가능한 피처 API를 만드는 것이다.

## Read First

- `agents/common.md`
- `src/features.py`
- Latest relevant entries in `context/research.md` and `context/experiments.md`

## Feature Priorities

- P0 text serialization:
  - `current_prompt`
  - recent user turns
  - recent assistant action names
  - assistant action args summaries
- P0 transition/meta:
  - last action, last 2-3 actions
  - history length, turn index
  - last CI status, git dirty
  - open file count/extensions
  - language preference and workspace language mix
- P1 lexical flags:
  - asks to run/test/lint/search/open/list/patch/write/plan/respond
  - Korean/English prompt indicators
  - command/path-like tokens
- P1 class-specific cues:
  - web/search phrasing
  - question/clarification phrasing for `ask_user`
  - final-answer phrasing for `respond_only`

## API Contract

Prefer explicit, testable functions:

- `session_id(sample_id: str) -> str`
- `extract_text(sample: dict) -> str`
- `extract_meta_features(sample: dict) -> dict`
- `extract_features(sample: dict) -> dict`
- batch helpers only if they remove duplication.

## Deliverables

- Update `src/features.py`.
- Add a small smoke path or simple assertions if useful.
- Log feature changes and expected impact in `context/experiments.md`.

## Do Not

- Do not duplicate feature logic in `submit/script.py`.
- Do not use train labels inside feature extraction.
