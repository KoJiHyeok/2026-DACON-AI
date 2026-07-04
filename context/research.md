# Research Log

> Codex + insane_search(메인 리서치), NotebookLM(문서 정리) 결과를 여기에 축적한다.
> 항목 형식: 주제 / 출처 / 핵심 내용 / 우리 문제에의 적용

## 리서치 큐

> w112 핸드오프(legacy/w112_handoff.md §8) 수신 후 우선순위 재편 — 미세 앙상블로는 컷(−0.055) 도달 불가, **구조적 신호**가 1순위.

- [ ] **P0 시뮬레이터 포렌식**: train이 `sess_sim_*` 시뮬레이션 산출물 → 생성 정책 역공학, (state→action) 결정성 분석. 결정적 규칙 발견 = 해당 구간 F1 ~1.0
- [ ] P1 버킷-게이트 블렌드 (`history_presence`) 프로브 — 정보 획득용
- [ ] P1 탐색 계열 4클래스(read_file·grep_search·list_directory·glob_pattern) 판별 신호 — per-class F1 약점 구간
- [ ] P2 4-way 앙상블 (enc block = base+small 확률 평균) — v2 홀드아웃 파일 도착 후
- [ ] (보류) 경량 인코더 비교, Macro-F1 최적화 일반론 — w112가 이미 e5-base + blend로 검증, 재리서치 불필요

## Feature Backlog

EDA Agent는 발견한 피처 후보를 아래 형식으로 추가한다.
(참고: 아래 P0 3종은 w112의 linear `E_+seq` 피처셋에 이미 구현되어 있음 — 신규 구현이 아니라 **분석·개선** 대상)

| Priority | Feature | Rationale | Owner | Status |
|---|---|---|---|---|
| P0 | last assistant action / action n-gram | 다음 행동 전이 신호 | Feature Agent | **w112에 구현됨** |
| P0 | prompt + recent history serialization | 베이스라인이 버리는 history 정보 회수 | Feature Agent | **w112에 구현됨** |
| P0 | GroupKFold session key | CV 누수 방지 | Modeling Agent | **w112에 구현됨** (9,429 세션) |
| P1 | workspace/open file meta | 작업 맥락과 도구 선택 상관 | Feature Agent | w112 linear에 일부 구현 |
| P1 | class-wise OOF error analysis | Macro-F1 희소 클래스 개선 | Modeling Agent | todo — OOF는 colab_out/oof/ 에 존재 |
| P0 | state→action 결정 규칙 (시뮬레이터 포렌식 산출) | 결정적 구간은 규칙이 모델을 이김 | 미정 | todo |
