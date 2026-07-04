# Research Log

> Codex + insane_search(메인 리서치), NotebookLM(문서 정리) 결과를 여기에 축적한다.
> 항목 형식: 주제 / 출처 / 핵심 내용 / 우리 문제에의 적용

## 리서치 큐

> w112 핸드오프(legacy/w112_handoff.md §8) 수신 후 우선순위 재편 — 미세 앙상블로는 컷(−0.055) 도달 불가, **구조적 신호**가 1순위.

- [x] ~~**P0 시뮬레이터 포렌식**: (state→action) 결정성 분석~~ → **1라운드 완료, 가설 기각** (D-007). 결정 규칙 coverage 0.03%로 규칙 노선 폐기. 산출 리드 R3/R4로 전환. [reports/forensics_r1.md](reports/forensics_r1.md)
- [ ] **P0 R4 explore 계층 분류 프로토타입** — 1단계 대분류 + 2단계 (last2,last1) 조건부 explore 분류기, 플랫 대비 로컬 CV 비교 (밤샘 task2 진행 중)
- [ ] **P0 R3 첫스텝(history_len==0, 12.9%) 전용 class-wise prior** — 분포 이질성 구조적 확인됨. calib_v1 전례 때문에 LB 게이트 필수 (밤샘 task2에 상한 실측 포함)
- [ ] P1 버킷-게이트 블렌드 (`history_presence`) 프로브 — 정보 획득용 (R3의 거친 버전, 로컬 delta +0.0036 gate 통과 상태)
- [ ] P1 sim/au 계열 검증 — train은 `sess_sim_*`(92.8%)+`sess_au_*`(7.2%, 라벨 분포 이질) 혼합. CV fold 대표성 점검
- [ ] ~~P1 탐색 계열 4클래스 판별 신호~~ → 프롬프트 표층 신호는 **전멸(음성 확정)**, 조건부 시퀀스 신호만 유효 — R4에 흡수
- [ ] P2 4-way 앙상블 (enc block = base+small 확률 평균) — colab job2/job3 산출물 도착 후
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
| P0 | ~~state→action 결정 규칙 (시뮬레이터 포렌식 산출)~~ | 결정 구간이 존재하지 않음이 실측 확인 | 포렌식 1R | **기각 (D-007)** |
| P1 | args 키 스키마·turn_index=step 중복 피처 정리 (R5) | 중복 제거 = 노이즈 감소 (점수 이득 아님) | Feature Agent | todo |
| P2 | budget/ci_status state-conditioned 보정 (R6) | 방향은 뚜렷하나 calib_v1 실패 모드 위험 | 보류 | R3에 흡수해 검증 |
