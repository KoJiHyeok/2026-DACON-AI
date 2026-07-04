# Research Log

> Codex + insane_search(메인 리서치), NotebookLM(문서 정리) 결과를 여기에 축적한다.
> 항목 형식: 주제 / 출처 / 핵심 내용 / 우리 문제에의 적용

## 리서치 큐

- [ ] 텍스트 + 정형 혼합 분류 기법 (경량 환경)
- [ ] Macro-F1 최적화: class weight vs focal loss vs threshold 튜닝
- [ ] 경량 다국어 인코더 비교 (DeBERTa-v3-small, multilingual-e5-small 등) — T4/10분/1GB 제약 하에서
- [ ] 유사 대회(행동 예측·대화 분류) 상위 솔루션

## Feature Backlog

EDA Agent는 발견한 피처 후보를 아래 형식으로 추가한다.

| Priority | Feature | Rationale | Owner | Status |
|---|---|---|---|---|
| P0 | last assistant action / action n-gram | 다음 행동 전이 신호 | Feature Agent | todo |
| P0 | prompt + recent history serialization | 베이스라인이 버리는 history 정보 회수 | Feature Agent | todo |
| P0 | GroupKFold session key | CV 누수 방지 | Modeling Agent | todo |
| P1 | workspace/open file meta | 작업 맥락과 도구 선택 상관 | Feature Agent | todo |
| P1 | class-wise OOF error analysis | Macro-F1 희소 클래스 개선 | Modeling Agent | todo |
