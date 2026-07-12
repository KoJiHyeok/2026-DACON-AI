# task1 — AAR 추론 속도 최적화 (본선 추론속도 10% 대비)

## 컨텍스트

DACON 236694 (AI 에이전트 다음 행동 14클래스, Macro-F1). 예선 점수 축은 전부 실측 종결(exp #50·#51)됐고 챔피언 0.7623이 자동 최종 선택 상태다. **본선 산식 = 예선 Private 50% + 추론속도 10% + 전문가심사 40%** — 07-12 speed 하네스 실측에서 병목은 **AAR stacker (CPU ~10ms/행) > 인코더 블록**으로 판명. 이 작업은 본선 진출/차순위 승계 시나리오 대비 속도 보험이다. **예측 등가성이 절대 조건** — 점수를 바꾸는 최적화는 실격이다.

## 목표 / 완료 조건 (DoD)

1. `scripts/aar_speed/fast_aar.py` — `submit/aar_infer.py`(수정 금지)의 `predict_aar` 경로와 **예측 등가**인 고속 구현. 후보 레버: transition_prior의 행별 파이썬 루프 벡터화, 4컴포넌트 TF-IDF transform 배치화, sparse 행렬 연산 정리, 불필요한 dense 변환 제거. 알고리즘·가중치·수치 계약 변경 금지.
2. **등가성 게이트 (필수)**: `data/train.jsonl`에서 세션 다양성 있는 ≥5,000행 표본에 대해 (a) argmax 예측 5,000/5,000 **100% 일치** (b) 확률 최대 절대 오차 ≤ 1e-9 (부동소수 연산 순서 변경으로 1e-9 초과가 불가피하면 실측 최대 오차와 그 원인을 report에 명시하고 argmax 일치는 무조건 100%).
3. **속도 실측**: 동일 표본에서 원본 대비 행당 평균 시간 비교 (단독 실행, 워밍업 제외, 3회 중앙값). 목표 ≥2× — 미달이어도 실측치와 시도한 레버·병목 분석을 report에 남기면 부분 성공으로 인정.
4. `tests/test_aar_speed.py` — 소표본(≥300행) 등가성 + 반환 shape/순서 계약 테스트, `.venv` CPU에서 통과.
5. `context/night/2026-07-13/report_aar_speed.md` — 실측표(행당 ms 원본/최적화/배속), 등가성 증빙, 레버별 기여, 통합 방법(제출물 반영 절차 제안 — 반영 결정은 Claude). **report 파일명 task로 시작 금지.**
6. `context/night/2026-07-13/task1.DONE` 생성 (요약 포함) — 러너 완료 판정 기준.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 원본 추론: 워크트리 내 `submit/aar_infer.py` (읽기 전용 — 추적 파일이라 워크트리에 있음)
- AAR 모델 실물: `C:\dev\2026-AI-DACON\submit\model\stacker\aar_models.joblib`(47MB)·`aar_config.json` (절대 경로 읽기 전용 — gitignore라 워크트리에 없음)
- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 계측 패턴 참고: 워크트리 내 `scripts/speed/` (07-12 병목 실측 코드)

## 금지

- 워크트리 밖(메인 리포 작업트리·팀 리포) 수정 금지, `git push` 금지, 수동 zip·제출 금지, 네트워크 코드 금지
- `submit/**` 수정 금지 (읽기만) — 최적화 코드는 `scripts/aar_speed/`에만
- 예측을 바꾸는 근사(양자화·프루닝·threshold 변경) 금지 — 등가 변환만
- 폐기 목록(experiments.md 재시도 금지 테이블) 재시도 금지 — 이 작업은 점수 실험이 아니다

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-13/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit** (커밋 실패 시 사유를 PROGRESS에 기록 — 아침에 Claude가 회수)
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋

## 작업 내용 (단계)

1. PROGRESS 생성 → 재료 존재 확인 (aar_models.joblib 로드 가능 여부부터)
2. 원본 프로파일: predict_aar 경로를 행 구간별(component transform / transition / hstack / LR)로 계측해 병목 분해
3. fast_aar.py 구현 (등가 변환만, 단계별로 등가성 확인하며 진행)
4. 등가성 게이트(5,000행) + 속도 실측(3회 중앙값) → tests 작성·통과
5. report_aar_speed.md → task1.DONE → 최종 커밋
