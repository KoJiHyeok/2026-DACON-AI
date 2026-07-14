# task6 (CX-C) — au_linear 업그레이드 후보 (AU 라우팅 축 강화)

## 컨텍스트

DACON 236694, D-1 공세 국면. AU 라우팅 축은 **전이율 ~1:1이 실증된 유일한 축** (#5 +0.0142, #6 +0.0069 — 리그 예측과 LB가 거의 일치). 현 `submit/model/au_linear`는 char-C1 선형 모델로 07-06 이후 미갱신. α 재튜닝(0.85)까지 끝난 지금, **AU 모델 자체를 강하게** 만들면 라우팅 축에서 추가 이득 여지가 있다. sess_au 스코핑은 5,025행(holdout 내 서브셋) — 작은 데이터라 CPU로 충분.

## 목표 / 완료 조건 (DoD)

1. `scripts/cx_au2/train_au2.py` — 현행 au_linear 레시피(재료의 au_route.py·기존 학습 스크립트 참조)를 기준선으로, 개선 변형 3~5개를 **세션 GroupKFold**로 비교: 예) word+char 결합 TF-IDF, C 재탐색(소수 그리드), 클래스 가중, calibrated SVC/로지스틱. **holdout id는 학습에서 명시적 제외** (common.py train_or_load_au_probs의 leak assert 패턴 재사용)
2. `scripts/cx_au2/eval_au2.py` — 최고 변형의 holdout AU 서브셋 확률로 신표면(w3.0/α0.85) 5지표 판정 (기준선 = 현행 au_linear 확률)
3. 최고 변형이 기준선을 넘으면 `scripts/cx_au2/au_linear_candidate/model.pkl` 저장 (submit/au_route.py의 predict_proba 인터페이스와 호환 — artifact 형식 동일)
4. `tests/test_cx_au2.py` — 최소 2케이스 (leak assert 동작 / candidate 아티팩트가 AU.predict_proba로 로드·추론되는지) — pytest 통과
5. `context/night/2026-07-14/report_cx_au2.md` — 변형 비교표 + 5지표 + 권고
6. **`context/night/2026-07-14/task6.DONE` 생성 (한 줄 요약)**

## 재료 (절대 경로, 읽기 전용)

- 현행 AU 코드·아티팩트: `C:\dev\2026-AI-DACON\submit\au_route.py`, `C:\dev\2026-AI-DACON\submit\model\au_linear\model.pkl`
- AU 확률 캐시·스코핑: `C:\dev\2026-AI-DACON\scripts\league4\common.py` (train_or_load_au_probs — sess_au 정의·leak assert)
- Qwen holdout 확률(신표면 판정용): `C:\dev\2026-AI-DACON\colab_out\qwen_i2ep_h85.npz`
- 데이터: `C:\dev\2026-AI-DACON\data\` / 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 필수)

## 금지

- 워크트리 밖 수정 금지 (cx_au2 신규 경로만 — 특히 `submit/model/au_linear` 원본 불변), `git push` 금지, 네트워크 금지
- 변형 수 5개 초과 금지 (다중비교 억제), holdout 직접 적합 금지 (GroupKFold OOF로만 변형 선택)
- 폐기 목록 위반 금지 (uniform enc block·버킷 가중 등은 enc 축 얘기지만 유사 신기루 주의)

## 진행 프로토콜 (재개 대비)

1. `context/night/2026-07-14/PROGRESS-task6.md` 확인 → 재개
2. 의미 단위 커밋 시도 (실패 시 PROGRESS 기록)
3. `task6.DONE` + 최종 커밋 시도
