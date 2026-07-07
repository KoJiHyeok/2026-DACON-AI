# task2 — linear 성분 개선 스윕 (피처·C 그리드 → 정직 OOF → 리그 판정)

## 컨텍스트

DACON 236694. 우리 4-way(LB 0.7480)의 성분 중 **linear(챔피언 TF-IDF+LinearSVC 계열)는 대회 초반 이후 한 번도 개선을 시도한 적이 없는 미탐색 축**이다. 인코더 축은 소진 판정(#29), 라우팅 축도 소진(밤샘 07-07 task3) — 남은 내부 개선 여지는 sparse 성분들이다. 주의할 반례: #29에서 '단독 강화 ≠ 앙상블 기여'가 확인됐으므로, 판정은 단독 F1이 아니라 **blend 전체의 리그 델타**로만 한다.

## 목표 / 완료 조건 (DoD)

1. `scripts/linear2/baseline_repro.py` — 현행 linear의 학습 레시피를 리포에서 역추적(src/train.py·submit/features.py·artifacts 문서)해 동일 폴드로 OOF를 재현, 기존 `artifacts/oof/oof_rebuild_2026_07_04/linear_probs.npy`와의 일치(또는 근접, macro-F1 차 < 0.002)를 확인. **재현이 안 되면 그 사실을 리포트에 명시하고 현행 OOF를 기준선으로 그대로 사용** (재현 실패가 태스크 실패는 아님).
2. `scripts/linear2/sweep.py` — 변형 그리드: char_wb ngram (2-4 / 3-5 / 2-5) × max_features (120k / 200k / 300k) × C (0.5 / 1.0 / 2.0) × [char 단독 vs word(1-2)+char union] 중 **유망 순서로 최소 8개 변형**. 각 변형: 기존 `fold_indices.json`과 **동일한 3-fold**로 정직 OOF 생성 → holdout 9,969행에 대해 lin 교체 후 4-way+soft-AU 리그 값.
3. 판정 표: 변형별 (단독 OOF F1, holdout 리그 델타 vs 기준 0.73877, 반반 안정성 — 최고 후보만). 게이트: **+0.005 = LB 후보**, +0.002~0.005 = 보고만.
4. `context/night/2026-07-08/report_linear2.md` — 표 + 판정 + '단독↑ vs 앙상블 기여' 관계 관찰 (**파일명 task로 시작 금지**).
5. `context/night/2026-07-08/task2.DONE` (한 줄 요약) + 최종 커밋.

## 재료 (절대 경로)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — **시스템 파이썬 금지**, 아티팩트 학습은 반드시 이 버전)
- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv` (읽기 전용)
- 기존 OOF·폴드: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` — `fold_indices.json`(동일 폴드 필수), `linear_probs.npy`, `classes.json`, `row_ids.json`
- 리그 조인: `scripts/league4/common.py` (커밋됨) — `load_league_data()`로 조인 + sanity(3-way 0.71726 / 4-way 0.72255) 자동 검증, `train_or_load_au_probs()`·`apply_soft_au()` 재사용. 기준선 B4+softAU = **0.73877**
- 현행 linear 레시피 추적 시작점: `src/train.py`, `submit/features.py`, `context/experiments.md` 초기 항목

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출 금지, submit/ 수정 금지
- **세션 프리픽스 GroupKFold 위반 금지** — 폴드는 기존 fold_indices.json 재사용이 원칙 (새 분할 재구현 금지, 누수 사고 방지)
- 시스템 파이썬(sklearn 1.9.0)으로 모델 학습 금지
- 폐기 레인 금지: char-ngram **4번째 성분 추가**는 폐기됐다 — 이 태스크는 성분 추가가 아니라 **기존 linear의 교체 후보** 탐색이다. blend 가중(weights.json) 변경도 금지 (enc 지분 축)
- GPU 금지

## 진행 프로토콜 (재개 대비)

1. 시작 시 `context/night/2026-07-08/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. **변형 1개 OOF 완료 = 커밋 1회** (OOF npy는 night_out/linear2/에 저장 — 대용량이면 gitignore 대상이라 커밋엔 결과 수치만 남긴다. PROGRESS에 변형별 수치 즉시 기록)
3. 끝나면 `task2.DONE` + 최종 커밋

## 작업 내용

1. 현행 linear 레시피 역추적 → baseline_repro (1시간 내 재현 안 되면 스킵하고 기존 OOF 기준선 채택).
2. 그리드를 '유망 순서'로 정렬해 실행: 기존 레시피 근방(소폭 변형)부터 → 급진 변형(union)은 뒤로. 중단돼도 유망한 것부터 결과가 남는다. 폴드당 학습이 느리면(>20분) max_features를 줄인 프록시로 1차 스크리닝 후 상위만 풀 그리드.
3. 각 변형: OOF 생성 → common.load_league_data의 lin만 교체한 4-way+soft-AU 리그 값 → 표 갱신.
4. 리포트: 게이트 통과 후보가 있으면 '제출 스테이징 변경 절차'(submit/model/model.pkl 교체 + full-train 재학습 필요 여부)까지 명시. 없으면 'linear 축 소진'을 명시.
