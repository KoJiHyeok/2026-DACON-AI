# task3 — 차기 AU형 서브모집단 라우팅 후보 스윕

## 컨텍스트

DACON 236694. 우리 최고 상승 2건(+0.0142, +0.0069)은 전부 **sess_au 서브모집단 라우팅**에서 나왔다 — '전용 학습 specialist가 blend를 크게 이기는 서브모집단'을 찾아 soft 라우팅하는 패턴. 반면 first-step(hist=0) 라우팅은 마진이 없어서 FAIL(exp #25 — 약세 원인이 정보 부족이면 라우팅 무효). sess_au 같은 그룹이 더 있는지 체계적으로 스윕한 적이 없다. 이 태스크는 후보 그룹을 전수 스윕해 '다음 AU'를 찾는다.

## 목표 / 완료 조건 (DoD)

1. `scripts/subroute/sweep.py` — 아래 후보 그룹 정의 전부에 대해: holdout 내 그룹 크기, 그룹 내 4-way blend macro-F1, train 내 그룹 크기를 표로 산출 (1차 스크리닝).
2. `scripts/subroute/probe.py` — 1차에서 '그룹 blend F1이 전체 대비 −0.03 이상 약세 & holdout ≥ 300행 & train(비holdout) ≥ 3,000행'인 그룹만: specialist(char_wb(3-5) 120k TF-IDF + LinearSVC C=1.0 — AU 검증 레시피) 를 **비holdout 그룹 행으로만** 학습 → holdout 그룹 행에서 격리 평가 → soft α ∈ {0.5, 0.6, 0.7, 0.8, 0.9} 그리드 → 전체 holdout 환산 델타.
3. `context/night/2026-07-07/report_subroute.md` — 1차 표 + 2차 프로브 표 + 판정: 환산 델타 **+0.005 이상 = LB 게이트 후보**, +0.002~0.005 = 보고만, 그 외 폐기. 후보가 하나도 없으면 '이 레인 소진'을 명시(그것도 가치 있는 결론). **파일명을 task로 시작하지 말 것.**
4. `context/night/2026-07-07/task3.DONE` 생성 (한 줄 요약 포함) + 최종 커밋.

## 후보 그룹 정의 (스윕 대상)

id 프리픽스 계열:
- `sess_sim` 외의 모든 id 프리픽스 패턴을 실제 데이터에서 열거하고 각각 그룹으로 (sess_au는 **제외** — 이미 라우팅 중)

session_meta 계열 (train.jsonl에서 추출):
- `user_tier` 값별 / `language_pref` 값별 / `workspace.last_ci_status` 값별 (특히 failing)
- `workspace.git_dirty` true/false / `budget_tokens_remaining` 버킷(0~1k, 1k~10k) / `workspace.loc` 극단 버킷
- `turn_index` 상위 버킷 (예: ≥ 8 — 긴 세션 말미). **turn_index==0 (first-step)은 제외** (exp #25 폐기)
- `open_files` 비어 있음 / `history` 길이 ≥ 10

교차 그룹 (1차 표에서 약세 신호가 겹치면 2~3개만 선별):
- 예: language_pref=en & tier 조합 등 — 단 holdout ≥ 300행 조건 유지

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv` (읽기 전용, 절대 경로)
- 평가 npz: `context/night/2026-07-05/holdout_base.npz` (커밋됨, 상대경로 OK)
- mBERT holdout: `C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz` (절대 경로)
- OOF: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` (절대 경로)
- specialist 레시피·격리 규약 전례: `scripts/au2/task4_grid.py`
- 조인 레시피·sanity assert: **task1.md의 '조인 레시피' 절과 동일** — 3-way 0.71726, 4-way 0.72255 (±0.0005) 통과 후 진행. 조인 코드는 scripts/subroute/ 안에 자체 포함.

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 수동 zip·제출 금지, submit/ 수정 금지
- 폐기 레인 재시도 금지: first-step(hist=0/turn_index==0) 라우팅, threshold/prior/calib 가족(D-009), enc 지분 조정
- specialist 학습에 holdout 행 사용 금지 (누수 — AU 격리 규약과 동일하게 세션 프리픽스가 아니라 holdout ids 기준으로 제외)
- GPU 사용 금지 (LinearSVC는 CPU로 충분 — AU 전례 기준 그룹당 수 분)

## 진행 프로토콜 (재개 대비)

1. 시작하자마자 `context/night/2026-07-07/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(조인+sanity / 1차 스윕 / 그룹별 프로브 각각 / 리포트)마다 PROGRESS 갱신 후 **git commit** — 프로브는 그룹 단위로 커밋해 중단 시 그룹부터 재개
3. 전부 끝나면 `task3.DONE` 생성 + 최종 커밋

## 작업 내용

1. 조인 + sanity → 4-way blend(+soft-AU 제외한 비AU 기준) 확률 확보. 라우팅 델타 환산은 soft-AU 적용 후 최종 확률 기준으로도 한 번 더 (AU 라우팅과 겹치는 행 처리: sess_au 행은 모든 그룹에서 제외해 이중 라우팅 배제).
2. 1차 스윕 표 산출 → 통과 그룹 목록을 PROGRESS에 기록.
3. 그룹별 프로브 (오래 걸리는 순서가 아니라 **약세가 큰 순서**로 — 중단돼도 유망한 것부터 결과가 남게).
4. 리포트: exp #25의 교훈('약세 원인이 정보 부족이면 라우팅 무효')을 판정에 반영 — specialist가 그룹에서 blend를 **+0.02 이상** 이기지 못하면 정보 부족형으로 분류하고 후보에서 제외.
