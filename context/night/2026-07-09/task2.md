# task2 — 현행 4-way(0.7480) 오류 분류 → 차기 레버 후보 (모델프리 분석)

## 컨텍스트

DACON 236694. 우리 최고는 4-way(lin+stk+1.2·e5+0.8·mBERT) + soft-AU(α0.9) = LB 0.7480.
soft-AU 라우팅(서브모집단 전용 모델)만이 리그→LB 전이가 실증된 축이었다. serialize 재심(hist12)이
Colab에서 도는 동안, **현행 앙상블이 아직 어디서 틀리는지**를 홀드아웃에서 분해해 serialize 이후의
차기 레버 방향을 잡는다. 이미 계산된 확률만 쓰는 **모델프리 분석**이라 GPU·재학습·LB 불필요.

## 목표 / 완료 조건 (DoD)

1. `scripts/errtax/analyze.py` — `scripts/league4/common.py`의 `load_league_data()`로 현행 최종 확률
   (4-way + soft-AU α0.9)을 재구성하고, 9,969행 홀드아웃에서 아래를 산출:
   - 전체 macro-F1(기준 0.73877 재현 확인) + **클래스별 F1**(약한 순 정렬)
   - **혼동 상위 10쌍**(true→pred, 빈도·비중) — 특히 탐색계열(read/grep/list/glob) 클러스터
   - **서브모집단 분해**: `sess_sim_*` vs `sess_au_*` 별 macro-F1과 클래스별 F1(어느 계열이 약점인지)
   - **macro-F1 갭 기여도**: 어떤 클래스들을 얼마나 올리면 macro가 얼마 오르는지(클래스별 부족분 ×1/14)
2. `context/night/2026-07-09/report_errtax.md` — 위 표들 + **차기 레버 후보 2~3개**(report-only, 구현 금지).
   후보는 반드시 **폐기 목록(experiments.md 재시도 금지 테이블) 비위반**이어야 하고, 각 후보에
   "왜 이 오류 구조가 그 레버로 풀릴 만한가"의 근거를 1~2줄로 단다. (예시 방향: 특정 혼동쌍 전용
   서브모집단 라우팅 — soft-AU가 실증한 축 — 이 가능한 결정적 키가 홀드아웃에 있는지 purity로 점검.)
3. 산출 로그를 `context/night/2026-07-09/task2_run.log`에 저장.
4. 마지막: `context/night/2026-07-09/task2.DONE` 생성(5줄 요약 — 최약 클래스 3개·최대 혼동쌍·추천 레버 1순위).

## 재료 (절대 경로)

- 리그 유틸(정본, 커밋됨): `C:\dev\2026-AI-DACON\scripts\league4\common.py`
  - `load_league_data()`는 확률을 **절대경로(메인 트리)**에서 읽는다: `colab_out\holdout_mbert.npz`,
    `artifacts\oof\oof_rebuild_2026_07_04\`, `context\night\2026-07-05\holdout_base.npz`, `data\`.
    워크트리에서도 절대경로라 그대로 로드된다. **조인 로직 재구현 금지 — 이 함수를 그대로 써라.**
  - 최종 확률 = `four_way_blend(data)` → `apply_soft_au(data, blend, au["probs"], DEFAULT_ALPHA)`.
    AU 확률은 `train_or_load_au_probs(data, <워크트리 내 out_dir>)`로 얻는다(첫 실행 시 수 초~분 학습, 캐시됨).
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — numpy/sklearn 충분, torch 불필요)
- 서브모집단 키: `id`가 `sess_sim_` / `sess_au_`로 시작(포렌식 r1 발견). `au_route.is_au(id)`도 사용 가능.
- 참고(중복 방지): 과거 오류 분석은 구(舊) 워크스페이스 linear/stacker 기준이었다 —
  이번은 **현행 4-way+softAU**라 신규. 재시도 금지 테이블(`context/experiments.md` 하단)을 먼저 읽고
  후보가 거기에 없는지 대조하라.

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 수동 zip 금지, 제출물·네트워크 코드 무관
- **분석·리포트만** — 어떤 레버도 구현·제출·스테이징 금지(후보 제안까지만)
- 폐기 목록(enc 지분 낮추기·calib/threshold/prior·seed soup·serialize 확장은 이미 재심 중이라 제외·
  linear 교체·인코더 다양성·형제 복원)에 있는 방향을 "차기 레버"로 제안 금지
- `*.npz` 등 대용량 바이너리 커밋 금지(경로·재생성 명령만 리포트에)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-09/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터.
2. 의미 단위(load_league_data 재현·기준 F1 확인 / 클래스별·혼동 / 서브모집단 분해 / 레버 후보 작성)마다
   PROGRESS 갱신 후 **git commit**.
3. 전부 끝나면 `task2.DONE` + 최종 커밋.

## 작업 내용

1. `common.load_league_data()` 호출 → 최종 확률 재구성 → 전체 macro-F1이 **0.73877 ± 5e-4**인지 먼저
   assert(리그 정합 확인, 아니면 즉시 실패·원인 기록).
2. `sklearn.metrics`로 클래스별 F1·confusion_matrix 산출, true→pred 오분류 빈도 상위 10쌍 추출.
3. `is_au`(또는 id 프리픽스)로 sim/au 마스크 분리 → 계열별 macro·클래스별 F1.
4. macro 갭 기여도: 각 클래스 (1.0 − F1_c)/14 로 "이 클래스를 완벽히 풀면 macro 최대 상승분" 근사표.
5. 재시도 금지 테이블 대조 후 차기 레버 후보 2~3개 작성(각 근거 + purity 점검이 가능하면 수치 첨부).
6. 로그·리포트·DONE.
