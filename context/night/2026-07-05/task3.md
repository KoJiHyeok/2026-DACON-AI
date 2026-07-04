# task3 — blend 확률 수집 + 가중치 그리드 도구

## 컨텍스트

DACON 236694. w112(LB 0.7208)는 3-way 확률 가중 평균이고, 인코더 재학습(Colab) 후에는 blend
가중치를 다시 정해야 한다. LB 제출(일 10회)을 아끼기 위해 **로컬에서 성분별 holdout 확률을 모아
가중치 그리드를 돌리는 도구**를 만든다. Colab에서 만들 인코더 holdout 확률(npz)과 합쳐 쓸 수
있도록 포맷을 통일한다.

## 목표 / 완료 조건 (DoD)

1. `scripts/blend/collect_probs.py` — 세션 프리픽스 StratifiedGroup 85/15 holdout(seed=42, 방식 주석 명기)에서
   linear·stacker 성분의 holdout 확률을 각각 npz(`ids`, `probs`(N×14), `y_true`)로 저장
2. `scripts/blend/grid_blend.py` — 성분 npz 여러 개를 받아 가중치 그리드(각 0~3, step 0.25) Macro-F1 표 + 상위 10 조합 출력.
   출력에 반드시 주의 문구 포함: "로컬 그리드는 encoder 지분을 LB 최적보다 낮게 잡는 경향 — 방향·랭킹 선택기로만 사용"
3. 실제 데이터로 e2e 1회 완주한 실행 로그를 `context/night/2026-07-05/task3_run.log`로 저장
4. `context/night/2026-07-05/task3_report.md` — 사용법, holdout 구성(세션 수·행 수·클래스 분포), 성분별 holdout Macro-F1, 2-way 그리드 상위 조합, 한계
5. 모든 산출물 git commit
6. 마지막으로 `context/night/2026-07-05/task3.DONE` 생성 (5줄 요약)

## 재료 (절대 경로)

- 데이터 (읽기 전용): `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv`
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)
- 팀 리포 (읽기 전용): `C:\dev\dacon-agent-action-api-boost`
  - linear 학습 코드·아티팩트: `linear_pipeline\` (E_+seq 피처, train_final.py — holdout 확률은 85% 재학습으로 생성)
  - stacker 아티팩트: 루트 `model\` — **재학습 금지**, 기존 아티팩트로 holdout 예측만
    (이 경우 stacker의 holdout 점수는 낙관 편향이 있음을 리포트에 명시 — stacker 학습에 holdout이 포함됐기 때문)
  - 확률 생성 방식은 `ensemble\script_3way.py`가 각 성분 확률을 만드는 코드를 그대로 재사용
- npz 포맷 계약: `C:\dev\2026-AI-DACON\colab\holdout_eval.py` (인코더용 — 있으면 포맷을 맞추고, 없으면 ids/probs/y_true 계약으로 진행)
- 그룹키: `id.rsplit("-step_", 1)[0]` (`src/features.py`의 `session_id` — 워크트리에 있음)

## 금지

- 메인 리포·팀 리포 수정 금지, `git push` 금지, 산출물은 이 워크트리 안에만
- 랜덤 split 금지 — 세션 group-split만
- *.npz 등 대용량 바이너리 커밋 금지 (경로와 재생성 명령만 리포트에)
- 폐기 목록 위반 금지 (특히 calib류 온도/bias 튜닝은 폐기됨 — 가중치 그리드만)

## 진행 프로토콜 (재개 대비 — 필수)

1. 시작하자마자 `context/night/2026-07-05/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(split 함수 / linear 확률 생성 / stacker 확률 생성 / grid 완주)마다 PROGRESS 갱신 + git commit
3. 전부 끝나면 task3.DONE + 최종 커밋

## 작업 내용

1. holdout split 유틸 작성 (세션 단위 85/15, 라벨 층화는 세션 대표 라벨 기준, seed 42) — collect와 grid가 같은 split을 공유하도록 ids를 npz에 포함.
2. linear: `linear_pipeline`의 피처·학습 코드를 워크트리로 복사해 85%로 재학습 → 15% 확률. 클래스 순서는 script_3way.py의 정렬 기준(ACTIONS)을 따르고 npz에 클래스 목록도 저장.
3. stacker: 기존 아티팩트 로드 → 15% 예측 확률 (낙관 편향 주석).
4. grid_blend.py로 linear+stacker 2-way 그리드 실행 — task1의 [1,1] 선택이 합리적인지 교차 확인 자료가 된다.
5. 포렌식 1라운드 발견 반영(`C:\dev\2026-AI-DACON\context\reports\forensics_r1.md` 부가 발견): train에는 `sess_sim_*`(92.8%)와 `sess_au_*`(7.2%, 라벨 분포 이질) 두 계열이 섞여 있다 — holdout의 sim/au 구성비와 계열별 Macro-F1을 리포트에 별도 표기하라 (CV 대표성 점검 자료).
6. 리포트 + DONE.
