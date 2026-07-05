# task3 — sim/au 분리 분석 + au 서브셋 프로브 (분석 중심, CPU 전용)

## 컨텍스트

DACON 236694: 14클래스, Macro-F1. train id는 `sess_sim_...`(92.8%)과 `sess_au_...`(7.2%) 두 계열. 현 blend의 au 서브셋 macro-F1은 0.514로 sim(0.730)보다 크게 낮고, **id 프리픽스로 추론 시 판별이 공짜**다. Macro-F1은 희소 클래스가 좌우하므로 au에서의 실패가 전체 점수를 깎고 있을 가능성이 있다 — 미개척 레인.

⚠️ 선행 실패 교훈 (반드시 리포트에서 인용·대조할 것):
- R4 계층 specialist, meta-selector: **약한 specialist를 강한 blend 위에 얹으면 진다** (2회 확인)
- calib_v1, R3 prior: train 분포 피팅 bias는 LB 비전이 (2회 확인)
→ 이 태스크의 1차 산출물은 **분석 리포트**다. 프로브가 로컬 PASS여도 "LB 게이트 필수, 선행 실패 유형과 같은 메커니즘인지" 자기 판정을 리포트에 포함할 것.

## 목표 / 완료 조건 (DoD)

1. **분석 리포트** `context/night/2026-07-06/task3_report.md`:
   - au vs sim 라벨 분포 (클래스별), 필드 분포 차이 (history 길이, session_meta 주요 필드, current_prompt 길이/언어)
   - 현 3-way 성분별(linear/stacker/encoder/blend) au 서브셋 per-class F1 vs sim 서브셋 — **어느 성분이 au에서 무너지는가**
   - au 오류의 혼동 행렬 상위 5쌍 + 실제 au 샘플 5개 원문 검토 (au가 왜 어려운지 정성 진단)
2. **프로브 (분석에서 가능성이 보일 때만)**: au 전용 linear (au 4,900행만으로 학습, 세션 그룹 3-fold) vs 현 blend의 au 서브셋 성능 — 정직 group-split로 비교. au 행수가 적어 과적합 위험이 크다는 점을 판정에 반영
3. **명시적 판정**: PASS(au 전용 처리로 리그 전체 macro-F1 +0.002 이상 기대 근거 있음, LB 게이트 후보) / FAIL(au는 데이터가 적어 전용 처리 무익) — 애매하면 FAIL로
4. `context/night/2026-07-06/task3.DONE` 생성 (판정 요약 포함) + 최종 커밋

## 재료 (절대 경로)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` + `data\train_labels.csv` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 리그 평가행·성분 확률: task1.md의 '재료' 절과 동일 (`holdout_base.npz` + `artifacts/oof/oof_rebuild_2026_07_04` 조인 코드 그대로 — 3-way 0.71726 재현 assert 필수)
- au 판별: `id.startswith('sess_au')` (holdout 9,969행 중 au는 약 700행 예상 — 실측해서 리포트에)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출물 접촉 금지, 네트워크 코드 금지
- blend 가중 튜닝 금지 (au 서브셋이라도 enc 지분 조정 실험은 금지 — 리그 신기루 축)
- class bias/prior 피팅을 결론으로 제시 금지 (calib_v1 유형 — 분석 각주로만)

## 진행 프로토콜 (재개 대비)

1. 시작하자마자 `context/night/2026-07-06/PROGRESS-task3.md` 확인 — 있으면 이어서
2. 의미 단위(분포 분석 → 성분별 F1 → 정성 검토 → 프로브)마다 PROGRESS 갱신 + git commit
3. 끝나면 `task3.DONE` + 최종 커밋

## 작업 내용

1. train 70k에서 au/sim 분리, 라벨·필드 분포 비교표 생성 (스크립트는 `scripts/au/analyze.py`).
2. 리그 조인으로 holdout au 행의 성분별 per-class F1 산출 — 무너지는 성분·클래스 특정.
3. au 오류 샘플 정성 검토 (원문 5개 이상 인용).
4. 근거가 보이면 au 전용 linear 프로브 (`scripts/au/probe_au_linear.py`, 정직 group 3-fold). au 전용 모델이 au에서 이기더라도, au가 holdout의 ~7%뿐이므로 전체 macro-F1 환산 기대치를 계산해 판정.
5. 리포트 + DONE.
