# task2 — 동료 0.7511 구성 요소의 리그 공정 판정 (스태커 최종 · 로그바이어스)

## 컨텍스트

DACON 236694. 동료의 ensau080.zip이 팀 최고 **0.7511**인데, 우리 라인(0.7467)과 구성이 다르다. 동료 구성의 차별 요소는 ① blend 평균이 아니라 **스태커 출력을 최종**으로 사용 ② read/grep/list 클래스 **로그바이어스** (+0.1 / −0.1 / −0.18) ③ AU α=0.8 (우리는 0.9). 이 요소들을 우리 리그(holdout 9,969행)에서 공정하게 실측해, 우리 4-way에 이식할 가치가 있는지 판정한다. α 비교는 task1의 grid_alpha가 다루므로 여기서는 ①②만.

**⚠️ D-009 주의**: 로그바이어스는 우리가 폐기한 threshold/prior/calib 가족이다(리그 위양성 전력 — context/decisions.md D-009). 이 태스크는 '재시도'가 아니라 **동료 LB 실측(0.7511)이 존재하는 요소의 사후 분석**이다. 어떤 결과가 나와도 이 태스크만으로 승격하지 않는다 — 리포트에 D-009 관계를 명기하고, 승격하려면 LB 게이트 + 메인 세션의 명시적 결정이 필요하다고 적는다.

## 목표 / 완료 조건 (DoD)

1. `scripts/mate_eval/eval_stacker_final.py` — holdout에서 ① 스태커 단독(최종) ② 우리 4-way blend ③ '스태커 최종 + soft-AU' 구성의 macro-F1 비교. 클래스별 F1 분해 포함 (스태커가 어느 클래스에서 blend를 이기는지).
2. `scripts/mate_eval/eval_logbias.py` — 로그바이어스 (read +0.1, grep −0.1, list −0.18)를 (a) 스태커 단독 위 (b) 우리 4-way blend 위 (c) 4-way+soft-AU 최종 위에 각각 적용했을 때의 델타. 추가로 세 값 각각의 ±50% 스케일(0.5배/1.5배) 민감도.
3. `context/night/2026-07-07/report_mate_eval.md` — 결과 표 + 해석: 동료 0.7511과 우리 0.7467의 격차(+0.0044)가 어느 요소에서 오는 것으로 보이는지 추정, D-009 관계 명기, 승격 조건 명기. **파일명을 task로 시작하지 말 것.**
4. `context/night/2026-07-07/task2.DONE` 생성 (한 줄 요약 포함) + 최종 커밋.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 평가 npz: `context/night/2026-07-05/holdout_base.npz` (커밋됨, 상대경로 OK)
- mBERT holdout: `C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz` (절대 경로)
- OOF: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` (절대 경로)
- 동료 구성 분석 기록: `context/daily/2026-07-06.md` (ensau080 분석 절)
- AU 모델 전례: `scripts/au2/task4_grid.py`
- 조인 레시피·sanity assert: **task1.md의 '조인 레시피' 절과 동일** — 3-way 0.71726, 4-way 0.72255 (±0.0005) 통과 후 진행. task1과 워크트리가 다르므로 조인 코드는 scripts/mate_eval/ 안에 자체 포함시킬 것.

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 수동 zip·제출 금지, submit/ 수정 금지
- **이 태스크 결과만으로 로그바이어스를 제출 스테이징에 넣는 것 금지** (D-009 — 위 주의 참조)
- 팀 리포(`C:\dev\dacon-agent-action-api-boost`)와 ensau080.zip 내용물은 읽기 전용 — 필요한 수치는 daily 기록에 이미 있으니 원칙적으로 접근 불필요
- GPU 사용 금지

## 진행 프로토콜 (재개 대비)

1. 시작하자마자 `context/night/2026-07-07/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(조인+sanity / stacker_final / logbias / 리포트)마다 PROGRESS 갱신 후 **git commit**
3. 전부 끝나면 `task2.DONE` 생성 + 최종 커밋

## 작업 내용

1. 조인 + sanity 2개 통과 (task1.md 레시피).
2. **스태커 최종**: 주의할 점 — 우리 stacker OOF는 정직 OOF(폴드 밖 예측)라 동료의 full-train 스태커보다 불리하게 나온다. 그래서 '스태커 단독이 blend를 이기면 강한 신호, 지더라도 격차가 작으면(< 0.005) 동료 환경(full-train)에선 역전 가능'이라는 두 단계 해석 기준을 리포트에 명기하고 수치를 그대로 보고한다.
3. **로그바이어스**: `log(p + 1e-12)`에 클래스 상수를 더하고 argmax (동료 방식 미러). read/grep/list의 클래스별 F1 변화와 전체 macro-F1 변화를 함께 표기 — 바이어스가 '어느 클래스를 희생해 어느 클래스를 사는지' 드러낼 것.
4. **리포트**: 격차 +0.0044의 요소별 귀속 추정(스태커 최종 / 로그바이어스 / α 차이 / e5 학습 차이 중 리그에서 측정 가능한 것만), D-009 명기, '승격은 LB 게이트 + 메인 세션 결정 필요' 명기.
