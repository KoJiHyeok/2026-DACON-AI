# task5 (CX-B) — 신표면 오답 국소 분석 (list_directory·glob_pattern 악화 + AU 커버리지)

## 컨텍스트

DACON 236694, D-1 공세 국면. exp #53 블렌드 재튜닝(w_q=3.0, α=0.85)이 전체 +0.00378을 벌었지만 per-class에서 **list_directory −0.0075, glob_pattern −0.0033** 악화를 확인했다 (probe 결과, 이득은 plan_task +0.016·ask_user +0.013·run_bash +0.008). Macro-F1은 클래스당 1/14 지분이라 국소 악화 수복 = 직접 점수. 어디서 잃는지 정확히 알면 다음 LB 프로브 후보(가중 미세조정·AU 커버리지·바이어스)가 나온다.

## 목표 / 완료 조건 (DoD)

1. `scripts/cx_errloc/analyze.py` — holdout에서 두 표면(구: w2.0/α0.90, 신: w3.0/α0.85)의 행별 예측을 비교:
   - list_directory·glob_pattern에서 **신표면이 새로 틀리는 행들**(구는 맞고 신은 틀림)의 목록·특성 분해: 어떤 클래스로 흘렀나(혼동 방향), AU 라우팅 대상 여부, 세션 특성(turn_index·history 길이·workspace 필드), 성분별 확률(lin/stk/qwen 중 누가 밀었나)
   - 반대 방향(신이 새로 맞히는 행)도 대칭 분석 — 순이득 구조 파악
2. `context/night/2026-07-14/report_cx_errloc.md` — 발견 + **실행 가능한 후보 제안 1~3개** (예: "w_q 3.0→2.8이면 list_directory 회복 대비 전체 손실 X", "glob 행은 stk가 유일하게 맞힘 → stk 가중 유지 필수" 같은 정량 근거 형태). 제안마다 기대 row Δ를 holdout에서 실측해 첨부
3. `tests/` 불요 (분석 티켓) — 단 analyze.py는 재실행 재현성(시드 고정·결정적) 확보
4. **`context/night/2026-07-14/task5.DONE` 생성 (한 줄 요약)**

## 재료 (절대 경로, 읽기 전용)

- Qwen holdout 확률: `C:\dev\2026-AI-DACON\colab_out\qwen_i2ep_h85.npz`
- 판정 인프라: `C:\dev\2026-AI-DACON\scripts\league4\common.py` (lin/stk/AU 확률 로드 포함)
- 원 데이터(행 특성 조인용): `C:\dev\2026-AI-DACON\data\train.jsonl`
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`

## 금지

- 워크트리 밖 수정 금지 (cx_errloc 신규 경로만), `git push` 금지, 네트워크 금지
- 후보 제안에 **per-class 하드 규칙 override(템플릿·first-step 등) 금지** — 폐기 목록 등재 축. 확률 공간 조정(가중·α·bias)만
- holdout 재적합형 대규모 탐색 금지 — 제안은 분석에서 도출된 소수 가설만

## 진행 프로토콜 (재개 대비)

1. `context/night/2026-07-14/PROGRESS-task5.md` 확인 → 재개
2. 의미 단위 커밋 시도 (실패 시 PROGRESS 기록)
3. `task5.DONE` + 최종 커밋 시도
