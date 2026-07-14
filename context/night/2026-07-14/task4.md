# task4 (CX-A) — Qwen 블록 캘리브레이션 후보 (calib.json)

## 컨텍스트

DACON 236694, D-1 공세 국면 (사용자 지시: 제출 예산 해제·최고점 갱신 집중). 현 챔피언 #14 = Qwen2.5-0.5B 단독 인코더 블록 하이브리드 (LB 0.77089). script.py는 `model/calib.json`(있으면 인코더 블록에 `p ← softmax(log(p)/T + bias)` 적용, script.py 233·290행)을 지원하지만 **Qwen 블록에는 미적용 상태**. e5 시절 calib_v1이 +0.005를 실증한 축이다. 지금 블렌드 재튜닝(w_q=3.0, α=0.85 — exp #53, LB 프로브 중)이 새 기준선이다.

## 목표 / 완료 조건 (DoD)

1. `scripts/cx_calib/fit_calib.py` — Qwen 블록 확률(`colab_out/qwen_i2ep_h85.npz`)에 대해 temperature T·class bias를 **holdout 내부 세션 GroupKFold 5-fold 정직 적합**(각 fold의 T/bias는 나머지 4 fold로 적합 → 해당 fold 평가, 최종 파라미터는 전체 재적합)으로 산출 → `scripts/cx_calib/calib_candidate.json` (`{"temperature": T, "class_bias": {action: b, ...}}` — script.py load_calib 형식과 동일 키)
2. `scripts/cx_calib/eval_calib.py` — **새 표면 기준 판정**: 기준선 = (lin+stk+3·qwen)/5 + soft-AU α0.85 (scripts/league4/common.py 재사용). 후보 = qwen 블록에 calib 적용 후 같은 블렌드. 5지표(row/세션균등/MC200/bootstrap CI/반반) 출력 — `scripts/league4` 판정 코드를 import하거나 동일 수식 재현 (시드 42 고정)
3. `tests/test_cx_calib.py` — 최소 3케이스 (fold 분할이 세션 기준인지 / calib 적용 수식이 script.py 290행과 동치인지 / json 형식 로드 왕복) — pytest 통과
4. `context/night/2026-07-14/report_cx_calib.md` — 5지표 결과표 + fold별 T/bias 안정성 + 권고 (채택/기각)
5. **`context/night/2026-07-14/task4.DONE` 생성 (한 줄 요약)**

## 재료 (절대 경로, 읽기 전용)

- Qwen holdout 확률: `C:\dev\2026-AI-DACON\colab_out\qwen_i2ep_h85.npz` (h85 instruct-2ep, solo 0.75932)
- 판정 인프라: `C:\dev\2026-AI-DACON\scripts\league4\common.py` (load_league_data·apply_soft_au·align_npz_probs)
- calib 적용 수식 원본: `C:\dev\2026-AI-DACON\submit\script.py` 233~245·282~292행 (load_calib·적용식)
- 데이터: `C:\dev\2026-AI-DACON\data\` / 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`

## 금지

- 워크트리 밖 수정 금지 (`submit/`·`scripts/league4/`·기록 파일 포함 — cx_calib 신규 경로만 생성), `git push` 금지, 네트워크 금지
- **T·bias 대규모 그리드 금지** — T는 스칼라 최적화(예: scipy minimize_scalar), bias는 fold-정직 닫힌형/로지스틱 재적합만. holdout 직접 적합(비-fold) 수치를 최종 권고에 쓰지 말 것
- 폐기 목록(experiments.md 재시도 금지 테이블) 위반 금지

## 진행 프로토콜 (재개 대비)

1. 시작 시 `context/night/2026-07-14/PROGRESS-task4.md` 확인 — 있으면 재개 지점부터
2. 의미 단위마다 PROGRESS 갱신 후 git commit 시도 (index.lock 실패 시 PROGRESS에 기록하고 계속)
3. 끝나면 `task4.DONE` + 최종 커밋 시도

## 작업 내용

1. npz 로드·정렬(align_npz_probs) → 세션 GroupKFold 5-fold (session_id = id의 `-step_` 앞부분)
2. fold별 T 적합(NLL 최소화) → OOF calib 확률 → bias 적합(로그공간, 릿지 권장) → OOF 재평가
3. 새 표면 블렌드로 5지표 산출 → 리포트 → 전체 재적합 파라미터로 calib_candidate.json
