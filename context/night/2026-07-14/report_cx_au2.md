# CX-C report — au_linear upgrade candidate

## 결론

**기각: 현행 char_wb C=1 기준선을 유지한다.** Frozen holdout을 완전히 제거한 AU 4,343행·946세션의 5-fold GroupKFold OOF에서 현행 레시피가 `0.680667`로 1위였다. 최고 개선안인 word+char C=1도 `0.679398`로 `-0.001269` 낮았다. 따라서 조건부 산출물인 `scripts/cx_au2/au_linear_candidate/model.pkl`은 만들지 않았고, `submit/**` 원본도 변경하지 않았다.

## 프로토콜

- 선택 데이터: 전체 AU 5,025행 중 frozen holdout AU 682행·153세션을 ID로 제거한 4,343행·946세션.
- 추가 누수 게이트: non-holdout AU 세션과 holdout AU 세션의 교집합 0을 assert.
- CV: `GroupKFold(n_splits=5, shuffle=True, random_state=42)`, 세션 prefix 그룹.
- 선택 지표: 14-class pooled OOF Macro-F1. Holdout label은 변형 선택에 사용하지 않음.
- 변형 수: 사전 고정 5개. TF-IDF는 각 fold의 train 행에만 fit.
- 아티팩트 계약: `{union, clf}`; `union.transform` + `LinearSVC.decision_function`을 `submit/au_route.py::predict_proba`가 softmax하는 현행 계약 유지.

## OOF 변형 비교

| 순위 | 변형 | pooled OOF Macro-F1 | fold 평균 | fold 표준편차 | 기준선 대비 |
|---:|---|---:|---:|---:|---:|
| 1 | baseline char_wb(3–5), C=1.0 | 0.680667 | 0.678467 | 0.016715 | +0.000000 |
| 2 | word(1–2)+char_wb(3–5), C=1.0 | 0.679398 | 0.677194 | 0.016710 | -0.001269 |
| 3 | word(1–2)+char_wb(3–5), C=0.5 | 0.677773 | 0.675916 | 0.021258 | -0.002894 |
| 4 | word(1–2)+char_wb(3–5), C=0.25 | 0.671152 | 0.669423 | 0.022069 | -0.009516 |
| 5 | char_wb(3–5), C=0.5 | 0.669084 | 0.667196 | 0.018785 | -0.011583 |

Fold별 기준선 점수는 `0.695250 / 0.656189 / 0.686915 / 0.693473 / 0.660509`였다. word+char C=1은 일부 fold에서 이겼지만 pooled OOF에서는 기준선보다 낮아 선택되지 않았다.

## 신표면 frozen-holdout 5지표

표면은 `(linear + stacker + 3*qwen) / 5`, AU soft route alpha `0.85`다. 배포 pickle은 AU 5,025행 full-train 아티팩트라 frozen holdout 평가에 직접 쓰지 않고, 동일 char-C1 레시피를 non-holdout AU 4,343행으로 정직하게 재적합했다. OOF 선택 결과도 같은 기준선이므로 후보와 기준 예측은 동일하다.

| 지표 | 기준선 | 후보 | 델타 |
|---|---:|---:|---:|
| row Macro-F1 | 0.771381 | 0.771381 | +0.000000 |
| 세션균등 Macro-F1 | 0.776300 | 0.776300 | +0.000000 |
| 세션당 1행 MC200 | — | — | 평균 +0.000000 ± 0.000000 |
| paired-session bootstrap 2000 | — | — | CI95 [0.000000, 0.000000], P(Δ>0)=0.000 |
| deterministic halves | — | — | half1 +0.000000 / half2 +0.000000 |

AU specialist 단독 holdout Macro-F1도 기준/후보 모두 `0.744158`이다. 5개 축 양수 게이트를 통과하지 못해 최종 판정은 `do_not_swap`이다.

## 산출물과 재현

- 학습: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/cx_au2/train_au2.py`
- 평가: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/cx_au2/eval_au2.py`
- 결과: `scripts/cx_au2/au_linear_candidate/train_summary.json`, `candidate_status.json`, `eval_summary.json`
- 환경: Python 3.13.13, NumPy 2.5.0, scikit-learn 1.8.0.
- Qwen holdout SHA256: `a6547fba022e95365358b38a520ab3f6019c68b5a0c79193b389345551af5ac4`.
- 현행 AU pickle SHA256: `bc01eb659eca930bcad238d9210beb6c2c72d11b4cdbb778fc136a0bd98725e0`.
- 학습 시간: 597.392초. 후보 pickle 상태: `not_written_baseline_won`.

## 검증 및 handoff

- 정적 compile PASS.
- `pytest tests/test_cx_au2.py -q`: `3 passed` (holdout ID 누수, 세션 누수, 실제 AU `predict_proba` pickle roundtrip).
- 모델 라우팅: Sol `gpt-5.6-sol`, reasoning `xhigh`, `branch/commit: N/A (read-only)`로 감사 실행을 시도했으나 sandbox 네트워크가 WebSocket/HTTPS를 차단해 응답 전 종료. 구현·판정 수치에는 사용하지 않음.
- 작업 branch: `night/2026-07-14/task6`; commit은 최종 검증 후 기록.
- 권고: Claude는 현행 `submit/model/au_linear`를 유지한다. `submit/**` 스왑·LB 제출 없음.
- reviewer/tester 후속 확인: JSON 수치 재계산, 외부 입력 해시 대조, `model.pkl` 부재가 조건부 DoD에 맞는지 확인. main 승격 또는 제출은 이 독립 검증 전 금지.

## 한계

- 후보군은 현행 직렬화/추론 계약을 그대로 지키는 LinearSVC 5개로 제한했다. calibrated classifier나 LogisticRegression은 `au_route.predict_proba`가 `decision_function`을 직접 softmax하는 현재 pickle 계약과 확률 의미가 달라 제외했다.
- Frozen holdout에서는 OOF 우승자가 기준선과 같아 새로운 모델의 외부 일반화 비교가 발생하지 않았다. 이는 holdout 사후선택을 피하기 위한 의도된 결과다.
