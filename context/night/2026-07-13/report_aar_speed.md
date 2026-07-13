# AAR inference speed report

## 결론

배포 AAR의 확률과 argmax를 바꾸지 않고 필수 속도 목표를 통과했다. 전체
70,000행에서 균등 간격으로 뽑은 5,000행(3,817개 세션)에 대해 원본 대비
중앙값 **2.823x**, **5,000/5,000 argmax 일치**, 확률 최대 절대 오차
**0.0**이었다. `submit/**`와 외부 모델/데이터는 수정하지 않았다.

## 입력과 환경

- Branch: `night/2026-07-13/task1`, base `e027f8b9`
- Python 3.13.13, scikit-learn 1.8.0, NumPy 2.5.0, SciPy 1.18.0
- Windows 11 CPU 실행, `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`
- 논리 프로세서 8개가 보이는 로컬 환경. 구현 자체는 단일 프로세스이며 평가 서버의
  3 vCPU 제한을 넘는 병렬화를 사용하지 않는다.
- 모델 SHA256: `31b10456d072ce0e7e4a868c1f02fe7451cf90fe786de53104fed3dec1e0ed6d`
- config SHA256: `f7eb7d95003bf003cf6cdd68c593547b486bb7a67ae201c644589a70fb362e27`
- train JSONL SHA256: `a60ed84b75285caee237142ce97622fb55bb59c36be6b36ddee523992f83df19`

모델 로드와 JSONL 파싱은 시간에서 제외했다. 원본과 최적화 모두 같은 records,
`record_to_text`, `record_to_prompt_text` 결과를 입력받는다. 32행으로 양쪽 경로를
워밍업한 뒤 각각 3회 실행했고 중앙값을 사용했다.

## 필수 5,000행 게이트

| 항목 | 원본 | 최적화 |
|---|---:|---:|
| 실행 시간 run 1 | 27.7270 s | 9.5794 s |
| 실행 시간 run 2 | 27.4398 s | 9.7202 s |
| 실행 시간 run 3 | 25.6304 s | 19.2024 s |
| 중앙값 | **27.4398 s** | **9.7202 s** |
| 중앙값 행당 시간 | **5.4880 ms** | **1.9440 ms** |
| 중앙값 배속 | 1.000x | **2.823x** |

최적화 run 3에는 시스템 잡음으로 보이는 큰 지연이 있었지만, 티켓에 명시된 3회
중앙값 규칙을 그대로 적용했다. 위 표는 CLI가 출력한 원시 run 값을 그대로 옮겼다.

등가성 결과:

- 확률 shape: `(5000, 14)`, `AAR.ACTIONS` 순서
- argmax: **5,000/5,000 일치 (100%)**
- 전체 확률 원소 최대 절대 오차: **0.0** (`<= 1e-9` 통과)
- 별도 300행 pytest에서도 vendor `predict_aar` 라벨과 확률/순서 계약 통과

## 병목과 레버별 기여

초기 1,000행 프로파일에서 hstack(0.0002 s)과 최종 LR(0.0116 s)은 병목이
아니었다. `prompt_context_sgd`가 지배했고 그 안의 `char_wb` TF-IDF transform이
대부분이었다.

같은 1,000행에서 각 레버를 3회 격리 실행한 중앙값은 다음과 같다.

| 레버 | 변경 전 | 변경 후 | 관찰 |
|---|---:|---:|---|
| 실제 config에 없는 view 생략 | 0.8991 s | 0.4291 s | `history/meta/rule/full` 등 미사용 view를 만들지 않음 |
| prompt-context 컴포넌트 | 2.9443 s | 1.1990 s | cached word→ngram 희소 곱, 2.46x |
| transition prior | 0.2116 s | 0.2116 s | key 조합 memoization은 이 표본에서 유효 이득 없음 |
| hstack | 0.0002 s | 사전할당 | 절대 기여는 미미, 중간 복사 한 번 제거 |

최종 알고리즘은 `char_wb`가 공백 경계를 넘는 n-gram을 만들지 않는 성질을 이용한다.
고유 단어별 fitted-vocabulary n-gram 정수 카운트를 희소 행렬 `W`에 한 번 만들고,
문서별 단어 정수 카운트 `D`를 만든 뒤 `D @ W`로 원래 document-term count를 얻는다.
CSR index를 정렬한 다음 **원본 fitted TF-IDF transformer와 원본 classifier를 그대로**
사용한다. 1,000행 원형 검증에서 원본/최적화 희소행렬의 다른 원소 수가 0이었고,
최종 5,000행 확률도 bit-identical이었다.

## 코드와 테스트

- `scripts/aar_speed/fast_aar.py`: 빠른 확률/라벨 경로, 원본 확률 reference,
  균등 표본 및 3회 중앙값 CLI
- `tests/test_aar_speed.py`: 실물 300행 확률/argmax, vendor 라벨, shape/순서,
  길이 오류 계약
- 실행: `python -m pytest -q tests/test_aar_speed.py`
- 재측정: `python -m scripts.aar_speed.fast_aar --data <train.jsonl> --model-dir <stacker-dir> --rows 5000 --repeats 3`
- 결과: **3 passed**, 21개 warning은 joblib이 NumPy 2.5 배열 shape를 복원할 때의
  deprecation warning이며 판정에 영향 없음

## 제출물 통합 제안 (결정/수정은 Claude)

현재 파일은 `scripts/` 아래 검증 구현이므로 제출 zip에 자동 포함되지 않는다.
Claude가 통합을 선택할 때 다음 순서를 권장한다.

1. 독립 reviewer가 private sklearn 속성 사용 범위와 원본 산술/열 순서를 감사한다.
2. 독립 tester가 패키징 환경의 sklearn 1.8.0에서 5,000행 확률 0 오차 게이트를 재실행한다.
3. helper를 제출물 루트의 vendor 모듈로 옮기고 `stacker_probs`가 이미 로드한 artifact를
   `fast_predict_proba`에 주입한다. 원본 경로는 fallback으로 보존한다.
4. `make_submit.py` 게이트와 전체 추론 smoke를 통과한 뒤에만 Claude가 반영 여부를 결정한다.

주의점: 빠른 경로는 sklearn 1.8.0의 `_white_spaces`, `_char_wb_ngrams`, `_tfidf`
private 구현을 의도적으로 사용한다. 패키지의 sklearn 버전이 달라지면 무조건 재검증해야
한다. 점수 변경, 양자화, pruning, threshold 변경, 네트워크 호출은 없다.

## 라우팅/인계 기록

- 요청 모델: `gpt-5.6-sol`, reasoning high, read-only
- branch/commit: N/A (read-only routed audit)
- 결과: sandbox가 WebSocket과 HTTPS를 모두 차단해 라우팅 실행은 결과 생성 전 실패
- 로컬 검증: 위 300행 pytest와 5,000행 3회 게이트 PASS
- 필요한 다음 검증: 작성자와 다른 Claude reviewer/tester의 코드 감사 및 독립 재실행

---

## 회수 검증 부기 (Claude, 07-13 낮)

- **tester(test-task1) PASS**: pytest 3 passed / 정식 게이트 독립 재실행 argmax 5000/5000, 확률오차 0.0, 3.32x / submit/** 무수정 / vendor 헬퍼 12개 시그니처 대조 일치.
- **reviewer(rev-task1) 조건부 병합**: (a) SHA 자릿수 지적은 재계산 결과 3개 모두 64자 정확 — 리뷰어 측 오집계로 종결. (b) 본 리포트의 등가성 게이트는 `use_bias=false` config 기준 — bias 분기·비표준 파이프라인 fallback 분기는 **미실측** (코드상 원본 위임이라 로직 리스크 낮음). (c) submit/ 통합 시 패키징 환경에서 5,000행 등가성+속도 게이트 재실행 필수 — 미이행 상태로 main 회수만 함.
- 맥락: 제출 #13(Qwen)이 T4 10분 초과 FAIL하면서 본 최적화가 시간 예산 카드로 승격됨.
