# 제3자 SOL 관점 모델 감사 — 놓친 점, 터닝포인트, 점수 개선안

> 작성일: 2026-07-10
> 기준 커밋: `1a0e8b8`
> 분석 범위: 활성 제출물, 70,000행 원시 데이터, OOF/홀드아웃 확률, 실험·제출 대장, 동료 후보 표면
> 주의: 아래 리더보드 수치는 저장소에 기록된 이력만 사용했다. 07-10 현재 외부 순위·컷은 재조회하지 않았으므로, `0.7807`은 **07-06 기록값**이지 현재 컷이 아니다.

---

## 0. 한 줄 판정

현재 0.7623 모델은 이미 “모델을 더 많이 붙이는 단계”를 지났다. 다음 점프는 **테스트 표본 단위를 닮은 검증으로 바꾸고, 가장 강한 e5에 아직 버린 상태 정보(action args)를 넣은 뒤, 그 강해진 확률을 동료의 스태커까지 OOF로 다시 학습시키는 것**에서 나올 가능성이 가장 높다.

냉정한 결론은 다음과 같다.

- 현재 챔피언은 기술적으로 건전하고, AU 라우팅·mBERT 상보성·e5 full-history라는 세 구조적 발견으로 0.71884에서 0.7623까지 올랐다.
- 반면 같은 9,969행/1,350세션 홀드아웃을 20회 이상 적응적으로 재사용했다. 이제 `+0.001~0.005` 수준의 로컬 차이는 대부분 신뢰하기 어렵다.
- “세션 누수 없는 GroupKFold”만으로 충분하지 않다. 저장소는 LB 델타를 근거로 숨은 테스트를 세션당 한 행이라고 추정하지만, 로컬 리그는 선택된 세션의 모든 스텝을 행 단위로 채점한다.
- 현 e5는 history를 전부 보지만 assistant action의 `args`를 전부 버린다. 마지막 action args 값 중 **32.4%는 현재 e5 직렬화의 user/result/current/open_files 어디에도 나타나지 않는다.** 최근 한 action의 args만 넣으면 384 토큰 잘림률은 8.57%에서 10.95%로만 오른다.
- e5 hist12는 동료 0.7511 표면을 병합한 실험들이 끝난 **뒤**에 발견됐다. 따라서 “동료 표면은 소진”이 아니다. 다만 단순 모델 스왑이 아니라 hist12 OOF로 스태커를 다시 학습해야 한다.

## 1. 현재 모델을 제3자가 다시 그리면

활성 제출 코드는 [submit/script.py](../../submit/script.py), 점수 이력은 [experiments.md](../experiments.md)와 [submissions.md](../submissions.md)에 있다.

| 층 | 현재 구현 | 최종 블렌드 실효 지분(비-AU) | 역할 |
|---|---|---:|---|
| Linear | TF-IDF word/char + history/action/meta + LinearSVC | 25% | 인코더와 다른 표층·구문 신호 |
| AAR stacker | prompt/context/action/transition 4-view 스태커 | 25% | 전이·정형·텍스트 결합 |
| e5-base | full train, max_len 384, history list 12개 전부 | 30% | 현재 가장 강한 의미·맥락 성분 |
| mBERT | full train, max_len 384, history list 6개 | 20% | e5와 다른 토크나이저/표현의 보완 신호 |
| AU specialist | `sess_au` char-TFIDF LinearSVC | AU 행에서 90% | 생성 계열이 다른 하위집단 전용 모델 |

가중치는 `[linear, stacker, encoder-block]=[1,1,2]`, encoder block 내부는 `[e5,mBERT]=[1.2,0.8]`이다. AU 행만 `0.9·P_AU + 0.1·P_blend`로 다시 결합한다. 세션 형제 복원은 안전한 보험이지만 LB 이득은 0이었다.

활성 모델의 대략적인 비압축 구성은 e5 546.7MB, mBERT 342.1MB, AAR 45.2MB, linear 8.0MB, AU 16.9MB다. 제출 zip은 867.9MB, 서버 추론 실측은 4분 14초다. 즉 시간은 아직 여유가 있지만, 모델 파일 용량은 큰 성분 하나를 더 넣을 공간이 없다.

## 2. 무엇이 실제로 점수를 올렸는가

| 전환 | LB 변화 | 본질 |
|---|---:|---|
| 재건 기준선 → hard AU | 0.71884 → 0.73310 | `sess_au`라는 결정적 하위집단을 별도 학습 |
| hard AU → soft AU | 0.73310 → 0.7400 | 경계에서 전문모델과 전역모델을 확률 결합 |
| mBERT holdout/full 추가 | 0.7400 → 0.7480 | 단독 성능보다 오류 상보성이 있는 이질 표현 추가 |
| e5 history 6→12 | 0.7480 → 0.7623 | 강한 성분에 실제 누락 맥락을 복원 |

공통점은 하이퍼파라미터가 아니라 **조건부 데이터 분포 또는 입력 정보량을 바꿨다**는 것이다. 반대로 새 인코더 5종, epoch 연장, GBDT 메타, 탐색 specialist, 텍스트-only linear, 버킷 가중, calibration은 대부분 실패했다.

따라서 과거의 진짜 터닝포인트는 `더 좋은 모델`이 아니라 다음 두 문장이었다.

1. **약점이 식별 가능한 모집단에 집중되면 전용 모델로 라우팅한다.**
2. **모든 모델이 같이 틀리면 결합기를 만지지 말고 가장 강한 성분이 보는 정보를 늘린다.**

현재는 2번의 후속 단계다.

## 3. 놓치고 있는 점

### 3.1 Group split은 맞지만, 평가 표본 단위는 아직 틀릴 수 있다

형제 복원 제출의 LB 델타가 0이어서 저장소는 숨은 테스트를 세션당 한 행으로 해석하고 있다. 이 결과는 가정을 지지하지만 숨은 파일을 직접 본 증거는 아니며, 여러 행이어도 복원 대상이 없거나 예측 변화가 상쇄됐을 가능성은 남는다. **이 가정이 맞다면** 현재 리그가 1,350개 홀드아웃 세션의 9,969개 스텝을 모두 채점하는 것은 평가 단위 불일치다.

원시 train의 구조는 다음과 같다.

- 70,000행 / 9,429세션
- 세션 길이 1~18, 평균 약 7.4행
- 세션 길이 1인 세션은 216개뿐이고, 6~9행 세션이 큰 비중을 차지
- 행 단위 분포와 “세션마다 균등한 한 행”의 기대 분포 차이: `respond_only +2.94%p`, `apply_patch -1.29%p`, `grep_search -0.98%p`, `edit_file -0.71%p`

현재 hist12 후보를 다시 계산하면 전체 행 평가는 `0.734509→0.756006(+0.021496)`, 세션 균등 가중 평가는 `0.738427→0.757013(+0.018585)`였다. 결론은 유지되지만 효과 크기가 달라진다.

더 큰 문제는 스텝별 이질성이다.

| step | hist12 - hist6 Macro-F1 |
|---:|---:|
| 1 | -0.0041 |
| 2 | -0.0099 |
| 3 | +0.0193 |
| 6 | **+0.0489** |
| 7 | **+0.0466** |
| 8 | **+0.0465** |
| 11 | **+0.0650** |

숨은 테스트가 어떤 step 분포로 한 행을 뽑는지 모르면, 같은 후보도 기대 LB가 크게 달라진다. 200회 “세션당 한 행 균등 추출” 시뮬레이션에서 hist12 델타 평균은 +0.0186이었지만, 표준편차가 0.0086이고 개별 추출은 -0.0020~+0.0396이었다.

또한 로컬 `data/test.jsonl` 5행은 모두 train에 **exact ID로 존재하고**, sample_submission의 action도 train label과 5/5 일치한다. 이 파일은 실행 계약용 smoke fixture이지 숨은 테스트 분포의 증거로 쓰면 안 된다.

**놓친 핵심:** 세션 누수 방지와 테스트 분포 일치는 다른 문제다.

### 3.2 하나의 홀드아웃이 사실상 두 번째 리더보드가 됐다

현재 리그는 고정된 9,969행/1,350세션에 모델 추가, 가중치, 라우팅, alpha, GBDT, specialist를 반복해서 맞췄다. 반반 분할 검사는 같은 홀드아웃을 나눈 것이므로 독립 검증이 아니다.

세션 단위 paired bootstrap 5,000회 결과:

| 비교 | 관측 델타 | 95% CI | 판정 |
|---|---:|---:|---|
| hist12 - hist6 | +0.021496 | **[+0.015104, +0.027827]** | 구조적 이득, 견고 |
| e5:mBERT 1.25:0.75 - 현 1.20:0.80 | +0.000508 | **[-0.000729, +0.001725]** | 잡음, 제출 금지 |

즉 지금부터는 “+0.005 게이트”도 단일 개발면에서 찾은 최고값이면 약하다. 후보 생성용 dev split과 마지막 한 번만 여는 shadow split을 분리해야 한다.

### 3.3 가장 강한 e5가 action args를 버린다

현재 e5 직렬화는 assistant history를 `act:{name} r:{result_summary}`로만 만든다. `args`는 AU specialist와 현재 AAR stacker의 일부 text view에는 들어가지만 가장 강한 e5에는 들어가지 않는다.

독립 실사 결과, 각 행의 마지막 assistant action에서 추출한 인자 값 75,584개 중:

- 46.8%는 result summary에 등장
- 22.9%는 앞선 user text에 등장
- 7.0%는 current prompt에 등장
- 36.7%는 open_files에 등장
- 중복을 합쳐도 **32.4%는 어느 곳에도 등장하지 않음**

특히 `run_bash.cmd`, `run_tests.target`, `ask_user.question`, `plan_task.goal`, `web_search.query`는 기존 텍스트에 거의 그대로 남지 않는다. 이는 다음 행동의 연속성을 설명할 수 있는 실제 상태 손실이다.

14,000행 실제 e5 토크나이저 표본에서 측정한 비용:

| 직렬화 | 평균 토큰 | 384 초과 | 512 초과 |
|---|---:|---:|---:|
| 현재 hist12 | 236.4 | 8.57% | 0.02% |
| 모든 action args | 269.3 | 26.59% | 1.40% |
| 최근 2 action args | 251.6 | 13.93% | 0.21% |
| **최근 1 action args** | **244.5** | **10.95%** | **0.06%** |

따라서 raw args 전체 추가는 좋지 않지만, **마지막 action 하나의 타입별 핵심 args를 40~50자로 넣는 변형**은 정보 대비 토큰 비용이 합리적이다.

기존 exp #15는 이 가설을 종결하지 못했다. `v3_hist6`은 args뿐 아니라 language/elapsed도 동시에 추가했고, TF-IDF 프록시는 hist12를 -0.031로 오판한 전력이 있다. “BoW에 유리했는데 실패했으니 args도 끝”은 분리 실험이 아니다.

### 3.4 history의 시간 순서도 아직 고정 가정이다

현재는 최신 정보를 truncation 앞쪽에 두려고 history list를 통째로 뒤집는다. 이 때문에 `assistant_action → 그 action을 요청했던 user` 순서가 되어 pretrained encoder가 익숙한 대화 인과 순서를 거스른다.

재시도할 가치가 있는 형태는 “오래된 turn부터”가 아니라, **최근 pair 우선 + pair 내부는 `user → assistant_action` 유지**다. 예를 들어 최신 `(user, action)` 쌍을 먼저 놓고 그 이전 쌍을 뒤에 둔다. 마지막 action/result는 별도 헤더로 한 번 더 명시한다. 이 변형은 history 정보량을 늘리지 않고 표현만 바꾸므로 args-lite 다음의 저비용 A/B다.

### 3.5 동료 0.7511 표면은 hist12 이후 재검토되지 않았다

기존 병합 실험은 동료 표면에 mBERT를 최종 확률로 0.2 섞거나 AU alpha를 바꿨고, 각각 -0.0008/-0.0010으로 실패했다. 그러나 e5 hist12의 +0.0143 발견은 그 뒤다.

동료 표면의 [script.py](../../submit_candidates/alpha09/script.py)는 `sparse → e5 → stacker → class fix → AU` 구조이며, stacker 입력에 e5의 14확률·margin·entropy를 직접 넣는다. 여기에 강해진 hist12 e5를 넣는 올바른 방법은 다음이다.

1. sparse 모델은 학습 계약대로 hist6 직렬화를 유지
2. e5만 hist12 직렬화 사용
3. hist12 e5의 group-OOF 확률을 새로 생성
4. 그 OOF로 동료 stacker를 다시 학습
5. AU alpha는 원본 0.8을 기준으로 다시 판단

단순히 모델 디렉터리와 `max_hist`만 바꾸면 안 된다. 기존 stacker는 hist6 확률 분포에 맞춰져 있고, sparse 모델까지 hist12를 먹이면 train/infer mismatch가 생긴다. 현재 챔피언에서 해결한 per-encoder serialization 원칙을 동료 표면에도 적용해야 한다.

이 레인은 구현 비용이 있지만, post-hoc mix가 아니라 **강한 base learner를 meta learner 내부에서 다시 학습**하는 것이므로 기존 실패와 다른 실험이다.

### 3.6 학습 목표가 세션 길이를 암묵적으로 가중한다

e5 학습은 row-frequency 기준 inverse class weight와 label smoothing 0.1을 쓴다. 같은 세션의 18개 state는 길이 1 세션보다 18배 학습에 기여한다. 숨은 테스트가 세션당 한 행이라면 이 가중은 의도한 것이 아니다.

검토할 후보는 `class_weight × 1/session_length` 또는 그 완화형 `class_weight × 1/sqrt(session_length)`다. 단, Macro-F1이 클래스 균등을 요구하므로 class weight를 제거하는 실험이 아니라 **클래스 안에서 긴 세션의 중복 영향만 줄이는 실험**이어야 한다.

### 3.7 실험 시스템과 실제 챔피언 사이의 단일 소스 계약이 깨져 있다

문서상 정본은 `src/features.py → src/train.py → src/infer.py`지만 세 파일은 여전히 `NotImplementedError` 스캐폴드다. 실제 챔피언은 `submit/features.py`, `submit/script.py`, 여러 Colab 스크립트와 외부에서 벤더한 joblib에 분산되어 있다.

추가로 `submit/requirements.txt`는 `transformers>=4.51`로 열려 있어 본선 재현성 원칙과 충돌한다. README도 07-04 상태를 가리킨다. 이것은 당장 0.001을 주는 피처는 아니지만, 강한 후보를 안전하게 재학습·이식하는 속도를 떨어뜨리는 현재의 병목이다.

## 4. 다음 터닝포인트

제3자 관점의 다음 터닝포인트는 **“앙상블 탐색”에서 “context compiler + honest OOF surface”로 전환**하는 것이다.

```text
raw state
  ├─ component별 직렬화 계약
  │    ├─ sparse: 기존 hist6 유지
  │    ├─ e5: full history + last-action args-lite
  │    └─ mBERT: full history 후보
  ├─ session/step-aware group OOF
  └─ 강한 surface의 stacker 재학습
       └─ AU만 결정적 prefix로 별도 route
```

hist6 리그 오류의 32.2%는 적어도 한 성분이 맞히지만 결합이 못 고친 행이었고, 나머지 큰 잔차는 read/grep/list/glob의 공동 오류였다. 약한 결합기·라우터를 하나 더 붙이는 것보다, base learner가 보지 못한 실행 상태를 복원해야 공동 오류 프론티어가 움직인다.

## 5. 점수 개선 실행안

### P0 — 먼저 검증면을 고친다 (반나절)

1. 현재 1,350세션 홀드아웃을 `dev-A`로 동결한다.
2. 그룹 해시가 다른 `shadow-B`를 만들고 후보 선택 중에는 점수를 열지 않는다. **B 점수를 낼 모델은 B 세션을 학습에서 제외해 별도로 fit**해야 한다. 기존 full/85% 모델로 B를 잘라 채점하면 누수다. GPU 후보는 `A 제외 control/candidate`로 선택한 뒤, 마지막 후보만 `B 제외 control/candidate` 한 쌍을 재학습한다.
3. 모든 결과에 다섯 지표를 함께 낸다.
   - row Macro-F1
   - session-inverse-weight Macro-F1
   - 세션당 한 행 Monte Carlo 평균±표준편차
   - step bucket별 F1/델타
   - AU/non-AU별 F1
4. paired session bootstrap 95% CI를 자동 생성한다.

승격 조건은 `dev-A +0.0075 이상`, `shadow-B 양수`, `paired CI 하한 > 0`을 기본으로 권한다. 현재 인코더-family의 로컬→LB 전이율 0.67을 적용하면 로컬 +0.0075가 LB 약 +0.005 기대치다.

### P1-A — 이미 준비된 빠른 카드

| 카드 | 실행 | 기대/위험 | 판정 기준 |
|---|---|---|---|
| Bet B: mBERT hist12 | 기존 [probe_b_mbert_hist12.py](../../scripts/league4/probe_b_mbert_hist12.py) 사용 | solo는 오를 수 있으나 e5와 상관이 커져 앙상블 기여가 사라질 수 있음 | 반드시 최종 blend 델타와 상보성으로 판정 |
| Bet A: e5 maxlen512 | 기존 [probe_a_maxlen512.py](../../scripts/league4/probe_a_maxlen512.py) 사용 | 384에서 잘린 8.6% 회수, 시간 증가 | 아래 length-bucket inference 적용 후 T4 8분 이내 |

maxlen512 전에 추론 배치를 토큰 길이순으로 정렬하고 원래 행 순서로 복원해야 한다. 현재처럼 입력 순서대로 64개씩 묶으면 384 초과 행이 8.57%일 때 한 배치에 적어도 하나 들어갈 확률이 약 **99.7%**라 거의 모든 배치가 긴 길이로 padding된다. 정렬은 점수를 바꾸지 않고 Bet A의 시간 위험을 줄인다.

### P1-B — 새 구조 카드: e5 args-lite

대조군과 후보의 유일 차이는 다음 한 줄이어야 한다.

- control: 현재 hist12
- candidate: 현재 hist12 + **마지막 assistant_action의 타입별 핵심 args만** 추가

권장 args는 `path`, `pattern/scope`, `cmd`, `target`, `goal/question/query`, `n_files`이고 각 값은 40~50자로 제한한다. language_mix, elapsed, 전체 args, maxlen 변경은 섞지 않는다.

판정은 e5 solo가 아니라 `linear + stacker + e5-candidate + mBERT + soft-AU` 전체 델타로 한다. 이 카드가 이기면 그 다음에만 최근 2 action args와 pair-order 변형을 각각 단일 변수로 연다.

### P1-C — 가장 큰 업사이드: hist12-aware 동료 stacker

동료 0.7511 표면을 hist12 OOF로 다시 학습한다. 목표는 “0.7511 + 0.0143” 단순 합산이 아니라, 현 0.7623과 다른 결정 경계를 가진 두 번째 강한 surface를 만드는 것이다.

최소 산출물:

- 동일 group fold의 hist6/hist12 e5 OOF
- sparse OOF와 구조 피처
- hist12 stacker OOF
- 현 챔피언과의 error overlap/paired bootstrap
- per-component serialization config

단독으로 현 챔피언을 못 이겨도 오류 상보성이 있으면 용량 안에서 **두 stacker의 작은 확률 결합**을 검토할 수 있다. e5 가중치를 두 벌 넣는 것이 아니라 한 번 계산한 e5 확률을 두 meta surface가 공유해야 한다.

### P2 — 학습 분포 정렬

e5 args-lite 결과가 나온 뒤 다음 세 변형만 비교한다.

1. 현 class weight
2. class weight × `1/sqrt(session_length)`
3. class weight × `1/session_length`

first-step과 late-step를 따로 잘 맞히는 대신 전체 surface에서 안정적으로 이기는지를 본다. 기존 calibration/threshold를 그대로 재개하지 않는다. 분포 정렬 후에도 class별 precision/recall이 일관되게 치우칠 때만, 새 shadow split에서 decision rule 최적화를 별도 가족으로 연다.

## 6. 당장 하지 말 것

- e5:mBERT를 1.20:0.80에서 1.25:0.75로 바꾸는 제출: +0.0005, CI가 0을 포함한다.
- AU ID의 `_000~014` 하위 코드를 피처로 넣는 제출: 동일 홀드아웃 프로브에서 전체 `0.756006→0.755274`, AU F1 `0.7923→0.7812`로 하락했다.
- raw args 전체 추가: 384 초과가 26.6%로 뛰어 full-history 이점을 다시 잘라먹는다.
- 새 인코더 zoo, epoch 연장, GBDT 라우터, 탐색 specialist 재시도: 이미 “solo 강화 ≠ 앙상블 기여”와 공동 오류 벽이 반복 확인됐다.
- 단일 홀드아웃 최고값만 보고 +0.001~0.005 후보 제출.
- 공개 5행의 step 분포나 exact train 중복을 숨은 테스트 구조로 일반화.

## 7. 현실적인 점수 전망

저장소에 남은 07-06 컷 0.7807과 비교하면 현 0.7623은 +0.0184가 더 필요하다. 현재 컷은 이보다 높을 수 있다. 따라서 미세 그리드 두세 개로는 부족하다.

- Bet A/B 중 하나가 이겨도 보통 한 자릿수 milli-F1 카드일 가능성이 높다.
- args-lite 또는 hist12-aware stacker가 성공하면 과거 구조적 카드처럼 +0.006~0.015 LB 구간을 노릴 수 있지만, 이 범위는 목표 시나리오이지 예측값이 아니다.
- 컷을 넘으려면 **중간급 구조적 승리 두 개** 또는 hist12급 큰 승리 하나가 더 필요하다.

현재 챔피언은 안전하게 보존하고, 제출 예산은 `Bet B 또는 A 1회`, `args-lite 1회`, `hist12-aware stacker 1회`처럼 각기 다른 가설에 써야 한다.

## 8. 최종 우선순위

1. **검증면 분리:** dev/shadow + session/step-aware 지표 + bootstrap
2. **e5 args-lite A/B:** 마지막 action 한 개만, 다른 변수 고정
3. **mBERT hist12 Bet B:** 이미 준비된 가장 싼 GPU 카드
4. **hist12-aware 동료 stacker 재학습:** 가장 큰 구조적 업사이드
5. **maxlen512 + length-bucket inference:** 속도 안전장치 후 실행
6. **session-length loss weighting:** 위 직렬화가 고정된 뒤

제3자 SOL의 결론은 간단하다. 이 팀이 놓친 것은 새 모델 이름이 아니라 **“테스트가 한 세션에서 한 행”이라는 자기 가정을 검증·학습 가중에 끝까지 반영하지 않은 것**, 그리고 **가장 강한 encoder에 실행 인자를 넘기지 않은 것**이다. 다음 점수는 앙상블 가중치가 아니라 그 두 계약을 고치는 데서 나올 가능성이 높다.

---

## 근거 문서

- [실험 대장](../experiments.md)
- [제출 대장](../submissions.md)
- [hist12 배포 독립 검증](verify_hist12_deploy_2026-07-10.md)
- [성분 상보성·혼동행렬 정찰](recon_complementarity_confusion_2026-07-09.md)
- [기존 강의 갭 분석](lecture-gap-analysis_2026-07-09.md)
- [활성 추론 코드](../../submit/script.py)
- [e5 holdout/full 학습 코드](../../colab/encoder_e5_holdout85_maxhist.py)
- [동료 stacker 표면](../../submit_candidates/alpha09/script.py)
