# 데이터 분석 초안 — 시뮬레이터 상태와 서브모집단

> 전문가심사 데이터분석 10점 대응용 초안. 모든 수치는 저장소 기록에 근거하며, 제출용 발표에서는 표와 그림을 보강한다.

## 1. 문제와 데이터

- 입력은 `current_prompt`, 0~12턴의 `history`, `session_meta`이며 다음 에이전트 행동 14개를 예측한다. 학습 데이터는 70,000행, 14클래스, 세션 시뮬레이터에서 생성된 행동 분류 데이터다. 평가 지표는 14클래스 Macro-F1이다. [근거: `CLAUDE.md`; `PLAN.md`]
- Macro-F1은 클래스별 F1의 산술평균이므로 다수 클래스 정확도보다 희소 클래스의 재현율·정밀도 균형이 중요하다. 따라서 이후 분석은 전체 점수뿐 아니라 `read_file`, `grep_search`, `list_directory`, `glob_pattern`, `ask_user` 등 취약 클래스별 F1을 함께 본다. [근거: `context/decisions.md` D-003; `context/experiments.md` 하단 per-class 규칙]
- 세션 프리픽스(`sess_...`)가 데이터의 자연스러운 그룹이다. 동일 세션의 여러 step을 행 단위로 무작위 분할하면 대화·행동 전이가 train과 validation에 함께 들어가 점수가 부풀려진다. 이 문제 때문에 모든 정식 CV는 session-prefix GroupKFold/StratifiedGroupKFold로 정의했다. [근거: D-003]

## 2. 데이터 성질을 확인한 방법

### 2.1 포렌식 r1: 결정 규칙의 부재

- r1은 상태→행동을 거의 그대로 복원하는 고순도 규칙이 있는지 확인했다. 템플릿·workspace·마지막 행동·탐색 계열 전이를 조사했지만, 전체 14클래스에 넓게 적용 가능한 결정 구간은 발견하지 못했다. `ask_user`도 특정 문형에서 purity가 올라가지만 0.99 수준의 안전한 신호가 아니며, explore 전이 신호도 “이미 explore family로 판별된 뒤”에만 강했다. [근거: `context/reports/forensics_r1.md` §§(b)~(f)]
- 결론은 규칙 기반 override가 아니라, 텍스트·history·meta를 함께 보는 모델과 검증된 서브모집단 라우팅을 사용해야 한다는 것이다. 결정 규칙 노선은 D-007에서 기각됐다.

### 2.2 포렌식 r2: ask_user/plan_task 착시

- r2에서는 질문부호, wh 의문사, plan 관련 어휘, `last_action`/`last2_action`, budget·elapsed·turn_index를 다시 스캔했다. 14클래스 전체 기준으로 purity가 0.52를 넘는 조건은 없었다. ask_user와 plan_task만 떼면 0.79~0.89가 보이는 조건도 있었지만 실제 모집단은 다른 12개 클래스가 대부분이었다. [근거: `context/reports/forensics_r2.md` §§(b)~(d)]
- 예를 들어 `last_action=='none' → plan_task` 후보는 90행 중 19행만 올바르게 바꾸고 71행을 망쳤다. 따라서 H1은 현재 입력 표면의 규칙/override가 아니라 annotation-contract 감사 대상으로만 남겼다. [근거: `forensics_r2.md` §(c), CX-003]

### 2.3 sim/au 이질성

- id prefix의 `sess_sim`과 `sess_au`는 같은 14클래스 문제 안에서도 난이도가 다르다. AU 5,025행을 별도 분석한 결과, AU에서 모든 기존 성분의 성능이 SIM보다 약했고 전 성분 공통 약세였다. 이는 특정 인코더의 프록시 오류가 아니라 실제 하위 모집단 차이라는 근거가 됐다. [근거: exp #23]
- 이 관찰은 단순히 AU를 버리는 결론이 아니라, 결정적 키로 식별 가능한 모집단에는 전용 모델을 학습하고 확률 수준에서 기존 모델과 결합할 수 있다는 설계 근거가 됐다.

## 3. 형제 행 라벨 구조와 한계

- train에서 같은 세션의 step k 라벨은 다음 step history의 마지막 `assistant_action.name`으로 복원되는지 검사했다. 58,326/58,326 쌍(100.00%)이 성립했고, gap 1~6의 231,664쌍에서도 예외가 없었다. [근거: exp #12; D-008]
- 그러나 test는 세션당 1개 step을 샘플링하므로 test 행에는 형제 행이 없다. 실제 제출에서 형제 라벨 복원을 켠 결과 LB 변화가 0이었다. 이 구조는 train 내부 전이의 강함과 test-time 이용 가능성을 구분해야 한다는 교훈이다. 보험 코드는 남겼지만 점수 레버로는 사용하지 않는다. [근거: 제출 대장 #2; D-008]
- 같은 이유로 train 전체를 보고 얻는 관측 세션 길이·상대 turn은 descriptive feature일 뿐, one-row-per-session 평가에서 그대로 inference feature로 가정할 수 없다. [근거: CX-003 §Step, turn, observed session length; exp #36]

## 4. AU 발견에서 모델 설계로

1. AU 전용 linear를 비holdout AU 세션으로 학습하고 session-group OOF로 점검했다. 초기 OOF AU F1은 0.68001이었고, holdout AU에서 기존 blend의 F1은 0.51381→0.69035로 올랐다. 전체 리그 상승은 +0.00935였다. [근거: exp #23]
2. reviewer가 최초 OOF 학습에 holdout 혼입 가능성(약 13%)을 발견했다. 비holdout 4,343행만으로 완전 격리해도 상승이 +0.01143으로 유지되어, 발견이 누수 부산물이 아님을 재검증했다. [근거: exp #23 독립 검증]
3. 하드 교체보다 char-only `C=1.0` 전용 확률을 기존 확률과 soft 결합하고 `alpha=0.9`를 선택했다. 리그에서 하드 대비 +0.0065, LB에서 0.73310→0.7400(+0.0069)로 전이됐다. [근거: exp #24; 제출 대장 #5~#6]

이 사례는 “전체 데이터에 임의 규칙을 덧씌운다”가 아니라, (a) 키가 사전에 정해져 있고, (b) 해당 모집단에서 전 성분 공통 약세가 관측되며, (c) 격리 학습과 확률 결합이 독립적으로 검증된 경우에만 라우팅한다는 원칙을 보여준다. 반대로 first-step 라우팅, explore specialist, template override는 이 조건을 충족하지 못했다. [근거: exp #8, #25, #40; D-007]

## 5. 데이터 분석 결론

- 데이터의 핵심은 표면 문장 하나의 결정 규칙이 아니라 세션 그룹 구조, history의 길이·정보량, sim/au 모집단 차이, 그리고 Macro-F1이 요구하는 희소 클래스 보호다.
- 분석 결과는 챔피언의 세 가지 선택으로 연결됐다: 세션 기준 누수 차단, e5에만 hist12를 허용하는 per-encoder 직렬화, AU의 soft 라우팅. [근거: exp #34~35; D-010; 제출 대장 #11]
- r1/r2와 D-008은 “보이는 규칙”의 과잉 해석을 막는 반증 자료이며, AU 사례는 데이터 분석이 실제 알고리즘 개선으로 이어진 양성 사례다.
