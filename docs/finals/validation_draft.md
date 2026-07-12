# 모델 검증 초안 — 누수 방지와 다섯 겹의 게이트

> 전문가심사 모델검증 10점 대응용 초안. 로컬 리그 점수는 LB가 아니며, 실험 가족별 전이 근거와 한계를 함께 제시한다.

## 1. 검증 단위와 누수 방어

- 기본 분할 단위는 `id`에서 `-step_\d+$`를 제거한 세션 prefix다. 같은 세션의 행을 train/valid에 나누지 않는 GroupKFold/StratifiedGroupKFold를 사용한다. [근거: D-003; `context/experiments.md` 검증 프로토콜]
- 행 단위 random split은 history와 전이 패턴을 공유하는 형제 행을 양쪽에 놓아 Macro-F1을 과대평가한다. 형제 라벨 복원 자체가 train 58,326/58,326쌍에서 100% 성립한 것은 이 누수 위험을 정량적으로 보여준다. [근거: exp #12; D-008]
- 별도 문제는 평가 표본 단위다. 리그는 9,969행/1,350세션으로 구성되지만 실제 test가 세션당 한 행이라는 가정은 직접 관측한 계약이 아니라 간접 추정이다. 따라서 group split만으로 충분하다고 주장하지 않고, session-uniform 점수와 one-row/session Monte Carlo를 병행한다. [근거: exp #36; `third_party_sol_model_audit_2026-07-10.md`]

## 2. 5지표 리그 게이트

후보는 row Macro-F1 하나가 아니라 다음 다섯 지표를 함께 통과해야 한다.

1. **row Macro-F1** — 기존 실험 대장과 직접 비교하는 기본 지표.
2. **session-uniform Macro-F1** — 세션별 점수를 먼저 계산해 긴 세션이 많은 행을 독점하지 않도록 한다.
3. **one-row/session MC200** — 세션마다 한 행을 뽑는 평가를 200회 반복해 표본 선택 변동을 본다.
4. **paired-session bootstrap** — control과 candidate의 같은 세션을 짝지어 2,000회 재표집하고 차이의 CI/P(Δ>0)를 계산한다. 일부 실험에서는 5,000회로 정밀 확인했다. [근거: exp #37]
5. **deterministic halves** — 세션을 고정된 두 반으로 나눠 개선 방향이 한쪽에만 의존하지 않는지 확인한다.

이 설계는 서로 다른 실패를 잡는다. row는 전체 평균, session-uniform은 세션 길이 편향, MC는 숨은 one-row 샘플링 변동, bootstrap은 paired 불확실성, halves는 특정 그룹 집중을 점검한다. 반반 안정성만으로는 같은 holdout 내부의 구조적 신기루를 못 잡으므로 단독 게이트로 쓰지 않는다. [근거: exp #20; audit report §3.2]

## 3. CV·리그·LB를 읽는 법

| 사례 | 로컬/리그 관측 | 실제 LB | 검증상 결론 |
|---|---:|---:|---|
| exp #19 history bucket | 리그 +0.0075 | 미제출 | enc 지분 하향 신호의 정황 추론을 LB 근거로 격상하면 안 됨 |
| exp #20 history_len3 | 리그 +0.0095, 반반 통과 | **−0.0061** (대장 #4) | 리그 내부 안정성도 전이 보장이 아님; enc 지분 축 금지 |
| exp #23 AU hard route | 리그 +0.00935~+0.01143 | **+0.0142** (대장 #5) | 결정적 키·전용 모델·격리 학습 라우팅은 전이 |
| exp #24 AU soft α=.9 | 리그 +0.0065 | **+0.0069** (대장 #6) | 라우팅 축의 두 번째 전이 확인 |
| exp #34 hist12 e5 | 리그 **+0.02150** | exp #35 **+0.0143** (대장 #11) | 인코더 가족 축은 할인되어도 방향이 전이 |

현재 기록된 가족별 로컬→LB 할인은 linear −0.002, base encoder −0.015, small/e8 encoder −0.019, stacker −0.033이다. 이는 보편 상수가 아니라 당시 핸드오프의 보수적 읽기 규칙이다. 리그에서 encoder 지분 최적이 0.33으로, LB 최적이 0.50으로 어긋난 사례가 있어 enc 지분 축은 리그를 방향 선택기로만 사용한다. [근거: `context/experiments.md` “로컬 CV → LB 할인율”]

## 4. 신기루와 방어 규칙

- **#19~20:** history 버킷 점수와 반반 안정성은 모두 좋아 보였지만 LB에서 완전히 역전됐다. 원인은 85% proxy encoder가 실전 full encoder와 다른 축을 만들기 때문이다. 이후 전역·버킷 enc 지분 하향은 재시도 금지로 고정했다. [근거: exp #16, #19, #20; D-009]
- **단독 성능 함정:** mBERT 3ep는 단독 0.67147→0.69117로 올랐지만 blend 개선은 +0.00070뿐이었다. #41, #42, #43~45, #48도 solo 상승이 blend에서 역전되거나 게이트 미달이었다. “성분 F1 상승”을 “앙상블 상승”으로 보고하지 않는다. [근거: exp #29, #41~45, #48]
- **진단과 승격 분리:** CX-002/#46 stacker는 메타-CV 0.73275로 diagnostic 50:50 proxy 0.73830보다 −0.00544였고, baseline이 alpha09 sparse OOF가 아닌 legacy proxy라 promotion-eligible이 아니다. reviewer/tester가 누수·재현은 PASS했지만 점수 승격은 하지 않았다. [근거: CX-002; exp #46]
- **고정 holdout 재사용 금지:** 후보 탐색용 holdout을 다시 카드 검증에 사용하지 않는다. 새 session-group shadow 또는 새 annotation-contract population을 만들고 다섯 지표를 재계산해야 한다. [근거: CX-003 §Population and scoring contract]

## 5. 독립 검증 운영

- 작성자와 다른 reviewer/tester가 각각 검증한다. reviewer는 누수, split, 피처 정합, 스코프·해석을 보고 tester는 명령 재실행, hash/shape/label/action 순서, 결정론성을 확인한다. 이는 `CLAUDE.md`와 `context/coordination.md`의 강제 규칙이다.
- 실제 사례에서 reviewer가 #23의 AU OOF holdout 혼입을 발견했고, tester가 격리 재검증 수치를 재현했다. #35는 BOM 폴백 버그를 smoke test가 발견해 수정한 뒤 tester 11/11, 오프라인 50.2초, 867.9MB 게이트를 통과했다. [근거: exp #23, #35]
- CX-003은 reviewer/tester가 전체 수치와 dead-axis 스코프를 확인했지만, `id↔probs` 내부 행 대응은 생성기 신뢰 경계로 남겼다. 따라서 해시 manifest와 생성 시 ids/probs 동시 저장을 요구한다. [근거: CX-003; `context/coordination.md` 07-11 새벽 노트]

## 6. 판정 원칙

1. group split이 없는 점수는 정식 근거로 쓰지 않는다(D-003).
2. 리그 후보는 5지표와 per-class F1을 함께 본다. 단일 row 상승이나 solo 상승은 승격 사유가 아니다.
3. 리그에서 방향을 고른 뒤 LB에서 확인하되, 실패 축은 재시도 금지표에 기록한다(D-009).
4. 챔피언 exp #35는 리그 +0.0215, LB +0.0143, 5지표·독립 검증을 모두 갖췄고, D-013에 따라 0.7623 챔피언 단독 유지로 결론냈다.
