# 알고리즘 초안 — 이질 성분 blend와 제한 조건 하의 국소 최적

> 전문가심사 알고리즘 15점 대응용 초안. 이 문서는 제출 챔피언과 진단용 실험을 구분한다.

## 1. 챔피언 구조

현재 챔피언은 다음 4-way 확률 구조다.

```text
linear(E_+seq) + AAR stacker + multilingual-e5-base(hist12) + mBERT(hist6)
                                  └ e5:mBERT block = 1.2:0.8
AU(sess_au)에서는 char_wb 전용 확률을 soft-AU alpha=0.9로 결합
```

- linear는 sparse text/history/meta view, AAR stacker는 이종 view와 transition prior, e5는 dense semantic/history, mBERT는 e5와 다른 다국어 오류를 담당한다. 챔피언 점수는 **0.7623**, 이전 우리 0.7480과 동료 0.7511을 넘었다. [근거: exp #35; 제출 대장 #11; D-013]
- 직렬화는 모델별 학습 계약을 보존한다. e5는 `max_hist=12`, mBERT는 `max_hist=6`으로 유지한다. 한 serializer를 두 encoder에 공통 적용하면 train↔infer 불일치가 생기므로 `per-encoder serialize`를 사용했다. [근거: exp #34~35; D-010]
- AU에서는 `sess_au` 키로 char_wb 전용 확률을 식별하고 기존 blend와 `alpha=.9` soft 결합한다. 하드 교체보다 리그 +0.0065가 좋았고 LB 0.7400까지 올렸다. [근거: exp #23~24; 제출 대장 #5~#6]

## 2. 성분을 선택한 근거

| 성분/결정 | 근거 | 채택 이유 |
|---|---|---|
| E_+seq linear | exp #1, #32 | baseline 재현과 sparse 보완성 유지; text-only replacement는 B4 리그 −0.00051로 기각 |
| AAR stacker | exp #2, 기존 챔피언 lineage | transition/다중 view를 보완; 새 hist12 stacker CX-002는 진단용으로만 보류 |
| e5 hist12 | exp #34~35 | hist6 대조 4-way+AU 0.73451→0.75601, 리그 +0.02150, LB +0.0143 |
| mBERT hist6, block 0.8 | exp #27~28 | holdout 리그 e5:mBERT 1.2:0.8에서 +0.00529, LB 0.7467; full-train으로 0.7480 |
| AU char soft route | exp #23~24 | AU 공통 약세를 격리 학습으로 보정, LB +0.021 누적 효과(하드 +0.0142, soft 추가 +0.0069) |

e5 hist12은 #15의 BoW proxy 실패를 그대로 반복하지 않고, 동일 encoder 레시피에서 hist6 대조군과 85% holdout을 동시에 학습해 판정했다. 12턴 확장에서 384 토큰 잘림은 8.5%로 측정됐고, 인코더 리그에서 처음으로 serialize 확장의 blend 전이가 확인됐다. [근거: exp #15, #34]

## 3. 탐색 공간과 폐기된 대안

- **규칙/threshold/prior/calibration:** r1/r2의 purity 착시, first-step route, meta-selector, template override가 모두 안전한 범용 규칙이 아님을 보였다. D-009는 threshold·prior·calibration 가족을 폐기했다. [근거: D-007, D-009; exp #9, #17, #25, #40]
- **encoder 입력 강화:** maxlen512(#41), mBERT hist12(#42), e5 args-lite(#43)는 solo 성능이 올라도 4-way blend에서 각각 −0.00102, −0.00106, −0.00268이었다. 세션 길이 weighting(#44~45)과 seed probability average(#48)도 5지표 게이트 미달이었다. 따라서 e5 입력 변형·시드 증설은 중단했다.
- **백본 대형화:** klue/roberta-large hist12(#49)는 solo 0.71271로 e5 0.73617보다 −0.023, blend row −0.00745, bootstrap CI도 전부 음수였다. 큰 모델이 자동으로 이질성을 주지 않는다는 결론이다.
- **새 stacker:** CX-002/#46의 hist12 stacker는 meta-CV 0.73275 대 diagnostic proxy blend 0.73830으로 −0.00544였다. read/list 계열 개선 신호는 남았지만 alpha09 sparse OOF와 frozen shadow/outer evaluation 없이는 승격하지 않는다. #47 local corrector도 row −0.01397, confidence override 전부 음수라 P1-C를 종결했다. [근거: CX-002~003; exp #46~47]
- **모델 추가:** e5-small(#18, #21), char fourth component(#22), mdeberta(#26), 한국어 encoder 3종(#31)은 단독 성능 또는 다양성만으로는 blend 개선을 만들지 못했다. mBERT는 “단독 최고”가 아니라 e5와의 보완성과 zip 예산을 함께 만족한 예외다.

이 탐색 결과에서 국소 최적이라는 표현은 전역 최적 증명이 아니다. 현재 확보된 성분·리그·제출 예산 안에서, 실제 LB 전이와 독립 검증을 동시에 통과한 조합이 챔피언이라는 뜻이다. [근거: D-013]

## 4. 배포 제약을 반영한 설계

- 평가 서버는 T4 16GB, 3 vCPU, 12GB RAM, 오프라인이며 추론 10분·zip 1GB 제한이다. 따라서 모든 encoder weight를 패키지하고 network call을 금지하며, serializer·action 순서·sample_submission 형식을 고정한다. [근거: `CLAUDE.md`; `PLAN.md`]
- mBERT 포함 챔피언 zip은 867.9MB, 서버 추론은 4분14초였다. e5 573MB와 mBERT 339MB를 함께 탑재하면서도 1GB 아래에 남고, e5를 한 번 더 넣는 seed ensemble은 1GB를 초과하므로 증류 없이는 배포하지 않는다. [근거: exp #27~28, #35, #48]
- e5와 mBERT를 같은 긴 history로 재학습하지 않은 것은 단순 성능 타협이 아니라 serialization contract와 실행시간을 함께 지키기 위한 결정이다. e5 hist12의 실효 이득은 확인했지만 mBERT hist12는 blend −0.00106이었다. [근거: exp #42]
- 제출 전에는 G1 tests, G4 출력/오프라인/시간 게이트, zip 크기와 독립 reviewer/tester 확인을 통과해야 한다. exp #35에서 G1 22 tests, G4 12/12, 오프라인 50.2초, 867.9MB, tester 11/11을 기록했다. [근거: exp #35; 제출 대장 #11]

## 5. 결론과 재현성

- 알고리즘의 핵심은 “큰 단일 모델”이 아니라 서로 다른 오류 표면을 가진 sparse, transition, e5 dense, mBERT dense를 계약이 맞는 방식으로 합치고, 데이터에서 검증된 AU key에만 국소 라우팅을 적용하는 것이다.
- 재현 코드 관점에서는 linear/e5/mBERT/AU char 학습 경로는 확인됐지만 AAR stacker 원 트레이너는 현재 보유하지 않아 재현성 gap이 남는다. 상위 12팀 진입 시 7/20 10:00 재현 코드 제출을 대비해 이 gap을 별도 보험 트랙으로 관리한다. [근거: D-012; `context/coordination.md` 07-11 오후 노트]
- 새 제출은 자동 최종 선택을 바꿀 수 있으므로, public 점수 탐색이 아니라 5지표·shadow·제약 검증을 통과해도 최종 선택으로 남겨도 되는 후보에만 허용한다. 현재 판단은 D-013에 따라 0.7623 챔피언 단독 유지다.
