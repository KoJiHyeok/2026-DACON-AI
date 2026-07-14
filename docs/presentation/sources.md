# 본선 발표 근거 사료집

이 문서는 전문가심사 40점의 네 항목을 저장소 기록에 매핑한 발표용 원천 문서이며, 확정되지 않은 해석은 결론으로 쓰지 않는다. `(exp #N)`은 `context/experiments.md`, `(D-NNN)`은 `context/decisions.md`, `(대장 #N)`은 `context/submissions.md`, `(reports/...)`는 `context/reports/...`를 뜻한다. (D-001)

## 1. 데이터분석 — 10점

### 1.1 문제와 데이터 구조

- 입력은 `current_prompt`, 최대 12개 history 항목, session meta를 사용해 다음 행동 14개 중 하나를 맞히는 70,000행 Macro-F1 문제이며, 데이터에는 9,429개 세션이 있다. 세션당 스텝은 평균 7.42, 중앙값 7, 최대 18이고 history 길이는 0~12의 짝수로 캡된다. (reports/eda_distribution.md)
- 클래스 빈도는 `edit_file` 11,171행(15.96%)에서 `web_search` 1,273행(1.82%)까지 8.8배 차이가 나므로, 정확도보다 14개 클래스의 F1을 똑같이 평균하는 Macro-F1 관점이 중요하다. (reports/eda_distribution.md)
- 희소 5개 클래스는 `web_search` 1.82%, `write_file` 2.12%, `lint_or_typecheck` 3.26%, `plan_task` 3.83%, `ask_user` 3.86%이며, 첫 턴 9,000행에서는 `list_directory` 20.2%, `plan_task` 12.5%, `write_file` 7.9%로 전체 분포와 달라진다. (reports/eda_distribution.md)
- 직전 행동만 쓴 최빈 예측 상한은 22.9%, 직전 두 행동의 바이그램은 28.6%여서 구조 피처만으로 충분하지 않지만, `read/grep→edit`, `edit→run_tests`, `list→read`, `glob→grep` 같은 전이 신호는 sparse 성분의 보조 피처가 된다. (reports/eda_distribution.md)
- session meta의 연관 강도는 open-files 수 0.291, 직전 행동 바이그램 0.274, turn-index 구간 0.273, git-dirty 0.260 순이었고 user-tier 0.022, language-pref 0.021은 거의 무관했다. (reports/eda_distribution.md)

### 1.2 시뮬레이터 포렌식: 규칙을 찾기보다 규칙 가설을 기각한 과정

- 1라운드는 70,000행·9,429세션에서 20종 state 정의를 검사했으며, purity≥0.99인 구조 규칙의 합집합 coverage는 min-rows≥5에서 0.03%(21행), min-rows≥10에서 0.0157%(11행), min-rows≥12에서 0%였다. 따라서 “결정적 state→action 규칙으로 점수 갭을 닫는다”는 가설을 기각했다. (reports/forensics_r1.md) (D-007)
- prompt 템플릿까지 포함한 purity≥0.99 coverage는 7.40%(5,181행)였지만 그중 49.7%(2,575행)는 이미 포화된 `respond_only`였고, 비-`respond_only` 순증 후보는 3.72%(2,606행)의 작은 템플릿들에 흩어져 있었다. (reports/forensics_r1.md)
- 포렌식은 `sess_sim` 64,975행·8,330세션과 `sess_au` 5,025행·1,099세션이라는 두 생성계열을 발견했으며, AU는 `read_file` 25.7% 대 SIM 12.3%, `glob_pattern` 1.8% 대 8.0%, `list_directory` 2.2% 대 6.5%로 라벨 분포가 달랐다. 이 결정적 ID prefix가 이후 AU specialist 라우팅의 출발점이 됐다. (reports/forensics_r1.md) (exp #23)
- `respond_only`는 관측된 세션의 마지막 step에서만 5,178/5,178건 등장했지만, 공개 test 5행은 각기 다른 세션의 1행뿐이어서 이 규칙을 숨은 테스트에 일반화하지 않았다. (reports/forensics_r1.md) (D-008)
- 2라운드는 `ask_user↔plan_task` 혼동을 겨냥했지만 질문부호·의문사·계획어의 14클래스 purity는 대부분 0.19~0.40였고, 실제 191개 오류행에서 `last_action=='ask_user'→plan_task`는 10건 중 9건을 악화시켰다. 후보 C1~C3는 모두 기각됐으며 독립 reviewer가 191행 재구성과 전 수치를 재현했다. (reports/forensics_r2.md)

### 1.3 hist12: 잘림 통념을 실제 토크나이저로 뒤집다

- 초기 exp #15는 args·language·elapsed를 함께 넣은 BoW 프록시에서 hist12가 −0.031이었고 “83% 잘림”을 폐기 근거로 삼았지만, 이는 hist 확장의 단일변수 실험도 아니고 encoder 판정도 아니었다. (exp #15) (D-010)
- 실제 e5 tokenizer 12k 표본에서는 세션의 51%가 6턴을 넘고 정확히 12턴인 비율이 30.7%였으며, hist12의 384-token 잘림률은 8.5%뿐이었다. 6턴 초과 세션은 평균 4.5턴을 더 살렸으므로 “83% 잘림” 가설은 재현되지 않았다. (exp #34) (D-010)
- 같은 85% split·같은 e5 레시피에서 history만 6→12로 바꾸자 e5 solo가 0.70066→0.73617, 최종 4-way+AU가 0.73451→0.75601로 올라 격리 델타 +0.02150과 반반 +0.018/+0.025를 기록했다. (exp #34)
- 이득은 모든 클래스에 균일하지 않았고 `web_search` +0.082, `apply_patch` +0.056, `lint_or_typecheck` +0.045, `run_tests` +0.033이 컸으며 `read_file`·`grep_search`는 각각 약 −0.001로 거의 변하지 않았다. (reports/recon_complementarity_confusion_2026-07-09.md)
- 결론은 “history는 길수록 좋다”가 아니라 “최근 맥락의 누락을 attention encoder 한 곳에서 복원했을 때만 앙상블에 상보적으로 기여했다”이다. e5 maxlen512는 solo +0.0027에도 blend −0.00102였고, mBERT hist12는 solo +0.02276에도 blend −0.00106이었다. (exp #41) (exp #42)

## 2. 모델검증 — 10점

### 2.1 세션 누수 방지와 평가 단위

- 같은 세션이 최대 18개 step으로 반복되므로 id에서 `-step_\d+$`를 제거한 세션 prefix를 그룹키로 쓰는 GroupKFold가 기본 계약이며, 행 랜덤 분할은 같은 대화의 앞뒤 state를 train과 valid에 나눠 누수시킨다. (reports/eda_distribution.md) (D-003)
- 정식 검증 프로토콜은 StratifiedGroupKFold, 9,429개 세션 그룹, Macro-F1과 per-class F1이며 중요 판정은 3-fold 이상으로 본다. (exp #51)
- 세션 누수 방지와 숨은 테스트의 표본단위 일치는 별개이므로, 고정 홀드아웃 9,969행·1,350세션을 row와 세션 관점에서 함께 읽었다. 행 분포와 세션당 한 행 기대분포는 `respond_only` +2.94%p, `apply_patch` −1.29%p, `grep_search` −0.98%p, `edit_file` −0.71%p 차이가 났다. (reports/third_party_sol_model_audit_2026-07-10.md)

### 2.2 5지표 승격 게이트

- 후보는 최종 blend 표면에서 ① row Macro-F1, ② session-inverse-weight Macro-F1, ③ 세션당 한 행 Monte Carlo 평균·표준편차, ④ paired-session bootstrap 95% CI와 P(Δ>0), ⑤ 고정 홀드아웃 반반 안정성을 함께 본다. (reports/third_party_sol_model_audit_2026-07-10.md) (exp #43)
- hist12는 row +0.02150, 세션균등 +0.01859, 세션당 한 행 MC +0.01860±0.00863, bootstrap 95% CI [+0.015104,+0.027827], 반반 +0.018/+0.025로 큰 구조적 신호였다. (exp #34) (exp #36) (reports/third_party_sol_model_audit_2026-07-10.md)
- 반대로 args-lite는 solo +0.0035였지만 다섯 지표가 row −0.00268, 세션균등 −0.00096, MC −0.00109±0.00547, CI [−0.00751,+0.00176], 반반 −0.0058/−0.0003으로 모두 실패해 제출하지 않았다. (exp #43)
- 세션길이 역가중은 일부 보조지표가 양수여도 row −0.00231, CI [−0.00721,+0.00271], 반반 −0.00329/−0.00159였으므로 승격하지 않았다. 이는 “solo 상승”이나 “양수 지표 개수”가 아니라 전체 표면의 강건성을 판정한다는 사례다. (exp #45)

### 2.3 holdout→LB 전이와 불확실성의 실측

- hist12의 holdout blend +0.02150은 LB에서 +0.0143으로 전이돼 약 67% 전이율을 보였고, 라우팅 축의 과거 1:1 전이와 달리 encoder-family 델타는 할인해서 읽어야 한다는 근거가 됐다. (exp #35) (대장 #11)
- KD born-again student는 solo +0.0138, holdout row +0.00360, 5지표 방향 전부 양수였지만 bootstrap CI [−0.00078,+0.00809]가 0을 포함했다. LB는 −0.0002로 끝나 CI가 경고한 비전이 시나리오가 실제로 발생했다. (exp #51) (대장 #12)
- 따라서 작은 양수 델타를 제출 신호로 보지 않고, encoder-family 전이 할인과 불확실성을 흡수하는 row≥+0.005·CI 하한>0 같은 엄격 문턱을 유지한 이유를 실측으로 설명할 수 있다. (exp #51)
- Qwen 하이브리드는 row +0.01160, 세션균등 +0.01393, MC +0.01430±0.00685, CI [+0.00614,+0.01758], 반반 +0.01562/+0.00753으로 #34 이후 처음 엄격 게이트를 통과했다. #14의 LB 이득 +0.00853은 holdout 이득의 73.5%로, hist12의 67% 할인과 같은 방향의 두 번째 데이터포인트가 됐다. (exp #52) (대장 #14)

### 2.4 작성자와 검증자 분리

- hist12 배포는 tester가 전체 22/22 테스트를 통과시키고 reviewer가 per-encoder serialize diff에서 finding 0건을 확인했으며, 모델 SHA256·BOM 없는 config·e5=12/mBERT=6 폴백 계약까지 독립 대조한 뒤 제출됐다. (reports/verify_hist12_deploy_2026-07-10.md) (exp #35)
- 5성분 parity 스태커는 작성자 2회 실행이 바이트 동일했고, 별도 reviewer가 수치를 소수 5자리까지 재현하고 meta-train과 holdout의 세션 교집합 0을 독립 계산했지만 결과가 row −0.00164여서 폐기됐다. 검증 통과는 코드·수치의 신뢰성을 뜻할 뿐 모델 승격을 뜻하지 않는다. (exp #50)
- 포렌식 2라운드도 reviewer가 191개 오류행과 모든 표 수치를 재구성했으며, 출처가 r1 산출물인 표와 스크립트 docstring 결함까지 분리해 기록했다. (reports/forensics_r2.md)

## 3. 알고리즘 — 15점

### 3.1 이전 챔피언: 5성분 이종 앙상블

- 제출 #11 비교 기준의 비-AU 확률은 Linear 25%, AAR stacker 25%, e5-base 30%, mBERT 20%의 이종 결합이다. Linear는 word/char TF-IDF+history/action/meta LinearSVC, AAR은 prompt/context/action/transition 4-view stacker, e5와 mBERT는 서로 다른 tokenizer와 표현공간의 encoder다. (reports/third_party_sol_model_audit_2026-07-10.md) (대장 #11)
- AU 행은 `sess_au`라는 결정적 prefix로 식별하고 char-TFIDF LinearSVC specialist의 확률을 α=0.9로 결합하며, 이 specialist의 AU 실효 지분은 90%다. (exp #24) (reports/third_party_sol_model_audit_2026-07-10.md)
- encoder block의 내부 가중은 e5:mBERT=1.2:0.8이며, 전체 [linear,AAR,encoder-block]=[1,1,2] 구조에 넣으면 비-AU 실효 지분이 25%/25%/30%/20%가 된다. (exp #27) (reports/third_party_sol_model_audit_2026-07-10.md)
- 직렬화는 encoder별로 분리해 e5는 history 12개, mBERT는 학습 계약과 같은 history 6개를 사용한다. 하나의 global max-history를 바꾸면 mBERT train↔infer 계약이 깨지므로 config 누락 시 6으로 폴백한다. (exp #34) (exp #35) (reports/verify_hist12_deploy_2026-07-10.md)

### 3.2 시간초과 대역전: #13 FAIL에서 #14 팀 최고까지

- exp #52는 유일하게 미검증이던 모델 패밀리를 Qwen2.5-0.5B 디코더로 바꿨다. instruct/base 2ep의 h85 solo는 0.75932/0.75941로 e5 0.73617보다 약 +0.023 높았고, linear+AAR+Qwen block+soft-AU 하이브리드는 mBERT를 빼고도 holdout 0.75601→0.76760, Δ +0.01160을 만들었다. (exp #52)
- 첫 배포 #13은 패키지 게이트를 통과했지만 평가 T4에서 추론 10분을 넘겨 채점되지 않았다. 챔피언 4분14초에서 Qwen도 4~5분일 것이라는 외삽은 디코더 24층과 hist12 장문 시퀀스를 빠뜨린 “파라미터 등가 ≠ 연산 등가” 오판이었다. (대장 #13) (daily 07-13)
- 연산활성량은 Qwen 360M/24L, 이전 e5·mBERT는 각각 약 86M/12L였다. 파라미터 파일 크기가 비슷해도 매 행에서 실제로 통과하는 층과 활성 가중치가 달라, 신규 모델 패밀리의 시간은 패키지 크기나 5행 smoke로 추정할 수 없었다. (exp #52) (docs/t4_rehearsal.md)
- 복구 레버는 두 가지였다. 길이정렬 배칭은 hist12 평균 220token·384 cap 도달 5.3%에서 패딩 연산을 1.70x 줄였고, fast_aar는 리허설 분해 기준 구경로 약 84초를 29.7초로 줄여 약 2.8x를 냈다. 두 경로 모두 출력 순서 복원과 확률·argmax 등가를 reviewer/tester가 확인했다. (exp #52) (daily 07-13) (docs/t4_rehearsal.md)
- 동일 T4에서 30,000행을 재현한 Colab 리허설은 로드 6.6초, linear 7.0초, fast_aar 29.7초, Qwen 471.5초, 총 515초(8.6분)를 기록했다. 이 실측 후 #14를 재제출해 LB 0.77089, 기존 0.76236 대비 +0.00853, 89→79등, 컷 갭 −0.0229를 달성했다. (docs/t4_rehearsal.md) (대장 #14) (daily 07-13)
- 이 사건의 배포 규칙은 단순하다. 새 모델 패밀리나 큰 연산경로 변경은 평가 서버와 동급인 T4·실평가 30,000행으로 먼저 재고, 출력등가를 보존한 속도 최적화만 승격한다. (exp #52) (docs/t4_rehearsal.md)

### 3.3 0.71884→0.77089 승격 궤적

- 재건 기준선은 3-way [1,1,2]에서 LB 0.71884였고, sibling label recovery는 같은 0.71884로 이득 0을 확인했다. (대장 #1) (대장 #2)
- e5-small 추가는 0.71280, history-length bucket weighting은 0.71270으로 하락해 “성분 추가”와 “로컬 가중 최적화”가 자동으로 전이되지 않음을 일찍 확인했다. (대장 #3) (대장 #4)
- hard-AU 라우팅은 0.71884→0.73310(+0.0142), soft-AU α=0.9는 0.73310→0.7400(+0.0069)으로 두 번 연속 팀 최고를 갱신했다. (대장 #5) (대장 #6)
- mBERT 60k holdout-trained 성분은 0.7467(+0.0067), full-train 70k 교체는 0.7480(+0.0013)을 만들었고, 전자는 서버 추론 4분14초와 zip 867.9MB로 10분·1GB 제약 안에 들어왔다. (대장 #7) (대장 #8)
- 동료 surface에 mBERT를 섞은 병합과 α 변경은 각각 0.7503과 0.7501로 동료 기준 0.7511보다 낮아 surface 의존성을 확인했고, 최종 hist12가 0.7480→0.7623(+0.0143)으로 팀 최고를 탈환했다. (대장 #9) (대장 #10) (대장 #11)
- 주요 구조적 승격은 hard-AU, soft-AU, mBERT, hist12의 네 전환으로 요약되며, 기준선 0.71884에서 최종 0.7623까지 +0.04346을 만들었다. 이 “4건”은 실험 행의 PASS 개수가 아니라 감사 보고서가 압축한 네 구조적 점프를 뜻한다. (reports/third_party_sol_model_audit_2026-07-10.md) (대장 #1) (대장 #11)
- KD student #12는 0.7621로 비승격됐고, Qwen #13은 시간초과로 채점되지 않았다. 속도레버를 적용한 #14가 0.77089를 기록하면서 기준선 0.71884 대비 누적 +0.05205, hist12 챔피언 대비 +0.00853을 완성했다. (대장 #1) (대장 #12) (대장 #13) (대장 #14)

### 3.4 실패를 자산으로 만든 폐기 서사

- 실험 대장 #1~#52의 핵심 교훈은 단일 성분 성능보다 최종 앙상블 기여와 오류 상보성이 중요하다는 것이다. mBERT 3ep는 solo +0.0197에도 blend +0.00070, 한국어 encoder 후보 최고는 solo 0.68147에도 blend +0.00154, maxlen512는 solo +0.0027에도 blend −0.00102였다. Qwen은 반대로 solo 약 +0.023뿐 아니라 하이브리드 blend +0.01160까지 함께 확보해 승격됐다. (exp #1) (exp #29) (exp #31) (exp #41) (exp #51) (exp #52)
- 시드 확률 평균은 solo +0.0041에도 blend row +0.00045와 CI [−0.00311,+0.00379]에 그쳤고, 2×e5는 1GB를 넘으므로 직접 배포할 수도 없었다. (exp #48)
- klue/roberta-large hist12는 solo 0.71271로 e5보다 −0.023 낮고 다섯 지표가 모두 음수여서 백본 대형화 축을 종결했다. (exp #49)
- 5성분 parity OOF와 frozen shadow로 #46의 proxy-baseline·nested-evaluation 결함을 해소한 스태커도 row −0.00164, CI [−0.00625,+0.00295]였으므로 “정확한 스태킹 구현”과 “챔피언보다 좋은 결정경계”가 다름을 확인했다. (exp #46) (exp #50)
- KD는 잘못된 tokenizer vocab=5가 teacher accuracy 6% 붕괴를 일으킨 실패를 assert로 차단한 뒤 student solo 0.74994를 얻어 born-again 효과는 입증했지만, blend LB가 −0.0002여서 배포 승격은 하지 않았다. (exp #51)
- 현재 최고 선택은 Qwen 하이브리드 #14의 0.77089이며, #13 시간초과도 채점 없음으로 이전 0.7623을 보존했다. 실패를 안전하게 만들고 엄격 게이트·동급 하드웨어 리허설을 통과한 변경만 최고점을 바꾸는 운영 원칙은 그대로 유지됐다. (대장 #13) (대장 #14)

## 4. 전달력 — 5점

### 4.1 발표 스토리라인 초안

1. **문제:** 14개 행동을 Macro-F1로 맞히되 70,000행이 9,429세션에 반복되고 클래스 빈도 차이가 8.8배여서, 행 랜덤 분할과 정확도 중심 평가는 위험하다고 제시한다. (reports/eda_distribution.md)
2. **탐색:** 결정 규칙 가설을 purity×coverage로 시험해 0.03% 상한으로 기각하고, 대신 SIM/AU 생성계열과 history 누락이라는 재현 가능한 구조를 발견했다고 전개한다. (reports/forensics_r1.md) (D-010)
3. **체계:** GroupKFold와 5지표 게이트, reviewer/tester 분리로 “좋아 보이는 후보”를 제출 가능한 후보로 좁히는 과정을 보여준다. (D-003) (exp #43) (reports/verify_hist12_deploy_2026-07-10.md)
4. **알고리즘:** sparse·전이·subpopulation specialist 위에서 두 encoder 블록을 Qwen 단일 블록으로 치환해 오류 상보성과 1GB 제약을 함께 지킨 구조를 설명한다. (reports/third_party_sol_model_audit_2026-07-10.md) (exp #52)
5. **결과:** LB 0.71884→0.77089의 계단 끝에서 #13 시간초과를 원인 규명→출력등가 속도레버→T4 30k 리허설→#14 성공으로 당일 뒤집은 장면을 결론으로 둔다. (대장 #1) (대장 #13) (대장 #14) (daily 07-13)

### 4.2 그림·표 후보와 원천 데이터

| 시각물 | 전달할 메시지 | 원천 데이터 |
|---|---|---|
| 클래스 분포 막대그래프 | 8.8배 불균형과 Macro-F1의 의미 | `context/reports/eda_distribution.md` §1 (reports/eda_distribution.md) |
| 세션 구조 도식 | 70,000행이 9,429세션·최대 18 step으로 반복돼 GroupKFold가 필요한 이유 | `context/reports/eda_distribution.md` §0·§2 (reports/eda_distribution.md) |
| purity×coverage 깔때기 | 구조 규칙 0.03%→0.0157%→0%로 가설을 기각한 과정 | `context/reports/forensics_r1.md` §b (reports/forensics_r1.md) |
| SIM 대 AU 분포 비교 | 결정적 prefix와 specialist 라우팅의 데이터 근거 | `context/reports/forensics_r1.md` 부가 발견 (reports/forensics_r1.md) |
| 이전→현 챔피언 아키텍처 | 25/25/30/20 5성분의 두 encoder가 Qwen 단일 block으로 치환되는 변화 | `context/reports/third_party_sol_model_audit_2026-07-10.md` §1 + exp #52 (reports/third_party_sol_model_audit_2026-07-10.md) (exp #52) |
| LB 계단 그래프 | #1~#14의 실패를 포함한 0.71884→0.77089 궤적 | `context/submissions.md` #1~#14 (대장 #1) (대장 #14) |
| 5지표 레이더 또는 소형 다중표 | hist12 PASS와 args-lite FAIL의 강건성 차이 | `context/experiments.md` #34·#43 (exp #34) (exp #43) |
| holdout→LB 산점/화살표 | hist12 67%, KD 비전이, Qwen 73.5%의 세 데이터포인트 | `context/experiments.md` #35·#51·#52 (exp #35) (exp #51) (exp #52) |
| 시간초과 대역전 타임라인 | #13 FAIL→원인→1.7x/2.8x 레버→T4 515s→#14 0.77089 | `context/daily/2026-07-13.md` + `docs/t4_rehearsal.md` (daily 07-13) (docs/t4_rehearsal.md) (대장 #13) (대장 #14) |
| 폐기 학습곡선이 아닌 “solo→blend” 대응표 | mBERT 3ep·maxlen512·seed avg의 solo 상승이 blend에서 사라진 이유 | `context/experiments.md` #29·#41·#48 (exp #29) (exp #41) (exp #48) |
| 검증 책임 분리 체크리스트 | 작성자 실행, reviewer 수치·diff, tester 회귀·계약 테스트의 서로 다른 책임 | `context/reports/verify_hist12_deploy_2026-07-10.md` (reports/verify_hist12_deploy_2026-07-10.md) |
