# 발표 핵심 수치 — 단일 기준표

발표 슬라이드의 수치는 아래 표에서만 가져오며, 단위와 비교 기준을 바꾸면 원문을 다시 대조한다. (D-001)

## 데이터와 평가

| 항목 | 발표 수치 | 출처 |
|---|---:|---|
| 학습 행 / 세션 | 70,000행 / 9,429세션 | (reports/eda_distribution.md) |
| 클래스 수 / 지표 | 14클래스 / Macro-F1 | (reports/eda_distribution.md) |
| 세션 길이 | 평균 7.42 / 중앙값 7 / 최대 18 step | (reports/eda_distribution.md) |
| history 길이 | 0~12, 짝수, 12에서 cap | (reports/eda_distribution.md) |
| 최다 클래스 | `edit_file` 11,171행 / 15.96% | (reports/eda_distribution.md) |
| 최소 클래스 | `web_search` 1,273행 / 1.82% | (reports/eda_distribution.md) |
| 클래스 불균형 | 8.8배 | (reports/eda_distribution.md) |
| 첫 턴 | 9,000행; `list_directory` 20.2% | (reports/eda_distribution.md) |
| GroupKFold 그룹키 | id에서 `-step_\d+$` 제거 | (D-003) |
| 고정 홀드아웃 | 9,969행 / 1,350세션 | (reports/third_party_sol_model_audit_2026-07-10.md) |

## 데이터 포렌식과 history

| 항목 | 발표 수치 | 출처 |
|---|---:|---|
| purity≥0.99 구조 규칙 coverage | min-rows≥5: 0.03%(21행); ≥10: 0.0157%(11행); ≥12: 0% | (reports/forensics_r1.md) |
| purity≥0.99 prompt-template coverage | 7.40%(5,181행) | (reports/forensics_r1.md) |
| 그중 `respond_only` | 49.7%(2,575/5,181행) | (reports/forensics_r1.md) |
| SIM 계열 | 64,975행 / 8,330세션 | (reports/forensics_r1.md) |
| AU 계열 | 5,025행 / 1,099세션 | (reports/forensics_r1.md) |
| `respond_only` 종료 위치 | 마지막 관측 step 5,178/5,178건 | (reports/forensics_r1.md) |
| 6턴 초과 / 정확히 12턴 | 51% / 30.7% | (D-010) |
| hist12@384 실제 잘림 | 8.5% | (D-010) |
| 6턴 초과 세션의 추가 보존 | 평균 +4.5턴 | (D-010) |
| #34 e5 solo | 0.70066→0.73617 | (exp #34) |
| #34 최종 blend | 0.73451→0.75601, Δ +0.02150 | (exp #34) |

## 이전 챔피언 구성 — 제출 #11 비교 기준

| 성분 | 비-AU 실효 지분 / 설정 | 출처 |
|---|---:|---|
| Linear | 25%; word/char TF-IDF + history/action/meta + LinearSVC | (reports/third_party_sol_model_audit_2026-07-10.md) |
| AAR stacker | 25%; prompt/context/action/transition 4-view | (reports/third_party_sol_model_audit_2026-07-10.md) |
| e5-base | 30%; full 70k, max-len 384, history 12 | (reports/third_party_sol_model_audit_2026-07-10.md) |
| mBERT | 20%; full 70k, max-len 384, history 6 | (reports/third_party_sol_model_audit_2026-07-10.md) |
| encoder 내부 가중 | e5:mBERT = 1.2:0.8 | (exp #27) |
| AU specialist | AU 행에서 90%, soft α=0.9 | (exp #24) (reports/third_party_sol_model_audit_2026-07-10.md) |
| per-encoder serialize | e5=12 / mBERT=6 | (exp #35) (reports/verify_hist12_deploy_2026-07-10.md) |
| 최종 zip / 로컬 offline smoke | 867.9MB / 50.2초 | (exp #35) |

## 현 챔피언 구성 — Qwen 하이브리드

| 항목 | 발표 수치 | 출처 |
|---|---:|---|
| 모델 패밀리 | Qwen2.5-0.5B 디코더 분류기, hist12/384, 2ep | (exp #52) |
| 하이브리드 | linear + AAR + Qwen block + soft-AU; mBERT 제외 | (exp #52) |
| Qwen h85 solo | instruct 0.75932 / base 0.75941; e5 대비 약 +0.023 | (exp #52) |
| 최종 blend holdout | 0.75601→0.76760, Δ +0.01160 | (exp #52) |
| 5지표 | 세션균등 +0.01393; MC +0.01430±0.00685; CI [+0.00614,+0.01758]; 반반 +0.01562/+0.00753 | (exp #52) |
| 제출 zip | 908.1MB | (대장 #14) |
| 연산활성 비교 | Qwen 360M/24L vs 이전 챔피언 86M×2/12L | (exp #52) |
| 속도 레버 | 길이정렬 배칭 1.7x + fast_aar 2.8x; 둘 다 출력등가 검증 | (exp #52) |
| T4 리허설 | 30,000행 총 515s(8.6분) | (docs/t4_rehearsal.md) (대장 #14) |

### 성분별 solo — 평가면을 섞어 비교하지 말 것

| 성분 | solo | 평가면 | 출처 |
|---|---:|---|---|
| Linear | 약 0.673 | 초기 LB solo | (exp #1) |
| AAR stacker | 약 0.671 | 초기 LB solo | (exp #2) |
| e5-hist12 | 0.73617 | 85% 고정 holdout | (exp #34) |
| mBERT-hist6 | 0.67147 | 85% 고정 holdout | (exp #27) |
| AU specialist | 0.68001 | AU 세션 3-fold OOF | (exp #23) |

초기 LB solo, 전체 holdout solo, AU-only OOF는 모집단과 평가 프로토콜이 달라 직접 순위를 매길 수 없으며, 최종 모델의 기여는 blend 델타로 판정한다. (exp #23) (exp #27) (exp #34)

## LB 궤적

| 제출 | 변화 | LB | 출처 |
|---:|---|---:|---|
| #1 | 3-way 재건 기준선 | 0.71884 | (대장 #1) |
| #2 | sibling recovery | 0.71884 | (대장 #2) |
| #3 | e5-small 추가 | 0.71280 | (대장 #3) |
| #4 | history bucket weighting | 0.71270 | (대장 #4) |
| #5 | hard-AU | 0.73310 | (대장 #5) |
| #6 | soft-AU α=0.9 | 0.7400 | (대장 #6) |
| #7 | +mBERT 60k | 0.7467 | (대장 #7) |
| #8 | mBERT full 70k | 0.7480 | (대장 #8) |
| #9 | 동료 surface+mBERT | 0.7503; 동료 기준 −0.0008 | (대장 #9) |
| #10 | 동료 surface α=0.9 | 0.7501; 동료 기준 −0.0010 | (대장 #10) |
| #11 | e5 hist12 | **0.7623; 직전 대비 +0.0143** | (대장 #11) |
| 기준선→hist12 챔피언 | 네 구조적 전환의 누적 | **+0.04346** | (대장 #1) (대장 #11) |
| #12 | KD student | 0.7621; 챔피언 대비 −0.0002 | (대장 #12) |
| #13 | Qwen 하이브리드 1차 | **시간초과 FAIL; 채점 없음** | (대장 #13) |
| #14 | Qwen + 속도레버 2종 | **0.77089; +0.00853, 89→79등** | (대장 #14) |
| 기준선→현 챔피언 | 0.71884→0.77089 | **+0.05205** | (대장 #1) (대장 #14) |

## 5지표 게이트와 전이

| 후보 | row Δ | 세션균등 Δ | MC Δ | bootstrap 95% CI | 반반 Δ | 결과 | 출처 |
|---|---:|---:|---:|---:|---:|---|---|
| hist12 | +0.02150 | +0.01859 | +0.01860±0.00863 | [+0.015104,+0.027827] | +0.018/+0.025 | LB +0.0143 | (exp #34) (exp #36) (reports/third_party_sol_model_audit_2026-07-10.md) (대장 #11) |
| args-lite | −0.00268 | −0.00096 | −0.00109±0.00547 | [−0.00751,+0.00176] | −0.0058/−0.0003 | 미제출 | (exp #43) |
| session-weight inv | −0.00231 | +0.00068 | +0.00132±0.00543 | [−0.00721,+0.00271] | −0.00329/−0.00159 | 미제출 | (exp #45) |
| seed avg s42+s43 | +0.00045 | +0.00180 | +0.00229±0.00417 | [−0.00311,+0.00379] | +0.00078/+0.00007 | 미제출 | (exp #48) |
| 5성분 parity stacker | −0.00164 | −0.0000028 | +0.00051±0.00585 | [−0.00625,+0.00295] | −0.00277/−0.00076 | 미제출 | (exp #50) |
| KD student | +0.00360 | +0.00394 | +0.00450±0.00545 | [−0.00078,+0.00809] | +0.00008/+0.00664 | LB −0.0002 | (exp #51) (대장 #12) |
| Qwen 하이브리드 | +0.01160 | +0.01393 | +0.01430±0.00685 | [+0.00614,+0.01758] | +0.01562/+0.00753 | LB +0.00853 | (exp #52) (대장 #14) |
| hist12 holdout→LB 전이 | +0.02150→+0.0143 | — | — | — | — | 약 67% | (exp #35) (대장 #11) |
| Qwen holdout→LB 전이 | +0.01160→+0.00853 | — | — | — | — | **73.5%** | (exp #52) (대장 #14) |

## 시간초과 대역전

| 단계 | 근거 수치 / 판정 | 출처 |
|---|---|---|
| #13 실패 | 평가 T4 추론 10분 초과, 채점 불가 | (대장 #13) (daily 07-13) |
| 원인 | 파라미터 등가를 연산 등가로 본 외삽 오류; 디코더 24층·hist12 장문 미반영 | (대장 #13) (daily 07-13) |
| 연산량 | Qwen 활성 360M/24L vs e5·mBERT 각 약 86M/12L | (exp #52) (docs/t4_rehearsal.md) |
| 레버 1 | 길이정렬 배칭: 평균 220tok·캡 5.3%, 패딩 연산 1.70x 절감, argmax 100% | (daily 07-13) (docs/t4_rehearsal.md) |
| 레버 2 | fast_aar: 84s→29.7s, 약 2.8x; 5,000행 오차 0.0·argmax 100% | (exp #52) (daily 07-13) (docs/t4_rehearsal.md) |
| 사전 실측 | Colab T4 30,000행 515s(8.6분) | (docs/t4_rehearsal.md) (대장 #14) |
| #14 성공 | LB 0.77089, +0.00853, 89→79등, 컷 갭 −0.0229 | (대장 #14) (daily 07-13) |
| 전이 | holdout +0.01160→LB +0.00853 = 73.5%; hist12 67% 할인과 부합 | (exp #52) (대장 #14) |

## solo 상승이 승격을 보장하지 않은 사례

| 후보 | solo 변화 | 최종 blend 변화 | 출처 |
|---|---:|---:|---|
| mBERT 3ep | +0.0197 | +0.00070 | (exp #29) |
| e5 maxlen512 | +0.0027 | −0.00102 | (exp #41) |
| mBERT hist12 | +0.02276 | −0.00106 | (exp #42) |
| e5 args-lite | +0.0035 | −0.00268 | (exp #43) |
| e5 seed avg | +0.0041 | +0.00045 | (exp #48) |
| klue-large hist12 | e5 대비 −0.023 | −0.00745 | (exp #49) |
| KD born-again | +0.0138 | holdout +0.00360, LB −0.0002 | (exp #51) (대장 #12) |
| Qwen solo 직접 배치 | +0.023 | row +0.0033, 세션균등 음수·CI 0 포함 — 단독 배치 부결 → 하이브리드로 우회해 승격 | (exp #52) (daily 07-13) |

## 검증 증거

| 항목 | 수치 | 출처 |
|---|---:|---|
| hist12 전체 회귀 테스트 | 22 passed / 0 failed | (reports/verify_hist12_deploy_2026-07-10.md) |
| serialize 설정 테스트 | 6/6 PASS | (reports/verify_hist12_deploy_2026-07-10.md) |
| encoder block 회귀 | 5/5 PASS | (reports/verify_hist12_deploy_2026-07-10.md) |
| reviewer diff | finding 0건 | (reports/verify_hist12_deploy_2026-07-10.md) |
| 5성분 frozen-shadow 누수 | meta-train∩holdout 세션 0 | (exp #50) |
| 최종 선택 | Qwen 하이브리드 0.77089 팀 최고 | (exp #52) (대장 #14) |

## 실험 운영 통계

| 항목 | 수치 | 출처 |
|---|---:|---|
| 번호 실험 범위 | #1~#52, 별도 기준선 #0 | (exp #1) (exp #52) |
| 주요 구조적 승격 | 기존 4개(hard-AU, soft-AU, mBERT, hist12) + Qwen 모델 패밀리 교체 | (reports/third_party_sol_model_audit_2026-07-10.md) (exp #52) |
| 제출 대장 | #1~#14 | (대장 #1) (대장 #14) |
| 최신 제출 흐름 | #12 KD 비승격 → #13 시간초과 → #14 팀 최고 | (대장 #12) (대장 #13) (대장 #14) |
