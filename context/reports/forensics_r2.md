# 시뮬레이터 포렌식 2라운드 (2026-07-11)

> 목적: CX-003 H1 가설(챔피언이 홀드아웃에서 `ask_user↔plan_task`를 상호 혼동 — directed confusion 1위, `ask_user→plan_task` 115건, eligible 191행, oracle-flip 시나리오 +0.0110, bootstrap CI `[+0.0085,+0.0138]` P=1.000) 후속. train은 시뮬레이터 산출물이므로 `ask_user`와 `plan_task`를 가르는 **결정적 상태 규칙**이 존재하는지 train.jsonl 70,000건 + train_labels.csv에서 재구성한다.
> 스크립트: `scripts/analysis/ask_plan_boundary.py` (재실행 가능, `scripts/analysis/common.py::load_frame` 재사용, 캐시 `data/_forensics_cache.pkl`). 출력 CSV는 `scripts/analysis/_out/ask_plan_*.csv`.
> 데이터: train 70,000행 중 `ask_user` 2,701행(2,491세션) / `plan_task` 2,679행(2,182세션).
> 재현 검증: `scripts/errtax_h12/analyze.py`로 CX-003의 191개 H1 eligible 홀드아웃 행을 `champion_holdout_preds.csv`에서 정확히 재현(191/191 일치).

## 결론 먼저 (TL;DR)

**`ask_user`와 `plan_task`를 가르는 고순도(purity≥0.80) state 규칙은 발견되지 않았다.** current_prompt 문형(질문부호, wh-의문사, "plan/roadmap" 계열 명사·동사, "not sure which" 등 10종 정규식 × step 조건 조합), last_action/last2_action 히스토리 조합(30+ 표본 전체 스캔), session_meta 연속값(budget_tokens_remaining, elapsed_session_sec, turn_index) 세 축 모두에서 **14클래스 전체 기준 purity가 0.52를 넘는 조건이 하나도 없었다**. ask_user/plan_task 두 클래스만 떼어놓고 보면 purity 0.79~0.89까지 오르는 조건이 여럿 나오지만(예: `last_action=='ask_user'` 0.858, `mentions_plan_verb&step1` 0.737), 이건 r1에서 이미 지적된 것과 동일한 착시다 — 그 조건이 걸리는 실제 모집단은 90% 가까이가 **다른 12개 클래스**로 채워져 있어(예: `has_question_mark&step==1` 1,997행 중 ask_user는 13.3%뿐, 최빈은 list_directory 21.7%), override 규칙으로 쓰면 이미 맞히던 다른 클래스 행을 대량으로 깨뜨린다. CX-003의 실제 191개 eligible 오류 행에 최선 후보들을 그대로 적용해 검증한 결과도 이를 뒷받침한다: `last_action=='none'→plan_task` 규칙은 90행에 적용되어 19행만 올바르게 뒤집고 71행(79%)을 오히려 틀리게 만든다. **H1은 "규칙 발견"으로 승격할 근거가 없다 — CX-003 권고(annotation-contract audit만, no-submit)를 재확인한다.**

---

## (a) 방법

1. `common.py::load_frame()`으로 70,000행 상태 프레임(session_key, step, last_action/last2/last3, current_prompt, session_meta·workspace 필드, 14클래스 라벨) 로드.
2. `ask_user`/`plan_task` 라벨 행만 골라 last_action/last2_action별 purity 테이블 작성 — **단, 두 라벨 서브셋 내부 purity와 14클래스 전체 purity를 분리해서 계산**(r1 함정 방지 규칙 그대로 적용).
3. current_prompt에 대해 10종 정규식 신호(질문부호, wh-의문사, "plan/roadmap" 계열, "not sure which", "how would you approach" 등)와 step==1/step>1 조합 20종을 스캔, 각 조건이 걸리는 행의 (i) 14클래스 전체 purity, (ii) ask_user/plan_task 서브셋 내 조건부 purity, (iii) unique 세션 수를 모두 산출.
4. session_meta 연속값(budget_tokens_remaining, elapsed_session_sec, turn_index)의 라벨별 분포(mean/std/quantile)를 비교.
5. 최선 후보 조건들을 **CX-003의 191개 H1 eligible 홀드아웃 오류 행**(`champion_holdout_preds.csv`에서 `errtax_h12/analyze.py` 로직으로 재구성, id로 조인)에 실제 override로 적용해 몇 행이 올바르게 뒤집히고 몇 행이 깨지는지 직접 계산.

## (b) 수치 — coverage × purity 표

### current_prompt 정규식 신호 (14클래스 전체 기준, min ask/plan 20행)

| 조건 | n_rows | n_sessions | 14클래스 top_purity | ask/plan 조건부 purity | favors |
|---|---:|---:|---:|---:|---|
| has_question_mark | 10,796 | 6,099 | 0.204 | 0.512 | ask_user |
| has_question_mark&step1 | 1,997 | 1,997 | 0.217 | 0.613 | plan_task |
| has_question_mark&step_gt1 | 8,799 | 5,277 | 0.223 | 0.568 | ask_user |
| wh_question_word | 5,084 | 2,554 | 0.219 | 0.523 | ask_user |
| mentions_plan_noun | 1,351 | 1,128 | 0.189 | 0.500 | ask_user |
| mentions_plan_verb | 347 | 323 | 0.395 | 0.537 | plan_task |
| mentions_plan_verb&step1 | 102 | 102 | 0.686 | 0.737 | plan_task |
| want_to_know_approach | 54 | 53 | 0.630 | 0.756 | ask_user |
| want_to_know_approach&step_gt1 | 43 | 42 | 0.628 | 0.794 | ask_user |

**핵심**: "ask/plan 조건부 purity" 열은 그 조건이 걸리는 행을 ask_user/plan_task로만 좁혔을 때의 값이다. 하지만 "14클래스 top_purity" 열이 보여주듯, 조건이 걸리는 **실제 모집단은 대부분 다른 12개 클래스**다. 예를 들어 `mentions_plan_verb&step1`은 표면상 0.686의 14클래스 purity를 보이지만 표본이 102행/102세션으로 작고, 상위 조건들(has_question_mark 계열)은 전체 표본이 커도(1,997~10,796행) 14클래스 purity가 0.20~0.22 수준에 머문다.

### last_action / (last2,last1) 히스토리 조합 (14클래스 전체, min 30행)

30행 이상 표본을 가진 모든 (last2, last1) 조합(수백 종) 중 **14클래스 전체 기준 최고 purity는 0.520**(`grep_search>glob_pattern`, 1,093행/1,007세션, 최빈 라벨 `glob_pattern` — explore 클래스 이슈이지 ask_user/plan_task와 무관)이었다. `ask_user`나 `plan_task`가 최빈 라벨로 나오는 (last2,last1) 조합은 **단 하나도 없다**:

| combo | n | n세션 | 14클래스 top_label | 14클래스 top_purity |
|---|---:|---:|---|---:|
| ask_user>ask_user | 43 | 43 | plan_task | 0.279 |
| plan_task>ask_user | 58 | 58 | plan_task | 0.224 |

(참고로 ap-서브셋으로 좁혔을 때만 `last_action=='ask_user'` → purity 0.858(plan_task 방향), `none>ask_user` → 0.894였다. 이 두 값은 (c)에서 오류 행 검증으로 완전히 기각됨.)

### session_meta 연속값 (분포 겹침 확인)

| 피처 | ask_user mean±std | plan_task mean±std |
|---|---:|---:|
| budget_tokens_remaining | 90,491 ± 52,788 | 97,028 ± 52,681 |
| elapsed_session_sec | 470 ± 237 | 427 ± 236 |
| turn_index (step) | 4.40 ± 3.38 | 3.44 ± 3.25 |

세 피처 모두 두 라벨 분포가 std 범위 안에서 거의 완전히 겹친다 — 결정적 문턱 없음.

## (c) 191개 H1 eligible 오류 행에 후보 규칙 적용 검증

CX-003 홀드아웃(9,969행)에서 챔피언이 실제로 `ask_user↔plan_task`를 혼동한 191행에, (b)에서 나온 최선 후보들을 override로 그대로 적용했을 때:

| 규칙 | 적용된 행 수 | 올바르게 뒤집힘 | 오히려 틀려짐 |
|---|---:|---:|---:|
| `last_action=='ask_user'` → plan_task | 10 | 1 | 9 |
| `last_action=='none'` → plan_task | 90 | 19 | 71 |
| `none>ask_user` → plan_task | 2 | 0 | 2 |
| `has_question_mark` (전체) | 70 | 46 (ask_user 실제) | 24 |
| `has_question_mark&step1` | 30 | 27 (ask_user 실제, 90%) — **but train ap-서브셋 다수결은 반대로 plan_task(61.3%)** | 3 |
| `has_question_mark&step_gt1` | 40 | 21 (plan_task 실제) | 19 |
| `mentions_plan_verb&step1` | 3 | 0 | 3 |

가장 좋아 보이는 `has_question_mark&step1`(191행 중 30행에 적용, 27/30=90%가 실제 ask_user)조차, 두 가지 이유로 규칙화가 불가능하다:

1. **모집단 오염**: 이 조건이 걸리는 전체 모집단(1,997행, train 전체 기준)에서는 `ask_user`가 13.3%에 불과하고 최빈은 `list_directory`(21.7%)다. 191행 밖의 수많은 다른 행(list_directory/read_file/grep_search 등)을 override가 망가뜨릴 게 거의 확실하다.
2. **방향 자체가 반전된다**: `ask_user`/`plan_task` 두 클래스로만 좁힌 train 전체 ap-서브셋(1,997행 중 684행이 ask_user 또는 plan_task)에서는 이 조건이 오히려 **plan_task를 61.3% 다수로 favor**한다(표 (b) 참조). 그런데 실제 챔피언이 혼동한 191개 오류 행에서 같은 조건이 걸리는 30행은 **90%가 ask_user**다 — 정반대 방향. 즉 "train 라벨 분포 기준 다수결"과 "챔피언이 실제로 틀리는 표본 기준 다수결"이 서로 뒤집혀 있어, 어느 모집단을 기준으로 규칙을 세우든 다른 모집단에서는 역효과를 낸다. 191행이라는 좁은 창에서만 좋아 보이는 표본 편향의 전형이다.

`last_action=='none'`(첫 스텝, history 없음) 조건은 90/191행(47%)이라는 큰 coverage를 보이지만 실제로는 **정반대 방향으로 작동**(71/90행이 override 시 틀림) — r1의 R3(첫 스텝 prior 보정) 가설과 마찬가지로 history_len==0 자체는 신호가 있지만(전체 turn_index 분포 참고) `ask_user vs plan_task`를 가르는 이진 판별자로는 쓸 수 없다.

## (d) 규칙 후보 목록 (최종 판정)

| # | 후보 | 14클래스 전체 purity | 191행 검증 결과 | 채택 권고 |
|---|---|---:|---|---|
| C1 | `last_action`/`(last2,last1)` 히스토리 조합 | 최고 0.520 (ask_user/plan_task 무관 라벨) | 적용 시 9/10, 71/90 오답 | **기각** |
| C2 | current_prompt 질문부호/의문사/계획-명사 정규식 | 0.19~0.40 (대부분) | `has_question_mark&step1`만 90% 방향이나 모집단 순도 13.3%, 게다가 train 다수결과 방향 자체가 반전 | **기각** — 모집단 오염 + 방향 반전이 191행 창을 넘어 압도적 |
| C3 | session_meta 연속값(budget/elapsed/turn_index) 문턱 | 분포 완전 겹침 | 검증 불필요(문턱 자체가 없음) | **기각** |
| C4 | (참고) `mentions_plan_verb&step1` | 0.686 (n=102, 소표본) | 191행 중 해당 조건 표본 없음(n<5) | 보류 — 표본 부족으로 판정 불가, 배포 가치 없음(coverage 극소) |

**결론: 규칙 후보가 리그 판정(row +0.005 & CI 하한>0 & MC>0)에 도전할 가치가 없다.** 모든 후보가 (i) 14클래스 전체에서 purity 0.52를 넘지 못하거나, (ii) 191행 eligible 창에서만 좋아 보이고 실제 배포 모집단에서는 반대 방향/저순도로 나타난다. CX-003의 최종 권고("annotation-contract audit only, no current-surface rule/route/threshold")를 이번 라운드가 데이터로 재확인한다.

## (e) 다음 라운드 제안

1. **H1을 규칙/override 축에서 완전히 닫는다.** CX-003이 이미 권고한 대로, 남은 유일한 경로는 시뮬레이터 생성 규칙 자체의 annotation-contract 감사(코드 레벨 규칙이 아니라 생성기 라벨링 계약 문제일 가능성) — 이는 Claude 승인 없이는 진행하지 않는다.
2. **H3(explore 4클래스 target-shape)로 라운드 전환**을 검토할 만하다 — CX-003 기준 절대 오라클 갭이 가장 크고(+0.1126), r1의 (e)절 (last2,last1) 조건부 80~90% purity(오라클 조건 하)도 여전히 미해결 리드다. 단, r1·CX-003 모두 "규칙/특화기/라우팅 금지"를 명시했으므로, 이번처럼 순수 서술적 분석(계층 분류 아키텍처가 실제로 로컬 CV에서 이득을 내는지)에 한정해야 한다.
3. **R2(템플릿 오버라이드, r1에서 보류)**는 여전히 저비용 옵션이지만 이번 라운드로 봐도 소표본 관용구 암기 위험이 그대로다 — 우선순위 낮음.
4. 새로운 state 축(예: current_prompt의 화용론적 구조 — 명령문 vs 의문문 vs 조건문, 또는 history의 사용자 발화 패턴)을 시도하려면, 이번 라운드에서 확인된 "14클래스 전체 purity와 이진 서브셋 조건부 purity를 반드시 분리해서 본다"는 방법론을 그대로 이어가야 한다.

## 산출물 위치

- 분석 스크립트: `scripts/analysis/ask_plan_boundary.py`
- 중간 산출물(CSV): `scripts/analysis/_out/ask_plan_by_last_action.csv`, `ask_plan_by_last2_last1.csv`, `ask_plan_lexical_signals.csv`, `ask_plan_lexical_x_step.csv`, `ask_plan_meta_signals.csv`, `ask_plan_strong_candidates.csv`, `ask_plan_h1_eligible_rule_application.csv`
- 재현 근거: `scripts/errtax_h12/analyze.py` (CX-003 소유, 191행 재현에만 사용, 수정 없음)

## 검증 부기 (2026-07-11 저녁, rev-forensics-r2 독립 재실행 — PASS)

- reviewer가 `ask_plan_boundary.py` 재실행으로 전 수치 재현, H1 eligible 191행(ask_user 115 + plan_task 76) 독립 재구성 일치, 함정 방지 규칙(조건별 unique 세션 수 병기 — 전 조건 n_rows==n_sessions) 준수 확인. main 소유 파일 미수정(신규 추가만) 확인.
- **출처 정정**: (b)절 히스토리 조합 표의 수치(최고 purity 0.520 `grep_search>glob_pattern` 등)는 이번 스크립트 산출물이 아니라 **r1 `transition_analysis.py`의 `last2_conditional_lift.csv`(전체 70,000행, min 30행 기준)** 에서 온 것 — reviewer가 해당 파일과 대조해 수치 일치를 확인했다. 이번 라운드의 `ask_plan_by_*.csv`는 2클래스 서브셋 기준(n_other_14class_rows=0)이라 (b)표의 재현 검증에는 r1 산출물을 사용해야 한다.
- **알려진 스크립트 결함 (결론 불변)**: `purity_table()`이 ap_df(2클래스 서브셋)에만 호출되어 docstring의 "14클래스 전체 purity" 목적을 자체 산출물로는 달성하지 못함. 리포트 본문은 r1 산출물로 우회했으므로 서술 자체는 정확 — 스크립트 재사용 시 주의.
- 완전성 부기: session_meta 스캔은 본문에 언급된 budget/elapsed/turn_index 외에 n_open_files·ci_status·git_dirty 축도 포함했다(9개 조건, `ask_plan_meta_signals.csv`) — n_open_files_0 0.194, ci_status_failed 0.180 등 전부 14클래스 purity 0.52 미만으로 결론 동일.
