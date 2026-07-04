# 시뮬레이터 포렌식 1라운드 (2026-07-04)

> 목적: train(`sess_sim_*`/`sess_au_*`)이 시뮬레이터 정책 산출물이라는 가설 아래, state→action 결정적 규칙을 찾아 "규칙이 모델을 이기는" 구간을 확보할 수 있는지 검증.
> 스크립트: `scripts/analysis/{common,determinism,template_forensics,transition_analysis,exploration_signals,session_meta_analysis,simulator_artifacts}.py` (전부 재실행 가능, `data/_forensics_cache.pkl`에 플랫 데이터프레임 캐시).
> 데이터: train 70,000행 / 9,429세션 (session key = id에서 `-step_\d+$` 제거).

## 결론 먼저 (TL;DR)

**구조적 state→action 결정 규칙은 사실상 없다.** 순수 구조 피처(last_action, history 길이, turn_index, ci_status, args 스키마, budget 등) 조합으로는 purity≥0.99 규칙의 누적 coverage가 **0.03%(21/70,000행, min_rows≥5)**에 불과했고, min_rows≥10으로 최소 표본 요건을 걸면 **0.0157%(11행)**로 줄어들며 min_rows≥12부터는 **0%**로 완전히 사라진다(생존자가 전부 5~11행짜리 소표본 노이즈였다는 뜻). current_prompt 템플릿까지 포함해도 순정미 상한은 **7.4%**이고, 그중 **49.7%**(2,575/5,181행)는 이미 F1 1.0인 `respond_only` 몫이라 순증 여지는 **3.7%**뿐이며 그 3.7%도 템플릿당 평균 2~3행짜리 소집단이라 과적합 위험이 크다. **"결정적 규칙 발견 → 컷 갭(−0.055) 해소"라는 1라운드 가설은 기각한다.** 대신 (1) explore 4클래스 계층 분류 아이디어, (2) 첫 스텝(history 없음) 별도 정책 활용, (3) respond_only 종료 구조 확인의 세 가지가 다음 라운드로 넘길 만한 근거 있는 리드다.

---

## (a) 기존 작업 요약

- `C:\dev\dacon-agent-action-api-boost\src\rules.py`: current_prompt 키워드 정규식 → action 오버라이드. **state→action 결정성 분석이 아니라 수작업 프롬프트 규칙**이며, 확신도가 낮을 때만 보정용으로 쓰는 별개 메커니즘. 중복 없음.
- `notes/*.md`, `context/{decisions,experiments,research}.md`, `legacy/w112_handoff.md`: 전부 **3-way 앙상블 가중치·버킷 블렌드 튜닝** 기록(w112, LB 0.7208). `history_presence` 버킷(있음/없음)은 이미 로컬 gate를 통과했고(§(g)에서 이 리포트가 그 근거를 구조적으로 보강함), `last_action_family4`/`turn_index_bin` 버킷은 로컬 gate 탈락.
- `research.md`에 "P0 시뮬레이터 포렌식"이 **todo로만 등록**되어 있었고 실제 착수 흔적은 없음 → 본 라운드가 첫 착수.
- 결론: **중복 작업 없음.** 백지에서 시작.

---

## (b) Purity≥0.99 누적 coverage 상한선

state 정의 20종(단일/조합) 전체에 대해 `groupby(state).action.value_counts()`로 버킷 purity·coverage 계산 (`scripts/analysis/determinism.py`, 최소 표본 min_rows=5).

| 기준 | 커버리지 | 비고 |
|---|---:|---|
| 순수 구조 state, min_rows≥5, 후보 7종 union upper bound | **0.03%** (21/70,000행) | "여러 규칙을 자유롭게 골라 쓸 수 있다면" 가정한 이론적 상한 |
| 순수 구조 state, min_rows≥10 (표본 안정화) | **0.0157%** (11/70,000행) | 유일한 생존자는 `last_action+last_result_summary` 정의의 `(edit_file, "ok; applied 1 edit (1+/1-) to Dockerfile")` 버킷(11행/11세션, purity 1.0) |
| 순수 구조 state, min_rows≥12 이상 | **0%** | 위 11행짜리 생존자도 사라짐 — min_rows≥5의 21행이 전부 세션 5~11개짜리 소표본 노이즈였음을 확인 |
| 개별 state 정의 20종 각각 | **0.0000~0.0005** | 어떤 단일/조합 정의도 유의미한 coverage를 못 냄 (아래 표) |
| current_prompt 정규화 템플릿, purity≥0.99 | **7.40%** (5,181행, 4,272세션) | `respond_only` 몫 **49.7%**(2,575행) 포함 |
| 위에서 `respond_only` 제외한 순증 커버리지 | **3.72%** (2,606행) | 템플릿 300여 개, 평균 2~3행/템플릿 |

`determinism_summary.csv`의 전체 표(threshold 0.95/0.99/1.00 × state_def 20종)는 예외 없이 **coverage 0.0000~0.0005** 구간이었다. 아래는 대표 예시:

| state 정의 | purity≥0.99 buckets | coverage |
|---|---:|---:|
| last_action 단독 | 0 | 0.0000 |
| last_action+last2_action | 0 | 0.0000 |
| last_action+last2+last3 | 1 (min_rows=5일 때만, 5행) | 0.0001 |
| last_action+last_args_sig | 0 | 0.0000 |
| last_ci_status+last_action | 0 | 0.0000 |
| history_len_bin(+last_action) | 0 | 0.0000 |
| turn_index(_bin)(+last_action) | 0 | 0.0000 |
| step+last_action | 0 | 0.0000 |
| last_action+prompt_glob_char | 0 | 0.0000 |
| last_action+last_result_summary(정확 문자열) | 2 buckets, 16행 | 0.0002 |

min_rows를 10으로 올리면 `last_action+last2+last3`의 5행짜리 버킷과 `last_action+last_result_summary`의 16행 중 5행짜리 셀(run_tests/edit_file)은 이미 최소 표본 미달로 탈락하고, 같은 `last_action+last_result_summary` 정의의 11행짜리 `(edit_file, "ok; applied 1 edit (1+/1-) to Dockerfile") → run_bash` 버킷만 남는다(0.0157%). min_rows≥12부터는 이마저 사라져 **0%**가 된다(재현: 더 무거운 5-way 콤보 `last1+last2+last3+turn_bin+ci`는 min_rows≥10에서 이미 0버킷). **즉 살아남은 소수 버킷은 세션 10여 개 안쪽의 우연의 일치이지 재현 가능한 규칙이 아니다.**

---

## (c) 상위 결정 버킷 표 (일반화 가능한 것만)

purity가 유의미하게 높으면서 세션 수가 충분한 버킷은 순수 구조 정의에서는 발견되지 않았다. 대신 **last_action 단독의 전체 분포**(가장 간단한 배포 가능 규칙 후보)를 참고용으로 남긴다 — purity가 모두 0.17~0.40대라 규칙화 불가함을 보여주는 음성 결과 표다.

| last_action | n행 | n세션 | 최빈 action | purity |
|---|---:|---:|---|---:|
| write_file | 1,446 | 1,443 | edit_file | 0.404 |
| run_bash | 4,797 | 3,125 | run_bash | 0.204 |
| edit_file | 10,620 | 6,123 | run_tests | 0.230 |
| list_directory | 4,223 | 3,512 | read_file | 0.251 |
| lint_or_typecheck | 2,016 | 1,507 | apply_patch | 0.249 |
| read_file | 8,887 | 6,008 | edit_file | 0.290 |
| none(첫 스텝) | 9,000 | 9,000 | list_directory | 0.202 |

last2_action까지 조건화하면 최대 **purity 0.52**(last2=grep_search→last1=glob_pattern→glob_pattern, 1,093행/1,007세션)까지 오르지만 0.95 문턱에는 한참 못 미친다. lift 분포(156개 셀) 평균 +0.064, 최대 +0.298 — "정보는 있으나 결정적이지 않다"는 게 핵심.

---

## (d) 템플릿 포렌식 결과

- 원문 완전 중복: 63,257개 고유 prompt 중 3,755개 그룹(size>1)이 10,498행(15.0%) 커버. purity≥0.99 그룹은 1,536개, 4,703행(6.72%).
- 숫자/경로/식별자/따옴표 인용어를 플레이스홀더로 정규화한 템플릿: 62,421개 고유. size>1 템플릿 4,319개가 11,898행(17.0%) 커버. purity≥0.99 템플릿 1,727개, **5,181행(7.40%), 4,272세션**.
- purity≥0.99 템플릿을 action별로 분해하면: `respond_only` 2,575행(548개 템플릿) — "마무리/정리해줘/wrap up/summarize" 계열 문구가 **거의 항상 purity 1.0**. 다음은 `edit_file` 736행, `run_bash` 365행, `grep_search` 312행, `read_file` 256행, `run_tests` 211행, `write_file` 176행, `plan_task` 135행, `ask_user` 129행 등 13개 클래스에 분산 — 템플릿당 평균 2~3행이라 각 템플릿이 사실상 "특정 세션들이 우연히 같은 관용구를 씀"에 가깝다.
- **`respond_only` 제외 순증 커버리지 3.72%(2,606행)** — 이미 팀 노트가 명시한 대로 `respond_only`는 F1 1.0으로 포화 상태라, 이 템플릿 규칙을 배포해도 **해당 클래스에서 얻을 추가 점수는 없음**. 나머지 13개 클래스 몫은 크기가 작고(템플릿당 2~11행) 표현이 매우 구어체·관용구적이라(예: "다시 빌드"→run_bash, "다시 타입체크"/"다시 린트"→lint_or_typecheck) 과적합 위험이 크다.
- `ask_user`는 "막혔는데/자꾸 나는데 도와줄래" 계열 템플릿에서 purity가 0.47~0.77 정도로 오르지만 0.99에는 못 미친다 — respond_only만큼 깨끗한 신호가 아니다.

---

## (e) 탐색 4클래스(read_file/grep_search/list_directory/glob_pattern) 판별 신호

### 프롬프트 표층 신호 — 음성 결과

explore-only 부분집합(28,782행, 기저 분포 grep_search 34.4%/read_file 32.2%/glob_pattern 18.4%/list_directory 15.0%)에서 13개 정규식 신호(glob 문자, 따옴표 인용어, "file(s)"/"directory" 언급, 검색·오픈 동사, 확장자 나열 등)를 테스트했다. **precision이 기저 분포 대비 거의 오르지 않는다**(0.32~0.44 vs 기저 0.15~0.34) — 즉 **프롬프트 단어 자체는 이 네 도구를 구분할 신호를 거의 담고 있지 않다.**

| 신호 | 켜졌을 때 n | explore 내 coverage | 최빈 클래스 | precision |
|---|---:|---:|---|---:|
| n_open_files_gt0 | 15,726 | 54.6% | grep_search | 0.436 |
| mentions_search_verb | 3,889 | 13.5% | grep_search | 0.391 |
| mentions_open_show | 7,629 | 26.5% | read_file | 0.352 |
| has_bare_filename | 4,459 | 15.5% | read_file | 0.349 |
| has_quoted_term | 450 | 1.6% | grep_search | 0.367 |
| mentions_directory | 923 | 3.2% | grep_search | 0.322 |

`last_args_had_scope`/`last_args_had_path`/`last_args_had_pattern`는 언뜻 강한 신호처럼 보이지만(예: scope 키 존재 시 read_file vs list_directory에서 P=0.908) **검증 결과 args 키 스키마가 action 이름의 결정적 함수임이 드러났다** — `scope`는 오직 grep_search만, `pattern`은 grep_search/glob_pattern만, `path`는 edit_file/list_directory/read_file/write_file만 갖는다(1:1 매핑, 확인 완료). 즉 이 "신호"는 last_action 그 자체를 다르게 인코딩한 것일 뿐 새 정보가 아니다.

### 진짜 신호: last2_action×last_action, 단 "explore로 판명된 후"에만 강함

explore-only 조건부로 보면 (last2, last1) 쌍이 **꽤 큰 표본에서 80~90% purity**를 낸다:

| last2 | last1 | n(explore 내) | 최빈 explore 클래스 | conditional purity | 전체 기준 purity | 전체 기준 explore 비율 |
|---|---|---:|---|---:|---:|---:|
| read_file | run_tests | 48 | grep_search | 0.896 | 0.250 | 27.9% |
| read_file | read_file | 477(전체 1,196) | grep_search | 0.881 | 0.351 | 39.9% |
| apply_patch | grep_search | 191(전체 402) | grep_search | 0.859 | 0.408 | 47.5% |
| read_file | glob_pattern | 318(전체 522) | grep_search | 0.852 | 0.519 | 60.9% |
| list_directory | grep_search | 480(전체 888) | read_file | 0.850 | 0.459 | 54.1% |
| read_file | list_directory | 82(전체 120) | grep_search | 0.756 | 0.517 | 68.3% |
| read_file | plan_task | 98(전체 212) | grep_search | 0.837 | 0.387 | 46.2% |
| grep_search | edit_file | 472(전체 1,952) | glob_pattern | 0.828 | 0.213 | 24.2% |

**해석에 주의**: 이 80~90%는 "이 행이 explore 클래스라는 걸 이미 안다"는 오라클 조건 하의 purity다. 조건 없이(14클래스 전체 기준) 같은 버킷을 보면 purity가 0.21~0.52로 뚝 떨어지고, 애초에 이 버킷이 explore로 판명되는 비율도 24~68%에 그친다. 즉 **평평한 14-way softmax에 이 신호를 피처로 더 넣는 것만으로는 큰 이득이 없을 가능성이 높다** — 이미 팀의 linear 피처셋(action n-gram)에 last1/last2가 들어가 있음. 진짜 기회는 **계층적(2단계) 분류 구조**다: 1단계가 대분류(explore/mutate_validate/coordinate/none)를 맞히면, 2단계에서 (last2,last1) 조건부 신호가 80~90% purity로 4개 explore 클래스를 갈라줄 여지가 크다. (e) 결론: 규칙표가 아니라 **아키텍처 제안**으로 다음 라운드에 넘긴다.

---

## (f) 규칙 후보 목록

| # | 적용 조건 → action | 예상 coverage | 예상 purity | w112 대비 예상 이득 | 채택 권고 |
|---|---|---:|---:|---|---|
| R1 | is_last_observed_step_in_file == False → `respond_only` 확률 0으로 강제 배제 | 87.1%(61,000행)에 배제 적용 | respond_only⟹is_last은 train에서 **100%**(5,178/5,178, sim/au 모두) | **≈0** — respond_only는 이미 F1 1.0(포화), 이 규칙이 잡아낼 오류가 현재 없음 | 보류. 안전망으로만 문서화, LB 프로브 불필요 |
| R2 | current_prompt 정규화 템플릿 purity≥0.99(비-respond_only, 2,606행/3.72%) → 해당 action 오버라이드 | 3.72% | 0.99+ (train 기준, 템플릿당 2~11행) | 매우 작음, 과적합 위험 큼(소표본·관용구 암기) | 보류. 고빈도·의미 명확한 소수 템플릿(예: "다시 빌드"→run_bash)만 골라 `rules.py` 확장 후보로 별도 LB 프로브 |
| R3 | history_len==0(첫 스텝, 12.9%) 전용 클래스 prior/bias 보정 | 12.9% | — (규칙 아님, prior shift) | 방향 확인됨 — 이미 gate 통과한 `history_presence` 버킷 블렌드(로컬 delta +0.0036)의 구조적 근거. blend 가중치보다 **class-wise local bias**가 더 정밀할 수 있음 | **R2 라운드 우선순위 1**. calib_v1 실패 교훈 때문에 반드시 LB 게이트 |
| R4 | explore 4클래스 계층 분류(1단계 대분류→2단계 last1/last2 조건부) | explore 41.1%(28,782행)에 적용 | 2단계 조건부 purity 80~90%(단 오라클 조건) | 미검증, 가장 유망하지만 구현 필요(모델 아키텍처 변경) | **R2 라운드 우선순위 2**. 프로토타입으로 로컬 CV 먼저 검증 |
| R5 | args 키 스키마·turn_index=step 중복 피처 정리 | — | — | 0(중복 제거는 노이즈 감소 목적, 점수 이득 아님) | 피처 위생 차원에서 채택, 점수 게이트 불필요 |
| R6 | last_ci_status/budget_tokens_remaining/turn_index 연속 신호를 state-conditioned 보정에 사용 | 전체 | — (연속 prior, 규칙 아님) | 방향은 뚜렷(budget<5k→respond_only 27~28%·ask_user 15~23%; CI failed→run_tests/apply_patch↑, respond_only↓) 하지만 calib_v1과 유사한 실패 모드 위험 | 참고만, R3에 흡수해서 같이 검증 |

**요약**: 표에 "규칙"으로 즉시 배포 가능한 항목은 없다. R1은 이미 포화, R2는 과적합 위험 대비 이득이 너무 작다. 진짜 가치 있는 리드(R3, R4)는 **규칙이 아니라 모델/앙상블 구조 변경**을 요구한다.

---

## 부가 발견 (item 6 — 시뮬레이터 아티팩트)

- **두 개의 서로 다른 생성 계열**을 발견: `sess_sim_*`(64,975행/8,330세션, step이 1..N으로 연속) vs `sess_au_*`(5,025행/1,099세션, step이 불연속 — 예: `[1,5,6,7,8,9]`, `[2,3,4]`). "au"는 아마도 추가 증강/재표집 배치로 보이며, **라벨 분포가 확연히 다르다**: read_file 25.7%(vs sim 12.3%), glob_pattern 1.8%(vs 8.0%), list_directory 2.2%(vs 6.5%). 이전에 팀 기록 어디에도 이 구분이 없었음 — **GroupKFold 시 sim/au 층화를 함께 고려할 가치 있음** (현재는 세션 프리픽스만으로 그룹핑해서 섞여 있음, 그 자체는 누수 아니지만 CV 대표성 문제일 수 있음).
- `respond_only`는 세션 종료 신호: 해당 세션에서 관측된 최대 step에서만 등장(5,178/5,178, 100%), 그 역은 아님(최대 step 행 중 54.9%만 respond_only). **테스트 시 이 규칙을 쓰려면 실제 히든 테스트 파일이 세션당 다중 행을 포함해야 하는데, 공개된 5행짜리 `data/test.jsonl` 샘플은 세션 5개가 각각 1행씩만 있어 구조가 다를 가능성을 시사** — 검증 없이 이 구조에 의존하는 건 위험.
- args 키 스키마는 action 이름의 결정적 함수(1:1) — `apply_patch`={n_files}, `ask_user`={question}, `edit_file`={path} or {path,target_symbol}(51/49), `glob_pattern`={pattern}, `grep_search`={pattern,scope}, `lint_or_typecheck`={target}, `list_directory`={path}, `plan_task`={goal}, `read_file`={path}, `run_bash`={cmd}, `run_tests`={target}, `web_search`={query}, `write_file`={path}. 새 정보 없음, 하지만 피처 중복 정리에 참고.
- `turn_index` == `step`(id의 숫자) 100% 일치 — 완전 중복 피처.
- id 숫자부(세션 시리얼 mod 14)와 라벨 상관 없음(최대 편차 2.1%, 노이즈 수준) — 생성 순서 누수 없음.
- 전체가 단일 날짜(`20260522`)로 생성됨 — 추가 시간 신호 없음.
- budget_tokens_remaining/elapsed_session_sec는 turn_index의 결정적 함수가 아님(표준편차 각각 ~50k토큰/~180초로 상당한 세션별 편차 존재) — 두 피처가 turn_index와 별개의 정보를 담고 있다는 뜻이라 현재처럼 별도 피처로 유지할 근거가 됨.

---

## (g) 다음 라운드 제안

1. **R3 (첫 스텝 prior 보정) 프로토타입 + LB 프로브** — `history_presence` 버킷 블렌드보다 세밀한 class-wise local bias를 first-step 전용으로 시도. calib_v1 실패(글로벌 피팅은 분포 이동에 취약)와 다르게, 이건 **train 전체에서 봐도 뚜렷한 state-conditioned 분포차**(list_directory 20.2%→4.1%, apply_patch 0.07%→7.9% 등)라 실패 모드가 다를 수 있음 — 그래도 LB 게이트 필수.
2. **R4 (explore 계층 분류) 로컬 CV 프로토타입** — 1단계 4-way 대분류 + 2단계 조건부 explore 분류기를 별도로 학습해 로컬 group-split Macro-F1을 현재 플랫 앙상블과 비교. 이 라운드에서 발견한 (last2,last1) 조건부 80~90% purity가 실제로 살아남는지가 핵심 검증 포인트.
3. **sim/au 계열 분리 검증** — CV fold별 sim/au 비율 확인, au 세션이 CV 점수를 왜곡하는지(혹은 실제 테스트 분포에 au 유사 데이터가 있는지) 점검.
4. **R1 재검토는 실제 테스트 구조가 밝혀진 뒤로 보류** — 세션당 다중 행 여부를 알 수 없는 한 시간 투자 우선순위 낮음.
5. R2(템플릿 오버라이드)는 저비용이므로 시간 남으면 소수 고빈도 템플릿만 추려 `rules.py` 확장 후보로 제출, 아니면 스킵.

---

## 산출물 위치

- 분석 스크립트: `scripts/analysis/common.py`, `determinism.py`, `template_forensics.py`, `transition_analysis.py`, `exploration_signals.py`, `session_meta_analysis.py`, `simulator_artifacts.py`
- 중간 산출물(CSV/JSON): `scripts/analysis/_out/` — `determinism_summary.csv`, `top_pure_buckets.csv`, `last_action_buckets.csv`, `exact_dup_groups.csv`, `template_groups.csv`, `transition_matrix_last1.csv`, `last2_conditional_lift.csv`, `exploration_signal_report.csv`, `exploration_pairwise_signals.csv`, `meta_*.csv`, `session_termination_dist.csv`, `session_size_dist.csv`
- 캐시: `data/_forensics_cache.pkl` (재실행 시 자동 생성, `force_rebuild=True`로 재생성 가능)

---

## 정정 이력 (2026-07-05, 독립 리뷰 반영)

1. (b)절: "min_rows≥10 → 0%"는 오기. 실측은 min_rows≥10에서 0.0157%(11행, `last_action+last_result_summary` 정의의 `(edit_file, "ok; applied 1 edit (1+/1-) to Dockerfile")` 버킷)가 남고, min_rows≥12부터 0%가 된다. 표와 본문을 정정.
2. TL;DR: "템플릿 purity≥0.99 중 respond_only 몫 66%"는 계산 오류. 정확한 값은 2,575/5,181 = **49.7%**(약 절반)이며 (d)절 원수치(2,575행, 548개 템플릿)는 애초에 정확했다.
3. (e)절 조건부 purity 표에서 `(read_file, list_directory)` 행에 실제로는 `(read_file, plan_task)` 쌍의 수치(n=98, purity=0.837)가 잘못 들어가 있었다. `(read_file, list_directory)`의 정확한 explore-조건부 값(n=82, purity=0.756)으로 정정하고, 원래 있던 `(read_file, plan_task)`(n=98, purity=0.837, 전체 기준 purity=0.387, explore 비율 46.2%) 행을 별도로 추가해 데이터 유실 없이 표를 복원했다.
4. (자체 발견, 리뷰 지적 외) (b)절 표에서 `last_action+last2+last3`의 purity≥0.99 버킷 수를 "6개"로 잘못 적었다. `determinism_summary.csv` 원본 확인 결과 **1개 버킷(5행)**이 정확하며, 이번 정정 과정에서 함께 바로잡았다.
