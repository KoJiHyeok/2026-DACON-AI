# EDA 리포트 — 분포 분석 (Phase: EDA)

- 작성: 2026-07-04 · EDA Agent
- 데이터: `data/train.jsonl` 70,000건 + `data/train_labels.csv` (라벨 결측 0)
- 재현: `python notebooks/eda_distribution.py` (train 데이터 필요, 읽기 전용)

## TL;DR (모델링이 바로 쓸 결론)

1. **불균형 8.8x, 희소 클래스가 Macro-F1을 지배** — `web_search`(1.8%), `write_file`(2.1%), `lint_or_typecheck`(3.3%), `plan_task`(3.8%), `ask_user`(3.9%). 이 5개가 점수 리스크.
2. **가장 강한 정형 신호 = 전이 + 위치 + 결과플래그** (Cramér's V: `open_files수` 0.29, `prev>last` 바이그램 0.27, `turn_index` 0.27, `git_dirty` 0.26, `last_act` 0.21). 하지만 정형 피처 최빈 예측 상한은 **~28%** → 나머지는 `current_prompt` 텍스트가 좌우.
3. **`result_summary`의 성공/실패가 다음 행동을 가른다** — `run_tests` 통과→`respond_only`(26%) vs 실패→`edit_file`(33%). `lint` 실패→`edit_file/apply_patch`(합 63%). **P0 피처.**
4. **첫 턴(history 없음, 9,000건)은 별개 분포** — `list_directory`·`plan_task`·`write_file`가 3.3~3.7x lift. `is_first_turn` 플래그 필수.
5. **`user_tier`·`language_pref`·`budget`는 라벨과 거의 무관** (V ≤ 0.04). 프롬프트 한글 비율도 클래스 무관(모든 클래스 ~71%). 언어/티어에 피처 예산 쓰지 말 것.
6. **로컬 `test.jsonl`은 5건짜리 train 부분집합 스모크 샘플** (id·정답 모두 train과 일치). 점수 추정 불가 → **세션 프리픽스 GroupKFold OOF만 신뢰**.

---

## 0. 데이터 형태·정합성

| 항목 | 값 |
|---|---|
| 행 수 | 70,000 |
| 유니크 세션 (프리픽스) | 9,429 |
| 세션당 스텝 | mean 7.42 / median 7 / max 18 |
| step 번호 | 1~18 (== `turn_index`) |
| history 길이 | 0~12, **항상 짝수** (user↔action 교대), 12에서 캡 |
| 라벨 결측 | 0 |

- `history_len ≈ min(2·(step-1), 12)` — step과 turn_index가 사실상 동일 축. history는 최근 6 action(12턴)에서 잘림.
- 세션 하나가 최대 18개 step(=행)으로 등장 → **GroupKFold 그룹키를 세션 프리픽스로 안 잡으면 누수 확정** (`src/features.py:session_id` 이미 구현됨).

## 1. 클래스 분포 (Macro-F1 → 희소 클래스가 점수 좌우)

| # | action | count | share | vs uniform |
|---|---|---:|---:|---:|
| 1 | edit_file | 11,171 | 15.96% | 2.23x |
| 2 | grep_search | 9,912 | 14.16% | 1.98x |
| 3 | read_file | 9,257 | 13.22% | 1.85x |
| 4 | glob_pattern | 5,284 | 7.55% | 1.06x |
| 5 | respond_only | 5,178 | 7.40% | 1.04x |
| 6 | run_bash | 5,068 | 7.24% | 1.01x |
| 7 | apply_patch | 4,823 | 6.89% | 0.96x |
| 8 | run_tests | 4,561 | 6.52% | 0.91x |
| 9 | list_directory | 4,329 | 6.18% | 0.87x |
| 10 | ask_user | 2,701 | 3.86% | 0.54x |
| 11 | plan_task | 2,679 | 3.83% | 0.54x |
| 12 | lint_or_typecheck | 2,283 | 3.26% | 0.46x |
| 13 | write_file | 1,481 | 2.12% | 0.30x |
| 14 | web_search | 1,273 | 1.82% | 0.25x |

- 불균형 **8.8x** (top `edit_file` vs bottom `web_search`).
- Macro-F1은 14클래스 단순평균 → **하위 5개(`web_search`, `write_file`, `lint_or_typecheck`, `plan_task`, `ask_user`)의 F1이 총점을 좌우**. class weight / threshold 튜닝 / OOF 오류분석의 1순위 대상.

## 2. 세션·스텝·history 구조

- 세션당 스텝 분포는 5~8에서 봉우리(각 1,000~1,275건), 롱테일 18까지.
- step 번호별 건수: s01=9,000 → s07=5,435 → s12=1,420 → s18=84 (뒤로 갈수록 감소).
- **첫 턴(history 없음) 9,000건은 분포가 확연히 다름:**

| action | first-turn% | overall% | lift |
|---|---:|---:|---:|
| write_file | 7.9% | 2.1% | **3.72x** |
| list_directory | 20.2% | 6.2% | **3.27x** |
| plan_task | 12.5% | 3.8% | **3.25x** |
| ask_user | 7.0% | 3.9% | 1.81x |
| run_bash | 11.2% | 7.2% | 1.55x |
| read_file | 16.5% | 13.2% | 1.25x |

→ 세션 시작 = "둘러보기(list/read)·계획(plan)·스캐폴딩(write)". `is_first_turn`(=history 비었는지) 플래그가 희소 클래스(`write_file`, `plan_task`) 회수에 직접 기여.

## 3. 전이 신호: 직전 행동 → 정답

각 마지막-행동별 정답 top-3:

| last_act (n) | top1 | top2 | top3 |
|---|---|---|---|
| edit_file (10,620) | run_tests 23% | edit_file 15% | apply_patch 10% |
| grep_search (9,412) | edit_file 22% | read_file 19% | grep_search 18% |
| **&lt;none&gt; (첫턴, 9,000)** | list_directory 20% | read_file 17% | plan_task 12% |
| read_file (8,887) | edit_file 29% | grep_search 15% | read_file 14% |
| glob_pattern (4,967) | grep_search 22% | glob_pattern 18% | read_file 16% |
| run_bash (4,797) | run_bash 20% | edit_file 18% | read_file 11% |
| apply_patch (4,417) | lint_or_typecheck 17% | respond_only 16% | apply_patch 16% |
| run_tests (4,251) | edit_file 24% | respond_only 18% | grep_search 13% |
| list_directory (4,223) | read_file 25% | grep_search 21% | glob_pattern 11% |
| plan_task (2,584) | apply_patch 18% | read_file 17% | list_directory 17% |
| ask_user (2,192) | grep_search 20% | read_file 15% | edit_file 14% |
| lint_or_typecheck (2,016) | apply_patch 25% | edit_file 22% | respond_only 10% |
| write_file (1,446) | **edit_file 40%** | run_bash 29% | read_file 7% |
| web_search (1,188) | edit_file 19% | grep_search 16% | respond_only 12% |

- 최빈 예측 상한: `last_act` **22.9%**, `prev>last` 바이그램 **28.6%**.
- 명확한 워크플로 체인: `read/grep → edit`, `edit → run_tests`, `list → read`, `glob → grep`, `write → edit/run_bash`, `lint → apply_patch`.
- **자기반복률 12.1%** (탐색 계열 높음: run_bash 19%, grep 17%, glob 17% / write_file 0%, ask_user 2%). `last_act == 후보`인지 자체가 피처.

### 3-1. `result_summary` 성공/실패 = 강한 조건부 신호 (P0)

`result_summary`를 ok/fail로 파싱(분포: ok 29,502 · fail 5,321 · other 35,177):

| 직전 action | 결과 OK → 다음 | 결과 FAIL → 다음 |
|---|---|---|
| run_tests | **respond_only 26%**, edit_file 20% | **edit_file 33%**, grep 15%, apply_patch 13% |
| lint_or_typecheck | apply_patch 22%, edit_file 16% | **edit_file 32% + apply_patch 31%** |
| run_bash | run_bash 21%, edit_file 20% | grep 18%, run_bash 17%, edit_file 17% |
| apply_patch | respond_only 19%, lint 17% | apply_patch 18%, run_tests 17% |

→ "통과하면 마무리(`respond_only`)/다음 단계, 실패하면 고치기(`edit_file`/`apply_patch`)". `last_result_status ∈ {pass, fail, none}` 피처는 특히 `respond_only`·`edit_file` 구분에 직접적.

## 4. `current_prompt` 텍스트 단서

- 길이: mean 61자, median 56, p90 103, max 346.
- **프롬프트 길이가 클래스 구분자**: 짧은 명령형(~40자) = `respond_only`(43)·`run_tests`(40)·`run_bash`(39)·`lint`(40); 긴 질문형(~85자) = `plan_task`(88)·`web_search`(85)·`ask_user`(86).
- **언어는 비구분**: 한글 비율>0.1 프롬프트가 전체 71%인데 클래스별로도 모두 69~75%로 평평 → 언어 피처는 라벨 신호 아님.

액션별 특징 영어 토큰 (log-odds, 상위):

| action | distinctive tokens |
|---|---|
| respond_only | recap, summarize, brief, done, wrap, stop, enough, helped |
| run_bash | compiles, compile, vet, rerun, rebuild, jest, sanity, reproduce |
| run_tests | regressed, rerun, didnt, suite, jest, kick, touched, safe |
| **ask_user** | assertionerror, nonetype, typeerror, connectionerror, attributeerror, stuck, help, whether |
| **plan_task** | practice, fastapi, sketch, steps, lay, best, vague, recommended, standard |
| **web_search** | recommended, fastapi, docs, standard, rotary, days, handles |
| **write_file** | rewrite, scratch, create, stub, fresh, write, dto, completion |
| lint_or_typecheck | dangling, boots, reproduce, smoke, compile, spin |
| glob_pattern | notebook, matches, find, references, pins, suspect |
| grep_search | grep, define, who, registered, defined, asserting |
| read_file | open, show, walk, understand, currently, configured |
| edit_file | tweak, swap, fix, toggle, bool, false, dependencies |

- 희소 클래스가 텍스트 단서는 오히려 뚜렷: `ask_user`=에러 타입(traceback), `plan_task`/`web_search`="best practice/recommended/standard", `write_file`="rewrite/scratch/create/stub". **char n-gram + 키워드 플래그가 희소 클래스 회수의 핵심.**
- `ask_user`↔`web_search`가 어휘(both "recommended/practice", error types) 겹침 → 혼동 위험. 정형 피처(길이·history 유무)로 분리 보강 필요.

## 5. `session_meta` ↔ 라벨 상관 (연관 강도 순)

Cramér's V (0~1, 라벨과의 연관):

| 피처 | V | 해석 |
|---|---:|---|
| open_files 개수 | **0.291** | 0개→탐색, 1개→편집, 2~3개→멀티패치 |
| prev>last 바이그램 | **0.274** | 전이 체인 |
| turn_index 구간 | **0.273** | 세션 단계(탐색→편집→마무리) |
| git_dirty | **0.260** | clean→read, dirty→edit |
| last_act | 0.207 | 전이 |
| last_ci_status | 0.120 | failed→edit/patch, passed→respond↑ |
| top workspace lang | 0.072 | yaml→grep, 컴파일언어→run_tests |
| budget 5분위 | 0.037 | 미약 |
| user_tier | 0.022 | 무관 |
| language_pref | 0.021 | 무관 |

세부:
- **open_files**: 0개(n=23,498)→`read_file 19%`/`list_directory`; 1개(n=40,257)→`edit_file 21%`; 2~3개(n=6,231)→`apply_patch 23%`. (4+는 n=14, 무시)
- **turn_index**: 1-2→`read_file/list_directory`(탐색); 3-5→`edit_file 22%`; 6-9·10+→`respond_only 13%`·`apply_patch` 상승(마무리·정리).
- **git_dirty**: False(n=16,299)→`read_file 19%`(탐색); True(n=53,701)→`edit_file`.
- **last_ci_status**: failed→`edit_file/apply_patch`(수리); passed→`respond_only 11%`(마무리); none→`grep/glob`(탐색).
- **budget**: `respond_only`가 저예산 8.8% vs 고예산 6.1% (약하지만 존재), `ask_user`는 거의 무변화. 단독 신호로는 약함.

## 6. Train/Test 주의 + CV 규약

- 로컬 `test.jsonl`은 **5건**, 5건 모두 train id와 중복, `sample_submission.csv` 정답도 train 라벨과 100% 일치 → **공개 스모크 샘플**(파이프라인 검증용). 로컬 test로 점수 추정 절대 불가.
- **신뢰 가능한 검증 = 세션 프리픽스 GroupKFold OOF Macro-F1**뿐. 리더보드는 제출로만.
- 실제 평가 test는 크기·분포 미공개 → train 분포에 과적합 경계. 특히 희소 클래스는 CV 분산이 크므로 **StratifiedGroupKFold 또는 반복 GroupKFold**로 OOF 안정화 권장.

---

## 피처 후보 (우선순위)

### P0 — 즉시 구현 (`src/features.py`)
- **전이 피처**: `last_action`(원핫/임베딩), `prev_action`, `prev>last` 바이그램, `last_action==각 클래스` 여부. (V 0.21~0.27)
- **`last_result_status`**: 직전 `result_summary` → {pass, fail, none/other}. `run_tests`·`lint` 뒤 `respond_only` vs `edit_file` 분기의 핵심. (§3-1)
- **`is_first_turn`** (history 비었는지) + `history_len`(0~12) + `turn_index`/`step`. 첫 턴 분포가 별개(§2). turn_index는 그대로 수치 피처.
- **`n_open_files`**(0/1/2~3 버킷) + `git_dirty`(bool). (V 0.29, 0.26)
- **프롬프트 텍스트**: char n-gram(2~5) + word TF-IDF (다국어 → char n-gram 필수). 베이스라인이 이미 TF-IDF 사용, 여기에 위 정형 피처 concat.
- **프롬프트 길이**(char/word 수) — 길이만으로 명령형 vs 질문형 분리(§4).

### P1 — 테스트 가치 있음
- **키워드 플래그**(정규식): 에러타입(`\w+Error`, `traceback`) → `ask_user`; `best practice|recommend|docs|standard` → `web_search`/`plan_task`; `rewrite|from scratch|create|stub` → `write_file`; `rerun|regress|suite|compile` → `run_tests`/`run_bash`. 희소 클래스 정밀도 보강.
- **`last_ci_status`** 원핫 + **history 내 action 카운트/비율**(직전 6 action 중 edit/test 빈도).
- **top workspace language** 원핫(특히 `yaml`→grep, 컴파일 언어→test 경향).
- **result_summary 수치 파싱**: 테스트 통과/실패 개수, 패치 파일 수(`patched N files`) 등 마지막 action 결과의 숫자.

### P2 — 시간 남으면
- `budget_tokens_remaining`(약신호), `elapsed_session_sec`, `loc` 수치 버킷.
- 세션 내 누적 통계(지금까지 edit 횟수, 마지막 test 이후 경과 step 등) — 시퀀스 피처.
- `user_tier`·`language_pref`: **버릴 것** (V≈0.02). 넣어도 노이즈.

## 모델링 시사점
- 정형 단독 최빈 상한 ~28% → **텍스트(char n-gram)가 주력, 정형은 booster**. Tier1 GBDT는 TF-IDF(희소) + 정형 피처 결합 필요.
- Macro-F1 최적화: 희소 5클래스에 **class_weight='balanced' 또는 클래스별 threshold 튜닝**을 OOF에서 직접 최적화.
- 혼동쌍 우선 관찰: `ask_user`↔`web_search`(어휘 겹침), `edit_file`↔`apply_patch`, `grep_search`↔`read_file`, `run_bash`↔`run_tests`.
