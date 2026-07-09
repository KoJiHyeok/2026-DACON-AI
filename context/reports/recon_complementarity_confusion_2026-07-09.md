# C·B 정찰 — 성분 상보성 + 혼동행렬 (lecture-gap-analysis 후속)

> 2026-07-09. 무제출 정찰. honest 9969행 리그(linear/stacker OOF + e5/mBERT holdout + soft-AU).
> 스크립트: `scripts/league4/diag_complementarity_confusion.py`, `inspect_confused_rows.py`, `diag_hist12_confusion_delta.py`.
> 출력: `night_out/league4/diag_{complementarity,confusion}.json`.
> 리그 정합: 최종 4-way+soft-AU = **0.73877** (exp #28 밤샘 수치와 소수점 일치 = 진단 신뢰 근거).

---

## TL;DR (판정)

- **보고서의 A(GBDT 메타 스태커)·B(피처)·C(상보성) 세 레버는 지금 전부 低-EV** — 이유가 데이터로 나왔다.
- **① A의 먹잇감(Type R 라우팅 가능 행)을 exp #34 hist12가 이미 먹는다.** hist12 +0.0215는 전부 edit↔apply_patch(−48행)·lint↔run_tests(−19행)·ask_user에서 발생 — 상보성 분석에서 linear가 0.48 복구하던 바로 그 라우팅 가능 행들. A로 다시 딸 게 남지 않는다. + 메타셀렉터 #17이 같은 행에서 이미 실패("routing 정보가 성분 출력에 없음", oracle 0.80163 실재).
- **② 잔차 = 탐색 클러스터(read/grep/list/glob)** — 최대 오류 질량(grep→read 379, read→list 215, grep→list 200…). **hist12가 손도 못 댐**(read/grep F1 델타 ≈ 0 → history 무관). 원시행 육안: 부분은 어휘 마커 갭(`찾아줘`→grep 놓침), 부분은 **비가역 라벨 모호성**(같은 표면형 "사용처부터 찾아줘"가 한 행 read_file·다른 행 grep).
- **③ 남은 잔차를 여는 어떤 레버도 이미 폐기됐거나 전이 실패**: 피처 증강(#15 args·#32 char-tfidf 스왑 → 앙상블 전이 0), 탐색 specialist 라우팅(#8/#14 R4 → 블렌드 −0.007~−0.016, **재시도 금지행**), 메타셀렉터(#17).
- **④ 라우터 확장(§R, 스코어차 역추적 필드 슬라이스 전수)도 사망**: 블렌드 이기는 성분 슬라이스 전무(comp_gain ≤ +0.009 노이즈), oracle 갭 전 슬라이스 ~0.073 균일 = 복구가능 오류 확산 → 집중 특화 표적 없음. **블렌드가 라우팅 프론티어에 포화.**
- **결론: 최고 EV 행동은 #34 배포**(e5 full-train 70k hist12 → LB). 확정 +0.0215가 손 안에 있고, 정찰이 "그게 옳은 표적"임을 확증. **라우터-슬라이스 확장은 死**(포화). A(GBDT per-row 결합)·B(좁은 스태커 피처)는 **veto 아님 — hist12 뒤 각 1회 LB 게이트, 낮은 기대**(제3자 평가 반영; 슬라이스 포화 ≠ per-row 결합기 사망). 유효 주축은 **성분 자체 강화(직렬화)**.

---

## C. 성분 오류-상보성

| 성분 | solo acc | solo macro-F1 |
|---|---:|---:|
| linear | 0.6733 | 0.6676 |
| stacker | 0.7039 | 0.7057 |
| e5 (hist6) | 0.7042 | 0.7051 |
| mbert | 0.6909 | 0.6715 |
| **oracle (≥1 맞음)** | **0.8121** | — |

- **복구가능 오류 = 838건 / 전체오류 2602건 = 32.2%** (블렌드는 틀렸으나 성분 ≥1은 맞음). 결합기가 노릴 수 있는 이론 상한.
- **pairwise Yule Q(정오상관) = 0.90~0.97 전 쌍 高** → 강의 Voting 규칙("서로 다른 약점")과 반대. 성분들이 **같이 맞고 같이 틀린다.**
  - 최상보(가장 낮은 Q): **linear|mbert = 0.8955**, 근소 2위 linear|e5 = 0.9002 (double_fault로는 linear|e5가 최소 0.2254) — linear가 인코더와 가장 다른 실수. 블렌드에서 linear 지분의 근거.
  - 최중복: **e5|mbert = 0.966** (두 인코더) → 인코더 다양성 소진(#29·#31)과 정합.
  - (reviewer 정정: 이전 서술 "최상보 linear|e5=0.900"은 스크립트의 double_fault 정렬을 Q 정렬로 오독한 것. Q 최소는 linear|mbert.)
- Q 0.9는 "복구가능 32%"의 대부분이 라우팅 불가 조인트 실패임을 뜻함 — 실현가능 상한은 32%보다 훨씬 작다.

## B. 혼동행렬 (최종 블렌드 argmax, hist6)

per-class F1 약한 순: **list_directory 0.516 · read_file 0.573 · lint 0.610 · grep 0.618 · ask_user 0.624 · glob 0.661** — 전부 탐색계열(할인표 경고와 일치). respond_only/write_file/edit_file은 이미 0.94+.

최대 오분류쌍 + **성분 복구율**(그 혼동행에서 성분 argmax가 정답인 비율):

| true → (블렌드)pred | n | 최대 성분 복구 | 유형 |
|---|---:|---|---|
| grep_search → read_file | 378 | linear 0.16 | **Type F** (성분 무력) |
| read_file → list_directory | 225 | linear 0.24 / mbert 0.23 | Type R |
| grep_search → list_directory | 205 | linear 0.05 | **Type F** |
| read_file → grep_search | 199 | linear 0.28 | Type R(약) |
| list_directory → read_file | 142 | stacker 0.14 | Type F |
| glob_pattern → list_directory | 129 | **전부 0.00** | **Type F(전무)** |
| glob_pattern → read_file | 116 | stacker 0.16 | Type F |
| edit_file → apply_patch | 103 | **linear 0.48 / stacker 0.36** | **Type R(강)** |
| ask_user → plan_task | 101 | linear 0.19 | Type R(약) |
| lint → run_tests | 73 | stacker 0.36 | **Type R** |

- **Type R (성분 20~48% 복구)** = A/라우팅이 노릴 수 있는 행. edit→patch, lint→tests, read→list.
- **Type F (성분 ~0)** = 결합기로 못 고침. 최대 질량(grep→read 378). GBDT 무력 — 어휘피처 or 비가역.

원시행 육안(`inspect_confused_rows.py`):
- grep→read/list: `찾아줘`·`어디 있나`·`where's`·`process.env 찾아봐` = grep 리터럴 마커 명백한데 e5(복구0.04) 미사용, linear만 일부.
- read↔grep: "사용처부터 찾아줘"가 라벨 read_file(다른 행 grep) → **시뮬레이터 라벨 모호성, 비가역.**
- edit→patch: "update main.py and app.py", "across both files" → linear 강복구, 라우팅 가능.

## Δ. hist12(#34)가 각 혼동쌍에 미치는 효과 (핵심)

동일 4-way+soft-AU에서 e5 슬롯만 hist6대조↔hist12 스왑:

| 오분류쌍 | hist6 | hist12 | 델타 |
|---|---:|---:|---:|
| grep→read_file | 379 | 379 | **0** |
| glob→list_directory | 126 | 126 | **0** |
| grep→list_directory | 207 | 200 | −7 |
| read→grep | 188 | 196 | +8 |
| read→list_directory | 215 | 216 | +1 |
| list→read_file | 150 | 144 | −6 |
| glob→read_file | 124 | 112 | −12 |
| **edit→apply_patch** | 92 | **44** | **−48** |
| **lint→run_tests** | 77 | **58** | **−19** |
| ask_user→plan_task | 117 | 115 | −2 |

per-class F1 델타(hist12−hist6, **전체 14클래스**): **web_search +0.082(최대)**, apply_patch +0.056, lint +0.045, run_tests +0.033, edit +0.025, ask_user +0.022, glob +0.011, list +0.010 / **read −0.001, grep −0.001**. 악화: grep→glob_pattern 쌍 +7, plan_task→ask_user 쌍 +2(소폭).

→ **hist12의 +0.0215는 history로 풀리는 클래스(web_search·apply_patch·lint·run_tests·edit·ask_user)에서 발생. 탐색 클러스터(read/grep)는 무변.** A가 노릴 Type R(edit→patch −48, lint→tests −19)을 hist12가 먹는다.
→ (reviewer 정정: 최초 리포트가 `diag_hist12_confusion_delta.py`의 8클래스 화이트리스트에 갇혀 **web_search +0.082(실제 최대 기여)·run_tests +0.033을 누락**했고 "전부 edit/patch·lint"로 좁게 서술했다. 결론 방향(read/grep 무변, A 착수 부결)은 불변이나 이득의 전체 그림은 더 넓다.)

---

## R. 라우터 확장 정찰 — "스코어 차 역추적" 필드 슬라이스 전수 (`diag_router_slices.py`)

AU 라우팅(+0.014)이 통한 조건을 필드 슬라이스로 재탐색. 추론 가능 필드(user_tier·language_pref·primary_lang·last_ci_status·git_dirty·turn·hist_len·open_files·budget·loc) 값별로 blend F1 vs best-single-component F1 vs oracle 갭을 support 가중 랭크. (#28은 id-prefix 세션그룹 스캔, 이건 필드 슬라이스 스캔 — 미시도 축.)

**① 성분 라우팅(단일 성분 스왑): 사망.** 블렌드를 이기는 성분이 있는 슬라이스가 **전무.** comp_gain(best 성분 − blend) 전부 ≤ +0.009, 유일 양수(budget<10000 stacker +0.009)조차 n=198 노이즈. 블렌드가 모든 식별가능 슬라이스에서 모든 성분을 이긴다.

**② 특화 라우팅(AU식 전용학습): 집중 표적 없음.** oracle 갭(복구가능 여유)이 **전 슬라이스 ~0.073 균일**(git_dirty=True 0.079, ko 0.074, py 0.072…). 복구가능 오류가 어느 서브모집단에도 집중 안 되고 **확산**. AU는 집중 약점(blend 0.51 vs 전역 0.74)이었으나 여기엔 그런 슬라이스가 없다:
- 최약 = hist_bucket=0(first-step) 0.501 → exp #25 이미 폐기(정보 부족), linear도 −0.087 나쁨 → 라우팅 불가.
- turn≥9(oracle 갭 0.108, n=1632)만 이론적 여지지만 best 성분 −0.010 → 전용학습 필요 = R4-in-blend 실패(#14).

**구조적 결론: 라우팅은 약점이 집중·식별가능할 때 먹힌다. 우리 복구가능 질량은 확산(균일 oracle 갭) = 블렌드가 라우팅 프론티어에 포화.** A(메타)·B(피처)·라우터확장이 같은 벽에 막히는 이유. #34(직렬화)가 유효한 이유도 여기 — 결합이 아니라 **성분 자체**를 밀어올려 프론티어를 통째로 이동.

## 권고

1. **#34 배포 (최우선, 신규 레버 아님)**: e5 full-train 70k hist12 재학습 → LB 게이트. 확정 +0.0215, 정찰이 표적 정당성 확증. per-encoder serialize(e5=12, mBERT=6) 계약 주의.
2. **A(GBDT 메타 스태커) — hist12 뒤 1회 LB 게이트, 낮은 기대** (제3자 평가 반영, 최초 "부결"에서 완화): 라우터 스캔이 죽인 건 *슬라이스 라우팅·단일성분 스왑*이지 **per-row 비선형 결합기가 아님.** #17은 27피처 LogReg override였지 full GBDT 스태커가 아니었다 → 복구가능 32%를 GBDT가 일부 열 여지 잔존. 단 Q 0.9·균일 oracle 갭이 기대치를 강하게 낮춤. 입력=성분 OOF확률+margin/entropy+구조피처, 강규제(num_leaves↓·min_child_samples↑), **LB 게이트 1회로만 판정.**
3. **B(탐색 피처) — 좁은 변형만, 낮은 기대**: 전체 linear 교체(#15·#32)·explore hard route(#8/#14)는 死. 단 **오분류쌍 전용 판별피처를 스태커 입력에 좁게 주입**하는 변형은 미시도 — 원시행에 마커(찾아줘→grep) 실재. R4식 대구조 변경 말고 보조피처/스태커 입력으로 한정.
4. **scale smoke test 추가** (제3자 평가): 공개 5행과 별도로 holdout 10k~30k 복제 → 추론시간·peak mem·zip 실측. (hist12는 max_len 384로 per-row 시간 불변이나 거버넌스로 실측.)
5. 잔차(탐색 클러스터)는 **features/label-ambiguity 벽**으로 기록. #34 배포 후 재평가. **우선순위: hist12 배포(확정 +0.0215) ≫ A/B(낮은 기대 1게이트).**
