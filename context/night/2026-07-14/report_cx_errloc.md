# CX-B 신표면 오답 국소 분석

## 결론

exp #53 신표면의 두 target 악화는 AU 커버리지 문제가 아니다. `list_directory`의
old-correct/new-wrong 13행과 `glob_pattern`의 4행이 **전부 non-AU**였다. 17행 중
Qwen이 target을 top-1으로 둔 행은 0개인 반면 STK는 16개를 맞혔다. 즉
`w_q=2→3`에서 Qwen의 `read_file`/`grep_search` 쏠림이 STK의 국소 정답을 덮은 구조다.

다음 단일 probe로는 Qwen 확률에 `list_directory`와 `glob_pattern` log-bias를 각각
`+0.08` 적용한 뒤 기존 신표면을 유지하는 후보를 우선한다. 이 후보의 같은 holdout
실측은 Macro-F1 **+0.00089349**, 정답 행 **+10**이며 target F1은 각각
`+0.00916673`, `+0.00187457`였다. 다만 같은 holdout으로 발견·평가했고 Qwen h85
프록시이므로 이 수치는 LB 전이 보장이 아니라 다음 probe의 방향 근거다.

## 재현 방법과 계약

```powershell
& C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\cx_errloc\analyze.py
```

- 구표면: `soft_AU((lin + stk + 2*qwen) / 4, alpha=0.90)`
- 신표면: `soft_AU((lin + stk + 3*qwen) / 5, alpha=0.85)`
- Qwen bias 후보: `q' = softmax(log(clip(q)) + class_bias)`, 이후 신표면과 동일
- 행 정렬: Qwen의 `ids`, `y_true`, `actions`를 holdout 기준으로 전수 검증했다.
- AU cache: holdout AU 682행과 id 순서 및 14개 action 순서가 정확히 일치한다.
- 난수 사용 없음. 전이 행은 target/transition/holdout index 순으로 고정 정렬한다.

상세 수치와 입력 해시는
[`analysis.json`](../../../scripts/cx_errloc/analysis.json), 전체 전이 18행과 prompt·성분별
확률은 [`transition_rows.csv`](../../../scripts/cx_errloc/transition_rows.csv)에 있다.

주요 입력 SHA256:

| 입력 | SHA256 |
|---|---|
| `qwen_i2ep_h85.npz` | `a6547fba022e95365358b38a520ab3f6019c68b5a0c79193b389345551af5ac4` |
| `holdout_base.npz` | `414c8bb112cb1815587efc0e1aa0e93845d0e1330938b46dd8830a70b7aefc0c` |
| AU holdout cache | `831f9953e57bc660913c0eb80226b1f4f42c3909ff2c2ac08fdc375525dd7c5d` |
| `linear_probs.npy` | `9ef3e8866223d2cd6767d8a10524bbe50485a0ff35de16867bd7e66228082a54` |
| `stacker_probs.npy` | `efad4549c95da23a07e9afdb92e79bc3671ecf0f3e9172715d9bfacf41f56a9e` |
| `train.jsonl` | `a60ed84b75285caee237142ce97622fb55bb59c36be6b36ddee523992f83df19` |
| `league4/common.py` | `e61b8aaba1980b9f6eb85d85bd319c59863cb6a7c893b61bf508e23eb9bf9034` |
| `submit/au_route.py` | `758d68452bdf8ffeb044ff822cf6703957afbf73cb8b117453100f3baa17ff10` |

## 기준 재현과 F1 분해

| 표면 | Macro-F1 | 정답 행 | list F1 | glob F1 |
|---|---:|---:|---:|---:|
| 구: w2.0 / α0.90 | 0.76760460 | 7,631 | 0.52552927 | 0.67700987 |
| 신: w3.0 / α0.85 | 0.77138130 | 7,658 | 0.51800379 | 0.67374381 |
| 신−구 | **+0.00377670** | **+27** | **−0.00752547** | **−0.00326607** |

두 target 모두 precision보다는 true-positive 감소가 핵심이다.

| class | surface | TP | FP | FN | precision | recall |
|---|---|---:|---:|---:|---:|---:|
| list_directory | 구 | 422 | 533 | 229 | 0.441885 | 0.648233 |
| list_directory | 신 | 410 | 522 | 241 | 0.439914 | 0.629800 |
| glob_pattern | 구 | 480 | 155 | 303 | 0.755906 | 0.613027 |
| glob_pattern | 신 | 476 | 154 | 307 | 0.755556 | 0.607918 |

## 행 전이와 특성

| true class | 구 정답→신 오답 | 구 오답→신 정답 | 신 오답 방향 | AU | turn / history 특징 | target top-1 (lin/stk/qwen) |
|---|---:|---:|---|---|---|---|
| list_directory | 13 | 1 | read_file 9, grep_search 4 | 0/14 | 손실 turn 중앙 1, 8/13이 turn1·history0, 13/13 open_files 0 | 손실 8/12/0 |
| glob_pattern | 4 | 0 | grep_search 3, read_file 1 | 0/4 | 손실 turn 중앙 7, history 중앙 11, open_files 0은 2/4 | 손실 2/4/0 |

`list_directory`는 first-step에 몰리지만 이를 이용한 hard override는 금지축이며 적용하지
않았다. 일부 prompt는 사람 눈에는 `read_file`에 가까워 보이므로 label ambiguity도 있다.
반면 `glob_pattern` 손실은 중후반 session에 있고 history가 길어 같은 구조적 규칙으로
묶이지 않는다. 공통점은 오직 **STK target 지지 / Qwen 경쟁 class 지지**다.

### list_directory: 구 정답 → 신 오답 13행

```text
sess_sim_20260522_007767-step_01  -> read_file
sess_sim_20260522_011483-step_01  -> read_file
sess_sim_20260522_030963-step_01  -> read_file
sess_sim_20260522_021454-step_01  -> read_file
sess_sim_20260522_035898-step_06  -> read_file
sess_sim_20260522_019267-step_01  -> read_file
sess_sim_20260522_023101-step_01  -> grep_search
sess_sim_20260522_044128-step_03  -> grep_search
sess_sim_20260522_038684-step_03  -> grep_search
sess_sim_20260522_040016-step_02  -> read_file
sess_sim_20260522_006295-step_04  -> read_file
sess_sim_20260522_024356-step_01  -> grep_search
sess_sim_20260522_031304-step_01  -> read_file
```

반대 전이는 `sess_sim_20260522_004772-step_05` 한 행이며, 구 `read_file`에서 신
`list_directory`로 교정됐다. 이 행은 Qwen만 target top-1이었다.

### glob_pattern: 구 정답 → 신 오답 4행

```text
sess_sim_20260522_033407-step_11  -> grep_search
sess_sim_20260522_042919-step_05  -> grep_search
sess_sim_20260522_015278-step_08  -> grep_search
sess_sim_20260522_005243-step_06  -> read_file
```

반대 전이는 없다.

## 실행 가능한 후보 3개

아래는 모두 템플릿·first-step·행별 override가 아닌 **Qwen 확률 전체에 적용하는 soft
class bias**다. 수치는 신표면 대비 같은 holdout 실측이다.

| 우선 | 후보 | Macro-F1 Δ | 정답 행 Δ | 전체 예측 변경 | target F1 Δ | target 교정/손실 |
|---:|---|---:|---:|---:|---|---|
| 1 | list +0.08, glob +0.08 | **+0.00089349** | **+10** | 21 | list +0.00916673, glob +0.00187457 | list 12/0, glob 2/0 |
| 2 | list +0.08 | +0.00056247 | +6 | 17 | list +0.00732454 | list 10/0 |
| 3 | glob +0.10 | +0.00019447 | +2 | 4 | glob +0.00280986 | glob 3/0 |

1순위 후보는 전체 14행을 새로 맞히고 4행을 잃어 순 +10이다. 세 target 관련 class
외에 `read_file`/`grep_search` F1도 감소하지 않았지만, 이는 holdout 내 결과일 뿐이다.
blast radius가 더 작은 probe가 필요하면 3순위(glob-only)는 예측 4행만 바뀐다.

### 기각 대조군

| 변경 | Macro-F1 Δ | 정답 행 Δ | target 효과 | 판정 |
|---|---:|---:|---|---|
| `w_q 3.0→2.8`, α0.85 유지 | −0.00032740 | −3 | list F1 +0.000281, glob 0; list 1행만 교정 | 기각 |
| `α 0.85→0.90`, w_q3 유지 | −0.00107405 | −13 | 두 target 변화 0 | 기각 |

전역 Qwen 비중 후퇴는 exp #53의 다른 class 이득을 잃고 target을 거의 복구하지 못한다.
α는 target 전이 행이 전부 non-AU라 구조적으로 직접 레버가 아니다.

## 권고와 한계

- 다음 probe를 하나만 연다면 `list_directory:+0.08`, `glob_pattern:+0.08` Qwen log-bias를
  우선한다. 배포 적용 전 full-70k Qwen 표면 또는 독립 split에서 방향을 다시 확인한다.
- CX-A의 `calib.json` 후보가 별도로 나오면 bias를 중복 합산하지 않는다. CX-A calibration과
  이 후보를 각각 같은 신표면에서 재평가한 뒤 하나의 Qwen calibration으로 합친다.
- 세 bias 값은 대규모 탐색 결과가 아니라 오류 방향에서 도출한 작은 진단 probe다. 그래도
  발견과 평가가 같은 holdout이므로 `+0.00089349`는 낙관 편향 가능성이 있다.
- per-class hard rule, template, first-step override, 제출, push, 외부 파일 수정은 수행하지 않았다.

## 검증 / handoff

- 모델 라우팅: `gpt-5.6-sol`, reasoning `high`, `read-only`, `ROUTED_TASK=1` 시도.
- routed 결과: WebSocket/HTTPS가 샌드박스에서 차단되어 모델 응답 전 실패; 파일 변경 없음.
- 폴백: 현재 Codex 주 세션이 구현·실데이터 검증 수행.
- routed branch/commit: N/A (read-only).
- 작업 branch: `night/2026-07-14/task5`.
- commit: 생성 불가. `git add`/`git commit` 모두 공용 Git 메타데이터
  `C:\dev\2026-AI-DACON\.git\worktrees\task5\index.lock` 쓰기 권한 거부로 실패했다.
  작업 파일은 이 worktree에 unstaged 상태로 완성돼 있다.
- reviewer/tester 필요: 별도 작성자가 실데이터 재실행, output SHA256·수치 대조,
  Qwen bias 수식의 배포 parity, 독립 split/프록시→full 전이 확인.
