# Mate Eval Report - stacker final and log-bias

## Verdict

현 holdout 리그에서는 동료 `ensau080`의 측정 가능한 두 요소를 그대로 승격할 근거가 없다.

- `stacker_final` 단독은 우리 4-way blend보다 `-0.016885` 낮다.
- `stacker_final + soft-AU(a=0.9)`도 `blend4 + soft-AU(a=0.9)`보다 `-0.019210` 낮다.
- 동료 로그바이어스(`read +0.1 / grep -0.1 / list -0.18`)는 4-way blend 위에서는 `+0.000236`이지만, 실제 비교해야 할 `4-way + soft-AU` 최종면에서는 `-0.000735`다.

따라서 동료 LB `0.7511`과 우리 `0.7467`의 `+0.0044` 격차는 이 리그에서 실측한 로그바이어스나 "우리 OOF stacker를 최종으로 쓰기"로 설명되지 않는다. 남는 후보는 동료의 더 강한 full-train e5 계열 인코더(70k, 6 epochs, max_len 384, label smoothing), 동료 환경의 full-train stacker와 성분 차이, alpha 차이(task1 담당), 그리고 public LB 표본 차이다.

## Protocol

- Python: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`
- Holdout: `9,969` rows (`AU 682`, `SIM 9,287`)
- Join sanity:
  - 3-way `(linear + stacker + 2*e5)/4`: `0.717259217`
  - 4-way `(linear + stacker + 1.2*e5 + 0.8*mbert)/4`: `0.722545825`
- Soft-AU: `char_wb(3,5) LinearSVC C=1.0`, nonholdout AU-only train (`4,343` rows, `946` sessions), `alpha=0.9`
- AU cache: `context/night/2026-07-07/mate_au_char_c1_holdout.npz`

## Stacker Final

| variant | macro-F1 | delta vs 4-way | AU macro | SIM macro |
|---|---:|---:|---:|---:|
| stacker_final | 0.705661 | -0.016885 | 0.491988 | 0.716994 |
| blend3_sanity | 0.717259 | -0.005287 | 0.513806 | 0.729707 |
| blend4 | 0.722546 | 0.000000 | 0.511765 | 0.735676 |
| stacker_final + soft-AU a0.9 | 0.719563 | -0.002983 vs 4-way | 0.743832 | 0.716994 |
| blend4 + soft-AU a0.9 | 0.738772 | +0.016226 vs 4-way | 0.770168 | 0.735676 |

Task 기준 해석: 우리 stacker OOF는 정직 OOF라 동료 full-train stacker보다 불리하다. 그래도 "작은 격차(<0.005)라 full-train에서 역전 가능"인 상황이 아니다. 단독 stacker는 4-way보다 `-0.0169`, soft-AU를 붙여도 현 최종면보다 `-0.0192` 낮아, 우리 성분 그대로는 stacker-final 전환 신호가 약하다.

Stacker가 4-way보다 이긴 클래스는 거의 없다.

| class | stacker F1 | 4-way F1 | delta |
|---|---:|---:|---:|
| list_directory | 0.488812 | 0.482759 | +0.006054 |
| web_search | 0.619718 | 0.616967 | +0.002752 |
| read_file | 0.507487 | 0.538608 | -0.031121 |
| glob_pattern | 0.618695 | 0.650281 | -0.031586 |
| ask_user | 0.580556 | 0.613514 | -0.032958 |
| plan_task | 0.621622 | 0.670256 | -0.048634 |

## Log-Bias

동료 방식 그대로 `log(p + 1e-12)`에 클래스 상수를 더한 뒤 argmax했다.

| base | none | 0.5x delta | 1.0x delta | 1.5x delta |
|---|---:|---:|---:|---:|
| stacker_final | 0.705661 | +0.000811 | +0.000298 | -0.000634 |
| blend4 | 0.722546 | -0.000281 | +0.000236 | -0.001036 |
| blend4 + soft-AU a0.9 | 0.738772 | -0.000485 | -0.000735 | -0.002378 |

최종면(`blend4 + soft-AU`)에서 full vector는 `read_file`을 살리지만 `grep_search`, `list_directory`, `run_bash`를 잃는다.

| class | baseline F1 | biased F1 | delta | pred-count delta |
|---|---:|---:|---:|---:|
| read_file | 0.572694 | 0.583906 | +0.011213 | +198 |
| grep_search | 0.617455 | 0.606405 | -0.011050 | -134 |
| list_directory | 0.515892 | 0.510746 | -0.005147 | -54 |
| run_bash | 0.818448 | 0.814652 | -0.003796 | -1 |

One-at-a-time sensitivity도 승격 신호는 아니다. `blend4` 위에서는 `list_only_1.5x`가 `+0.000688`로 가장 크지만, soft-AU 최종면에서는 `read_only_0.5x`가 `+0.000091`에 그친다. 이는 D-009에서 폐기한 threshold/prior/calibration 가족의 전형적인 작은 holdout 조정으로 보인다.

## D-009 Relation

로그바이어스는 D-009의 threshold/prior/calibration 폐기 가족이다. 이 태스크는 동료 LB 실측값이 있는 요소의 사후 분석일 뿐이며, 결과만으로 제출 스테이징에 반영하지 않는다.

승격 조건은 별도다.

1. 메인 세션에서 D-009 예외를 명시적으로 결정한다.
2. 제출 후보를 clean commit으로 만들고 LB gate를 통과한다.
3. private 전이 위험을 리포트에 남긴다.

현재 수치로는 그 예외를 요청할 이유가 약하다.

## Next Actions

- 우리 4-way+soft-AU 라인은 유지한다.
- 로그바이어스는 submit/에 넣지 않는다.
- 동료 `0.7511` 격차 분석은 full-train 강인코더와 동료 stacker 성분 차이 쪽을 우선한다.
- alpha `0.8` vs `0.9`는 task1 grid 결과로만 판단한다.

## Artifacts

- `scripts/mate_eval/common.py`
- `scripts/mate_eval/eval_stacker_final.py`
- `scripts/mate_eval/eval_logbias.py`
- `context/night/2026-07-07/mate_stacker_final.json`
- `context/night/2026-07-07/mate_stacker_final_summary.csv`
- `context/night/2026-07-07/mate_stacker_final_per_class.csv`
- `context/night/2026-07-07/mate_logbias.json`
- `context/night/2026-07-07/mate_logbias_summary.csv`
- `context/night/2026-07-07/mate_logbias_per_class.csv`

## Reproduction

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m py_compile scripts\mate_eval\common.py scripts\mate_eval\eval_stacker_final.py scripts\mate_eval\eval_logbias.py
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\mate_eval\eval_stacker_final.py
C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\mate_eval\eval_logbias.py
```
