# hist12 혼동행렬 정식 산출물 (2026-07-09)

리그 프레임: `scripts/league4/common.py` honest 9969행 (linear/stacker OOF + e5/mBERT holdout + AU 라우팅), 4-way+soft-AU 블렌드(alpha=0.9), e5 슬롯만 hist6↔hist12 스왑.

- hist6(현행 배포) 블렌드 macro-F1 = **0.73451**
- hist12 블렌드 macro-F1 = **0.75601** (델타 +0.02150)
- baseline 참고값(4-way+soft-AU) = 0.75601

## e5 성분 단독 복구율 (hist6 vs hist12)

| 지표 | hist6 e5 | hist12 e5 | 델타 |
|---|---:|---:|---:|
| e5 argmax accuracy | 70.8% | 73.6% | 2.8% |
| e5 argmax macro-F1 | 0.7007 | 0.7362 | +0.0355 |

## per-class P/R/F1/support (hist12 블렌드)

| action | support | precision | recall | F1 | F1(hist6) | 델타F1 |
|---|---:|---:|---:|---:|---:|---:|
| edit_file | 1580 | 0.971 | 0.963 | 0.9670 | 0.9424 | +0.0245 |
| grep_search | 1473 | 0.717 | 0.544 | 0.6191 | 0.6202 | -0.0012 |
| read_file | 1284 | 0.547 | 0.611 | 0.5771 | 0.5784 | -0.0013 |
| glob_pattern | 783 | 0.742 | 0.610 | 0.6699 | 0.6591 | +0.0109 |
| respond_only | 734 | 0.989 | 1.000 | 0.9946 | 0.9939 | +0.0007 |
| run_bash | 689 | 0.827 | 0.826 | 0.8264 | 0.8163 | +0.0101 |
| apply_patch | 666 | 0.929 | 0.961 | 0.9446 | 0.8889 | +0.0558 |
| run_tests | 653 | 0.810 | 0.836 | 0.8229 | 0.7898 | +0.0331 |
| list_directory | 651 | 0.434 | 0.644 | 0.5182 | 0.5087 | +0.0096 |
| ask_user | 393 | 0.690 | 0.567 | 0.6229 | 0.6008 | +0.0221 |
| plan_task | 370 | 0.656 | 0.705 | 0.6797 | 0.6667 | +0.0130 |
| lint_or_typecheck | 320 | 0.662 | 0.650 | 0.6562 | 0.6117 | +0.0445 |
| write_file | 199 | 0.980 | 0.995 | 0.9875 | 0.9900 | -0.0025 |
| web_search | 174 | 0.638 | 0.770 | 0.6979 | 0.6162 | +0.0817 |

## 가장 약한 클래스 top5 (hist12 F1 기준)

- **list_directory**: F1=0.5182 (support=651, hist6 대비 +0.0096)
- **read_file**: F1=0.5771 (support=1284, hist6 대비 -0.0013)
- **grep_search**: F1=0.6191 (support=1473, hist6 대비 -0.0012)
- **ask_user**: F1=0.6229 (support=393, hist6 대비 +0.0221)
- **lint_or_typecheck**: F1=0.6562 (support=320, hist6 대비 +0.0445)

## 최대 오분류쌍 top15 (hist12, count 기준)

| true → pred | count | row_frac(정답 행 내 비율) |
|---|---:|---:|
| grep_search → read_file | 379 | 25.7% |
| read_file → list_directory | 216 | 16.8% |
| grep_search → list_directory | 200 | 13.6% |
| read_file → grep_search | 196 | 15.3% |
| list_directory → read_file | 144 | 22.1% |
| glob_pattern → list_directory | 126 | 16.1% |
| ask_user → plan_task | 115 | 29.3% |
| glob_pattern → read_file | 112 | 14.3% |
| grep_search → glob_pattern | 79 | 5.4% |
| plan_task → ask_user | 76 | 20.5% |
| run_bash → run_tests | 63 | 9.1% |
| glob_pattern → grep_search | 60 | 7.7% |
| lint_or_typecheck → run_tests | 58 | 18.1% |
| read_file → glob_pattern | 56 | 4.4% |
| list_directory → grep_search | 54 | 8.3% |

## hist6 → hist12 오분류쌍 델타 (개선/악화 상위, count 기준)

| true → pred | hist6 count | hist12 count | 델타 |
|---|---:|---:|---:|
| edit_file → apply_patch | 92 | 44 | -48 |
| list_directory → grep_search | 59 | 54 | -5 |
| read_file → glob_pattern | 56 | 56 | +0 |
| lint_or_typecheck → run_tests | 77 | 58 | -19 |
| glob_pattern → grep_search | 65 | 60 | -5 |
| run_bash → run_tests | 75 | 63 | -12 |
| plan_task → ask_user | 74 | 76 | +2 |
| grep_search → glob_pattern | 72 | 79 | +7 |
| glob_pattern → read_file | 124 | 112 | -12 |
| ask_user → plan_task | 117 | 115 | -2 |
| glob_pattern → list_directory | 126 | 126 | +0 |
| list_directory → read_file | 150 | 144 | -6 |
| read_file → grep_search | 188 | 196 | +8 |
| grep_search → list_directory | 207 | 200 | -7 |
| read_file → list_directory | 215 | 216 | +1 |
| grep_search → read_file | 379 | 379 | +0 |

## 14x14 정규화 혼동행렬 (행=recall, hist12)

| true\pred | apply_patch | ask_user | edit_file | glob_pattern | grep_search | lint_or_typecheck | list_directory | plan_task | read_file | respond_only | run_bash | run_tests | web_search | write_file |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **apply_patch** | 96% |  | 4% |  |  |  |  |  |  |  |  |  |  |  |
| **ask_user** | 0% | 57% | 0% | 0% | 0% |  | 0% | 29% | 1% |  |  |  | 12% |  |
| **edit_file** | 3% |  | 96% |  | 0% | 0% | 0% |  | 0% | 0% | 0% | 0% |  | 0% |
| **glob_pattern** | 0% |  | 0% | 61% | 8% |  | 16% |  | 14% |  | 1% |  |  |  |
| **grep_search** |  | 0% | 0% | 5% | 54% | 0% | 14% |  | 26% |  | 0% | 0% |  |  |
| **lint_or_typecheck** | 0% |  | 0% |  | 0% | 65% | 0% |  | 0% |  | 15% | 18% |  |  |
| **list_directory** |  |  | 0% | 4% | 8% |  | 64% | 0% | 22% |  | 0% | 0% |  |  |
| **plan_task** |  | 21% | 1% | 0% |  |  |  | 71% | 0% |  |  |  | 8% | 0% |
| **read_file** | 0% | 0% | 0% | 4% | 15% | 0% | 17% | 0% | 61% | 0% | 1% | 0% |  | 0% |
| **respond_only** |  |  |  |  |  |  |  |  |  | 100% |  |  |  |  |
| **run_bash** | 0% |  | 0% |  | 0% | 7% | 0% |  | 1% | 0% | 83% | 9% |  |  |
| **run_tests** |  | 0% |  | 0% | 0% | 8% |  |  | 1% | 0% | 7% | 84% |  |  |
| **web_search** |  | 10% |  |  | 1% |  |  | 11% |  |  | 1% |  | 77% | 1% |
| **write_file** |  |  | 1% |  |  |  |  |  |  |  |  |  |  | 99% |

## 요약: hist6 대비 hist12에서 무엇이 바뀌었나

- 오분류 감소 쌍: 9개 (합계 델타 -116건)
- 오분류 증가 쌍: 4개 (합계 델타 +18건)
- 가장 크게 준 쌍: edit_file→apply_patch(-48), list_directory→grep_search(-5), lint_or_typecheck→run_tests(-19)
- 가장 크게 는 쌍: read_file→grep_search(+8), grep_search→glob_pattern(+7), plan_task→ask_user(+2)
- F1 상승 클래스(10개): web_search(+0.0817), apply_patch(+0.0558), lint_or_typecheck(+0.0445), run_tests(+0.0331), edit_file(+0.0245), ask_user(+0.0221), plan_task(+0.0130), glob_pattern(+0.0109), run_bash(+0.0101), list_directory(+0.0096)
- F1 하락 클래스(3개): grep_search(-0.0012), read_file(-0.0013), write_file(-0.0025)
