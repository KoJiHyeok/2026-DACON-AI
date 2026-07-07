# task3 subroute sweep report

## Setup

- Join sanity passed:
  - 3-way `(lin + stk + 2*e5) / 4` = `0.71725922`
  - 4-way `(lin + stk + 1.2*e5 + 0.8*mbert) / 4` = `0.72254583`
- Current final mirror for probe deltas: 4-way + soft-AU `alpha=0.9` = `0.73877228`.
- All candidate masks exclude `sess_au` rows to avoid double-routing with the existing AU route.
- `turn_index == 0` was not used as a candidate route. The screened candidates below also have `turn0% = 0.0`, so this is not a first-step rerun.
- Artifacts:
  - `night_out/task3_subroute/sweep.json`
  - `night_out/task3_subroute/sweep_rows.csv`
  - `night_out/task3_subroute/probe.json`
  - `night_out/task3_subroute/probe_route_rows.csv`

## 1st-pass Sweep

Screen rule: group F1 <= overall 4-way F1 - `0.03`, holdout >= `300`, train(nonholdout) >= `3000`.

No id-prefix candidate remained after excluding `sess_au`; the train data only exposed `sess_sim` and `sess_au` at the family level.

| Group | H | Train nonholdout | turn0% | F1 | Delta | Screen |
|---|---:|---:|---:|---:|---:|---|
| `budget_bucket=0000_1000` | 10 | 59 | 0.0% | 0.090909 | -0.631637 | - |
| `budget_bucket=1000_10000` | 188 | 1183 | 0.0% | 0.569290 | -0.153256 | - |
| `cross:open_files_empty&git_dirty=false` | 1324 | 8140 | 0.0% | 0.583398 | -0.139147 | PASS |
| `open_files_empty` | 3243 | 18604 | 0.0% | 0.633040 | -0.089505 | PASS |
| `turn_index>=12` | 608 | 3657 | 0.0% | 0.633106 | -0.089440 | PASS |
| `git_dirty=false` | 1891 | 11886 | 0.0% | 0.635397 | -0.087148 | PASS |
| `turn_index>=8` | 2158 | 13548 | 0.0% | 0.660674 | -0.061872 | PASS |
| `turn_index>=10` | 1174 | 7369 | 0.0% | 0.670218 | -0.052328 | PASS |
| `history_len>=10` | 3822 | 23311 | 0.0% | 0.687130 | -0.035416 | PASS |
| `workspace_loc>=p90_49691` | 898 | 5601 | 0.0% | 0.722155 | -0.000391 | - |
| `user_tier=free` | 2849 | 16939 | 0.0% | 0.723479 | +0.000933 | - |
| `language_pref=en` | 2376 | 14234 | 0.0% | 0.726992 | +0.004446 | - |
| `last_ci_status=passed` | 3586 | 21663 | 0.0% | 0.731778 | +0.009232 | - |
| `git_dirty=true` | 7396 | 43802 | 0.0% | 0.733880 | +0.011335 | - |
| `budget_bucket=50000_plus` | 6562 | 39190 | 0.0% | 0.735818 | +0.013273 | - |
| `language_pref=ko` | 6005 | 35895 | 0.0% | 0.736027 | +0.013481 | - |
| `last_ci_status=none` | 2752 | 15537 | 0.0% | 0.736073 | +0.013527 | - |
| `last_ci_status=failed` | 2949 | 18488 | 0.0% | 0.737319 | +0.014773 | - |
| `user_tier=pro` | 5119 | 30508 | 0.0% | 0.739614 | +0.017068 | - |
| `workspace_loc<=p10_3200` | 986 | 5517 | 0.0% | 0.739632 | +0.017086 | - |
| `budget_bucket=10000_50000` | 2527 | 15256 | 0.0% | 0.743825 | +0.021279 | - |
| `user_tier=enterprise` | 1319 | 8241 | 0.0% | 0.746056 | +0.023510 | - |
| `language_pref=mixed` | 906 | 5559 | 0.0% | 0.751005 | +0.028459 | - |

The only selected cross group was `open_files_empty & git_dirty=false`, because those were the overlapping weak non-threshold signals that still kept enough rows.

## 2nd-pass Probe

Specialist recipe: `char_wb(3-5)`, `max_features=120000`, `LinearSVC(C=1.0, class_weight=balanced)`. Training rows are only nonholdout rows inside the candidate group.

Decision rules:

- `delta_vs_soft_au >= +0.005`: LB gate candidate.
- `+0.002 <= delta_vs_soft_au < +0.005`: report only.
- Specialist hard group F1 must beat blend group F1 by at least `+0.02`; otherwise classify as information-limited, following exp #25.

| Group | H | Train | Blend F1 | Specialist hard F1 | Hard margin | Best alpha | Best soft delta | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `cross:open_files_empty&git_dirty=false` | 1324 | 8140 | 0.583398 | 0.404322 | -0.179077 | 0.5 | -0.000727 | discard_info_limited |
| `open_files_empty` | 3243 | 18604 | 0.633040 | 0.449146 | -0.183895 | 0.5 | -0.001075 | discard_info_limited |
| `turn_index>=12` | 608 | 3657 | 0.633106 | 0.221190 | -0.411916 | 0.6 | +0.000584 | discard_info_limited |
| `git_dirty=false` | 1891 | 11886 | 0.635397 | 0.407858 | -0.227540 | 0.6 | -0.000560 | discard_info_limited |
| `turn_index>=8` | 2158 | 13548 | 0.660674 | 0.270946 | -0.389728 | 0.6 | -0.000537 | discard_info_limited |
| `turn_index>=10` | 1174 | 7369 | 0.670218 | 0.251418 | -0.418799 | 0.5 | +0.000067 | discard_info_limited |
| `history_len>=10` | 3822 | 23311 | 0.687130 | 0.327452 | -0.359678 | 0.5 | +0.000439 | discard_info_limited |

Top soft rows, for scale:

| Group | alpha | Group F1 | group delta | delta vs 4way | delta vs soft-AU | changed |
|---|---:|---:|---:|---:|---:|---:|
| `turn_index>=12` | 0.6 | 0.640781 | +0.007674 | +0.000572 | +0.000584 | 19 |
| `turn_index>=12` | 0.5 | 0.639212 | +0.006106 | +0.000456 | +0.000465 | 15 |
| `history_len>=10` | 0.5 | 0.688981 | +0.001851 | +0.000423 | +0.000439 | 117 |
| `turn_index>=12` | 0.7 | 0.637532 | +0.004426 | +0.000281 | +0.000287 | 30 |
| `turn_index>=10` | 0.5 | 0.670195 | -0.000023 | +0.000058 | +0.000067 | 26 |

## Verdict

No candidate reaches the `+0.005` LB gate, and none even reaches the `+0.002` report-only band. The best final delta is `+0.000584` on `turn_index>=12` at `alpha=0.6`, far below the submission threshold.

More importantly, every specialist hard model is worse than the 4-way blend inside its own group. The weakness signals are real in the first-pass table, but they are not AU-like: the dedicated char specialist does not recover them. This matches the exp #25 lesson: when the weak slice is information-limited or already better handled by the ensemble, routing is ineffective.

Conclusion: this candidate lane is exhausted for the tested metadata/id-prefix groups under the AU specialist recipe. No submission should be made from task3.
