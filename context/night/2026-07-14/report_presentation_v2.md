# task2 handoff — 발표 사료 v2

## 작업 결과

- 07-13 미병합 원본 3개 파일을 `docs/presentation/`에 복원하고 exp #52와 제출 #13/#14까지 현행화했다.
- `key_numbers.md`에 이전 5성분 챔피언과 현 Qwen 하이브리드 챔피언을 분리하고, LB 궤적을 #1~#14·0.71884→0.77089로 확장했다.
- `sources.md`에 독립 절 `3.2 시간초과 대역전`을 추가했다. #13 시간초과, 활성 연산량 360M/24L vs 86M×2/12L, 길이정렬 1.7x, fast_aar 2.8x, 출력등가, T4 30k 515s, #14 0.77089, 73.5% 전이를 원문 근거에 연결했다.
- `verify_citations.py`를 무작위 20건 표본에서 기존 24건 + 신규 9건 = 33건 전수 검사로 바꿨다. 신규 검사는 exp #52 2건, 제출 #13/#14 2건, daily 07-13 3건, T4 리허설 2건이다.

## 입력과 SHA256

```text
6902cf978c56203cbe367222d995bab9689ee19866a0b2a5e1241397ac314585  C:\dev\2026-AI-DACON\context\experiments.md
abdb7258bcd3db61dd48c6517a3ac35b95ea28ba8ae7104570bca292e4ca4747  C:\dev\2026-AI-DACON\context\submissions.md
e85a1be784e4ead8c529400332b1ccb2d93cd97b1765f5a02e5e03801a2461a5  C:\dev\2026-AI-DACON\context\daily\2026-07-13.md
06f8d17e264ee80739bcfbee93573f10f967bde1be96d86ac9d92f32c479b13b  C:\dev\2026-AI-DACON\docs\t4_rehearsal.md
1252fbb24e50f9c06bf3f9d18fe101ff4677b80fd9deaa7dd14f7b0e99a00154  C:\dev\night\2026-07-13\task3\docs\presentation\key_numbers.md
e309698034fbc01e2395df549626428c75b1cfa1248e4bcff1e312d674ed0564  C:\dev\night\2026-07-13\task3\docs\presentation\sources.md
0ddd4e8fcba6ce9c899d8e9d4909da914059aee5e5c7c15a66e0a2a2dc61b311  C:\dev\night\2026-07-13\task3\docs\presentation\verify_citations.py
```

## 인용 검증 실행 출력 전문

명령:

```powershell
& 'C:\dev\2026-AI-DACON\.venv\Scripts\python.exe' docs/presentation/verify_citations.py
```

출력:

```text
checks=33/33 mode=exhaustive
| check | citation | source | result |
|---|---|---|---|
| C01 | reports/eda_distribution.md | `context/reports/eda_distribution.md` | PASS |
| C02 | reports/eda_distribution.md | `context/reports/eda_distribution.md` | PASS |
| C03 | reports/eda_distribution.md | `context/reports/eda_distribution.md` | PASS |
| C04 | reports/forensics_r1.md | `context/reports/forensics_r1.md` | PASS |
| C05 | reports/forensics_r1.md | `context/reports/forensics_r1.md` | PASS |
| C06 | reports/forensics_r1.md | `context/reports/forensics_r1.md` | PASS |
| C07 | reports/forensics_r2.md | `context/reports/forensics_r2.md` | PASS |
| C08 | D-010 | `context/decisions.md` | PASS |
| C09 | exp #34 | `context/experiments.md` | PASS |
| C10 | D-003 | `context/decisions.md` | PASS |
| C11 | audit | `context/reports/third_party_sol_model_audit_2026-07-10.md` | PASS |
| C12 | exp #43 | `context/experiments.md` | PASS |
| C13 | exp #35 | `context/experiments.md` | PASS |
| C14 | exp #51 | `context/experiments.md` | PASS |
| C15 | hist12 deploy review | `context/reports/verify_hist12_deploy_2026-07-10.md` | PASS |
| C16 | model audit components | `context/reports/third_party_sol_model_audit_2026-07-10.md` | PASS |
| C17 | model audit weights | `context/reports/third_party_sol_model_audit_2026-07-10.md` | PASS |
| C18 | ledger #1 | `context/submissions.md` | PASS |
| C19 | ledger #5 | `context/submissions.md` | PASS |
| C20 | ledger #6 | `context/submissions.md` | PASS |
| C21 | ledger #7 | `context/submissions.md` | PASS |
| C22 | ledger #11 | `context/submissions.md` | PASS |
| C23 | exp #50 | `context/experiments.md` | PASS |
| C24 | D-013 | `context/decisions.md` | PASS |
| C25 | exp #52 model | `context/experiments.md` | PASS |
| C26 | exp #52 runtime | `context/experiments.md` | PASS |
| C27 | ledger #13 | `context/submissions.md` | PASS |
| C28 | ledger #14 | `context/submissions.md` | PASS |
| C29 | daily 07-13 timeout | `context/daily/2026-07-13.md` | PASS |
| C30 | daily 07-13 recovery | `context/daily/2026-07-13.md` | PASS |
| C31 | daily 07-13 equivalence | `context/daily/2026-07-13.md` | PASS |
| C32 | T4 rehearsal timing | `docs/t4_rehearsal.md` | PASS |
| C33 | T4 rehearsal compute | `docs/t4_rehearsal.md` | PASS |
mismatches=0 PASS
```

## 추가 검증

```text
stale_markers=0
ast_parse=PASS
baseline_to_qwen=0.05205
transfer=73.5%
fast_aar=2.8x
repeat_output=identical
```

- stale marker 대상: `#1~#51`, `#1~#11`, `0.71884→0.7623`, `최종 선택은 e5-hist12`, `챔피언 0.7623 단독`.
- `verify_citations.py`를 연속 2회 실행해 출력이 동일함을 확인했다.

## 모델 라우팅과 Git handoff

- Routed model/reasoning: `gpt-5.6-luna`, medium, `-s read-only`, `ROUTED_TASK=1`.
- Routed result: 외부 네트워크 차단으로 OpenAI Responses API 연결 실패. 파일 수정은 없었고, 사실 추출은 주 에이전트의 로컬 `rg -n`과 33건 전수 검사로 대체했다.
- Routed branch/commit: N/A (read-only; 실행 실패).
- 작업 branch: `night/2026-07-14/task2`.
- 작업 commit: 생성 불가. 샌드박스가 `C:\dev\2026-AI-DACON\.git\worktrees\task21`을 읽기 전용으로 제공해 `index.lock: Permission denied`가 발생했다.
- Push/merge/submission: 수행하지 않음.

## 알려진 제한과 다음 검증자

- 독립 reviewer/tester 검증이나 main 승격은 수행하지 않았다. 다음 owner는 이 작업트리 변경을 리뷰하고, 쓰기 가능한 Git 메타데이터 환경에서 커밋한 뒤 필요한 commit만 승격해야 한다.
- 권장 커밋 메시지: `docs: refresh presentation through Qwen submission 14`.
