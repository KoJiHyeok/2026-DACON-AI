# CX-B progress

- 2026-07-14: 신규 시작. `context/coordination.md`와 task5 ticket 확인 완료.
- Sol (`gpt-5.6-sol`, reasoning high) 읽기 전용 라우팅을 시도했으나 샌드박스 네트워크 차단(os error 10013)으로 실패. 파일 변경 없음.
- 외부 입력 스키마와 행 정렬 확인: Qwen/holdout 모두 9,969행 x 14클래스, ids/y_true/actions 정렬 일치. AU cache 682행은 holdout AU 순서 및 actions 일치.
- 기준 재현: 구표면 0.7676045974, 신표면 0.7713812961. target F1 delta는 list_directory -0.0075254750, glob_pattern -0.0032660656.
- 1차 오류 위치: list_directory old-correct/new-wrong 13, 반대 1; glob_pattern 4, 반대 0. 전부 non-AU.
- `scripts/cx_errloc/analyze.py`, `analysis.json`, `transition_rows.csv`, `report_cx_errloc.md` 완료.
- 후보 실측(신표면 대비): list+glob Qwen log-bias +0.08은 Macro-F1 +0.00089349 / 정답 +10; list-only +0.08은 +0.00056247 / +6; glob-only +0.10은 +0.00019447 / +2.
- 기각 대조: w_q 2.8은 -0.00032740 / 정답 -3, alpha 0.90은 -0.00107405 / -13.
- 검증 PASS: py_compile, AST/JSON/CSV semantic assertions, 18개 전이 행, 외부 원본 git status clean, 재실행 SHA256 동일.
  - `analysis.json`: `f5baa3d977518ed5ffc018386ea27f79530ee5389775df1f925c4601d8c3e264`
  - `transition_rows.csv`: `5837f2e35a0d70eb4e3495ce1fb8eaeaf35d0fc7d57e2f4e8d22b849bb6e9af6`
- DONE 생성 완료. git add/commit 시도는 `C:\dev\2026-AI-DACON\.git\worktrees\task5\index.lock` 쓰기 권한 거부로 실패했다. 작업 파일은 unstaged 상태로 보존.
- reviewer/tester는 독립 재실행·배포 bias parity·독립 split/full 전이를 확인할 것.
