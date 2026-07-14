# Freeze check 도구 보고서

## 사용법

프로젝트 가상환경에서 인자 없이 실행한다.

```powershell
& C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\freeze_check\check.py
```

기본 입력은 읽기 전용 `C:\dev\2026-AI-DACON`의 `submit/`,
`context/submissions.md`, Git 작업트리다. 결과는 이 도구가 있는 worktree의
`context/reports/freeze_manifest_2026-07-14.json`과
`context/reports/freeze_checklist_2026-07-14.md`에 쓴다.

경로를 바꿔야 할 때는 선택 인자 또는 다음 환경변수를 쓸 수 있다. 필수 argparse 인자는 없다.

- `FREEZE_PROJECT_ROOT`, `FREEZE_OUTPUT_ROOT`, `FREEZE_SUBMIT_DIR`
- `FREEZE_LEDGER_PATH`, `FREEZE_GIT_ROOT`, `FREEZE_TEMP_DIR`, `FREEZE_DISK_ROOT`
- `FREEZE_DATE`, `FREEZE_EXPECTED_SUBMISSION`, `FREEZE_EXPECTED_LB`, `FREEZE_MIN_FREE_GB`

FAIL이 하나라도 있으면 exit 1, FAIL 없이 WARN만 있으면 exit 0이다. 검사기는 외부
`submit/`, 대장, TEMP 잔해를 수정하거나 삭제하지 않으며 서버에 접속하지 않는다.

## 실제 실행 출력 전문

명령: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\freeze_check\check.py`

```text
[PASS] submit/ SHA256 매니페스트: 25개 파일, 1931.6 MiB → C:\dev\night\2026-07-14\task3\context\reports\freeze_manifest_2026-07-14.json
[PASS] public LB 최고 제출: numeric LB 최고 = #14 (0.77089); 기대값 = #14 (0.77089)
[PASS] mBERT rollback 자산: 존재: C:\dev\2026-AI-DACON\colab_out\mbert_encoder2_backup
[PASS] fast_aar rollback 자산: 존재: C:\dev\2026-AI-DACON\submit\fast_aar.py
[PASS] encoder serialize 계약: C:\dev\2026-AI-DACON\submit\model\encoder\serialize_config.json: {"max_hist": 12}; 기대값 = {"max_hist": 12}
[PASS] 서버 rollback 경로: 문서 기재만(접속 시도 안 함): ~/models/champ_encoder_s42, ~/out/qwen05i_2ep_full
[WARN] TEMP 제출 잔해: 1개 발견(삭제하지 않음): C:\Users\wlgur\AppData\Local\Temp\dacon_submit_api-0.1.2-py3-none-any.whl (file, 4958 bytes)
[WARN] 디스크 여유: C:\: 3.96 GiB free (WARN 기준 10.00 GiB 미만)
[PASS] git 작업트리: clean: C:\dev\2026-AI-DACON
[REPORT] manifest: C:\dev\night\2026-07-14\task3\context\reports\freeze_manifest_2026-07-14.json
[REPORT] checklist: C:\dev\night\2026-07-14\task3\context\reports\freeze_checklist_2026-07-14.md
[RESULT] exit 0
```

종합은 PASS 7 / WARN 2 / FAIL 0이다. WARN은 TEMP 잔해 1개와 C: 여유 3.96 GiB이며,
요청대로 삭제·수정하지 않았다.

## 검증

- `python -m pytest -q tests/test_freeze_check.py`: **3 passed in 0.48s**
- 독립 재열거·재해시: **25개 파일 전부 SHA256 일치**
- `scripts/freeze_check/check.py` AST 파싱: **PASS**
- 실데이터 실행: **exit 0**, 매니페스트·체크리스트 2파일 생성 확인
- 외부 `C:\dev\2026-AI-DACON` Git 작업트리: **clean**

## 라우팅 / handoff

- 조사 라우팅: `gpt-5.6-luna`, reasoning medium, read-only — 네트워크 소켓/HTTPS 차단으로 모델 실행 전 실패
- 최종 감사 라우팅: `gpt-5.6-sol`, reasoning high, read-only — 같은 네트워크 차단으로 모델 실행 전 실패
- 폴백: 현재 Codex 주 세션이 구현·로컬 검증 수행
- branch: `night/2026-07-14/task3`
- commit: 생성 불가 — 공용 Git 메타데이터
  `C:\dev\2026-AI-DACON\.git\worktrees\task31\index.lock` 쓰기 권한이 현재 샌드박스에 없음
- 제출, push, main 승격, 외부 파일 수정: 수행하지 않음

