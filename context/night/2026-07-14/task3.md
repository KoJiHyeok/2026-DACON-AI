# task3 — 동결(freeze) 점검 도구: 최종 제출 확정 전 자동 체크리스트

## 컨텍스트

DACON 236694. 예선 마감 **07-15 09:59**, 오늘(07-14) 18시 동결 방침: 최고 제출(#14, LB 0.77089)의 정상 채점·아티팩트 해시·롤백 경로를 확인하고 더 이상 건드리지 않는다. 지금은 이 확인이 수작업이라 누락 위험이 있다 — 실행 한 번으로 동결 전 점검을 끝내는 도구를 만든다. (최종 선택은 public 최고 자동 — D-013 추록 2)

## 목표 / 완료 조건 (DoD)

1. `scripts/freeze_check/check.py` 신규 (argparse required 금지 — 인자 없이 실행 가능, env 폴백):
   - (a) `submit/` 스테이징 전 파일 SHA256 매니페스트 생성 → `context/reports/freeze_manifest_2026-07-14.json` (파일별 크기·해시, 총량)
   - (b) `context/submissions.md` 표 파싱 → LB(public) 최고 행 식별, **#14(0.77089)인지 assert** (숫자 아닌 셀·"시간초과 FAIL" 같은 텍스트 셀은 건너뜀)
   - (c) 롤백 자산 점검: 로컬 확인 가능 항목 존재 여부 (`colab_out/mbert_encoder2_backup`, `submit/model/encoder/serialize_config.json` 내용 == {"max_hist": 12}, `submit/fast_aar.py` 존재) + 서버 보존 경로는 문서 기재만 (`~/models/champ_encoder_s42`, `~/out/qwen05i_2ep_full` — 접속 시도 금지)
   - (d) 위생 점검: `%TEMP%\dacon_submit_*` 잔해 목록, C: 드라이브 여유 GB, git 작업트리 clean 여부
   - (e) 종합 결과 → `context/reports/freeze_checklist_2026-07-14.md` (항목별 PASS/WARN/FAIL + 사람이 볼 요약) — 모두 PASS면 exit 0, FAIL 있으면 exit 1
2. `tests/test_freeze_check.py` — 최소 3케이스 (대장 파싱이 #14를 최고로 뽑는지 / 매니페스트 생성 스모크 / FAIL 조건 시 exit 1) — pytest 통과
3. 실제 실행 1회: check.py exit 0 + 리포트 2파일 생성 확인 (현행 상태 기준 전 항목 PASS 기대 — FAIL이 있으면 고치지 말고 리포트에 그대로 남길 것, 판단은 아침 회수 몫)
4. `context/night/2026-07-14/report_freeze_tool.md` — 도구 사용법 + 실행 출력 전문
5. **`context/night/2026-07-14/task3.DONE` 생성 (한 줄 요약 포함)**

## 재료 (절대 경로)

- 대장: `C:\dev\2026-AI-DACON\context\submissions.md` (읽기 전용)
- 스테이징: `C:\dev\2026-AI-DACON\submit\` (읽기 전용 — 해시만)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`
- 참고 규약: `C:\dev\2026-AI-DACON\docs\t4_rehearsal.md`, `C:\dev\2026-AI-DACON\scripts\make_submit.py` (검증 스타일 참고)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 네트워크 호출 금지 (서버 SSH 접속 시도 포함), 제출 금지
- `submit/`·대장·메인 리포 파일 변경 금지 — 도구는 **읽고 리포트만** 생성
- %TEMP% 잔해 발견 시 삭제하지 말고 목록만 기록 (삭제 판단은 아침 회수 몫)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-14/PROGRESS-task3.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신 후 **git commit**
3. 전부 끝나면 `task3.DONE` 생성 + 최종 커밋

## 작업 내용

1. check.py 골격: 점검 항목을 (이름, 함수, PASS/WARN/FAIL) 리스트로 등록하는 단순 구조 — 외부 의존성 stdlib만
2. (a)~(d) 구현 → (e) 리포트 렌더링·exit code
3. tests 작성 (대장 파싱은 임시 fixture 마크다운으로도 검증) → pytest 통과
4. 실제 1회 실행·출력 검토 → report_freeze_tool.md → DONE
