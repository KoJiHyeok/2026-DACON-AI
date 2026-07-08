# task1 — serialize() 계약 회귀 테스트 (byte-identical 자동 가드)

## 컨텍스트

DACON 236694. `serialize()`는 인코더의 **학습·추론 계약**이다 — 학습 텍스트와 추론 텍스트가
한 글자라도 달라지면 조용한 오답이 난다. 현재 이 함수가 **4곳에 복제**돼 있는데(제출 script,
학습 스크립트 3종), 지금은 reviewer가 수동 AST 비교로만 동일성을 확인했다. 누가 한 곳만 고치면
드리프트가 조용히 생긴다. 이를 **자동 회귀 테스트**로 고정해 CI/제출 게이트가 잡게 한다.

## 목표 / 완료 조건 (DoD)

1. `tests/test_serialize_contract.py` 작성 — 아래 4개 파일의 `serialize()`와 `_bucket()` 정의를
   **AST(docstring 제외) 비교**해 전부 동일함을 assert한다. 기준(정본)은 `submit/script.py`.
   - `submit/script.py`
   - `colab/encoder_v2_s42_repro.py`
   - `colab/mdeberta_finetune.py`
   - `colab/encoder_e5_holdout85_maxhist.py`
2. **char-cap 상수 존재 검증**: 각 `serialize()`가 `[:800]`(query)·`[:120]`(result_summary)·
   `[:200]`(user content)·`[:5]`(open_files)와 `reversed(hist[-max_hist:])` 패턴을 포함하는지
   소스 문자열 수준으로도 확인(AST 동일성의 이중 안전망).
3. **테스트가 실제로 드리프트를 잡는지 자기증명**: 정본 `serialize()` AST 덤프를 프로그램적으로
   한 글자 변형한 뒤 비교 함수가 **불일치를 반환**함을 assert하는 케이스를 포함(가드에 이빨이
   있음을 증명 — 항상-통과 테스트 방지).
4. `.venv`의 pytest로 **전체 통과** 확인, 실행 로그를 `context/night/2026-07-09/task1_run.log`에 저장.
5. `context/night/2026-07-09/task1_report.md` — 무엇을 어떻게 비교했는지, 4파일 동일성 결과,
   드리프트 감지 자기증명 결과, 한계(예: 문자열 수준 검사가 못 잡는 케이스).
6. 마지막: `context/night/2026-07-09/task1.DONE` 생성(5줄 요약).

## 재료 (절대 경로)

- 검사 대상 4파일: `C:\dev\2026-AI-DACON\{submit\script.py, colab\encoder_v2_s42_repro.py,
  colab\mdeberta_finetune.py, colab\encoder_e5_holdout85_maxhist.py}` (읽기 전용 — 워크트리에 커밋돼 있음)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (pytest 설치됨, 시스템 파이썬 금지)
- 표준 라이브러리 `ast`만 사용(외부 의존성 추가 금지)
- 기존 테스트 스타일 참고: `C:\dev\2026-AI-DACON\tests\` (특히 features 모듈 캐시 격리 패턴은
  이 작업엔 불필요 — 순수 소스 파싱이라 import 안 함)

## 금지

- 워크트리 밖(메인 리포 작업트리·팀 리포) 수정 금지, `git push` 금지, 수동 zip 제출 금지
- `serialize()` 정의 자체를 **수정 금지** — 이 작업은 "동일성 검증"만, 리팩터링 아님
  (만약 4파일이 이미 불일치라면 **고치지 말고** 리포트에 불일치를 명시하고 테스트는 xfail 표시)
- 외부 패키지 설치 금지(ast 표준 라이브러리로 충분)
- 폐기 목록 위반 없음(이 작업은 하이진 테스트라 무관)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-09/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터.
2. 의미 단위(파서 유틸 / 4파일 비교 / char-cap 검증 / 드리프트 자기증명 / pytest 통과)마다
   PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit**.
3. 전부 끝나면 `task1.DONE` + 최종 커밋.

## 작업 내용

1. `ast` 헬퍼: 파일 경로 → 소스 파싱 → 최상위(또는 중첩) `FunctionDef` 중 이름이 `serialize`/`_bucket`인
   노드를 찾아, **docstring 노드를 제거**한 뒤 `ast.dump(node, annotate_fields=True)` 문자열 반환.
   (docstring 제거: 함수 body 첫 노드가 `ast.Expr`이고 값이 `ast.Constant(str)`이면 pop.)
2. 정본(`submit/script.py`)의 덤프를 기준으로 나머지 3파일과 각각 비교 — 불일치 시 어느 파일·어느
   함수가 다른지 diff 요약과 함께 실패시킨다(파일:함수 단위 assert).
3. char-cap 문자열 검사: 각 파일 소스에서 `[:800]`·`[:120]`·`[:200]`·`open_files[:5]`·
   `reversed(hist[-max_hist:])`(공백 무시 정규화) 존재를 assert.
4. 드리프트 자기증명: 정본 덤프 문자열의 `120`→`121`처럼 안전하게 1치환한 가짜 덤프를 만들어
   비교 함수가 False를 반환함을 assert(테스트 로직에 이빨이 있음을 고정).
5. pytest 실행 로그 저장 + 리포트 + DONE.
