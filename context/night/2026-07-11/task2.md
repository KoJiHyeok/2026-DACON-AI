# task2 — 추론속도 프로파일링 하네스 + 등가성 검증 (본선 속도 10% 대비)

## 컨텍스트

DACON 236694 본선 산식이 확정됐다: 예선 Private 50% + **추론속도 10%** + 전문가심사 40%. 챔피언 제출물의 오프라인 추론 실측은 50.2초(T4)로 우수하지만, 본선에서 속도가 점수인 이상 **어디서 시간이 쓰이는지 성분별로 계측하는 도구**와 **출력 불변(등가성)을 증명하는 검증기**가 필요하다. 이 작업은 도구·리포트만 만들고 제출물(`submit/script.py`)은 건드리지 않는다.

## 목표 / 완료 조건(DoD)

1. `scripts/speed/profile_infer.py` — 챔피언 추론 파이프라인을 단계별(데이터 로드 / 피처·직렬화 / linear / AAR stacker / e5 인코더 / mBERT 인코더 / AU 라우팅 / blend·후처리)로 계측하는 하네스. 행 수(`SPEED_ROWS`, 기본 300)·모델 루트(`SPEED_MODEL_DIR`)·디바이스(`SPEED_DEVICE`, 기본 cpu)를 env로 받아 **로컬 CPU와 서버 GPU에서 같은 스크립트가 돌게** 한다 (env 폴백, required argparse 금지).
2. 등가성 검증: 하네스 경유 예측 == 기준 경로(`submit/script.py` 로직 그대로) 예측이 **행 단위 100% 일치**함을 같은 서브셋에서 증명하는 `scripts/speed/equivalence_check.py` (불일치 시 해당 행 id 덤프).
3. `scripts/speed/REPORT.md`: CPU 서브셋 기준 단계별 시간 분해표(절대·비율), 병목 상위 2개, **script.py 수정 없이 가능한 안전 최적화 후보 2개**(예: 토크나이저 배치 크기, fp16 캐스팅 시점, 직렬화 캐시)를 근거와 함께 — **report-only, 구현 금지**.
4. `pytest` 스모크(하네스가 30행에서 완주 + 등가성 PASS) 통과.
5. **`context/night/2026-07-11/task2.DONE` 생성(3줄 요약 포함)**.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` (읽기 전용 — test.jsonl은 없다; train에서 서브셋을 만들어 계측)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (torch CPU 포함 — 시스템 파이썬 금지)
- 제출 코드(계측 대상, git 추적됨): 워크트리 내 `submit/script.py`, `submit/features.py`, `submit/aar_infer.py`, `submit/au_route.py`
- 모델 가중치: `C:\dev\2026-AI-DACON\submit\model\` (읽기 전용, 워크트리엔 없음 — `SPEED_MODEL_DIR` 기본값으로 이 절대 경로 사용)

## 금지

- **`submit/**` 수정 금지** (계측은 import·서브프로세스로만). 워크트리 밖 수정 금지, `git push` 금지, 네트워크 코드 금지.
- 최적화 "구현" 금지 — 후보 제안까지만 (수정은 아침에 Claude가 reviewer/tester 게이트로).
- GPU 강제 금지 — 기본 CPU. e5·mBERT의 CPU 추론이 느리므로 서브셋(기본 300행)을 넘기지 말 것.

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-11/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터.
2. 의미 단위(하네스 골격 → 단계 계측 → 등가성 → 리포트)마다 PROGRESS 갱신 후 **git commit**.
3. 전부 끝나면 `task2.DONE` 생성 + 최종 커밋.

## 작업 내용

1. `submit/script.py`를 읽고 추론 경로를 단계 함수로 분해 가능한 지점을 파악한다 (수정 없이 import하거나, 불가피하면 로직을 하네스에 복제하되 등가성 검증으로 복제 정확성을 증명).
2. `profile_infer.py`: train.jsonl 앞쪽 N행(세션 단위로 자르지 말고 행 단위 서브셋, 재현 가능하게 고정) → 단계별 `time.perf_counter()` 계측, 3회 반복 중앙값, 결과 JSON+표 출력.
3. `equivalence_check.py`: 같은 서브셋에 대해 기준 경로 예측 CSV vs 하네스 예측 CSV 행 일치율 100% 확인.
4. CPU 실측 → REPORT.md 작성 (서버 GPU 재실행 커맨드 한 줄 포함: env만 바꾸면 되게).
5. pytest → DONE.
