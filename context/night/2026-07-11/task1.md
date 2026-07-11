# task1 — AAR stacker 트레이너 재구현 (재현성 보험)

## 컨텍스트

DACON 236694(AI 에이전트 다음 행동 14클래스, Macro-F1) 예선 마감 7/15. 최종 제출은 챔피언 0.7623으로 확정(D-013)이고, 상위 12팀이 되면 7/20 재현 코드 제출이 필요하다. 재현성 감사(coordination 07-11 오후 노트) 결과 챔피언 5성분 중 **AAR stacker 트레이너만 소실** — 원본 `train_tscar.py`는 동료 리포 work2(`C:\dev\dacon-agent-action-api-boost-work2`)에 있었고 그 디렉토리는 삭제됐다. 메타데이터와 추론 소비자는 남아 있으므로 프로토콜 동등 재구현이 목표다. top-12 조건부 보험이지만, 컷 진입 순간 없으면 실격 리스크다.

## 목표 / 완료 조건(DoD)

1. `scripts/aar_rebuild/train_aar.py` — meta.json에 기록된 레시피(4-view SGD 후보 + greedy_blend inner selection + oof_stack_validation(logreg), 3-fold 세션 group)를 재구현한 트레이너. 시드 고정, env 폴백 argparse(required 인자 금지).
2. 산출물 포맷 계약: 학습 산출물을 `submit/aar_infer.py`가 **수정 없이** 로드·추론할 수 있어야 한다(로드 스모크 테스트 포함). 기존 `submit/model`의 AAR 아티팩트 파일 구조·키를 먼저 역공학해 스키마 문서(`scripts/aar_rebuild/SCHEMA.md`)로 남길 것.
3. 재현 검증 리포트(`scripts/aar_rebuild/REPRODUCTION.md`): 재학습 stacker의 세션 group 3-fold OOF Macro-F1 vs 기존 기록(OOF ~0.71 안팎, exp #2 holdout 0.7098) 비교, 기존 submit/model AAR 예측과의 행 일치율, 정확 재현이 불가능한 지점(초기화·버전·소실 하이퍼파라미터)의 명시적 목록.
4. `pytest` 신규 테스트(스키마 계약 + 트레이너 스모크) 통과, 기존 스위트 무손상.
5. **`context/night/2026-07-11/task1.DONE` 생성(3줄 요약 포함)** — 러너 완료 판정 기준.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `data\train_labels.csv` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)
- 레시피 메타: `C:\dev\2026-AI-DACON\artifacts\oof\oof_2026_07_03\meta.json`, `artifacts\oof\oof_rebuild_2026_07_04\meta.json` (읽기 전용 — `stacker` 키에 레시피 서술, `inner_weights`·`greedy_blend`·클래스 순서 포함)
- 추론 소비자(계약의 기준): 워크트리 내 `submit/aar_infer.py` (git 추적됨). 기존 학습 아티팩트: `C:\dev\2026-AI-DACON\submit\model\` (읽기 전용 — 필요한 파일은 워크트리 밖에서 읽기만)
- 기존 OOF 산출물(비교 기준): `C:\dev\2026-AI-DACON\artifacts\oof\**` (읽기 전용)

## 금지

- 워크트리 밖(메인 리포 작업트리·`C:\dev\2026-AI-DACON\submit\model` 등) **수정 금지** — 읽기만.
- `git push` 금지, 수동 zip 금지, 네트워크 호출 코드 금지(오프라인 규약).
- 폐기 목록(experiments.md 재시도 금지 테이블) 재시도 금지 — 이 작업은 신규 모델링이 아니라 **기존 성분의 재현**이다. 성능 개선 시도 금지(레시피 이탈 금지).
- GPU 사용 금지 — 전부 CPU(SGD/LogReg 계열이라 가능).

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-11/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터 이어서.
2. 의미 단위 작업(스키마 역공학 → 트레이너 골격 → view 재구현 → greedy_blend → 검증 리포트)마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit**.
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋.

## 작업 내용

1. `submit/aar_infer.py` 전체를 읽고 로드하는 아티팩트(파일명·joblib 구조·기대 키)를 역공학 → `SCHEMA.md`.
2. `C:\dev\2026-AI-DACON\submit\model\`에서 AAR 관련 파일을 열어(읽기 전용) 실제 구조를 스키마와 대조.
3. meta.json의 `stacker` 레시피 서술을 기반으로 `train_aar.py` 구현: build_views(4-view) → 후보 SGD 학습 → greedy_blend inner selection → logreg oof_stack_validation, 세션 group 3-fold(`id.rsplit("-step_",1)[0]` 그룹).
4. 소실 하이퍼파라미터는 meta.json·aar_infer.py의 흔적에서 복원하고, 복원 불가 항목은 합리적 기본값 + REPRODUCTION.md에 "추정" 표기.
5. 학습 실행(70,000행, CPU) → 산출물을 aar_infer.py로 로드해 스모크 추론 → OOF F1·행 일치율 산출 → REPRODUCTION.md.
6. pytest 작성·통과 → DONE.
