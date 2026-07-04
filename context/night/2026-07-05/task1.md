# task1 — 2-way(linear+stacker) 제출 후보 조립·검증

## 컨텍스트

DACON 236694 (AI 에이전트 다음 행동 14클래스 분류, Macro-F1). 우리 팀 최고 기록 w112(LB 0.7208)는
linear+stacker+encoder 3-way 앙상블인데 인코더 가중치가 아직 없다(Colab 재학습 중). 인코더가
도착하는 즉시 3-way를 조립할 수 있도록, 먼저 **인코더 없는 2-way(linear+stacker) 제출 후보**를
지금 있는 재료로 조립해 전 과정을 검증해 둔다. 이게 통과되면 3-way 재건은 성분 하나 추가일 뿐이다.

## 목표 / 완료 조건 (DoD)

1. 이 워크트리의 `submit_candidates/two_way/` 아래에 `script.py` + `requirements.txt` + `model/` 구성
2. zip 패키징 후 검증 하네스 **12/12 PASS** — 실행 명령과 전체 출력을 `context/night/2026-07-05/task1_validate.log`로 저장
3. `context/night/2026-07-05/task1_report.md` 작성: 조립 방법(어떤 파일을 어디서 가져왔는지), 검증 결과, 예상 LB(성분 solo: linear 0.6732 / stacker 0.6708, uniform 평균이면 ~0.69 추정), 아침에 실제 제출할 가치가 있는지 의견
4. 모든 산출물 git commit (이 워크트리 브랜치에)
5. 마지막으로 `context/night/2026-07-05/task1.DONE` 생성 (5줄 요약 포함) — 이 파일이 러너의 완료 신호다

## 재료 (절대 경로 — 이 워크트리에는 gitignore된 파일이 없다)

- 팀 리포 (읽기 전용, 필요 파일은 이 워크트리로 **복사**): `C:\dev\dacon-agent-action-api-boost`
  - 3-way 추론 코드: `ensemble\script_3way.py`, `ensemble\aar_infer.py`, `ensemble\features.py`
  - linear 아티팩트: `linear_pipeline\submit\model\model.pkl` (8MB)
  - stacker 아티팩트: 루트 `model\` (aar_models.joblib 31.5MB, model.joblib 29.5MB, prompt_model.joblib 17.4MB, aar_config.json)
  - 참고: 루트 `submit.zip`(78.5MB)은 stacker 단독 계열 제출물 — 구조 참고용
- 데이터 (읽기 전용): `C:\dev\2026-AI-DACON\data\` (test.jsonl 5행 샘플, sample_submission.csv)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — **시스템 파이썬 금지**, 팀 아티팩트가 sklearn 1.9에서 역직렬화 실패함)
- 검증 하네스: `C:\dev\2026-AI-DACON\scripts\validate_submit.py` (워크트리에도 있음 — 워크트리 사본 사용)

## 금지

- 팀 리포·메인 리포(C:\dev\2026-AI-DACON) 작업 트리 수정 금지 — 산출물은 전부 이 워크트리 안에만
- `git push` 금지 (커밋만; 병합은 아침에 사람이)
- zip 파일(수십 MB)은 커밋하지 말 것 (.gitignore가 막지만 확인)
- 제출 script.py에 네트워크 호출 코드 금지 (평가 서버는 오프라인)
- requirements.txt에 서버 기본 패키지(pandas/numpy/sklearn/joblib/torch/transformers) 재핀 금지 — 팀 정책은 주석만

## 진행 프로토콜 (재개 대비 — 필수)

1. 시작하자마자 `context/night/2026-07-05/PROGRESS-task1.md`를 확인 — 존재하면 '다음 재개 지점'부터 이어서, 없으면 생성
2. 의미 단위(파일 복사 완료 / script.py 조정 완료 / 첫 실행 성공 / 12검증 통과)마다 PROGRESS 갱신 + git commit
3. 전부 끝나면 task1.DONE 생성 + 최종 커밋

## 작업 내용

1. `ensemble\script_3way.py`를 정독하고, encoder 성분 없이 linear+stacker만으로 동작하는 조립 방식을 파악하라
   (서빙 메커니즘은 전부 default-off: weights.json / calib.json / 다중 encoder 블록 / bucket_weights.json.
   model/ 디렉터리 구성으로 성분을 감지하는 방식일 가능성이 높다 — 코드로 확인하고 리포트에 기록).
2. `submit_candidates/two_way/`에 script.py(=script_3way.py 기반), requirements.txt(팀 정책: 주석만), model/(linear+stacker 아티팩트 배치) 구성.
   weights는 [1,1] uniform (인코더 없음). 필요 시 weights.json 생성.
3. 로컬 스모크: 워크트리에서 서버 레이아웃 재현해 script.py 직접 실행, output/submission.csv 5행 생성 확인.
4. zip 패키징 후 `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts\validate_submit.py <zip> --data-dir C:\dev\2026-AI-DACON\data` 12/12 확인.
5. task1_report.md + DONE 작성.
