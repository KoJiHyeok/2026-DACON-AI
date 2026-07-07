# task1 — merge080 회귀 테스트 (align_by_label · mbert_mix · validate_mbert)

## 컨텍스트

DACON 236694 (14클래스 Macro-F1). 오늘 동료 0.7511 제출물에 우리 mBERT를 병합한 `submit_candidates/merge080/`를 제출했다(대장 #9, exp #30). reviewer가 "가장 위험한 학습/추론 불일치 범주(라벨 재정렬·믹스 수식)에 자동 테스트가 없다"고 지적했다 — 이 후보가 승격되면 앞으로 계속 수정될 파일이므로 회귀 테스트가 필요하다. 어제 enc_block_weights BOM 사고도 테스트 부재로 G4까지 가서야 잡혔던 전례.

## 목표 / 완료 조건 (DoD)

1. `tests/test_merge080_script.py` 추가 — 메인 `.venv`(torch/transformers 없음)에서 pytest 통과. 커버:
   - **align_by_label 방향**: 합성 확률 행렬 + 알파벳순 id2label 가짜 config로, 결과 열 k가 정확히 ACTIONS[k](스펙 순서)의 확률이 되는지. (인코더 실행 없이 순수 재정렬 로직만 검증 — 재정렬 코드를 함수로 뽑아내지 말고 script.py 그대로 테스트할 방법을 먼저 찾되, 불가능하면 최소 추출 리팩토링은 허용)
   - **믹스 수식**: mix=0.2에서 (1-w)·final + w·mbert 후 행합 1, mix=0이면 mbert 경로 미실행(원본 동일).
   - **validate_mbert**: mix>0 + 디렉토리 없음 → FileNotFoundError / mix=0 → 통과(스킵) / 라벨 셋 불일치 config → RuntimeError.
   - **env 오버라이드**: MBERT_MIX env가 config보다 우선.
2. torch/transformers/joblib/scipy 의존은 `tests/test_enc_block_weights.py`의 모듈 스텁 패턴(sys.modules 주입 + 캐시 격리)을 따라 해결 — src/ 패키지 import도 merge080 디렉토리 sys.path 격리로 처리하고 테스트 후 원상복구.
3. 전체 스위트 회귀 없음: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe -m pytest -q tests/` 전부 통과 (pytest는 .venv에 설치돼 있음).
4. `context/night/2026-07-08/report_merge080_tests.md` — 커버리지 요약 (**파일명 task로 시작 금지**).
5. `context/night/2026-07-08/task1.DONE` 생성(한 줄 요약) + 최종 커밋.

## 재료 (절대 경로)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (pytest 있음, torch 없음 — 스텁 필수)
- 대상: `submit_candidates/merge080/script.py` (커밋됨 — 워크트리에 있음. encoder_predict의 align_by_label 부분, validate_mbert, main의 mbert_mix 블록)
- 참조 패턴: `tests/test_enc_block_weights.py` (torch 스텁 + features 캐시 격리 + spec loader)
- merge080의 model/ 바이너리는 워크트리에 없다(gitignore) — 테스트는 tmp_path 가짜 config/파일로만 구성할 것

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출/수동 zip 금지
- `submit_candidates/merge080/script.py`의 **동작 변경 금지** — 테스트를 위해 불가피한 최소 리팩토링(함수 추출)만 허용하고, 그 경우 리팩토링 전후 diff가 대수적으로 동일함을 리포트에 명시
- submit/ 수정 금지, GPU 금지

## 진행 프로토콜 (재개 대비)

1. 시작 시 `context/night/2026-07-08/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(스텁 로더 구성 / 테스트 그룹별 / 스위트 통과)마다 PROGRESS 갱신 + **git commit**
3. 끝나면 `task1.DONE` + 최종 커밋

## 작업 내용

1. merge080/script.py를 모듈 스텁으로 로드하는 헬퍼 구축 (torch·transformers·joblib 스텁, src 패키지는 실제 것 사용 — merge080/src는 커밋돼 있어 워크트리에 있음, constants.ACTIONS가 스펙 순서인 것 활용).
2. 위 DoD 1의 테스트 케이스 작성 — 각 케이스는 독립적으로 tmp_path에 가짜 model/mbert_full(config.json만) 구성.
3. 전체 스위트 실행으로 기존 테스트와의 오염 없음 확인 (어제 features 모듈 캐시 오염 전례 — sys.modules 정리 철저히).
