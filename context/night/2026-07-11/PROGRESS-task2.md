# task2 진행 기록

- [x] `submit/script.py` 추론 단계와 모델 경로를 읽고 계측 경계를 확정했다.
- [x] `profile_infer.py`/`stages.py`에 env 폴백, 3회 중앙값, 단계별 JSON 출력을 구현했다.
- [x] `equivalence_check.py`에 별도 `script.py` subprocess 기준 비교와 불일치 id 덤프를 구현했다.
- [x] CPU 측정 명령을 시도했다.
- [ ] CPU/GPU 실측 및 pytest 완료 — 지정 env 모두 torch 미설치로 실행 차단.
- [x] report-only 리포트 작성.

다음 재개 지점: torch가 설치된 오프라인 환경에서 30행 스모크와 등가성 검사를 실행하고,
이후 300행·3회 CPU 실측 및 reviewer/tester 검증을 수행한다.
