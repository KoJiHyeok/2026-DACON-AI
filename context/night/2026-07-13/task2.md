# task2 — 재현 리허설 하네스 (7/20 코드 제출 + 본선 모델검증 40% 대비)

## 컨텍스트

DACON 236694. 상위 12팀은 **7/20(월) 10:00까지 재현 코드 제출**, 본선 심사에 모델검증 10%가 들어간다. 챔피언(LB 0.7623)은 5성분 — linear·AAR·e5-hist12·mBERT-h6·AU(soft α0.9) — 인데 학습 코드가 리포 곳곳(scripts/linear2, scripts/aar_rebuild, colab/, scripts/league4)에 흩어져 있고, "무엇을 어떤 순서로 돌리면 제출물이 재생되는가"의 단일 문서·검증기가 없다. 이 작업은 그 **재현 플레이북 + 자동 검증기**를 만든다. 컷 갭상 조건부 보험이지만 본선 발표의 모델검증 서사 재료로도 쓰인다.

## 목표 / 완료 조건 (DoD)

1. `docs/repro_playbook.md` — 챔피언 5성분 각각에 대해: ① 정확한 학습 커맨드(env 포함, GPU 성분은 서버/Colab 커맨드) ② 환경(패키지 버전 — 실측 기록이 있는 run json들 인용) ③ 기대 산출물(파일명·크기·해시가 기록된 곳) ④ 기대 지표와 허용 오차(예: AAR OOF 0.7034, linear OOF 0.6639, e5 holdout 0.73617 등 — context/ 기록에서 인용) ⑤ 소요 시간. **실측으로 검증한 항목과 문서로만 검증한 항목을 표로 명확히 구분**한다.
2. **CPU 성분 실측 재현 (필수 2종)**: 이 밤에 실제로 재실행해 수치 대조 —
   - AAR: `scripts/aar_rebuild/train_aar.py` (약 14분) → 3-fold OOF 0.7034 ± 0.005 재현
   - linear: `scripts/linear2/baseline_repro.py` 경로 (기존 검증 exp #32 커맨드) → OOF 0.6639 ± 0.005 재현
   - GPU 성분(e5·mBERT)·AU는 커맨드 실검증 없이 문서+기존 run json 근거로 정리 (실행 금지 — GPU 없음)
3. `scripts/repro_rehearsal/verify.py` — 플레이북의 기대 산출물·지표를 기계 검증: 지정 경로 파일 존재/해시 대조(기록이 있는 것), 재현 런 로그에서 지표 파싱해 허용 오차 판정, 결과 JSON 출력. `--component aar` 처럼 성분별 실행 가능, required argparse 금지(전부 기본값 동작).
4. `tests/test_repro_rehearsal.py` — verify.py의 파싱·판정 로직 소형 고정 입력 테스트, `.venv` CPU 통과.
5. `context/night/2026-07-13/report_repro.md` — 실측 2종 결과표, 5성분 커버리지 표(실측/문서), 갭 목록(재현 불가능하거나 기록 부재인 것 — 있으면 그것이 이 작업의 최대 가치다). **파일명 task 시작 금지.**
6. `context/night/2026-07-13/task2.DONE` 생성 (요약 포함).

## 재료 (절대 경로)

- 데이터: `C:\dev\2026-AI-DACON\data\` (읽기 전용) / 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`
- 기록 원천 (워크트리 내 추적 파일): `context/experiments.md`(#32·#34·#51 등), `context/coordination.md`(07-11 재현성 감사 노트·Handoff 해시들), `scripts/aar_rebuild/REPRODUCTION.md`, `colab/encoder_e5_holdout85_maxhist.py`·`colab/mdeberta_finetune.py` 헤더, `scripts/kd_run/` run json 패턴
- 실물 대조(절대 경로 읽기 전용, gitignore): `C:\dev\2026-AI-DACON\submit\model\**`, `C:\dev\2026-AI-DACON\artifacts\**`
- 재현 런 출력: 워크트리 안 `out_repro/` (gitignore 경로면 커밋 불필요 — 지표는 report에 전사)

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출 금지, 네트워크 코드 금지, GPU 학습 금지
- `submit/**`·`scripts/aar_rebuild/**`·`scripts/linear2/**` 수정 금지 (실행·인용만) — 신규 코드는 `scripts/repro_rehearsal/`에만
- **task1과 경로 격리**: `scripts/aar_speed/**`·`tests/test_aar_speed.py`·`report_aar_speed.md` 건드리지 말 것

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-13/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신 + **git commit** (실패 시 사유 기록)
3. 전부 끝나면 `task2.DONE` + 최종 커밋
4. 장시간 학습(AAR 14분·linear)은 로그를 파일로 남기고, 중단 재개 시 완료된 런은 로그 파싱으로 재사용 (재학습 반복 금지)

## 작업 내용 (단계)

1. PROGRESS 생성 → 기록 원천 통독 → 5성분 × (커맨드/env/산출물/지표/시간) 매트릭스 초안
2. AAR 재현 런 실행 (백그라운드 + 로그) → 그동안 linear 재현 커맨드 확정 → linear 런
3. verify.py + tests 구현 (런 로그·기존 기록 기반)
4. repro_playbook.md 완성 (실측/문서 구분 표 포함) → report_repro.md → task2.DONE
