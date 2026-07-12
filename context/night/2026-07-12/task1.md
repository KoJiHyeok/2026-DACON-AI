# task1 — CX-004: KD 증류 도구 세트 (teacher prob-dump + student trainer + 검증 테스트)

> ⚠️ **SUPERSEDED (07-12 저녁)**: 이 티켓은 무효 — CX-004는 `scripts/kd_run/`으로 당일 대체 구현·제출까지 완료(exp #51), CX-005는 exp #50 축 종결로 불필요. 러너 철수됨. 기록 보존용으로만 남김.

## 컨텍스트

DACON 236694 (AI 에이전트 다음 행동 14클래스, Macro-F1). D-014로 공세 재개 — 시드 앙상블(exp #48: avg(s42,s43) 블렌드 row +0.00045, 유일한 로컬 양수 후보)은 e5 인코더 2개가 submit.zip 1GB 제한을 초과해 **증류(KD)만이 유일한 배포 경로**다. 서버에서 s43·s44 full-70k teacher가 오늘 학습 중이고(s42 = 배포 챔피언 인코더, `submit/model/encoder`에 실존), 내일 아침 Claude가 GPU로 실행할 **증류 학습 도구**가 이 작업의 산출물이다. GPU 실행·판정·제출은 전부 Claude 담당 — 이 작업은 코드와 CPU 검증만.

## 목표 / 완료 조건 (DoD)

1. `scripts/distill/dump_teacher_probs.py` — 저장된 e5 fp16 모델 디렉토리(HF from_pretrained 호환)를 로드해 지정 행들의 14클래스 확률을 npz(ids, probs, actions)로 저장. env 기반 설정(아래), required argparse 금지(env 폴백 + parse_known_args).
2. `scripts/distill/train_student.py` — N개 teacher npz의 확률 평균(soft label)과 hard label을 결합한 KD 손실로 e5-base student를 챔피언 레시피(hist12/6ep/b16/lr2e-5/maxlen384)로 학습. `DIS_MODE=holdout85|full` 지원 — holdout85면 15% 홀드아웃 npz 표면 산출(리그 판정용), full이면 fp16 모델 저장(배포용).
3. KD 손실 = `(1-λ)·CE(y_hard) + λ·T²·KL(softmax(z_s/T) ‖ p_teacher^(T))` — λ=`DIS_LAMBDA`(기본 0.5), T=`DIS_T`(기본 2.0). teacher 확률의 온도 재적용은 log-공간 근사(`p^(1/T)` 재정규화)로 구현하고 docstring에 수식 명기.
4. **직렬화 계약**: student 입력 직렬화는 챔피언과 바이트 동일해야 한다. `colab/encoder_e5_holdout85_maxhist.py`(수정 금지)의 직렬화 함수를 import 또는 복사하되, 복사 시 원본 대비 바이트 동일성을 검증하는 테스트를 포함(샘플 20행 직렬화 결과를 원본 모듈 호출과 대조).
5. `tests/test_distill.py` — 전부 로컬 CPU(`C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`, **torch 없음**)에서 통과해야 한다:
   - KD 손실 수식의 **numpy 참조 구현** 대조: λ=0 → CE와 일치, λ=1·T=1 → 순수 KL, teacher 평균 계산 정확성
   - teacher npz 정렬 검증 로직: ids 불일치·행 순서 셔플·확률 비정규화 입력에서 명시적 에러
   - env 설정 파싱·모드 분기 (torch import는 함수 내부 lazy import로 격리 — 테스트가 torch 없이 돌게)
   - 직렬화 바이트 동일성 (위 4)
6. `context/night/2026-07-12/report_distill.md` — 사용법(서버 실행 커맨드 예시 포함), 설계 결정, 알려진 한계. **파일명이 task로 시작하면 안 됨(러너 glob 함정).**
7. `context/night/2026-07-12/task1.DONE` 생성 (결과 요약 포함) — 러너 완료 판정 기준.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 서버 미러 — 시스템 파이썬 금지, **torch 미설치임을 전제로 테스트 설계**)
- 챔피언 인코더(구조·토크나이저 참조): `C:\dev\2026-AI-DACON\submit\model\encoder\` (읽기 전용 — config.json, tokenizer.json, serialize_config.json)
- 직렬화 원본: 워크트리 내 `colab/encoder_e5_holdout85_maxhist.py` (수정 금지, import/복사만)
- 홀드아웃 분할 기준: `C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_split_meta.json` (세션 프리픽스 15% — holdout85 모드는 이 분할을 그대로 사용, 새 분할 생성 금지)
- env 설계 참고: `colab/encoder_e5_holdout85_maxhist.py`의 ENC_* 패턴 (DIS_*로 통일: DIS_MODE, DIS_TEACHER_NPZ=콤마 목록, DIS_LAMBDA, DIS_T, DIS_DATA_DIR, DIS_OUT, DIS_SEED 기본 42)

## 금지

- 워크트리 밖(메인 리포 작업트리·팀 리포) 수정 금지, `git push` 금지, 수동 zip·제출 금지, 제출물에 네트워크 코드 금지
- `colab/**`·`submit/**`·canonical context(`context/experiments.md` 등) 수정 금지
- 폐기 목록(experiments.md 재시도 금지 테이블) 재시도 금지 — 특히 이 작업은 "3-seed 프로브"가 아니라 **증류 도구 제작**이다. 판정·점수 주장 금지(promotion 판단은 Claude)
- 메인 `.venv`에 패키지 설치 금지. torch 실행 테스트가 꼭 필요하면 워크트리 내부에 별도 venv를 만들고 report에 기록(선택 사항 — DoD 아님)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-12/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터 이어서
2. 의미 단위 작업마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit** (커밋이 안 되면 PROGRESS에 사유 기록 — 아침에 Claude가 회수 커밋)
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋

## 작업 내용 (단계)

1. PROGRESS 파일 생성, 재료 존재 확인 (특히 submit/model/encoder의 config·tokenizer)
2. `scripts/distill/common.py` — env 파싱, npz 정렬 검증(ids↔probs 동일 순회 저장 계약), holdout 분할 로드
3. `dump_teacher_probs.py` (torch lazy import) — 배치 추론, fp16→fp32 캐스팅 명시, DIS_ROWS=train85|full 행 선택
4. `train_student.py` (torch lazy import) — KD Trainer(custom loss), holdout85/full 두 모드, run json에 패키지 버전·env 전체 기록
5. `tests/test_distill.py` — DoD 5의 전 항목. numpy 참조 구현은 테스트 파일 안에 독립 작성(구현 코드 재사용 금지 — 대조 의미 소멸)
6. report_distill.md + task1.DONE + 최종 커밋
