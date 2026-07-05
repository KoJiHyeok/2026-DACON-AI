# task2 — mdeberta-v3-base Colab 학습 팩 (아침 즉시 가동용, CPU 검증 전용)

## 컨텍스트

DACON 236694: 14클래스 분류, Macro-F1. 현 인코더 성분은 multilingual-e5-base 단일 가족 — e5-small 추가는 전 가중에서 마이너스로 폐기됐다(같은 백본 가족이라 다양성 없음). **이질 백본**(mdeberta-v3-base, 멀티링구얼 — 데이터에 language_pref 필드 존재)이 남은 최우선 성분 후보다.

이 태스크는 GPU 학습이 아니라 **아침에 사용자가 Colab에 붙여넣기만 하면 도는 완성 스크립트**를 만들고 CPU로 파이프라인을 검증하는 것이다.

## 목표 / 완료 조건 (DoD)

1. `colab/mdeberta_finetune.py` 완성 — 아래 '스크립트 요구사항' 전부 충족
2. **CPU 스모크 PASS**: `SMOKE=1`로 ≤200행·tiny 모델·1 epoch가 로컬 CPU에서 end-to-end 완주 (직렬화→토크나이즈→학습 루프→저장→holdout npz 생성까지). 스모크 로그를 리포트에 첨부
3. 리포트 `context/night/2026-07-06/task2_report.md`: 사용법(Colab 붙여넣기 순서), 예상 학습 시간 추정(근거 포함), T4 함정 회피 목록
4. `context/night/2026-07-06/task2.DONE` 생성 (요약 포함) + 최종 커밋

## 스크립트 요구사항 (전부 필수)

- **붙여넣기 실행 규약** (`context/decisions.md` D-00x, 재발 방지 기록): required argparse 금지. 모든 설정은 env 폴백 — `MDEB_MODEL`(기본 `microsoft/mdeberta-v3-base`), `MDEB_DATA_DIR`, `MDEB_OUT`, `MDEB_SEED`(42), `MDEB_EPOCHS`(2), `MDEB_MODE`(`full`|`holdout85`), `SMOKE`. argparse를 쓰려면 `parse_known_args` + 전부 optional.
- **직렬화 계약**: `submit/script.py`의 `serialize()`(max_hist=6, v2)를 **verbatim 복사** — 학습·추론 직렬화 동일 계약. 임의 변경 금지 (serialize 확장은 재시도 금지 테이블에서 폐기됨).
- **CV/split**: `MDEB_MODE=holdout85`면 세션 프리픽스 기준 85/15 split(시드 42) — `context/night/2026-07-05/holdout_base.npz`의 ids와 **같은 holdout 행**이 나오도록, 그 npz의 ids를 직접 읽어 valid 행을 지정하는 방식을 권장(재현 확실). `full`이면 70k 전체 학습.
- **T4 함정 회피 (필수 내장)**:
  - DeBERTa-v3는 **fp16 학습에서 NaN/overflow가 잦다** (disentangled attention 스케일 이슈, 커뮤니티 다수 보고). T4는 bf16 미지원 → **학습은 fp32 고정** (`fp16=False`). 대신 batch 8~16 + gradient_accumulation으로 메모리 대응, `max_len=384`.
  - 저장은 학습 후 `model.half().save_pretrained()` fp16 변환본 별도 저장 (추론은 fp16 안전) — 제출 zip 1GB 제약 대비 크기 로그 출력.
  - loss가 NaN이 되면 즉시 중단하고 마지막 정상 체크포인트 보존 + 로그에 명시하는 가드 포함.
- **출력**: (a) 모델 디렉토리 (config에 id2label = 14클래스 알파벳순), (b) holdout85 모드면 `holdout_mdeb.npz` (ids/probs/y_true/actions — `holdout_base.npz`와 동일 스키마, 리그 조인용), (c) 진행 로그에 epoch별 valid macro-F1.
- **중간 저장**: epoch마다 checkpoint 저장 (Colab 끊김 대비), `MDEB_RESUME=1`이면 마지막 checkpoint에서 재개.

## 재료 (절대 경로)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` + `data\train_labels.csv` (읽기 전용)
- serialize 원본: `C:\dev\2026-AI-DACON\submit\script.py` (읽기 전용 — serialize()만 복사)
- 참고 구현: `C:\dev\2026-AI-DACON\colab\holdout_eval.py`, `colab\encoder_v3_repro.py` (env 폴백 패턴·npz 스키마 참고)
- **CPU 스모크용 파이썬**: `C:\dev\Second-Brain-Project\Hoseo\ai-2026\.venv\Scripts\python.exe` (torch 2.12.1+cpu, transformers 5.12.1 — 이 인터프리터에만 torch가 있다)
- 스모크용 tiny 모델: HF에서 `hf-internal-testing/tiny-random-DebertaV2ForSequenceClassification` 다운로드 시도. 네트워크가 없으면 `DebertaV2Config(hidden_size=32, num_hidden_layers=2, ...)` 랜덤 초기화 + 토크나이저는 문자 단위 더미로 대체하고 리포트에 "토크나이저 실물 검증은 Colab에서" 명기.

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출물(submit/) 접촉 금지
- GPU 학습 시도 금지 (이 머신에 GPU 없음 — 스모크는 CPU tiny만)
- serialize() 변경 금지 (verbatim 복사만)

## 진행 프로토콜 (재개 대비)

1. 시작하자마자 `context/night/2026-07-06/PROGRESS-task2.md` 확인 — 있으면 이어서
2. 의미 단위(스크립트 골격 → 스모크 통과 → 리포트)마다 PROGRESS 갱신 + git commit
3. 끝나면 `task2.DONE` + 최종 커밋

## 작업 내용

1. `colab/holdout_eval.py`를 베이스로 mdeberta용 학습 스크립트 작성 (위 요구사항 반영).
2. tiny 모델로 CPU 스모크: `SMOKE=1 MDEB_MODE=holdout85`, 200행, 1 epoch — npz 스키마까지 검증 (ids/probs shape/actions 알파벳순 assert).
3. 예상 시간 추정: mdeberta-base fp32, T4, 70k×2epoch, max_len 384, batch 8×accum 2 기준 — 토큰 수 실측(e5 토크나이저 실측 chars/token≈2.43 참고, mdeberta는 별도 실측)으로 step 수 계산해 근거 있는 추정치 제시. 12시간 초과 추정이면 epoch/max_len 절충안을 리포트에 제시.
4. 리포트 + DONE.
