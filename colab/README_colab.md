# 인코더 재학습 (Colab) — w112 앙상블 재건 + 4번째 성분 + blend 탐색용 holdout

팀 앙상블(w112, LB 0.7208)에서 가장 강한 성분인 `intfloat/multilingual-e5-base` 파인튜닝
가중치(fp32 ~1.1GB)가 git에 없고 팀원 로컬에만 있어 우리 쪽에서 입수 불가능하다. 이 폴더는
로컬 GPU 없이 Google Colab(T4)에서 관련 인코더들을 재학습·평가하기 위한 자료다.

## 작업 우선순위 (job1 → job2 → job3)

| job | 스크립트 | 목적 | 필수 여부 |
|---|---|---|---|
| **job1** | `encoder_v2_s42_repro.py` | w112 재건 — 없으면 팀 최고 앙상블(LB 0.7208) 자체가 성립 안 함 | **최우선, 필수** |
| job2 | `encoder_small_repro.py` | e5-small 4번째 성분 추가(오류 이질성, 이득은 미검증) | 선택 — Colab 세션 여유가 있을 때 |
| job3 | `holdout_eval.py` | base/small의 holdout 확률을 뽑아 로컬 CPU에서 blend weight 그리드 탐색(LB 소모 없이) | 선택 — job1/job2 이후 |

job1만 끝나도 w112를 재건해 제출할 수 있다. job2·job3는 그 위에 얹는 실험이니, Colab
세션을 더 돌릴 여유가 있을 때만 진행해도 된다.

핵심 스펙(job1, 변경 금지): `intfloat/multilingual-e5-base` (XLM-RoBERTa 기반, 14클래스
head), train.jsonl **전량 7만 행** + train_labels.csv만 사용(홀드아웃 없음), **seed 42**,
6 epochs, label_smoothing 0.1, max_len 384. 텍스트 직렬화는 팀 리포
`ensemble/script_3way.py`의 `serialize()`와 완전히 동일해야 한다 — 이 계약이 한 글자라도
어긋나면 학습 때 보던 입력과 추론 때 보는 입력이 달라져 **조용한 오답**이 나온다.

## 0. serialize() 대조 결과

이 폴더의 세 스크립트(`encoder_v2_s42_repro.py`, `encoder_small_repro.py`,
`holdout_eval.py`)에 들어간 `serialize()`/`_bucket()`을 AST 단위(주석·docstring 제외,
실제 실행되는 코드만)로 다음 팀 리포 소스와 비교했다 — **전부 코드 동일**:

- `dacon-agent-action-api-boost/colab/encoder_finetune.py` (v1)
- `dacon-agent-action-api-boost/colab/encoder_v2_full.py` (v2, job1의 원본)
- `dacon-agent-action-api-boost/ensemble/script_3way.py` (제출 추론 스크립트 — 정본)
- `dacon-agent-action-api-boost/ensemble/soup_encoders.py`

차이는 오직 docstring 문구와 for문 옆 인라인 주석 하나뿐이며 둘 다 실행에 영향 없다.
14개 `ACTIONS` 클래스 목록·순서도 전부 동일 확인.
**주의**: 같은 팀 리포의 `src/serialize.py`는 이름이 비슷하지만 전혀 다른(더 정교한) 직렬화
모듈이며 `train.py`/`train_tscar.py`라는 별개 트랙에서만 쓰인다 — 인코더 학습과는 무관하니
혼동하지 말 것.

## 1. Colab 런타임 설정 (job1/2/3 공통)

1. https://colab.research.google.com 에서 새 노트북 생성.
2. 상단 메뉴 `런타임 → 런타임 유형 변경 → 하드웨어 가속기: T4 GPU` 선택 후 저장.
3. 무료 티어는 세션이 끊길 수 있다 — job1/job2 학습 스크립트는 `save_strategy="epoch"`,
   `save_total_limit=1`이라 마지막 epoch 체크포인트(`OUT_DIR/ckpt`)에서 재개할 수 있지만,
   재개 로직을 자동화하지 않았으므로 끊기면 수동으로 `Trainer`를 체크포인트에서
   재시작해야 한다(팀 원본 스크립트들도 동일한 한계). job3(`holdout_eval.py`)는
   `save_strategy="no"`라 재개 대상 자체가 없다 — 끊기면 처음부터 다시 돌린다(짧은 job이라
   감내 가능한 수준).

## 1.5 멀티 계정 병렬 실행 (job을 동시에 여러 개 돌릴 때 — 필독)

병렬 세션은 **계정마다 Colab GPU 할당이 따로**라서 보조 Google 계정 여러 개로 돌리는 게 정석이다.
단, 함정이 하나 있다 (2026-07-05 실제로 겪은 오류):

> **보조 계정 세션에서 `drive.mount`는 그 계정의 MyDrive만 마운트한다.**
> 메인 계정이 소유한 `dacon2026/` 폴더는 보조 계정에선 '공유 문서함(Shared with me)'에
> 있고, 공유 문서함은 `/content/drive/MyDrive/`에 **나타나지 않는다** →
> `FileNotFoundError: /content/drive/MyDrive/dacon2026/train.jsonl`.

해결 (보조 계정마다 둘 중 하나, A 권장):

- **(A) 바로가기 추가 — 계정당 1회 설정**: 보조 계정으로 Drive 웹 접속 → 공유 문서함에서
  `dacon2026` 우클릭 → `정리(Organize)` → `바로가기 추가(Add shortcut)` → `내 드라이브` 선택.
  이후 Colab 마운트에서 `/content/drive/MyDrive/dacon2026/...` 경로가 그대로 동작한다
  (바로가기는 FUSE 마운트에서 실제 폴더처럼 따라감). 산출물도 이 경로에 쓰면 실제로는
  공유 폴더 원본에 저장되므로 메인 계정에서 전부 보인다.
- **(B) 마운트 계정 선택**: `drive.mount('/content/drive')` 인증 팝업에서 런타임 계정이 아닌
  **폴더 소유 계정을 선택**해 마운트. 설정이 필요 없지만 매 세션 인증 때 계정을 잘 골라야 한다.

병렬 운영 팁: 세션(계정)마다 `ENC_SEED` 환경변수나 `--seed`를 다르게 주면 산출 폴더
(`enc_v2_s42`, `enc_v2_s7`, ...)가 겹치지 않는다. 세 학습 스크립트 모두 시작 시
`check_drive_root()`가 경로를 검사해서, 위 상황이면 현재 MyDrive 목록과 해결법을 에러
메시지에 바로 띄워준다.

## 2. 업로드할 파일 (job1/2/3 공통)

Google Drive에 `dacon2026/` 폴더를 만들고 아래 두 파일을 업로드한다(Drive 마운트 방식
권장 — Colab 업로드 위젯은 대용량 파일에서 끊기기 쉽다). 세 job이 이 데이터를 공유한다.

| 파일 | 크기 | 로컬 위치(참고) |
|---|---:|---|
| `train.jsonl` | 약 103MB (98.5MiB) | `C:\dev\2026-AI-DACON\data\train.jsonl` |
| `train_labels.csv` | 약 3.1MB | `C:\dev\2026-AI-DACON\data\train_labels.csv` |

업로드 후 Drive 구조가 `내 드라이브/dacon2026/train.jsonl` + `.../train_labels.csv`가
되도록 한다(스크립트들의 `DRIVE_ROOT`/`--data-dir` 기본값과 일치).

## 3. job1 — e5-base v2 s42 (w112 재건, 최우선)

Colab 셀을 아래 순서로 실행한다. `encoder_v2_s42_repro.py`는 `# %%` 주석으로 셀이
구분되어 있으니 그대로 잘라 붙이면 된다.

1. **셀 1 — 의존성 설치** (약 1~2분):
   ```
   !pip install -q "transformers>=4.51" accelerate
   ```
2. **셀 2 — Drive 마운트** (약 10~30초):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **셀 3 — 학습 본체**: `encoder_v2_s42_repro.py` 전체 내용을 붙여넣고 실행.
   - 예상 소요 시간: **약 2.5시간/런** (원본 `encoder_v2_full.py` 작성자의 T4 실측
     기준 주석). 70,000행 × 6 epochs × max_len 384 × batch 16(e5-base, fp16 mixed
     precision)라는 규모를 감안하면 Colab 무료 T4의 세션별 변동(다른 사용자와의
     공유, 스로틀링)까지 포함해 **현실적으로 2~4시간 범위**로 잡는 것이 안전하다.
4. 학습이 끝나면 `/content/drive/MyDrive/dacon2026/enc_v2_s42/model/` 과
   `.../enc_v2_s42/run.json`이 생성된다. 산출물 목록·검증은 4절 참고.

## 4. job1 산출물 목록과 검증

`OUT_DIR/model/`(예: `enc_v2_s42/model/`)에 다음이 생성되어야 한다:

- `config.json` — `num_labels: 14`, `id2label`/`label2id`에 14개 액션 클래스 전부 포함.
- `model.safetensors` — fp32, 약 **1.1GB** (XLM-RoBERTa-base 기반 ~278M 파라미터 × 4byte).
- 토크나이저 파일 일체 (`tokenizer.json`, `tokenizer_config.json`,
  `special_tokens_map.json`, `sentencepiece.bpe.model` 등 — 정확한 파일 목록은 실행된
  transformers 버전이 자동으로 결정하므로 개수보다는 "로드가 되는지"로 검증).
- `OUT_DIR/run.json` (model/의 형제 디렉터리) — `{"model": ..., "seed": 42, "epochs": 6,
  "max_len": 384, "label_smoothing": 0.1, "n_train": 70000, "full_train": true}`.

검증 방법(Colab 셀에서):
```python
import json, os
model_dir = "/content/drive/MyDrive/dacon2026/enc_v2_s42/model"
cfg = json.load(open(os.path.join(model_dir, "config.json")))
assert cfg["num_labels"] == 14, cfg["num_labels"]
assert len(cfg["id2label"]) == 14
size_mb = os.path.getsize(os.path.join(model_dir, "model.safetensors")) / 1e6
print(f"num_labels OK, model.safetensors = {size_mb:.0f} MB")  # ~1100 근처면 정상(fp32)
```

## 5. fp16 변환 후 다운로드 (job1, job2 공통)

다운로드 용량을 절반으로 줄이기 위해 fp32 → fp16 변환을 거친다(`to_fp16.py`, 이 폴더).
같은 Colab 런타임에서(다운로드 전에) 실행하는 것을 권장한다 — GPU가 없어도 동작한다.
job1(base)과 job2(small) 모두 같은 스크립트를 `--src`/`--dst`만 바꿔서 쓴다:

```python
# job1 (base)
!python /content/drive/MyDrive/dacon2026/to_fp16.py \
    --src /content/drive/MyDrive/dacon2026/enc_v2_s42/model \
    --dst /content/drive/MyDrive/dacon2026/enc_v2_s42/model_fp16

# job2 (small) — job2를 진행했다면
!python /content/drive/MyDrive/dacon2026/to_fp16.py \
    --src /content/drive/MyDrive/dacon2026/enc_small_s42/model \
    --dst /content/drive/MyDrive/dacon2026/enc_small_s42/model_fp16
```

(스크립트 파일 자체를 Drive에 올려두거나, Colab 셀에 내용을 붙여넣고 `%run`으로 실행해도
된다.) 성공하면 fp32 `model.safetensors` 옆에 fp16 폴더가 생긴다 — base는 ~1.1GB→**~550MB**,
small은 ~470MB(118M 파라미터 fp32 추정)→**~235MB**. 스크립트가 자체적으로
dtype(`float16`)과 `num_labels==14`를 재확인하므로, 콘솔에
`OK — dtype=float16 확인, num_labels=14 확인`이 뜨는지 확인한다.

변환이 끝나면 fp16 폴더 전체를 로컬로 다운로드한다. Colab 파일 브라우저에서 우클릭 →
다운로드는 폴더 단위를 지원하지 않으므로, 아래처럼 zip으로 묶은 뒤 다운로드하는 편이 안전하다:

```python
import shutil
shutil.make_archive("/content/enc_v2_s42_fp16", "zip",
                    "/content/drive/MyDrive/dacon2026/enc_v2_s42/model_fp16")
# job2도 진행했다면:
# shutil.make_archive("/content/enc_small_s42_fp16", "zip",
#                     "/content/drive/MyDrive/dacon2026/enc_small_s42/model_fp16")
# 좌측 파일 탐색기에서 zip 파일을 다운로드
```

## 6. 로컬 배치 위치

다운로드한 zip을 풀어서 아래 경로에 배치한다(우리 리포 기준):

```
C:\dev\2026-AI-DACON\artifacts\enc_v2_s42\model\      ← job1 (base, ~550MB fp16)
├── config.json
├── model.safetensors
├── tokenizer.json / tokenizer_config.json / special_tokens_map.json / (기타 토크나이저 파일)

C:\dev\2026-AI-DACON\artifacts\enc_small_s42\model\   ← job2 (small, ~235MB fp16, 진행했다면)
└── (구성 동일)
```

`artifacts/` 디렉터리는 아직 리포에 없으므로 직접 생성한다
(`mkdir -p artifacts/enc_v2_s42/model` 또는 PowerShell
`New-Item -ItemType Directory -Force artifacts/enc_v2_s42/model`).
`.gitignore`에 이미 `*.safetensors`가 등록되어 있어 무거운 가중치 파일(수백MB)은 자동으로
커밋에서 제외된다 — `config.json`/토크나이저 파일 등 작은 텍스트 파일만 추적된다.

앙상블 패키징 시(팀 `ensemble/package_ensemble.py` 패턴 참고) job1은 `model/encoder`,
job2는 `model/encoder_2`로 넣으면 `ensemble/script_3way.py`의 `encoder_dirs()`가 이름순으로
모아 uniform 평균을 낸다 — 코드 수정 없이 4-way가 된다. zip 예산은
`encoder_small_repro.py`의 docstring에 계산해뒀다(현재 w112 ~634MB + e5-small fp16
~235MB ≈ 869MB, 1024MB 한도 대비 여유 ~155MB, 팀 보고 수치 기준).

## 7. job2 — e5-small (4번째 성분, 선택)

job1과 같은 방식(셀 1/셀 2 공유, 이미 실행했다면 재실행 불필요)으로 셀 3에
`encoder_small_repro.py` 전체 내용을 붙여넣고 실행한다.

- 예상 소요 시간: **약 2~3.5시간**. 팀 README의 v1(e5-small, 85% 데이터=약 5.95만 행,
  3 epochs, batch 32) 실측/추정 1.5~3시간에, 데이터가 100%로 늘어난 비율(7만/5.95만
  ≈ 1.18배)만 곱한 추정치다 — e5-small 자체의 full-70k 실측 학습곡선은 없으므로
  (base처럼 "피크 ep6"를 실측한 적이 없음) epoch 수는 v1의 3을 그대로 유지했다.
- 완료 후 `/content/drive/MyDrive/dacon2026/enc_small_s42/model/` +
  `.../enc_small_s42/run.json` 생성. 검증은 4절과 동일한 방식(단 `model.safetensors`는
  fp32 기준 ~470MB 근처, e5-small ~118M 파라미터 × 4byte).
- ⚠️ `ensemble/soup_encoders.py`의 가중치 평균(model soup)과 혼동 금지 — 팀 보고 기준
  seed-soup은 LB 0.697로 폐기 확정됐다. 이건 soup가 아니라 **확률 평균용 별도 모델**이다.

## 8. job3 — holdout_eval.py (blend weight 탐색용, 선택)

base/small 어느 모델이든 세션-프리픽스 StratifiedGroup 85/15 holdout으로 빠르게 학습해
holdout 확률을 npz로 뽑는다. LB를 소모하지 않고 로컬 CPU에서 linear/stacker 확률과
blend weight 그리드 탐색을 하기 위한 용도다 — **이 npz의 모델은 제출용이 아니다**(85%만
학습, label_smoothing 없음).

```python
# base 인코더 holdout 확률
!python /content/drive/MyDrive/dacon2026/holdout_eval.py \
    --model base --out /content/drive/MyDrive/dacon2026/holdout_base.npz

# small 인코더 holdout 확률 (job2를 하지 않았어도 독립적으로 실행 가능 — 사전학습 체크포인트에서 새로 파인튜닝함)
!python /content/drive/MyDrive/dacon2026/holdout_eval.py \
    --model small --out /content/drive/MyDrive/dacon2026/holdout_small.npz
```

- split: `StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42)`의 첫 fold를
  holdout으로 사용(1/7 ≈ 14.3%, 목표 15%에 근접), group=세션 프리픽스(정규식
  `-step_\d+$` 제거), stratify=action 라벨 — 팀 리포 `train_tscar.py`의 기존 홀드아웃
  패턴(`n_holdout_splits = round(1/valid_size)` → `StratifiedGroupKFold`)과 동일 원칙.
  **로컬에서 실제 train.jsonl(7만행)로 이 분할을 미리 검증**: 세션 누수 0건, holdout
  10,001행(14.3%), 14개 클래스 전부 train/holdout 양쪽에 충분히 존재(최소 클래스
  web_search도 holdout 182건) — 정상 동작 확인.
- 예상 소요 시간: `--model base`는 약 **1~1.5시간**(85% 데이터·3 epochs·batch16을 job1의
  2.5h/6epoch/batch16 실측치에서 row-epoch 비율로 환산한 추정), `--model small`은
  **약 1.5~3시간**(사실상 팀 v1 스크립트와 거의 같은 설정이라 팀의 원래 실측/추정치와 동일).
- 산출: `ids`(str), `probs`(N×14, `ACTIONS` 알파벳순), `y_true`(str) 를 담은 npz. 다운로드해서
  로컬 CPU에서 `np.load(...)`로 불러와 linear/stacker 확률과 정렬 후 가중 평균 그리드 탐색에
  사용하면 된다(그리드 탐색 코드 자체는 이 배포물 범위 밖 — npz 생성까지가 job3의 역할).

## 재현성에 대한 정직한 한 줄

`SEED=42`로 `random`/`numpy`/`torch`/`TrainingArguments(seed=...)`를 전부 고정하지만,
GPU 비결정적 커널(cuDNN 알고리즘 선택)과 `fp16=True` 혼합정밀도의 연산 순서 비결합성,
그리고 Colab이 세션마다 배정하는 T4 개별 개체·드라이버 버전 차이 때문에 **비트 단위로
동일한 가중치가 나온다는 보장은 없다** — 다만 같은 하이퍼파라미터·데이터·seed이므로
성능(macro-F1, 클래스별 분포)은 원본 성분과 사실상 동등한 수준으로 재현될 것으로
기대할 수 있다는 정도로 이해하면 된다. job2(e5-small)는 애초에 원본 성분 자체가
존재하지 않는 새 모델이라 "재현"이 아니라 "동일 레시피 신규 학습"이며, job3은 제출용이
아닌 probe이므로 이 재현성 논의가 적용될 필요조차 없다.

## 파일 목록

- `encoder_v2_s42_repro.py` — job1: e5-base 학습 본체 (원본 `encoder_v2_full.py`와
  하이퍼파라미터·`serialize()` 동일, 경로 상수 정리 + `ENC_SEED` 환경변수로 시드 파라미터화).
- `encoder_small_repro.py` — job2: e5-small 학습 본체 (원본 `encoder_finetune.py` 기반,
  holdout 제거 + full-70k로 조정, 하이퍼파라미터는 원본 그대로 유지).
- `holdout_eval.py` — job3: base/small 공용 holdout 확률 추출 (StratifiedGroupKFold
  85/15, npz 출력).
- `to_fp16.py` — job1/job2 공용 fp32 → fp16 변환 (독립 실행, GPU 불요).
- `README_colab.md` — 이 문서.
