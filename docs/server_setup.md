# 학교 대여 서버 셋업 가이드 (GPU 학습용 가상환경)

> 목적: Colab(세션 킬·쿼터·체크포인트 손상 리스크)을 학교 서버로 대체.
> `colab/` 스크립트들은 전부 env 변수 기반이라 **수정 없이 서버에서 그대로 실행**된다 —
> Drive 마운트 → 로컬 경로, `files.upload()` → git clone + scp로 바뀔 뿐.

## 0-A. 실측 사양 (2026-07-10 수령, dclab419-ESC4000-E10)

- GPU: **RTX A5000 24GB × 4** (수령 시점 전부 유휴) — 드라이버 580.159.03, CUDA 13.0. 공용 랩 서버(동시 사용자 여럿) → 점유는 2~3장까지만, 잡기 전 `nvidia-smi` 확인.
- Python 3.10.12, 인터넷 O (HF 직접 다운로드 가능).
- **디스크 함정: 홈(`/`)이 98% 사용(잔여 ~25GB)** — 홈에 venv/캐시/산출물 두지 말 것. `/mnt/ssd2`(3.3T 여유) 아래 `WORK=/mnt/ssd2/$USER`를 만들어 venv·HF_HOME·data·out 전부 거기에. (쓰기 불가면 ssd3 → 그것도 안 되면 관리자 문의)
- torch는 cu128 휠 사용 (`--index-url https://download.pytorch.org/whl/cu128` — CUDA 13 드라이버는 하위 런타임 호환).
- A5000이 T4보다 훨씬 빨라도 **레시피(배치/에폭/lr/maxlen/maxhist) 변경 금지** — 벽시계만 단축.

## 0. 서버 받으면 먼저 확인 (5분)

```bash
nvidia-smi                 # GPU 종류·VRAM·CUDA 드라이버 버전
python3 --version          # 3.10+ 필요
df -h ~                    # 디스크 여유 ~50GB 권장 (ckpt가 회당 ~3.3GB)
curl -sI https://huggingface.co | head -1   # 인터넷 여부 (모델 다운로드)
```

- **VRAM이 T4(16GB)보다 커도 배치/레시피 변경 금지** — 재현성 계약 (encoder_v2 레시피 고정, exp #34 판정 오염 방지).
- 인터넷이 막혀 있으면: 로컬 PC에서 HF 모델(`intfloat/multilingual-e5-base`, `bert-base-multilingual-cased`)을 `snapshot_download`로 받아 scp → `HF_HUB_OFFLINE=1` + 로컬 경로를 모델명 대신 지정.

## 1. 가상환경 (venv 기준 — conda 있으면 동일 구성)

```bash
python3 -m venv ~/venv-dacon
source ~/venv-dacon/bin/activate
pip install --upgrade pip

# torch는 서버 CUDA에 맞춰 (nvidia-smi의 CUDA Version 확인 후 pytorch.org 명령 사용)
pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA 12.x 예시

pip install "transformers>=4.51" accelerate sentencepiece scikit-learn numpy pandas
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 확인
```

## 2. 코드·데이터 배치

```bash
git clone <repo> ~/dacon && cd ~/dacon
mkdir -p ~/dacon-data ~/dacon-out
# 로컬 PC에서 (data/는 gitignore — 직접 전송):
#   scp data/train.jsonl data/train_labels.csv <server>:~/dacon-data/
#   scp context/night/2026-07-05/holdout_base.npz <server>:~/dacon-data/   # holdout85 모드용
```

## 3. 학습 실행 (예: e5 holdout85 프로브)

```bash
cd ~/dacon
export ENC_MODE=holdout85 ENC_MAXHIST=12 ENC_MAXLEN=384 ENC_SEED=42 \
       ENC_DATA_DIR=~/dacon-data ENC_HOLDOUT_NPZ=~/dacon-data/holdout_base.npz \
       ENC_OUT=~/dacon-out/e5_h12

# tmux 권장 (SSH 끊겨도 계속) — Colab식 세션 킬 없음
tmux new -s train
python colab/encoder_e5_holdout85_maxhist.py 2>&1 | tee ~/dacon-out/e5_h12/train.log
# 분리: Ctrl+B, D / 복귀: tmux attach -t train
```

- mBERT/mdeberta 계열은 `colab/mdeberta_finetune.py` + `MDEB_*` env (MODEL/MODE/MAXHIST/EPOCHS/BATCH/ACCUM…).
- 배포용 full-train은 `ENC_MODE=full` — `model_fp16/` 산출.
- **중단 재개는 Colab과 동일**: `$ENC_OUT/ckpt/checkpoint-*`가 있으면 자동 이어감. 서버는 세션 킬이 없어 사실상 불필요.

## 4. 산출물 회수

```bash
# 로컬 PC에서:
scp <server>:~/dacon-out/e5_h12/holdout_e5_h12.npz  C:/dev/2026-AI-DACON/colab_out/
scp -r <server>:~/dacon-out/<full런>/model_fp16     <임시폴더>   # 배포 인코더
```

리그 판정·배포 게이트는 기존 그대로 로컬에서 (`scripts/league4/probe_*.py` → `/submit`).

## 주의

- 서버 pip 버전을 `pip freeze > ~/dacon-out/server_requirements.txt`로 스냅샷 — 학습 재현 기록용 (제출 requirements.txt와는 무관, 그건 서버 기본 패키지 계약 유지).
- 시드 고정은 스크립트에 내장(SEED=42). GPU가 달라지면 fp16 연산 순서로 미세한 수치 차이는 있을 수 있으나 레시피 계약(에폭·배치·lr·maxlen·maxhist)이 동일하면 판정에 유효.
