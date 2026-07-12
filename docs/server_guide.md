# 학교 서버 작업 가이드 — 내가 할 일 순서대로

> 2026-07-10 기준. 서버 = dclab419-ESC4000-E10 (RTX A5000 24GB × 4, CUDA 13.0, Python 3.10.12, 인터넷 O).
> 일반 원칙·배경은 `docs/server_setup.md` 참고. 이 문서는 **오늘 밤 실행 체크리스트**다.
> 모든 블록은 통째로 붙여넣기 가능 (부분 수정 불필요).

## 절대 규칙 4개

1. **디스크 예산 엄수** — 홈이 걸린 루트가 98% 사용(잔여 ~25GB)인데 ssd1~3 전부 권한 없음(07-10 실측) → 당분간 홈 모드. **런 끝나는 즉시 ckpt 삭제**, 시작 전후 `df -h /` 확인, 동시 2job까지만. ssd 권한 확보 시 data/out을 `/mnt/ssd2/$USER`로 이전.
2. **GPU는 2장까지만** (GPU 0, 1) — 공용 랩 서버. 잡기 전 `nvidia-smi`로 유휴 확인.
3. **레시피 변경 금지** — A5000이 T4보다 빨라도 배치/에폭/lr/maxlen/maxhist 그대로 (실험 판정 오염 방지). 빨라지는 건 벽시계뿐.
4. **학습은 반드시 tmux 안에서** — 노트북 꺼도 계속 돈다. 분리 `Ctrl+B, D`, 복귀 `tmux attach -t <이름>`.

---

## ☑ 0. 작업 공간 — 실측 결과(07-10): 홈 모드 확정

- `/mnt/ssd1`·`ssd2`·`ssd3` **전부 Permission denied** (계정이 자기 그룹 1011에만 속함, `id` 실측).
- → **임시로 홈 모드**: `WORK=$HOME`. 홈이 걸린 루트가 98%(잔여 ~25GB)라 오늘 밤 2job(≈10GB)까지만 가능. **5-fold OOF 연쇄는 홈에서 금지** — ssd 확보 후에.
- 병행으로 서버 담당자에게 요청 발송:
  `sudo mkdir /mnt/ssd2/u20220876 && sudo chown u20220876:u20220876 /mnt/ssd2/u20220876`
  (확보되면 venv 제외 data/out만 옮기고 이 문서의 `$WORK`를 `/mnt/ssd2/$USER`로 전환)

```bash
export WORK=$HOME
mkdir -p $WORK/data $WORK/out
echo "export WORK=$HOME" >> ~/.bashrc
```

## ☑ 1. venv — 이미 완료 (07-10 실측: torch 2.13.0+cu130, CUDA True)

`~/venv-dacon` 재사용. 나머지 패키지만 보강 + 재현 스냅샷:

```bash
source ~/venv-dacon/bin/activate
pip install "transformers>=4.51" accelerate sentencepiece scikit-learn numpy pandas
pip freeze > $WORK/out/server_requirements.txt
```

(재현 기록 스냅샷은 7/20 재현 코드 제출 산출물의 일부, D-012. torch가 Colab과 버전이 다르지만 레시피 계약이 같으면 fp16 미세 수치차는 판정에 유효 — setup.md 주의 참조.)

## ☐ 2. 코드 + 데이터 배치

서버에서:

```bash
git clone <리포 주소> $WORK/dacon
```

(리포가 private라 clone이 번거로우면 최소한만 로컬에서 전송해도 된다:
`scp -r C:\dev\2026-AI-DACON\colab u20220876@210.119.108.236 (⚠️ SSH 포트 419 — `ssh -p 419`/`scp -P 419`, 포트 22는 방화벽 차단):dacon/colab` — 콜론 뒤 상대경로는 홈 기준)

**로컬 PC PowerShell에서** (data/는 gitignore라 항상 직접 전송):

```powershell
scp C:\dev\2026-AI-DACON\data\train.jsonl C:\dev\2026-AI-DACON\data\train_labels.csv u20220876@210.119.108.236 (⚠️ SSH 포트 419 — `ssh -p 419`/`scp -P 419`, 포트 22는 방화벽 차단):data/
scp C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz u20220876@210.119.108.236 (⚠️ SSH 포트 419 — `ssh -p 419`/`scp -P 419`, 포트 22는 방화벽 차단):data/
```

## ☐ 3. 작업 1 — args-lite candidate (GPU 0, 최우선 · D-011)

```bash
tmux new -s args1
source $WORK/venv-dacon/bin/activate && cd $WORK/dacon
export CUDA_VISIBLE_DEVICES=0
export ENC_MODE=holdout85 ENC_MAXHIST=12 ENC_MAXLEN=384 ENC_SEED=42 ENC_ARGSLITE=1 \
       ENC_DATA_DIR=$WORK/data ENC_HOLDOUT_NPZ=$WORK/data/holdout_base.npz \
       ENC_OUT=$WORK/out/e5_h12_args1
python colab/encoder_e5_holdout85_maxhist.py 2>&1 | tee $WORK/out/e5_h12_args1.log
```

시작 로그에 `[cfg] max_hist=12 args_lite=1 ...`이 찍히는지 확인 후 `Ctrl+B, D`로 분리.

## ☐ 4. 작업 2 — Bet B: mBERT hist12 (GPU 1)

```bash
tmux new -s betb
source $WORK/venv-dacon/bin/activate && cd $WORK/dacon
export CUDA_VISIBLE_DEVICES=1
export MDEB_MODEL=bert-base-multilingual-cased MDEB_MODE=holdout85 MDEB_MAXHIST=12 \
       MDEB_EPOCHS=2 MDEB_BATCH=8 MDEB_ACCUM=2 MDEB_MAXLEN=384 MDEB_SEED=42 \
       MDEB_DATA_DIR=$WORK/data MDEB_HOLDOUT_NPZ=$WORK/data/holdout_base.npz \
       MDEB_OUT=$WORK/out/mbert_h12
python colab/mdeberta_finetune.py 2>&1 | tee $WORK/out/mbert_h12.log
```

`Ctrl+B, D`로 분리. 이후 노트북을 꺼도 된다.

## ☐ 5. 진행 확인 (아무 때나)

```bash
tmux attach -t args1        # 또는 betb (분리는 Ctrl+B, D)
tail -5 $WORK/out/e5_h12_args1.log
nvidia-smi                  # 두 GPU가 돌고 있는지
```

완료 신호 = 로그 마지막에 `[npz] ... macro-F1=...` 라인. A5000 기준 각 ~40-60분 예상.

## ☐ 6. 산출물 회수 (로컬 PC PowerShell)

```powershell
scp u20220876@210.119.108.236 (⚠️ SSH 포트 419 — `ssh -p 419`/`scp -P 419`, 포트 22는 방화벽 차단):out/e5_h12_args1/holdout_e5_h12_args1.npz C:\dev\2026-AI-DACON\colab_out\
```

mBERT는 `~/out/mbert_h12/` 안의 holdout npz 파일명을 먼저 확인(`ls ~/out/mbert_h12/`)하고, 판정 프로브가 기대하는 이름으로 받는다:

```powershell
scp u20220876@210.119.108.236 (⚠️ SSH 포트 419 — `ssh -p 419`/`scp -P 419`, 포트 22는 방화벽 차단):out/mbert_h12/<확인한 npz명> C:\dev\2026-AI-DACON\colab_out\holdout_mbert_h12.npz
```

**→ npz 두 개가 colab_out/에 도착하면 Claude 세션에 알리기.** 판정(probe_c_args_lite / probe_b_mbert_hist12, 게이트 +0.005 & bootstrap CI & MC)과 실험 기록은 로컬에서 진행.

## ☐ 7. 뒷정리 (런 완료 후, 디스크 위생)

```bash
rm -rf $WORK/out/e5_h12_args1/ckpt $WORK/out/mbert_h12/ckpt   # 옵티마이저 포함 ckpt는 회수 후 즉시 삭제
```

## 다음 (내일 이후, 판정 나온 뒤)

- args-lite **PASS** → full-train 재학습(`ENC_MODE=full`) + GPU 2·3으로 **P1-C: 5-fold group-OOF e5 연쇄** (hist12-aware 동료 stacker 레인 — 이 서버의 최대 활용처)
- args-lite **FAIL** → pair-order 변형 단일 A/B (D-011 후속 카드)
- 병행: 서버를 7/20 재현 환경 기준으로 삼아 성분별 원커맨드 재학습 검증 (D-012 재현성 트랙)
