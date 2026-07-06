# Colab 실행 기록 — mdeberta-v3-base holdout85 학습 (2026-07-06)

> 사후 기록 (reviewer 지적: 실행 로그가 context/에 없으면 일어나지 않은 것).
> 스크립트: `colab/mdeberta_finetune.py` (수정 없이 통짜 붙여넣기 실행, env 셀로 설정 주입)

## 설정 (env)

- MDEB_MODEL=microsoft/mdeberta-v3-base, MDEB_MODE=holdout85 (valid = holdout_base.npz ids 직접 사용)
- MDEB_OUT=/content/drive/MyDrive/mdeb_out (소실 사고 2회 후 Drive 고정 + assert)
- MDEB_EPOCHS=2, MDEB_MAXLEN=384, MDEB_LR=2e-5, MDEB_SEED=42
- OOM 대응: MDEB_GRAD_CKPT=1, MDEB_BATCH=4, MDEB_ACCUM=4 (유효 배치 16)
- 학습 fp32 고정 (DeBERTa-v3 T4 fp16 NaN 함정), fp16은 저장 사본만

## 종료 로그 (사용자 전달 원문)

```
[npz] holdout_mdeb.npz rows=9969 final macro-F1=0.66998
[save] fp16 사본 574MB — e5-base(573MB)와 합산 1GB 초과 여부 확인할 것
[DONE]
```

## 산출물

- `colab_out/holdout_mdeb.npz` (3.0MB, ids/probs/y_true/actions 9,969행) — 리그 판정 완료 (exp #26)
- Drive `mdeb_out/model_fp32`, `mdeb_out/model_fp16` (574MB) — zip 크기 제약으로 현재 미사용
- 격리 검증: reviewer가 holdout85 로직 정적 검증 + npz id 집합 == holdout_base ids 일치 확인 (full 모드면 npz 자체가 안 생기는 구조라 격리 학습 확정)

## 판정 → exp #26

교체 FAIL(−0.005) / 블록 분할 [1,1,e5+mdeb] 리그 +0.0053이나 zip 초과로 미제출.
후속: mBERT(356MB)로 같은 구조 재시도 (exp #27 예정).
