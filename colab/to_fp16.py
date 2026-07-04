# -*- coding: utf-8 -*-
"""학습 산출물(fp32 safetensors)을 fp16으로 변환 — Colab 다운로드 용량 절감용.

배경: e5-base(XLM-R base, ~278M 파라미터) fp32 체크포인트는 ~1.1GB, fp16으로 변환하면
~550MB로 절반이 된다(팀 리포 notes/models.md 기준 실측 fp16 아티팩트 ~556MB와 일치).
변환 로직은 팀 리포의 검증된 패턴을 따른다:
  - src/transformer_classifier.py: AutoModelForSequenceClassification.from_pretrained(...)
    → model.half() → model.save_pretrained(...)
  - ensemble/package_ensemble.py 는 이미 fp16으로 변환된 model dir을 그대로 조립만 함
    (이 스크립트가 그 "이미 fp16으로 변환된" 단계를 담당).

이 스크립트는 encoder_v2_s42_repro.py의 산출물(OUT_DIR/model, fp32)을 입력으로 받아
별도 디렉터리(기본: <src>_fp16)에 fp16 사본을 만든다. 원본(fp32)은 보존된다.
GPU 불필요 — dtype 캐스팅 후 저장만 하므로 Colab에서든 로컬(CPU)에서든 동작한다.

사용 (Colab, 학습 직후 같은 런타임에서 실행 권장 — 다운로드 전에 용량을 줄여둠):
    python to_fp16.py --src /content/drive/MyDrive/dacon2026/enc_v2_s42/model \
                       --dst /content/drive/MyDrive/dacon2026/enc_v2_s42/model_fp16

기본 --dst를 생략하면 "<src>_fp16" 이 사용된다.
"""
import argparse
import json
import os
import shutil

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

EXPECTED_NUM_LABELS = 14


def dir_size_mb(path):
    total = 0
    for base, _, files in os.walk(path):
        total += sum(os.path.getsize(os.path.join(base, f)) for f in files)
    return total / 1e6


def convert(src_dir, dst_dir):
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"src model dir 없음: {src_dir}")
    os.makedirs(dst_dir, exist_ok=True)

    print(f"[1/4] fp32 모델 로드: {src_dir}")
    model = AutoModelForSequenceClassification.from_pretrained(src_dir)
    n_params = sum(p.numel() for p in model.parameters())
    n_labels = model.config.num_labels

    print(f"[2/4] fp16 변환 ({n_params / 1e6:.0f}M 파라미터, num_labels={n_labels})")
    model = model.half()

    print(f"[3/4] 저장: {dst_dir}")
    model.save_pretrained(dst_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(src_dir)
    tok.save_pretrained(dst_dir)

    # run.json이 model/ 상위에 있으면(원본 학습 스크립트 산출 구조) 참고용으로 복사
    run_json = os.path.join(os.path.dirname(os.path.normpath(src_dir)), "run.json")
    if os.path.isfile(run_json):
        shutil.copy(run_json, os.path.join(os.path.dirname(os.path.normpath(dst_dir)),
                                            "run.json"))

    print("[4/4] 검증")
    if n_labels != EXPECTED_NUM_LABELS:
        raise ValueError(f"num_labels={n_labels} (예상 {EXPECTED_NUM_LABELS}) — "
                          "잘못된 체크포인트일 수 있음")
    check = AutoModelForSequenceClassification.from_pretrained(dst_dir)
    bad_dtypes = {n for n, p in check.named_parameters() if p.dtype != torch.float16}
    if bad_dtypes:
        raise RuntimeError(f"fp16 변환 실패 — float16이 아닌 파라미터: {sorted(bad_dtypes)[:5]}")
    id2label = check.config.id2label
    if len(id2label) != EXPECTED_NUM_LABELS:
        raise ValueError(f"저장된 id2label 길이={len(id2label)} (예상 {EXPECTED_NUM_LABELS})")

    src_mb = dir_size_mb(src_dir)
    dst_mb = dir_size_mb(dst_dir)
    print(f"      OK — dtype=float16 확인, num_labels={EXPECTED_NUM_LABELS} 확인")
    print(f"      크기: fp32 {src_mb:.0f}MB → fp16 {dst_mb:.0f}MB")
    print(f"      완료: {dst_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True,
                    help="fp32 model dir (예: .../enc_v2_s42/model)")
    ap.add_argument("--dst", default=None,
                    help="fp16 출력 dir (기본: <src>_fp16)")
    args = ap.parse_args()
    src = os.path.normpath(args.src)
    dst = args.dst or (src.rstrip(os.sep) + "_fp16")
    convert(src, dst)


if __name__ == "__main__":
    main()
