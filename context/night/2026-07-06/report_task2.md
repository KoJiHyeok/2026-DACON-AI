# task2 리포트 — mdeberta Colab 팩 (밤샘에서 낮으로 선행 완료, 2026-07-05 오후)

밤샘 task2로 예정했던 작업을 GPU 시간 확보를 위해 메인 세션이 07-05 오후에 선행 수행했다.
**task2.md는 삭제됨 — 밤샘 러너는 task1(char-ngram)·task3(sim/au)만 돈다.**

## 산출물

- `colab/mdeberta_finetune.py` — env 폴백(붙여넣기 규약), serialize() verbatim(리뷰로 문자 단위 대조),
  학습 fp32 고정(DeBERTa-v3 T4 fp16 NaN 함정), holdout split은 npz ids 직접 사용(부분집합 assert),
  MDEB_CKPT_STEPS 중간 체크포인트 + 동일 셔플 skip 재개, npz는 fp16 변환 **전** 계산.

## 검증 (작성자·검증자 분리)

- reviewer(계약 리뷰): serialize verbatim PASS, 결함 3건 발견(약한 assert / fp16 in-place 후 npz 오염 /
  에폭 단위 ckpt만) → 전부 반영.
- tester 1차(스모크): end-to-end PASS (npz 스키마·softmax·오프라인·resume 로그).
- tester 2차(수정본): ① 중간 ckpt 저장 확인, ② npz가 fp16 저장보다 먼저 출력(순서 증거),
  ③ step=10 강제 후 `[resume] epoch 0 step 10부터 재개 (동일 셔플 skip)` + skip 정상,
  ④ 스테일 npz(가짜 id 3개) → 학습 진입 전 AssertionError 즉사. 전 시나리오 PASS.

## Colab 사용법 (순서대로 셀 3개)

1. `!pip install -q "transformers>=4.51" accelerate sentencepiece`
2. 준비 셀: drive.mount → train.jsonl 자동 탐색 → `MDEB_DATA_DIR`/`MDEB_HOLDOUT_NPZ`/`MDEB_OUT`(Drive)/`MDEB_RESUME=1` env 설정 (서브계정은 공유 폴더 바로가기 필요 — colab-multi-account 규약)
3. `colab/mdeberta_finetune.py` 통짜 붙여넣기

시작 확인: `[cfg]`에 Drive 경로, `[split] valid=9969`(리그와 동일 행). 첫 200 step elapsed로
총 시간 재추정 — fp32는 T4 텐서코어를 못 써 에폭당 60~100분 추정, 2에폭이 부담이면 `MDEB_EPOCHS=1` 먼저.
OOM 시 `MDEB_GRAD_CKPT=1`.

## 미검증 (Colab에서 확인할 것)

- 실제 mdeberta-v3-base 로딩(vocab 250k, sentencepiece)·T4 시간/메모리 실측
- fp16 사본 실측 크기 (~550MB 추정 — e5-base 573MB와 합산 시 zip 1GB 초과 → **대체 후보로 평가**가 기본)

## 다음 판단

holdout_mdeb.npz 도착 시 리그에서: mdeberta solo vs e5-base solo(0.70509 프록시) → 3-way에서
인코더 교체/추가 add-test. 성분 추가/제거 축만 리그 신뢰(enc 지분 조정 금지 — exp #16/#20).
