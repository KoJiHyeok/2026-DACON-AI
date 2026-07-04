# 팀 리포 조사 리포트 — dacon-agent-action-api-boost (2026-07-04)

> clone 위치: `C:\dev\dacon-agent-action-api-boost` (main @ 12452b7, 작업 트리 clean)
> 목적: w112 재건에 필요한 것 중 무엇이 리포에 있고 무엇이 없는지 확정.

## 요약 (w112 재건 관점)

| w112 성분 | 리포에 있나 | 위치 |
|---|---|---|
| linear 아티팩트 (model.pkl 8MB) | ✅ | `linear_pipeline/submit/model/model.pkl` |
| stacker 아티팩트 (aar_config + aar_models.joblib 31.5MB) | ✅ | `model/` (루트) |
| 3-way 추론 코드 (script_3way.py, aar_infer.py, features.py) | ✅ | `ensemble/` |
| 빌더 (package_ensemble.py) | ✅ | `ensemble/` — ⚠️ 경로가 팀원 로컬 레이아웃 가정, 조정 필요 |
| weights.json `[1,1,2]` | ❌ (사소) | 한 줄이라 재생성 가능: `{"weights":[1.0,1.0,2.0]}` |
| **encoder v2 s42 가중치 (fp32 ~1.1GB / fp16 547MB)** | ❌ **핵심 결손** | 팀원 로컬 `colab_out/enc_v2_s42/model/`. git에 없음 — **Drive 링크 요청** 또는 `colab/encoder_v2_full.py`(seed 42)로 재학습 |

→ **인코더 가중치만 오면 w112 완전 재건 가능.** 그 전에도 linear+stacker 2-way blend 검증은 지금 재료로 가능.

## 리포 구조

- **루트 `submit.zip` (78.5MB)** = AAR-Max 스태커 계열 제출물 (w112 아님). README의 CV 0.7075는 누수 split 값 — 정직 LB는 **0.669** (RESULTS §1~2에 원인 분석: 세션 누수 + 분포 이동).
- `linear_pipeline/` = linear 가족 완본 (E_+seq features, train_final.py, tune_bias.py, LB 0.6732)
- `ensemble/` = 3-way 전체 코드 + calib_v1.json + soup/bucket/oof 실험 랩
- `colab/` = 인코더 학습 스크립트 (encoder_finetune.py=e5-small, encoder_v2_full.py=e5-base full-70k)
- `src/` = aar_features, transformer_classifier, serialize, **rules.py**(규칙 실험 흔적 — 포렌식 착수 전 확인 가치), ensemble, io_utils
- `notes/` + `RESULTS.md` = 실측 기록 (핸드오프보다 정밀한 부분 있음)

## 검증 결과 (우리 하네스)

- 루트 `submit.zip` → **12/12 PASS**, 오프라인 실행 36.3초 (5행 기준)
- **교훈 1**: 팀 requirements.txt는 주석뿐 — DACON 서버가 pandas/numpy/sklearn/joblib/torch/transformers를 기본 제공, "필요한 것 외 추가 금지" 정책. 우리 검증기의 버전 고정 체크를 FAIL→권고로 완화함.
- **교훈 2**: 로컬 sklearn 1.9.0으로는 팀 아티팩트 역직렬화 실패(`_loss` 모듈) → **서버 미러 venv 필수**. `C:\dev\2026-AI-DACON\.venv` = sklearn 1.8.0 + joblib 1.5.3 + pandas + numpy. 앞으로 모든 로컬 검증은 이 venv로: `.venv\Scripts\python.exe scripts\validate_submit.py <zip>`

## RESULTS.md에서 얻은 추가 실측 (핸드오프에 없거나 더 정밀)

- **`history`(직전 행동 시퀀스)가 최대 레버: +0.127** — 텍스트 인코더가 아니라 시퀀스가 본질. 시뮬레이터 포렌식 방향을 강하게 지지.
- E_+seq(+0.073)의 이득은 탐색 4클래스(glob/grep/read/list)에 집중. char n-gram은 +0.001로 무의미.
- per-class bias 튜닝 +0.009. 전역 temperature는 argmax 불변 → Macro-F1에 무효 (비대칭 shift만 유효).
- 3-way 성분별 solo LB: linear 0.6732 / stacker 0.6708 / e5-small 0.6696 → uniform 평균 0.6930 (+0.02, 오류 이질성의 직접 증거).
- `script_3way.py` 서빙 메커니즘 (전부 default-off, 미사용 시 byte-identical 검증됨): `weights.json`, `calib.json`, 다중 encoder 블록 평균, `bucket_weights.json`.

## 다음 액션

1. **팀원에게 인코더 가중치 Drive 링크 요청** (`colab_out/enc_v2_s42/model/` — fp16이면 547MB)
2. 그 사이: linear+stacker 2-way blend를 우리 하네스로 재현 (ensemble/script_3way.py + 리포 내 아티팩트 조립)
3. `src/rules.py` 내용 확인 → 시뮬레이터 포렌식과 중복/재사용 판단
