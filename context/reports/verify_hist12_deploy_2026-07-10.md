# hist12 배포 독립 검증 리포트 (2026-07-10 아침)

> 실행: ultracode 워크플로우 `morning-verify-hist12-prep` (tester·reviewer·recon 병렬 3에이전트, Sonnet) + 메인 세션 해시 대조.
> 대상: 밤샘 07-09 리포트 "내일 아침 우선순위" 1·2번 + 제출 #11 패키징 정합성.
> 주의: 이 검증은 **병렬 세션의 #11 패키징(커밋 7b59293, 대장 #11 07:58)과 동시 진행**됐다 — 아래는 그 산출물에 대한 독립 재검증이다.

## TL;DR — 전부 PASS, #11 LB 업로드 블로커 없음

| 항목 | 판정 | 근거 |
|---|---|---|
| BOM 수정 재검증 (우선순위 1) | **PASS 6/6** | tester: `test_encoder_serialize_maxhist.py` 6/6 (이전 FAIL이던 `test_reads_bom_json` 포함), block_weights 5/5 무회귀, **전체 스위트 22/22** → G1 충족 |
| per-encoder serialize diff 리뷰 | **approve, finding 0건** | reviewer: `serialize(s, max_hist=6)` 시그니처 일치(script.py:112↔226), config 없으면 default=6 무회귀 계약 확인, `json.JSONDecodeError ⊂ ValueError` 실증, 신규 코드 os/json만 사용(오프라인 안전) |
| experiments.md 수치 대조 | **일치** | reviewer가 `night_out/night_hist12/*.json` 원시 산출물 직접 대조 — GBDT −0.0097/−0.0119/−0.0174, specialist −0.00043, baseline 0.75601, 격리 +0.0215 전부 정확 |
| sklearn 1.9↔1.8 로컬 env 이슈 (우선순위 2) | **해소 확인** | .venv·.venv-merge 둘 다 scikit-learn 1.8.0 단일 설치. 메인 세션 실측: `.venv-merge`에서 `aar_models.joblib` **load OK** — 밤샘 리포트의 피클 불일치는 더 이상 재현 안 됨 (병렬 세션이 아침에 정합시킨 것으로 추정) |
| **인코더 스왑 진위 (해시 대조)** | **진짜 hist12** | staged `submit/model/encoder/model.safetensors` SHA256 `1F2AD870…` ≠ 백업 `encoder_e5h6_fp16_lb0.7480_bak` `09343462…`, 백업 == 원본 `enc_v2_s42` (바이트 수는 556,132,972로 셋 다 동일 — 같은 아키텍처 fp16이라 정상, 해시만이 판별자) |
| serialize_config.json | **정합** | `submit/model/encoder/serialize_config.json` = `{"max_hist": 12}` BOM 없음(reviewer xxd 확인), encoder_2(mBERT)에는 부재 → 6 폴백 = exp #34 계약(e5=12/mBERT=6) 그대로 |
| submit.zip | **한도 내** | 910,049,870B = 867.9MiB (07:57 생성) — 대장 #11 기재와 일치, 1GiB 한도 여유 ~156MiB |

## 세부

### 1. tester (pytest, .venv Python 3.13.13)
- `tests/test_encoder_serialize_maxhist.py` 6/6 PASS — 누락→6, 명시 12, **BOM 12**, malformed→6, ≤0→6, 실절단 차이.
- `tests/test_enc_block_weights.py` 5/5 PASS (무회귀).
- `pytest tests/ -q` → **22 passed, 0 failed** (test_features, test_merge080_script 포함).

### 2. reviewer (diff + 신규 테스트 + 수치)
- finding 0건. 확인 못 한 것으로 명시: T4 실측 추론시간(로컬 CPU로는 판정 불가 — scale_smoke docstring도 동일 입장), au_route.py의 독립 serialize는 별도 계약이라 범위 외.

### 3. env recon
- scale_smoke는 `sys.executable` 서브프로세스 spawn, 의도 인터프리터는 `.venv-merge`.
- `.venv-merge`는 `scripts/league4/probe_serialize_maxhist.py` 전용 — sklearn 핀 변경의 부작용 범위는 그 1개 스크립트로 국한, 학습·리그 파이프라인(.venv)은 무관.
- scale_smoke 풀스케일(N=30000)은 로컬 CPU에서 수십 분 추정 + 시간판정은 어차피 T4 몫 → 로컬 풀런 비권장.

## 남은 것 (이 검증 범위 밖)

1. **#11 LB 업로드 + 결과 기입** (사용자/병렬 세션) — 리그 격리 +0.0215 기준 기대선은 0.7501+α.
2. T4 실측 스모크 (추론 ≤10분 확인) — 로컬에서 불가.
3. hist12 LB 결과 후 maxlen 512 승격 프로브 우선순위 재검토 (밤샘 우선순위 5).
