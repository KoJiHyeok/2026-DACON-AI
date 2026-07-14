# CX-A Qwen block calibration report

## 결론

**기각(REJECT).** 세션 GroupKFold 5-fold OOF 파라미터로만 판정했을 때 새 표면의 row Macro-F1이 `0.771381 → 0.770775`(`-0.000606`)로 하락했다. 세션균등·MC200·반반도 모두 음수이고 paired session bootstrap 95% CI가 0을 가로지른다. `calib_candidate.json`은 전체 재적합 기술 후보로 보존하지만 현 챔피언에 배포하지 않는 것을 권고한다.

## 적합 계약

- 입력: Qwen instruct-2ep h85, 9,969행 / 1,350세션 / 14클래스.
- 정렬: 기존 `scripts/league4/common.py::align_npz_probs`로 holdout 기준 ID·라벨·클래스 순서를 강제했다.
- 분할: `GroupKFold(n_splits=5)`, group은 ID의 `-step_` 앞 세션 prefix. 각 행은 정확히 한 validation fold에 있고 train/validation 세션 교집합은 0이다.
- 적합: 각 fold에서 나머지 4개 fold의 NLL을 최소화하도록 scalar temperature를 `minimize_scalar`로 적합하고, mean-zero class intercept를 ridge `1e-3` logistic NLL로 적합했다. 대규모 grid는 사용하지 않았다.
- 판정 확률: 각 validation 행에는 그 fold 밖 데이터로만 적합한 T/bias를 적용했다. 전체 9,969행 재적합값은 배포 후보 JSON 생성에만 사용했고 아래 지표에는 사용하지 않았다.
- 적용식: `softmax(log(clip(p, 1e-12))/T + bias)`, 현 `submit/script.py`와 동치.

## 5지표 — 새 표면

기준선은 `(lin + stk + 3*qwen)/5` 뒤 soft-AU `alpha=0.85`, 후보는 Qwen 블록만 fold-honest calibration한 뒤 같은 조합이다. seed는 42다.

| 지표 | 기준 | 후보 / 분포 | 델타 | 판정 |
|---|---:|---:|---:|---|
| row Macro-F1 | 0.771381 | 0.770775 | **-0.000606** | 실패 |
| 세션균등 Macro-F1 | 0.776300 | 0.776023 | **-0.000277** | 실패 |
| 세션당 1행 MC200 | — | mean -0.000377 ± 0.002315 | min -0.006899 / max +0.007696 | 평균 음수 |
| paired session bootstrap 1,000회 | — | 95% CI **[-0.002460, +0.001060]** | P(Δ>0)=0.267 | 0 포함 |
| 반반 안정성 | — | half1 -0.000123 / half2 -0.000961 | 양쪽 음수 | 실패 |

엄격 LB gate(`row ≥ +0.005`, `MC mean > 0`, `bootstrap CI lower > 0`)를 전부 충족하지 못한다. 보고 기준 `+0.002`에도 미달한다.

## Fold별 안정성

| fold | valid 행/세션 | T | bias min / max | bias L2 | valid NLL Δ | Qwen solo F1 Δ | 최종 표면 F1 Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1,994 / 270 | 1.098402 | -0.230415 / +0.198062 | 0.558096 | -0.002245 | +0.001751 | -0.001842 |
| 1 | 1,994 / 270 | 1.103426 | -0.315525 / +0.183528 | 0.570931 | -0.003412 | +0.002584 | +0.002629 |
| 2 | 1,994 / 270 | 1.101214 | -0.400956 / +0.181119 | 0.516269 | -0.002407 | -0.007111 | +0.000178 |
| 3 | 1,994 / 270 | 1.099872 | -0.347877 / +0.157989 | 0.521076 | -0.002347 | +0.002662 | -0.001737 |
| 4 | 1,993 / 270 | 1.106334 | -0.205155 / +0.153412 | 0.447196 | -0.002459 | +0.000594 | -0.002309 |

T의 범위는 `1.0984~1.1063`으로 좁고 모든 fold의 NLL은 개선됐다. 그러나 class bias는 희소 클래스 영향으로 fold 변동이 있고, fold 2의 Qwen solo F1이 크게 하락했다. pooled Qwen solo OOF도 NLL은 `0.611941 → 0.609367`로 좋아졌지만 Macro-F1은 `+0.000348`에 그쳤고, 실제 3x-Qwen 블렌드 표면에서는 역전됐다. 즉 확률 NLL 개선이 현 argmax 블렌드의 순위 개선으로 이어지지 않았다.

전체 재적합 배포 후보는 T=`1.1020982801`이며 class bias는 `scripts/cx_calib/calib_candidate.json`에 있다. 이 값은 OOF 판정에 사용하지 않았다.

## 산출물·재현성

- `scripts/cx_calib/fit_calib.py`: fold-honest 적합 및 전체 재적합 후보 생성.
- `scripts/cx_calib/eval_calib.py`: 새 표면 구성과 5지표 판정.
- `scripts/cx_calib/fit_summary.json`: fold 파라미터·세션 해시·OOF 진단.
- `scripts/cx_calib/eval_results.json`: 원시 5지표·입력 SHA256·판정.
- `scripts/cx_calib/calib_candidate.json`: `submit/script.py::load_calib` 호환 최종 재적합 후보.
- Qwen NPZ SHA256: `a6547fba022e95365358b38a520ab3f6019c68b5a0c79193b389345551af5ac4`.
- AU holdout 확률 SHA256: `831f9953e57bc660913c0eb80226b1f4f42c3909ff2c2ac08fdc375525dd7c5d`.
- 테스트: `pytest tests/test_cx_calib.py -q` → 3 passed. 추가로 confusion 집계 최적화가 sklearn Macro-F1과 unweighted/weighted 모두 수치 동치임을 대조했다.
- 결정론성: 연속 실데이터 fit/eval 재실행에서 candidate `f852cc42…`, fit summary `e9386be3…`, eval results `ac25b537…`가 모두 byte-identical이었다.

## Handoff

- Task ID: CX-A / task4
- Branch / commit: `night/2026-07-14/task4` / 커밋 불가 (`C:/dev/2026-AI-DACON/.git/worktrees/task4/index.lock` 쓰기 권한 없음)
- 사용 모델: 현재 primary Codex 세션 모델, routed `codex exec` 미사용
- Reasoning: 현재 세션 기본값(모델 hot-swap 없음)
- 권고: **기각, calib 미배포**. `submit/**`, `scripts/league4/**`, canonical context, 공식 LB 제출은 수정·수행하지 않았다.
- 독립 검증 필요: reviewer는 외부 입력 SHA256과 5지표 독립 재실행, tester는 fold 세션 교집합 0·후보 JSON 로더 호환·결정론적 재실행을 확인할 것.
