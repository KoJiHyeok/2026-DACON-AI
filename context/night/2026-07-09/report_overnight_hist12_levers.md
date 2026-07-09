# 밤샘 리포트 — hist12 기반 A/B 레버 프로토타입 + 배포코드 검증 (2026-07-09→10)

> 실행: ultracode 워크플로우 `overnight-hist12-levers` (9 에이전트, 무제출·로컬).
> 산출물: `night_out/night_hist12/*.json`, `scripts/night_hist12/*.py`.
> baseline: hist12 4-way+soft-AU 리그 = **0.75601** (격리 델타 +0.0215 재확정, 반반 안정 half1 +0.01807 / half2 +0.02475).

## TL;DR

- **A(GBDT 메타 스태커)·B(탐색 specialist) 4개 레버 전부 로컬에서부터 손해 → LB 예산 0으로 조기 폐기 확정.** 반반 동부호 음수라 신기루 아닌 진짜 마이너스. 내 07-09 recon의 "낮은 기대"가 "실측 로컬 마이너스"로 격상. #17(메타셀렉터)·#32(피처 전이 실패)와 정합.
- **hist12 리그 게이트는 재확정 PASS** — 배포는 여전히 유효한 유일 레버.
- **배포 스모크가 진짜 버그 2건 포착**: ① `_encoder_max_hist` BOM 미처리(→hist12가 조용히 hist6로 폴백하는 계약파손) **[수정 완료: utf-8-sig]** ② scale_smoke 로컬 env sklearn 1.9↔stacker 1.8 피클 불일치(서버는 1.8 → 배포 아닌 로컬 tooling 이슈).

## A. GBDT 메타 스태커 (3-config, hist12 리그 nested SGKF)

| config | 규제 | 델타 | half1 / half2 | 판정 |
|---|---|---:|---|---|
| strongreg | leaf15·minleaf200·l2=1.0 | **−0.0097** | −0.004 / −0.015 | 폐기 |
| medreg | leaf31·minleaf100·l2=0.3 | **−0.0119** | −0.008 / −0.016 | 폐기 |
| probsonly | 확률만(구조피처 없음) | **−0.0174** | −0.015 / −0.020 | 폐기 |

- 규제를 강하게 할수록 손해가 줄지만(strong > med > probsonly) **어떤 config도 hist12 선형결합을 못 이김.** 구조피처가 있는 쪽(strong/med)이 확률만(probsonly)보다 나음 = 구조 신호가 약하게나마 도움되나 선형결합 천장을 못 넘음.
- 07-09 recon 근거와 정합: Q 0.9(성분 정오 고상관) + 균일 oracle 갭 = 결합기가 열 독립신호 부족. GBDT per-row 비선형도 #17 LogReg와 같은 벽. **"낮은기대 1게이트" → "로컬 마이너스, 게이트 불필요"로 확정.**

## B. 탐색 클러스터 판별피처 specialist

| 레버 | 델타 | half1/half2 | 판정 |
|---|---:|---|---|
| explore specialist (soft/add best) | **−0.00043** | −0.0004 / −0.0005 | 폐기 |

- 좁은 어휘 마커(찾아→grep/열어→read/디렉토리→list/몇개→glob) + char_wb specialist를 정직 프로토콜(holdout 제외 학습)로 soft-route/blend-add — 사실상 중립(−0.0004). #32(char-tfidf 전이 0)·#14(R4 explore route 실패)와 정합. **탐색 잔차는 features/label-ambiguity 벽 재확인.**

## 배포코드 검증

- **pytest**: 신규 serialize 테스트 6개 중 5 PASS, `test_reads_bom_json` 1 FAIL → **BOM 버그 수정 완료**(`_encoder_max_hist` utf-8-sig). 재검증 필요(tester). block_weights 5개는 무회귀 PASS.
- **scale_smoke 하네스**: 정상 동작(40행 샌드박스→서브프로세스→peak RSS 467.7MB→zip 910MB 측정). 단 로컬 stacker 로드가 sklearn 1.9↔1.8 피클 불일치로 실패 → **로컬 스모크 env를 sklearn 1.8.0으로 맞추거나 실 스모크는 T4에서**. submit.zip 910MB/한도 1024(여유 114MB) 확인.
- **hist12 리그**: 0.75601 정확 재현, 격리 델타 +0.0215, 대조군 정합 −0.00426(레시피 일치).

## 내일 아침 우선순위

1. **BOM 수정 tester 재검증** → 6/6 PASS 확인.
2. **scale_smoke 로컬 env 정합**(.venv-merge sklearn 1.8.0 핀) 또는 T4 스모크로 실추론시간 실측.
3. **e5 hist12 fp16 도착 시**: model/encoder 스왑 + `serialize_config.json{max_hist:12}` 투하(BOM 없이) → `/submit` 게이트(G1~G5, 수동 zip 금지) → LB.
4. A/B 4개 레버 → experiments.md 폐기 등재(#35~#38).
5. hist12 배포 후 maxlen 512 승격 프로브(현 384에서 8.5% 잘림) 우선순위 재검토.
