# task2 linear2 report

## 요약

- baseline 재현: `scripts/linear2/baseline_repro.py --all-folds --evaluate --tune-bias`
- 원본 OOF: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\linear_probs.npy`
- 고정 fold: 동일 디렉터리의 `fold_indices.json` 3-fold 사용. `load_saved_folds()`에서 valid 중복과 group leakage를 검사함.
- baseline 재현 결과: OOF macro-F1 `0.663895` vs 원본 `0.663307`, delta `+0.000588`로 `< 0.002` 조건 통과.
- sweep 결과: 9개 config 완료. 최고 후보도 B4+soft-AU 리그 기준 `-0.000513`이라 전부 폐기.

## baseline 재현

원본 생성기는 `C:\dev\Second-Brain-Project\Hoseo\ai-2026\src\oof_lab_2026_07_03.py`에 남아 있었다. 이를 반영해 `baseline_repro.py`를 다음 조건으로 맞췄다.

- feature source: `submit/features.py`
- feature set: `E_+seq`
- model: `LinearSVC(C=0.1, class_weight=balanced, max_iter=1000, tol=1e-3, dual=True)`
- inner bias split: `GroupShuffleSplit(test_size=0.2, random_state=43)`
- class-bias tuner: 원본의 3-stage coordinate delta grid

결과:

| item | macro-F1 |
|---|---:|
| rebuilt OOF | 0.663895 |
| reference OOF | 0.663307 |
| delta | +0.000588 |

확률 차이는 남아 있다: mean abs prob diff `0.002314`, argmax disagreement `2.846%`. 하지만 task 기준인 macro-F1 차이는 허용 범위 안이다. 리그에 재현 linear를 넣으면 `0.738930`으로 reference B4+soft-AU `0.738772` 대비 `+0.000158`이지만, 이는 재현 오차 수준이라 후보 판정에는 사용하지 않았다. 이후 sweep 비교 기준은 원본 reference OOF의 `0.738772`로 고정했다.

## sweep 설정

- serializer: `compact`
- vectorizer: `TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", min_df=1, dtype=float32)`
- char-only: `analyzer="char_wb"`
- union: `FeatureUnion(word(1,2), char_wb)`
- model: `LinearSVC(class_weight="balanced", C=config, max_iter=3000, tol=1e-3, random_state=42, dual=True)`
- output: `night_out/linear2/<variant>/summary.json`, `sweep_results.csv`, `sweep_summary.json`
- evaluation: full 70k OOF macro-F1 + holdout 9,969 row 4-way+soft-AU league replacement

## sweep 결과

| rank | variant | OOF F1 | league | delta vs 0.738772 | decision |
|---:|---|---:|---:|---:|---|
| 1 | `char_2-5_mf120k_C1` | 0.620063 | 0.738260 | -0.000513 | discard |
| 2 | `char_3-5_mf120k_C1` | 0.618173 | 0.738181 | -0.000592 | discard |
| 3 | `char_2-4_mf120k_C1` | 0.616013 | 0.737933 | -0.000839 | discard |
| 4 | `char_3-5_mf120k_C2` | 0.613350 | 0.737882 | -0.000891 | discard |
| 5 | `char_3-5_mf120k_C0.5` | 0.618553 | 0.737865 | -0.000907 | discard |
| 6 | `union_2-5_mf120k_C1` | 0.617453 | 0.737716 | -0.001056 | discard |
| 7 | `union_3-5_mf120k_C1` | 0.616445 | 0.737655 | -0.001117 | discard |
| 8 | `char_3-5_mf200k_C1` | 0.617955 | 0.737590 | -0.001183 | discard |
| 9 | `char_3-5_mf300k_C1` | 0.617785 | 0.737590 | -0.001183 | discard |

PASS 기준:

- `+0.005`: LB 후보
- `+0.002 ~ +0.005`: report-only 후보
- 이번 최고: `-0.000513`

따라서 linear 교체 후보 없음.

## 가설 vs 결과

가설: char ngram을 더 크게/넓게 잡거나 word(1-2)+char union을 쓰면 원래 linear와 다른 sparse 신호가 생기고 4-way blend에서 보완 성분이 될 수 있다.

결과:

- OOF 단독 F1이 `0.616~0.620`으로 reference linear OOF `0.6633`보다 크게 낮다.
- reference linear와 argmax disagreement는 `26~28%`로 다양성은 있지만, 틀린 방향의 다양성이라 4-way+soft-AU에서 모두 손해다.
- `char_2-5`가 sweep 내 최상위였지만 delta `-0.000513`으로 노이즈보다 작고 음수다.
- `max_features 120k -> 200k/300k`는 개선이 없었다. 200k와 300k 모두 league `0.737590`.
- union은 char-only보다 일관되게 나빴다. `union_2-5`도 `char_2-5` 대비 league delta가 약 `-0.000543` 더 낮다.
- C축은 `3-5/120k` 안에서 `C=1.0`이 league 최상이고, `C=0.5`, `C=2.0` 모두 하락했다.

## 다음 task에 줄 결론

- 이 레인의 결론은 "linear 교체 불가"다. 현재 B4+soft-AU의 linear 자리에 compact char/union SVC를 넣는 방식은 LB 후보가 아니다.
- #29의 교훈과 동일하게, 단독 또는 OOF 지표가 조금 변해도 blend 기여가 자동으로 생기지 않는다. 이번에는 단독 F1 자체도 낮고 blend delta도 전부 음수다.
- sparse linear를 더 한다면 "전체 linear 교체"가 아니라 특정 class/subgroup의 확률 보정처럼 영향 면적을 좁혀야 한다. 다만 AU 라우팅 레인은 이미 소진 판정이므로 새 subgroup 근거가 먼저 필요하다.
- submit/model/model.pkl full-train 교체 작업은 하지 말 것. 현 결과로는 제출물 변경 가치가 없다.
