# 템플릿(current_prompt) 완전일치 라우팅 카드 판정 — G4 후속 CPU 분석

> 작성일: 2026-07-10
> 배경: [deep_research_gap_check_2026-07-10.md](deep_research_gap_check_2026-07-10.md) §G4
> 스크립트: `scripts/analysis/template_dup_probe.py` (재실행 가능, 시드 고정, argparse `--out-dir` 기본값 있음)
> 산출물: `scripts/analysis/_out/template_dup_probe_result.json`, `_grids.csv`, `_overrides.csv`

## 0. 한 줄 판정

**폐기.** 템플릿 완전일치 커버리지는 홀드아웃의 13.85%로 예상보다 크지만, 대부분(테일)은 이미 챔피언이 거의 다 맞히는 `respond_only` 종료형 프롬프트라 오버라이드 이득이 없다. 신뢰 가능한 그리드(purity/n 조건)로 제한하면 커버리지가 1.6~2.2%까지 줄고, 오버라이드 델타는 세 그리드 모두 게이트(+0.005)에 크게 못 미친다 — 최고값도 +0.00009(MC 평균), row 기준 최고는 사실상 0.000000. 절대 GPU/코드 예산을 쓰지 말 것.

## 1. 방법

### 1.1 챔피언 재현 (재발명 없이 기존 코드 재사용)

`scripts/league4/common.py`의 `load_league_data()` / `four_way_blend()` / `apply_soft_au()` / `train_or_load_au_probs()`를 그대로 호출했다. 단, `common.py`의 `HOLDOUT_BASE`(`context/night/2026-07-05/holdout_base.npz`)는 e5 hist6 구버전이라, 배포된 챔피언(exp #34/#35 승격, exp #37로 재확인)과 맞추기 위해 `scripts/league4/probe_b_mbert_hist12.py` / `diag_hist12_confusion_delta.py`와 동일한 패턴으로 `dataclasses.replace(data, e5=h12)`를 이용해 e5 슬롯만 `colab_out/holdout_e5_h12.npz`(hist12)로 교체했다. mBERT 슬롯은 `colab_out/holdout_mbert.npz`(hist6) 그대로 두었다 — `holdout_mbert_h12.npz`는 Bet B 후보로만 언급되고(`probe_b_mbert_hist12.py`), 실제 파일이 저장소에 없으며 exp #37에서 "FAIL/유지"로 종결된 미배포 성분이기 때문이다. 블록 가중은 `common.BASE_E5_WEIGHT=1.2` / `BASE_MBERT_WEIGHT=0.8`, soft-AU는 `common.DEFAULT_ALPHA=0.9`.

### 1.2 템플릿 그룹핑 (누수 방지)

- `data/train.jsonl` 70,000행 전량 로드, `current_prompt`는 트림만 (완전일치가 목적이므로 추가 정규화 없음).
- 세션 그룹키 = id에서 `-step_\d+$` 제거 (`league4/common.py`의 `session_id`와 동일 로직).
- 홀드아웃 1,350세션(9,969행, `holdout_base.npz`의 ids에서 세션 프리픽스 추출)에 속하는 모든 train 행을 오버라이드 소스에서 제외 → 오버라이드 소스 60,031행, train측 유니크 `current_prompt` 그룹 54,734개. 코드에서 `assert`로 누수(홀드아웃 세션이 오버라이드 소스에 남는 경우) 여부를 검증했고, 위반 없음을 확인했다.
- 홀드아웃 9,969행 각각의 `current_prompt`를 이 그룹과 완전일치 매칭.

## 2. 챔피언 재현 검증

| 항목 | 값 |
|---|---|
| 재현 macro_f1 | **0.756006** |
| 기록값(exp #37, 리그 0.756006) | 0.756006 |
| 차이 | **0.000000 (완전 일치)** |

기록값과 정확히 일치했다 — 5단계 전체가 별도 조사 없이 신뢰 가능한 기반 위에 있다.

## 3. 템플릿 커버리지 및 오류율

| 구분 | 행수 | accuracy | error rate | macro_f1 |
|---|---|---|---|---|
| 템플릿 행 (완전일치 그룹 존재) | 1,381 (13.85%) | 0.8016 | 0.1984 | 0.7592 |
| 비템플릿 행 | 8,588 (86.15%) | 0.7465 | 0.2535 | 0.7565 |

템플릿 행에서 챔피언 accuracy가 비템플릿보다 5.5%p 높다 — 챔피언이 이미 템플릿 신호를 상당 부분 흡수하고 있다는 뜻(템플릿 대부분이 `respond_only` 세션 종료형 프롬프트라 예측이 쉬움, 전체 exact-dup 최상위 그룹은 `context/reports`가 참조하는 `exact_dup_groups.csv` 최상위 항목과 일치 — "정리해줘/마무리해줘"류 142/93/92행 그룹이 모두 purity 1.0 respond_only).

챔피언이 템플릿 행에서 자주 틀리는 상위 오류쌍: `grep_search→read_file`(35), `read_file→list_directory`(21), `ask_user→plan_task`(18), `grep_search→list_directory`(17), `plan_task→ask_user`(16), `read_file→grep_search`(16) — 탐색계열/AU-plan 혼동으로, 기존에 알려진 전역 혼동 패턴과 동일하다(템플릿 특유 오류가 아님).

## 4. 그리드별 다수라벨 정확도 및 오버라이드 시뮬레이션

| 그리드 | 조건 | 적용 그룹수 | 커버리지 | 다수라벨 정확도 | row baseline | row override | **row delta** | **MC mean delta** | MC std | MC range |
|---|---|---|---|---|---|---|---|---|---|---|
| A | purity≥0.90, n≥5 | 41 | 1.75% (174행) | 0.9828 | 0.756006 | 0.755889 | **−0.000117** | **+0.000090** | 0.000232 | [0.000000, +0.000907] |
| B | purity≥0.95, n≥10 | 22 | 1.57% (157행) | 1.0000 | 0.756006 | 0.756006 | **0.000000** | **0.000000** | 0.000000 | [0.000000, 0.000000] |
| C | purity≥0.99, n≥3 | 293 | 2.18% (217행) | 0.9401 | 0.756006 | 0.755243 | **−0.000763** | **−0.000353** | 0.000889 | [−0.002595, +0.000907] |

MC = 세션당 1행 랜덤 샘플링 50회 반복(numpy `RandomState(42+i)`, i=0..49).

관찰: 그리드 B는 다수라벨 정확도 100%지만 이는 애초에 챔피언이 이미 다 맞히던 157행이라 델타가 0에 수렴한 것 — "오버라이드가 필요 없을 만큼 쉬운 부분집합"일 뿐, 이득의 증거가 아니다. 그리드 A/C는 커버리지를 넓히자 오히려 순손실(다수라벨이 챔피언보다 부정확한 사례 포함). 세 그리드 모두 게이트 +0.005에 크게 못 미친다.

## 5. 판정

**게이트: 홀드아웃 Macro-F1 델타 ≥ +0.005 → 후속 검토, 미만 → 폐기.**

최고 성과 그리드조차 row 기준 0.000000(B), MC 평균 기준 +0.000090(A) — 게이트 대비 두 자릿수 이상 미달. **카드 폐기를 제안한다.** 완전일치 템플릿 신호는 이미 챔피언 블렌드(특히 e5/mBERT 인코더가 attention으로 반복 프롬프트 패턴을 학습)에 대부분 흡수되어 있고, 남은 잔차는 오버라이드로 개선되지 않는다.

## 6. 한계 및 리스크

- **숨은 테스트 분포 불일치 위험**: 홀드아웃 템플릿 커버리지(13.85%, 좁은 그리드는 1.6~2.2%)가 시뮬레이터의 한 시점 스냅샷일 뿐, 실제 숨은 테스트셋의 템플릿 재사용 비율이 다를 수 있다. 다만 델타가 이미 게이트 대비 압도적으로 작아 분포가 다소 바뀌어도 결론이 뒤집힐 여지는 낮다.
- **템플릿 대부분이 저위험 클래스에 편중**: 상위 exact-dup 그룹은 거의 전부 `respond_only`(세션 종료 인사말류)로, 챔피언이 이미 매우 잘 맞히는 클래스다. 희소 클래스(예: `web_search`, `run_tests`)에서의 템플릿 재사용은 표본이 적어 (그리드 C조차 217행) 신뢰구간이 넓다 — MC std 최대 0.00089로 표본 수 자체가 작다는 신호.
- **오버라이드가 오히려 손해가 되는 경로 존재**: 그리드 A/C에서 다수라벨 정확도(0.94~0.98)가 챔피언 정확도(0.80, 템플릿 행 한정)보다 높아 보이지만 macro-F1(클래스 불균형 가중)에서는 오히려 순손실 — 이는 다수라벨이 흔한 클래스(`respond_only` 등)로 쏠리고, 챔피언이 이미 그 행들에서 맞히고 있던 희소 클래스 예측을 깨뜨리기 때문으로 추정된다(교차표 상세는 `_out/template_dup_probe_overrides.csv` 참고, 세부 행 단위 대조는 산출물에 미포함이라 추가 조사 시 재실행 필요).
- **AU 특수처리 정의 재확인**: 배경 프롬프트는 "AU(ask_user 관련) 행"이라 표현했지만, 실제 챔피언 코드(`submit/au_route.py`)의 AU는 `ask_user` 라벨이 아니라 **id가 `sess_au`로 시작하는 세션**을 뜻한다. 본 분석은 기존 코드의 정의를 그대로 따랐으며 이 차이가 결과에 영향을 주지는 않는다(오버라이드 로직 자체는 AU/non-AU 무관하게 텍스트 완전일치 매칭이므로).

## 7. 생성 파일

- `C:\dev\2026-AI-DACON\scripts\analysis\template_dup_probe.py` (분석 스크립트)
- `C:\dev\2026-AI-DACON\scripts\analysis\_out\template_dup_probe_result.json` (전체 결과 JSON)
- `C:\dev\2026-AI-DACON\scripts\analysis\_out\template_dup_probe_grids.csv` (그리드별 커버리지/다수라벨 정확도)
- `C:\dev\2026-AI-DACON\scripts\analysis\_out\template_dup_probe_overrides.csv` (그리드별 오버라이드 델타, row+MC)
- `C:\dev\2026-AI-DACON\context\reports\template_dup_probe_2026-07-10.md` (본 보고서)
