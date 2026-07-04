# w112 챔피언 핸드오프 — DACON AI Agent Action 예측 (2026-07-04)

> **이 문서 목적**: 새 Claude Code 세션(또는 협업 동료·검증 하네스)이 우리 최고 제출물 **w112(LB 0.7208)**를
> 처음부터 이해하고 **재현·검증**할 수 있게 만드는 단일 핸드오프.
> 대회 규칙·데이터 스키마는 `../CLAUDE.md`, 폴더 운영 규칙은 `ai-2026/CLAUDE.md` 1절이 정본.

---

## 0. TL;DR (한눈에)

- **최고 제출 = w112 = LB Macro-F1 `0.7208`.** 14-class 행동 분류.
- **w112 = 3개 이종 모델의 가중 확률 평균**, 가중치 `[linear=1, stacker=1, encoder=2]` (= encoder 지분 0.50).
- **w112를 만든 "행동" 한 줄**: v2 uniform 앙상블(0.7190)에서 **재학습 없이** `weights.json`만 `[1,1,1] → [1,1,2]`로 바꿔 인코더 지분을 0.33→0.50으로 올린 것. (그 사이 0.7200 = `[1,1,1.5]` 경유.)
- **핵심 게이트**: 점수 판정은 **LB 실측** 또는 **세션 group-split** 로만. accuracy·누수 split·단일 holdout 금지.
- **현 위치**: top-12 컷 `0.77426`, 우리 81등. 미세 가중 프로브로는 컷 도달 불가 → 다음은 구조적 신호(시뮬레이터 포렌식) 우선.

---

## 1. 점수 여정 — "무슨 행동 → 몇 점"

모든 점수는 **LB 실측**(리더보드 제출 결과). 로컬 CV는 §5에서 별도.

| # | 행동(무엇을 바꿨나) | 제출물 | weights (lin,stk,enc) | enc 지분 | LB | 델타 |
|---|---|---|---|---:|---:|---:|
| 0 | **베이스라인** (`current_prompt`만 TF-IDF+LogReg, `history`·`session_meta` 버림) | 공개 baseline | — | — | ~0.43 (local group-split) | 기준 |
| 1 | **linear 구축**: 입력 3필드 전부 사용, 피처셋 `E_+seq`, `LinearSVC(C=0.1)` | `submit/` | 단독 | — | ~0.673 (성분 solo) | — |
| 2 | **stacker 구축**: work2 AAR 스태커(4개 SGD view + transition prior) | work2 | 단독 | — | ~0.671 (성분 solo) | — |
| 3 | **encoder 파인튜닝**: `intfloat/multilingual-e5-base` 파인튜닝 | colab | 단독 | — | ~0.701 (성분 solo) | — |
| 4 | **첫 3-way 앙상블**(v1 인코더, uniform) | base zip | 1,1,1 | 0.33 | **0.7130** | 이종 blend 시작 |
| 5 | **인코더 교체**: v1 → **v2 s42**(full 70k 학습), uniform 유지 | `ensemble_v2s42_submit.zip` | 1,1,1 | 0.33 | **0.7190** | +0.0060 |
| 6 | **인코더 지분↑**: weights `[1,1,1.5]` | w115 zip | 1,1,1.5 | 0.43 | **0.7200** | +0.0010 |
| 7 | **인코더 지분 더↑**: weights `[1,1,2]` 🏆 | `ensemble_v2s42_w112_submit.zip` | **1,1,2** | **0.50** | **0.7208** | +0.0008 |

**가중 곡선 (LB 실측)**: enc 지분 `0.33 → 0.43 → 0.50` = `0.7190 → 0.7200 → 0.7208`. 아직 우상향이나 체감(diminishing). 다음 점 `1,1,2.5`(지분 0.556)는 로컬 그리드에서 하락 시작점이라 후순위.

### 폐기된 제출(❌ 재시도 금지)

| 제출물 | weights / 변경 | LB | 왜 폐기 |
|---|---|---:|---|
| soup2 (seed 앙상블) | enc = s42+s7 가중치 평균 | **0.697** | seed별 분류 head 초기화가 달라 **다른 basin** 가중치 평균 = 파괴적 간섭. soup3도 동일 이유로 중단. |
| calib_v1 | enc block에 `softmax(log p /1.34 + bias)` | **0.7169** | 정직 holdout에선 +0.005였으나 LB에선 −0.002. train 클래스 분포에 피팅된 bias가 **분포 이동에 취약** = 비전이. |

---

## 2. w112가 무엇으로 이루어졌나 (정확한 구성)

**w112 = 3개 성분의 14-class 확률을 클래스명으로 정렬 → 가중 평균 → argmax.**
추론 코드는 `ensemble_submit_draft/script_3way.py` (제출 zip 안에서 `script.py`가 됨).

```
최종확률(row) = ( 1·P_linear + 1·P_stacker + 2·P_encoder ) / 4
예측 = argmax(최종확률)
```

가중치 출처: `model/weights.json = {"weights":[1.0,1.0,2.0]}` (없으면 uniform, `ENS_WEIGHTS` env가 최우선).

### 성분 A — linear (`model/linear/model.pkl`, 8.35 MB)

- **피처셋 `E_+seq`** (train·추론 동일 계약, `features.py`):
  - word + char **TF-IDF** on `current_prompt`
  - `hist_text` (history의 user 발화·action 텍스트)
  - **action 시퀀스 n-gram**(마르코프) + `last_action` / `last2_action`
  - `session_meta` (budget_tokens, git_dirty, CI status, turn_index, open_files 등)
  - 구문 regex 플래그(glob/grep/read/list 분기)
- **분류기**: `LinearSVC(C=0.1, loss=squared_hinge, class_weight="balanced")`
- **확률 경로**: `softmax(decision_function + class_bias)` — pkl에 `class_bias` 벡터 동봉(3-way에서 softmax 전에 더함).
- 성분 solo ≈ **0.673 LB**.

### 성분 B — stacker (`model/stacker/aar_config.json` + `aar_models.joblib`, 47 MB)

- work2 저장소의 **AAR 스태커**. `aar_infer.py`가 work2 `script.py`를 **verbatim 벤더**(자기완결).
- **4개 sub-view의 가중 blend** (config 실측):
  | sub-component | kind | view | weight |
  |---|---|---|---:|
  | prompt_context_sgd | text | prompt+context | 0.600 |
  | prompt_sgd | text | prompt | 0.200 |
  | action_sgd | text | action seq | 0.120 |
  | transition_prior | transition | Markov | 0.080 |
- `use_bias=false`, `use_stacker=true`. 내부 holdout `final_valid_macro_f1 = 0.7098`, solo ≈ **0.671 LB**.

### 성분 C — encoder block (`model/encoder/`, e5-base v2 s42)

- **베이스**: `intfloat/multilingual-e5-base` (MIT, `xlm-roberta-base` 초기화 → `XLMRobertaForSequenceClassification` 14-class head).
- **파인튜닝**: 대회 제공 `train.jsonl`(**full 70k**) + `train_labels.csv`만, **seed 42** (= "v2 s42"). 외부 데이터 무추가.
- `max_len=384`. 텍스트 계약 = `script_3way.py`의 `serialize()` (변경 시 재학습·재검증 필수 — 안 그러면 조용한 오답).
- 로컬 원본 = `colab_out/enc_v2_s42/model/model.safetensors` (~1.1 GB fp32) → **패키징 시 fp16 복사(~547 MB)**로 1GB 제약 통과.
- encoder block은 `encoder/`, `encoder_2/`… 여러 개면 **uniform 평균**. w112는 **단일 인코더**.
- 성분 solo ≈ **0.701 LB** (3개 중 최강). 이종 blend가 여기에 +0.02 추가.

### 크기·시간 예산 (제약 = ≤1GB / 추론 ≤10분 / T4 16GB, 오프라인)

- 크기: linear 8 MiB + stacker 92 MiB + encoder fp16 547 MiB ≈ **647 MiB** < 1 GB ✅
- 시간: encoder 배치추론(T4 fp16) 수십초 + linear/stacker TF-IDF 수분 → 여유 ✅
- 오프라인: `HF_HUB_OFFLINE=1`, 가중치·토크나이저 전부 `model/` 동봉, 원격 호출 0 ✅

---

## 3. w112 submit.zip 재현 (빌드 → 검증)

⚠️ `ensemble_v2s42_w112_submit.zip` 원본은 **07-04 디스크 정리로 삭제됨**. 아래로 **정확히 재빌드** 가능(`ensemble_submit_draft/`에 인코더 가중치 뺀 전부가 있음).

### 제출 zip 구조

```
submit.zip
├── script.py            # = script_3way.py 를 빌더가 복사
├── requirements.txt     # transformers>=4.51 (기본 패키지 재핀 금지)
└── model/
    ├── linear/model.pkl
    ├── stacker/{aar_config.json, aar_models.joblib}
    ├── encoder/{config.json, model.safetensors(fp16), tokenizer.json, ...}
    └── weights.json     # {"weights":[1.0,1.0,2.0]}  ← 이게 w112의 정체
```

### 빌드 명령 (조율 세션 소유 lane)

```powershell
# 빌드 전: 800MB+ python 프로세스 없는지 확인 (RAM 7.8GB, 동시 1개 원칙)
tasklist /FI "IMAGENAME eq python.exe" /FO CSV

.\.venv\Scripts\python.exe src\package_ensemble.py `
  --rows 2000 `
  --encoder-model colab_out\enc_v2_s42\model `   # fp16 dir 없으면 fp32 원본 지정 → 빌더가 fp16 변환
  --weights 1,1,2 `
  --out-zip ensemble_v2s42_w112_submit.zip
```

### 검증 (제출·push 전 필수)

- `package_and_verify.py` / `package_ensemble.py`의 오프라인 재현: 스모크5 + 스케일3k + 30k 시간추정.
- `check_ensemble_package.py`로 구조/인코더수/가중/arch 확인 (하나라도 실패 시 제출 금지).
- **`dacon-verifier` 서브에이전트 자동 호출**(제출·패키징·push 직전, CLAUDE.md 9절).
- 게이트: **"수정 전 LB(0.7208)"보다 오를 때만** push/제출 채택. 하락·동일이면 금지.

---

## 4. "공유 요청 3종" 매핑 (검증 하네스가 받을 것)

받는 쪽이 요청한 우선순위 항목 → 우리 저장소 위치:

**① 제출했던 submit.zip 그대로 (script.py + model/ + requirements.txt)**
- 원본 zip은 삭제됨. **`ensemble_submit_draft/`가 인코더 가중치만 뺀 완전한 draft**:
  - `script_3way.py`(=zip의 script.py) · `aar_infer.py` · `features.py` · `requirements.txt`
  - `model/linear/model.pkl` · `model/stacker/*` · `model/weights.json [1,1,2]`
  - 인코더는 `colab_out/enc_v2_s42/model/` (별도, ~1.1GB fp32). → §3 빌드 명령으로 zip 완성.
- 소형 성분만 먼저 넘길 땐 linear+stacker+weights.json(≈55MB)만으로도 blend 로직 검증 가능.

**② 학습 코드/노트북 (피처·모델·하이퍼파라미터)**
- linear: `src/train_final.py` + `src/features.py`(`FEATURE_SETS` E_+seq) — `LinearSVC(C=0.1, class_weight=balanced)`.
- stacker: work2 저장소 AAR(`aar_config.json`에 view 가중 명시, 벤더본 `aar_infer.py`).
- encoder: `colab/encoder_v2_full.py` + `encoder_finetune.py` (Colab, e5-base full-70k s42, max_len 384).

**③ 아는 정보 (로컬 CV·제출 이력·폐기한 것)**
- 제출 이력·LB: §1 표. 로컬 CV vs LB 갭: §5. 폐기 목록: §6.

---

## 5. 로컬 CV vs LB (가족별 갭 — 반드시 할인해서 읽기)

로컬 세션 group-split/holdout 점수는 LB보다 **높게** 나오며 가족마다 갭이 다르다. 로컬 그리드는 **방향·랭킹 선택기**일 뿐, LB 보장 아님.

| 성분 가족 | 로컬 → LB 할인 |
|---|---:|
| linear | −0.002 |
| **encoder (base)** | **−0.015** |
| encoder (small/e8) | −0.019 |
| stacker | −0.033 |

- **실증된 함정**: 로컬 프록시(v1 인코더)는 **최적 enc 지분을 LB보다 낮게** 잡는다. 로컬 그리드 최적은 `1,1,1`(지분 0.33)인데 **LB 최적은 `1,1,2`(지분 0.50)**. → 그리드는 "지분 환산 + 오른쪽 보정"으로 읽을 것 (`/dacon-wgrid`).
- 검증 프로토콜: `StratifiedGroupKFold`, 그룹키 = `id`에서 `-step_\d+$` 제거(9,429 세션). 중요한 판정은 3-fold CV. **per-class F1 항상 확인** — 약한 곳은 탐색 계열(read_file·grep_search·list_directory·glob_pattern), 이미 1.0인 respond_only·write_file은 건드리지 말 것.

---

## 6. 폐기 확정 (❌ 재시도 금지 — 검증 후 버린 것)

| 레버 | 결과 | 왜 |
|---|---|---|
| seed soup (soup2/soup3) | LB 0.697 | seed head 초기화 상이 → 파괴적 간섭 |
| calib_v1 (enc T=1.34+bias) | LB 0.7169 | 정직 +0.005가 LB −0.002 비전이 |
| flat 피처 추가 F~W | 이득 없음 | — |
| stacker 변형 4종 | 이득 없음 | — |
| cascade base | +0.0019 (노이즈) | — |
| max_len 512 추론 | ±0.000 | — |
| 환각 피처(비저항/토양/관개, When2Tool 히든스테이트) | 스키마에 없음 | 우리 입력에 존재하지 않는 필드 |
| 옛 회차(2025) 전략(확률출력·ROC-AUC·MIL) | 대회 지표 불일치 | Macro-F1(14-class)엔 무관 |

---

## 7. 파일·아티팩트 지도 (어디에 뭐가 있나)

| 무엇 | 경로 | 비고 |
|---|---|---|
| 3-way 추론 코드 | `ensemble_submit_draft/script_3way.py` | zip의 script.py |
| w112 가중치 | `ensemble_submit_draft/model/weights.json` | `[1,1,2]` = 챔피언 정체 |
| linear 아티팩트 | `submit/model/model.pkl`, `ensemble_submit_draft/model/linear/model.pkl` | E_+seq, LinearSVC C=0.1 |
| linear 피처(계약) | `src/features.py` = `submit/features.py` = `ensemble_submit_draft/features.py` | 셋 항상 동기화 |
| stacker 아티팩트 | `ensemble_submit_draft/model/stacker/{aar_config.json,aar_models.joblib}` | work2 원본 |
| encoder 원본(fp32) | `colab_out/enc_v2_s42/model/` | ~1.1GB, 삭제 금지 |
| 빌더 | `src/package_ensemble.py` (앙상블), `src/package_and_verify.py` (linear) | 오프라인 재현 내장 |
| 정직 OOF (유일본) | `colab_out/oof/` | linear·stacker 정직 확률, **삭제 절대 금지** |
| 검증 프로토콜 | `/dacon-score`, `/dacon-submit`, `/dacon-wgrid` | 프로젝트 커맨드 |
| soup s7 유일 흔적 | `ensemble_soup2_w112_submit.zip`, `colab_out/enc_v2_soup2_fp16/` | 폐기됐지만 s7 유일본 |

### 협업 저장소 (push 게이트 준수)

- repo: `https://github.com/wnstj0128-sudo/dacon-agent-action-api-boost.git`
- **push는 LB(0.7208)보다 오를 때만** + 오프라인 재현 통과 + `git remote -v` 확인 후.

---

## 8. 다음 (07-04 저녁 기준 우선순위)

1. **시뮬레이터 포렌식(1순위)**: train이 `sess_sim_*` 시뮬레이션 산출물 → 정책 역공학·(state→action) 결정성 분석. top-12 컷(0.77426)까지 −0.0535는 미세 앙상블로 불가, 구조적 신호 필요.
2. 버킷-게이트 블렌드(`history_presence`) 프로브 — 정보 획득용.
3. 4-way(enc block = base+small 확률 평균) — v2 홀드아웃 4파일 도착 후 판정.
4. 후순위: `1,1,2.5`, 동료 4번째 성분.
