# task1 — char-ngram 이질 성분 (4번째 성분 후보, CPU 전용)

## 컨텍스트

DACON 236694: AI 에이전트 다음 행동 14클래스 분류, Macro-F1. 현 최고 blend = linear + AAR stacker + e5-base 인코더 3-way (LB 0.71884, 팀 최고 0.7242, 컷 0.77665).

**왜 이 작업인가**: 오늘(07-05) 가중치 튜닝 계열 레인이 전부 LB 실측으로 폐기됐다 (버킷 가중 −0.0061, e5-small 전 형태 마이너스, enc 지분 조정 전면 금지). 남은 방향은 **이질 성분 추가**뿐이다. 로컬 LB 시뮬레이션 리그는 "성분 추가/제거" 축에서만 신뢰된다(3-way 재현 오차 0.0016) — char-ngram linear는 기존 linear(word 피처)와 상관이 낮을 수 있는 CPU 학습 가능 성분이다.

⚠️ 재시도 금지 유의: "flat 피처 추가 F~W"는 기존 linear에 피처를 **더한** 실험이었고, 이 작업은 **독립 성분**을 새로 만드는 것이라 금지 대상이 아니다. 단 blend 가중 튜닝(특히 enc 지분 변화)은 금지 — 성분 추가 여부만 판정한다.

## 목표 / 완료 조건 (DoD)

1. `scripts/components/char_svm/train_oof.py` — 70k 전체에 대한 정직 3-fold OOF 확률 생성 (세션 프리픽스 StratifiedGroupKFold)
2. OOF solo Macro-F1 보고 (기대: 0.55~0.65면 정상 범위)
3. **리그 add-test**: 3-way flat [1,1,2] 위에 4번째 성분으로 w4 ∈ {0.25, 0.5, 0.75, 1.0} 추가 시 리그 점수 표. 베이스라인 0.71726 대비 델타 명시
4. 판정 기준 명기: **어떤 w4에서든 +0.002 이상이면 PASS**(아침에 LB 게이트 후보), 전부 +0.002 미만이면 FAIL(폐기) — 애매하게 긍정으로 포장하지 말 것
5. full-train 모델 아티팩트 저장 (`night_out/task1/char_svm_full.pkl`, PASS 시 제출 조립용)
6. 리포트 `context/night/2026-07-06/task1_report.md` (방법·수치·판정·재현 커맨드)
7. `context/night/2026-07-06/task1.DONE` 생성 (판정 요약 3줄 포함) + 최종 커밋

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl` + `data\train_labels.csv` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)
- 리그 평가행: `C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz` (ids/probs/y_true/actions, 9,969행, 인코더 85% 프록시 확률. actions는 알파벳순)
- linear·stacker 정직 OOF: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` — `linear_probs.npy`, `stacker_probs.npy`, `classes.json`(⚠️ 알파벳순 아님 — 재정렬 필수), `row_ids.json`
- 리그 조인 코드 (검증된 패턴 — 그대로 쓸 것):
  ```python
  import json, numpy as np
  enc = np.load(r'C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_base.npz', allow_pickle=True)
  eids = [str(x) for x in enc['ids']]; eprobs = enc['probs']
  y = [str(x) for x in enc['y_true']]; acts = [str(a) for a in enc['actions']]
  oof = r'C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04'
  classes = json.load(open(oof+'/classes.json')); row_ids = json.load(open(oof+'/row_ids.json'))
  col = [classes.index(a) for a in acts]; idx = {r:i for i,r in enumerate(row_ids)}
  rows = [idx[i] for i in eids]
  lin = np.load(oof+'/linear_probs.npy')[:,col][rows]
  stk = np.load(oof+'/stacker_probs.npy')[:,col][rows]
  # 3-way 리그 베이스 = (lin + stk + 2*eprobs)/4 → macro-F1 0.71726 이 나와야 조인이 맞은 것 (assert 필수)
  ```

## 금지

- 워크트리 밖(메인 리포 작업트리 `C:\dev\2026-AI-DACON`, 팀 리포) 수정 금지 — 읽기만
- `git push` 금지, 수동 zip/제출 금지, 네트워크 코드 금지
- blend 가중 튜닝 금지 (w4 그리드 4점은 add-test 판정용으로만 — enc 지분(2/4)은 절대 건드리지 말 것)
- 재시도 금지 테이블(`context/experiments.md`) 위반 금지

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-06/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터
2. **fold 하나 끝날 때마다** OOF 부분 저장(`night_out/task1/oof_fold{k}.npz`) + PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) + git commit — 세션이 언제 죽어도 fold 단위로 재개 가능해야 한다
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋

## 작업 내용

1. **직렬화**: 자기완결 함수로 샘플→텍스트. current_prompt 전문 + history를 `act:<name>` 토큰 나열 + session_meta 주요 필드 문자열화. 기존 linear(word TF-IDF)와 **다른 view**가 되도록 char 정보를 살린다.
2. **모델**: `TfidfVectorizer(analyzer='char_wb', ngram_range=(2,5), max_features=300_000, sublinear_tf=True)` + `LinearSVC(C=0.1)` (확률 = decision_function softmax, script.py linear와 동일 방식). 70k×3fold가 메모리/시간 초과면 max_features를 150k로, 그래도 안 되면 `SGDClassifier(loss='modified_huber')`로 강등하고 리포트에 명기.
3. **CV**: StratifiedGroupKFold 3-fold, 그룹키 = id에서 `-step_\d+$` 제거. (누수 절대 금지 — 같은 세션이 train/valid에 갈라지면 안 됨)
4. **평가**: OOF solo macro-F1 → 리그 조인(위 코드) → add-test 표 → 판정.
5. **부가 진단**: char 성분과 기존 linear의 예측 불일치율(diversity 지표), per-class F1 (특히 탐색 4클래스 read_file/grep_search/list_directory/glob_pattern — 여기가 blend 약점).
6. 리포트 작성 → DONE.
