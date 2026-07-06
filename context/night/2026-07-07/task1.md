# task1 — enc_block_weights 회귀 테스트 + 4-way 리그 재구축 (블록 비율 미세 그리드 + AU α 재그리드)

## 컨텍스트

DACON 236694 (AI 에이전트 다음 행동 14클래스, Macro-F1). 오늘 4-way 앙상블(linear + stacker + e5 1.2 + mBERT 0.8 블록, soft-AU α=0.9)이 LB **0.7467**로 승격됐다(제출 #7, 리그 예측 +0.0053 → 실측 +0.0067). 이제 리그의 기준선이 3-way가 아니라 4-way가 되어야 하고, 4-way 위에서 (a) 블록 비율이 아직 최적인지 (b) AU α=0.9가 3-way에서 튜닝된 값이라 4-way에선 최적이 달라졌는지 재판정이 필요하다. 또 오늘 제출 검증에서 enc_block_weights.json의 UTF-8 BOM으로 script.py가 크래시하는 사고가 있었다 — G1(pytest)에 이 버그 클래스를 잡는 회귀 테스트가 없어서 G4까지 가서야 드러났다. reviewer가 회귀 테스트 추가를 권고했다.

## 목표 / 완료 조건 (DoD)

1. `tests/test_enc_block_weights.py` 추가 — pytest 통과. 커버: ① BOM 포함 JSON 파일도 정상 파싱(utf-8-sig) ② 가중치 개수 != 인코더 수 → ValueError ③ 음수 가중치 → ValueError ④ 파일·env 둘 다 없으면 None(uniform 폴백) ⑤ env가 파일보다 우선.
2. `scripts/league4/rebuild.py` — 4-way 리그 기준선 재구축: 4-way blend 값과 **4-way + soft-AU α=0.9** 값(= 현 제출 #7의 리그 미러)을 산출·기록.
3. `scripts/league4/grid_block.py` — 블록 비율 미세 그리드 (mbert 가중 x ∈ 0.60~1.00, step 0.05, e5 = 2−x). 각 점: 전체 / 비AU 실효 / soft-AU 최종 3개 값 + 반반(half1/half2) 안정성.
4. `scripts/league4/grid_alpha.py` — 4-way blend 기준 AU α 재그리드 (α ∈ 0.70~1.00, step 0.05). AU 모델은 char_wb(3-5) 120k TF-IDF + LinearSVC C=1.0 (scripts/au2/task4_grid.py의 격리 규약 그대로: 학습 = **비holdout** AU 행만).
5. `context/night/2026-07-07/report_league4.md` — 결과 표 + 판정(현 설정 [1.2,0.8]·α0.9 대비 델타, 게이트 +0.005 통과 후보 유무). **파일명을 task로 시작하지 말 것** (러너 glob 함정).
6. `context/night/2026-07-07/task1.DONE` 생성 (한 줄 요약 포함) + 최종 커밋.

## 재료 (절대 경로 — 워크트리에는 gitignore된 것이 없다)

- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)
- 데이터: `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv` (읽기 전용)
- 평가 npz: `context/night/2026-07-05/holdout_base.npz` (커밋됨 — 워크트리 상대경로 OK. ids/probs=e5 프록시/y_true/actions 알파벳순, 9,969행)
- mBERT holdout: `C:\dev\2026-AI-DACON\colab_out\holdout_mbert.npz` (gitignore — 절대 경로로 읽기)
- OOF: `C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04\` (gitignore — 절대 경로): `linear_probs.npy`, `stacker_probs.npy`, `classes.json`, `row_ids.json`
- AU 그리드 전례: `scripts/au2/task4_grid.py` (커밋됨 — 격리 규약·모델 스펙 참조)
- 테스트 대상: `submit/script.py`의 `enc_block_weights()` (236행 부근 — import 시 main() 실행 여부를 먼저 확인하고, 실행된다면 함수를 subprocess 스텁 실행 또는 소스 exec 추출로 테스트할 것. 기존 tests/ 패턴 참조)

### 조인 레시피 (조용한 오답 방지 — 반드시 이 순서)

```python
import json, numpy as np
enc = np.load(r'context/night/2026-07-05/holdout_base.npz', allow_pickle=True)
eids = [str(x) for x in enc['ids']]; e5 = enc['probs'].astype(np.float64)
acts = [str(a) for a in enc['actions']]  # 알파벳순
oof = r'C:\dev\2026-AI-DACON\artifacts\oof\oof_rebuild_2026_07_04'
classes = json.load(open(oof + r'\classes.json'))  # ★ 알파벳순 아님 — 재정렬 필수
col = [classes.index(a) for a in acts]
idx = {r: i for i, r in enumerate(json.load(open(oof + r'\row_ids.json')))}
rows = [idx[i] for i in eids]
lin = np.load(oof + r'\linear_probs.npy')[:, col][rows].astype(np.float64)
stk = np.load(oof + r'\stacker_probs.npy')[:, col][rows].astype(np.float64)
# mbert npz도 id 조인 + actions 재정렬 + y_true 일치 assert (scripts/au2 및 아래 sanity 참조)
```

**Sanity asserts (하나라도 깨지면 조인 버그 — 진행 금지)**:
- 3-way `(lin+stk+2·e5)/4` macro-F1 = **0.71726** (±0.0005)
- 4-way `(lin+stk+1.2·e5+0.8·mbert)/4` = **0.72255** (±0.0005)

## 금지

- 워크트리 밖(메인 리포 작업트리·팀 리포) 수정 금지, `git push` 금지, 수동 zip·제출 금지
- 제출물(submit/)에 손대지 말 것 — 이 태스크는 tests/와 scripts/league4/만 만든다
- 폐기 목록(experiments.md '재시도 금지' 테이블) 위반 금지: enc **지분**(3슬롯 가중) 조정 금지 — 이 태스크의 그리드는 블록 **내부** 비율만 (총가중 2 고정)
- GPU 사용 금지 (전부 사전계산 확률로 CPU 연산)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-07/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(테스트 완성 / rebuild / grid_block / grid_alpha / 리포트)마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit**
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋

## 작업 내용

1. **(워밍업) 회귀 테스트**: `tests/test_enc_block_weights.py`. BOM 케이스는 `tmp_path`에 `codecs.BOM_UTF8 + b'{"weights": [1.2, 0.8]}'`를 써서 검증. pytest 통과 확인 후 커밋.
2. **rebuild**: 조인 → sanity 2개 통과 → soft-AU 적용(sess_au 프리픽스 행에 `0.9·P_au + 0.1·P_blend`; AU 모델은 비holdout AU 행으로 학습 — task4_grid.py 재사용) → 4-way+soft-AU 리그 값 산출. 이것이 새 기준선 B4.
3. **grid_block**: x ∈ {0.60, 0.65, …, 1.00}에 대해 blend를 재계산하고 soft-AU까지 적용한 최종값으로 B4 대비 델타 표. 최고점이 x=0.8이 아니면 반반 안정성(seed42 permutation half)까지 확인.
4. **grid_alpha**: 고정 블록 [1.2, 0.8]에서 α ∈ {0.70, …, 1.00} 스캔, B4 대비 델타 표.
5. **리포트**: 표 3개 + 결론. 판정 기준 명기 — 신규 제출 후보는 B4 대비 **+0.005 이상**, +0.002~0.005는 '보고만'. 블록 비율·α는 각각 전이 검증된 축(오늘 오차 0.0014, 라우팅 오차 0.0004)이지만 최종 승격은 LB 게이트.
