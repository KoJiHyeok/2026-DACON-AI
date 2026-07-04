# 2026 DACON — 디지털 경진대회 AI 부문 : 워크플로우

## Overview : <https://dacon.io/competitions/official/236694/overview/description>

* **대회 소개:** AI 코딩 에이전트 세션의 특정 시점 상태를 보고, 에이전트가 다음에 수행할 행동을 **14개 클래스 중 하나로 분류**하는 NLP·분류 챌린지. 총상금 1,220만 원.
* **대회 기간:** 예선 2026.07.01(수) ~ **2026.07.15(수) 09:59** 온라인 진행. 본선 발표평가·시상식 2026.08.11(화).
* **참가 조건:** 동일 대학 소속 학부생(재학생·휴학생) 1~5인 팀. 졸업생 불가, 대학당 최대 7팀.
* **평가 방식:** **Macro-F1** (14클래스). 예선은 리더보드 Private Score로 상위 12팀 본선 진출. 일일 제출 10회.
* **제출 방식:** 예측 CSV가 아닌 **코드 제출 대회** — 추론 코드(script.py) + 학습된 모델을 submit.zip으로 제출하면 서버가 오프라인 환경에서 실행.

---

## Steps

### 1. [요구사항] 대회 룰 파악 ✅ (2026.07.04 완료)

1. 대회 룰 관련 링크
   1. <https://dacon.io/competitions/official/236694/codeshare> — 코드공유 목록 (주기적으로 새 글 확인)
   2. <https://dacon.io/competitions/official/236694/codeshare/14026> — 공식 베이스라인 **학습** (TF-IDF + LogReg)
   3. <https://dacon.io/competitions/official/236694/codeshare/14025> — 공식 베이스라인 **추론**
   4. <https://dacon.io/competitions/official/236694/talkboard/416930> — 배포용 데이터 명세 (필드 구조 원문)
2. 제출 제약 (submit.zip = `model/` + `script.py` + `requirements.txt`)
   1. 추론 코드 실행 시간 ≤ 10분
   2. 패키지(라이브러리) 설치 시간 ≤ 10분
   3. 제출 파일 용량 ≤ 1GB
   4. **오프라인 환경 실행** (패키지 설치 외 인터넷 연결 불가 → API 호출·모델 다운로드 불가, 모든 가중치를 zip에 동봉)
   5. T4 GPU(16GB VRAM), 3 vCPU, 12GB RAM / **Python만 허용**
3. 규칙 요점
   1. 외부 데이터·사전학습 모델·API **사용 가능** (법적 제한 없음 + 출처 명시 조건) → 학습 단계에서는 LLM API 활용 가능, 추론만 오프라인
   2. 본선 진출 시 **코드 검증(7/24)** 있음 → 학습 재현성(시드·버전 고정)이 필수
4. 현재 대회 상태 state (2026.07.04 기준 — 계속 변하므로 확인 시점 기록)
   1. 1등 : 0.79015 / 12등 (본선 커트라인) : 0.77585
   2. → 1등~커트라인 갭이 겨우 **0.014p**. 리더보드가 극도로 촘촘함 = 커트라인 통과가 곧 상위권 수준. 목표는 **0.78+ 도달**이며, 소수점 셋째 자리 개선(피처 하나, threshold 튜닝)이 순위를 가른다

### 2. [문제 정의 및 리서치] Problem Understanding ↔ Data Understanding

> Product 해커톤의 Market Research 자리를 이 대회에서는 **EDA(데이터 리서치)** 가 대신한다.

1. **Problem Statement (한 문장 정의)**
   * "`current_prompt`(현재 발화) + `history`(직전 0~12턴 대화·행동) + `session_meta`(요금제·토큰예산·워크스페이스 상태)를 입력으로, 다음 행동 14클래스를 Macro-F1 최적화로 분류한다."
2. **EDA — 반드시 확인할 것** (train.jsonl 70,000건 + train_labels.csv)
   1. 클래스 분포 → 불균형 정도 파악 (Macro-F1이라 **희소 클래스가 점수를 좌우**)
   2. 행동 전이 행렬: history 마지막 행동 → 정답 행동 (예: `edit_file` 다음 `run_tests` 확률)
   3. `current_prompt` 텍스트 패턴: 언어 비율(ko/en), 길이, 클래스별 키워드
   4. `session_meta`와 라벨 상관: `last_ci_status=failed` ↔ `run_tests`/`edit_file`, `budget_tokens_remaining` ↔ `respond_only`, `git_dirty` 등
   5. history 길이(0~12)별 분포 — history 없는 샘플(첫 턴)은 별도 패턴일 가능성
3. **Research → Context Building**
   1. 유사 문제 기법 리서치: 텍스트+정형 혼합 분류, class imbalance 대응(class weight, focal loss, threshold 튜닝), 경량 다국어 인코더(DeBERTa-v3-small, multilingual-e5-small 등)
   2. 도구: NotebookLM(문서 정리), Claude Code(EDA·구현), **Codex + insane_search(메인 리서치)**
   3. 리서치 결과는 `docs/research.md`에 축적 → 이후 콘텍스트 주입 재료로 사용
4. **Opportunity Sizing → 점수 갭 분석으로 대체**
   * 베이스라인(TF-IDF+LogReg) 점수 vs 리더보드 상위권 점수의 갭을 보고, 어느 피처/모델 계층에 개선 여지가 큰지 추정. 리더보드를 "시장"으로 읽는다.

### 3. [솔루션 & 실험 설계]

1. **Problem Solutioning — 후보 솔루션 계층 (Tier)**
   * **Tier 0:** 공식 베이스라인 재현 + 제출 → 파이프라인 검증이 목적 (점수는 무관)
   * **Tier 1:** 피처 엔지니어링 + GBDT(LightGBM/XGBoost) — 전이 피처(직전 행동 n-gram), 메타 피처, TF-IDF/문자 n-gram
   * **Tier 2:** 경량 사전학습 인코더 파인튜닝 — prompt+history를 직렬화해 입력, 메타는 별도 헤드 또는 텍스트에 병기
   * **Tier 3:** Tier 1 + Tier 2 앙상블 + 클래스별 threshold 튜닝(Macro-F1 직접 최적화)
2. **Solution Prioritization**
   * 순서 고정: **제출 파이프라인 완성(Tier 0) → 빠른 반복이 가능한 Tier 1 → Tier 2/3** (커트라인 0.776 수준이면 Tier 2/3까지 사실상 필수)
   * 판단 기준: (예상 점수 상승) ÷ (구현+학습 시간). 마감 D-3부터는 새 모델 금지, 튜닝·앙상블만.
3. **Solution Evaluation / Measurement Plan**
   1. **로컬 CV 체계 먼저 구축**: `id`의 세션 프리픽스(`sess_...`) 기준 **GroupKFold** — 같은 세션의 step이 train/valid에 갈라지면 누수
   2. 로컬 Macro-F1과 리더보드 점수의 상관을 초기에 확인 → 이후 로컬 CV로 의사결정, 제출은 검증용
   3. 제출 예산 관리: 일 10회. 초반엔 파이프라인 검증 1~2회, 이후 유의미한 개선만 제출. **Private Score 기준이므로 public 과적합 경계**
   4. 실험 로그: `docs/experiments.md`에 (가설 → 변경점 → 로컬 CV → 리더보드) 기록

### 4. [엔지니어링 / 모델 구현]

1. **콘텍스트 주입** — `CLAUDE.md` 작성: 대회 룰·제약·데이터 명세·현재 최고 점수·실험 규칙을 담아 에이전트가 항상 대회 맥락 위에서 작업하게 한다.
2. **Scaffolding**
   1. File structure: ✅ 구축 완료 (2026.07.04)
      ```
      2026-AI-DACON/
      ├── CLAUDE.md            # 대회 콘텍스트 (에이전트용)
      ├── PLAN.md              # 이 문서
      ├── data/                # 대회 데이터 (git 제외)
      ├── notebooks/           # EDA
      ├── src/
      │   ├── features.py      # 피처 추출 (학습·추론 공용 — 단일 소스)
      │   ├── train.py         # 학습 → submit/model/ 산출
      │   └── infer.py         # script.py의 원형
      ├── submit/              # 제출 스테이징: script.py + requirements.txt + model/ (대회 규정 구조)
      ├── tests/               # 단위 테스트 (피처 불변식)
      ├── docs/                # research.md, experiments.md, validation.md
      └── scripts/
          ├── make_submit.py       # submit/ → submit.zip 패키징 + 검증 자동 실행
          └── validate_submit.py   # 대회 기준 시뮬레이션 (구조·오프라인·시간·출력 형식)
      ```
   2. README / CLAUDE.md 초안 작성 ✅
   3. 서버 실행 규약 (baseline_submit.zip에서 확인): 서버가 `./data/test.jsonl` + `./data/sample_submission.csv` 제공 → script.py가 `./output/submission.csv` 생성
3. **학습/추론 분리 원칙**: 피처 코드는 학습·추론이 같은 모듈을 import (불일치 = 조용한 점수 하락). script.py는 `model/`만 읽어 예측.
4. **오프라인 제약 체크리스트** (제출 전 매번)
   - [ ] script.py에 네트워크 호출 없음 (HF `from_pretrained`는 로컬 경로 + `local_files_only=True`)
   - [ ] requirements.txt 설치 10분 이내 (버전 고정)
   - [ ] zip ≤ 1GB / 추론 ≤ 10분 (T4보다 느린 환경 가정, 로컬에서 시간 측정)
   - [ ] sample_submission.csv와 동일한 형식(id, action)·행 수 출력
5. **재현성** (본선 코드 검증 대비): 시드 고정, 패키지 버전 잠금, 학습 스크립트 원커맨드 실행 가능하게 유지

### 5. [확장 및 자동화] Loop Engineering

> 나의 한계치를 agent / automation으로 끌어올리는 방법 설계 — 이 대회에서는 **실험 루프의 회전 속도**가 곧 한계치다.

1. **실험 루프 자동화**: 설정 파일(config) 하나 바꾸면 학습→CV 평가→experiments.md 기록까지 원커맨드로
2. **Claude Code agent 활용**
   1. EDA·피처 아이디어 탐색을 병렬 에이전트로 fan-out
   2. 코드 리뷰(/code-review)로 피처 누수·학습/추론 불일치 점검
   3. 반복 잡무(패키징, 스모크 테스트, 로그 정리)는 스크립트화 후 에이전트에 위임
3. **제출 자동화**: `make_submit.py` = 패키징 + 오프라인 시뮬레이션(네트워크 차단 + 시간 측정 + 출력 형식 검증) → 통과해야 제출

---

## 일정 (예선 마감 07.15 09:59 역산)

| 날짜 | 목표 |
|---|---|
| 07.04(금)~07.05(토) | 데이터 다운로드, EDA, Tier 0 베이스라인 제출 (파이프라인 검증) |
| 07.06(일)~07.09(목) | CV 체계 확립, Tier 1 피처엔지니어링 + GBDT, 매일 1~2회 제출 |
| 07.09(목)~07.12(일) | Tier 2 경량 인코더 파인튜닝 (오프라인 패키징 포함) |
| 07.12(일)~07.14(화) | Tier 3 앙상블 + threshold 튜닝, 최종 후보 2~3개 선정 |
| 07.14(화)~07.15(수) 오전 | **최종 제출 확정 (마감 09:59 주의 — 전날 밤까지 완료 권장)** |
| 07.20(월) | 본선 자료 제출 마감 (발표자료·코드 정리) |
| 07.24(금) | 코드 검증 → 재현성 확보 상태 유지 |
