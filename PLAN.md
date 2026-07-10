# 2026 DACON — 디지털 경진대회 AI 부문 : 워크플로우

## Overview : <https://dacon.io/competitions/official/236694/overview/description>

* **대회 소개:** AI 코딩 에이전트 세션의 특정 시점 상태를 보고, 에이전트가 다음에 수행할 행동을 **14개 클래스 중 하나로 분류**하는 NLP·분류 챌린지. 총상금 1,220만 원.
* **대회 기간:** 예선 2026.07.01(수) ~ **2026.07.15(수) 09:59** 온라인 진행. **상위 12팀 재현 코드 제출 마감 2026.07.20(월) 10:00** (공식 rules 확인, 2026-07-10 / D-012). 본선 발표평가·시상식 2026.08.11(화).
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
   2. 본선 진출 시 코드 검증 있음 — **재현 코드(학습 코드+자원 출처) 제출 마감은 7/20(월) 10:00** (~~7/24~~ 정정, D-012). 예선 종료 후 5일뿐 → 학습 재현성(시드·버전 고정)은 예선과 **병행 트랙**
4. 현재 대회 상태 state (2026.07.06 기준 — 계속 변하므로 확인 시점 기록)
   1. 1등 : 0.7936 / 12등 (본선 커트라인) : 0.7807 / **팀 최고 : 0.7400** (3-way blend [1,1,2] + soft-AU 라우팅, 대장 #6)
   2. 커트라인 상승 중: 07-04→07-06 2일 평균 **+0.0024/일** (직전일 07-05→06은 +0.004로 가속) → 마감(07-15) 외삽 **~0.79-0.80**. 목표를 **0.79+**로 상향 — 갭 −0.041은 미세 튜닝으로는 못 메우고, AU 라우팅급 **구조적 발견 2~3개**가 더 필요하다
   3. 검증된 성공 패턴: "결정적 키(id 프리픽스 등)로 판별되는 서브모집단 + 전용 학습이 전 성분을 큰 마진(+0.15급)으로 이길 때"만 라우팅이 전이된다 (soft-AU: 리그 +0.0065 → LB +0.0069). 파생 신호 기반 행 선택(R4, meta-selector, first-step)은 전부 실패
   4. 로컬 LB 시뮬레이션 리그(holdout_base.npz 9,969행)의 축별 신뢰도: 라우팅·성분 추가/제거 축은 전이 검증됨, **enc 지분 조정 축은 전역·버킷 불문 전면 금지** — LB 역전 실측 2회(w112 핸드오프 §5 가중 곡선, exp #20 버킷 가중 −0.0061)로 리그가 enc 지분 하향을 일관되게 과대평가함이 확인됨 (#16은 이 근거로 미제출 기각)

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
   1. 유사 문제 기법 리서치: 텍스트+정형 혼합 분류, class imbalance 대응(class weight, focal loss, ~~threshold 튜닝~~ — 이후 폐기, §3.1/D-009), 경량 다국어 인코더(DeBERTa-v3-small, multilingual-e5-small 등)
   2. 도구: NotebookLM(문서 정리), Claude Code(EDA·구현), Codex + insane_search(어떻게 보면 메인 리서치)
   3. 리서치 결과는 `context/research.md`에 축적 → 이후 콘텍스트 주입 재료로 사용
4. **Opportunity Sizing → 점수 갭 분석으로 대체**
   * 베이스라인(TF-IDF+LogReg) 점수 vs 리더보드 상위권 점수의 갭을 보고, 어느 피처/모델 계층에 개선 여지가 큰지 추정. 리더보드를 "시장"으로 읽는다.

### 3. [솔루션 & 실험 설계]

1. **Problem Solutioning — 후보 솔루션 계층 (Tier)**
   * **Tier 0:** 공식 베이스라인 재현 + 제출 → 파이프라인 검증이 목적 (점수는 무관)
   * **Tier 1:** 피처 엔지니어링 + GBDT(LightGBM/XGBoost) — 전이 피처(직전 행동 n-gram), 메타 피처, TF-IDF/문자 n-gram
   * **Tier 2:** 경량 사전학습 인코더 파인튜닝 — prompt+history를 직렬화해 입력, 메타는 별도 헤드 또는 텍스트에 병기
   * **Tier 3:** Tier 1 + Tier 2 앙상블 + **서브모집단 라우팅**(결정적 키 + 전용 모델 — soft-AU가 실증) + 이질 인코더 추가
   * ~~클래스별 threshold 튜닝~~ — **폐기** (2026.07.06, D-009): threshold/prior/calibration 가족은 calib_v1에서 리그 이득의 LB 비전이 실측(holdout +0.005 → LB −0.002), 같은 가족인 R3 첫스텝 prior는 그 근거로 미제출 기각. 재시도 금지 목록은 `context/experiments.md` 참조 (버킷 가중, enc 지분 조정, e5-small, char-ngram 4성분, first-step 라우팅, R4 specialist, meta-selector, seed soup 등)
2. **Solution Prioritization**
   * 순서 고정: **제출 파이프라인 완성(Tier 0) → 빠른 반복이 가능한 Tier 1 → Tier 2/3** — 현재 Tier 3 단계 진행 중 (커트라인 0.78+ 수준이라 Tier 3의 구조적 발견이 필수)
   * 판단 기준: (예상 점수 상승) ÷ (구현+학습 시간). 마감 D-3부터는 새 모델 금지, 튜닝·앙상블만.
3. **Solution Evaluation / Measurement Plan**
   1. **로컬 CV 체계 먼저 구축**: `id`의 세션 프리픽스(`sess_...`) 기준 **GroupKFold** — 같은 세션의 step이 train/valid에 갈라지면 누수
   2. 로컬 Macro-F1과 리더보드 점수의 상관을 초기에 확인 → 이후 로컬 CV로 의사결정, 제출은 검증용
   3. 제출 예산 관리: 일 10회. 초반엔 파이프라인 검증 1~2회, 이후 유의미한 개선만 제출. **Private Score 기준이므로 public 과적합 경계**
   4. 실험 로그: `context/experiments.md`에 (가설 → 변경점 → 로컬 CV → 리더보드) 기록

### 4. [엔지니어링 / 모델 구현]

1. **콘텍스트 주입** — `CLAUDE.md` 작성: 대회 룰·제약·데이터 명세·현재 최고 점수·실험 규칙을 담아 에이전트가 항상 대회 맥락 위에서 작업하게 한다.
2. **Scaffolding**
   1. File structure:
      ```
      2026-AI-DACON/
      ├── CLAUDE.md            # 대회 콘텍스트 (에이전트용)
      ├── PLAN.md              # 이 문서
      ├── data/                # train.jsonl, train_labels.csv, test.jsonl, sample_submission.csv (git 제외)
      ├── notebooks/           # EDA
      ├── src/
      │   ├── features.py      # 피처 추출 (학습·추론 공용 — 단일 소스)
      │   ├── train.py         # 학습 → model/ 산출
      │   └── infer.py         # script.py의 원형
      ├── submit/              # script.py, requirements.txt, model/ → submit.zip
      ├── docs/                # research.md, experiments.md
      └── scripts/make_submit.py  # 패키징 + 로컬 스모크 테스트
      ```
   2. README / CLAUDE.md 초안은 scaffolding 시점에 에이전트가 작성
3. **학습/추론 분리 원칙**: 피처 코드는 학습·추론이 같은 모듈을 import (불일치 = 조용한 점수 하락). script.py는 `model/`만 읽어 예측.
4. **오프라인 제약 체크리스트** (제출 전 매번)
   - [ ] script.py에 네트워크 호출 없음 (HF `from_pretrained`는 로컬 경로 + `local_files_only=True`)
   - [ ] requirements.txt 설치 10분 이내 (버전 고정)
   - [ ] zip ≤ 1GB / 추론 ≤ 10분 (T4보다 느린 환경 가정, 로컬에서 시간 측정)
   - [ ] sample_submission.csv와 동일한 형식(id, action)·행 수 출력
5. **재현성** (재현 코드 제출 7/20 10:00 대비 — D-012로 병행 트랙 승격): 시드 고정, 패키지 버전 잠금, 학습 스크립트 원커맨드 실행 가능하게 유지
   * 챔피언 5성분 중 리포 내 재현 가능한 것은 e5 계열·AU specialist뿐 — **linear 원본 트레이너·mBERT 파인튜닝·AAR stacker 학습 코드는 동료 측에 있음 → 즉시 확보 요청** (reports/deep_research_gap_check_2026-07-10.md G1)
   * `submit/requirements.txt`의 `transformers>=4.51` → 검증된 정확 버전으로 고정
   * `src/train.py`/`src/infer.py`/`src/features.py` 스캐폴드를 실제 챔피언 파이프라인으로 정리 (또는 성분별 학습 스크립트 목록 문서로 대체)

### 5. [확장 및 자동화] Loop Engineering

> 나의 한계치를 agent / automation으로 끌어올리는 방법 설계 — 이 대회에서는 **실험 루프의 회전 속도**가 곧 한계치다.

1. **실험 루프 자동화**: 설정 파일(config) 하나 바꾸면 학습→CV 평가→experiments.md 기록까지 원커맨드로
2. **Claude Code agent 활용**
   1. EDA·피처 아이디어 탐색을 병렬 에이전트로 fan-out
   2. 코드 리뷰(/code-review)로 피처 누수·학습/추론 불일치 점검
   3. 반복 잡무(패키징, 스모크 테스트, 로그 정리)는 스크립트화 후 에이전트에 위임
3. **제출 자동화**: `make_submit.py` = G1 tests → G2 git clean → G3 패키징 → G4 12개 검증(네트워크 차단·시간·출력 형식) → G5 제출 대장 기록. 하나라도 실패하면 zip이 남지 않는다.
4. **기록 강제화 (context/)** ✅ 2026.07.04 구축
   1. `context/INDEX.md` 진입점: decisions(ADR-lite) · experiments · research · submissions(자동 대장) · daily · reports
   2. 게이트: 데일리(`new_day.py`) / 제출(`make_submit.py` G1~G5) / 실험(train.py 자동 로그 — 구현 예정) / 페이즈(지정 산출물 파일 존재 = 완료 조건)
   3. 원칙: 기록이 없으면 일어나지 않은 것. 이 기록이 본선(08.11) 발표 자료의 원천.

---