# 2026 DACON - AI·SW중심대학 디지털 경진대회 : AI부문

> AI Agent 행동(Action) 의사결정 예측 챌린지
> https://dacon.io/competitions/official/236694

AI 코딩 에이전트 세션의 특정 시점 상태를 보고, 에이전트가 다음에 수행할 행동을 **14개 클래스** 중 하나로 예측하는 NLP·분류 대회 저장소입니다.

이 저장소의 정본 운영 문서는 [PLAN.md](PLAN.md), 에이전트 공통 콘텍스트는 [CLAUDE.md](CLAUDE.md), 실험·결정·제출 기록의 진입점은 [context/INDEX.md](context/INDEX.md)입니다.

## 현재 상태 (2026-07-04 기준)

| 항목 | 상태 |
|---|---|
| 최고 기록 | **LB Macro-F1 0.7208**: w112, 3-way 앙상블 weights `[linear=1, stacker=1, encoder=2]` |
| 순위 / 커트라인 | 81등, 12등 커트라인 0.77585 |
| 검증 하네스 | `scripts/validate_submit.py` 구축, 팀 스태커 `submit.zip` 기준 12/12 PASS |
| 기록 시스템 | `context/` 기반 의사결정·실험·제출·데일리 로그 구축 |
| 현재 리포 코드 | `src/train.py`, `src/infer.py`, `submit/script.py`는 스캐폴드/TODO 상태 |
| w112 재건 | `legacy/w112_handoff.md`와 `context/reports/team_repo_map.md`에 정리. 인코더 v2 s42 가중치가 현재 리포에는 없음 |

현재 우선순위는 미세 앙상블 튜닝이 아니라 **시뮬레이터 포렌식(state→action 결정 규칙 분석)** 입니다. 근거와 폐기한 실험은 [context/experiments.md](context/experiments.md), [context/research.md](context/research.md)에 기록되어 있습니다.

## 문제 정의

각 샘플의 입력은 세 부분으로 구성됩니다.

| 입력 | 내용 |
|---|---|
| `current_prompt` | 현재 사용자 발화. 이 직후의 행동이 예측 대상 |
| `history` | 직전까지의 대화·행동 기록 0~12턴 (`user` ↔ `assistant_action` 교대) |
| `session_meta` | 요금제, 언어 선호, 잔여 토큰, 턴 번호, 경과 시간, 워크스페이스 상태 등 |

**14개 행동 클래스**

`read_file`, `grep_search`, `list_directory`, `glob_pattern`, `edit_file`, `write_file`, `apply_patch`, `run_bash`, `run_tests`, `lint_or_typecheck`, `ask_user`, `plan_task`, `web_search`, `respond_only`

데이터 명세 요약:

- `data/train.jsonl` 70,000건
- `data/train_labels.csv` 컬럼: `id`, `action`
- 서버 실행 시 `./data/test.jsonl`, `./data/sample_submission.csv`가 제공됨
- `script.py`는 `./output/submission.csv`를 생성해야 함

## 평가 및 제출 제약

| 항목 | 내용 |
|---|---|
| 평가 지표 | Macro-F1 |
| 제출 형식 | `submit.zip` 루트에 `model/`, `script.py`, `requirements.txt` |
| 실행 환경 | T4 16GB, 3 vCPU, 12GB RAM |
| 시간 제한 | 추론 10분 이하, 패키지 설치 10분 이하 |
| 용량 제한 | 제출 zip 1GB 이하 |
| 실행 방식 | 오프라인. 추론 중 네트워크 호출 금지 |
| 일정 | 예선 2026.07.01 ~ 2026.07.15 09:59, 본선 발표평가 2026.08.11 |

CV는 `id`의 세션 프리픽스 기준 GroupKFold/StratifiedGroupKFold를 사용합니다. 같은 세션의 step이 train/valid에 갈라지면 누수로 간주합니다.

## 저장소 구조

```text
├── AGENTS.md                 # 에이전트 진입 지시: CLAUDE.md 확인
├── CLAUDE.md                 # 대회 제약, 데이터, 실험 규칙, 코드 구조
├── PLAN.md                   # 전체 워크플로우·전략·일정
├── agents/                   # EDA, Feature, Modeling, Encoder, Ensemble 등 역할별 지시서
├── configs/                  # 실험 설정 placeholder
├── context/                  # decisions, experiments, research, submissions, daily, reports
├── data/                     # 대회 데이터 위치 (GitHub에는 포함하지 않음)
├── docs/                     # 검증 체계와 agent workflow 문서
├── legacy/                   # w112 핸드오프 등 계승 문서
├── notebooks/                # EDA/실험 노트북
├── scripts/                  # 데일리 로그, 제출 패키징, 제출 검증 하네스
├── src/                      # features / train / infer
├── submit/                   # 제출 스테이징: script.py, requirements.txt, model/
└── tests/                    # 피처 불변식 테스트
```

## 주요 명령

### 테스트

```powershell
python -m pytest tests/
```

`pytest`가 없으면 아래처럼 직접 실행할 수 있습니다.

```powershell
python tests/test_features.py
```

### 하루 기록 생성

```powershell
python scripts/new_day.py --lb1 0.79015 --lb12 0.77585 --ours 0.7208
```

생성된 데일리 로그는 `context/daily/`에 저장됩니다.

### 제출물 검증

```powershell
python scripts/validate_submit.py submit/submit.zip --data-dir data
```

검증 항목은 zip 구조, 용량, requirements 확인, 서버 입력 레이아웃 재현, 네트워크 차단, 실행 시간, `submission.csv` 컬럼·행 수·id 순서·action 유효성입니다.

### 제출 패키징

```powershell
python scripts/make_submit.py --cv 0.7xx --note "변경 요약"
```

`make_submit.py`는 다음 게이트를 순서대로 통과해야 `submit/submit.zip`을 남깁니다.

1. `tests/` 통과
2. git 작업 트리 clean
3. `submit/` 패키징
4. `validate_submit.py` 검증
5. `context/submissions.md` 자동 기록

개발 중 검증만 할 때는 `--allow-dirty`로 git clean 게이트와 제출 대장 기록을 생략할 수 있습니다.

## 구현 원칙

- 피처 생성은 [src/features.py](src/features.py) 단일 소스를 사용합니다.
- 학습과 추론의 피처 불일치는 조용한 점수 하락으로 봅니다.
- 제출은 수동 zip이 아니라 [scripts/make_submit.py](scripts/make_submit.py) 경유를 원칙으로 합니다.
- 의사결정은 [context/decisions.md](context/decisions.md)에 ADR-lite 형식으로 기록합니다.
- 실험은 [context/experiments.md](context/experiments.md)에 가설 → 변경점 → 로컬 CV → 리더보드 순서로 기록합니다.

## 참고 문서

- [docs/validation.md](docs/validation.md): 제출 전 검증 체계
- [docs/agent_workflow.md](docs/agent_workflow.md): 역할별 작업 흐름
- [legacy/w112_handoff.md](legacy/w112_handoff.md): w112 최고 제출물 구성과 계승 정보
- [context/reports/team_repo_map.md](context/reports/team_repo_map.md): 팀 리포 조사와 w112 재건 결손 항목
