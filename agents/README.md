# Sub-agent Operating Model

이 폴더는 대회 실험을 병렬로 굴리기 위한 역할별 instruction 모음이다.

각 sub-agent는 `common.md`를 먼저 읽고, 자기 역할 파일만 추가로 읽은 뒤 작업한다. 작업 결과는 코드 변경보다 명확한 산출물을 남기는 것을 우선한다.

## 역할 분리

| 역할 | Instruction | 주 산출물 | 주요 수정 영역 |
|---|---|---|---|
| EDA Agent | `eda_agent.md` | 데이터 관찰, 누수 리스크, 피처 후보 | `context/research.md`, `context/experiments.md`, `notebooks/` |
| Feature Agent | `feature_agent.md` | 학습/추론 공용 피처 구현 | `src/features.py`, `context/experiments.md` |
| Modeling Agent | `modeling_agent.md` | CV, Tier 0/1 학습, 모델 저장 | `src/train.py`, `configs/`, `submit/model/` |
| Encoder Agent | `encoder_agent.md` | Tier 2 인코더 실험 설계/구현 | `src/`, `configs/`, `context/experiments.md` |
| Ensemble Agent | `ensemble_agent.md` | OOF 기반 앙상블, threshold 튜닝 | `src/`, `configs/`, `context/experiments.md` |
| Submission Agent | `submission_agent.md` | 제출 zip, 오프라인 검증 | `submit/`, `scripts/` |
| Review Agent | `review_agent.md` | 누수/불일치/재현성 리뷰 | 변경 파일 전반 |

## Handoff 원칙

1. 모든 실험은 `context/experiments.md`에 남긴다.
2. CV 점수는 세션 프리픽스 기준 GroupKFold Macro-F1만 공식 기록으로 인정한다.
3. `src/features.py`는 학습과 추론의 단일 소스다. 추론 스크립트에 별도 피처 로직을 복붙하지 않는다.
4. 제출 전에는 `scripts/make_submit.py`와 `scripts/validate_submit.py`를 통과해야 한다.
5. sub-agent가 확신하지 못한 가정은 `Open questions`로 남긴다.

## 작업 시작 템플릿

```text
역할: <EDA|Feature|Modeling|Encoder|Ensemble|Submission|Review>
읽을 파일: agents/common.md + agents/<role>_agent.md + 관련 코드/문서
목표: <이번 작업의 한 문장 목표>
제약: GroupKFold, Macro-F1, offline submit, feature single source
산출물: <수정 파일 또는 문서 기록>
```
