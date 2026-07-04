# 검증 체계 설계

> 원칙: "리더보드에 내기 전에, 대회 기준의 실패 요인을 로컬에서 전부 걸러낸다."
> 제출은 일 10회 제한 — 형식 오류·시간 초과로 제출을 태우는 일이 없어야 한다.

## Layer 1 — 제출물 게이트 (대회 기준 시뮬레이션) ✅ 구축 완료

`python scripts/validate_submit.py <submit.zip>` — 평가 서버 환경을 로컬에서 재현한다.

| # | 검증 항목 | 대회 규정 |
|---|---|---|
| 1 | zip 루트 구조 (model/ + script.py + requirements.txt) | 제출 형식 |
| 2 | zip 용량 ≤ 1GB | 용량 제한 |
| 3 | requirements 버전 고정(==) | 재현성·설치 안정성 |
| 4 | 샌드박스 실행: 서버 레이아웃 재현(`./data` 제공 → `./output` 생성) | 서버 실행 규약 |
| 5 | **네트워크 차단** 상태에서 실행 (socket 연결 몽키패치) | 오프라인 규정 |
| 6 | 실행 시간 ≤ 10분 (로컬 경고선 8분 — 서버가 더 느릴 수 있음) | 시간 제한 |
| 7 | output/submission.csv: 컬럼(id, action)·행 수·id 순서 = sample_submission | 채점 형식 |
| 8 | action 값 ∈ 14클래스 | 라벨 유효성 |

검증 기록 (2026.07.04):
- 공식 `baseline_submit.zip` → **12개 검증 전부 PASS** (실행 28.5초) — 하네스 자체가 공식 제출물로 입증됨
- 네거티브 테스트: 차단 환경에서 `urllib.request.urlopen()` → `RuntimeError: NETWORK BLOCKED` 확인

한계(알려진 것): 네트워크 차단은 Python socket 수준 — C 확장이 자체 소켓을 열면 못 잡는다.
로컬 GPU 부재 시 시간 측정은 CPU 기준 참고치로만 사용.

## Layer 2 — 모델 검증 (점수 신뢰성)

- **GroupKFold (세션 프리픽스 기준)**: 같은 세션의 step들이 train/valid에 갈라지면 누수 → `src/features.py::session_id()` 를 그룹 키로 사용
- 지표는 **Macro-F1** + 클래스별 F1 리포트 (희소 클래스 붕괴 감지)
- 로컬 CV ↔ 리더보드 상관을 초기 2~3회 제출로 캘리브레이션 → 이후 로컬 CV로 의사결정
- 모든 실험은 `context/experiments.md` 에 기록

## Layer 3 — 코드 검증 (조용한 버그 방지)

- `tests/` 단위 테스트: `python -m pytest tests/` (또는 `python tests/test_features.py`)
- 핵심 불변식:
  - 학습·추론이 `src/features.py` 의 **같은 함수**로 피처 생성 (불일치 = 조용한 점수 하락)
  - `session_id()` 파싱 정확성 (GroupKFold 무결성의 전제)
  - 14클래스 목록이 명세와 일치
- 제출 전 워크플로우: `tests 통과` → `make_submit.py` (패키징+Layer 1 자동 실행) → 제출

## 제출 전 체크리스트 (매번)

1. [ ] `python -m pytest tests/` 통과
2. [ ] `python scripts/make_submit.py` → 12개 검증 전부 PASS
3. [ ] 로컬 CV가 직전 최고 기록 대비 개선 확인 (`context/experiments.md`)
4. [ ] 제출 예산 확인 (일 10회)
