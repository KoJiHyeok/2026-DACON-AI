# 추론 속도 프로파일링 보고서

## 실행 계약

기본값은 `SPEED_ROWS=300`, `SPEED_REPEATS=3`, `SPEED_DEVICE=cpu`,
`SPEED_MODEL_DIR=C:\dev\2026-AI-DACON\submit\model`이다.

```powershell
$env:SPEED_ROWS=300; $env:SPEED_DEVICE="cpu"
& C:\dev\2026-AI-DACON\.venv\Scripts\python.exe scripts/speed/profile_infer.py
```

서버 GPU에서는 환경 변수만 바꾼다: `$env:SPEED_DEVICE="cuda"`.

## CPU 결과

이 워크트리에서는 실측을 완료하지 못했다. 지정 Python 실행 파일은 존재하지만 `torch`가
설치되어 있지 않아 `submit/script.py` import 단계에서
`ModuleNotFoundError: No module named 'torch'`가 발생했다. 수치를 추정해 기록하지 않는다.

| 단계 | 중앙값(초) | 전체 대비 |
|---|---:|---:|
| 데이터 로드 | 미측정 | 미측정 |
| 피처·직렬화 | 미측정 | 미측정 |
| linear | 미측정 | 미측정 |
| AAR stacker | 미측정 | 미측정 |
| e5 / mBERT 인코더 | 미측정 | 미측정 |
| AU 라우팅 | 미측정 | 미측정 |
| blend·후처리 | 미측정 | 미측정 |

실측 전에는 병목 상위 2개를 확정할 수 없다. 우선 확인할 후보는 인코더
forward/tokenization과 AAR·linear 아티팩트 로드지만, 이는 측정 결과가 아니다.

## report-only 최적화 후보

1. GPU 메모리 여유 범위에서 토크나이저 배치 크기(`BATCH`)를 비교해 호출 횟수와 padding 비용을
   줄이는 후보를 검토한다. 행 단위 등가성 검사를 통과한 값만 채택 후보로 남긴다.
2. 반복 계측에서 직렬화/토크나이저 입력 캐시를 별도로 측정한다. 행 순서·메모리 사용량·프로세스
   수명 조건을 확인해야 하며, 구현은 이 작업에 포함하지 않는다.

## 실측 결과 (2026-07-12, 학교서버 A5000, 평가서버 미러 venv-speed: py3.11 + torch 2.7.1+cu128 + sklearn 1.8.0)

측정 조건: SPEED_ROWS=30(CPU)·300(GPU), repeats=3 중앙값, 모델 = 배포 zip과 동일 아티팩트(959MB 미러).
**등가성 게이트: 300행에서 원본 script.py 경로와 예측 100% 일치 (match_rate 1.0, mismatch 0).**

| 단계 | CPU 30행 중앙값(s) | GPU 300행 중앙값(s) | 비고 |
|---|---:|---:|---|
| 데이터 로드 | 0.025 | 0.005 | |
| 피처·직렬화 | 0.004 | 0.030 | |
| linear | 0.399 | 0.678 | |
| AAR stacker | 2.372 | 3.059 | **CPU 단계 — 행수 준비례 (~10ms/행), 대량 추론 병목 1순위** |
| 인코더 블록 (e5+mBERT) | 5.373 | 3.922 | GPU에서 고정비 지배적 — 300행도 30행 CPU보다 빠름 |
| AU 라우팅 | 0.582 | 0.775 | char TF-IDF, CPU 단계 |
| blend·후처리 | ~0.000 | ~0.001 | |

- 첫 실행은 모델 로드·워밍업으로 인코더가 ~1.5배 느림 (run1 vs run2/3) — 본선 시간 측정 시 워밍업 분리 필요.
- 병목 상위 2개 (실측 확정): ① AAR stacker (CPU sklearn, 행수 비례 — TF-IDF transform 지배) ② 인코더 블록 (GPU, 고정비+행수). T4는 A5000 대비 인코더가 2~3배 느릴 것으로 예상되나 챔피언 T4 실측 50.2초가 이미 존재해 여유 충분.
- 환경 함정 기록: 학교서버 venv-dacon(sklearn 1.7.2)에서는 AAR LogisticRegression unpickle이 `multi_class` AttributeError로 실패 — **제출물 실측은 반드시 평가서버 스펙(py3.11+, sklearn 1.8.0) 환경에서** (서버 ~/venv-speed 구축 완료, 재사용 가능).
