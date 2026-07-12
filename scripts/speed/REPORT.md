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
