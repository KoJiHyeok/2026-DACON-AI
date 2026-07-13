# Colab T4 제출 리허설 규약 (07-13 실증)

> 제출 #13 시간초과 FAIL → #14 성공(LB 0.77089)을 만든 절차. **새 모델 패밀리/大변경 제출 전 필수.**
> 근거: "파라미터 등가 ≠ 연산 등가" 오판 방지 — 로컬(5행 G4)로는 T4 30k 시간을 예측할 수 없다.

## 원칙

- 평가 서버 = **T4 16GB / 3 vCPU / 12GB RAM / 추론 10분 / 실평가 30,000행** (대회 데이터 페이지 명시)
- Colab T4 런타임 = 동일 GPU + 2 vCPU (CPU 구간은 실전이 더 빠름 → 보수적 측정)
- 리허설 zip은 `make_submit.py --allow-dirty`로 생성 (G2·G5 생략, 대장 오염 없음). **제출용은 풀게이트 재실행.**
- 판정: **총 8분 미만 GO / 8~10분 경계(마진 분석 후 결정 — #14는 8.6분 GO로 성공) / 10분+ 중단**

## 셀 시퀀스 (완결형 — colab-run 철칙 준수)

1. **GPU 확인 + 서버 환경 재현 pip**: `assert 'T4' in torch.cuda.get_device_name(0)` → `pip install "scikit-learn==1.8.0" "joblib==1.5.3"` (서버 기본과 버전 일치 — AAR joblib/fast_aar private API 호환) + requirements.txt 그대로 (`transformers>=4.51`)
2. **Drive 마운트 + submit.zip 해제** (`find_in_drive` 깊이4 탐색) → `/content/rehearsal`, 구조 assert (script.py/fast_aar.py/model/encoder/serialize_config.json)
3. **30,000행 test.jsonl 재현**: train.jsonl에서 `random.seed(42)` 샘플 + 같은 id로 sample_submission.csv 생성 (`respond_only` 더미)
4. **본 실측**: `subprocess.Popen([sys.executable,'-u','script.py'])` 스트리밍에 경과시간 프리픽스 — 단계별([N/5]) 타임스탬프 확보, EXIT/총시간/10분 판정
5. **분해 리포트**: [N/5] 마커별 +Δt — 병목 단계 식별

## 07-13 실측 기준치 (30k행, 레버 반영본)

| 단계 | 소요 |
|---|---|
| 로드+파싱 | 6.6s |
| linear | 7.0s |
| stacker (fast_aar) | 29.7s (구경로 ~84s) |
| encoder (Qwen 0.5B fp16 + 길이정렬 배칭) | 471.5s (구경로 ~800s) |
| **총** | **515s (8.6분)** — 실서버 채점 성공 |

## 연산량 추정 공식 (사전 스크리닝용 — 리허설 대체 불가)

인코더 시간 ∝ **연산활성 파라미터(임베딩 제외) × 평균 패딩 토큰 × 행수**.
예: Qwen 0.5B = 494M 중 임베딩 135M → 활성 360M/24층 (e5/mBERT는 각 ~86M/12층).
hist12 직렬화 실측: 평균 220tok, 384캡 도달 5.3% → 정렬 배칭 이득 1.7x.
