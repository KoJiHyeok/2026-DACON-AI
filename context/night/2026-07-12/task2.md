# task2 — CX-005: 5성분 frozen-shadow 스태커 판정 하네스 (독립 대조 구현)

## 컨텍스트

DACON 236694. D-014 레인 A = 챔피언 5성분(linear·AAR·e5-hist12·mBERT-h6·AU) 전체의 parity OOF 스태커 — exp #46(CX-002)이 비승격이었던 자격 미달 2건(① baseline이 proxy ② 메타층만 cross-fit)을 해소하는 재실험이다. Claude 측이 같은 스펙의 판정 파이프라인을 별도 구현 중이며, **이 작업은 그 독립 대조 구현**이다(CX-003 이중검증 패턴). 두 구현의 수치가 일치해야 판정이 성립한다. 판정 선언·승격·제출은 Claude 단독 권한 — 이 하네스는 `promotion_eligible=false` 고정.

## 목표 / 완료 조건 (DoD)

1. `scripts/shadow_eval/` — 아래 **고정 스펙**의 frozen-shadow 평가 하네스. 스펙 임의 변경 금지(변경 제안은 report에만):
   - **메타 피처 (76열, 이 순서 고정)**: linear 확률 14 + AAR 14 + e5 14 + mBERT 14 + AU 14(비-AU 행은 0) + au_mask 1 + 성분별 엔트로피 5 (자연로그, AU 비-AU 행 엔트로피=0)
   - **메타 모델**: `LogisticRegression(C=1.0, max_iter=3000, random_state=42)` + `MaxAbsScaler` (CX-002와 동일, 스케일러는 학습 폴드 내부에서만 fit)
   - **frozen shadow**: `holdout_split_meta.json`의 15% 홀드아웃 세션은 메타 학습에서 **완전 배제**. 메타 학습 = 나머지 85% 행들의 OOF 피처(fold_map 5-fold). 진단용 메타-CV는 85% 안에서 fold_map 기준 cross-fit, 최종 메타 = 85% 전체 적합.
   - **평가**: 홀드아웃 15% 행에서, 피처는 85%-학습 표면(아래 재료의 holdout npz들)으로 구성 → 스태커 예측 vs **챔피언 블렌드 표면**(scripts/league4/common.py의 4-way+soft-AU α=0.9, BASELINE_SOFT_AU=0.73877 앵커로 재현 검증) Macro-F1 비교.
   - **판정 지표 5종** (probe_c 계열과 동일 정의 — `scripts/league4/probe_c_args_lite.py` 참조): ①row 델타 ②세션균등 델타 ③세션당1행 MC200 평균±표준편차 ④paired session bootstrap CI(1000회, seed 42) ⑤홀드아웃 세션 반반 분할 델타
2. 실행 결과 `out_shadow/verdict.json` — 전 지표 수치 + 입력 npz SHA256 전부 + 패키지 버전 + `promotion_eligible: false`. 같은 커맨드 2회 실행 시 시간 제외 바이트 동일(결정론).
3. `tests/test_shadow_eval.py` — 로컬 CPU 통과: 피처 조립 순서·AU 마스킹·엔트로피 계산의 소형 고정 입력 대조, 홀드아웃 세션 배제 assert(누수 네거티브 테스트 — 홀드아웃 id가 메타 학습 행에 있으면 실패), 지표 5종의 소형 예제 검증
4. `context/night/2026-07-12/report_shadow.md` — 수치 요약표, 스펙 준수 확인, 알려진 한계. **파일명 task 시작 금지.**
5. `context/night/2026-07-12/task2.DONE` 생성 (요약 포함).

## 재료 (절대 경로 — 전부 읽기 전용, 사용 전 SHA256SUMS 대조 후 report에 해시 기록)

- fold_map: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_h12\fold_map.csv` (sha256 56074c16c400fbccc389e15c01c05adc4db810533516340f15e9826dd44fe295)
- OOF 5성분 (각 폴더의 SHA256SUMS로 검증):
  - e5: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_h12\oof_fold{0..4}.npz`
  - mBERT: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_mbert_h6\oof_mbert_fold{0..4}.npz`
  - linear: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_linear\` (2026-07-12 생성 — SHA256SUMS 필수 확인)
  - AU: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_au\` (sess_au 행 5,025개만 커버 — run_oof_au.json에서 스코핑 확인)
  - AAR: `C:\dev\2026-AI-DACON\artifacts\experiments\oof_aar\` (run_oof_aar.json의 deviation_from_train_aar 필드 참조)
- 홀드아웃(15%) 85%-학습 표면: `C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_linear.npz`·`holdout_stacker.npz`(=AAR)·`holdout_base.npz`, `C:\dev\2026-AI-DACON\colab_out\holdout_e5_h12.npz`, `holdout_mbert.npz`; AU 표면·챔피언 블렌드 = 워크트리 내 `scripts/league4/common.py` 재사용 (표면 캐시가 gitignore면 절대 경로로 로드)
- 분할: `C:\dev\2026-AI-DACON\context\night\2026-07-05\holdout_split_meta.json`
- 데이터·라벨: `C:\dev\2026-AI-DACON\data\` / 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 제출 금지, 네트워크 코드 금지
- `scripts/oof5/**`·`scripts/stacker_h12/**`(기존)·`submit/**`·canonical context 수정 금지 — 이 작업의 소유는 `scripts/shadow_eval/**`와 전용 테스트뿐
- **점수 주장·승격 판단 금지** — verdict.json은 수치 보고서다. "게이트 통과/실패" 문구는 판정 지표 정의에 따른 기계 출력으로만.
- 폐기 목록 재시도 금지 (특히 탐색 specialist·템플릿 라우팅 계열 접근 금지)

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-12/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신 + **git commit** (커밋 불가 시 PROGRESS에 사유 기록)
3. 전부 끝나면 `task2.DONE` + 최종 커밋

## 작업 내용 (단계)

1. PROGRESS 생성 → 재료 존재·해시 검증 (oof_linear/au/aar가 아직 없으면 **1시간 대기 후 재확인**을 PROGRESS에 기록하며 반복 — 낮 세션 빌더가 생성 중)
2. `scripts/shadow_eval/features.py` — 76열 조립 (OOF 뷰·홀드아웃 표면 뷰 공용, 순서 고정 상수로 명시)
3. `scripts/shadow_eval/evaluate.py` — 메타 학습(85% OOF) → 홀드아웃 평가 → 지표 5종 → verdict.json
4. 챔피언 baseline 재현 검증 — league4.common으로 홀드아웃 챔피언 표면 Macro-F1이 0.73877(±1e-5)로 재현되는지 assert. 어긋나면 진행 중단하고 PROGRESS에 기록(입력 오염 신호)
5. tests 작성·통과 → 2회 실행 결정론 확인 → report_shadow.md → task2.DONE
