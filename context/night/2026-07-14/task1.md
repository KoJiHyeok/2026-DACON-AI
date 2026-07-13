# task1 — 재현 플레이북 현행화 (챔피언 #14 = Qwen 하이브리드 기준)

## 컨텍스트

DACON 236694 (AI 에이전트 다음 행동 14클래스, Macro-F1). 어제(07-13) 제출 #14가 LB **0.77089**로 팀 최고를 갱신하며 챔피언이 교체됐다: 인코더 블록 = **Qwen2.5-0.5B 단독**(mBERT 제거), script.py에 길이정렬 배칭 + fast_aar 고속 경로 통합. 07-13 밤에 만든 재현 플레이북(task2 산출물)은 그 이전 챔피언(#12 시점, e5+mBERT 2인코더) 기준이라 **해시표·구성이 전부 stale** — reviewer가 조건부 병합으로 판정했고 병합이 보류됐다. 본선 코드 검증(7/24)에 쓸 자료이므로 현행 main 기준으로 갱신해야 한다.

## 목표 / 완료 조건 (DoD)

1. `docs/repro_playbook.md` — 현행 챔피언(#14, 커밋 `d16fbc8` 이후 main) 기준으로 갱신:
   - 배포 아티팩트 해시표가 **현행 `submit/` 스테이징 실측 SHA256과 일치** (encoder=Qwen, encoder_2 항목 삭제, fast_aar.py 추가)
   - 구성 요약 갱신: 인코더 블록 Qwen 단독(serialize max_hist=12), enc_block_weights [1.0], weights [1,1,2], soft-AU α0.9, fast_aar 경로
   - `docs/t4_rehearsal.md`(T4 리허설 규약) 상호참조 1줄 추가
   - "어느 features.py가 계약인가" 명시 (submit/features.py가 배포 계약, src/features.py는 학습측 — 07-13 reviewer 지적사항)
2. `scripts/repro_rehearsal/verify.py` — 현행 구성(Qwen 단독·fast_aar)에 맞게 갱신 후 실행해 `"status": "pass"` 출력
3. `tests/test_repro_rehearsal.py` — pytest 통과 (구성 변화 반영)
4. `context/night/2026-07-14/report_repro_v2.md` — 변경 요약 + verify.py 실행 출력 전문
5. **`context/night/2026-07-14/task1.DONE` 생성 (한 줄 요약 포함)** ← 러너 완료 판정 기준

## 재료 (절대 경로, 전부 읽기 전용)

- **07-13 원본 산출물** (미병합 — 여기서 복사해 갱신 시작): `C:\dev\night\2026-07-13\task2\docs\repro_playbook.md`, `C:\dev\night\2026-07-13\task2\scripts\repro_rehearsal\`, `C:\dev\night\2026-07-13\task2\tests\test_repro_rehearsal.py` — 이 워크트리 자체는 절대 수정 금지
- 현행 배포 스테이징: `C:\dev\2026-AI-DACON\submit\` (script.py·fast_aar.py·aar_infer.py·model/)
- 리허설 실측 규약: `C:\dev\2026-AI-DACON\docs\t4_rehearsal.md`
- 07-13 reviewer 판정 원문: `C:\dev\2026-AI-DACON\context\night\2026-07-13\report_aar_speed.md` 하단 부기 참고
- 데이터: `C:\dev\2026-AI-DACON\data\` (읽기 전용)
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (서버 미러 sklearn 1.8.0 — 시스템 파이썬 금지)

## 금지

- 워크트리 밖 수정 금지 (메인 리포 작업트리·07-13 워크트리·팀 리포 포함)
- `git push` 금지, 수동 zip·제출 금지, 네트워크 코드 금지
- `submit/` 파일 내용 변경 금지 — 해시는 읽어서 기록만
- 폐기 목록(experiments.md 재시도 금지 테이블) 재시도 금지

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-14/PROGRESS-task1.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신(체크리스트 + '다음 재개 지점' 한 줄) 후 **git commit**
3. 전부 끝나면 `task1.DONE` 생성 + 최종 커밋

## 작업 내용

1. 07-13 산출물 3종을 워크트리의 같은 경로로 복사 (docs/, scripts/repro_rehearsal/, tests/)
2. `submit/` 전 파일 SHA256 실측 → 플레이북 섹션 6 해시표 재작성 (encoder=Qwen 항목·fast_aar.py 추가, encoder_2 행 삭제)
3. 구성 서술 갱신 (위 DoD 1의 4개 항목)
4. verify.py를 현행 구성에 맞게 수정 (검사 대상 컴포넌트: aar/linear/qwen-encoder/au — mbert 제거) → 실행 pass 확인
5. pytest 실행 → 통과 확인 → report_repro_v2.md 작성 → DONE
