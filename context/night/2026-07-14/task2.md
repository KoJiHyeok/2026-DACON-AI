# task2 — 발표 사료 최신화 (07-13 Qwen 대역전 스토리 반영)

## 컨텍스트

DACON 236694. 07-13 밤 task3가 만든 발표 사료(docs/presentation/)는 "실험 #1~#51, 제출 #1~#12" 시점 기준. 그 직후 하루 동안 스토리가 크게 진행됐다: exp #52 Qwen 문샷 → 제출 #13 **시간초과 FAIL** → 원인 규명("파라미터 등가 ≠ 연산 등가") → 속도레버 2종(길이정렬 배칭 1.7x + fast_aar 2.8x, 출력등가 검증) → **Colab T4 30k 리허설 515s 실측** → 제출 #14 **LB 0.77089 팀 최고 갱신 (89→79등)**, holdout→LB 전이율 73.5%. 발표(본선) 관점에서 가장 극적인 챕터이므로 사료에 반드시 들어가야 한다. 07-13 reviewer 판정은 조건부 병합(시차 stale)이었고 병합 보류 상태다.

## 목표 / 완료 조건 (DoD)

1. `docs/presentation/` 전체를 현행화:
   - `key_numbers.md`·`sources.md`의 "실험 #1~#51" 표기 → **#52까지**, 제출 대장 **#13·#14 추가** (LB 궤적 0.71884→…→0.7623→0.77089)
   - 새 챕터/섹션: "시간초과 대역전" — #13 FAIL 원인(연산활성 파라미터 360M/24L vs 86M×2/12L), 레버 2종 수치, T4 리허설 515s, #14 성공, 전이율 73.5% (인코더 가족 축 67% 할인 재확인)
2. `docs/presentation/verify_citations.py` — 신규 인용 **6건 이상** 추가 (experiments.md exp #52 행·submissions.md #13/#14 행·daily 07-13에서 정확한 문자열 인용), 실행 시 **기존+신규 전건 PASS·불일치 0건**
3. `context/night/2026-07-14/report_presentation_v2.md` — 변경 요약 + verify_citations.py 실행 출력 전문
4. **`context/night/2026-07-14/task2.DONE` 생성 (한 줄 요약 포함)**

## 재료 (절대 경로, 전부 읽기 전용)

- **07-13 원본 산출물** (미병합 — 복사해서 갱신 시작): `C:\dev\night\2026-07-13\task3\docs\presentation\` — 이 워크트리 자체는 절대 수정 금지
- 사실 원본: `C:\dev\2026-AI-DACON\context\experiments.md` (#52 행), `C:\dev\2026-AI-DACON\context\submissions.md` (#13·#14 행), `C:\dev\2026-AI-DACON\context\daily\2026-07-13.md`, `C:\dev\2026-AI-DACON\docs\t4_rehearsal.md`
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe`

## 금지

- 워크트리 밖 수정 금지, `git push` 금지, 네트워크 코드 금지
- **수치는 반드시 원본 문서에서 grep으로 확인 후 인용** — 기억/추정 수치 기재 금지 (verify_citations.py가 문자열 대조하므로 원문과 다르면 FAIL)
- 폐기 목록 재시도 금지

## 진행 프로토콜 (재개 대비 — 핵심)

1. 시작하자마자 `context/night/2026-07-14/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위마다 PROGRESS 갱신 후 **git commit**
3. 전부 끝나면 `task2.DONE` 생성 + 최종 커밋

## 작업 내용

1. 07-13 docs/presentation/ 을 워크트리 같은 경로로 복사
2. 원본 3문서에서 #52·#13·#14·리허설 관련 정확한 문구·수치 추출 (grep)
3. 새 챕터 작성 + key_numbers/sources 갱신
4. verify_citations.py에 신규 인용 등록 → 실행 전건 PASS 확인
5. report_presentation_v2.md 작성 → DONE
