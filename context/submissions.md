# Submissions Ledger — 제출 대장

> `scripts/make_submit.py`가 게이트 통과 시 자동으로 행을 추가한다. LB 점수는 제출 후 수기로 채운다.
> 제출 예산: **일 10회**. Private Score 기준 — public 점수에 과적합하지 말 것.

| # | 일시 | 커밋 | zip 크기 | 검증 | 로컬 CV | LB (public) | 메모 |
|---|---|---|---|---|---|---|---|
| 1 | 07-05 09:37 | `f305fc9` | 546.0MB | PASS | - | **0.71884** | 3way w112 rebuild (original encoder fp16) — 팀 w112 0.7208 대비 −0.002 (fp16 재변환분 추정), 기준선 복구 |
| 2 | 07-05 10:17 | `963396e` | 546.0MB | PASS | - | **0.71884** | 3way + sibling label recovery (D-008 probe) — #1과 델타 0 → test에 복원 가능한 형제 행 없음 (세션당 1스텝 샘플링 확정적). 보험 코드는 잔류 |
