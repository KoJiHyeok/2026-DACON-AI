# Submissions Ledger — 제출 대장

> `scripts/make_submit.py`가 게이트 통과 시 자동으로 행을 추가한다. LB 점수는 제출 후 수기로 채운다.
> 제출 예산: **일 10회**. Private Score 기준 — public 점수에 과적합하지 말 것.

| # | 일시 | 커밋 | zip 크기 | 검증 | 로컬 CV | LB (public) | 메모 |
|---|---|---|---|---|---|---|---|
| 1 | 07-05 09:37 | `f305fc9` | 546.0MB | PASS | - | **0.71884** | 3way w112 rebuild (original encoder fp16) — 팀 w112 0.7208 대비 −0.002 (fp16 재변환분 추정), 기준선 복구 |
| 2 | 07-05 10:17 | `963396e` | 546.0MB | PASS | - | **0.71884** | 3way + sibling label recovery (D-008 probe) — #1과 델타 0 → test에 복원 가능한 형제 행 없음 (세션당 1스텝 샘플링 확정적). 보험 코드는 잔류 |
| 3 | 07-05 14:35 | `06175c2` | 757.0MB | PASS | - | **0.71280** | 4way: +e5-small as encoder_2 (uniform enc block) — 3way 대비 **−0.006**. small이 블록 내 uniform 평균으로 base 지분을 반토막내며 희석. 폐기 (uniform은 재시도 금지, 가중 블록은 holdout_small.npz 도착 후 리그 판정) |
| 4 | 07-05 15:05 | `40ff8ca` | 546.0MB | PASS | - | **0.71270** | 3-way 복귀 + bucket_weights.json history_len3 스킴 blend — flat [1,1,2] 대비 **−0.0061**. 리그 +0.0095가 LB 마이너스로 역전 = enc 지분 축 신기루 3차 확인. 버킷 가중 폐기, 스테이징은 flat 복귀 |
