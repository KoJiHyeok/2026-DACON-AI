# Claude 세션 핸드오프 — 2026-07-15 새벽 (D-day 예선 마감)

> 새 세션 시작 시 이 문서 + `context/INDEX.md` + `context/coordination.md`를 먼저 읽어라.
> **오늘 09:59 예선 마감** — 아래 "즉시 할 일"이 마감 크리티컬 패스다.

## 한 줄 상황

D-day (**예선 마감 07-15 09:59**). **팀 최고 0.77350 확정·자동 선택** (하루에 0.7623→0.77350, +0.0112, 6연속 갱신). 오케스트레이션 Fable→**Opus 이관** (사용자 지시 07-15). 남은 축 전부 실측 종결 — s44 시드 확인런만 학습 중, 판정 대기.

## 즉시 할 일 (순서대로)

0. **🔥 발악: 남은 제출 전량 소진 (사용자 07-15 지시).** 오늘 제출 10회 리셋됨. 자동선택이 0.77350을 지키므로 **모든 드로는 다운사이드 0**. 안 써본 축 = **linear/stacker 가중** (지금껏 weights [1,1,x]로 lin/stk=1 고정). 스윕(`%TEMP%\linstk_sweep.py`) 결과 홀드아웃 배포본(0.77227) 상회 구성 다수. **제출 큐 (weights.json=[wl,ws,we] 3슬롯 + calib.json bias, α0.85 고정)** — 서로 다른 (wl,ws,we) 영역으로 다양성 확보해 public/private 분산 도박:
   | 순번 | weights [wl,ws,we] | bias (list,glob) | holdout row |
   |---|---|---|---|
   | A | [0.9, 1.0, 3.0] | 0.10, 0.12 | 0.77278 (+0.00050) |
   | B | [0.9, 1.1, 3.2] | 0.10, 0.10 | 0.77274 (+0.00046) |
   | C | [0.9, 0.9, 2.8] | 0.10, 0.12 | 0.77255 (+0.00027) |
   | D | [1.1, 1.1, 3.2] | 0.10, 0.10 | 0.77244 (+0.00017) |
   | E | [0.8, 1.0, 2.8] | 0.10, 0.10 | 0.77247 (+0.00020) |
   | F | s44 통과분 (아래 1번) | — | — |
   각각 weights.json/calib.json 수정 → make_submit(CRLF 분리 러너) → dacon_submit → 대장 기록 → 스테이징 원복. **매 제출 게이트 ~11분**, 채점 병렬 ~45-75분. 마감 09:59 역산: **늦어도 08:30까지 마지막 제출** (채점 완료 여유). LB 갱신 시 대장·daily 기록.

1. **s44 h85 판정** (학습 중 GPU0, ~03:30 완료): 분리 폴러 pid 17216이 완료 시 `colab_out/qwen_i2ep_h85_s44.npz` 자동 회수 + `scratchpad/s44_h85.status` 생성. 판정 = `%TEMP%\s43_judge.py`의 s43→s44 치환 (배포 #17 표면 기준선 0.77227 대비 5지표). **통과(row>0 & CI하한>0) 시에만** full-s44 학습(~2.5h)→배포→제출. **s43이 solo 0.748로 크게 FAIL(exp #58)했으므로 s44도 기대 낮음** — Qwen 0.5B 시드 민감성 큼, s42가 상단 시드. NO-GO면 그냥 폐기(제출 낭비 0).
2. **마감 의식 (~08:30, s44 결과 무관하게 필수)**:
   - `python scripts/freeze_check/check.py` → exit 0 확인 (기대값 #20/0.77350으로 갱신됨)
   - LB 최종 스냅샷 (WebFetch, 팀 jj)
   - **DACON 제출 페이지에서 최종 선택이 0.77350인지 눈으로 확인** (자동 public 최고 — D-013, 사용자 07-14 재확인)
3. **개별 점수 미확인 항목**: #19(w2.9)·#20(w3.1) 중 어느 것이 0.77350인지 제출 페이지 미확인. 대장에 "≤/최고" 표기만 있음 — 확인되면 대장 확정.

## 오늘 성과 (07-14~15, 대장 #14~#23)

| # | 구성 | LB | 비고 |
|---|---|---|---|
| 14 | Qwen 하이브리드 + 속도레버 | 0.77089 | 시간초과 #13 회복, T4 리허설 |
| 15/16 | 블렌드 재튜닝 w3/w4 | 0.77237/0.77216 | exp #53, +0.00147 전이 |
| 17 | w3 + list/glob bias +0.08 | **0.77302** | exp #55(CX-B), 최종 자동선택 후보 |
| 18 | w2.5 | 0.77230 | 고원 확인 |
| **19/20** | **w2.9/w3.1 + bias** | **0.77350** | **max-draw 최고** 🏆 |
| 21/22/23 | 비대칭·마이크로 bias | ≤0.77350 | 미갱신 |

**평가 T4 실측 6분37초** (리허설 8.6분보다 빠름 — 3vCPU 효과, 여유 3.4분).

## 배포 상태 (중요 — 재현/롤백)

- **최고 0.77350 = #19(weights[1,1,2.9]) 또는 #20([1,1,3.1]) + calib bias(list+0.08,glob+0.08) + AU α0.85**. 인코더 = **Qwen2.5-0.5B-Instruct s42** (`colab_out/qwen_i2ep_h85.npz` solo 0.75932, 서버 `~/out/qwen05i_2ep_full`).
- ⚠️ **`submit/model/weights.json`·`calib.json`은 gitignore** → 커밋으로 구성 복원 불가. **구성은 experiments.md #53·#55 메모로만 재현.** 대장 커밋 해시는 코드만 가리킴.
- ⚠️ **현재 스테이징 = weights[1,1,3.05] + calib bias(0.10,0.12)** = #23 드로(미갱신). 다음 제출용일 뿐 자동선택엔 무관. **재현 시 최고 구성으로 되돌릴 것.**
- 인코더 블록 = Qwen 단독 (mBERT 제거, `enc_block_weights [1.0]`, serialize max_hist 12). script.py에 길이정렬 배칭 + fast_aar 통합(둘 다 출력등가 검증).

## 종결된 축 (재시도 금지 — 전부 이중검증 FAIL)

- exp #54 Qwen 블록 캘리브레이션 (NLL 개선 argmax 비전이)
- exp #56 au_linear 업그레이드 (현행 char-C1이 OOF 1위)
- exp #57 Qwen2.5-Coder 스왑 (solo·하이브리드 전패, 코드 특화 사전학습이 이 분류엔 범용판 이하)
- exp #58 시드 s43 (solo 0.748, −0.011)
- 블렌드 가중·α·bias 강도 그리드 소진 — holdout에서 배포본 넘는 구성 없음

## 진행 중 / 자산

- **s44 h85 학습** (서버 GPU0, ~03:30 완료). 폴러 `scratchpad/poll_s44.cmd` pid 17216.
- s43/full 중단됨. 서버 자산: `~/out/qwen05i_2ep_{h12,full}`(s42), h85 s43 npz, kd·teacher 일체. GPU 3장 유휴 복귀 예정.
- 밤샘 CX 산출물 3종 회수·병합됨 (`scripts/cx_calib`·`cx_errloc`·`cx_au2`). freeze_check·플레이북·발표사료 현행화 완료.

## 함정 (재발 주의)

1. **`.cmd`는 CRLF 필수** — Write 도구는 LF 저장, cmd가 무출력 exit 255 즉사 (메모리 `write-tool-cmd-crlf`). 작성 후 `-replace "`r?`n","`r`n"` + UTF8 no-BOM.
2. **백그라운드 Bash 워처는 세션 이벤트에 킬당함** → 장기 감시는 `Start-Process cmd` 분리 폴러 + `.status` 파일 + PowerShell block-wait(≤9.8분 사이클).
3. **weights/calib gitignore** → 배포 구성은 반드시 experiments.md에 수치로 기록 (커밋 복원 불가).
4. WebFetch 리더보드는 87등 부근에서 이웃 행으로 판독 (jj 직접 표시될 때도 있음).
5. make_submit G2는 clean 요구 — freeze_check 실행이 자기 리포트를 재생성하니 제출 전 그 2파일 커밋 or 무시.

## 규율 (불변)

- 제출은 make_submit 게이트 경유만. 최종 선택 = public 최고 자동 (D-013, 사용자 재확인).
- 판정은 작성자와 다른 reviewer/tester 이중검증 후 canonical 기록.
- 기록 없으면 일어나지 않은 것: experiments.md(#58까지)·submissions.md(#23까지)·daily/2026-07-15 갱신 유지.
- 오케스트레이션 Opus, 서브에이전트 Sonnet 5 고정.
