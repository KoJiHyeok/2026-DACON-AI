# Claude 세션 핸드오프 — 2026-07-13 새벽 (Qwen 문샷 제출 직전)

> 새 세션 시작 시 이 문서 + `context/INDEX.md` + `context/coordination.md`를 먼저 읽어라.
> 지금은 **제출 크리티컬 패스 한가운데**다 — 아래 "즉시 할 일"부터 처리.

## 한 줄 상황

D-2 (예선 마감 **7/15 09:59**). 챔피언 LB 0.7623(89등, 컷 0.79326, 갭 −0.031). 밤새 문샷 성공: **exp #52 Qwen2.5-0.5B 하이브리드가 holdout +0.01160으로 엄격 게이트 통과** (reviewer·tester 이중검증 PASS) → 지금 make_submit 게이트가 분리 프로세스로 돌고 있고, **사용자가 "게이트 통과하면 바로 제출" 지시함**.

## 즉시 할 일 (순서대로)

1. **게이트 결과 확인**: `C:\Users\wlgur\AppData\Local\Temp\claude\C--dev-2026-AI-DACON\82ae2973-4a60-4992-8679-1b8f4b2f0948\scratchpad\make_submit.log.status` — `GATE_OK`/`GATE_FAIL`. 같은 폴더 `make_submit.log`에 상세. (모니터 blolta8vr가 감시 중이었음 — 새 세션이면 파일 직접 확인)
2. **GATE_OK면 즉시 제출** (사용자 사전 승인):
   ```
   .venv\Scripts\python.exe scripts\dacon_submit.py --check
   .venv\Scripts\python.exe scripts\dacon_submit.py --file submit\submit.zip --memo "exp #52 Qwen2.5-0.5B hybrid (encoder block=Qwen solo, no mBERT) — holdout +0.0116, strict gate pass" --yes
   ```
   제출 후: 대장(`context/submissions.md`) #13행은 make_submit이 이미 기록 — zip 크기·커밋 확인, git push.
3. **GATE_FAIL이면**: 최우선 의심 = **zip > 1GB** (model/ raw 1017MB, zip 예상 990~1005MB 턱걸이). 대응 순서: ① `submit/model/encoder/chat_template.jinja` 삭제(불필요) ② 그래도 초과면 stacker(AAR 46MB) 제외 검토 — 단 이건 블렌드 표면이 바뀌므로 **3-way(lin+Qwen+AU) 재판정 필수** (`scratchpad/probe_qwen_hybrid.py` 변형해 stk 제외 표면으로 5지표 재실행, 엄격 게이트 통과 시에만 제출).
4. **제출 후 채점 폴링**: WebFetch `dacon.io/competitions/official/236694/leaderboard` — 팀 `jj`(직전 0.76236, 카운트 47, 89등 부근). **페이지가 89등 언저리에서 잘리므로** 88등 `팀4층`(0.76263) 대비 상대 위치로 판독. 점수 갱신 시: 대장 #13·exp #52 LB 기입, daily 기록, PushNotification. 어제 채점은 4시간+ 걸렸다 — 20~40분 간격 ScheduleWakeup으로.

## 현재 패키징 상태 (완료됨)

- `submit/model/` = **encoder(Qwen2.5-0.5B-Instruct full-70k 2ep fp16 954MB** + serialize_config.json `{"max_hist":12}` + chat_template.jinja**)** + linear(8MB) + stacker(46MB) + au_linear(11MB) + `enc_block_weights.json={"weights":[1.0]}` + weights.json [1,1,2]. **encoder_2(mBERT) 제거** → 백업 `colab_out/mbert_encoder2_backup`.
- `submit/script.py`에 pad_token 폴백 커밋됨(a352d80) — Qwen 필수, BERT류 무영향.
- KD student 인코더(제출 #12, LB 0.7621 FAIL)는 로컬 삭제 — 서버 `~/out/kd_run/student_full_3t_l05t2/model_fp16`에 보존.
- 챔피언 인코더(s42)는 서버 `~/models/champ_encoder_s42`에 SHA 검증본. 롤백 = 그걸 scp + mBERT 백업 복원 + enc_block [1.2,0.8].

## exp #52 판정 근거 (검증 완료 — 재검증 불필요)

- h85 solo: instruct-2ep **0.75932** / base-2ep 0.75941 (챔피언 e5 0.73617 +0.023). 4ep는 ep3 과적합(0.74569) — **2ep 최적**.
- 하이브리드(lin+AAR+Qwen블록+softAU α0.9) vs 챔피언(0.75601): row **+0.01160**(→0.76760), 세션균등 +0.01393, MC +0.01430±0.00685, bootstrap CI **[+0.00614,+0.01758]** P=1.000, 반반 +0.0156/+0.0075. 판정 스크립트 `scratchpad/probe_qwen_hybrid.py` (⚠️ league4 기본 e5는 stale hist6 — 반드시 `replace(data, e5=holdout_e5_h12.npz)` 후 비교).
- 독립 검증: qwen-gate-reviewer(수치 5자리 재현·SHA·오염없음·블록치환 수식 동치성) + qwen-smoke-tester(서버 300행 acc 0.8167·14클래스·pad 정상·CPU 20행 13.8s) 전부 PASS.
- 시간 제약 판단: 테스터의 "T4 29.7분" 외삽은 고정비 오류 — 진짜 앵커는 **챔피언(인코더 456M 합)이 평가서버 T4 실측 4m14s**. Qwen 494M ≈ 등가 연산 → 4~5분 예상, 10분 내 안전.

## 병행 중인 것

- **Codex 밤샘 러너** (스케줄러 DACON-night-shift 등록됨, 30분 자기치유): `night/2026-07-13/task{1,2,3}.md` = AAR 속도 최적화 / 재현 리허설 / 발표 사료. 아침 회수 시 reviewer·tester 검증 → 병합. 워크트리 `C:\dev\night\2026-07-13\`.
- 서버 GPU 전량 유휴 복귀. 자산: `~/out/qwen05i_2ep_{h12,full}`, `qwen05b_2ep_h12`, KD 일체, teacher 덤프, h85 ckpt들. venv-speed(torch 2.7.1+tf 5.13.1+accelerate).

## 이 세션에서 배운 함정 (재발 주의)

1. **백그라운드 Bash 태스크는 세션 이벤트에 킬당한다** → 장기 작업(학습·push·make_submit)은 `Start-Process cmd /c 래퍼.cmd` 분리 실행 + `.status` 파일 패턴. (OOF 체인·git push·make_submit 3회 실증)
2. `env VAR=x cmd &`의 env는 pkill 패턴에 안 잡힘 — 죽일 땐 pid, 기다릴 땐 산출물 마커.
3. transformers 5.13: 토크나이저 없는 checkpoint 디렉토리에서 **vocab=5 깡통 토크나이저를 조용히 생성** — dump/추론 유틸엔 vocab 하한 assert 필수 (`scripts/kd_run/dump_probs.py` 참조).
4. WebFetch 리더보드는 89등 부근에서 잘리고 소형 모델이 오독함 — 이웃 행(팀4층) 상대 위치로 판독.
5. `git push`가 행처럼 보이면 LFS 업로드다 (`submit/*.zip` LFS 추적) — `lfs.fetchexclude=submit/*.zip` 설정해서 워크트리 팽창은 막아둠.
6. 디스크 상시 부족 (C: 7GB 안팎) — G4 실패 시 `%TEMP%\dacon_submit_*` 잔해 정리부터.

## 규율 (불변)

- 제출은 make_submit 경유만, 엄격 게이트(row≥+0.005 & CI하한>0) 통과 후보만 (D-014 완화는 #51 실패로 사실상 폐기 — #52는 엄격 통과라 무관).
- 최종 선택 = public 최고 자동 (D-013 추록 2) — #52가 0.7623 넘으면 자동 교체 (의도된 것).
- 판정은 작성자와 다른 reviewer/tester 이중검증 후에만 canonical 기록.
- 기록 없으면 일어나지 않은 것: experiments.md(#52까지 기록됨)·submissions.md·daily/2026-07-13.md 갱신 유지.
