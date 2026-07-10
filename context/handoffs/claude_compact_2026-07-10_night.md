# Claude 세션 재개 프롬프트 (2026-07-10 밤 — 서버 자동 루프 인계)

> 새 세션 시작 시 이 파일 전체를 읽고 그대로 이어간다. 규칙 원천: CLAUDE.md + context/coordination.md (반드시 먼저 읽기).

## 역할·규칙 요약

- 나(Claude) = control owner: 실험 큐·GPU 배정·공식 기록(context/*)·submit/·LB 제출. Codex는 codex/* 브랜치에서만, main 직접 수정 금지, 산출물은 cherry-pick.
- 리뷰·테스트는 reviewer/tester 서브에이전트 — 작성자 자기검증 금지. GPU 투입 전 코드 변경은 reviewer 승인 필수.
- 커밋 전 `git status/log` 재확인(병렬 세션 활동 있음). 커밋 트레일러: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- 모델·NPZ·zip은 git에 추가 금지 (coordination rule 4). 인계는 `artifacts/experiments/<task>/` + SHA256 manifest.

## 학교 서버 (직접 제어 가능)

- `ssh -p 419 u20220876@210.119.108.236` — **키 인증 완료** (BatchMode=yes로 비대화 실행 가능). scp도 `-P 419`.
- 홈 모드: ssd1~3 권한 없음(관리자 요청 중), 루트 98% 사용·잔여 ~23GB → **런 종료 즉시 ckpt 삭제** 정책. GPU는 A5000×4 중 **2장까지만**(공용 서버). 레시피(에폭/배치/lr/maxlen) 변경 금지.
- venv: `source ~/venv-dacon/bin/activate` (torch 2.13+cu130, transformers 5.13). 데이터 `~/data/`, 코드 `~/dacon/colab/`(git 아님 — scp로 동기화, 로컬 수정 시 재전송 필수), 산출물 `~/out/`, 코드 백업 `~/backup/dacon-main-35fdb15.bundle`.
- tmux 세션으로 학습(노트북 무관). 디스크 8GB 미만 경보 모니터 운용 중이었음 — 새 세션에서 Monitor 재장전.

## 이 시점의 진행 중 작업

1. **P2 sessw (감사 P2)**: GPU0=`ENC_SESSW=sqrt`, GPU1=`inv` 학습 중 (tmux swsqrt/swinv, 로그 `~/out/e5_h12_sw{sqrt,inv}.log`). `[losscheck] sw present=True` 검증 통과. **완료 체인**: `[npz] holdout_e5_h12_sw*.npz` 확인 → scp로 `colab_out/`에 회수 → `scripts/league4/probe_c_args_lite.py`를 `E5_H12_ARGS=<후보 npz>` env로 실행(대조군=배포 hist12) → 5지표 게이트(row +0.005 & bootstrap CI 하한>0 & MC 평균>0) → exp #44/#45로 experiments.md 기록·커밋 → 서버 `ckpt_sw*` 삭제.
2. **OOF 생성 (P1-C 선행, coordination 실행순서 #3)**: `colab/encoder_e5_oof_fold.py`(신규) + 래퍼 `run_oof.sh`(스크래치패드) — **reviewer 검토 대기 중**. 통과 시 사용자 지시대로: 수정 반영·커밋 → base 스크립트와 함께 서버 전송 → `OOF_MAP_ONLY=1`로 `~/out/oof_h12/fold_map.csv` 사전 생성(그룹 무결성 assert 확인, CPU라 P2와 병행 가능) → P2 종료 후 tmux로 `~/run_oof.sh 0 0 1 2`(GPU0) / `~/run_oof.sh 1 3 4`(GPU1) → fold당 npz 확인·fold 디렉토리 삭제 자동 → 5개 npz 병합 + SHA256 manifest → `artifacts/experiments/oof_h12/` → coordination Handoff 형식으로 Codex 인계.
3. **push 차단 상태**: origin은 54+커밋 뒤. GH008 — 히스토리의 LFS zip 3개(각 868MB)를 GitHub이 요구. 사용자 결정="코드만"(2.6GB 업로드 거부). LFS 정리(히스토리에서 zip 제거)는 **Codex worktree(base `35fdb15`)와 조율 후에만**. 그전까지 원격 백업은 서버의 git bundle로 대체됨.

## 오늘 밤 확정된 결과 (재시도 금지 참조)

- exp #41 maxlen512 −0.0010 / **#42 mBERT hist12 −0.0011** / **#43 args-lite 5지표 전패** — 전부 solo↑·블렌드 역전. **e5 입력 강화 축 종결**. pair-order는 D-011 게이트상 미개봉. coordination에 Codex compiler v2 보류 권고 노트 기록됨.
- 남은 구조 레버: **hist12-aware stacker**(OOF 선행) + **P2 sessw**(진행 중). 챔피언 LB 0.7623 유지. 12등 ~0.79(Codex 보고, 공식 미검증).
- 템플릿 완전일치 라우팅 폐기(exp #40). threshold/calibration 가족 D-009 veto 유지.

## 사용자 대기 항목 (독촉 금지, 오면 즉시 처리)

- 평가탭 심사기준 **스크린샷** → Public/Private 정의 확정 → 최종 제출 선택 전략(강건성 vs public 최고) 재판정 (D-012)
- 동료 학습 코드(linear 트레이너/mBERT 파인튜닝/AAR stacker) — **재현 코드 제출 7/20(월) 10:00** (D-012)
- ssd 권한 (`sudo mkdir /mnt/ssd2/u20220876 && chown ...`)

## 제출 관련

- `scripts/dacon_submit.py` + DACON_TOKEN/DACON_TEAM env — API 제출 가능(일 10회). 반드시 `make_submit.py` 게이트 경유, 수동 zip 금지. P2 승격 시에만 full-train 재학습(`ENC_MODE=full`) → /submit.
