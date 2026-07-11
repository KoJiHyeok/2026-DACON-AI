# Claude 세션 재개 프롬프트 (2026-07-11 저녁 — 신규 축 2종 학습 중 인계)

> 새 세션 시작 시 이 파일 전체를 읽고 그대로 이어간다. 규칙 원천: CLAUDE.md + context/coordination.md (반드시 먼저 읽기).

## 역할·규칙 요약

- 나(Claude) = control owner: 실험 큐·GPU 배정·공식 기록(context/*)·submit/·LB 제출. Codex는 codex/* 브랜치, main 수정 금지, 산출물은 검증 후 cherry-pick.
- 리뷰·테스트는 reviewer/tester 서브에이전트(Sonnet 고정) — 작성자 자기검증 금지. GPU 투입 전 코드 변경은 reviewer 승인 필수 (기존 env 파라미터만 쓰면 코드 변경 아님).
- 커밋 전 `git status/log` 재확인. 트레일러: `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`. 모델·NPZ·zip git 금지 — `artifacts/experiments/<task>/` + SHA256 (artifacts/는 gitignore, 해시는 coordination Handoff 블록에 영구화).
- **사용자 지시: 동료(팀원) 요청은 배제** — alpha09 OOF·동료 코드 의존 계획 금지.

## 학교 서버 (키 인증 복구됨)

- `ssh -p 419 u20220876@210.119.108.236` — 07-11 저녁 관리자가 **홈을 `/mnt/ssd2/u20220876`로 이동 (3.3TB 여유, 디스크 제약 소멸)**. 키 재등록 완료.
- 자산은 구 홈에서 새 홈으로 복사 완료(data/dacon/out/backup/.cache/run_*.sh). **venv는 재배치 불가라 구 홈 절대경로로 source**: `source /home/u20220876/venv-dacon/bin/activate`.
- GPU 4장 중 동시 2장 규약 유지. 코드는 `~/dacon/colab/`(scp 동기화, git 아님).

## 지금 돌아가는 것 (2026-07-11 18:24 투입)

1. **GPU0 = klue/roberta-large + hist12** (tmux `klue`, 로그 `~/out/klue_large_h12.log`) — 남은 유일한 도약 후보 축(백본 대형화×hist12, 둘 다 미조합). 완료 신호 `[npz] holdout_mdeb.npz ... final macro-F1` → `~/out/klue_large_h12/holdout_mdeb.npz`. ETA ~20:30.
   **판정 체인**: scp 회수 → 리그 5지표, **e5 슬롯 교체 프로브** (probe_c 패턴: 대조군=colab_out/holdout_e5_h12.npz 스왑 표면 0.75601, 후보=klue npz를 e5 자리에). 비교 기준 e5 solo 0.73617. 배포 시 zip = 867−573+674 ≈ 968MB ✓ (mBERT 유지).
2. **GPU1 = e5 seed43 hist12** (tmux `e5s43`, 로그 `~/out/e5_h12_s43.log`) — 시드 확률 앙상블 축(soup=weight-space만 폐기, prob-space 미시도). npz `~/out/e5_h12_s43/holdout_e5_h12.npz`. ETA ~19:10.
   **판정 체인**: 회수 → avg(s42,s43) npz 생성(로컬 s42=colab_out/holdout_e5_h12.npz) → 5지표 프로브(신규 스크립트 필요 — probe_c 복제, reviewer 검증). ⚠️ 통과해도 **2×e5 fp16은 zip 1GB 초과** — 배포는 증류(student 재학습) 경로만, 시간 비용 감안해 판정.
3. **Codex = CX-003 오답 택소노미** 실행 중 — 챔피언 홀드아웃 오답 2,451행 구조 분석 → 가설 카드 3~5개. 입력: `artifacts/experiments/errtax_h12/` (preds csv sha ce1bee87…, probs npz sha 77476558…). 완료 시 reviewer/tester 이중검증 후 가설 채택 판정.
4. **모니터**: 학습 감시 + 디스크 경보 운용 중이었음 — **세션 바뀌면 재장전** (npz 경로·tmux 이름 위 참조).

## 오늘 확정 사항 (재시도 금지 참조)

- exp #44/#45 sessw 폐기, #46 스태커 전역 대체 비승격, **#47 OOF 국소 보정 폐기 → P1-C 축 종결** (champion이 보정 대상을 이미 흡수). 재시도 금지 테이블 갱신됨.
- CX-001/002 main 승격 (ded0e67·7e6552a), 전체 스위트 30 passed (stale merge080 테스트 제거 d9a79fa).
- OOF 자산 완비: e5 5-fold (`artifacts/experiments/oof_h12/`) + mBERT 5-fold (`oof_mbert_h6/`, 서버 단일환경) — 해시는 coordination Handoff 블록.
- **D-013: 최종 제출 선택 = 챔피언 0.7623 (대장 #11) 단독** (선택 1개 제한 확인). 마감 7/15 09:59 전 제출 페이지에서 지정 확인만 남음. 새 후보가 5지표+LB 게이트를 통과하면 교체 재판정.
- 재현성 감사: 5성분 중 AAR stacker 트레이너만 부재(동료 work2 삭제) — 재현 제출은 top-12 조건부라 보류. fallback(stk 제외) 리그 −0.0053 실측.
- 평가 확정(스크린샷): 예선=Private LB 100%, 본선=Private 50%+추론속도 10%+전문가심사 40%, 코드검증 실격 시 차순위 승계.

## 상황판

- LB: 우리 0.7623 (팀 최고) / 컷(12등) ~0.7906 / 갭 −0.028 / 예선 마감 7/15(수) 09:59.
- 챔피언 리그 표면 0.75601 (e5-hist12 4-way + soft-AU α0.9). 판정은 항상 5지표 게이트 (row +0.005 & CI 하한>0 & MC>0).
- 제출 예산: 게이트 통과 후보 없으면 쓰지 않는다. 통과 시 full-train(`ENC_MODE=full`) → `/submit` (make_submit.py 경유, 수동 zip 금지).
- Artifact 상황판(공유용): https://claude.ai/code/artifact/29a01fc5-96bf-4044-b739-34524cf2e9a5 — 큰 변동 시 갱신.
