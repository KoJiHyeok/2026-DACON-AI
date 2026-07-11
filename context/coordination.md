# Claude-Codex Coordination

> 목적: Claude와 Codex가 같은 실험 또는 파일을 동시에 수정하지 않도록 역할, 경로, 인계 규약을 고정한다.
> 이 문서는 Claude 세션과 Codex 세션 모두 작업 시작 전에 읽는다.

## 역할

### Claude (control owner)

- 실험 우선순위, 중단/승격, GPU 배정, 공식 LB 제출을 최종 결정한다.
- `submit/`, 서버 운영 문서, canonical context 기록을 단독 소유한다.
- Codex 산출물은 reviewer/tester 검증 후 필요한 commit만 cherry-pick한다.

### Codex (independent implementation owner)

- 별도 worktree와 `codex/*` branch에서만 작업한다.
- context compiler, OOF 소비/stacker 평가, 검증 자동화처럼 Claude의 활성 GPU 작업과 분리 가능한 코드를 담당한다.
- main, 제출물, canonical 실험 대장을 직접 수정하거나 공식 제출하지 않는다.

## 현재 소유권

| Task | Owner | Status | Owned paths |
|---|---|---|---|
| args-lite e5 A/B | Claude | **done — FAIL/폐기 (exp #43)** | `colab/encoder_e5_holdout85_maxhist.py`, GPU output, full-train decision |
| mBERT hist12 Bet B | Claude | **done — FAIL/폐기 (exp #42)** | `colab/mdeberta_finetune.py`, GPU output, probe decision |
| P2 세션길이 가중 (ENC_SESSW sqrt/inv) | Claude | **done — FAIL/폐기 (exp #44·#45, 07-11)** | `colab/encoder_e5_holdout85_maxhist.py`(ENC_SESSW 게이트), GPU output, probe 판정 |
| hist12 group-OOF 생성 (P1-C 선행) | Claude | **done — 5/5 fold 완료·회수·검증, Codex 인계 (07-11 02:40, 아래 Handoff)** | `colab/encoder_e5_oof_fold.py`, `artifacts/experiments/oof_h12/**` + SHA256 manifest |
| mBERT group-OOF 생성 (P1-C 보완, 5성분 OOF) | Claude | **done — 서버 이관 후 5/5 완료·검증·인계 (07-11 13:30, 아래 Handoff 추록)** | `colab/mdeberta_finetune.py`(MDEB_FOLD 게이트), `artifacts/experiments/oof_mbert_h6/**` |
| 서버 실행·제출 | Claude | active | `docs/server_guide.md`, `submit/**`, `scripts/dacon_submit.py` |
| 공식 기록 | Claude | active | `context/experiments.md`, `context/decisions.md`, `context/submissions.md`, `context/daily/**`, `context/coordination.md` |
| context compiler v2 | Codex | ready — **⚠️ 방향 재검토 필요 (아래 07-10 밤 노트)** | `experiments/context_compiler_v2/**`, `scripts/eval_v2/**`, 전용 tests |
| hist12 stacker consumer (CX-001) | Codex | 구현 완료 — 검증 후 main cherry-pick (ded0e67·685a755) | `scripts/stacker_h12/**`, 전용 tests, handoff report |
| hist12 stacker 실물 진단 (CX-002) | Codex | **done — 비승격 판정 (exp #46, main 7e6552a). 재실험은 alpha09 sparse OOF + frozen shadow 확보 후** | 위와 동일 + `context/handoffs/codex/CX-002.md` |
| 챔피언 오답 택소노미 (CX-003) | Codex | **done — 이중검증 PASS, main 승격 (1e45be9). 가설 판정: H2·H4 종결, H1 포렌식 r2로 규칙축 종결, H3 보류 (아래 07-11 저녁 노트)** | `scripts/errtax_h12/**`, `context/handoffs/codex/CX-003.md` |

> **07-11 저녁 노트 — CX-003 판정 (Claude, control owner)**: reviewer(독립 재실행 + 수치 전수 대조 + 스코프 감사)·tester(2회 실행 결정론성 + 해시 게이트 변조 네거티브 테스트) 양쪽 PASS로 main 승격. 가설 카드 판정: **H2(110행)·H4(66행) 기각** — 카드 자체가 NO-SUBMIT(게이트 미달) 자체 판정. **H1(ask_user↔plan_task, scenario 57행 +0.0110, CI [+0.0085,+0.0138] P=1.000) 보류-우선** — 수치는 5지표 게이트를 유일하게 넘지만 oracle-flip 시나리오라 실행 가능한 개입(라벨 생성 규칙 재구성 = 포렌식 라운드 계열)이 나와야 승격 대상. **H3(explore 오라클 갭 +0.1126) 보류** — 갭은 최대지만 재시도 금지축(탐색 specialist·템플릿 라우팅) 최근접이라 새 개입 형태 없이는 착수 금지. 부기: 리포트에 기재된 codex tester JSON 해시(41d23b96…)는 우리 재실행 해시(30ba7539…)와 불일치 — 수치는 전부 일치라 무해하나 "byte-identical" 주장은 미검증으로 남김. **→ 추록 (같은 날 저녁): 포렌식 r2(`reports/forensics_r2.md`, reviewer 재실행 PASS)가 ask/plan 고순도 규칙 부재를 실증 — H1도 규칙/override 축 종결.** 시드 앙상블 s43/s44 npz는 `colab_out/holdout_e5_h12_s43.npz`(sha `001e8b95…`, solo 0.73737)·`_s44.npz`(sha `6bcecf42…`, solo 0.74109)로 아카이브(exp #48 폐기, 재프로브 금지 — 증류 teacher 후보로만 보존).
>
> **07-11 새벽 노트 (Claude, control owner)**: ① P2 sessw도 FAIL(#44 sqrt 5지표 전패 / #45 inv solo +0.0037→블렌드 역전) — **e5 단일 성분 개선 축 4연속 역전으로 전면 종결**, 남은 구조 레버는 P1-C 스태커뿐. ② CX-001은 Claude 측 reviewer·tester 독립 재검증 PASS 후 main 승격 — 단 tester가 **미문서화 갭 1건** 발견: npz 내부 id↔probs 행 순서가 어긋난 경우 소비측 구조 검증으로 탐지 불가(생성기 신뢰 경계). 완화책 = 생성측(`encoder_e5_oof_fold.py`)이 같은 `va` 순회에서 ids/probs를 동시 저장 + 인계 시 npz SHA256 manifest 고정. Codex는 CX-001.md Known limitations에 이 갭 추가 검토. ③ e5 OOF(서버)·mBERT OOF(Colab, 같은 fold_map) 동시 생성 중 — 5성분 스태커 입력(강의 갭 분석 §입력설계) 중 GPU 산출물 2종이 이번 밤에 나온다.
>
> **07-11 오후 노트 — 재현성 감사 (Claude, control owner)**: 챔피언 5성분 학습 코드 보유 현황 실사. ① linear ✓ (`ai-2026/submit/features.py` 실존 + `scripts/linear2` 재현 검증 exp #32 + `ai-2026/src/run_oof_rebuild_2026_07_04.py`) ② e5 ✓ (`colab/encoder_e5_holdout85_maxhist.py` full 모드) ③ mBERT ✓ (`colab/mdeberta_finetune.py`) ④ AU char_wb ✓ (`scripts/league4/common.py` 학습 프로토콜) ⑤ **AAR stacker ✗ — 트레이너(`train_tscar.py`)는 동료 리포 `C:\dev\dacon-agent-action-api-boost-work2`에 있었고 해당 디렉토리는 삭제됨** (oof_lab이 sys.path로 임포트하던 구조, 사본·원본 리포 이 머신에 없음). 메타데이터는 풍부(성분명·inner_weights·greedy_blend·logreg stack·3fold — `artifacts/oof/*/meta.json`)해서 재구현은 가능하나 정확 재현 보장 없음. **재현 코드 제출(7/20)은 상위 12팀만 해당** — 컷 진입 시나리오에서만 필요한 보험. fallback 실측: stk 제외 3-way+AU = 리그 **−0.00533**, linear 제외 = −0.00175.
>
> **07-10 밤 노트 (Claude, control owner)**: 서버 실측으로 e5 입력 강화 축이 3연속 블렌드 역전으로 종결됐다 — #41 maxlen512 −0.0010, #42 mBERT h12 −0.0011, #43 args-lite −0.0027(5지표 전패). **compiler v2의 pair-order·동적 token allocation은 같은 축의 변형**이라 사전 기대값이 크게 낮아졌다. Codex 권장 우선순위: ① stacker consumer 준비를 앞당기고(입력 스키마·mock OOF로 선개발 가능) ② compiler v2는 e5 입력 변형이 아니라 **stacker 피처 뷰**(직렬화가 아닌 확률·전이·구조 피처 결합)로 방향 전환 검토. 최종 결정 전 Claude 판정 대장(exp #41~43) 참조.

소유권 표에 없는 기존 파일은 수정 전에 Claude가 owner를 지정한다. 특히 Codex는 `submit/**`, 활성 `colab/*` 파일, canonical context 기록을 수정하지 않는다.

## Codex Worktree

- Path: `C:\dev\codex-context-v2`
- Branch: `codex/context-v2`
- Base commit: `35fdb15107b09ba31b54c04234da157def722db5`
- 대형 추적 아티팩트 복제를 피하기 위해 sparse checkout을 사용한다.
- main의 미커밋 변경은 이 worktree에 포함되지 않는다. 필요한 입력은 파일 자체를 복사하지 말고 명시적 commit 또는 해시가 고정된 artifact 경로로 인계한다.

## 파일 충돌 방지 규칙

1. 한 경로에는 동시에 한 owner만 둔다.
2. Codex는 main에서 작업하지 않고 자기 branch에만 commit한다.
3. Claude는 Codex branch를 merge하지 않고 우선 commit 단위로 리뷰한 뒤 cherry-pick한다.
4. 모델, NPZ, zip은 Git에 추가하지 않는다. `artifacts/experiments/<task-id>/` 같은 task별 경로와 SHA256 manifest로 인계한다.
5. `context/experiments.md` 등 canonical 문서는 Claude만 갱신한다. Codex는 `context/handoffs/codex/<task-id>.md`로 결과를 전달한다.
6. 작성자와 다른 reviewer/tester가 검증하기 전에는 main 승격 또는 제출하지 않는다.

## Task Ticket

```text
Task ID:
Base SHA:
Owner:
Goal:
Owned files:
Forbidden files:
Inputs and SHA256:
Validation / gate:
Definition of Done:
Next owner:
```

## Handoff

Codex 결과는 아래 형식으로 Claude에 전달한다.

```text
Task ID:
Branch / commit:
Files changed:
Inputs and artifact hashes:
Validation results:
Known limitations:
Recommended decision:
Required reviewer/tester checks:
```

## Handoff — OOF-H12 실물 인계 (Claude → Codex, 2026-07-11 02:40)

```text
Task ID: OOF-H12 (P1-C 선행 — CX-001 stacker consumer 입력)
Branch / commit: main d9a79fa 시점 생성. 산출물은 git 외부 artifacts/experiments/oof_h12/ (rule 4)
Files changed: oof_fold{0..4}.npz + fold_map.csv + run_oof_fold{0..4}.json + SHA256SUMS
Inputs and artifact hashes (SHA256):
  fold_map.csv  56074c16c400fbccc389e15c01c05adc4db810533516340f15e9826dd44fe295
  oof_fold0.npz 8ad89e6329b4d992cd79cb8695151ba2a48bfb092925b600db9aadae6acaff94
  oof_fold1.npz 7c0ac6411c43709d416f1e28ac33a16867a1ee8851e0d338fe2f8dbf7856e220
  oof_fold2.npz 97470e9d2cdce188e406413c75fd41fabd950e34909f9473636c31a35fe2c282
  oof_fold3.npz c34bd72709ac6a81e3c63090fd82021509b27a246caf4111f911c149122df0f6
  oof_fold4.npz 324ce7ef3de2b12f14fba975a42cf88e117dfae1ab4598b85e9b2585b5b487f8
Validation results:
  - 레시피 = 배포 챔피언 동일 (hist12/6ep/b16/lr2e-5/384/s42, SESSW=none, serialize import 단일소스)
  - fold Macro-F1: 0.73484 / 0.73565 / 0.73750 / 0.73700 / 0.73325 (holdout 0.73617과 정합)
  - 합계 70,000행, fold_map id 집합 완전 일치, 확률 정규화 OK, 세션 그룹 누수 0 (독립 awk 검증)
  - CX-001 --validate-only PASS (validate_real/validation.json — 입력 해시 전부 위와 일치)
Known limitations:
  - baseline_origin은 legacy_linear_proxy만 확인(비-parity plumbing) — alpha09 sparse OOF manifest 별도 공급 전까지 teammate_parity=false 유지
  - id↔probs 페어링은 생성기 신뢰 경계 (07-11 새벽 노트) — 위 SHA256으로 사후 변조만 커버
Recommended decision: Codex는 이 입력으로 stacker 학습·진단 진행 가능. 점수 주장·승격은 frozen shadow/outer-fold 평가 + Claude 승인 필요 (promotion_eligible=false 유지)
Next owner: Codex (scripts/stacker_h12), 판정·승격·제출은 Claude
```

## Handoff 추록 — OOF-MBERT-H6 (Claude → Codex, 2026-07-11 13:30)

```text
Task ID: OOF-MBERT-H6 (OOF-H12 인계의 5성분 확장 — CX-002 선택 항목 6의 실물 입력)
Files: artifacts/experiments/oof_mbert_h6/oof_mbert_fold{0..4}.npz + run json + SHA256SUMS
  (fold_map은 OOF-H12와 동일 파일 공유 — sha 56074c16…)
Inputs and artifact hashes (SHA256):
  oof_mbert_fold0.npz e228346dfd204fbcfe99143484ca850de4d0d60a6405ef6f5b406975451a9563
  oof_mbert_fold1.npz fb0e068e418552ce1cd11c59f20385c827c57fd9b5d89e883798874bfb8fe13a
  oof_mbert_fold2.npz 098f3e5002e582c521972b91e454dd83283dbb1f6ede470d7d7174b352828989
  oof_mbert_fold3.npz 964ba24df48347d7acd6419c583f29a3c785bea50561b24e5ebf969184d3cc3b
  oof_mbert_fold4.npz 4b064e844e2b75c27592b4082e213ba31efa29273b3715b1762a32726c52971a
Validation results:
  - 레시피 = 배포 mBERT 계약 (bert-base-multilingual-cased, hist6, 2ep/b8×2/384/s42, fp32)
  - 전 fold 서버 A5000 단일 환경 생성 (Colab 혼용 금지 판정 근거: fold0 교차 실측
    Colab↔서버 F1 동률이나 argmax 8.1% 상이 — fold 간 환경차는 스태커 입력 오염)
  - fold Macro-F1: 0.66153 / 0.66432 / 0.66438 / 0.67062 / 0.67027 (holdout 0.67147 정합)
  - 합계 70,000행, fold 배정·actions 순서(e5 OOF와 동일)·run json 전항목 정합
  - Colab산 fold0은 oof_mbert_fold0_colab.npz로 보존 (교차 환경 검증용, canonical 아님)
Recommended decision: CX-002의 mBERT 성분 플래그 개발 시 이 세트를 사용. e5와 동일
  fold_map이므로 행 정렬 검증 로직 재사용 가능.
Next owner: Codex (scripts/stacker_h12), 판정·승격·제출은 Claude
```

## 현재 실행 순서

1. ~~Claude가 args-lite와 Bet B 서버 작업을 계속 소유한다.~~ → 07-10 밤 완료: 둘 다 FAIL/폐기 (exp #42·#43). Claude의 활성 GPU 레인은 P2 sessw → hist12 OOF 생성 순.
2. Codex는 args-lite를 다시 구현하지 않는다. ~~context compiler v2에서는 pair order와 동적 token allocation을 독립 변수로 준비한다.~~ → **보류 권고** (소유권 표의 07-10 밤 노트 참조 — e5 입력 강화 축 3연속 역전으로 종결. pair-order는 D-011 게이트 규칙상 미개봉).
3. Claude가 동일 fold의 hist12 OOF를 생성하고 manifest와 함께 넘긴다.
4. Codex는 OOF를 읽는 stacker/진단 파이프라인만 구현한다.
5. Claude reviewer/tester가 Codex commit을 검증한다.
6. Claude가 통과한 commit만 main에 cherry-pick하고 GPU 승격 또는 LB 제출을 결정한다.
