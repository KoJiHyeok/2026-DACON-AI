# Experiments Log

> 규칙: 모든 실험은 (가설 → 변경점 → 로컬 CV → 리더보드) 순으로 기록.
> CV는 세션 프리픽스 GroupKFold Macro-F1. 리더보드 제출은 일 10회 예산 관리.
> **점수 판정은 LB 실측 또는 세션 group-split만** — accuracy·누수 split·단일 holdout 금지 (w112 핸드오프 계승).

## 현재 최고 기록

| 구분 | Macro-F1 | 비고 |
|---|---|---|
| 리더보드 (우리 팀) | **0.7400** | 3-way + soft-AU 라우팅 char-C1 α=0.9 (2026-07-06, 대장 #6, exp #24) 🏆. 이전: 하드-AU 0.7331 / 동료 0.7242 / 재건 3-way 0.71884 |
| 커트라인(12등) | 0.7807 | 2026.07.06 기준 (전일 0.77665 → +0.004/일로 가속 중) — 갭 **−0.048** |
| 순위 | 81등 | 핸드오프 시점 |

## 실험 기록 (w112까지의 여정 — 전부 LB 실측, 핸드오프 §1에서 계승)

| # | 날짜 | 가설/행동 | 변경점 | 로컬 CV | LB | 결론 |
|---|---|---|---|---|---|---|
| 0 | ~07.03 | 공개 베이스라인 | current_prompt만 TF-IDF+LogReg | ~0.43 (group-split) | - | 기준점. history·meta를 버림 |
| 1 | ~07.03 | 3필드 전부 사용 | linear: E_+seq 피처 + LinearSVC(C=0.1) | - | ~0.673 solo | 피처만으로 +0.24 |
| 2 | ~07.03 | 이종 view 스태킹 | stacker: AAR 4-view SGD + transition prior | holdout 0.7098 | ~0.671 solo | linear와 유사 수준 |
| 3 | ~07.03 | 사전학습 인코더 | multilingual-e5-base 파인튜닝 (full 70k, s42, max_len 384) | - | ~0.701 solo | 단일 최강 성분 |
| 4 | ~07.03 | 이종 blend | 3-way uniform [1,1,1] (v1 인코더) | - | 0.7130 | blend 시작 |
| 5 | ~07.04 | 인코더 업그레이드 | v1 → v2 s42 (full 70k) | - | 0.7190 | +0.0060 |
| 6 | ~07.04 | 인코더 지분↑ | weights [1,1,1.5] (enc 0.43) | - | 0.7200 | +0.0010 |
| 7 | 07.04 | 인코더 지분↑↑ | **weights [1,1,2] (enc 0.50) = w112** 🏆 | - | **0.7208** | +0.0008, 체감 중. [1,1,2.5]는 로컬 하락 시작점 → 후순위 |
| 8 | 07.05 | R4: explore 4클래스는 계층 분류가 낫다 (밤샘 task2, 프로토타입 = linear 단독) | 1단계 family gate(F1 0.984) → 2단계 explore 전용 분류기. flat 0.6638 → override 0.6812 → strict route **0.6883** | 5-fold SGKF **+0.0246** (explore 4클래스 각 +0.020~0.028) | 미제출 | **생존** (독립 리뷰 통과: 누수 없음·수치 재현 확인). 단 **조건부** — hard-label 스왑이라 확률 blend와 비호환 → 확률 레벨(family 확률 마스킹) 재설계 + override/strict 각각 LB 게이트 후 승격. 리포트: night/2026-07-05/task2_report.md |
| 9 | 07.05 | R3: 첫 스텝(history=0) prior 보정 (밤샘 task2) | first-step log-prior bias, λ 그리드 | 최적 λ=0.125에서 **+0.0008** (첫스텝 자체 F1은 하락) | 미제출 | **보류** — calib_v1과 같은 유형(분포 피팅 bias)이라 LB 비전이 위험 대비 이득 없음 |
| 10 | 07.05 | w112 재조립: 원본 인코더(fp16)로 3-way 복원 | ai-2026 draft(linear+stacker+weights[1,1,2]) + artifacts/enc_v2_s42 fp16 → submit/ 스테이징 (커밋 e4cd2b4) | - (재현 제출) | **0.71884** | 기준선 복구 (팀 w112 0.7208 대비 −0.002 — fp16 재변환 or 체크포인트 미세 차이 추정, 추적 비용 대비 무가치). 제출 대장 #1 |
| 12 | 07.05 | **세션 형제 행 라벨 복원**: step k 라벨 == step k+1 history 마지막 assistant_action.name | train 검증 스크립트 (스크래치) | **train 58,326/58,326 쌍 100.00% 성립** (sim·au 계열 모두, gap1~6 231,664쌍 무예외) | **0.71884 (델타 0)** | **이득 0 확정** — test는 세션당 1스텝 샘플링(형제 행 없음). 상위권 0.79도 순수 모델링이라는 뜻. 폴백 안전하므로 보험 코드는 script.py에 잔류 (D-008). 점수 기대는 인코더 v3·R4로 이동 |
| 11 | 07.05 | blend 그리드 도구 (밤샘 task3) | scripts/blend/{collect_probs,grid_blend}.py — 성분별 holdout 확률 npz + 가중 그리드 | ⚠️ 수치 판정 불가 — stacker가 full-train 아티팩트라 holdout 누수 (0.7385는 오염값) | 미제출 | 도구만 채택. 사용하려면 성분 전부를 85% split로 재학습한 npz 필요 |
| 13 | 07.05 | **로컬 LB 시뮬레이션 리그 완성** | 평가행 = job3 holdout_base.npz의 9,969행, linear·stacker는 정직 OOF(artifacts/oof) id 조인 | 3-way [1,1,2] = **0.71726** vs 실제 LB 0.71884 (**오차 0.0016**) | - | 채택 — 후보 스크리닝은 LB 소모 없이 리그에서. 단 #15의 축별 한계 참고 |
| 14 | 07.05 | R4 확률 레벨 통합 (r4-integrate) | V1 override(gate행만 explore 질량 재배분) / V2 soft route(전 family 재조정) — 3-way 리그 베이스 | V1 **−0.0073**, V2 **−0.0158** (explore 4클래스 전부 하락) | 미제출 | **폐기** — linear 단독의 +0.0246은 강한 베이스에서 역전. 인코더+stacker가 explore를 specialist보다 잘 풂. 재시도 금지행 |
| 15 | 07.05 | serialize v3 (args+hist12+lang+elapsed) 프록시 A/B | scripts/encoding/serialize_ab.py — TF-IDF+LinearSVC 3-fold, 70k | v2 0.5017 / v3 **−0.031** / v3_no_args −0.026 / v3_hist6(v2+args) **−0.0036**, explore4 개선 없음(grep −0.007) | 미제출 (GPU 미투입) | **폐기** — hist 확장은 명백한 해, args는 explore를 못 올림(TF-IDF가 유리한 리터럴 신호인데도) → GPU 정당화 실패. 부산물: 실제 토크나이저 실측(v3는 384에서 29.3% 잘림 — chars/4 근사는 1.64배 과소) |
| 16 | 07.05 | blend 가중 재튜닝 (리그 그리드) | [2,2,1.75](enc 지분 0.30)가 로컬 +0.0093 | 로컬 0.72654 | 미제출 | **기각** — 핸드오프 §5 편향 재확인: 리그는 enc 지분 축에서 기울기가 LB와 반대 (LB 실측: 지분 0.33→0.50 단조 상승). 리그는 성분 추가/제거 판정용으로만, enc 지분 축은 LB만 신뢰 |
| 17 | 07.05 | meta-selector (코덱스 1순위 아이디어) | scripts/meta/meta_selector.py — 27피처(마진·불일치·rank) LogReg, 확신 임계 override, 9,969행 중첩 그룹 5-fold | **전 임계·전 설정 마이너스** (θ=0.9에서도 −0.003, fixed:broken = 1:8~17). oracle 상한 0.80163은 실재 확인 | 미제출 | **폐기** — 파생 피처에는 oracle 갭을 열 정보가 없음. R4와 동일 교훈: 약한 메타로 블렌드 못 이김. (후속 후보로 '성분 신뢰 3-way 분류'가 제안됐으나 저순위 보류) |
| 18 | 07.05 | 4-way: e5-small을 encoder_2로 추가 | submit/model/encoder_2 (fp16 235MB), 인코더 블록 uniform 평균, weights [1,1,2], zip 757MB | - | **0.71280 (−0.006)** | **폐기 (uniform 블록)** — small(솔로 할인 −0.019 가족)이 base와 5:5 평균되며 최강 성분을 희석. 제출 대장 #3. 가중 블록(base≫small)은 holdout_small.npz 도착 후 리그에서만 재검토 |
| 19 | 07.05 | history 유무 버킷별 blend 가중 (동료 리포 notes/bucket-blend-2026-07-04.md + bucket-weights JSON) | 동료 JSON을 리그에서 우리 성분으로 평가: hist_empty=[0.75,0.5,2], hist_present=[0.75,1,0.75] | 리그 **0.72477 (+0.0075** vs flat 0.71726**)**. 이득 전부 history-있음 버킷의 enc 지분 하향(0.30)에서 발생 | - | ⚠️ **판단 오류 기록**: "동료 0.7242 = 이 레시피"는 정황 추론(0.7200+프록시 델타 0.0036≈0.7236)이었을 뿐 LB 실측 아님 — 동료의 +0.0036도 **자기 프록시 측정치**였다. 이걸 "LB 검증"으로 격상해 #16 금지에 예외를 뚫은 것이 #20 실패의 원인 |
| 20 | 07.05 | 버킷 정밀화: history_presence(2버킷) → history_len3(0 / 1-4 / 5+) | script.py에 history_len3 스킴 추가 + model/bucket_weights.json: hist_0=[0.75,0.5,2], hist_1_4=[0.75,1,1], hist_5_plus=[0.75,1.5,0.75] | 리그 **0.72679** (flat 대비 +0.0095, 반반 안정성도 통과) | **0.71270 (−0.0061)** | **폐기** — 리그 +0.0095 → LB −0.0061 완전 역전. **enc 지분 축 신기루 3차 확인** (핸드오프 §5, #16에 이어). 반반 안정성 체크도 이 신기루는 못 거른다(리그 내부 일관성일 뿐). 대장 #4. 교훈: enc 지분을 낮추는 어떤 스킴(전역·버킷 불문)도 리그 점수 무효 — 85% 프록시 enc가 실전 enc보다 약해서 생기는 구조적 편향 |
| 21 | 07.05 | e5-small 가중 블록 (holdout_small.npz 도착, #18 후속) | 리그에서 버킷 3-way 위에 base:small 블록 비율 0.95:0.05~0.5:0.5 + 4번째 독립 성분 w4 0.1~0.5 전수 | small solo 0.59374 (base 0.70509). **전 구간 단조 마이너스** — 최소 희석(0.95:0.05)조차 −0.0006, w4=0.1도 −0.0001 | 미제출 | **e5-small 완전 폐기** — 어떤 가중으로도 기여 없음. #18 LB −0.006과 정합. 이질 성분은 밤샘 mdeberta로 |
| 22 | 07.06 night | char-ngram LinearSVC 독립 4번째 성분 (밤샘 task1) | `scripts/components/char_svm/train_oof.py`, char_wb(2,5) max_features=300k + LinearSVC(C=0.1, balanced), 3-fold SGKF OOF + 리그 w4 add-test | OOF **0.59369**, explore4 0.41445. 리그 baseline 0.717259, best w4=1.0 → **0.718163 (+0.00090)** | 미제출 | **FAIL/폐기** — PASS 기준 +0.002 미달. 28.2% disagreement로 다양성은 있으나 성분이 약해 3-way 보정 신호가 부족. reviewer가 FAIL 판정 타당성 확인(조인 assert·그리드 정확, w4 축은 오히려 리그에 유리한 방향인데도 미달) |
| 23 | 07.06 night | AU/SIM id-prefix 분리 + AU 전용 linear 라우팅 (밤샘 task3) | `sess_au` 5,025행만으로 TF-IDF+LinearSVC, 세션 Group 3-fold OOF → 리그 holdout AU 682행의 **예측을 교체**(하드 라우팅), SIM은 3-way 유지 | AU OOF **0.68001**, 리그 AU `0.51381→0.69035`, all **+0.00935**. 독립 검증: ① tester 전 수치 소수점 재현 ② reviewer가 OOF의 holdout 혼입(fold train의 ~13%) 발견 → **완전 격리 재검증(비holdout 4,343행만 학습)에서 오히려 +0.01143** ③ AU 약세는 전 성분 공통(lin 0.544/stk 0.492/enc 0.509 vs SIM 0.67~0.72) — enc 프록시 고유 아님 | **0.73310 (+0.0142)** 🏆 | **승격 — 팀 최고 갱신**. 리그 예측(+0.0094~0.0114)보다 LB 전이가 더 컸다(+0.0142) — 신기루 우려는 기우, AU 약세는 실전 인코더에서도 실재하며 오히려 더 컸음. '결정적 키 서브모집단 라우팅' 패턴 최초 실증. test에 sess_au 실재 확인(델타>0). 다음: task4 soft α=0.9 변형(리그 하드 대비 +0.0065) LB 게이트 |
| 24 | 07.06 night | AU 라우팅 심화: soft 결합·모델 그리드 (밤샘 task4) | scripts/au2/task4_grid.py — 격리 규약(학습=비holdout AU), C×피처 그리드 + α 그리드 + sim 가중 | 최고 = **char-only C=1.0, soft α=0.9** → 리그 0.73314 (**하드 대비 +0.0065**). sim 가중 활용은 열세 | **0.7400 (+0.0069)** 🏆 | **승격 — 팀 최고 재갱신** (대장 #6). 리그 예측 +0.0065 → LB +0.0069, 오차 0.0004 — 라우팅 축에서 리그 전이 2연속 정확. 다음 후보: mdeberta 성분(Colab 진행 중), AU 라우팅 3차 정밀화는 한계효용 체감 예상 |
| 25 | 07.06 night | first-step(hist=0) 라우팅 프로브 (밤샘 task5) | scripts/firststep/ — 성분별 hist_0 분석 + 격리 프로브 | hist_0에서 전 성분 0.40 안팎 균등, specialist OOF 0.426 — 마진 없음. soft α=0.7 최대 **+0.0011** (< +0.005 문턱), hard는 −0.0137 | 미제출 | **FAIL/폐기** — AU와 달리 first-step 약세는 정보 부족이 원인(모델 문제 아님). 라우팅 패턴은 '전용 학습이 크게 이기는 서브모집단'에만 유효 |
| 26 | 07.06 | mdeberta-v3-base 이질 인코더 (Colab 학습, [reports/colab_mdeb_run_2026-07-06.md](reports/colab_mdeb_run_2026-07-06.md)) | holdout85 격리 학습(fp32, 2ep, maxlen384) → holdout_mdeb.npz 리그 판정 (스크래치 league_mdeb.py, reviewer 수치 재현·조인 검증 완료) | mdeb 단독 **0.66998** (e5-proxy 0.70509, argmax 일치율 0.864). **교체 [1,1,2·mdeb] −0.00505 FAIL** / 추가 [1,1,2,1] +0.0023 미달 / **블록 분할 [1,1,e5+mdeb] +0.00533** (반반 +0.0063/+0.0046) | 미제출 (zip 불가) | **교체 폐기, 블록 분할은 잠정 보류** — ① e5+mdeb 동시 탑재 = 573+574MB로 zip 1GB 초과, 그대로 제출 불가 ② reviewer 경고: 블록 분할은 #18/#21(e5-small)과 동일 구조의 **미검증 서브타입** + 게이트 마진 6% + 반반 체크는 #20 위양성 전력 → 리그만으로 승격 금지, LB 게이트 필수 ③ 후속: 크기가 맞는 mBERT(356MB)로 같은 구조 재시도 (#27 예정, bert-base-multilingual-cased holdout85 Colab 진행) |
| 27 | 07.06 | mBERT 이질 인코더 — 탑재 가능한 블록 분할 (#26 후속) | bert-base-multilingual-cased holdout85 격리 학습(fp32, 2ep, 384len, 유효배치 16) → holdout_mbert.npz. 리그 그리드: 블록 총가중 2 고정 비율 스캔 + mdeb+mBERT 교체 조합 | mBERT 단독 **0.67147** (mdeb보다 근소 우위, e5와 argmax 일치 0.862). uniform [1,1,e5+mbert] +0.00467 **게이트 0.0003 미달** → 비율 그리드: **e5 1.2 + mbert 0.8 = +0.00529 게이트 통과** (반반 +0.0047/+0.0060, 비AU 실효 +0.0060, 비율 곡선 매끄러움). mdeb+mBERT로 e5 교체는 전 구간 마이너스(−0.002~−0.005) — **e5-base 대체 불가 확정** | **0.7467 (+0.0067)** | **승격 — 우리 최고 갱신** (대장 #7, 팀 최고는 동료 0.7511). 리그 예측 +0.0053 → LB +0.0067, 오차 0.0014 — **블록 분할 서브타입 리그 전이 첫 검증** (단 실측이 예측보다 후하게 나온 방향, #18/#21 실패 전례와 달리 '비율 가중'이 핵심이었던 것으로 해석). 제출 구성: encoder_2(mBERT fp16 339MB) + enc_block_weights.json [1.2,0.8] + soft-AU 유지, zip 867.9MB. **서버 추론 4분14초** — mBERT는 e5-base 동급 연산(12L·768h, 임베딩만 차이)이라 인코더당 ~2분 예산. 다음: mBERT full-train(70k) 재학습(현재 60k 격리 학습분이라 공짜 상승 여지) |
| 28 | 07.07 | mBERT full-train 교체 (#27 후속) + 밤샘 리그 3종 | Colab full 모드(70k, 2ep, 동일 레시피) → encoder_2 교체 (script.py 무변경, 해시로 교체 확증, tester 본실행 PASS). 밤샘: task1 4-way 리그 재구축+그리드 / task2 동료 요소 판정 / task3 서브모집단 스윕 | 밤샘 결과 ① 4-way B4+soft-AU 리그 = **0.73877**, 블록 그리드 최고 [1.15,0.85] +0.0003뿐·α=0.9 유지 — **현 설정 최적 확인** ② 동료 요소 이식 불가: 스태커최종 −0.017, 로그바이어스 −0.0007 ③ 차기 AU 후보 없음 (23그룹, specialist 마진 전부 음수) — **라우팅 레인 소진** | **0.7480 (+0.0013)** | **승격 — 우리 최고 갱신** (대장 #8). '격리판→full-train 공짜 상승' 패턴 2번째 확인 (e5 전례 재현). 리그 판정 불가 축(홀드아웃 오염)이라 LB 단독 판정이었음. 60k판 Downloads/mbert_fp16.zip 보존. 부산물: tests/test_enc_block_weights.py 회귀 테스트(BOM 등 5케이스, 모듈 캐시 격리 수정 포함), pytest 설치로 G1 실제 pytest 실행 전환. 다음: mBERT 3ep 연장 프로브(보조 계정 npz 대기) → 리그 판정 후 3ep full-train 여부 |
| 29 | 07.07 | mBERT 3ep 연장 프로브 (#28 후속, Colab ckpt 재개로 +1ep) | 보조 계정에서 mbert_out ckpt(ep1 완료) 재개 → holdout_mbert_3ep.npz. league4 도구로 블록 그리드 (주의: load_league_data 기본 인자가 def 시점 바인딩이라 monkeypatch 무효 — mbert_holdout 명시 전달 필요, 단독 F1 assert로 교체 확인) | 3ep 단독 **0.69117** (2ep 0.67147 대비 +0.0197) — 그러나 블록 그리드 최고 x=0.8에서 **+0.00070뿐**, 반반 부호 갈림(+0.0036/−0.0022) | 미제출 | **FAIL/폐기 — 3ep full-train 재학습 불필요.** 교훈: **단독 강화 ≠ 앙상블 기여** — 블록은 mBERT의 보완 신호를 이미 추출하고 있고, 에폭을 더 돌리면 e5와 상관만 높아진다. 인코더 성분 강화 축(에폭·단독 성능)은 이것으로 소진 판정 |
| 30 | 07.07 | **라인 병합**: 동료 0.7511(ensau080) + 우리 mBERT (submit_candidates/merge080) | 가설: 우리 라인 레인 소진(#29·밤샘 3종) → 동료 표면에 우리 mBERT를 이식하면 우리 4-way 기여(+0.0067)가 재현될 것. 변경: ① 동료 script.py에 mbert 믹스 — apply_fix 후·AU 전 `(1-0.2)·P + 0.2·P_mbert` (0.2 = 우리 4-way mBERT 실효 지분), 라벨명 재정렬(그들 ACTIONS는 스펙순) ② attack_model.joblib 슬림: 죽은 키(memory_model·transition_model, 참조 0건) 제거 + SGD coef f32 + lzma4 = 271→179MB (parity max Δp 3.8e-8, argmax 불일치 0/500 — reviewer 재측정) ③ mbert_full = LB 0.7480 encoder_2와 SHA256 동일 ④ make_submit --submit-dir 게이트 확장 | serialize 계약: 우리 학습 serialize ↔ 동료 e5_compatible **70,000건 전수 diff=0** (reviewer). tester: 본실행·폴백(MBERT_MIX=0=동료 원본과 동일)·에러 케이스 PASS, 5행 중 1행 예측 개입 확인. zip 근사 988.8MB (여유 3.4%) | LB 게이트 예정 | 리그 판정 불가(동료 스태커 full-train — holdout 오염)라 **LB 단독 판정**. 기준: 동료 0.7511 대비 상승 여부. 폴백 안전(mbert 블록 제거 시 동료 원본과 대수 동일). 잔여 위험: T4 실측 시간(동료 원본 +mBERT 1회분 ~2분 추가 추정), zip 여유 36.9MB → **LB 0.7503 (기준 0.7511 대비 −0.0008) FAIL** — post-fix 믹스 서브타입 폐기. 교훈: mBERT 기여는 표면 의존적 — flat blend(+0.0067)에선 살고 스태커-최종 표면에선 죽는다(스태커가 이미 e5·sparse 정보를 비선형으로 흡수 → 약한 성분의 선형 믹스는 희석만). w 축소(0.1) 재프로브는 곡선상 기대이득 ~0이라 보류. 병합 잔여 카드: 동료 표면 α 0.8→0.9 프로브(1변수, config만), 스태커 피처에 mbert 주입(재학습 필요 — 동료 협업 사안) |
| 31 | 07.08 | 한국어 인코더 다양성 프로브 3종 (Colab 밤샘, holdout85 2ep 동일 레시피) | klue/roberta-base · koelectra-base-v3 · kykim/bert-kor-base — 각각 mBERT 슬롯 대체 후보로 리그 블록 그리드 (x 0.6~1.0, soft-AU 최종). 후보 파일 매핑은 다운로드 순서 추정(a=klue, b=koelectra, c=bertkor) — 전부 FAIL이라 확정 불요 | 단독: a **0.68147**(mBERT 0.67147 초과!) / c 0.67412 / b 0.65210. 리그 최고: a x=0.9 **+0.00154** (보고 문턱 0.002도 미달), b/c 마이너스 | 미제출 (리그 차단) | **FAIL/폐기 — 인코더 다양성 축 소진 선언** (mdeberta·mBERT 3ep·한국어 3종 전패). '단독↑ ≠ 앙상블 기여' 3번째 확인 — e5+X 블록에서 X의 단독 성능보다 e5와의 오차 상보성이 지배적이며, mBERT(다국어 BERT)가 그 자리의 사실상 최적. npz는 colab_out/holdout_cand_{a,b,c}.npz 보존 |
| 32 | 07.08 night | linear2 replacement sweep (밤샘 task2) | `scripts/linear2/` 추가. baseline_repro: `E_+seq`+LinearSVC C=0.1+bias 재현 OOF 0.663895 vs reference 0.663307 (Δ +0.000588, tolerance pass). Sweep: compact text char/word+char TF-IDF LinearSVC, exact 3-fold OOF, 9 variants | best `char_2-5_mf120k_C1`: OOF **0.62006**, B4+soft-AU **0.73826** vs baseline 0.73877 (**−0.00051**). All variants negative; union and 200k/300k worse | 미제출 | **FAIL/폐기 — text-only linear replacement lane exhausted.** 현재 E_+seq linear가 blend 안에서 더 강하고 보완적이다. 단독 OOF가 그럴듯해도 앙상블 기여는 없었음. 리포트: `context/night/2026-07-08/report_linear2.md` |
| 33 | 07.08 | 동료 표면 soft-AU α 0.8→0.9 프로브 (#30 잔여 카드, submit_candidates/alpha09) | merge080에서 mbert_full 제거 + attack_config.json 2값만 변경(mbert 키 삭제=믹스 비활성, au_alpha 0.9) — **동료 0.7511 원본 대비 순수 1변수**. script.py·src·모델은 merge080 검증분과 동일 파일 | 근거: 우리 표면 α그리드 정점 0.9 (exp #24 LB 전이 오차 0.0004) + 밤샘 07-07 task1 리그 α그리드도 0.9 유지. tester 6/6 PASS — sess_au 3행 주입 스텁에서 `au_route alpha=0.9 rows=3` 실측 | **0.7501 (−0.0010)** | **FAIL/폐기 — α=0.9는 우리 표면 한정** (동료 AU 모델은 자체 serialize 재학습 + 베이스가 강해 α=0.8이 최적). 병합 라인 값싼 카드 소진 — 잔여는 스태커 피처 mbert 주입(동료 협업, 재학습 필요)뿐 |
| 34 | 07.08 | **serialize 확장 재심** — e5 `max_hist 6→12` (D-010, exp #15 폐기 인코더 재검증) | 가설: #15의 hist12 폐기(프록시 −0.031)는 BoW 프록시 + args-부풀림 변형 판정이라 인코더 비전이. 변경점: encoder_v2 레시피(e5-base 6ep/b16/lr2e-5/ls0.1/fp16/balanced) 그대로 + `max_hist`만 env화, **85% split hist6 대조군 + hist12 후보** 동시 학습(colab/encoder_e5_holdout85_maxhist.py) → 리그에서 e5 슬롯 스왑, `hist12 − hist6대조`로 serialize 효과 격리 | **실측 선판정(로컬 e5 토크나이저 12k)**: 세션 51%가 6턴 초과(12턴=30.7%), hist12 잘림 384에서 **8.5%뿐**(#15의 83% 재현 실패), >6턴 세션 hist12 평균 +4.5턴 실살림, query 트림 무효. **리그(85% split): hist6대조 e5solo 0.70066/4way+AU 0.73451 → hist12 e5solo 0.73617/4way+AU 0.75601** | **격리 델타 +0.02150 (게이트 4배 초과), 반반 +0.018/+0.025 안정** | **PASS → 승격 진행.** serialize 확장이 처음으로 앙상블 전이(e5 solo +0.0355 → blend +0.0215) — #15 폐기는 BoW 프록시 판정이라 비전이였음을 입증(인코더는 attention으로 최근 히스토리 활용, 탐색계열 혼동 해소). **다음: e5 full-train(70k 6ep) hist12 재학습 → 배포.** ⚠️ 계약 주의: 추론 serialize를 hist12로 바꾸면 hist6로 학습된 mBERT가 train↔infer 불일치 → **per-encoder serialize(e5=12, mBERT=6)로 분리** 또는 mBERT도 hist12 재학습(Bet B). 후속 프로브 승인됨(D-010): maxlen512(Bet A)·mBERT hist12(Bet B). 실측 스크립트: scratchpad/measure_serialize.py |

## ❌ 폐기 확정 — 재시도 금지 (검증 후 버린 것, 핸드오프 §6)

| 레버 | LB/결과 | 왜 |
|---|---|---|
| seed soup (soup2/3: s42+s7 가중치 평균) | 0.697 | seed별 head 초기화가 다른 basin → 파괴적 간섭 |
| R4 계층 분류 (explore gate+specialist) | 리그 −0.007~−0.016 | linear 단독에서만 +0.0246, 3-way 베이스에서 역전 — blend가 이미 explore를 더 잘 풂 (#14) |
| serialize 확장 (hist12 / args 추가) | 프록시 −0.031 / −0.004 | 길이 희석 실해 + args가 explore F1을 못 올림. hist12는 384 토큰에서 83% 잘림 문제도 (#15) |
| 세션 형제 행 라벨 복원 | LB 델타 0 | test가 세션당 1스텝 샘플링 — 복원 대상 없음 (#12, 보험 코드는 잔류) |
| e5-small 성분 추가 (모든 형태) | LB −0.006 / 리그 전 가중 마이너스 | uniform·가중 블록·4번째 독립 성분 전부 마이너스 — solo 0.594가 너무 약함 (#18·#21). 완전 폐기 |
| **enc 지분 낮추기 — 전역·버킷 불문 전면 금지** | LB 실측 −0.0061 (#20) | 리그 기울기가 enc 지분 축에서 LB와 반대 — **세 번 확인** (핸드오프 §5, #16, #20 LB 실측). 버킷별도 예외 아님 (#19의 '예외' 판단은 정황 추론을 LB 검증으로 오독한 오류). 리그는 오직 성분 추가/제거 판정용, enc 지분이 변하는 모든 스킴은 LB 게이트 필수 |
| calib_v1 (enc T=1.34 + class bias) | 0.7169 | holdout +0.005가 LB −0.002로 비전이 — train 분포 피팅 bias는 분포 이동에 취약 |
| flat 피처 추가 F~W | 이득 없음 | |
| stacker 변형 4종 | 이득 없음 | |
| cascade base | +0.0019 | 노이즈 수준 |
| max_len 512 추론 | ±0.000 | |
| 스키마에 없는 환각 피처 | 무효 | 입력에 존재하지 않는 필드 |
| 옛 회차(2025) 전략 | 무관 | Macro-F1 14-class 지표 불일치 |

## 로컬 CV → LB 할인율 (핸드오프 §5 — 반드시 할인해서 읽기)

| 성분 가족 | 할인 |
|---|---:|
| linear | −0.002 |
| encoder (base) | −0.015 |
| encoder (small/e8) | −0.019 |
| stacker | −0.033 |

- 로컬 그리드는 최적 enc 지분을 LB보다 **낮게** 잡는다 (로컬 최적 0.33 vs LB 최적 0.50) → 그리드는 방향·랭킹 선택기로만, 오른쪽 보정해서 읽기.
- 검증 프로토콜: **StratifiedGroupKFold**, 그룹키 = `-step_\d+$` 제거 (9,429 세션), 중요 판정은 3-fold.
- **per-class F1 항상 확인**: 약점 = 탐색 계열(read_file·grep_search·list_directory·glob_pattern). respond_only·write_file은 이미 1.0 — 건드리지 말 것.

## Agent Handoff Log

Sub-agent 작업 결과는 필요할 때 아래 형식으로 남긴다.

```text
Role:
Goal:
Files changed:
Validation:
Experiment log entry:
Open questions:
Next recommended owner:
```

<!-- (task2 브랜치의 중복 기록 섹션은 본 테이블 #8·#9로 통합됨 — 상세는 context/night/2026-07-05/task2_report.md) -->
