# task2 — explore 계층 분류(R4) 프로토타입 + 첫스텝 prior(R3) 실측

## 컨텍스트

DACON 236694 (14클래스, Macro-F1). 오늘 포렌식 1라운드(`C:\dev\2026-AI-DACON\context\reports\forensics_r1.md` — 반드시 먼저 정독)가
"결정적 state→action 규칙" 가설을 **기각**했다 (purity≥0.99 구조 규칙 coverage 0.03%, 최소표본 걸면 0%).
대신 두 리드가 남았다:
- **R4 (이 작업의 본체)**: explore 4클래스(read_file/grep_search/list_directory/glob_pattern)는 Macro-F1 약점인데,
  "explore로 판명된" 조건부에서는 (last2_action, last_action) 쌍이 80~90% purity를 낸다 (무조건부론 0.21~0.52).
  → **계층 분류(1단계 대분류 → 2단계 조건부 explore 분류기)가 플랫 분류보다 나은지 로컬 CV로 검증**한다.
- **R3 (부차)**: 첫 스텝(history_len==0, 12.9%)은 라벨 분포가 구조적으로 다르다
  (list_directory 20.2% vs 4.1%, apply_patch 0.07% vs 7.9% 등) → first-step 전용 class-wise bias의 이득 상한을 실측한다.

## 목표 / 완료 조건 (DoD)

1. `scripts/hierarchy/proto_hier.py` — 계층 분류 프로토타입: 같은 피처·같은 fold에서 (A) 플랫 14-way baseline vs (B) 계층(1단계 대분류 + 2단계 explore 세부) 비교
2. 판정은 **세션 프리픽스 StratifiedGroupKFold(5-fold)** Macro-F1 + explore 4클래스 per-class F1 — 랜덤 split 절대 금지
3. `scripts/hierarchy/first_step_bias.py` — 첫 스텝 부분집합에서 baseline 모델의 per-class F1과, first-step 전용 prior 보정(간단한 class-wise logit/확률 bias 그리드)을 fold-valid에서 실측 — 이득 상한과 함께 "calib_v1 실패 전례(holdout 이득이 LB로 비전이)" 경고를 리포트에 명시
4. `context/night/2026-07-05/task2_report.md` — (A) vs (B) 표, explore per-class F1 변화, 첫스텝 bias 상한, 결론(다음 낮에 LB 프로브할 가치 있는가), 한계
5. 모든 산출물 git commit
6. 마지막으로 `context/night/2026-07-05/task2.DONE` 생성 (5줄 요약)

## 재료 (절대 경로 — 이 워크트리에는 gitignore/미커밋 파일이 없을 수 있다)

- 포렌식 리포트·산출물 (읽기 전용): `C:\dev\2026-AI-DACON\context\reports\forensics_r1.md`, `C:\dev\2026-AI-DACON\scripts\analysis\_out\*.csv`, 분석 캐시 `C:\dev\2026-AI-DACON\data\_forensics_cache.pkl` (플랫 데이터프레임 — 로딩 시간 절약에 사용 가능, 스키마는 `scripts\analysis\common.py` 참조)
- 데이터 (읽기 전용): `C:\dev\2026-AI-DACON\data\train.jsonl`, `train_labels.csv`
- 파이썬: `C:\dev\2026-AI-DACON\.venv\Scripts\python.exe` (sklearn 1.8.0 — 시스템 파이썬 금지)
- 피처 참고 (읽기 전용, 필요 코드는 워크트리로 복사): 팀 리포 `C:\dev\dacon-agent-action-api-boost\linear_pipeline\` (E_+seq 피처 — history 시퀀스가 +0.127 최대 레버), `ensemble\features.py`
- 그룹키: `id.rsplit("-step_", 1)[0]` (`src/features.py`의 `session_id`)

## 금지

- 메인 리포·팀 리포 수정 금지, `git push` 금지, 산출물은 이 워크트리 안에만
- 랜덤 split·단일 holdout 판정 금지 — 세션 group-split 5-fold만
- GPU·transformers 사용 금지 (CPU 선형 모델로 프로토타입 — LinearSVC/SGD/LogReg 계열)
- 폐기 목록(`C:\dev\2026-AI-DACON\context\experiments.md` 재시도 금지 테이블) 위반 금지 — 특히 글로벌 temperature/calibration 재시도 금지 (first_step_bias는 state-conditioned라 허용, 단 경고 명시)

## 진행 프로토콜 (재개 대비 — 필수)

1. 시작하자마자 `context/night/2026-07-05/PROGRESS-task2.md` 확인 — 있으면 '다음 재개 지점'부터
2. 의미 단위(피처 파이프라인 완성 / 플랫 baseline CV 완주 / 계층 CV 완주 / first-step 실측 / 리포트)마다 PROGRESS 갱신 + git commit
3. 전부 끝나면 task2.DONE + 최종 커밋

## 작업 내용

1. forensics_r1.md의 (e)절과 (f)절 R3/R4, (g)절을 정독하고, `_out\last2_conditional_lift.csv`·`exploration_signal_report.csv`로 조건부 신호의 형태를 파악하라.
2. 피처 파이프라인: 팀 linear의 E_+seq 피처를 재사용(복사)하거나 동등한 경량 버전(프롬프트 TF-IDF/해싱 + last1/last2/last3 action + history 길이 + session_meta 이산화)을 구성. 플랫과 계층이 **같은 피처, 같은 fold**를 쓰는 것이 비교의 전제다.
3. (A) 플랫 14-way baseline: LinearSVC(C=0.1, class_weight=balanced) 계열로 5-fold CV Macro-F1 + per-class F1.
4. (B) 계층: 1단계 대분류 — 제안 그룹: explore={read_file,grep_search,list_directory,glob_pattern}, mutate={edit_file,write_file,apply_patch}, validate={run_bash,run_tests,lint_or_typecheck}, coordinate={ask_user,plan_task,web_search,respond_only} (필요시 근거와 함께 조정 가능) / 2단계 — 각 대분류 내부 분류기(특히 explore는 last1/last2 조건부 신호 강조). fold마다 1→2단계 파이프라인으로 예측해 14클래스 Macro-F1 산출. 1단계 오류 전파를 완화하는 soft 변형(1단계 확률 × 2단계 확률)도 시간이 되면 비교.
5. (A) vs (B) 비교표 + explore 4클래스 F1 변화 분석. 개선이 없으면 없다고 정직하게 — 음성 결과도 가치 있다 (다음 라운드가 이 길을 다시 안 파게).
6. first_step_bias.py: baseline 모델의 fold-valid 확률에서 history_len==0 부분집합만 골라 class-wise bias 그리드(각 클래스 확률에 곱/덧셈 보정) → Macro-F1 델타 상한 실측.
7. 리포트 + DONE.
