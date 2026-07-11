# Model Routing (Codex)

현재 세션의 기반 모델은 실행 중 hot-swap할 수 없다. 대신 Codex 주 에이전트는 작업을 분류한 뒤
`codex exec -m <model>`로 적합한 모델의 별도 작업을 실행할 수 있다. 이 규칙은 **Codex CLI 작업에만**
적용하며, `CLAUDE.md`의 Claude Opus 오케스트레이션 및 Sonnet 5 서브에이전트 고정 규칙을 변경하지 않는다.
이 저장소의 Codex 작업에 한해 아래 라우팅은 사용자 사전 승인 없이 허용한다.

- **Luna (`gpt-5.6-luna`, medium)**: 읽기 전용 검색, 로그/표 요약, 문서 정리, 범위가 명확한 단순 테스트.
- **Terra (`gpt-5.6-terra`, medium/high)**: 기본값. 일반 구현, 디버깅, 평가 스크립트, OOF 정렬, 리팩터링.
- **Sol (`gpt-5.6-sol`, high/xhigh)**: 예상 LB 영향 `>=0.005`, train/infer 계약, CV/OOF/split,
  `submit/`, 복합 설계, 최종 감사, 또는 Terra가 같은 원인으로 2회 실패한 작업.

라우팅 규칙:

1. 단순 조회는 Luna, 일반 작업은 Terra, 고위험 의사결정은 Sol을 선택한다.
2. Sol의 기본 reasoning은 low이므로 고난도 작업은 반드시 high 또는 xhigh로 지정한다.
3. `codex exec`는 스스로 격리를 보장하지 않는다. 읽기 전용 작업은 반드시 `-s read-only`로 실행한다.
   쓰기 작업은 실행 전 `C:\dev\codex-context-v2`와 `codex/context-v2` branch를 확인하고 `-C`로
   그 worktree를 지정한다. main 또는 Claude 작업트리에서 쓰기 가능한 routed 실행을 금지한다.
4. delegated 실행에는 `ROUTED_TASK=1`을 전달하고, routed agent는 다시 `codex exec`를 호출하지 않는다.
5. `context/coordination.md`의 소유권을 전부 따른다. 특히 `submit/**`, 활성 Claude 경로, canonical
   `context/` 기록, 공식 LB 제출, main merge는 routed Codex가 수정하거나 수행하지 않는다.
6. 작성자와 다른 reviewer/tester 검증 전에는 main 승격 또는 제출하지 않는다.
7. 결과에는 사용 모델, reasoning, 검증 결과를 handoff에 기록한다. 쓰기 작업은 branch/commit을,
   읽기 전용 작업은 `branch/commit: N/A (read-only)`를 기록한다.

참조 범위: `CLAUDE.md` 중 **절대 규칙(제출 제약)** 과 **기록 시스템(context/) 게이트** 절만 Codex에 적용된다.
모델 역할 분담·서브에이전트(reviewer/tester) 규칙 등 Claude 전용 절은 Codex에 적용하지 않는다 — Codex의 모델 선택은 이 문서의 라우팅이 우선한다.
