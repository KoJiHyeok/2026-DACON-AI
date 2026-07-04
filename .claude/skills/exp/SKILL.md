---
name: exp
description: 실험 기록 — 가설→변경점→로컬 CV→LB 형식으로 context/experiments.md에 남기고, 폐기 실험은 재시도 금지 테이블로 보낸다. Trigger: /exp
---

# 실험 기록

"기록이 없으면 일어나지 않은 것." 실험 하나가 끝날 때마다 즉시 남긴다.

## 절차

1. `context/experiments.md` 실험 테이블에 행 추가: # / 날짜 / 가설·행동 / 변경점 / 로컬 CV / LB / 결론.
2. **판정 규칙**:
   - 점수 판정은 **LB 실측 또는 세션 프리픽스 group-split CV만**. accuracy·누수 split·단일 holdout 금지.
   - CV는 할인율을 적용해 읽는다: linear −0.002, encoder base −0.015, encoder small −0.019, stacker −0.033.
   - per-class F1을 항상 확인 — 약점은 탐색 4클래스(read/grep/list/glob), respond_only·write_file은 이미 1.0.
3. **폐기 확정**이면 "재시도 금지" 테이블로 이동 — 반드시 '왜'를 적는다 (같은 실험을 두 번 하지 않기 위한 장치).
4. 방향 전환·규약 변경 수준의 결정이면 `context/decisions.md`에 D-00x ADR을 추가하고 실험 행에서 번호로 인용한다.
