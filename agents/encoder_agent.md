# Encoder Agent

## Mission

Tier 2 lightweight pretrained encoder experiments를 설계하고, 제출 제약을 만족하는 경우에만 구현한다.

## Read First

- `agents/common.md`
- `PLAN.md`
- Current Tier 1 results in `context/experiments.md`

## Decision Gate

Encoder work starts only after:

- Tier 0 submission path works.
- GroupKFold CV is established.
- Tier 1 baseline has a recorded score.

## Candidate Direction

- Lightweight multilingual encoders only.
- Sequence input should serialize prompt, recent history, and compact meta.
- Any pretrained weights used at inference must be bundled under `submit/model/`.
- In submitted code, use local paths only and offline/local-files-only mode.

## Deliverables

- Feasibility note: size, expected inference time, package impact.
- Training/inference code only if feasibility passes.
- Experiment entry with per-class impact vs Tier 1.

## Do Not

- Do not add a model that makes `submit.zip` exceed 1GB.
- Do not rely on runtime model download.
