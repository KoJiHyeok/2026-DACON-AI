# -*- coding: utf-8 -*-
"""AU(sess_au) 전용 linear 라우팅 (exp #23) — 학습·추론 공용 단일 소스.

serialize()는 밤샘 task3 scripts/au/probe_au_linear.py와 동일 로직 (타입힌트만 제거,
AST 수준 동일 — 독립 리뷰로 확인. train·추론 계약이므로 변경 금지).
- 학습: scripts/au/train_full_au.py 가 이 모듈을 import → model/au_linear/model.pkl
- 추론: script.py 가 pkl 존재 + ENS_AU_ROUTE!=0 이면 sess_au 행의 최종 예측을 교체
  (test에 sess_au 행이 없으면 no-op — 하방 위험 0)
"""
import json
import re


def is_au(sample_id):
    return str(sample_id).startswith("sess_au")


def compact_json(value, limit=240):
    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    text = re.sub(r"\s+", " ", text)
    return text[:limit]


def serialize(sample):
    meta = sample.get("session_meta") or {}
    workspace = meta.get("workspace") or {}
    parts = [
        "id_prefix=sess_au" if is_au(str(sample.get("id", ""))) else "id_prefix=sess_sim",
        f"user_tier={meta.get('user_tier', '')}",
        f"language_pref={meta.get('language_pref', '')}",
        f"turn_index={meta.get('turn_index', '')}",
        f"elapsed_session_sec={meta.get('elapsed_session_sec', '')}",
        f"budget_tokens_remaining={meta.get('budget_tokens_remaining', '')}",
        f"git_dirty={workspace.get('git_dirty', '')}",
        f"last_ci_status={workspace.get('last_ci_status', '')}",
        f"loc={workspace.get('loc', '')}",
        "language_mix=" + compact_json(workspace.get("language_mix") or {}),
        "open_files=" + " ".join(str(x) for x in (workspace.get("open_files") or [])),
        "current_prompt=" + str(sample.get("current_prompt") or ""),
    ]
    for item in sample.get("history") or []:
        role = str(item.get("role", ""))
        if role == "assistant_action":
            parts.append(
                "history_action="
                + str(item.get("name", ""))
                + " args="
                + compact_json(item.get("args") or {}, 180)
                + " result="
                + str(item.get("result_summary", ""))
            )
        else:
            parts.append(f"history_{role}=" + str(item.get("content", "")))
    return "\n".join(parts)


def predict_proba(artifact, samples):
    """artifact = {"union": vectorizer, "clf": LinearSVC} → (probs, classes).

    probs = softmax(decision_function) — 밤샘 task4 그리드(scripts/au2/task4_grid.py)의
    확률 변환과 동일 (soft 결합 계약)."""
    import numpy as np
    texts = [serialize(s) for s in samples]
    x = artifact["union"].transform(texts)
    z = np.asarray(artifact["clf"].decision_function(x), dtype=np.float64)
    if z.ndim == 1:
        z = np.vstack([-z, z]).T
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    probs = e / e.sum(axis=1, keepdims=True)
    classes = [str(c) for c in artifact["clf"].classes_]
    return probs, classes


def predict(artifact, samples):
    """예측 라벨 리스트 (하드 라우팅용 — soft는 predict_proba 사용)."""
    probs, classes = predict_proba(artifact, samples)
    return [classes[i] for i in probs.argmax(axis=1)]
