"""Exact, faster consumer for the deployed AAR stacker.

The deployment implementation in :mod:`aar_infer` remains the contract
and is never modified.  This module preserves its float32 component alignment,
stacking order, transition arithmetic, and final classifier while avoiding
unused views and caching repeated char_wb word n-gram lookups.

scripts/aar_speed/fast_aar.py 벤더본 — 유일한 차이는 zip 루트 배포용 import 한 줄
(`from submit import aar_infer` → `import aar_infer`). 등가성 게이트: 5,000행
argmax 5000/5000·확률오차 0.0 (night 07-13 task1, tester 독립 재실행 PASS).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time
from typing import Any, Dict, Iterable, Mapping, Sequence

import joblib
import numpy as np
from scipy import sparse

import aar_infer as AAR


Record = Dict[str, Any]


def _align(raw: np.ndarray, model: object) -> np.ndarray:
    """Match ``aar_infer.predict_proba_aligned`` byte for byte."""
    raw32 = np.asarray(raw, dtype=np.float32)
    classes = AAR._model_classes(model)  # contract helper from the vendor file
    action_to_idx = {action: idx for idx, action in enumerate(AAR.ACTIONS)}
    aligned = np.zeros((raw32.shape[0], len(AAR.ACTIONS)), dtype=np.float32)
    for src_idx, label in enumerate(classes or AAR.ACTIONS):
        dst_idx = action_to_idx.get(str(label))
        if dst_idx is not None and src_idx < raw32.shape[1]:
            aligned[:, dst_idx] = raw32[:, src_idx]
    return aligned


def _cached_char_wb_transform(vectorizer: object, documents: Sequence[str]):
    """Run fitted ``TfidfVectorizer(char_wb)`` with exact cached word counts.

    sklearn's analyzer recreates and looks up the same within-word character
    n-grams for every occurrence.  Counts are integers, CSR indices are sorted
    exactly as in ``CountVectorizer._count_vocab``, and the fitted transformer
    still performs the original sublinear-TF/IDF/normalization operations.
    """
    vocabulary = vectorizer.vocabulary_
    preprocess = vectorizer.build_preprocessor()
    # W stores fitted-vocabulary n-gram counts once per unique word. D stores
    # word counts per document. D @ W is then exactly the integer document-term
    # count matrix that CountVectorizer would build with repeated Python lookups.
    word_ids: dict[str, int] = {}
    word_indices: list[int] = []
    word_values: list[int] = []
    word_indptr = [0]
    document_indices: list[int] = []
    document_values: list[int] = []
    document_indptr = [0]

    for raw_document in documents:
        document = preprocess(vectorizer.decode(raw_document))
        document_counter: dict[int, int] = {}
        words = vectorizer._white_spaces.sub(" ", document).split()
        for word in words:
            word_idx = word_ids.get(word)
            if word_idx is None:
                word_idx = len(word_ids)
                word_ids[word] = word_idx
                word_counter: dict[int, int] = {}
                for feature in vectorizer._char_wb_ngrams(word):
                    feature_idx = vocabulary.get(feature)
                    if feature_idx is not None:
                        word_counter[feature_idx] = word_counter.get(feature_idx, 0) + 1
                word_indices.extend(word_counter.keys())
                word_values.extend(word_counter.values())
                word_indptr.append(len(word_indices))
            document_counter[word_idx] = document_counter.get(word_idx, 0) + 1

        document_indices.extend(document_counter.keys())
        document_values.extend(document_counter.values())
        document_indptr.append(len(document_indices))

    word_index_dtype = (
        np.int64 if len(word_indices) > np.iinfo(np.int32).max else np.int32
    )
    document_index_dtype = (
        np.int64 if len(document_indices) > np.iinfo(np.int32).max else np.int32
    )
    word_matrix = sparse.csr_matrix(
        (
            np.asarray(word_values, dtype=np.intc),
            np.asarray(word_indices, dtype=word_index_dtype),
            np.asarray(word_indptr, dtype=word_index_dtype),
        ),
        shape=(len(word_ids), len(vocabulary)),
        dtype=vectorizer.dtype,
    )
    document_matrix = sparse.csr_matrix(
        (
            np.asarray(document_values, dtype=np.intc),
            np.asarray(document_indices, dtype=document_index_dtype),
            np.asarray(document_indptr, dtype=document_index_dtype),
        ),
        shape=(len(documents), len(word_ids)),
        dtype=vectorizer.dtype,
    )
    csr = (document_matrix @ word_matrix).tocsr()
    csr.sort_indices()
    if getattr(vectorizer, "binary", False):
        csr.data.fill(1)
    return vectorizer._tfidf.transform(csr, copy=False)


def _feature_union_transform(union: object, documents: Sequence[str]):
    transformed = []
    weights = getattr(union, "transformer_weights", None) or {}
    for name, transformer in union.transformer_list:
        if transformer == "drop":
            continue
        if transformer == "passthrough":
            matrix = documents
        elif getattr(transformer, "analyzer", None) == "char_wb" and hasattr(
            transformer, "_tfidf"
        ):
            matrix = _cached_char_wb_transform(transformer, documents)
        else:
            matrix = transformer.transform(documents)
        if name in weights:
            matrix = matrix * weights[name]
        transformed.append(matrix)

    if not transformed:
        return np.zeros((len(documents), 0), dtype=np.float64)
    if any(sparse.issparse(matrix) for matrix in transformed):
        return sparse.hstack(transformed).tocsr()
    return np.hstack(transformed)


def _component_proba(model: object, documents: Sequence[str]) -> np.ndarray:
    """Use the exact model, replacing only supported char_wb transformation."""
    steps = getattr(model, "named_steps", None)
    if not isinstance(steps, Mapping) or list(steps) != ["features", "clf"]:
        return AAR.predict_proba_aligned(model, documents)
    features = steps["features"]
    classifier = steps["clf"]
    if hasattr(features, "transformer_list"):
        matrix = _feature_union_transform(features, documents)
    elif getattr(features, "analyzer", None) == "char_wb" and hasattr(features, "_tfidf"):
        matrix = _cached_char_wb_transform(features, documents)
    else:
        matrix = features.transform(documents)
    return _align(classifier.predict_proba(matrix), classifier)


def _selected_views(
    records: Sequence[Record],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
    required: Iterable[str],
) -> dict[str, list[Any]]:
    """Build only views named by deployed components, using vendor functions."""
    factories = {
        "full": lambda: list(texts),
        "prompt": lambda: list(prompt_texts),
        "prompt_context": lambda: [AAR.aar_prompt_context_text(row) for row in records],
        "history": lambda: [AAR.aar_history_text(row) for row in records],
        "action": lambda: [AAR.aar_action_text(row) for row in records],
        "meta_text": lambda: [AAR.aar_meta_text(row) for row in records],
        "meta_dict": lambda: [AAR.aar_metadata_features(row) for row in records],
        "rule_dict": lambda: [AAR.aar_rule_features(row) for row in records],
    }
    views: dict[str, list[Any]] = {}
    for name in dict.fromkeys(required):
        if name not in factories:
            raise ValueError(f"Unsupported AAR view: {name}")
        views[name] = factories[name]()
    return views


def _transition_proba(spec: Mapping[str, Any], records: Sequence[Record]) -> np.ndarray:
    """Memoize repeated transition-key combinations with original float order."""
    global_vec = np.asarray(spec["global"], dtype=np.float32)
    weights = spec.get("weights", {})
    groups = spec.get("groups", {})
    group_names = tuple(weights)
    result = np.zeros((len(records), len(AAR.ACTIONS)), dtype=np.float32)
    cache: dict[tuple[str | None, ...], np.ndarray] = {}

    for row_idx, record in enumerate(records):
        keys = AAR.aar_transition_keys(record)
        signature = tuple(keys.get(group) for group in group_names)
        probability = cache.get(signature)
        if probability is None:
            total = global_vec * float(spec.get("global_weight", 0.3))
            weight_sum = float(spec.get("global_weight", 0.3))
            for group, weight in weights.items():
                values = groups.get(group, {}).get(keys.get(group))
                if values is None:
                    continue
                total += np.asarray(values, dtype=np.float32) * float(weight)
                weight_sum += float(weight)
            probability = total / max(weight_sum, 1e-6)
            cache[signature] = probability
        result[row_idx] = probability
    return result


def reference_predict_proba(
    records: Sequence[Record],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> np.ndarray:
    """Exact probability form of the unmodified vendor ``predict_aar`` path."""
    views = AAR.aar_views(records, texts, prompt_texts)
    component_probas: dict[str, np.ndarray] = {}
    for component in config.get("components", []):
        name = str(component.get("name"))
        if str(component.get("kind")) == "transition":
            component_probas[name] = AAR.aar_transition_predict_proba(
                artifact["transition"], records
            )
        else:
            model = artifact["components"][name]
            component_probas[name] = AAR.predict_proba_aligned(
                model, views[str(component.get("view"))]
            )
    if config.get("use_stacker"):
        names = [str(name) for name in config.get("stacker_components", [])]
        matrix = np.hstack([component_probas[name] for name in names]).astype(np.float32)
        probas = AAR.predict_proba_aligned(artifact["stacker"], matrix)
    else:
        probas = AAR.weighted_average(
            (component_probas[str(c["name"])], float(c.get("weight", 0.0)))
            for c in config.get("components", [])
        )
    if config.get("use_bias"):
        probas = AAR.aar_apply_bias(
            probas, config.get("class_bias", [0.0] * len(AAR.ACTIONS))
        )
    return np.asarray(probas)


def fast_predict_proba(
    records: Sequence[Record],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
) -> np.ndarray:
    """Return probabilities in the canonical ``AAR.ACTIONS`` order."""
    if not (len(records) == len(texts) == len(prompt_texts)):
        raise ValueError("records, texts, and prompt_texts must have equal length")
    required_views = [
        str(component.get("view"))
        for component in config.get("components", [])
        if str(component.get("kind")) != "transition"
    ]
    views = _selected_views(records, texts, prompt_texts, required_views)
    component_probas: dict[str, np.ndarray] = {}
    for component in config.get("components", []):
        name = str(component.get("name"))
        if str(component.get("kind")) == "transition":
            transition = artifact.get("transition")
            if not isinstance(transition, Mapping):
                raise ValueError("AAR transition component is missing.")
            component_probas[name] = _transition_proba(transition, records)
        else:
            model = artifact.get("components", {}).get(name)
            if model is None:
                raise ValueError(f"AAR component is missing: {name}")
            component_probas[name] = _component_proba(
                model, views[str(component.get("view"))]
            )

    if config.get("use_stacker"):
        stacker = artifact.get("stacker")
        if stacker is None:
            raise ValueError("AAR stacker is missing.")
        names = [str(name) for name in config.get("stacker_components", [])]
        matrix = np.empty(
            (len(records), len(names) * len(AAR.ACTIONS)), dtype=np.float32
        )
        for idx, name in enumerate(names):
            start = idx * len(AAR.ACTIONS)
            matrix[:, start : start + len(AAR.ACTIONS)] = component_probas[name]
        probas = AAR.predict_proba_aligned(stacker, matrix)
    else:
        probas = AAR.weighted_average(
            (component_probas[str(c["name"])], float(c.get("weight", 0.0)))
            for c in config.get("components", [])
        )
    if config.get("use_bias"):
        probas = AAR.aar_apply_bias(
            probas, config.get("class_bias", [0.0] * len(AAR.ACTIONS))
        )
    return np.asarray(probas)


def predict_aar(
    records: Sequence[Record],
    texts: Sequence[str],
    prompt_texts: Sequence[str],
    config: Mapping[str, Any],
    artifact: Mapping[str, Any] | None = None,
    model_dir: str | Path = "model",
) -> list[str]:
    """Drop-in label result; callers may inject the already-loaded artifact."""
    if artifact is None:
        artifact = joblib.load(
            Path(model_dir) / str(config.get("model_file", "aar_models.joblib"))
        )
    return AAR.labels_from_proba(
        fast_predict_proba(records, texts, prompt_texts, config, artifact)
    )


def sample_evenly(path: Path, rows: int) -> list[Record]:
    """Select deterministic positions spanning the complete JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        total = sum(1 for line in handle if line.strip())
    if rows < 1 or rows > total:
        raise ValueError(f"rows must be in [1, {total}], got {rows}")
    positions = set(np.linspace(0, total - 1, rows, dtype=np.int64).tolist())
    selected = []
    nonempty_idx = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if nonempty_idx in positions:
                selected.append(json.loads(line))
            nonempty_idx += 1
    if len(selected) != rows:
        raise RuntimeError(f"sample selection returned {len(selected)} rows, expected {rows}")
    return selected


def _session_key(record: Mapping[str, Any]) -> str:
    return str(record.get("id", "")).rsplit("-step_", 1)[0]


def benchmark(
    records: Sequence[Record],
    config: Mapping[str, Any],
    artifact: Mapping[str, Any],
    repeats: int = 3,
    warmup_rows: int = 32,
) -> dict[str, Any]:
    texts = [AAR.record_to_text(record) for record in records]
    prompt_texts = [AAR.record_to_prompt_text(record) for record in records]
    warm = min(len(records), max(1, warmup_rows))
    reference_predict_proba(
        records[:warm], texts[:warm], prompt_texts[:warm], config, artifact
    )
    fast_predict_proba(records[:warm], texts[:warm], prompt_texts[:warm], config, artifact)

    reference_runs = []
    fast_runs = []
    reference = fast = None
    for _ in range(repeats):
        started = time.perf_counter()
        reference = reference_predict_proba(records, texts, prompt_texts, config, artifact)
        reference_runs.append(time.perf_counter() - started)
        started = time.perf_counter()
        fast = fast_predict_proba(records, texts, prompt_texts, config, artifact)
        fast_runs.append(time.perf_counter() - started)

    assert reference is not None and fast is not None
    reference_labels = np.asarray(AAR.labels_from_proba(reference), dtype=object)
    fast_labels = np.asarray(AAR.labels_from_proba(fast), dtype=object)
    reference_median = statistics.median(reference_runs)
    fast_median = statistics.median(fast_runs)
    return {
        "rows": len(records),
        "unique_sessions": len({_session_key(record) for record in records}),
        "repeats": repeats,
        "warmup_rows": warm,
        "reference_seconds_runs": reference_runs,
        "fast_seconds_runs": fast_runs,
        "reference_seconds_median": reference_median,
        "fast_seconds_median": fast_median,
        "reference_ms_per_row": 1000.0 * reference_median / len(records),
        "fast_ms_per_row": 1000.0 * fast_median / len(records),
        "speedup": reference_median / fast_median,
        "max_abs_probability_error": float(np.max(np.abs(reference - fast))),
        "argmax_matches": int(np.count_nonzero(reference_labels == fast_labels)),
        "argmax_total": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = json.loads((args.model_dir / "aar_config.json").read_text(encoding="utf-8"))
    artifact = joblib.load(args.model_dir / str(config.get("model_file", "aar_models.joblib")))
    result = benchmark(
        sample_evenly(args.data, args.rows), config, artifact, repeats=args.repeats
    )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
