from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler

from .common import (
    ACTIONS,
    OOFBundle,
    align_bundle,
    hash_files,
    load_hist12_oof,
    load_legacy_oof,
    session_group,
)
from .features import (
    FEATURE_SOURCE,
    FEATURE_SOURCE_COMMIT,
    FEATURE_SOURCE_SHA256,
    build_teammate_matrix,
    feature_dicts,
    numeric_feature_names,
    teammate_numeric_features,
)


METHODOLOGY_LIMITATION = (
    "diagnostic_meta_cv cross-fits only the meta layer over fixed base OOF features. "
    "Base models used for meta-train rows may have trained on diagnostic validation "
    "groups, so this is not end-to-end nested CV and cannot authorize promotion. "
    "Promotion requires a frozen shadow set excluded from every base/meta fit or "
    "outer-fold base regeneration."
)
BASELINE_PARITY_LIMITATION = (
    "baseline_origin is caller-declared provenance, not artifact verification. "
    "Until a pinned alpha09 sparse-OOF manifest and hash are supplied and checked, "
    "teammate baseline parity and overall teammate parity remain false."
)


def load_records(path: Path, reference_ids: np.ndarray) -> list[dict[str, Any]]:
    wanted = set(str(value) for value in reference_ids)
    by_id: dict[str, dict[str, Any]] = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            sample_id = str(record.get("id", ""))
            if not sample_id:
                raise ValueError(f"{path}:{line_number} has no id")
            if sample_id in by_id:
                raise ValueError(f"{path} contains duplicate id {sample_id}")
            if sample_id in wanted:
                by_id[sample_id] = record
    missing = wanted - set(by_id)
    if missing:
        raise ValueError(f"train records missing {len(missing)} reference IDs")
    return [by_id[str(sample_id)] for sample_id in reference_ids]


def _model(c_value: float, max_iter: int, seed: int) -> Pipeline:
    # loc and count-valued structured features otherwise dominate the optimizer's
    # scale and make lbfgs exhaust thousands of iterations. The scaler is fitted
    # inside each meta fold, so validation statistics never enter meta training.
    return Pipeline(
        [
            ("scale", MaxAbsScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c_value),
                    max_iter=int(max_iter),
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=int(seed),
                ),
            ),
        ]
    )


def _aligned_model_probs(
    model: Pipeline,
    matrix: Any,
    n_actions: int,
) -> np.ndarray:
    raw = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    out = np.zeros((matrix.shape[0], n_actions), dtype=np.float64)
    for source_column, label in enumerate(model.classes_):
        index = int(label)
        if not 0 <= index < n_actions:
            raise ValueError(f"classifier emitted invalid class {label}")
        out[:, index] = raw[:, source_column]
    row_sum = out.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("classifier emitted an empty probability row")
    return out / row_sum


def _score(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    labels = np.arange(len(ACTIONS), dtype=np.int64)
    pred = probs.argmax(axis=1)
    per_class = f1_score(
        y_true,
        pred,
        labels=labels,
        average=None,
        zero_division=0,
    )
    return {
        "macro_f1": float(
            f1_score(
                y_true,
                pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": {
            action: float(per_class[index]) for index, action in enumerate(ACTIONS)
        },
    }


def diagnostic_meta_cv(
    baseline: np.ndarray,
    e5: np.ndarray,
    records: list[dict[str, Any]],
    y_int: np.ndarray,
    folds: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    unique_folds = sorted(int(value) for value in np.unique(folds))
    if len(unique_folds) < 2:
        raise ValueError("diagnostic_meta_cv requires at least two folds")
    oof = np.zeros((len(y_int), len(ACTIONS)), dtype=np.float64)
    fold_rows: list[dict[str, Any]] = []
    for fold in unique_folds:
        valid = np.asarray(folds == fold)
        train = ~valid
        if not train.any() or not valid.any():
            raise ValueError(f"fold {fold} has an empty train or validation partition")
        train_records = [record for index, record in enumerate(records) if train[index]]
        valid_records = [record for index, record in enumerate(records) if valid[index]]
        train_matrix, vectorizer = build_teammate_matrix(
            baseline[train],
            e5[train],
            train_records,
            fit_vectorizer=True,
        )
        valid_matrix, _ = build_teammate_matrix(
            baseline[valid],
            e5[valid],
            valid_records,
            vectorizer=vectorizer,
        )
        model = _model(c_value, max_iter, seed)
        model.fit(train_matrix, y_int[train])
        oof[valid] = _aligned_model_probs(model, valid_matrix, len(ACTIONS))
        fold_score = _score(y_int[valid], oof[valid])
        fold_rows.append(
            {
                "fold": fold,
                "train_rows": int(train.sum()),
                "valid_rows": int(valid.sum()),
                "macro_f1": fold_score["macro_f1"],
                "optimizer_iterations": int(
                    model.named_steps["classifier"].n_iter_.max()
                ),
            }
        )
    result = _score(y_int, oof)
    result.update(
        {
            "name": "diagnostic_meta_cv",
            "promotion_eligible": False,
            "methodology_limitation": METHODOLOGY_LIMITATION,
            "folds": fold_rows,
        }
    )
    return oof, result


_STEP_RE = re.compile(r"-step_(\d+)$")


def _step(sample_id: str) -> int:
    match = _STEP_RE.search(sample_id)
    if match is None:
        raise ValueError(f"cannot parse step from id {sample_id}")
    return int(match.group(1))


def _correction_rows(
    ids: np.ndarray,
    y_int: np.ndarray,
    blend_probs: np.ndarray,
    stacker_probs: np.ndarray,
    baseline_origin: str,
) -> dict[str, Any]:
    blend_pred = blend_probs.argmax(axis=1)
    stacker_pred = stacker_probs.argmax(axis=1)
    blend_wrong_stacker_right = (blend_pred != y_int) & (stacker_pred == y_int)
    blend_right_stacker_wrong = (blend_pred == y_int) & (stacker_pred != y_int)

    def summarize(mask: np.ndarray, keys: np.ndarray) -> dict[str, int]:
        return {
            str(key): int(np.sum(mask & (keys == key)))
            for key in sorted(np.unique(keys).tolist())
        }

    steps = np.asarray([_step(str(sample_id)) for sample_id in ids], dtype=np.int64)
    classes = np.asarray([ACTIONS[index] for index in y_int], dtype=object)
    return {
        "current_blend_definition": f"0.5 * {baseline_origin} + 0.5 * hist12_e5_oof",
        "current_blend_is_parity_surface": False,
        "blend_wrong_stacker_right_total": int(blend_wrong_stacker_right.sum()),
        "blend_right_stacker_wrong_total": int(blend_right_stacker_wrong.sum()),
        "net_corrected_rows": int(
            blend_wrong_stacker_right.sum() - blend_right_stacker_wrong.sum()
        ),
        "blend_wrong_stacker_right_by_step": summarize(
            blend_wrong_stacker_right, steps
        ),
        "blend_right_stacker_wrong_by_step": summarize(
            blend_right_stacker_wrong, steps
        ),
        "blend_wrong_stacker_right_by_class": summarize(
            blend_wrong_stacker_right, classes
        ),
        "blend_right_stacker_wrong_by_class": summarize(
            blend_right_stacker_wrong, classes
        ),
    }


def _numeric_contributions(model: Pipeline, matrix: Any) -> dict[str, Any]:
    names = list(numeric_feature_names())
    scaled = model.named_steps["scale"].transform(matrix)
    classifier = model.named_steps["classifier"]
    mean_abs_value = np.asarray(np.abs(scaled[:, : len(names)]).mean(axis=0)).ravel()
    coefficients = np.asarray(classifier.coef_[:, : len(names)], dtype=np.float64)
    mean_abs_coefficient = np.abs(coefficients).mean(axis=0)
    contributions = mean_abs_value * mean_abs_coefficient

    def row(index: int) -> dict[str, Any]:
        return {
            "feature": names[index],
            "mean_abs_scaled_value": float(mean_abs_value[index]),
            "mean_abs_coefficient": float(mean_abs_coefficient[index]),
            "mean_abs_logit_contribution": float(contributions[index]),
        }

    descending = np.argsort(-contributions, kind="stable")
    ascending = np.argsort(contributions, kind="stable")
    return {
        "definition": (
            "mean over rows and classifier classes of abs(scaled_value * coefficient); "
            "limited to the 34 probability/confidence meta features"
        ),
        "feature_count": len(names),
        "top": [row(int(index)) for index in descending[:10]],
        "bottom": [row(int(index)) for index in ascending[:10]],
    }


def _diagnostics(
    ids: np.ndarray,
    y_int: np.ndarray,
    baseline: np.ndarray,
    e5: np.ndarray,
    stacker_oof: np.ndarray,
    diagnostic: dict[str, Any],
    final_model: Pipeline,
    final_matrix: Any,
    baseline_origin: str,
) -> dict[str, Any]:
    blend = 0.5 * baseline + 0.5 * e5
    fold_scores = np.asarray(
        [row["macro_f1"] for row in diagnostic["folds"]], dtype=np.float64
    )
    support = Counter(int(value) for value in y_int)
    class_rows = []
    component_scores = {
        baseline_origin: _score(y_int, baseline),
        "hist12_e5_oof": _score(y_int, e5),
        "current_equal_blend_proxy": _score(y_int, blend),
        "diagnostic_meta_cv": _score(y_int, stacker_oof),
    }
    for index, action in enumerate(ACTIONS):
        class_rows.append(
            {
                "action": action,
                "support": int(support[index]),
                **{
                    name: float(score["per_class_f1"][action])
                    for name, score in component_scores.items()
                },
            }
        )
    return {
        "promotion_eligible": False,
        "methodology_limitation": METHODOLOGY_LIMITATION,
        "fold_macro_f1": fold_scores.tolist(),
        "fold_macro_f1_mean": float(fold_scores.mean()),
        "fold_macro_f1_population_variance": float(fold_scores.var(ddof=0)),
        "fold_macro_f1_sample_variance": float(fold_scores.var(ddof=1)),
        "fold_macro_f1_std": float(fold_scores.std(ddof=0)),
        "component_macro_f1": {
            name: float(score["macro_f1"])
            for name, score in component_scores.items()
        },
        "class_f1_sorted_by_support": sorted(class_rows, key=lambda row: row["support"]),
        "corrections": _correction_rows(
            ids, y_int, blend, stacker_oof, baseline_origin
        ),
        "numeric_feature_contributions": _numeric_contributions(
            final_model, final_matrix
        ),
    }


def _feature_name_hash(names: list[str]) -> str:
    return hashlib.sha256("\0".join(names).encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_joblib(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def train_stacker(
    *,
    ids: np.ndarray,
    y_true: np.ndarray,
    folds: np.ndarray,
    baseline: np.ndarray,
    e5: np.ndarray,
    records: list[dict[str, Any]],
    baseline_origin: str,
    output_dir: Path,
    c_value: float = 1.0,
    max_iter: int = 3000,
    seed: int = 42,
    input_hashes: dict[str, str] | None = None,
    validate_only: bool = False,
) -> dict[str, Any]:
    allowed_origins = {"alpha09_sparse_oof", "legacy_linear_proxy"}
    if baseline_origin not in allowed_origins:
        raise ValueError(f"baseline_origin must be one of {sorted(allowed_origins)}")
    ids = np.asarray([str(value) for value in ids], dtype=object)
    y_true = np.asarray([str(value) for value in y_true], dtype=object)
    folds = np.asarray(folds)
    if len(ids) != len(y_true) or len(ids) != len(folds) or len(ids) != len(records):
        raise ValueError("ids, labels, folds, and records must have equal lengths")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("ids contain duplicates")
    if not np.issubdtype(folds.dtype, np.integer):
        raise ValueError("folds must have an integer dtype")
    record_ids: list[str] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"record {index} is not a mapping")
        record_id = str(record.get("id", ""))
        if not record_id:
            raise ValueError(f"record {index} has no id")
        record_ids.append(record_id)
    if len(set(record_ids)) != len(record_ids):
        raise ValueError("record IDs contain duplicates")
    if record_ids != ids.tolist():
        raise ValueError("record IDs are not exactly aligned with ids")
    group_fold: dict[str, int] = {}
    for sample_id, fold in zip(ids, folds, strict=True):
        group = session_group(sample_id)
        fold_int = int(fold)
        prior = group_fold.setdefault(group, fold_int)
        if prior != fold_int:
            raise ValueError(f"session {group} is split across folds")
    action_to_index = {action: index for index, action in enumerate(ACTIONS)}
    unknown = sorted(set(y_true.tolist()) - set(ACTIONS))
    if unknown:
        raise ValueError(f"unknown labels: {unknown}")
    y_int = np.asarray([action_to_index[label] for label in y_true], dtype=np.int64)
    teammate_numeric_features(baseline, e5)
    feature_dicts(records)

    fold_distribution = {
        str(int(fold)): int(np.sum(folds == fold)) for fold in sorted(np.unique(folds))
    }
    manifest: dict[str, Any] = {
        "task_id": "CX-002",
        "rows": int(len(ids)),
        "sessions": int(len({session_group(sample_id) for sample_id in ids})),
        "fold_distribution": fold_distribution,
        "actions": list(ACTIONS),
        "numeric_feature_names": list(numeric_feature_names()),
        "baseline_origin": baseline_origin,
        "teammate_baseline_parity_claimed": baseline_origin == "alpha09_sparse_oof",
        "teammate_baseline_parity_verified": False,
        "teammate_structured_feature_contract_pinned": True,
        "teammate_parity": False,
        "baseline_parity_limitation": BASELINE_PARITY_LIMITATION,
        "feature_source": FEATURE_SOURCE,
        "feature_source_commit": FEATURE_SOURCE_COMMIT,
        "feature_source_sha256": FEATURE_SOURCE_SHA256,
        "input_sha256": dict(input_hashes or {}),
        "promotion_eligible": False,
        "methodology_limitation": METHODOLOGY_LIMITATION,
        "final_fit_scored_in_sample": False,
        "validate_only": bool(validate_only),
    }
    output_dir = Path(output_dir)
    if validate_only:
        _atomic_json(output_dir / "validation.json", manifest)
        return manifest

    stacker_oof, diagnostic = diagnostic_meta_cv(
        baseline,
        e5,
        records,
        y_int,
        folds,
        c_value=c_value,
        max_iter=max_iter,
        seed=seed,
    )
    final_matrix, vectorizer = build_teammate_matrix(
        baseline,
        e5,
        records,
        fit_vectorizer=True,
    )
    stacker = _model(c_value, max_iter, seed)
    stacker.fit(final_matrix, y_int)
    vectorizer_names = list(vectorizer.get_feature_names_out())
    model_metadata = {
        "task_id": "CX-002",
        "baseline_origin": baseline_origin,
        "teammate_baseline_parity_claimed": baseline_origin == "alpha09_sparse_oof",
        "teammate_baseline_parity_verified": False,
        "teammate_structured_feature_contract_pinned": True,
        "teammate_parity": False,
        "numeric_feature_names": list(numeric_feature_names()),
        "structured_feature_count": len(vectorizer_names),
        "structured_feature_names_sha256": _feature_name_hash(vectorizer_names),
        "classifier_classes": [int(value) for value in stacker.classes_],
        "optimizer_iterations": int(
            stacker.named_steps["classifier"].n_iter_.max()
        ),
        "preprocessing": "MaxAbsScaler fitted with the classifier",
        "promotion_eligible": False,
        "final_fit_scored_in_sample": False,
    }
    payload = {
        "dict_vectorizer": vectorizer,
        "stacker": stacker,
        "actions": list(ACTIONS),
        "metadata": model_metadata,
    }
    model_path = output_dir / "stacker_h12.joblib"
    _atomic_joblib(model_path, payload)
    manifest.update(
        {
            "diagnostic_meta_cv": diagnostic,
            "diagnostics": _diagnostics(
                ids,
                y_int,
                baseline,
                e5,
                stacker_oof,
                diagnostic,
                stacker,
                final_matrix,
                baseline_origin,
            ),
            "model": {
                **model_metadata,
                "path": str(model_path),
                "sha256": hash_files([model_path])[str(model_path)],
                "C": float(c_value),
                "max_iter": int(max_iter),
                "seed": int(seed),
            },
        }
    )
    _atomic_json(output_dir / "manifest.json", manifest)
    return manifest


def _load_inputs(args: argparse.Namespace) -> tuple[
    OOFBundle,
    np.ndarray,
    list[dict[str, Any]],
    dict[str, str],
]:
    legacy_bundle, legacy_components = load_legacy_oof(
        args.legacy_oof_dir,
        components=(args.baseline_component,),
        target_actions=ACTIONS,
    )
    label_map = dict(zip(legacy_bundle.ids, legacy_bundle.y_true, strict=True))
    hist12 = load_hist12_oof(
        args.hist12_oof_dir,
        reference_ids=legacy_bundle.ids,
        reference_y=label_map,
        expected_actions=ACTIONS,
    )
    aligned_legacy = align_bundle(
        legacy_bundle,
        hist12.ids,
        reference_y=dict(zip(hist12.ids, hist12.y_true, strict=True)),
    )
    legacy_position = {
        sample_id: index for index, sample_id in enumerate(legacy_bundle.ids)
    }
    order = np.asarray(
        [legacy_position[sample_id] for sample_id in aligned_legacy.ids],
        dtype=np.int64,
    )
    baseline = legacy_components[args.baseline_component][order]
    records = load_records(args.train_jsonl, hist12.ids)
    paths = [
        *hist12.sources,
        *legacy_bundle.sources,
        Path(args.train_jsonl),
    ]
    return hist12, baseline, records, hash_files(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hist12-oof-dir", type=Path, required=True)
    parser.add_argument("--legacy-oof-dir", type=Path, required=True)
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--baseline-component", default="linear")
    parser.add_argument(
        "--baseline-origin",
        choices=("alpha09_sparse_oof", "legacy_linear_proxy"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--C", dest="c_value", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hist12, baseline, records, input_hashes = _load_inputs(args)
    if hist12.folds is None:
        raise ValueError("hist12 OOF is missing fold provenance")
    manifest = train_stacker(
        ids=hist12.ids,
        y_true=hist12.y_true,
        folds=hist12.folds,
        baseline=baseline,
        e5=hist12.probs,
        records=records,
        baseline_origin=args.baseline_origin,
        output_dir=args.output_dir,
        c_value=args.c_value,
        max_iter=args.max_iter,
        seed=args.seed,
        input_hashes=input_hashes,
        validate_only=args.validate_only,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
