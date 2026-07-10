from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACTIONS = (
    "read_file",
    "grep_search",
    "list_directory",
    "glob_pattern",
    "edit_file",
    "write_file",
    "apply_patch",
    "run_bash",
    "run_tests",
    "lint_or_typecheck",
    "ask_user",
    "plan_task",
    "web_search",
    "respond_only",
)
PROB_ATOL = 1e-5
_FOLD_RE = re.compile(r"^oof_fold(\d+)\.npz$")


@dataclass(frozen=True)
class OOFBundle:
    ids: np.ndarray
    y_true: np.ndarray
    actions: tuple[str, ...]
    probs: np.ndarray
    folds: np.ndarray | None
    sources: tuple[Path, ...]


def session_group(sample_id: str) -> str:
    return str(sample_id).rsplit("-step_", 1)[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_files(paths: Sequence[Path]) -> dict[str, str]:
    return {str(Path(path)): sha256_file(Path(path)) for path in paths}


def _strings(values: object, name: str, *, unique: bool = False) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    out = np.asarray([str(value) for value in array], dtype=object)
    if any(not value for value in out):
        raise ValueError(f"{name} contains an empty value")
    if unique and len(set(out.tolist())) != len(out):
        raise ValueError(f"{name} contains duplicate values")
    return out


def _actions(values: object, name: str = "actions") -> tuple[str, ...]:
    out = tuple(_strings(values, name, unique=True).tolist())
    if not out:
        raise ValueError(f"{name} is empty")
    return out


def _probabilities(values: object, rows: int, columns: int, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must have a numeric dtype")
    out = np.asarray(raw, dtype=np.float64)
    if out.shape != (rows, columns):
        raise ValueError(f"{name} shape {out.shape} != {(rows, columns)}")
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values")
    if np.any(out < 0.0) or np.any(out > 1.0):
        raise ValueError(f"{name} contains values outside [0, 1]")
    if not np.allclose(out.sum(axis=1), 1.0, atol=PROB_ATOL, rtol=0.0):
        raise ValueError(f"{name} rows do not sum to one")
    return out


def align_probs(
    probs: np.ndarray,
    src_actions: Sequence[str],
    dst_actions: Sequence[str],
) -> np.ndarray:
    src = _actions(src_actions, "src_actions")
    dst = _actions(dst_actions, "dst_actions")
    if set(src) != set(dst):
        missing = sorted(set(dst) - set(src))
        extra = sorted(set(src) - set(dst))
        raise ValueError(f"action set mismatch: missing={missing}, extra={extra}")
    array = _probabilities(probs, len(probs), len(src), "probs")
    positions = {action: index for index, action in enumerate(src)}
    return array[:, [positions[action] for action in dst]]


def _reference_labels(
    reference_ids: np.ndarray,
    reference_y: Mapping[str, str] | Sequence[str] | None,
) -> np.ndarray | None:
    if reference_y is None:
        return None
    if isinstance(reference_y, Mapping):
        missing = [sample_id for sample_id in reference_ids if sample_id not in reference_y]
        extra = set(str(key) for key in reference_y) - set(reference_ids.tolist())
        if missing or extra:
            raise ValueError(
                f"reference label IDs mismatch: missing={len(missing)}, extra={len(extra)}"
            )
        return np.asarray([str(reference_y[sample_id]) for sample_id in reference_ids], dtype=object)
    labels = _strings(reference_y, "reference_y")
    if len(labels) != len(reference_ids):
        raise ValueError("reference_y length does not match reference_ids")
    return labels


def align_bundle(
    bundle: OOFBundle,
    reference_ids: Sequence[str],
    *,
    reference_y: Mapping[str, str] | Sequence[str] | None = None,
) -> OOFBundle:
    ref = _strings(reference_ids, "reference_ids", unique=True)
    source_ids = _strings(bundle.ids, "bundle.ids", unique=True)
    missing = set(ref.tolist()) - set(source_ids.tolist())
    extra = set(source_ids.tolist()) - set(ref.tolist())
    if missing or extra:
        raise ValueError(f"ID mismatch: missing={len(missing)}, extra={len(extra)}")
    position = {sample_id: index for index, sample_id in enumerate(source_ids)}
    order = np.asarray([position[sample_id] for sample_id in ref], dtype=np.int64)
    y_true = np.asarray(bundle.y_true, dtype=object)[order]
    expected_y = _reference_labels(ref, reference_y)
    if expected_y is not None and not np.array_equal(y_true, expected_y):
        raise ValueError("y_true mismatch after ID alignment")
    folds = None if bundle.folds is None else np.asarray(bundle.folds)[order]
    return OOFBundle(
        ids=ref,
        y_true=y_true,
        actions=bundle.actions,
        probs=np.asarray(bundle.probs)[order],
        folds=folds,
        sources=bundle.sources,
    )


def _read_fold_map(path: Path, expected_folds: set[int]) -> tuple[list[str], dict[str, int]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "fold"]:
            raise ValueError(f"{path} must have exactly id,fold columns")
        ordered_ids: list[str] = []
        fold_of: dict[str, int] = {}
        for row_number, row in enumerate(reader, start=2):
            sample_id = str(row["id"])
            if not sample_id:
                raise ValueError(f"{path}:{row_number} has an empty id")
            if sample_id in fold_of:
                raise ValueError(f"{path} contains duplicate id {sample_id}")
            try:
                fold = int(row["fold"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{row_number} has a non-integer fold") from exc
            if fold not in expected_folds:
                raise ValueError(f"{path}:{row_number} has unexpected fold {fold}")
            ordered_ids.append(sample_id)
            fold_of[sample_id] = fold
    if set(fold_of.values()) != expected_folds:
        raise ValueError(f"{path} does not contain every expected fold")
    group_fold: dict[str, int] = {}
    for sample_id, fold in fold_of.items():
        group = session_group(sample_id)
        prior = group_fold.setdefault(group, fold)
        if prior != fold:
            raise ValueError(f"session {group} is split across folds")
    return ordered_ids, fold_of


def load_hist12_oof(
    oof_dir: Path,
    *,
    reference_ids: Sequence[str] | None = None,
    reference_y: Mapping[str, str] | Sequence[str] | None = None,
    expected_actions: Sequence[str] = ACTIONS,
    expected_folds: Sequence[int] = range(5),
    require_fold_map: bool = True,
) -> OOFBundle:
    directory = Path(oof_dir)
    expected = set(int(value) for value in expected_folds)
    if not expected:
        raise ValueError("expected_folds is empty")
    discovered: dict[int, Path] = {}
    for path in directory.glob("oof_fold*.npz"):
        match = _FOLD_RE.match(path.name)
        if not match:
            raise ValueError(f"invalid fold filename: {path.name}")
        fold = int(match.group(1))
        if fold in discovered:
            raise ValueError(f"duplicate fold file for fold {fold}")
        discovered[fold] = path
    if set(discovered) != expected:
        raise ValueError(
            f"fold files mismatch: missing={sorted(expected - set(discovered))}, "
            f"extra={sorted(set(discovered) - expected)}"
        )

    target_actions = _actions(expected_actions, "expected_actions")
    id_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    prob_parts: list[np.ndarray] = []
    fold_parts: list[np.ndarray] = []
    sources: list[Path] = []
    seen: set[str] = set()
    required_keys = {"ids", "probs", "y_true", "actions", "fold"}
    for fold in sorted(discovered):
        path = discovered[fold]
        with np.load(path, allow_pickle=True) as item:
            missing_keys = required_keys - set(item.files)
            if missing_keys:
                raise ValueError(f"{path} missing keys: {sorted(missing_keys)}")
            ids = _strings(item["ids"], f"{path}:ids", unique=True)
            overlap = seen.intersection(ids.tolist())
            if overlap:
                raise ValueError(f"duplicate IDs across folds: {sorted(overlap)[:3]}")
            seen.update(ids.tolist())
            actions = _actions(item["actions"], f"{path}:actions")
            probs = align_probs(item["probs"], actions, target_actions)
            y_true = _strings(item["y_true"], f"{path}:y_true")
            if len(y_true) != len(ids):
                raise ValueError(f"{path} y_true length does not match ids")
            if not set(y_true).issubset(set(target_actions)):
                raise ValueError(f"{path} y_true contains unknown actions")
            folds_raw = np.asarray(item["fold"])
            if folds_raw.ndim != 1 or len(folds_raw) != len(ids):
                raise ValueError(f"{path} fold must be one-dimensional and match ids")
            if not np.issubdtype(folds_raw.dtype, np.integer):
                raise ValueError(f"{path} fold must have an integer dtype")
            folds = np.asarray(folds_raw, dtype=np.int64)
            if not np.all(folds == fold):
                raise ValueError(f"{path} fold values do not match its filename")
        id_parts.append(ids)
        y_parts.append(y_true)
        prob_parts.append(probs)
        fold_parts.append(folds)
        sources.append(path)

    bundle = OOFBundle(
        ids=np.concatenate(id_parts),
        y_true=np.concatenate(y_parts),
        actions=target_actions,
        probs=np.vstack(prob_parts),
        folds=np.concatenate(fold_parts),
        sources=tuple(sources),
    )
    fold_map_path = directory / "fold_map.csv"
    if require_fold_map or fold_map_path.exists():
        if not fold_map_path.exists():
            raise ValueError(f"missing fold map: {fold_map_path}")
        ordered_ids, fold_of = _read_fold_map(fold_map_path, expected)
        npz_ids = set(bundle.ids.tolist())
        map_ids = set(ordered_ids)
        if npz_ids != map_ids:
            raise ValueError(
                f"fold-map IDs mismatch: missing={len(npz_ids-map_ids)}, extra={len(map_ids-npz_ids)}"
            )
        for sample_id, fold in zip(bundle.ids, bundle.folds, strict=True):
            if fold_of[sample_id] != int(fold):
                raise ValueError(f"fold-map mismatch for {sample_id}")
        sources.append(fold_map_path)
        bundle = OOFBundle(
            ids=bundle.ids,
            y_true=bundle.y_true,
            actions=bundle.actions,
            probs=bundle.probs,
            folds=bundle.folds,
            sources=tuple(sources),
        )
        if reference_ids is None:
            reference_ids = ordered_ids

    if reference_ids is not None:
        bundle = align_bundle(bundle, reference_ids, reference_y=reference_y)
    elif reference_y is not None:
        raise ValueError("reference_y requires reference_ids")
    return bundle


def load_legacy_oof(
    oof_dir: Path,
    *,
    components: Sequence[str] = ("linear", "stacker"),
    target_actions: Sequence[str] = ACTIONS,
) -> tuple[OOFBundle, dict[str, np.ndarray]]:
    directory = Path(oof_dir)
    names = tuple(str(name) for name in components)
    if not names or len(set(names)) != len(names):
        raise ValueError("components must be non-empty and unique")
    row_ids_path = directory / "row_ids.json"
    classes_path = directory / "classes.json"
    y_path = directory / "y_true.json"
    row_ids = _strings(json.loads(row_ids_path.read_text(encoding="utf-8")), "row_ids", unique=True)
    y_true = _strings(json.loads(y_path.read_text(encoding="utf-8")), "y_true")
    if len(y_true) != len(row_ids):
        raise ValueError("legacy y_true length does not match row_ids")
    src_actions = _actions(json.loads(classes_path.read_text(encoding="utf-8")), "classes")
    dst_actions = _actions(target_actions, "target_actions")
    if not set(y_true).issubset(set(dst_actions)):
        raise ValueError("legacy y_true contains unknown actions")

    arrays: dict[str, np.ndarray] = {}
    sources = [row_ids_path, classes_path, y_path]
    for name in names:
        path = directory / f"{name}_probs.npy"
        raw = np.load(path, allow_pickle=False)
        arrays[name] = align_probs(raw, src_actions, dst_actions)
        if len(arrays[name]) != len(row_ids):
            raise ValueError(f"legacy component {name} row count mismatch")
        sources.append(path)
    bundle = OOFBundle(
        ids=row_ids,
        y_true=y_true,
        actions=dst_actions,
        probs=arrays[names[0]],
        folds=None,
        sources=tuple(sources),
    )
    return bundle, arrays
