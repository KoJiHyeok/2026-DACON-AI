# -*- coding: utf-8 -*-
"""D-014 Lane A phase 2 — frozen-shadow 5-component parity stacker: feature builder.

Builds the 76-column meta feature matrix specified in the frozen-shadow protocol:

  1-14   linear OOF probs
  15-28  AAR probs
  29-42  e5 probs
  43-56  mBERT probs
  57-70  AU probs (rows outside AU scope = all zeros)
  71     au_mask (1.0 if row is sess_au, else 0.0)
  72-76  per-component entropy (natural log; order linear/aar/e5/mbert/au;
         AU entropy = 0.0 for non-AU rows)

Class column order within each 14-wide block is ACTIONS from
scripts/oof5/common.py, which mirrors artifacts/experiments/oof_h12/oof_fold0.npz's
`actions` array exactly. Every source npz's `actions` array is verified against
ACTIONS before use; mismatch is a hard failure (no silent realignment).

This module only builds feature matrices — no model fitting, no scoring.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np

from common import ACTIONS, ROOT, sha256_file, verify_fold_map, load_fold_map

N_CLASSES = len(ACTIONS)
assert N_CLASSES == 14

OOF_LINEAR_DIR = ROOT / "artifacts" / "experiments" / "oof_linear"
OOF_AAR_DIR = ROOT / "artifacts" / "experiments" / "oof_aar"
OOF_E5_DIR = ROOT / "artifacts" / "experiments" / "oof_h12"
OOF_MBERT_DIR = ROOT / "artifacts" / "experiments" / "oof_mbert_h6"
OOF_AU_DIR = ROOT / "artifacts" / "experiments" / "oof_au"

# NOT USED — spec names these as the linear/AAR holdout surface, but they are
# a disjoint, incompatible holdout split (see _load_oof_rebuild_linear_aar).
# Kept only so the path is visible/greppable next to the substitution reason.
HOLDOUT_LINEAR_NPZ = ROOT / "context" / "night" / "2026-07-05" / "holdout_linear.npz"
HOLDOUT_STACKER_NPZ = ROOT / "context" / "night" / "2026-07-05" / "holdout_stacker.npz"
HOLDOUT_E5_NPZ = ROOT / "colab_out" / "holdout_e5_h12.npz"
HOLDOUT_MBERT_NPZ = ROOT / "colab_out" / "holdout_mbert.npz"
HOLDOUT_AU_NPZ = ROOT / "night_out" / "league4" / "au_charwb_C1_holdout_probs.npz"
HOLDOUT_BASE_NPZ = ROOT / "context" / "night" / "2026-07-05" / "holdout_base.npz"

# champion-parity linear/stacker(AAR) source actually consumed by
# scripts/league4/common.py::load_league_data (see deviation note in report —
# holdout_linear.npz / holdout_stacker.npz are a DIFFERENT, incompatible
# StratifiedGroupKFold(7) split from 2026-07-05/task3 and are NOT usable here).
OOF_REBUILD_DIR = ROOT / "artifacts" / "oof" / "oof_rebuild_2026_07_04"


def _verify_actions(npz_path: Path, z) -> None:
    got = [str(x) for x in z["actions"]]
    if got != list(ACTIONS):
        raise AssertionError(f"{npz_path}: actions mismatch.\n got={got}\n exp={list(ACTIONS)}")


def _entropy(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def load_oof_fold_set(dir_path: Path, prefix: str, n_folds: int = 5) -> dict[str, np.ndarray]:
    """Load and concatenate a component's 5-fold OOF npz set.

    Returns dict with concatenated ids/probs/y_true/fold, each row i built
    from its own fold's OOF prediction (no cross-fold leakage).
    """
    all_ids, all_probs, all_y, all_fold = [], [], [], []
    for i in range(n_folds):
        p = dir_path / f"{prefix}_fold{i}.npz"
        z = np.load(p, allow_pickle=True)
        _verify_actions(p, z)
        all_ids.append(np.asarray([str(x) for x in z["ids"]], dtype=object))
        all_probs.append(np.asarray(z["probs"], dtype=np.float64))
        all_y.append(np.asarray([str(x) for x in z["y_true"]], dtype=object))
        all_fold.append(np.asarray(z["fold"], dtype=np.int64))
    ids = np.concatenate(all_ids)
    probs = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_y)
    fold = np.concatenate(all_fold)
    if len(ids) != len(set(ids.tolist())):
        raise AssertionError(f"{dir_path}: duplicate ids across fold files")
    return {"ids": ids, "probs": probs, "y_true": y_true, "fold": fold}


def sha256sums_dir(dir_path: Path) -> dict[str, str]:
    """Hash every fold npz + fold_map/mask file actually read from dir_path."""
    out: dict[str, str] = {}
    for p in sorted(dir_path.glob("*.npz")):
        out[str(p.relative_to(ROOT))] = sha256_file(p)
    return out


def build_meta_train_features(meta_train_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the 76-col meta-training feature matrix for the given ids.

    Each row's per-component features come from that row's own fold's OOF
    prediction (via id-indexed lookup into the concatenated 5-fold sets).
    """
    verify_fold_map()
    meta_train_ids = np.asarray([str(x) for x in meta_train_ids], dtype=object)
    n = len(meta_train_ids)

    hashes: dict[str, str] = {}

    linear_set = load_oof_fold_set(OOF_LINEAR_DIR, "oof_linear")
    aar_set = load_oof_fold_set(OOF_AAR_DIR, "oof_aar")
    e5_set = load_oof_fold_set(OOF_E5_DIR, "oof")
    mbert_set = load_oof_fold_set(OOF_MBERT_DIR, "oof_mbert")
    au_set = load_oof_fold_set(OOF_AU_DIR, "oof_au")  # AU-scope rows only (~subset)

    for name, d in (
        ("oof_linear", OOF_LINEAR_DIR),
        ("oof_aar", OOF_AAR_DIR),
        ("oof_h12", OOF_E5_DIR),
        ("oof_mbert_h6", OOF_MBERT_DIR),
        ("oof_au", OOF_AU_DIR),
    ):
        hashes[name] = sha256sums_dir(d)

    au_mask_npz = OOF_AU_DIR / "au_row_mask.npz"
    hashes["au_row_mask.npz"] = sha256_file(au_mask_npz)
    zmask = np.load(au_mask_npz, allow_pickle=True)
    mask_ids = np.asarray([str(x) for x in zmask["ids"]], dtype=object)
    mask_is_au = np.asarray(zmask["is_au"], dtype=bool)
    au_mask_lookup = dict(zip(mask_ids.tolist(), mask_is_au.tolist()))

    # y_true consistency across sources (per-id) — build lookups once.
    def y_lookup(s: dict[str, np.ndarray]) -> dict[str, str]:
        return dict(zip(s["ids"].tolist(), s["y_true"].tolist()))

    y_lin = y_lookup(linear_set)
    y_aar = y_lookup(aar_set)
    y_e5 = y_lookup(e5_set)
    y_mbert = y_lookup(mbert_set)
    y_au = y_lookup(au_set)

    idx_lin = {v: i for i, v in enumerate(linear_set["ids"].tolist())}
    idx_aar = {v: i for i, v in enumerate(aar_set["ids"].tolist())}
    idx_e5 = {v: i for i, v in enumerate(e5_set["ids"].tolist())}
    idx_mbert = {v: i for i, v in enumerate(mbert_set["ids"].tolist())}
    idx_au = {v: i for i, v in enumerate(au_set["ids"].tolist())}

    missing_core = [
        i for i in meta_train_ids
        if i not in idx_lin or i not in idx_aar or i not in idx_e5 or i not in idx_mbert
    ]
    if missing_core:
        raise AssertionError(f"{len(missing_core)} meta-train ids missing a core (non-AU) OOF source, e.g. {missing_core[:3]}")

    X = np.zeros((n, 76), dtype=np.float64)
    y_true = np.empty(n, dtype=object)
    au_scope = np.zeros(n, dtype=bool)

    for row, sample_id in enumerate(meta_train_ids.tolist()):
        li, ai, ei, mi = idx_lin[sample_id], idx_aar[sample_id], idx_e5[sample_id], idx_mbert[sample_id]

        yset = {y_lin[sample_id], y_aar[sample_id], y_e5[sample_id], y_mbert[sample_id]}
        is_au_row = bool(au_mask_lookup.get(sample_id, False))
        if is_au_row and sample_id in y_au:
            yset.add(y_au[sample_id])
        if len(yset) != 1:
            raise AssertionError(f"y_true mismatch across sources for id={sample_id}: {yset}")
        y_true[row] = next(iter(yset))

        p_lin = linear_set["probs"][li]
        p_aar = aar_set["probs"][ai]
        p_e5 = e5_set["probs"][ei]
        p_mbert = mbert_set["probs"][mi]

        X[row, 0:14] = p_lin
        X[row, 14:28] = p_aar
        X[row, 28:42] = p_e5
        X[row, 42:56] = p_mbert

        if is_au_row and sample_id in idx_au:
            p_au = au_set["probs"][idx_au[sample_id]]
            X[row, 56:70] = p_au
            X[row, 70] = 1.0
            au_scope[row] = True
            au_ent = _entropy(p_au[None, :])[0]
        else:
            X[row, 56:70] = 0.0
            X[row, 70] = 1.0 if is_au_row else 0.0
            au_scope[row] = is_au_row
            au_ent = 0.0

        X[row, 71] = _entropy(p_lin[None, :])[0]
        X[row, 72] = _entropy(p_aar[None, :])[0]
        X[row, 73] = _entropy(p_e5[None, :])[0]
        X[row, 74] = _entropy(p_mbert[None, :])[0]
        X[row, 75] = au_ent

    meta = {
        "n_rows": n,
        "n_au_scope_rows": int(au_scope.sum()),
        "source_hashes": hashes,
    }
    return X, y_true.astype(object), meta


def _load_oof_rebuild_linear_aar(holdout_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Champion-parity linear+AAR holdout surface.

    DEVIATION FROM SPEC: the spec names context/night/2026-07-05/holdout_linear.npz
    and holdout_stacker.npz as the eval-time linear/AAR surfaces. Those two files
    were verified (see module docstring / final report) to come from a DIFFERENT,
    incompatible StratifiedGroupKFold(7) split built in 2026-07-05/task3 — only
    1,248 of holdout_base.npz's 9,969 ids overlap (~12.5%), so aligning by id
    against holdout_base would silently drop/misalign ~87% of rows. This is not
    a resplit of the same rows; it is a disjoint holdout definition.

    Instead this uses artifacts/oof/oof_rebuild_2026_07_04/{linear,stacker}_probs.npy
    + row_ids.json, which is the actual source scripts/league4/common.py::load_league_data
    and probe_c_args_lite.py already consume for the champion baseline (full 70k
    row_ids, honest per-fold OOF at inference position, sliced to holdout_base.ids).
    This keeps baseline and candidate on the same, verified-correct linear/AAR surface.
    """
    import json as _json

    row_ids = _json.loads((OOF_REBUILD_DIR / "row_ids.json").read_text(encoding="utf-8"))
    classes = _json.loads((OOF_REBUILD_DIR / "classes.json").read_text(encoding="utf-8"))
    classes = [str(c) for c in classes]
    col = [classes.index(a) for a in ACTIONS]
    row_index = {str(x): i for i, x in enumerate(row_ids)}
    missing = [i for i in holdout_ids.tolist() if i not in row_index]
    if missing:
        raise AssertionError(f"{len(missing)} holdout ids missing from oof_rebuild_2026_07_04 row_ids ({missing[:3]})")
    rows = np.asarray([row_index[i] for i in holdout_ids.tolist()], dtype=np.int64)
    lin = np.load(OOF_REBUILD_DIR / "linear_probs.npy")[rows][:, col].astype(np.float64)
    aar = np.load(OOF_REBUILD_DIR / "stacker_probs.npy")[rows][:, col].astype(np.float64)
    h = hashlib.sha256()
    h.update((OOF_REBUILD_DIR / "linear_probs.npy").read_bytes())
    lin_hash = h.hexdigest()
    h2 = hashlib.sha256()
    h2.update((OOF_REBUILD_DIR / "stacker_probs.npy").read_bytes())
    aar_hash = h2.hexdigest()
    return lin, aar, np.array([lin_hash, aar_hash])


def build_holdout_features(holdout_ids: Sequence[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build the 76-col holdout feature matrix from the 85%-trained surfaces
    (not the OOF npz sets), aligned to holdout_ids order.

    Linear/AAR come from artifacts/oof/oof_rebuild_2026_07_04 (champion-parity
    substitution — see _load_oof_rebuild_linear_aar docstring for why the
    spec-named holdout_linear.npz/holdout_stacker.npz could not be used).
    """
    holdout_ids = np.asarray([str(x) for x in holdout_ids], dtype=object)
    n = len(holdout_ids)
    hashes: dict[str, str] = {}

    def load_full(p: Path) -> dict:
        z = np.load(p, allow_pickle=True)
        _verify_actions(p, z)
        ids = np.asarray([str(x) for x in z["ids"]], dtype=object)
        probs = np.asarray(z["probs"], dtype=np.float64)
        y_true = np.asarray([str(x) for x in z["y_true"]], dtype=object)
        return {"ids": ids, "probs": probs, "y_true": y_true}

    e5_full = load_full(HOLDOUT_E5_NPZ)
    mbert_full = load_full(HOLDOUT_MBERT_NPZ)

    lin_probs_by_row, aar_probs_by_row, rebuild_hashes = _load_oof_rebuild_linear_aar(holdout_ids)
    hashes["oof_rebuild_2026_07_04/linear_probs.npy"] = str(rebuild_hashes[0])
    hashes["oof_rebuild_2026_07_04/stacker_probs.npy"] = str(rebuild_hashes[1])

    for name, p in (
        ("holdout_e5_h12.npz", HOLDOUT_E5_NPZ),
        ("holdout_mbert.npz", HOLDOUT_MBERT_NPZ),
        ("au_charwb_C1_holdout_probs.npz", HOLDOUT_AU_NPZ),
    ):
        hashes[name] = sha256_file(p)

    zau = np.load(HOLDOUT_AU_NPZ, allow_pickle=True)
    _verify_actions(HOLDOUT_AU_NPZ, zau)
    au_ids = np.asarray([str(x) for x in zau["ids"]], dtype=object)
    au_probs = np.asarray(zau["probs"], dtype=np.float64)
    idx_au = {v: i for i, v in enumerate(au_ids.tolist())}

    idx_e5 = {v: i for i, v in enumerate(e5_full["ids"].tolist())}
    idx_mbert = {v: i for i, v in enumerate(mbert_full["ids"].tolist())}

    missing = [
        i for i in holdout_ids.tolist()
        if i not in idx_e5 or i not in idx_mbert
    ]
    if missing:
        raise AssertionError(
            f"{len(missing)} holdout ids missing from a component holdout surface "
            f"(e.g. {missing[:5]}) — id alignment hard-fail per spec"
        )

    X = np.zeros((n, 76), dtype=np.float64)
    y_true = np.empty(n, dtype=object)
    au_scope = np.zeros(n, dtype=bool)

    for row, sample_id in enumerate(holdout_ids.tolist()):
        ei, mi = idx_e5[sample_id], idx_mbert[sample_id]
        yset = {e5_full["y_true"][ei], mbert_full["y_true"][mi]}
        is_au_row = sample_id in idx_au
        if len(yset) != 1:
            raise AssertionError(f"holdout y_true mismatch across sources for id={sample_id}: {yset}")
        y_true[row] = next(iter(yset))

        p_lin = lin_probs_by_row[row]
        p_aar = aar_probs_by_row[row]
        p_e5 = e5_full["probs"][ei]
        p_mbert = mbert_full["probs"][mi]

        X[row, 0:14] = p_lin
        X[row, 14:28] = p_aar
        X[row, 28:42] = p_e5
        X[row, 42:56] = p_mbert

        if is_au_row:
            p_au = au_probs[idx_au[sample_id]]
            X[row, 56:70] = p_au
            X[row, 70] = 1.0
            au_scope[row] = True
            au_ent = _entropy(p_au[None, :])[0]
        else:
            X[row, 56:70] = 0.0
            X[row, 70] = 0.0
            au_ent = 0.0

        X[row, 71] = _entropy(p_lin[None, :])[0]
        X[row, 72] = _entropy(p_aar[None, :])[0]
        X[row, 73] = _entropy(p_e5[None, :])[0]
        X[row, 74] = _entropy(p_mbert[None, :])[0]
        X[row, 75] = au_ent

    meta = {
        "n_rows": n,
        "n_au_scope_rows": int(au_scope.sum()),
        "source_hashes": hashes,
        "deviation_note": (
            "holdout_linear.npz/holdout_stacker.npz as named in the frozen-shadow "
            "spec are from an INCOMPATIBLE StratifiedGroupKFold(7) split built in "
            "context/night/2026-07-05/task3 (only 1248/9969 ids overlap holdout_base.npz). "
            "This build uses those two files as literally named as inputs but they were "
            "cross-checked and found unusable; see shadow_eval.py / final report for the "
            "champion-parity substitution actually used (artifacts/oof/oof_rebuild_2026_07_04)."
        ),
    }
    return X, y_true.astype(object), meta
