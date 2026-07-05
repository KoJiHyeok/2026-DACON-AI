"""R4 probability-level explore hierarchy integration.

`scripts/hierarchy/proto_hier.py` showed that a hard-label swap (family gate ->
explore specialist) beats the flat 14-way baseline on 5-fold CV
(0.6638 -> 0.6883 macro-F1, strict route). That swap operates on argmax labels
and is not compatible with a probability-weighted blend (e.g. the w112 3-way
average of linear/stacker/encoder probabilities): overwriting a label after
the blend throws away the blend's calibration on every other row.

This script re-implements R4 at the probability level so it can sit on top of
a blended base probability matrix P (N x 14) instead of a single model's
labels:

  V1 (override-like): for rows the family gate calls "explore", keep P's
  mass on the other 10 classes untouched, and redistribute P's explore-4-class
  *mass* (the sum of P over {read_file, grep_search, list_directory,
  glob_pattern}) according to the explore specialist's internal distribution.
  Non-explore rows are untouched.

  V2 (soft route): for every row and every family f in
  {explore, mutate, validate, coordinate}, replace the family's probability
  mass with the gate's probability for f, and replace the family's internal
  distribution with the specialist's distribution (explore) or P's own
  renormalized internal distribution (other families). This is the
  probability analogue of proto's "strict family route".

Both transforms are mass-preserving re-allocations, so if the inputs are
valid probability rows (sum to 1), the outputs are too.

Base probability matrix (per team-lead direction after the encoder holdout
npz arrived): the canonical evaluation set is the encoder's own holdout
(`holdout_base.npz`, 9,969 rows) rather than the earlier `holdout_linear.npz`
one (the two npz turned out to disagree on fold membership -- only 1,248 rows
overlap out of ~10k, suspected sklearn/stratify-encoding version drift between
the local venv and the Colab environment that produced the encoder npz). The
default base blend is w112's real-world weight recipe [linear 1, stacker 1,
encoder 2], with linear/stacker probabilities taken from the honest 3-fold OOF
bundle (artifacts/oof/oof_rebuild_2026_07_04) joined onto the encoder's
holdout ids.

Design choices carried over from proto_hier.py / collect_probs.py for
consistency with the existing base npz artifacts:
  - Same feature builder (`proto_hier._row`) and preprocessor
    (`proto_hier.build_preprocessor`), same `LinearSVC(C=0.1,
    class_weight="balanced")` classifier config for both the family gate and
    the explore specialist.
  - Probabilities are obtained via softmax(decision_function), not
    CalibratedClassifierCV. `collect_probs.py` already produces the linear
    base npz this way, so using the same recipe for the gate/specialist keeps
    every probability in this pipeline calibrated on the same scale, and
    avoids spending extra CV folds on Platt scaling inside an already
    holdout-constrained 85% training split.

Gate/specialist training split: the encoder's holdout was built with
StratifiedGroupKFold(n_splits=7, shuffle=True, random_state=42), groups =
session id with the trailing "-step_<n>" removed, stratify = the raw action
string (see colab/holdout_eval.py). This script recomputes that split locally
as a diagnostic and asserts whether it reproduces the encoder's holdout id set
exactly. Regardless of that diagnostic's outcome, the actual gate/specialist
training partition is always built directly from id membership against the
encoder's holdout ids (train = every row whose id is *not* in the encoder
holdout; valid = exactly the encoder holdout, in the encoder's own row order).
This is deliberately the same thing as using the recomputed split whenever the
diagnostic matches, and is the explicit fallback the moment it does not, so
gate/specialist training is always leakage-safe with respect to the
canonical evaluation set even if local sklearn reproduces a different fold
assignment than the Colab run that produced holdout_base.npz. A session-group
disjointness check between the two partitions is also run and reported.

Usage:
    python scripts/hierarchy/r4_integrate.py
    python scripts/hierarchy/r4_integrate.py --encoder-weight 2 --linear-oof-weight 1 --stacker-oof-weight 1
    python scripts/hierarchy/r4_integrate.py --npz extra_component.npz:0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "hierarchy"))
import proto_hier as PH  # noqa: E402  (reuse feature builder / classifier config)

DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_NIGHT_DIR = ROOT / "context" / "night" / "2026-07-05"
DEFAULT_OOF_DIR = ROOT / "artifacts" / "oof" / "oof_rebuild_2026_07_04"
DEFAULT_ENCODER_NPZ = DEFAULT_NIGHT_DIR / "holdout_base.npz"
DEFAULT_OUT_DIR = ROOT / "scripts" / "hierarchy" / "_out"

SEED = 42
VALID_FRAC = 0.15
EXPLORE_IDX = [i for i, a in enumerate(PH.ACTIONS) if a in PH.EXPLORE]
FAMILY_IDX = {
    family: [i for i, a in enumerate(PH.ACTIONS) if a in members]
    for family, members in PH.FAMILIES.items()
}


def load_samples_and_labels(data_root: Path):
    train_jsonl = data_root / "train.jsonl"
    labels_csv = data_root / "train_labels.csv"
    if not train_jsonl.exists() or not labels_csv.exists():
        raise FileNotFoundError(f"missing train files under {data_root}")

    samples = []
    with train_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    with labels_csv.open(encoding="utf-8", newline="") as handle:
        labels = {row["id"]: row["action"] for row in csv.DictReader(handle)}

    ids = np.array([str(s["id"]) for s in samples])
    y = np.array([labels[str(i)] for i in ids])
    groups = np.array([PH.session_id(i) for i in ids])
    return samples, ids, y, groups


def make_holdout_split(y: np.ndarray, groups: np.ndarray, seed: int, valid_frac: float):
    n_splits = max(2, int(round(1.0 / valid_frac)))
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
    overlap = set(groups[tr_idx]) & set(groups[va_idx])
    if overlap:
        raise AssertionError(f"session leakage: {len(overlap)} overlapping groups")
    return tr_idx, va_idx, n_splits


def softmax(scores: np.ndarray) -> np.ndarray:
    z = np.asarray(scores, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def decision_probs(clf, x_matrix) -> tuple[np.ndarray, list[str]]:
    scores = clf.decision_function(x_matrix)
    classes = [str(c) for c in clf.classes_]
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return softmax(scores), classes


def parse_npz_spec(spec: str) -> tuple[Path, float]:
    """Parse "path[:weight]". Windows paths contain a drive-letter colon, so
    only the *last* colon is treated as a weight separator, and only if what
    follows it actually parses as a float."""
    head, sep, tail = spec.rpartition(":")
    if sep:
        try:
            return Path(head), float(tail)
        except ValueError:
            pass
    return Path(spec), 1.0


def load_npz_component(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    ids = data["ids"].astype(str)
    actions = [str(a) for a in data["actions"]]
    probs = np.asarray(data["probs"], dtype=np.float64)
    y_true = data["y_true"].astype(str)

    missing = set(PH.ACTIONS) - set(actions)
    if missing:
        raise ValueError(f"{path}: npz is missing action columns {sorted(missing)}")
    col_idx = [actions.index(a) for a in PH.ACTIONS]
    probs = probs[:, col_idx]

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(
            f"{path}: probability rows do not sum to 1 (min={row_sums.min():.4f}, "
            f"max={row_sums.max():.4f})"
        )
    return ids, probs, y_true


def load_oof_component(oof_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load an honest 3-fold OOF probability set (artifacts/oof/oof_rebuild_2026_07_04).

    These cover all 70,000 train rows (not just a holdout): each row's
    probability comes from a StratifiedGroupKFold(n_splits=3) fold model that
    did not see that row's session during training, so it is safe to slice
    out just the encoder's holdout rows and treat them as honest for that
    holdout, even though the 3-fold split is unrelated to the 7-fold 85/15
    split used to build holdout_base.npz.
    """
    with (oof_dir / "classes.json").open(encoding="utf-8") as handle:
        classes = [str(c) for c in json.load(handle)]
    with (oof_dir / "row_ids.json").open(encoding="utf-8") as handle:
        ids = np.array([str(x) for x in json.load(handle)])
    with (oof_dir / "y_true.json").open(encoding="utf-8") as handle:
        y_true = np.array([str(x) for x in json.load(handle)])
    probs = np.load(oof_dir / f"{name}_probs.npy").astype(np.float64)

    missing = set(PH.ACTIONS) - set(classes)
    if missing:
        raise ValueError(f"{oof_dir}/{name}: missing action columns {sorted(missing)}")
    col_idx = [classes.index(a) for a in PH.ACTIONS]
    probs = probs[:, col_idx]

    row_sums = probs.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        raise ValueError(
            f"{oof_dir}/{name}: probability rows do not sum to 1 "
            f"(min={row_sums.min():.4f}, max={row_sums.max():.4f})"
        )
    return ids, probs, y_true


def align_component(
    ids: np.ndarray,
    probs: np.ndarray,
    y_true: np.ndarray,
    canonical_ids: np.ndarray,
    canonical_y: np.ndarray,
    label: str,
    require_exact_order: bool,
) -> np.ndarray:
    if require_exact_order:
        assert np.array_equal(ids, canonical_ids), (
            f"{label}: ids do not match the canonical holdout order"
        )
        assert np.array_equal(y_true, canonical_y), (
            f"{label}: y_true does not match the canonical holdout"
        )
        return probs

    id_set = set(ids)
    missing_ids = [c for c in canonical_ids if c not in id_set]
    if missing_ids:
        raise ValueError(
            f"{label}: missing {len(missing_ids)} holdout ids, e.g. {missing_ids[:3]}"
        )
    pos = {v: k for k, v in enumerate(ids)}
    order = np.array([pos[c] for c in canonical_ids])
    aligned_probs = probs[order]
    aligned_y = y_true[order]
    if not np.array_equal(aligned_y, canonical_y):
        bad = int(np.sum(aligned_y != canonical_y))
        raise ValueError(f"{label}: y_true mismatch after id-alignment on {bad} rows")
    return aligned_probs


def build_base_probs(components: list[dict], canonical_ids: np.ndarray, canonical_y: np.ndarray) -> np.ndarray:
    """components: list of {label, ids, probs, y_true, weight, exact}."""
    total_weight = sum(c["weight"] for c in components)
    if total_weight <= 0:
        raise ValueError("sum of component weights must be positive")

    base = np.zeros((len(canonical_ids), len(PH.ACTIONS)), dtype=np.float64)
    for component in components:
        aligned_probs = align_component(
            component["ids"],
            component["probs"],
            component["y_true"],
            canonical_ids,
            canonical_y,
            component["label"],
            component.get("exact", False),
        )
        base += component["weight"] * aligned_probs

    base /= total_weight
    row_sums = base.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "combined base probs do not sum to 1"
    return base


def integrate_v1_override(
    p_base: np.ndarray, gate_hard: np.ndarray, spec_dist: np.ndarray
) -> np.ndarray:
    """Gate-selected rows only: redistribute P's explore-4-class mass using
    the specialist's internal distribution. Non-explore classes and
    non-explore rows are untouched."""
    v1 = p_base.copy()
    explore_mask = gate_hard == "explore"
    if explore_mask.any():
        explore_mass = p_base[np.ix_(explore_mask, EXPLORE_IDX)].sum(axis=1, keepdims=True)
        v1[np.ix_(explore_mask, EXPLORE_IDX)] = explore_mass * spec_dist[explore_mask]
    assert np.allclose(v1.sum(axis=1), 1.0, atol=1e-6), "V1 rows do not sum to 1"
    return v1


def integrate_v2_soft_route(
    p_base: np.ndarray, gate_probs: np.ndarray, family_classes: list[str], spec_dist: np.ndarray
) -> np.ndarray:
    """Every row: family mass <- gate probability for that family. Internal
    distribution within a family <- specialist (explore) or P's own
    renormalized internal distribution (other families)."""
    eps = 1e-9
    v2 = np.zeros_like(p_base)
    for family, idxs in FAMILY_IDX.items():
        gate_col = gate_probs[:, family_classes.index(family)]
        if family == "explore":
            v2[:, idxs] = gate_col[:, None] * spec_dist
            continue
        fam_mass = p_base[:, idxs].sum(axis=1)
        low_mass = fam_mass < eps
        internal = np.zeros((len(fam_mass), len(idxs)))
        internal[~low_mass] = p_base[~low_mass][:, idxs] / fam_mass[~low_mass, None]
        if low_mass.any():
            internal[low_mass] = 1.0 / len(idxs)
        v2[:, idxs] = gate_col[:, None] * internal
    assert np.allclose(v2.sum(axis=1), 1.0, atol=1e-6), "V2 rows do not sum to 1"
    return v2


def evaluate(probs: np.ndarray, y_true: np.ndarray) -> dict:
    preds = np.array(PH.ACTIONS)[probs.argmax(axis=1)]
    macro_f1 = f1_score(y_true, preds, labels=PH.ACTIONS, average="macro", zero_division=0)
    explore_labels = sorted(PH.EXPLORE)
    explore_macro_f1 = f1_score(y_true, preds, labels=explore_labels, average="macro", zero_division=0)
    per_explore = f1_score(y_true, preds, labels=explore_labels, average=None, zero_division=0)
    row = {"macro_f1": float(macro_f1), "explore_macro_f1": float(explore_macro_f1)}
    for action, value in zip(explore_labels, per_explore):
        row[f"f1_{action}"] = float(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--encoder-npz", default=str(DEFAULT_ENCODER_NPZ))
    parser.add_argument("--encoder-weight", type=float, default=2.0)
    parser.add_argument(
        "--oof-dir",
        default=str(DEFAULT_OOF_DIR),
        help="Honest 3-fold OOF probability bundle (linear_probs.npy, stacker_probs.npy, "
        "row_ids.json, y_true.json, classes.json).",
    )
    parser.add_argument("--linear-oof-weight", type=float, default=1.0)
    parser.add_argument("--stacker-oof-weight", type=float, default=1.0)
    parser.add_argument(
        "--npz",
        action="append",
        default=None,
        help='Extra base probability component "path[:weight]", repeatable. '
        "Added on top of the default 3-way [linear-oof, stacker-oof, encoder] blend.",
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--valid-frac", type=float, default=VALID_FRAC)
    args = parser.parse_args()

    data_root = Path(args.data_root)
    oof_dir = Path(args.oof_dir)
    encoder_npz = Path(args.encoder_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    extra_specs = [parse_npz_spec(s) for s in args.npz] if args.npz else []

    print("[data] loading train.jsonl + train_labels.csv")
    samples, ids, y, groups = load_samples_and_labels(data_root)
    y_lookup = dict(zip(ids, y))

    print(f"[base] loading encoder holdout npz: {encoder_npz}")
    encoder_ids, encoder_probs, encoder_y = load_npz_component(encoder_npz)
    canonical_ids = encoder_ids
    canonical_y = encoder_y
    print(f"[base] canonical holdout rows (encoder) = {len(canonical_ids):,}")

    # Sanity: train_labels.csv agrees with the encoder npz's y_true for these ids.
    our_y_for_canonical = np.array([y_lookup[i] for i in canonical_ids])
    assert np.array_equal(our_y_for_canonical, canonical_y), (
        "encoder npz y_true does not match train_labels.csv for the same ids"
    )

    # Diagnostic: does a local recompute of the 7-fold split reproduce the
    # encoder's holdout id set? (Documented as a possible mismatch by
    # team-lead -- suspected sklearn/stratify-encoding drift between this
    # venv and the Colab environment that produced holdout_base.npz.)
    tr_idx_r, va_idx_r, n_splits_r = make_holdout_split(y, groups, seed=args.seed, valid_frac=args.valid_frac)
    recompute_ids = ids[va_idx_r]
    set_match = set(recompute_ids) == set(canonical_ids)
    order_match = set_match and np.array_equal(recompute_ids, canonical_ids)
    print(
        f"[diagnostic] local StratifiedGroupKFold(n_splits={n_splits_r}) recompute vs "
        f"encoder holdout: set_match={set_match} order_match={order_match} "
        f"(overlap={len(set(recompute_ids) & set(canonical_ids))}/{len(canonical_ids)})"
    )
    if not order_match:
        print(
            "[fallback] recompute did not reproduce the encoder split exactly -> gate/specialist "
            "train/valid partition is built directly from id membership against the encoder's "
            "holdout ids (train = id not in encoder holdout; valid = encoder holdout, encoder's "
            "own row order). This is always leakage-safe w.r.t. the canonical evaluation set "
            "regardless of whether the local recompute matches."
        )

    print("[features] building rows for all samples (proto_hier._row)")
    df = pd.DataFrame([PH._row(s) for s in samples])
    df_ids = df["id"].to_numpy()
    val_mask = np.isin(df_ids, canonical_ids)
    train_mask = ~val_mask

    # Session-group disjointness check for the id-membership partition.
    groups_all = np.array([PH.session_id(i) for i in df_ids])
    train_sessions = set(groups_all[train_mask])
    val_sessions = set(groups_all[val_mask])
    session_overlap = train_sessions & val_sessions
    if session_overlap:
        print(
            f"[WARNING] {len(session_overlap)} sessions appear on both sides of the "
            "encoder-holdout-based train/valid partition -- possible leakage."
        )
    else:
        print("[ok] no session overlap between gate/specialist train and the encoder holdout")

    x_train = df.loc[train_mask].reset_index(drop=True)
    y_train = y[train_mask]

    # x_valid must be reindexed into the encoder's own row order so gate/spec
    # outputs align 1:1 with canonical_ids / canonical_y / encoder_probs.
    df_by_id = df.set_index("id", drop=False)
    x_valid = df_by_id.loc[canonical_ids].reset_index(drop=True)

    print(f"[split] gate/specialist train={len(x_train):,} valid={len(x_valid):,}")

    y_family_train = np.array([PH.action_family(a) for a in y_train])
    y_family_valid = np.array([PH.action_family(a) for a in canonical_y])

    print("[fit] preprocessor + family gate + explore specialist on the encoder-holdout-complement train split")
    preprocessor = PH.build_preprocessor()
    x_train_matrix = preprocessor.fit_transform(x_train)
    x_valid_matrix = preprocessor.transform(x_valid)

    family_clf = PH.build_classifier()
    family_clf.fit(x_train_matrix, y_family_train)
    gate_probs, family_classes = decision_probs(family_clf, x_valid_matrix)
    gate_hard = np.array(family_classes)[gate_probs.argmax(axis=1)]

    explore_mask_train = np.isin(y_train, list(PH.EXPLORE))
    explore_clf = PH.build_classifier()
    explore_clf.fit(x_train_matrix[explore_mask_train], y_train[explore_mask_train])
    spec_probs_raw, explore_classes = decision_probs(explore_clf, x_valid_matrix)
    reorder = [explore_classes.index(PH.ACTIONS[i]) for i in EXPLORE_IDX]
    spec_dist = spec_probs_raw[:, reorder]  # aligned to EXPLORE_IDX order, rows sum to 1

    stage1_family_f1 = f1_score(
        y_family_valid, gate_hard, labels=sorted(PH.FAMILIES), average="macro", zero_division=0
    )
    print(f"[gate] stage-1 family macro-F1 on holdout = {stage1_family_f1:.5f}")
    print(f"[gate] holdout explore rate (predicted) = {float(np.mean(gate_hard == 'explore')):.5f}")

    print(f"[base] loading honest 3-fold OOF linear+stacker from {oof_dir}")
    lin_ids, lin_probs, lin_y = load_oof_component(oof_dir, "linear")
    stk_ids, stk_probs, stk_y = load_oof_component(oof_dir, "stacker")

    # Solo sanity check against team-lead's reproduced reference numbers
    # (linear=0.66765, stacker=0.70566, encoder=0.70509 on this same holdout).
    solo_rows = []
    for label, cids, cprobs, cy, exact in [
        ("linear_oof", lin_ids, lin_probs, lin_y, False),
        ("stacker_oof", stk_ids, stk_probs, stk_y, False),
        ("encoder", encoder_ids, encoder_probs, encoder_y, True),
    ]:
        aligned = align_component(cids, cprobs, cy, canonical_ids, canonical_y, label, exact)
        metrics = evaluate(aligned, canonical_y)
        solo_rows.append({"component": label, **metrics})
    print("[solo] component macro-F1 on the canonical (encoder) holdout:")
    print(pd.DataFrame(solo_rows).to_string(index=False))

    components = [
        {"label": "oof_linear", "ids": lin_ids, "probs": lin_probs, "y_true": lin_y, "weight": args.linear_oof_weight},
        {"label": "oof_stacker", "ids": stk_ids, "probs": stk_probs, "y_true": stk_y, "weight": args.stacker_oof_weight},
        {
            "label": "encoder_holdout_base",
            "ids": encoder_ids,
            "probs": encoder_probs,
            "y_true": encoder_y,
            "weight": args.encoder_weight,
            "exact": True,
        },
    ]
    for path, weight in extra_specs:
        eids, eprobs, ey = load_npz_component(path)
        components.append({"label": str(path), "ids": eids, "probs": eprobs, "y_true": ey, "weight": weight})

    weight_desc = ", ".join(f"{c['label']}={c['weight']:g}" for c in components)
    print(f"[base] 3-way blend weights: {weight_desc}")
    p_base = build_base_probs(components, canonical_ids, canonical_y)

    v1 = integrate_v1_override(p_base, gate_hard, spec_dist)
    v2 = integrate_v2_soft_route(p_base, gate_probs, family_classes, spec_dist)

    rows = []
    for variant_name, probs in [("base_argmax", p_base), ("v1_override", v1), ("v2_soft_route", v2)]:
        metrics = evaluate(probs, canonical_y)
        rows.append({"variant": variant_name, **metrics})

    summary = pd.DataFrame(rows)
    summary_path = out_dir / "r4_integrate_summary.csv"
    summary.to_csv(summary_path, index=False)

    print()
    print(f"[base] weighted 3-way base ({weight_desc}) vs V1/V2:")
    print(summary.to_string(index=False))
    print()
    base_macro = rows[0]["macro_f1"]
    for row in rows[1:]:
        print(f"[delta] {row['variant']} vs base_argmax: {row['macro_f1'] - base_macro:+.5f} macro-F1")
    print(
        "[reference] proto_hier.py hard-label 5-fold CV (context/night/2026-07-05/task2_report.md): "
        "flat=0.66378 -> strict_route=0.68834 (+0.02456)"
    )
    print(
        "[reference] team-lead reproduced base numbers on this holdout: linear=0.66765 "
        "stacker=0.70566 encoder=0.70509 3-way[1,1,2]=0.71726 (LB=0.71884)"
    )
    print(f"[out] wrote {summary_path}")


if __name__ == "__main__":
    main()
