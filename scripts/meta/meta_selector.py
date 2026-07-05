"""Meta-selector experiment on top of the 3-way (linear/stacker/encoder) blend.

Background
----------
The encoder's probabilities only exist for a 9,969-row holdout produced by an
85%-trained encoder checkpoint (`context/night/2026-07-05/holdout_base.npz`).
Linear/stacker probabilities for the same rows come from the honest 3-fold
OOF rebuild (`artifacts/oof/oof_rebuild_2026_07_04/`). Because the encoder OOF
does not exist, all meta-model fitting and evaluation happens strictly inside
these 9,969 rows via a nested session-group K-fold (never touching the wider
70k OOF) so the meta layer is not evaluated on rows it could have leaked into.

Baseline: linear=1, stacker=1, encoder=2 weighted blend, macro-F1 0.71726 on
this holdout (matches the validated LB blend, LB 0.71884). This script
verifies that baseline, trains a meta-selector, and reports whether it beats
the blend under a leakage-safe evaluation.

Outputs (scripts/meta/_out/):
  fold{k}_C{c}.csv        -- per-fold predictions, written immediately (crash-safe)
  meta_oof_C{c}.csv       -- assembled 9,969-row meta-OOF predictions
  summary.json            -- baseline/meta/oracle scores + threshold sweep + coefficients
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "_out"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUT_BASE = ROOT / "context/night/2026-07-05/holdout_base.npz"
OOF_DIR = ROOT / "artifacts/oof/oof_rebuild_2026_07_04"
TRAIN_JSONL = ROOT / "data/train.jsonl"

BLEND_WEIGHTS = {"linear": 1.0, "stacker": 1.0, "encoder": 2.0}
EXPLORE = {"read_file", "grep_search", "list_directory", "glob_pattern"}
N_FOLDS = 5
C_VALUES = [1.0, 0.1]
THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 42


def session_id(sample_id: str) -> str:
    return sample_id.rsplit("-step_", 1)[0]


def load_components():
    base = np.load(HOLDOUT_BASE, allow_pickle=True)
    ids = list(base["ids"])
    enc_probs = base["probs"].astype(float)
    y_true = list(base["y_true"])
    actions = list(base["actions"])  # alphabetical, shared column order for everything below

    lin_all = np.load(OOF_DIR / "linear_probs.npy")
    stk_all = np.load(OOF_DIR / "stacker_probs.npy")
    row_ids = json.load(open(OOF_DIR / "row_ids.json", encoding="utf-8"))
    classes = json.load(open(OOF_DIR / "classes.json", encoding="utf-8"))
    y_true_all = json.load(open(OOF_DIR / "y_true.json", encoding="utf-8"))

    id2idx = {rid: i for i, rid in enumerate(row_ids)}
    missing = [i for i in ids if i not in id2idx]
    if missing:
        raise RuntimeError(f"{len(missing)} holdout ids missing from OOF row_ids (e.g. {missing[:3]})")

    perm = [classes.index(a) for a in actions]  # reorder OOF columns -> alphabetical (matches holdout_base)
    idxs = [id2idx[i] for i in ids]
    lin = lin_all[idxs][:, perm]
    stk = stk_all[idxs][:, perm]

    oof_y_sub = [y_true_all[i] for i in idxs]
    mismatches = sum(1 for a, b in zip(oof_y_sub, y_true) if a != b)
    if mismatches:
        raise RuntimeError(f"y_true mismatch between OOF and holdout_base: {mismatches} rows")

    return ids, actions, lin, stk, enc_probs, np.array(y_true)


def join_session_meta(ids):
    wanted = set(ids)
    meta = {}
    with open(TRAIN_JSONL, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] in wanted:
                sm = d.get("session_meta", {})
                meta[d["id"]] = (float(sm.get("turn_index", 0)), float(len(d.get("history", []))))
                if len(meta) == len(wanted):
                    break
    turn_index = np.array([meta[i][0] for i in ids])
    n_history = np.array([meta[i][1] for i in ids])
    return turn_index, n_history


def entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def top2(p):
    order = np.argsort(-p, axis=1)
    top1_idx = order[:, 0]
    top2_idx = order[:, 1]
    rows = np.arange(len(p))
    return top1_idx, p[rows, top1_idx], p[rows, top2_idx]


def rank_of(p, target_idx):
    """0-indexed rank (0 = most confident) of the class at target_idx within each row."""
    target_val = p[np.arange(len(p)), target_idx]
    return (p > target_val[:, None]).sum(axis=1)


def build_features(lin, stk, enc, turn_index, n_history, n_actions):
    lin_top1_idx, lin_top1, lin_top2 = top2(lin)
    stk_top1_idx, stk_top1, stk_top2 = top2(stk)
    enc_top1_idx, enc_top1, enc_top2 = top2(enc)

    lin_margin, stk_margin, enc_margin = lin_top1 - lin_top2, stk_top1 - stk_top2, enc_top1 - enc_top2
    lin_ent, stk_ent, enc_ent = entropy(lin), entropy(stk), entropy(enc)

    blend = lin * BLEND_WEIGHTS["linear"] + stk * BLEND_WEIGHTS["stacker"] + enc * BLEND_WEIGHTS["encoder"]
    blend = blend / blend.sum(axis=1, keepdims=True)
    blend_top1_idx, blend_top1, blend_top2 = top2(blend)
    blend_margin = blend_top1 - blend_top2

    pairs_equal = (
        (lin_top1_idx == stk_top1_idx).astype(int)
        + (stk_top1_idx == enc_top1_idx).astype(int)
        + (lin_top1_idx == enc_top1_idx).astype(int)
    )
    agree_all = (pairs_equal == 3).astype(float)
    agree_two = (pairs_equal == 1).astype(float)
    agree_none = (pairs_equal == 0).astype(float)

    enc_rank_in_lin = rank_of(lin, enc_top1_idx) / (n_actions - 1)
    enc_rank_in_stk = rank_of(stk, enc_top1_idx) / (n_actions - 1)
    lin_rank_in_stk = rank_of(stk, lin_top1_idx) / (n_actions - 1)
    stk_rank_in_lin = rank_of(lin, stk_top1_idx) / (n_actions - 1)

    rows = np.arange(len(lin))
    lin_prob_at_blend = lin[rows, blend_top1_idx]
    stk_prob_at_blend = stk[rows, blend_top1_idx]
    enc_prob_at_blend = enc[rows, blend_top1_idx]

    feature_names = [
        "lin_top1", "lin_top2", "lin_margin", "lin_ent",
        "stk_top1", "stk_top2", "stk_margin", "stk_ent",
        "enc_top1", "enc_top2", "enc_margin", "enc_ent",
        "blend_top1", "blend_top2", "blend_margin",
        "agree_all", "agree_two_one", "agree_all_differ",
        "enc_rank_in_lin", "enc_rank_in_stk", "lin_rank_in_stk", "stk_rank_in_lin",
        "lin_prob_at_blend_top", "stk_prob_at_blend_top", "enc_prob_at_blend_top",
        "turn_index", "n_history",
    ]
    feats = np.column_stack([
        lin_top1, lin_top2, lin_margin, lin_ent,
        stk_top1, stk_top2, stk_margin, stk_ent,
        enc_top1, enc_top2, enc_margin, enc_ent,
        blend_top1, blend_top2, blend_margin,
        agree_all, agree_two, agree_none,
        enc_rank_in_lin, enc_rank_in_stk, lin_rank_in_stk, stk_rank_in_lin,
        lin_prob_at_blend, stk_prob_at_blend, enc_prob_at_blend,
        turn_index, n_history,
    ])
    return feats, feature_names, blend_top1_idx, lin_top1_idx, stk_top1_idx, enc_top1_idx


def run_cv(X, y_idx, groups, actions, c_value, blend_pred_idx, ids):
    n, n_actions = len(y_idx), len(actions)
    oof_proba = np.zeros((n, n_actions))
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, te) in enumerate(gkf.split(X, y_idx, groups)):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced")
        clf.fit(scaler.transform(X[tr]), y_idx[tr])
        proba = clf.predict_proba(scaler.transform(X[te]))
        full_proba = np.zeros((len(te), n_actions))
        full_proba[:, clf.classes_] = proba
        oof_proba[te] = full_proba

        fold_df = pd.DataFrame({
            "id": [ids[i] for i in te],
            "fold": fold,
            "y_true": [actions[j] for j in y_idx[te]],
            "meta_pred": [actions[j] for j in full_proba.argmax(axis=1)],
            "meta_conf": full_proba.max(axis=1),
            "blend_pred": [actions[j] for j in blend_pred_idx[te]],
        })
        fold_df.to_csv(OUT / f"fold{fold}_C{c_value}.csv", index=False)
        print(f"  C={c_value} fold{fold}: train={len(tr)} test={len(te)} -> saved")
    return oof_proba


def evaluate(y_true_idx, blend_pred_idx, meta_proba, actions):
    meta_pred_idx = meta_proba.argmax(axis=1)
    meta_conf = meta_proba.max(axis=1)
    actions_arr = np.array(actions)
    y_true_labels = actions_arr[y_true_idx]
    blend_labels = actions_arr[blend_pred_idx]

    baseline_f1 = f1_score(y_true_labels, blend_labels, labels=actions, average="macro")
    meta_only_f1 = f1_score(y_true_labels, actions_arr[meta_pred_idx], labels=actions, average="macro")

    sweep = []
    for th in THRESHOLDS:
        override = (meta_pred_idx != blend_pred_idx) & (meta_conf >= th)
        final_idx = np.where(override, meta_pred_idx, blend_pred_idx)
        final_labels = actions_arr[final_idx]
        f1 = f1_score(y_true_labels, final_labels, labels=actions, average="macro")
        explore_f1 = f1_score(y_true_labels, final_labels, labels=sorted(EXPLORE), average="macro", zero_division=0)
        blend_correct = blend_labels == y_true_labels
        final_correct = final_labels == y_true_labels
        fixed = int(np.sum(override & ~blend_correct & final_correct))
        broken = int(np.sum(override & blend_correct & ~final_correct))
        sweep.append({
            "threshold": th,
            "n_intervene": int(override.sum()),
            "macro_f1": f1,
            "delta_vs_baseline": f1 - baseline_f1,
            "explore_f1": explore_f1,
            "n_fixed": fixed,
            "n_broken": broken,
        })
    return baseline_f1, meta_only_f1, sweep


def component_oracle(y_true_idx, blend_pred_idx, lin_idx, stk_idx, enc_idx, actions):
    actions_arr = np.array(actions)
    mask_lin = lin_idx == y_true_idx
    mask_stk = stk_idx == y_true_idx
    mask_enc = enc_idx == y_true_idx
    any_correct = mask_lin | mask_stk | mask_enc
    oracle_idx = blend_pred_idx.copy()
    oracle_idx[mask_lin] = lin_idx[mask_lin]
    oracle_idx[mask_stk] = stk_idx[mask_stk]
    oracle_idx[mask_enc] = enc_idx[mask_enc]
    oracle_f1 = f1_score(actions_arr[y_true_idx], actions_arr[oracle_idx], labels=actions, average="macro")
    return float(any_correct.mean()), oracle_f1


def coefficient_report(X, feature_names, y_idx, c_value):
    """Full-data fit (not used for scoring) purely to inspect which features the
    meta-model leans on -- in particular, whether it over-trusts raw encoder
    confidence (enc_top1/enc_margin) vs. cross-component agreement/rank signals."""
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=c_value, max_iter=5000, class_weight="balanced")
    clf.fit(scaler.transform(X), y_idx)
    norms = np.linalg.norm(clf.coef_, axis=0)  # L2 norm across classes per feature
    order = np.argsort(-norms)
    ranked = [{"feature": feature_names[i], "coef_l2_norm": float(norms[i])} for i in order]
    enc_related = {"enc_top1", "enc_top2", "enc_margin", "enc_ent"}
    enc_share = sum(n["coef_l2_norm"] for n in ranked if n["feature"] in enc_related) / norms.sum()
    return ranked, float(enc_share)


def main():
    print("Loading components...")
    ids, actions, lin, stk, enc, y_true_labels = load_components()
    n_actions = len(actions)
    action2idx = {a: i for i, a in enumerate(actions)}
    y_idx = np.array([action2idx[a] for a in y_true_labels])
    groups = np.array([session_id(i) for i in ids])
    print(f"rows={len(ids)} groups={len(set(groups))} actions={n_actions}")

    print("Joining session_meta from train.jsonl...")
    turn_index, n_history = join_session_meta(ids)

    X, feature_names, blend_pred_idx, lin_idx, stk_idx, enc_idx = build_features(
        lin, stk, enc, turn_index, n_history, n_actions
    )

    actions_arr = np.array(actions)
    baseline_f1_check = f1_score(y_true_labels, actions_arr[blend_pred_idx], labels=actions, average="macro")
    print(f"Baseline [1,1,2] blend macro-F1 (sanity check, expect ~0.71726): {baseline_f1_check:.5f}")

    lin_f1 = f1_score(y_true_labels, actions_arr[lin_idx], labels=actions, average="macro")
    stk_f1 = f1_score(y_true_labels, actions_arr[stk_idx], labels=actions, average="macro")
    enc_f1 = f1_score(y_true_labels, actions_arr[enc_idx], labels=actions, average="macro")
    print(f"Component solo scores: linear={lin_f1:.5f} stacker={stk_f1:.5f} encoder={enc_f1:.5f}")

    oracle_hit_rate, oracle_f1 = component_oracle(y_idx, blend_pred_idx, lin_idx, stk_idx, enc_idx, actions)
    print(f"Component oracle: any-correct row rate={oracle_hit_rate:.5f} oracle macro-F1={oracle_f1:.5f}")

    summary = {
        "n_rows": len(ids),
        "n_groups": len(set(groups)),
        "n_folds": N_FOLDS,
        "blend_weights": BLEND_WEIGHTS,
        "solo_scores": {"linear": lin_f1, "stacker": stk_f1, "encoder": enc_f1},
        "baseline_blend_macro_f1": baseline_f1_check,
        "component_oracle": {"any_correct_row_rate": oracle_hit_rate, "oracle_macro_f1": oracle_f1},
        "feature_names": feature_names,
        "by_C": {},
    }

    for c_value in C_VALUES:
        print(f"\n=== Meta-selector, C={c_value} ===")
        meta_proba = run_cv(X, y_idx, groups, actions, c_value, blend_pred_idx, ids)
        baseline_f1, meta_only_f1, sweep = evaluate(y_idx, blend_pred_idx, meta_proba, actions)

        oof_df = pd.DataFrame({
            "id": ids,
            "y_true": y_true_labels,
            "blend_pred": actions_arr[blend_pred_idx],
            "meta_pred": actions_arr[meta_proba.argmax(axis=1)],
            "meta_conf": meta_proba.max(axis=1),
            "lin_pred": actions_arr[lin_idx],
            "stk_pred": actions_arr[stk_idx],
            "enc_pred": actions_arr[enc_idx],
        })
        oof_df.to_csv(OUT / f"meta_oof_C{c_value}.csv", index=False)

        ranked_coefs, enc_share = coefficient_report(X, feature_names, y_idx, c_value)

        print(f"baseline blend macro-F1: {baseline_f1:.5f}")
        print(f"meta-only (always use meta_pred) macro-F1: {meta_only_f1:.5f}")
        for row in sweep:
            print(f"  th={row['threshold']}: n_intervene={row['n_intervene']:4d} "
                  f"macro_f1={row['macro_f1']:.5f} delta={row['delta_vs_baseline']:+.5f} "
                  f"fixed={row['n_fixed']} broken={row['n_broken']} explore_f1={row['explore_f1']:.5f}")
        print(f"encoder-raw-confidence coefficient share (L2 norm): {enc_share:.4f}")
        print("top-5 features by coef L2 norm:", [r["feature"] for r in ranked_coefs[:5]])

        summary["by_C"][str(c_value)] = {
            "baseline_blend_macro_f1": baseline_f1,
            "meta_only_macro_f1": meta_only_f1,
            "threshold_sweep": sweep,
            "encoder_raw_confidence_coef_share": enc_share,
            "coef_ranking": ranked_coefs,
        }

    with open(OUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()
