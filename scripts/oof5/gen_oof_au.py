# -*- coding: utf-8 -*-
"""Generate 5-fold session-group OOF for the AU (char_wb C1) specialist.

Recipe (scripts/league4/common.py::train_or_load_au_probs, the exact
protocol behind the champion's alpha-0.9 soft-routing surface used in
apply_soft_au / BASELINE_SOFT_AU=0.73877):
  - au_route.serialize(sample) text view
  - TfidfVectorizer(analyzer=char_wb, ngram_range=(3,5), min_df=1,
    max_features=120_000, sublinear_tf=True, strip_accents=unicode)
  - LinearSVC(C=1.0, class_weight=balanced, max_iter=5000, random_state=42)
  - decision_function -> softmax -> align_probs to the 14-action order

The AU specialist is scoped to sess_au rows only (au_route.is_au). For each
of the 5 global folds, we train on sess_au rows NOT in that fold and predict
sess_au rows IN that fold -- consistent with "for fold k, train only on rows
whose fold_map fold != k, predict rows with fold == k" restricted to the
sess_au subset the component actually covers. Non-AU rows are NOT part of
this component's OOF (the specialist has no opinion on them); this matches
scripts/au/probe_au_linear.py's routing_eval pattern, which only ever scores
the AU subset for this component and lets non-AU rows fall back to the
league blend elsewhere.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "submit"))
from scripts.oof5 import common as C
import au_route  # noqa: E402

OUT_DIR = C.ROOT / "artifacts" / "experiments" / "oof_au"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t_start = time.time()
    fold_map = C.load_fold_map()
    print("[load] train.jsonl + labels")
    samples, ids, y, groups = C.load_train()
    folds = C.fold_assignment_for_ids(ids, fold_map)
    n_folds = int(folds.max()) + 1

    au_mask = np.asarray([au_route.is_au(str(i)) for i in ids], dtype=bool)
    au_idx_all = np.where(au_mask)[0]
    print(f"[au] total rows={len(ids)} au rows={len(au_idx_all)} ({au_mask.mean():.4%})")

    texts_cache: dict[int, str] = {}

    def get_text(i: int) -> str:
        if i not in texts_cache:
            texts_cache[i] = au_route.serialize(samples[i])
        return texts_cache[i]

    fold_reports = []
    all_ids_out = []
    all_probs_out = []
    all_y_out = []

    for fold in range(n_folds):
        t0 = time.time()
        va_mask_fold = folds == fold
        au_va_idx = au_idx_all[va_mask_fold[au_idx_all]]
        au_tr_idx = au_idx_all[~va_mask_fold[au_idx_all]]
        print(f"[fold {fold}] au_train={len(au_tr_idx)} au_valid={len(au_va_idx)}")

        train_texts = [get_text(int(i)) for i in au_tr_idx]
        valid_texts = [get_text(int(i)) for i in au_va_idx]

        vec = TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=1,
            max_features=120_000, sublinear_tf=True, strip_accents="unicode",
        )
        x_train = vec.fit_transform(train_texts)
        x_valid = vec.transform(valid_texts)

        clf = LinearSVC(C=args.c, class_weight="balanced", max_iter=args.max_iter, random_state=args.seed)
        clf.fit(x_train, y[au_tr_idx])

        raw_scores = clf.decision_function(x_valid)
        probs = C.align_probs(C.softmax(raw_scores), [str(c) for c in clf.classes_], C.ACTIONS)

        fold_ids = ids[au_va_idx]
        fold_y = y[au_va_idx]
        macro = C.macro_f1_probs(probs, fold_y, C.ACTIONS)
        elapsed = time.time() - t0

        C.save_fold_npz(
            OUT_DIR / f"oof_au_fold{fold}.npz",
            ids=fold_ids, probs=probs, y_true=fold_y, fold=fold,
        )
        fold_reports.append({
            "fold": fold, "au_train_rows": int(len(au_tr_idx)), "au_valid_rows": int(len(au_va_idx)),
            "n_features": int(x_train.shape[1]), "classes_seen": [str(c) for c in clf.classes_],
            "macro_f1_au_subset": macro, "elapsed_sec": round(elapsed, 3),
        })
        all_ids_out.append(fold_ids)
        all_probs_out.append(probs)
        all_y_out.append(fold_y)
        print(f"[fold {fold}] au_macro_f1={macro:.6f} elapsed={elapsed:.1f}s")

        del vec, clf, x_train, x_valid, raw_scores, probs
        gc.collect()

    pooled_ids = np.concatenate(all_ids_out)
    pooled_probs = np.vstack(all_probs_out)
    pooled_y = np.concatenate(all_y_out)

    # Coverage check restricted to the AU subset (this component's actual scope).
    au_ids_expected = set(str(x) for x in ids[au_mask])
    au_ids_got = set(str(x) for x in pooled_ids)
    if au_ids_got != au_ids_expected:
        raise AssertionError(
            f"AU OOF coverage mismatch: missing={len(au_ids_expected - au_ids_got)} extra={len(au_ids_got - au_ids_expected)}"
        )
    if len(pooled_ids) != len(set(str(x) for x in pooled_ids)):
        raise AssertionError("duplicate ids in concatenated AU OOF")

    pooled_macro_au = C.macro_f1_probs(pooled_probs, pooled_y, C.ACTIONS)

    # Save the full-70k AU boolean mask (indexed against fold_map.csv id
    # order via ids, not position) so phase 2 can build the AU feature block
    # for non-AU rows (zeros + indicator) without recomputing au_route.is_au.
    au_mask_path = OUT_DIR / "au_row_mask.npz"
    np.savez_compressed(
        au_mask_path,
        ids=np.asarray(ids, dtype=object),
        is_au=au_mask,
    )

    total_elapsed = time.time() - t_start
    run_meta = {
        "component": "au_charwb_C1",
        "recipe": {
            "serializer": "au_route.serialize (submit/au_route.py)",
            "vectorizer": "TfidfVectorizer(char_wb, ngram_range=(3,5), min_df=1, max_features=120000, sublinear_tf=True, strip_accents=unicode)",
            "clf": f"LinearSVC(C={args.c}, class_weight=balanced, max_iter={args.max_iter}, random_state={args.seed})",
            "proba_conversion": "decision_function -> softmax -> align_probs",
            "provenance": "scripts/league4/common.py::train_or_load_au_probs (alpha-0.9 soft-routing specialist, BASELINE_SOFT_AU=0.73877)",
            "scope": "sess_au rows only (au_route.is_au) -- this component has no opinion on non-AU rows",
        },
        "npz_coverage": {
            "note": "oof_au_fold{0..4}.npz together cover ONLY the 5,025 sess_au ids, NOT all 70,000 fold_map ids. "
                     "This is a scope difference from the linear/AAR/e5/mBERT OOF sets, which cover all 70,000. "
                     "Phase 2 must build the AU feature block for non-AU rows itself (e.g. zeros + an is_au "
                     "indicator column, matching apply_soft_au's alpha-0.9 gating which only ever touches "
                     "au_mask rows and leaves everything else untouched) rather than expecting a dense 70,000-row "
                     "AU probability matrix.",
            "au_row_mask_path": str(au_mask_path),
            "au_row_mask_schema": "ids (70000,), is_au (70000,) bool -- same id order as fold_map.csv iteration, "
                                    "provided so phase 2 doesn't need to recompute au_route.is_au(id).",
        },
        "fold_map_sha256": C.FOLD_MAP_SHA256,
        "n_folds": n_folds,
        "total_rows": int(len(ids)),
        "au_rows": int(len(au_idx_all)),
        "au_share": float(au_mask.mean()),
        "fold_reports": fold_reports,
        "pooled_au_subset_macro_f1": pooled_macro_au,
        "global_macro_f1_note": "not applicable -- AU specialist covers only sess_au rows; see pooled_au_subset_macro_f1",
        "total_elapsed_sec": round(total_elapsed, 3),
    }
    C.write_json(OUT_DIR / "run_oof_au.json", run_meta)
    C.write_sha256sums(
        OUT_DIR,
        [f"oof_au_fold{f}.npz" for f in range(n_folds)] + ["au_row_mask.npz", "run_oof_au.json"],
    )
    print(f"[done] pooled_au_subset_macro_f1={pooled_macro_au:.6f} total_elapsed={total_elapsed:.1f}s")


if __name__ == "__main__":
    main()
