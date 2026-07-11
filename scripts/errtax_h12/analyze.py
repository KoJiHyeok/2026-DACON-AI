"""Deterministic descriptive error taxonomy for the 9,969-row H12 holdout.

No fitting, optimization, embeddings, clustering, or learned inference occurs here.
The only counterfactual changes are transparent oracle corrections used for score
impact accounting.  Object arrays are loaded only after their enclosing artifact
matches a pinned SHA-256 digest.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


DEFAULT_ROOT = Path(r"C:\dev\2026-AI-DACON")
EXPECTED = {
    "champion_holdout_preds.csv": "ce1bee8757de84777015090d4e616dd2c3f21cf539a38cecb9d91aa9b67eb6ce",
    "champion_holdout_probs.npz": "77476558ca9794f7d6c823757d2300c227a7d088b08c266ddc3ca220f30f91ef",
    "train.jsonl": "a60ed84b75285caee237142ce97622fb55bb59c36be6b36ddee523992f83df19",
}
EXPECTED_E5 = {
    "fold_map.csv": "56074c16c400fbccc389e15c01c05adc4db810533516340f15e9826dd44fe295",
    "oof_fold0.npz": "8ad89e6329b4d992cd79cb8695151ba2a48bfb092925b600db9aadae6acaff94",
    "oof_fold1.npz": "7c0ac6411c43709d416f1e28ac33a16867a1ee8851e0d338fe2f8dbf7856e220",
    "oof_fold2.npz": "97470e9d2cdce188e406413c75fd41fabd950e34909f9473636c31a35fe2c282",
    "oof_fold3.npz": "c34bd72709ac6a81e3c63090fd82021509b27a246caf4111f911c149122df0f6",
    "oof_fold4.npz": "324ce7ef3de2b12f14fba975a42cf88e117dfae1ab4598b85e9b2585b5b487f8",
}
EXPECTED_MBERT = {
    "oof_mbert_fold0.npz": "e228346dfd204fbcfe99143484ca850de4d0d60a6405ef6f5b406975451a9563",
    "oof_mbert_fold1.npz": "fb0e068e418552ce1cd11c59f20385c827c57fd9b5d89e883798874bfb8fe13a",
    "oof_mbert_fold2.npz": "098f3e5002e582c521972b91e454dd83283dbb1f6ede470d7d7174b352828989",
    "oof_mbert_fold3.npz": "964ba24df48347d7acd6419c583f29a3c785bea50561b24e5ebf969184d3cc3b",
    "oof_mbert_fold4.npz": "4b064e844e2b75c27592b4082e213ba31efa29273b3715b1762a32726c52971a",
}
STEP_RE = re.compile(r"^(.*)-step_(\d+)$")
MARGIN_BINS = [(0.05, "[0,.05)"), (0.10, "[.05,.10)"), (0.20, "[.10,.20)"), (0.40, "[.20,.40)"), (math.inf, "[.40,1]")]
PATTERNS = {
    "question": re.compile(r"\?"),
    "path_like": re.compile(r"(?:^|\s)[\w.-]+(?:/[\w./-]+|\\[\w.\\-]+)"),
    "test_term": re.compile(r"\b(?:test|tests|pytest|unittest|spec|ci)\b|테스트", re.I),
    "lint_term": re.compile(r"\b(?:lint|mypy|typecheck|type-check|eslint|ruff)\b", re.I),
    "shell_term": re.compile(r"\b(?:run|execute|command|terminal|bash|shell|npm|pip)\b|실행", re.I),
    "search_term": re.compile(r"\b(?:grep|search|find|where|locate)\b|검색|찾", re.I),
    "list_term": re.compile(r"\b(?:list|directory|folder|files|tree|contents)\b|목록|폴더|디렉", re.I),
    "plan_term": re.compile(r"\b(?:plan|steps|sequence|approach)\b|계획|순서", re.I),
    "write_term": re.compile(r"\b(?:create|add|implement|write|edit|change|fix|refactor)\b|추가|수정|고쳐", re.I),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require_hash(path: Path, expected: str) -> str:
    actual = sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f"SHA256 mismatch for {path}: {actual} != {expected}")
    return actual


def macro_f1(y: Sequence[str], pred: Sequence[str], actions: Sequence[str], weights: np.ndarray | None = None) -> float:
    y = np.asarray(y); pred = np.asarray(pred)
    scores = []
    w = np.ones(len(y), dtype=float) if weights is None else np.asarray(weights, dtype=float)
    for label in actions:
        tp = w[(y == label) & (pred == label)].sum()
        fp = w[(y != label) & (pred == label)].sum()
        fn = w[(y == label) & (pred != label)].sum()
        scores.append(0.0 if 2 * tp + fp + fn == 0 else float(2 * tp / (2 * tp + fp + fn)))
    return float(np.mean(scores))


def session_id(row_id: str) -> str:
    m = STEP_RE.fullmatch(row_id)
    if not m:
        raise ValueError(f"invalid id schema: {row_id}")
    return m.group(1)


def step_of(row_id: str) -> int:
    m = STEP_RE.fullmatch(row_id)
    if not m:
        raise ValueError(f"invalid id schema: {row_id}")
    return int(m.group(2))


def load_trusted_npz(path: Path, expected_hash: str) -> dict[str, np.ndarray]:
    require_hash(path, expected_hash)
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    for key, value in out.items():
        if value.dtype == object:
            flat = value.ravel().tolist()
            if not all(isinstance(x, str) for x in flat):
                raise TypeError(f"untrusted object payload in {path}:{key}")
    return out


def parse_fold(value: Any, source: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{source}: boolean is not a valid fold")
    if isinstance(value, (int, np.integer)):
        fold = int(value)
    elif isinstance(value, str) and re.fullmatch(r"[0-4]", value.strip()):
        fold = int(value.strip())
    else:
        raise ValueError(f"{source}: invalid fold value {value!r}")
    if fold not in range(5):
        raise ValueError(f"{source}: fold out of range: {fold}")
    return fold


def read_sha256_manifest(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+?)\s*", raw)
        if not match:
            raise ValueError(f"{path}: malformed SHA256SUMS line {line_no}")
        name = match.group(2)
        if name in entries:
            raise ValueError(f"{path}: duplicate manifest entry {name}")
        entries[name] = match.group(1).lower()
    return entries


def validate_prob_bundle(bundle: dict[str, np.ndarray], source: str, fold: int | None = None) -> None:
    needed = {"ids", "probs", "y_true", "actions"}
    if not needed.issubset(bundle):
        raise ValueError(f"{source}: missing {sorted(needed - set(bundle))}")
    ids = np.asarray(bundle["ids"]); labels = np.asarray(bundle["y_true"]); action_values = np.asarray(bundle["actions"])
    if ids.ndim != 1 or labels.ndim != 1 or action_values.ndim != 1:
        raise ValueError(f"{source}: ids, y_true, and actions must be one-dimensional")
    if not all(isinstance(x, (str, np.str_)) for x in ids.tolist() + labels.tolist() + action_values.tolist()):
        raise ValueError(f"{source}: ids, labels, and actions must be strings")
    n = len(ids); probs = np.asarray(bundle["probs"], dtype=float)
    actions = [str(x) for x in action_values]
    if len(set(map(str, ids))) != n or len(labels) != n:
        raise ValueError(f"{source}: duplicate ids or row-length mismatch")
    if probs.shape != (n, len(actions)) or len(set(actions)) != len(actions):
        raise ValueError(f"{source}: probability/action shape mismatch")
    if not set(map(str, labels)).issubset(actions):
        raise ValueError(f"{source}: y_true contains an unknown action")
    if not np.isfinite(probs).all() or (probs < 0).any() or not np.allclose(probs.sum(1), 1, atol=2e-5):
        raise ValueError(f"{source}: invalid probabilities")
    if fold is not None:
        tags = np.asarray(bundle.get("fold"))
        if tags.shape != (n,) or any(parse_fold(value, source) != fold for value in tags.tolist()):
            raise ValueError(f"{source}: fold tag mismatch")


def load_champion(root: Path) -> dict[str, Any]:
    base = root / "artifacts/experiments/errtax_h12"
    csv_path = base / "champion_holdout_preds.csv"
    npz_path = base / "champion_holdout_probs.npz"
    require_hash(csv_path, EXPECTED[csv_path.name])
    z = load_trusted_npz(npz_path, EXPECTED[npz_path.name]); validate_prob_bundle(z, str(npz_path))
    with csv_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    required = ["id", "y_true", "pred", "pred2", "p_top1", "p_top2", "is_au", "correct"]
    if list(rows[0]) != required or len(rows) != 9969:
        raise ValueError("champion CSV schema/row count mismatch")
    ids = np.asarray([r["id"] for r in rows]); y = np.asarray([r["y_true"] for r in rows]); pred = np.asarray([r["pred"] for r in rows])
    if not np.array_equal(ids, z["ids"].astype(str)) or not np.array_equal(y, z["y_true"].astype(str)):
        raise ValueError("champion CSV/NPZ id or label order mismatch")
    actions = [str(x) for x in z["actions"]]; probs = np.asarray(z["probs"], dtype=float)
    order = np.argsort(-probs, axis=1, kind="stable")
    arg1 = np.asarray(actions)[order[:, 0]]; arg2 = np.asarray(actions)[order[:, 1]]
    if not np.array_equal(pred, arg1) or not np.array_equal(np.asarray([r["pred2"] for r in rows]), arg2):
        raise ValueError("CSV predictions disagree with NPZ ranks")
    if sum(r["correct"] == "0" for r in rows) != 2451 or int((y != pred).sum()) != 2451:
        raise ValueError("expected exactly 2,451 errors")
    return {"rows": rows, "ids": ids, "y": y, "pred": pred, "pred2": arg2, "probs": probs, "actions": actions,
            "margin": np.asarray([float(r["p_top1"]) - float(r["p_top2"]) for r in rows]),
            "au": np.asarray([r["is_au"] == "1" for r in rows])}


def load_train(root: Path, holdout_ids: set[str]) -> tuple[dict[str, dict[str, Any]], Counter[str], str]:
    path = root / "data/train.jsonl"; digest = require_hash(path, EXPECTED[path.name])
    selected: dict[str, dict[str, Any]] = {}; lengths: Counter[str] = Counter(); seen: set[str] = set()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line); rid = row.get("id")
            if not isinstance(rid, str) or rid in seen:
                raise ValueError(f"train id invalid/duplicate at line {line_no}")
            seen.add(rid); lengths[session_id(rid)] += 1
            if rid in holdout_ids: selected[rid] = row
    if len(seen) != 70000 or set(selected) != holdout_ids:
        raise ValueError(f"train coverage mismatch: rows={len(seen)}, holdout={len(selected)}")
    return selected, lengths, digest


def confusion_damage(data: dict[str, Any]) -> list[dict[str, Any]]:
    y, pred, actions = data["y"], data["pred"], data["actions"]; baseline = macro_f1(y, pred, actions)
    out = []
    for true in actions:
        for guessed in actions:
            if true == guessed: continue
            mask = (y == true) & (pred == guessed); n = int(mask.sum())
            if not n: continue
            fixed = pred.copy(); fixed[mask] = y[mask]
            out.append({"true": true, "pred": guessed, "errors": n, "damage_macro_f1": macro_f1(y, fixed, actions) - baseline,
                        "error_share": n / int((y != pred).sum()), "pred2_true_rate": float(np.mean(data["pred2"][mask] == y[mask])),
                        "margin_median": float(np.median(data["margin"][mask]))})
    return sorted(out, key=lambda r: (-r["damage_macro_f1"], -r["errors"], r["true"], r["pred"]))


def margin_summary(data: dict[str, Any]) -> dict[str, Any]:
    y, pred, pred2, margin = data["y"], data["pred"], data["pred2"], data["margin"]
    wrong = y != pred
    def q(mask: np.ndarray) -> dict[str, float]:
        vals = margin[mask]
        return {str(k): float(v) for k, v in zip([0, .1, .25, .5, .75, .9, 1], np.quantile(vals, [0, .1, .25, .5, .75, .9, 1]))}
    bins = []
    for upper, label in MARGIN_BINS:
        lower = 0 if not bins else MARGIN_BINS[len(bins)-1][0]
        mask = wrong & (margin >= lower) & (margin < upper if math.isfinite(upper) else margin <= 1)
        bins.append({"bin": label, "rows": int(mask.sum()), "error_share": float(mask.sum()/wrong.sum()),
                     "pred2_true_rate": float(np.mean(pred2[mask] == y[mask])) if mask.any() else None})
    slices = {}
    for name, mask in {"all_errors": wrong, "near_miss_lt_0.10": wrong & (margin < .10), "confident_wrong_ge_0.40": wrong & (margin >= .40),
                       "au_errors": wrong & data["au"], "non_au_errors": wrong & ~data["au"]}.items():
        slices[name] = {"rows": int(mask.sum()), "pred2_true_rate": float(np.mean(pred2[mask] == y[mask])) if mask.any() else None}
    return {"quantiles_correct": q(~wrong), "quantiles_wrong": q(wrong), "wrong_bins": bins, "pred2_slices": slices}


def bucket(value: int, cuts: Sequence[tuple[int, str]]) -> str:
    for upper, label in cuts:
        if value <= upper: return label
    return cuts[-1][1]


def slice_stats(data: dict[str, Any], train: dict[str, dict[str, Any]], lengths: Counter[str]) -> dict[str, Any]:
    rows = []
    for i, rid in enumerate(data["ids"]):
        sample = train[str(rid)]; meta = sample.get("session_meta") or {}; step = step_of(str(rid)); turn = int(meta.get("turn_index"))
        rows.append({"step": step, "turn": turn, "length": lengths[session_id(str(rid))], "wrong": bool(data["y"][i] != data["pred"][i])})
    mismatch = sum(r["step"] != r["turn"] for r in rows)
    axes = {
        "step": [(str(r["step"]), r) for r in rows],
        "turn_index": [(str(r["turn"]), r) for r in rows],
        "observed_session_length": [(bucket(r["length"], [(3,"1-3"),(6,"4-6"),(9,"7-9"),(12,"10-12"),(999,"13+")]), r) for r in rows],
        "relative_turn": [(bucket(math.ceil(4*r["turn"]/r["length"]), [(1,"Q1"),(2,"Q2"),(3,"Q3"),(4,"Q4+")]), r) for r in rows],
    }
    out: dict[str, Any] = {"step_turn_mismatches": mismatch, "support_safeguard": "rates suppressed when rows < 50", "axes": {}}
    for axis, keyed in axes.items():
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for key, row in keyed: groups[key].append(row)
        vals = []
        for key in sorted(groups, key=lambda x: (len(x), x)):
            g = groups[key]; n=len(g); e=sum(r["wrong"] for r in g)
            vals.append({"bucket": key, "rows": n, "errors": e, "error_density": e/n if n >= 50 else None, "supported": n >= 50})
        out["axes"][axis] = vals
    return out


def load_fold_map(root: Path) -> tuple[dict[str, int], str]:
    base = root / "artifacts/experiments/oof_h12"
    path = base / "fold_map.csv"
    manifest_entries = read_sha256_manifest(base / "SHA256SUMS")
    expected = EXPECTED_E5[path.name]
    if manifest_entries.get(path.name) != expected:
        raise ValueError("e5 manifest does not pin the expected fold_map.csv digest")
    digest = require_hash(path, expected)
    mapping: dict[str, int] = {}
    session_folds: dict[str, int] = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["id", "fold"]:
            raise ValueError(f"fold_map.csv schema mismatch: {reader.fieldnames}")
        for line_no, row in enumerate(reader, 2):
            rid = row.get("id")
            if not isinstance(rid, str) or not rid or rid in mapping:
                raise ValueError(f"fold_map.csv invalid/duplicate id at line {line_no}")
            session = session_id(rid)
            fold = parse_fold(row.get("fold"), f"fold_map.csv line {line_no}")
            if session in session_folds and session_folds[session] != fold:
                raise ValueError(f"fold_map.csv splits session group {session}")
            mapping[rid] = fold
            session_folds[session] = fold
    if len(mapping) != 70000 or set(mapping.values()) != set(range(5)):
        raise ValueError(f"fold_map.csv coverage mismatch: {len(mapping)} rows")
    return mapping, digest


def load_oof_folds(root: Path, subdir: str, expected: dict[str, str], prefix: str,
                   fold_map: dict[str, int]) -> dict[str, tuple[np.ndarray, str, list[str]]]:
    base = root / subdir
    manifest_entries = read_sha256_manifest(base / "SHA256SUMS")
    out: dict[str, tuple[np.ndarray, str, list[str]]] = {}
    for fold in range(5):
        name = f"{prefix}{fold}.npz"; digest = expected[name]
        if manifest_entries.get(name) != digest: raise ValueError(f"manifest digest mismatch for {name}")
        z = load_trusted_npz(base/name, digest); validate_prob_bundle(z, str(base/name), fold)
        for rid, yy, pp in zip(z["ids"].astype(str), z["y_true"].astype(str), np.asarray(z["probs"], dtype=float)):
            if rid in out: raise ValueError(f"duplicate OOF id: {rid}")
            if rid not in fold_map or fold_map[rid] != fold:
                raise ValueError(f"{name}: canonical fold_map mismatch for {rid}")
            out[rid] = (pp, yy, [str(x) for x in z["actions"]])
    if len(out) != 70000 or set(out) != set(fold_map):
        raise ValueError(f"OOF canonical coverage mismatch: {len(out)} rows")
    return out


def component_analysis(root: Path, data: dict[str, Any]) -> dict[str, Any]:
    fold_map, fold_map_digest = load_fold_map(root)
    e5 = load_oof_folds(root, "artifacts/experiments/oof_h12", EXPECTED_E5, "oof_fold", fold_map)
    mb = load_oof_folds(root, "artifacts/experiments/oof_mbert_h6", EXPECTED_MBERT, "oof_mbert_fold", fold_map)
    legacy = root / "artifacts/oof/oof_rebuild_2026_07_04"
    ids = json.loads((legacy/"row_ids.json").read_text(encoding="utf-8")); ys = json.loads((legacy/"y_true.json").read_text(encoding="utf-8"))
    classes = json.loads((legacy/"classes.json").read_text(encoding="utf-8")); linear = np.load(legacy/"linear_probs.npy", allow_pickle=False)
    if (len(ids)!=70000 or len(set(ids))!=70000 or set(ids)!=set(fold_map) or linear.shape!=(70000,14) or len(ys)!=70000
            or not all(isinstance(x,str) for x in ids) or not all(isinstance(x,str) for x in ys)
            or not isinstance(classes,list) or len(classes)!=14 or len(set(classes))!=14 or not all(isinstance(x,str) for x in classes)
            or set(classes)!=set(data["actions"]) or not set(ys).issubset(classes)):
        raise ValueError("legacy linear shape/action/label/coverage schema mismatch")
    if not np.isfinite(linear).all() or (linear < 0).any() or not np.allclose(linear.sum(1), 1, atol=2e-5):
        raise ValueError("legacy linear probabilities must be finite, nonnegative, and row-normalized")
    lin = {rid:(linear[i],ys[i],classes) for i,rid in enumerate(ids)}
    sources={"e5_oof_h12":e5,"mbert_oof_h6":mb,"linear_oof_legacy":lin}; preds={}; rows=[]
    for name, source in sources.items():
        aligned=[]
        for rid, yy in zip(data["ids"],data["y"]):
            if str(rid) not in source: raise ValueError(f"{name}: holdout id missing: {rid}")
            pp, sy, acts=source[str(rid)]
            if sy != yy or set(acts)!=set(data["actions"]): raise ValueError(f"{name}: label/actions mismatch: {rid}")
            idx=[acts.index(a) for a in data["actions"]]; aligned.append(np.asarray(pp)[idx])
        mat=np.asarray(aligned); p=np.asarray(data["actions"])[mat.argmax(1)]; preds[name]=p
        champ_wrong=data["pred"]!=data["y"]; right=p==data["y"]
        rows.append({"component":name,"aligned_rows":len(p),"solo_macro_f1":macro_f1(data["y"],p,data["actions"]),
                     "argmax_accuracy":float(np.mean(right)),"right_when_champion_wrong":int((right&champ_wrong).sum()),
                     "champion_error_suppression_rate":float(np.mean(right[champ_wrong]))})
    oracle=np.zeros(len(data["y"]),dtype=bool)
    for p in preds.values(): oracle |= p==data["y"]
    wrong=data["pred"]!=data["y"]
    return {"alignment": {"holdout_rows":len(data["ids"]),"exact_id_intersection_each":len(data["ids"]),"exact_label_match":True,
             "canonical_fold_map":{"rows":len(fold_map),"sha256":fold_map_digest,"session_group_consistent":True,
               "e5_all_id_fold_tags_match":True,"mbert_all_id_fold_tags_match":True,
               "mbert_fold_map_provenance":"mBERT has no separate fold_map file; every mBERT NPZ ID and fold tag was verified against the pinned e5 fold_map.csv shared canonical assignment."},
             "limitation":"Valid same-ID OOF diagnostics only. These 5-fold OOF models/training fractions are not proven identical to the component artifacts used to form the 9,969-row champion surface; do not algebraically reconstruct or causally attribute the champion blend from them."},
            "components":rows,"component_oracle": {"champion_errors_any_component_right":int((wrong&oracle).sum()),
            "rate_among_champion_errors":float(np.mean(oracle[wrong])),"all_three_disagree_rows":int(np.sum((preds["e5_oof_h12"]!=preds["mbert_oof_h6"])&(preds["e5_oof_h12"]!=preds["linear_oof_legacy"])&(preds["mbert_oof_h6"]!=preds["linear_oof_legacy"])))},
            "legacy_observed_hashes": {n:sha256(legacy/n) for n in ["row_ids.json","y_true.json","classes.json","linear_probs.npy"]},
            "legacy_hash_status":"No SHA256SUMS was present in this root; schema, 70k unique-ID coverage, labels, and exact holdout joins were validated, but hashes are observed rather than coordination-pinned."}


def prompt_patterns(data: dict[str, Any], train: dict[str, dict[str, Any]], damage: list[dict[str, Any]]) -> dict[str, Any]:
    features=[]
    for rid in data["ids"]:
        s=train[str(rid)]; prompt=str(s.get("current_prompt") or ""); ws=(s.get("session_meta") or {}).get("workspace") or {}
        f={name:bool(rx.search(prompt)) for name,rx in PATTERNS.items()}
        f.update({f"ci={ws.get('last_ci_status','unknown')}":True, f"dirty={bool(ws.get('git_dirty'))}":True,
                  f"open_files={bucket(len(ws.get('open_files') or []),[(0,'0'),(2,'1-2'),(5,'3-5'),(999,'6+')])}":True})
        mix=ws.get("language_mix") or {}; top=max(mix,key=lambda k:float(mix[k] or 0)) if mix else "unknown"; f[f"top_lang={top}"]=True
        features.append(f)
    targets=[]
    for pair in damage[:8]:
        true_mask=data["y"]==pair["true"]
        mask=true_mask&(data["pred"]==pair["pred"])
        base=float(mask.sum()/true_mask.sum()); candidates=[]
        names=sorted(set().union(*(f.keys() for f in features)))
        for name in names:
            has=np.asarray([f.get(name,False) for f in features])&true_mask
            denom=int(has.sum()); hits=int((has&mask).sum())
            if denom>=50 and hits>=5:
                rate=hits/denom; candidates.append({"pattern":name,"denominator_rows":denom,"pair_errors":hits,"pair_rate":rate,"lift_vs_all_rows":rate/base})
        candidates.sort(key=lambda r:(-r["lift_vs_all_rows"],-r["pair_errors"],r["pattern"]))
        targets.append({"pair":f"{pair['true']}->{pair['pred']}","pair_errors":int(mask.sum()),"true_label_rows":int(true_mask.sum()),"base_confusion_rate":base,"top_patterns":candidates[:8]})
    return {"examples":"No verbatim prompts emitted; only aggregate regex/workspace categories.","minimum_support":{"denominator_rows":50,"pair_errors":5},"groups":targets}


def oracle_metrics(data: dict[str, Any], corrected: Sequence[int]) -> dict[str, Any]:
    """Recompute the five league diagnostics after explicit oracle corrections."""
    y, base = data["y"], data["pred"]
    chosen=np.asarray(corrected,dtype=int)
    cand=base.copy(); cand[chosen]=y[chosen]
    sess=np.asarray([session_id(str(x)) for x in data["ids"]]); groups=defaultdict(list)
    for i,s in enumerate(sess): groups[s].append(i)
    counts=Counter(sess); weights=np.asarray([1/counts[s] for s in sess],dtype=float)
    row_delta=macro_f1(y,cand,data["actions"])-macro_f1(y,base,data["actions"])
    session_delta=macro_f1(y,cand,data["actions"],weights)-macro_f1(y,base,data["actions"],weights)
    rng=np.random.default_rng(42); gl=list(groups.values()); mc=[]
    for _ in range(200):
        idx=np.asarray([g[rng.integers(len(g))] for g in gl])
        mc.append(macro_f1(y[idx],cand[idx],data["actions"])-macro_f1(y[idx],base[idx],data["actions"]))
    keys=list(groups); action_index={a:i for i,a in enumerate(data["actions"])}; k=len(data["actions"])
    base_cm=np.zeros((len(keys),k,k),dtype=np.int32); cand_cm=np.zeros_like(base_cm)
    for si,key in enumerate(keys):
        for i in groups[key]:
            ti=action_index[str(y[i])]; base_cm[si,ti,action_index[str(base[i])]]+=1; cand_cm[si,ti,action_index[str(cand[i])]]+=1
    multiplicity=rng.multinomial(len(keys),np.full(len(keys),1/len(keys)),size=2000)
    bcm=(multiplicity@base_cm.reshape(len(keys),-1)).reshape(-1,k,k)
    ccm=(multiplicity@cand_cm.reshape(len(keys),-1)).reshape(-1,k,k)
    def batch_macro(cm: np.ndarray) -> np.ndarray:
        tp=np.diagonal(cm,axis1=1,axis2=2); fp=cm.sum(1)-tp; fn=cm.sum(2)-tp; den=2*tp+fp+fn
        return np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den!=0).mean(1)
    boot=batch_macro(ccm)-batch_macro(bcm)
    perm=np.random.RandomState(42).permutation(len(y)); halves=np.array_split(perm,2)
    half_delta=[macro_f1(y[h],cand[h],data["actions"])-macro_f1(y[h],base[h],data["actions"]) for h in halves]
    return {"corrected_rows":int(len(chosen)),"row_macro_f1_delta":row_delta,"macro_f1_after_oracle_correction":macro_f1(y,cand,data["actions"]),"session_uniform_delta":session_delta,
      "one_row_per_session_mc200_delta_mean":float(np.mean(mc)),"one_row_per_session_mc200_delta_std":float(np.std(mc)),
      "paired_session_bootstrap2000_ci95":[float(x) for x in np.quantile(boot,[.025,.975])],"paired_session_bootstrap_p_positive":float(np.mean(boot>0)),
      "half_deltas":[float(x) for x in half_delta]}


def card_oracles(data: dict[str, Any], eligible: np.ndarray, correction_rate: float) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates=np.flatnonzero(eligible & (data["y"]!=data["pred"]))
    n=min(len(candidates),int(round(len(candidates)*correction_rate)))
    chosen=sorted(candidates,key=lambda i:(float(data["margin"][i]),str(data["ids"][i])))[:n]
    common={"eligible_error_rows":int(len(candidates)),
      "eligible_pred2_true_rate":float(np.mean(data["pred2"][candidates]==data["y"][candidates])) if len(candidates) else None,
      "eligible_margin_quantiles":{str(q):float(v) for q,v in zip([.25,.5,.75],np.quantile(data["margin"][candidates],[.25,.5,.75]))} if len(candidates) else {}}
    ceiling={**common,**oracle_metrics(data,candidates),
      "meaning":"Absolute eligible-set oracle ceiling: correct every eligible named error with y_true and exactly recompute Macro-F1. This is a true ceiling only for interventions limited to this eligible set, and remains oracle/unimplementable."}
    scenario={**common,"assumed_realistic_correction_rate":correction_rate,"judgmental_rate_assumption":True,
      "correction_rate_basis":"Judgmental hypothesis-card assumption, not estimated or validated from the discovery holdout.",
      "favorable_oracle_selection":"Among eligible named errors, select the assumed count by lowest champion margin and correct with y_true; this favors the hypothesis and is not an implementable selector.",
      **oracle_metrics(data,chosen),
      "meaning":"Rate-conditioned favorable oracle scenario, not a bound and not measured candidate performance."}
    return ceiling,scenario


def hypothesis_projections(data: dict[str, Any], train: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    y,p=data["y"],data["pred"]
    specs=[
      ("H1 clarification-versus-plan label contract", ((y=="ask_user")&(p=="plan_task"))|((y=="plan_task")&(p=="ask_user")), .30,
       "Bidirectional speech-act ambiguity; 161/191 (84.3%) place the truth at rank 2.",
       "Tighten simulator annotation/generation contract around blocking missing information versus an explicitly requested plan; do not use thresholding or pred2 promotion."),
      ("H2 test-versus-lint operation contract", ((y=="lint_or_typecheck")&(p=="run_tests"))|((y=="run_tests")&(p=="lint_or_typecheck")), .25,
       "The two directed errors total 110 and 84/110 (76.4%) place the truth at rank 2.",
       "Annotation-schema audit/relabeling only: make existing generated labels mutually exclusive for explicit test versus static-check intent. Never add an auxiliary feature, model input, prompt field, or other inference-time strengthening."),
      ("H3 explore target-shape annotation contract", np.isin(y,["read_file","grep_search","glob_pattern","list_directory"])&np.isin(p,["read_file","grep_search","glob_pattern","list_directory"])&(y!=p), .15,
       "Explore-family boundaries dominate damage; outgoing-to-list errors concentrate in open_files=0, while pred2 evidence is mixed, so the rate is conservatively capped at 15%.",
       "Annotation-schema audit/relabeling only: review existing labels for known-file, directory-enumeration, filename-pattern, or content-predicate intent. Do not expose target shape as a new field or auxiliary inference feature/input; lexical specialists and overrides remain excluded."),
      ("H4 clarification-versus-web-source contract", ((y=="ask_user")&(p=="web_search"))|((y=="web_search")&(p=="ask_user")), .20,
       "The high-damage ask_user->web_search direction has 48 errors; question/search categories lift the conditional confusion rate, but support is small.",
       "Audit generator labels for requests that require current external facts versus missing user-specific requirements; use only a high-precision deterministic contract audit."),
    ]
    cards=[]
    for name,mask,rate,mechanism,intervention in specs:
        ceiling,scenario=card_oracles(data,mask,rate)
        cards.append({"name":name,"mechanism":mechanism,"intervention":intervention,
          "validation_population":"The 9,969-row discovery holdout is forbidden for validation and may be used only for taxonomy/hypothesis generation. Validate on an untouched session-group shadow population or a newly generated annotation-contract population.",
          "required_validation":"On that independent population, compute row Macro-F1, session-uniform Macro-F1, one-row/session MC200, paired-session bootstrap2000, and deterministic halves. No model was fit in CX-003.",
          "absolute_eligible_set_oracle_ceiling":ceiling,
          "rate_conditioned_favorable_oracle_scenario":scenario,
          "no_submit":scenario["row_macro_f1_delta"]<.005,
          "decision_basis":"The conservative no-submit rule uses the rate-conditioned favorable oracle scenario, never the absolute eligible-set oracle ceiling. Independent validation and Claude authority are still required."})
    return cards


def au_summary(data: dict[str, Any]) -> list[dict[str, Any]]:
    out=[]
    for name,mask in [("all",np.ones(len(data["y"]),bool)),("AU",data["au"]),("non-AU",~data["au"])]:
        wrong=(data["y"]!=data["pred"])&mask
        out.append({"slice":name,"rows":int(mask.sum()),"errors":int(wrong.sum()),"error_density":float(wrong.sum()/mask.sum()),
                    "macro_f1":macro_f1(data["y"][mask],data["pred"][mask],data["actions"]),
                    "pred2_true_rate_errors":float(np.mean(data["pred2"][wrong]==data["y"][wrong])) if wrong.any() else None})
    return out


def markdown(report: dict[str, Any]) -> str:
    d=report["confusion_damage"][:12]; lines=["# CX-003 deterministic H12 error taxonomy","","Generated by `scripts/errtax_h12/analyze.py`; the handoff contains interpretation and hypothesis cards.","",
      f"- Rows/errors: **{report['summary']['rows']:,} / {report['summary']['errors']:,}**","- Counterfactual damage: correct every row in one directed `true -> pred` pair, leave all other predictions unchanged, then recompute 14-class Macro-F1 exactly.","",
      "## Highest-damage directed confusions","","| true → pred | errors | exact Macro-F1 damage | pred2=true | median margin |","|---|---:|---:|---:|---:|"]
    for r in d: lines.append(f"| {r['true']} → {r['pred']} | {r['errors']} | {r['damage_macro_f1']:.6f} | {r['pred2_true_rate']:.3f} | {r['margin_median']:.3f} |")
    lines += ["","## AU split","","| slice | rows | errors | density | Macro-F1 | pred2=true on errors |","|---|---:|---:|---:|---:|---:|"]
    for r in report["au_vs_non_au"]: lines.append(f"| {r['slice']} | {r['rows']} | {r['errors']} | {r['error_density']:.3f} | {r['macro_f1']:.6f} | {r['pred2_true_rate_errors']:.3f} |")
    lines += ["","## Alignment limitation","",report["components"]["alignment"]["limitation"],"","Full deterministic detail is in the JSON output."]
    return "\n".join(lines)+"\n"


def analyze(root: Path) -> dict[str, Any]:
    data=load_champion(root); train,lengths,train_hash=load_train(root,set(map(str,data["ids"])))
    damage=confusion_damage(data)
    return {"summary":{"rows":len(data["ids"]),"errors":int(np.sum(data["y"]!=data["pred"])),"macro_f1":macro_f1(data["y"],data["pred"],data["actions"]),"actions":data["actions"]},
      "input_hashes":{**EXPECTED,"train_jsonl_verified":train_hash,"e5_manifest_pinned":EXPECTED_E5,"mbert_manifest_pinned":EXPECTED_MBERT},
      "counterfactual_definition":"For each directed true->pred error pair, replace pred with y_true on all and only rows in that pair; keep every other prediction fixed; recompute 14-label Macro-F1 exactly. This is an oracle damage attribution, not an achievable intervention estimate.",
      "confusion_damage":damage,"margins":margin_summary(data),"step_session":slice_stats(data,train,lengths),
      "components":component_analysis(root,data),"prompt_workspace_patterns":prompt_patterns(data,train,damage),"au_vs_non_au":au_summary(data),
      "hypothesis_cards":hypothesis_projections(data,train)}


def self_check() -> None:
    y=np.array(["a","a","b","b"]); p=np.array(["a","b","b","a"])
    assert abs(macro_f1(y,p,["a","b"])-.5)<1e-12
    assert session_id("sess_x-step_04")=="sess_x" and step_of("sess_x-step_04")==4
    assert bucket(4,[(3,"short"),(9,"long")])=="long"
    print("self-check PASS")


def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--root",type=Path,default=DEFAULT_ROOT); ap.add_argument("--json-out",type=Path); ap.add_argument("--markdown-out",type=Path); ap.add_argument("--self-check",action="store_true")
    args=ap.parse_args()
    if args.self_check: self_check(); return
    report=analyze(args.root)
    payload=json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n"
    if args.json_out: args.json_out.parent.mkdir(parents=True,exist_ok=True); args.json_out.write_text(payload,encoding="utf-8")
    else: print(payload,end="")
    if args.markdown_out: args.markdown_out.parent.mkdir(parents=True,exist_ok=True); args.markdown_out.write_text(markdown(report),encoding="utf-8")


if __name__ == "__main__": main()
