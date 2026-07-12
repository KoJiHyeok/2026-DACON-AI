# AAR artifact schema

The inference contract is defined by `submit/aar_infer.py` and is intentionally
not changed by this rebuild.

This schema was written by opening the real, surviving artifacts directly:
`submit/model/stacker/aar_config.json` (1.4 KB, on disk) and
`submit/model/stacker/aar_models.joblib` (47 MB, on disk, loaded with
`joblib.load` and inspected field by field). Both files exist and are not
lost. Earlier drafts of this document claimed the originals were missing and
invented a different 4-view all-sklearn recipe; that claim was false and has
been retracted (see REPRODUCTION.md's "Prior attempt" section).

## `model/aar_config.json`

Verified real keys (loaded from the actual file):

* `enabled: true`
* `model_file`: `"aar_models.joblib"`
* `final_valid_macro_f1`: `0.7098004360266162` (the recovery target)
* `fallback_macro_f1`: `0.6073847881131159`
* `blend_macro_f1` / `bias_calibrated_macro_f1`: diagnostic-only, not consumed
  by `aar_infer.py`
* `use_bias: false`, `class_bias`: 14 zeros
* `use_stacker: true`
* `stacker_components`: `["prompt_context_sgd", "prompt_sgd", "action_sgd", "transition_prior"]`
  (this exact order determines the column order of the stacker's input
  matrix — `aar_infer.predict_aar` does `np.hstack([component_probas[name] for name in names])`)
* `components`: 4 objects, each `{name, kind, view, weight}`:
  * `prompt_context_sgd` — `kind="text"`, `view="prompt_context"`, `weight=0.6003`
  * `prompt_sgd` — `kind="text"`, `view="prompt"`, `weight=0.2001`
  * `action_sgd` — `kind="text"`, `view="action"`, `weight=0.1196`
  * `transition_prior` — `kind="transition"`, `view="transition"`, `weight=0.08`
  * The `weight` fields are only used by `aar_infer.py` in the
    `weighted_average` fallback path (when `use_stacker` is false); since
    `use_stacker` is true here, the live path is the stacker, and these
    weights are carried through only for compatibility/diagnostics.

## `model/aar_models.joblib`

Verified real structure (`joblib.load` returns a `dict`):

* `actions`: the 14 canonical action names, in `aar_infer.ACTIONS` order.
* `components`: a `dict` with exactly the 3 keys
  `{prompt_context_sgd, prompt_sgd, action_sgd}` — **not** 4, and **not**
  named `sgd_<view>`. Each value is an `sklearn.pipeline.Pipeline`:
  * `prompt_context_sgd` and `prompt_sgd`: `Pipeline([("features", FeatureUnion([
    ("word", TfidfVectorizer(ngram_range=(1,2), analyzer="word", sublinear_tf=True,
    token_pattern=r"(?u)\b[^\s]+\b", min_df=2, max_features=...)),
    ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3,5), sublinear_tf=True,
    min_df=2, max_features=...))
    ])), ("clf", SGDClassifier(loss="log_loss", penalty="l2", class_weight="balanced",
    tol=1e-4, max_iter=25, random_state=42, alpha=...))])`.
    `prompt_context_sgd`: word max_features=120000, char max_features=80000, alpha=2.5e-05.
    `prompt_sgd`: word max_features=220000, char max_features=180000, alpha=2e-05.
  * `action_sgd`: `Pipeline([("features", TfidfVectorizer(ngram_range=(1,3), analyzer="word",
    sublinear_tf=True, min_df=2, max_features=60000, token_pattern=r"(?u)\b[^\s]+\b")),
    ("clf", SGDClassifier(loss="log_loss", penalty="l2", class_weight="balanced",
    tol=1e-4, max_iter=25, random_state=42, alpha=5e-05))])` — single TF-IDF,
    **no** `FeatureUnion` (confirmed: this pipeline has no `word`/`char` sub-steps).
* `transition`: a `dict` with keys `{actions, global, groups, weights, global_weight}`.
  This is **not** an sklearn estimator — it's a rule-based lookup table
  consumed by `aar_infer.aar_transition_predict_proba`:
  * `actions`: the 14 canonical action names (same order as `actions` above).
  * `global`: a 14-float vector, the unconditional empirical class prior
    (normalized label frequency over the full training set). Verified:
    sums to 1.0.
  * `global_weight`: `0.3` (float).
  * `weights`: a `dict` of 7 group names to blend weights:
    `{"last_action_rule": 0.7, "last2": 0.55, "last_action": 0.45,
    "prompt_rule": 0.35, "history_len": 0.15, "language_pref": 0.12,
    "ci_dirty": 0.08}`.
  * `groups`: a `dict` keyed by the same 7 group names. Each value is a
    `dict` mapping a group-specific key string (produced by
    `aar_infer.aar_transition_keys`, e.g. `last_action="apply_patch"`,
    `last2="run_tests>apply_patch"`, `ci_dirty="passed|True"`) to a 14-float
    vector — the empirical `P(action | group_key=value)` conditional
    distribution. Verified: each vector sums to 1.0 (plain normalized
    counts, no visible smoothing constant beyond "keys with zero support are
    simply absent from the dict" — `aar_transition_predict_proba` skips
    missing keys and renormalizes by `weight_sum`).
  * Blend formula (from `aar_transition_predict_proba`, unchanged):
    `total = global * global_weight`; for each group with a matching key,
    `total += group_vector * group_weight`; final = `total / sum_of_weights_used`.
* `stacker`: `sklearn.linear_model.LogisticRegression` with
  `C=1.0, class_weight="balanced", max_iter=500, solver="lbfgs", random_state=42`.
  Verified: `classes_` has the 14 actions (alphabetical sklearn order, not
  the canonical `ACTIONS` order — `aar_infer.predict_proba_aligned` handles
  the remapping), `coef_.shape == (14, 56)`, `n_features_in_ == 56`
  (= 4 components x 14 classes, in `stacker_components` order).

Models must expose `classes_` containing the 14 action names (or a subset),
or the consumer falls back to canonical order — verified via
`aar_infer._model_classes`.

## Rebuild fidelity

`scripts/aar_rebuild/train_aar.py` reproduces this structure exactly: the
same 3 named components with the same view functions (already defined,
unchanged, in `submit/aar_infer.py`: `aar_prompt_context_text`,
`record_to_prompt_text`, `aar_action_text`), the same per-component
TF-IDF/SGD hyperparameters read directly off the live pipelines above, a
transition table built as empirical frequency tables over the same 7 group
keys with the same fixed blend weights recovered from the real
`aar_config.json`/artifact, and a `LogisticRegression(C=1.0,
class_weight="balanced", max_iter=500, solver="lbfgs", random_state=42)`
stacker over the 4 components hstacked in `stacker_components` order.

What is *not* recoverable from the surviving files: the original SGD weight
initialization/shuffle order (SGD is order-sensitive across epochs even at a
fixed seed if row order differs), the exact TF-IDF fit vocabulary (depends on
exact preprocessing/tokenization history if it differed at training time,
though the vectorizer parameters themselves are verified above), and whether
the transition frequency tables were built on the full 70k rows or a
train-only split at each fold (the rebuild uses fold-train-only tables for
OOF estimation and full-data tables for the final artifact, which is the
standard OOF-safe convention and matches how the 3 SGD components are
trained).
