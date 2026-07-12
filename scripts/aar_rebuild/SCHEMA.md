# AAR artifact schema

The inference contract is defined by `submit/aar_infer.py` and is intentionally
not changed by this rebuild.

`model/aar_config.json` contains:

* `enabled: true`
* `model_file` (normally `aar_models.joblib`)
* `components`: objects with `name`, `kind`, `view`, and optional `weight`
* `use_stacker`, `stacker_components`, and optional `use_bias`/`class_bias`

`model/aar_models.joblib` is a dictionary with `components` (a mapping from
component name to an object implementing `predict_proba`), optional
`transition`, and `stacker` (an object implementing `predict_proba`).  Models
must expose `classes_` containing the 14 action names, or the consumer falls
back to the canonical order.  The rebuild stores sklearn Pipelines containing
`TfidfVectorizer` and `SGDClassifier(loss="log_loss")` for four views, and a
`LogisticRegression` stacker over the concatenated 4 x 14 probability columns.

The available checkout did not contain the original 31.5 MB AAR binary or its
JSON config; this schema is therefore consumer-derived and the generated
artifact is a compatible replacement, not a byte-for-byte recovery.
