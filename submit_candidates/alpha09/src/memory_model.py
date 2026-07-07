from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

from .aar_features import action_sequence
from .constants import ACTIONS
from .features import make_views
from .template_signature import record_template_features


N_ACTIONS = len(ACTIONS)
ACTION_TO_IDX = {name: idx for idx, name in enumerate(ACTIONS)}


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=np.float32)
    out = np.clip(out, 0.0, None)
    denom = np.maximum(out.sum(axis=1, keepdims=True), 1e-12)
    return out / denom


def _entropy(proba: np.ndarray) -> np.ndarray:
    safe = np.clip(proba, 1e-12, 1.0)
    return -(safe * np.log(safe)).sum(axis=1) / math.log(max(proba.shape[1], 2))


def _labels_to_prior(labels: Sequence[str]) -> np.ndarray:
    counts = np.ones(N_ACTIONS, dtype=np.float32) * 0.1
    for label in labels:
        if label in ACTION_TO_IDX:
            counts[ACTION_TO_IDX[label]] += 1.0
    return counts / counts.sum()


def _safe_texts(records: Sequence[Dict[str, Any]], view: str) -> List[str]:
    texts = []
    for record in records:
        if view == "template":
            texts.append(record_template_features(record)["template_signature"])
        elif view == "action_seq":
            texts.append(" ".join(action_sequence(record)) or "none")
        else:
            value = make_views(record).get(view, "")
            texts.append(str(value))
    return texts


@dataclass
class MemoryPrediction:
    proba: np.ndarray
    top1_similarity: np.ndarray
    top3_similarity_mean: np.ndarray
    label_entropy: np.ndarray


class TfidfNearestLabelMemory:
    def __init__(
        self,
        view: str,
        *,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 1,
        top_k: int = 3,
    ) -> None:
        self.view = view
        self.analyzer = analyzer
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.top_k = top_k
        self.vectorizer: TfidfVectorizer | None = None
        self.nn: NearestNeighbors | None = None
        self.y_idx: np.ndarray | None = None
        self.prior: np.ndarray | None = None
        self._empty = False

    def fit(self, records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "TfidfNearestLabelMemory":
        self.prior = _labels_to_prior(labels)
        self.y_idx = np.asarray([ACTION_TO_IDX.get(label, -1) for label in labels], dtype=np.int32)
        texts = _safe_texts(records, self.view)
        self.vectorizer = TfidfVectorizer(
            analyzer=self.analyzer,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            sublinear_tf=True,
            lowercase=True,
            max_features=200000,
        )
        try:
            x = self.vectorizer.fit_transform(texts)
        except ValueError:
            self._empty = True
            return self
        if x.shape[0] == 0 or x.shape[1] == 0:
            self._empty = True
            return self
        self.nn = NearestNeighbors(n_neighbors=min(self.top_k, x.shape[0]), metric="cosine")
        self.nn.fit(x)
        return self

    def predict(self, records: Sequence[Dict[str, Any]]) -> MemoryPrediction:
        n = len(records)
        if self._empty or self.vectorizer is None or self.nn is None or self.y_idx is None or self.prior is None:
            proba = np.tile(self.prior if self.prior is not None else np.ones(N_ACTIONS) / N_ACTIONS, (n, 1))
            return MemoryPrediction(proba.astype(np.float32), np.zeros(n), np.zeros(n), _entropy(proba))
        texts = _safe_texts(records, self.view)
        x = self.vectorizer.transform(texts)
        if sparse.issparse(x) and x.shape[1] == 0:
            proba = np.tile(self.prior, (n, 1))
            return MemoryPrediction(proba.astype(np.float32), np.zeros(n), np.zeros(n), _entropy(proba))
        distances, indices = self.nn.kneighbors(x, return_distance=True)
        sims = np.clip(1.0 - distances, 0.0, 1.0)
        proba = np.zeros((n, N_ACTIONS), dtype=np.float32)
        for row_idx in range(n):
            weights = sims[row_idx].astype(np.float32)
            if float(weights.sum()) <= 1e-8:
                proba[row_idx] = self.prior
                continue
            weights = weights / weights.sum()
            for neighbor_pos, weight in zip(indices[row_idx], weights):
                label_idx = int(self.y_idx[int(neighbor_pos)])
                if label_idx >= 0:
                    proba[row_idx, label_idx] += float(weight)
            if float(proba[row_idx].sum()) <= 1e-8:
                proba[row_idx] = self.prior
        proba = _normalize_rows(proba)
        top1 = sims[:, 0] if sims.shape[1] else np.zeros(n)
        top3 = sims[:, : min(3, sims.shape[1])].mean(axis=1) if sims.shape[1] else np.zeros(n)
        return MemoryPrediction(proba, top1.astype(np.float32), top3.astype(np.float32), _entropy(proba).astype(np.float32))


class TemplateLabelMemory:
    def __init__(self, alpha: float = 0.2) -> None:
        self.alpha = alpha
        self.tables: Dict[str, np.ndarray] = {}
        self.counts: Dict[str, int] = {}
        self.prior: np.ndarray | None = None

    def fit(self, records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "TemplateLabelMemory":
        self.prior = _labels_to_prior(labels)
        self.tables = {}
        self.counts = {}
        for record, label in zip(records, labels):
            idx = ACTION_TO_IDX.get(label)
            if idx is None:
                continue
            features = record_template_features(record)
            for key in (
                features["template_signature"],
                features["intent_template_signature"],
                features["last_action_template"],
                features["last_2_actions_template"],
                features["keyword_template"],
            ):
                if key not in self.tables:
                    self.tables[key] = np.ones(N_ACTIONS, dtype=np.float32) * self.alpha
                    self.counts[key] = 0
                self.tables[key][idx] += 1.0
                self.counts[key] += 1
        return self

    def predict(self, records: Sequence[Dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.prior is None:
            self.prior = np.ones(N_ACTIONS, dtype=np.float32) / N_ACTIONS
        proba = np.zeros((len(records), N_ACTIONS), dtype=np.float32)
        same_count = np.zeros(len(records), dtype=np.float32)
        for i, record in enumerate(records):
            features = record_template_features(record)
            matches = [
                features["template_signature"],
                features["intent_template_signature"],
                features["last_action_template"],
                features["last_2_actions_template"],
                features["keyword_template"],
            ]
            count = 0
            for key in matches:
                table = self.tables.get(key)
                if table is None:
                    continue
                proba[i] += table / np.maximum(table.sum(), 1e-12)
                count += 1
                same_count[i] = max(same_count[i], float(self.counts.get(key, 0)))
            if count:
                proba[i] /= float(count)
            else:
                proba[i] = self.prior
        proba = _normalize_rows(proba)
        return proba, same_count, _entropy(proba).astype(np.float32)


class MemoryModel:
    def __init__(self, top_k: int = 3, views: Sequence[str] | None = None) -> None:
        self.top_k = top_k
        wanted = set(views or ("prompt", "full", "char", "action_seq", "template"))
        candidates = {
            "prompt": ("nn_prompt", TfidfNearestLabelMemory("prompt", analyzer="word", ngram_range=(1, 2), top_k=top_k)),
            "full": ("nn_full", TfidfNearestLabelMemory("full", analyzer="word", ngram_range=(1, 2), top_k=top_k)),
            "char": ("nn_char", TfidfNearestLabelMemory("prompt", analyzer="char_wb", ngram_range=(3, 5), top_k=top_k)),
            "action_seq": ("nn_action_seq", TfidfNearestLabelMemory("action_seq", analyzer="word", ngram_range=(1, 3), top_k=top_k)),
        }
        self.models = [candidates[name] for name in candidates if name in wanted]
        self.use_template = "template" in wanted
        self.template_model = TemplateLabelMemory() if self.use_template else None

    def fit(self, records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "MemoryModel":
        for _, model in self.models:
            model.fit(records, labels)
        if self.template_model is not None:
            self.template_model.fit(records, labels)
        return self

    def predict_features(self, records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        probas: Dict[str, np.ndarray] = {}
        features: Dict[str, np.ndarray] = {}
        for name, model in self.models:
            pred = model.predict(records)
            probas[name] = pred.proba
            features[f"{name}_top1_similarity"] = pred.top1_similarity
            features[f"{name}_top3_similarity_mean"] = pred.top3_similarity_mean
            features[f"{name}_label_entropy"] = pred.label_entropy
        if self.template_model is not None:
            template_proba, same_count, template_entropy = self.template_model.predict(records)
            probas["nn_template"] = template_proba
            features["nn_same_template_count"] = same_count
            features["nn_same_template_label_entropy"] = template_entropy
        return {"probas": probas, "features": features}

    def predict_proba(self, records: Sequence[Dict[str, Any]]) -> np.ndarray:
        result = self.predict_features(records)
        arrays = list(result["probas"].values())
        return _normalize_rows(np.mean(arrays, axis=0))


def memory_oof(
    records: Sequence[Dict[str, Any]],
    labels: Sequence[str],
    folds: Sequence[int],
    *,
    top_k: int = 3,
    views: Sequence[str] | None = None,
) -> Dict[str, Any]:
    folds_arr = np.asarray(folds, dtype=np.int32)
    labels_arr = np.asarray(labels)
    n = len(records)
    probas: Dict[str, np.ndarray] = {}
    features: Dict[str, np.ndarray] = {}
    for fold in sorted(set(int(x) for x in folds_arr.tolist())):
        valid_idx = np.where(folds_arr == fold)[0]
        train_idx = np.where(folds_arr != fold)[0]
        model = MemoryModel(top_k=top_k, views=views).fit(
            [records[int(i)] for i in train_idx],
            labels_arr[train_idx].tolist(),
        )
        result = model.predict_features([records[int(i)] for i in valid_idx])
        for name, arr in result["probas"].items():
            if name not in probas:
                probas[name] = np.zeros((n, N_ACTIONS), dtype=np.float32)
            probas[name][valid_idx] = arr
        if result["probas"]:
            if "nn_memory_mean" not in probas:
                probas["nn_memory_mean"] = np.zeros((n, N_ACTIONS), dtype=np.float32)
            mean_arr = np.mean(list(result["probas"].values()), axis=0)
            probas["nn_memory_mean"][valid_idx] = _normalize_rows(mean_arr)
        for name, values in result["features"].items():
            if name not in features:
                features[name] = np.zeros(n, dtype=np.float32)
            features[name][valid_idx] = values
    for name, arr in probas.items():
        probas[name] = _normalize_rows(arr)
    return {"probas": probas, "features": features}
