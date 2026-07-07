from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .aar_features import action_sequence, keyword_flags, prompt_text
from .constants import ACTIONS


class TransitionPrior:
    def __init__(self, alpha: float = 0.35, actions: Sequence[str] = ACTIONS) -> None:
        self.alpha = float(alpha)
        self.actions = list(actions)
        self.action_to_idx = {name: idx for idx, name in enumerate(self.actions)}
        self.global_counts = np.full(len(self.actions), self.alpha, dtype=np.float64)
        self.tables: Dict[str, Dict[str, np.ndarray]] = {}

    def _keys(self, record: Dict[str, Any]) -> Dict[str, str]:
        seq = action_sequence(record)
        flags = [name for name, value in keyword_flags(prompt_text(record)).items() if value]
        first_flag = flags[0] if flags else "none"
        keys = {
            "last_1": seq[-1] if seq else "none",
            "last_2": ">".join(seq[-2:]) if len(seq) >= 2 else "none",
            "last_3": ">".join(seq[-3:]) if len(seq) >= 3 else "none",
            "last_1_keyword": f"{seq[-1] if seq else 'none'}|{first_flag}",
        }
        return keys

    def fit(self, records: Sequence[Dict[str, Any]], labels: Sequence[str]) -> "TransitionPrior":
        tables: Dict[str, Dict[str, np.ndarray]] = {
            "last_1": defaultdict(lambda: np.full(len(self.actions), self.alpha, dtype=np.float64)),
            "last_2": defaultdict(lambda: np.full(len(self.actions), self.alpha, dtype=np.float64)),
            "last_3": defaultdict(lambda: np.full(len(self.actions), self.alpha, dtype=np.float64)),
            "last_1_keyword": defaultdict(lambda: np.full(len(self.actions), self.alpha, dtype=np.float64)),
        }
        self.global_counts = np.full(len(self.actions), self.alpha, dtype=np.float64)
        for record, label in zip(records, labels):
            idx = self.action_to_idx.get(str(label))
            if idx is None:
                continue
            self.global_counts[idx] += 1.0
            for name, key in self._keys(record).items():
                tables[name][key][idx] += 1.0
        self.tables = {name: dict(values) for name, values in tables.items()}
        return self

    def _normalize(self, counts: np.ndarray) -> np.ndarray:
        return counts / np.maximum(counts.sum(), 1e-12)

    def predict_proba(self, records: Sequence[Dict[str, Any]]) -> np.ndarray:
        out = np.zeros((len(records), len(self.actions)), dtype=np.float32)
        global_proba = self._normalize(self.global_counts)
        for row_idx, record in enumerate(records):
            keys = self._keys(record)
            proba = None
            for name in ("last_3", "last_2", "last_1_keyword", "last_1"):
                counts = self.tables.get(name, {}).get(keys.get(name, ""))
                if counts is not None:
                    proba = self._normalize(counts)
                    break
            if proba is None:
                proba = global_proba
            out[row_idx] = proba
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "alpha": self.alpha,
            "actions": self.actions,
            "global_counts": self.global_counts.astype(float).tolist(),
            "tables": {
                name: {key: counts.astype(float).tolist() for key, counts in values.items()}
                for name, values in self.tables.items()
            },
        }

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "TransitionPrior":
        prior = cls(alpha=float(obj.get("alpha", 0.35)), actions=obj.get("actions", ACTIONS))
        prior.global_counts = np.asarray(obj.get("global_counts", [prior.alpha] * len(prior.actions)), dtype=np.float64)
        prior.tables = {
            name: {key: np.asarray(counts, dtype=np.float64) for key, counts in values.items()}
            for name, values in obj.get("tables", {}).items()
        }
        return prior
