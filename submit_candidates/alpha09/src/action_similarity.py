from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from .constants import ACTIONS


ACTION_DESCRIPTIONS: Dict[str, str] = {
    "read_file": "open or inspect the content of a specific file",
    "grep_search": "search text symbol function error or references across files",
    "list_directory": "list files or folders in a directory",
    "glob_pattern": "find files by wildcard extension or filename pattern",
    "edit_file": "modify an existing file",
    "write_file": "create or write a new file",
    "apply_patch": "apply a patch or diff to change files",
    "run_bash": "run a shell command",
    "run_tests": "execute tests such as pytest or unittest",
    "lint_or_typecheck": "run linting or type checking such as ruff mypy eslint",
    "ask_user": "ask the user for clarification or a choice",
    "plan_task": "produce a plan architecture or step by step approach",
    "web_search": "search the web for current or external information",
    "respond_only": "answer directly without using tools",
}


class ActionDescriptionSimilarity:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.prototype_matrix = None

    def fit(self) -> "ActionDescriptionSimilarity":
        descriptions = [ACTION_DESCRIPTIONS[action] for action in ACTIONS]
        self.prototype_matrix = self.vectorizer.fit_transform(descriptions)
        return self

    def transform(self, texts: Sequence[str]) -> np.ndarray:
        if self.prototype_matrix is None:
            self.fit()
        text_matrix = self.vectorizer.transform([str(x) for x in texts])
        scores = text_matrix @ self.prototype_matrix.T
        return np.asarray(scores.toarray(), dtype=np.float32)

    def to_dict(self) -> Dict[str, object]:
        return {"actions": ACTIONS, "descriptions": ACTION_DESCRIPTIONS}
