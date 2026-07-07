from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .constants import ACTIONS

# Conservative keyword rules. They are used only when the model confidence is low
# or the prompt contains a highly direct instruction.
RULES: Dict[str, List[str]] = {
    "run_tests": [
        r"\bpytest\b", r"\bunittest\b", r"\bunit tests?\b", r"test.*run", r"run.*test",
        r"테스트(를|도)?\s*(돌|실행|해|검증|확인)", r"(검증|확인).*테스트",
    ],
    "lint_or_typecheck": [
        r"\blint\b", r"\bmypy\b", r"\bruff\b", r"\beslint\b", r"type\s*check",
        r"타입\s*(체크|검사|확인)", r"린트",
    ],
    "web_search": [
        r"최신", r"웹\s*검색", r"인터넷", r"뉴스", r"사이트.*찾", r"search the web",
        r"look it up", r"검색해서.*알", r"찾아봐.*웹",
    ],
    "grep_search": [
        r"\bgrep\b", r"\brg\b", r"어디.*(정의|사용|참조)", r"찾아(봐|줘)?",
        r"검색(해|해줘)?", r"definition", r"references?", r"where.*defined",
    ],
    "list_directory": [
        r"디렉토리", r"폴더", r"파일\s*목록", r"구조.*보여", r"\bls\b", r"\btree\b",
        r"list.*director", r"what.*inside", r"목록.*보여",
    ],
    "glob_pattern": [r"glob", r"\*\.\w+", r"패턴.*파일", r"확장자.*파일", r"all .*files"],
    "read_file": [
        r"파일.*(읽|열|확인|보여)", r"(읽|열)어(봐|줘)", r"show.*file", r"open.*file",
        r"cat\s+", r"내용.*보여",
    ],
    "apply_patch": [r"patch", r"diff", r"패치", r"apply_patch"],
    "edit_file": [
        r"수정(해|해줘|해봐)", r"고쳐(줘|봐)?", r"바꿔(줘|봐)?", r"edit", r"fix",
        r"update.*file", r"change.*file",
    ],
    "write_file": [r"새 파일", r"파일.*(만들|생성|작성)", r"write.*file", r"create.*file"],
    "run_bash": [
        r"터미널", r"명령어", r"bash", r"shell", r"실행(해|해줘)?", r"npm\s+",
        r"pip\s+", r"python\s+", r"command",
    ],
    "ask_user": [
        r"선택.*해", r"어느.*원해", r"확인.*필요", r"clarify", r"which .* do you want",
        r"물어봐", r"사용자.*확인",
    ],
    "plan_task": [r"계획", r"설계", r"로드맵", r"아키텍처", r"단계별", r"plan", r"approach"],
}

HIGH_PRECISION_ACTIONS = {
    "run_tests",
    "lint_or_typecheck",
    "web_search",
    "list_directory",
    "glob_pattern",
    "apply_patch",
}


def load_extra_rules(path: str | Path | None) -> Dict[str, List[str]]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    out: Dict[str, List[str]] = {}
    for action, patterns in obj.items():
        if action in ACTIONS and isinstance(patterns, list):
            out[action] = [str(x) for x in patterns]
    return out


def match_rules(text: str, extra_rules: Dict[str, List[str]] | None = None) -> List[str]:
    candidates: List[str] = []
    merged = dict(RULES)
    for action, patterns in (extra_rules or {}).items():
        merged.setdefault(action, []).extend(patterns)
    for action, patterns in merged.items():
        for pattern in patterns:
            try:
                if re.search(pattern, text, flags=re.IGNORECASE):
                    candidates.append(action)
                    break
            except re.error:
                continue
    return candidates


def override_prediction(text: str, pred: str, confidence: float, extra_rules: Dict[str, List[str]] | None = None) -> str:
    matches = match_rules(text, extra_rules)
    if not matches:
        return pred

    # Direct, high-precision instructions can override unless the model is already very confident.
    for action in matches:
        if action in HIGH_PRECISION_ACTIONS and confidence < 0.82:
            return action

    # For general rules, only override weak predictions.
    if confidence < 0.55:
        return matches[0]
    return pred
