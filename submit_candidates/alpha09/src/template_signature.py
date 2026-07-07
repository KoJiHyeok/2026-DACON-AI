from __future__ import annotations

import re
from typing import Any, Dict, List

from .aar_features import action_sequence, clean_text, prompt_text, record_to_action_text
from .serialize import record_to_text


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
QUOTE_RE = re.compile(r"(['\"])(?:(?=(\\?))\2.)*?\1")
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s'\"<>|]+")
PATH_RE = re.compile(
    r"(?<!\w)(?:[./~\w-]+[/\\])+[\w .@()+-]+\.[A-Za-z0-9]{1,8}(?!\w)"
)
DIR_RE = re.compile(r"(?<!\w)(?:[./~\w-]+[/\\])+[\w .@()+-]+[/\\]?(?!\w)")
EXT_RE = re.compile(r"(?<!\w)\*\.[A-Za-z0-9]{1,8}\b|\.[A-Za-z0-9]{1,8}\b")
NUM_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)?(?!\w)")
SYMBOL_RE = re.compile(r"(?<![\w<])[A-Za-z_][A-Za-z0-9_]{2,}(?:\.[A-Za-z_][A-Za-z0-9_]*)?(?![\w>])")

COMMAND_WORDS = {
    "python",
    "python3",
    "pytest",
    "unittest",
    "pip",
    "npm",
    "pnpm",
    "yarn",
    "node",
    "ruff",
    "mypy",
    "eslint",
    "git",
    "bash",
    "sh",
    "cmd",
    "powershell",
}

KEEP_WORDS = {
    "read",
    "open",
    "show",
    "list",
    "ls",
    "tree",
    "find",
    "search",
    "grep",
    "reference",
    "definition",
    "edit",
    "modify",
    "fix",
    "patch",
    "diff",
    "write",
    "create",
    "run",
    "test",
    "lint",
    "typecheck",
    "plan",
    "explain",
    "ask",
    "web",
    "파일",
    "열어",
    "읽어",
    "검색",
    "찾아",
    "목록",
    "구조",
    "수정",
    "고쳐",
    "작성",
    "생성",
    "실행",
    "테스트",
    "설명",
    "계획",
}


def _normalize_commands(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        return token.lower() if token.lower() in COMMAND_WORDS else token

    return SYMBOL_RE.sub(repl, text)


def template_signature(text: Any) -> str:
    value = clean_text(text, 4000)
    value = URL_RE.sub(" <URL> ", value)
    value = QUOTE_RE.sub(" <QUOTE> ", value)
    value = WINDOWS_PATH_RE.sub(" <PATH> ", value)
    value = PATH_RE.sub(" <PATH> ", value)
    value = DIR_RE.sub(" <DIR> ", value)
    value = EXT_RE.sub(" <EXT> ", value)
    value = NUM_RE.sub(" <NUM> ", value)
    value = _normalize_commands(value)
    value = re.sub(r"\s+", " ", value).strip().lower()
    for tag in ("url", "quote", "path", "dir", "ext", "num", "cmd", "symbol"):
        value = value.replace(f"<{tag}>", f"<{tag.upper()}>")
    return value


def intent_template_signature(text: Any) -> str:
    value = template_signature(text)

    def repl(match: re.Match[str]) -> str:
        token = match.group(0)
        lower = token.lower()
        if lower in COMMAND_WORDS or lower in KEEP_WORDS or token.startswith("<"):
            return token
        if any(ord(ch) > 127 for ch in token):
            return token
        return "<SYMBOL>"

    value = SYMBOL_RE.sub(repl, value)
    value = re.sub(r"(?:<SYMBOL>\s*){3,}", "<SYMBOL> ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def keyword_tokens(text: Any) -> List[str]:
    sig = intent_template_signature(text)
    tokens = []
    for token in re.findall(r"<[A-Z]+>|[\w가-힣]+", sig):
        lower = token.lower()
        if lower.startswith("<") or lower in COMMAND_WORDS or lower in KEEP_WORDS or len(lower) >= 2:
            tokens.append(lower)
    return tokens[:40]


def record_template_features(record: Dict[str, Any]) -> Dict[str, str]:
    prompt_sig = template_signature(prompt_text(record))
    prompt_intent = intent_template_signature(prompt_text(record))
    full_sig = template_signature(record_to_text(record))
    seq = action_sequence(record)
    last = seq[-1] if seq else "none"
    last2 = ">".join(seq[-2:]) if len(seq) >= 2 else "none"
    token_sig = " ".join(keyword_tokens(prompt_text(record))[:20])
    return {
        "template_signature": prompt_sig,
        "intent_template_signature": prompt_intent,
        "full_template_signature": full_sig,
        "last_action_template": f"{last} {prompt_sig}",
        "last_2_actions_template": f"{last2} {prompt_sig}",
        "keyword_template": f"{token_sig} {prompt_sig}",
        "action_sequence_template": f"{record_to_action_text(record)} {prompt_sig}",
    }
