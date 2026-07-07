from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List

from .aar_features import action_sequence, clean_text, prompt_text


FILE_PATH_RE = re.compile(r"(?i)(?:^|\s)(?:[\w.-]+/)+[\w.-]+\.[a-z0-9]+")
FILE_EXT_RE = re.compile(r"(?i)\.[a-z0-9]{1,8}\b")
DIR_PATH_RE = re.compile(r"(?i)(?:^|\s)(?:[\w.-]+/)+[\w.-]+/?(?:\s|$)")
GLOB_RE = re.compile(r"(?i)(?:\*\.[a-z0-9]+|\*\*/|[\w.-]*\*[\w.-]*)")


def _history_text(record: Dict[str, Any]) -> str:
    history = record.get("history")
    if not isinstance(history, list):
        return ""
    parts: List[str] = []
    for item in history[-8:]:
        if isinstance(item, dict):
            parts.append(clean_text(item, 1000))
    return " ".join(parts).lower()


def _full_text(record: Dict[str, Any]) -> str:
    return f"{prompt_text(record)} {_history_text(record)}".lower()


def _contains(text: str, tokens: Iterable[str]) -> float:
    return float(any(token in text for token in tokens))


def _last_action_flags(record: Dict[str, Any], actions: Iterable[str]) -> Dict[str, float]:
    seq = action_sequence(record)
    last = seq[-1] if seq else "none"
    return {f"last_action_{action}": float(last == action) for action in actions}


def _common_features(record: Dict[str, Any]) -> Dict[str, float]:
    prompt = prompt_text(record)
    text = prompt.lower()
    hist = _history_text(record)
    return {
        "prompt_len": float(len(prompt)),
        "prompt_token_count": float(len(prompt.split())),
        "history_contains_file_path": float(bool(FILE_PATH_RE.search(hist))),
        "history_contains_search_result": float(any(x in hist for x in ("found", "match", "matches", "검색", "결과", "src/", ".py:"))),
        "history_contains_opened_file": float(any(x in hist for x in ("read_file", "open", "opened", "파일", "src/"))),
        "history_contains_previous_patch": float(any(x in hist for x in ("apply_patch", "patch", "diff", "패치"))),
        "has_specific_file_path": float(bool(FILE_PATH_RE.search(text))),
        "has_file_extension": float(bool(FILE_EXT_RE.search(text))),
        "has_directory_path": float(bool(DIR_PATH_RE.search(text))),
        "has_glob_pattern": float(bool(GLOB_RE.search(text))),
        "has_wildcard": float("*" in text or "wildcard" in text),
    }


def file_navigation_features(record: Dict[str, Any]) -> Dict[str, float]:
    text = _full_text(record)
    out = _common_features(record)
    out.update({
        "has_symbol_word": _contains(text, ("symbol", "function", "class", "method", "함수", "클래스", "심볼")),
        "has_reference_word": _contains(text, ("reference", "references", "usage", "where used", "참조", "사용처")),
        "has_definition_word": _contains(text, ("definition", "defined", "where is", "정의", "선언")),
        "has_search_word": _contains(text, ("search", "grep", "rg ", "검색", "찾아", "찾아줘")),
        "has_find_word": _contains(text, ("find", "locate", "찾", "어디")),
        "has_open_word": _contains(text, ("open", "cat ", "열어", "열어줘")),
        "has_read_word": _contains(text, ("read", "inspect", "읽", "확인")),
        "has_show_content_word": _contains(text, ("show content", "내용", "보여줘", "출력")),
        "has_list_word": _contains(text, ("list", "folder", "directory", "목록", "구조", "폴더", "디렉터리", "디렉토리")),
        "has_tree_word": _contains(text, ("tree", "트리", "구조")),
        "has_ls_word": _contains(text, ("ls ", " ls", "`ls`")),
    })
    out.update(_last_action_flags(record, ("read_file", "grep_search", "list_directory", "glob_pattern")))
    return out


def edit_apply_write_features(record: Dict[str, Any]) -> Dict[str, float]:
    text = _full_text(record)
    out = _common_features(record)
    out.update({
        "has_new_file_word": _contains(text, ("new file", "새 파일", "신규 파일")),
        "has_create_word": _contains(text, ("create", "make", "generate", "scaffold", "생성", "만들")),
        "has_write_word": _contains(text, ("write", "작성", "저장")),
        "has_edit_word": _contains(text, ("edit", "수정", "편집")),
        "has_modify_word": _contains(text, ("modify", "change", "update", "replace", "변경", "바꿔")),
        "has_fix_word": _contains(text, ("fix", "bug", "고쳐", "수정")),
        "has_patch_word": _contains(text, ("patch", "apply_patch", "패치")),
        "has_diff_word": _contains(text, ("diff", "---", "+++", "@@")),
        "has_apply_patch_word": _contains(text, ("apply_patch", "apply patch", "패치 적용")),
        "has_existing_file_path": float(bool(FILE_PATH_RE.search(text)) or _contains(text, ("existing file", "기존 파일"))),
    })
    out.update(_last_action_flags(record, ("read_file", "grep_search", "edit_file")))
    return out


def run_test_lint_features(record: Dict[str, Any]) -> Dict[str, float]:
    text = _full_text(record)
    out = _common_features(record)
    out.update({
        "has_pytest": _contains(text, ("pytest",)),
        "has_unittest": _contains(text, ("unittest",)),
        "has_test_word": _contains(text, ("test", "tests", "테스트", "검증")),
        "has_run_test_phrase": _contains(text, ("run test", "run tests", "test run", "테스트 돌", "테스트 실행")),
        "has_ruff": _contains(text, ("ruff",)),
        "has_mypy": _contains(text, ("mypy",)),
        "has_eslint": _contains(text, ("eslint",)),
        "has_typecheck": _contains(text, ("typecheck", "type check", "tsc", "타입")),
        "has_lint_word": _contains(text, ("lint", "린트")),
        "has_npm": _contains(text, ("npm ", "npm run")),
        "has_pip": _contains(text, ("pip ", "pip install")),
        "has_python_command": _contains(text, ("python ", "python.exe", "파이썬")),
        "has_bash_word": _contains(text, ("bash", "shell", "쉘")),
        "has_terminal_word": _contains(text, ("terminal", "터미널")),
        "has_shell_command": _contains(text, ("command", "명령", "실행")),
    })
    return out


def response_planning_features(record: Dict[str, Any]) -> Dict[str, float]:
    text = _full_text(record)
    history = record.get("history")
    history_len = len(history) if isinstance(history, list) else 0
    out = _common_features(record)
    out.update({
        "has_question_mark": float("?" in prompt_text(record)),
        "has_choice_word": _contains(text, ("choose", "choice", "which", "선택", "어느")),
        "has_clarify_word": _contains(text, ("clarify", "confirm", "확인", "명확")),
        "has_need_more_info_word": _contains(text, ("need more info", "정보 부족", "물어봐", "확인 필요")),
        "has_plan_word": _contains(text, ("plan", "steps", "approach", "계획", "단계")),
        "has_design_word": _contains(text, ("design", "설계")),
        "has_architecture_word": _contains(text, ("architecture", "아키텍처", "구조")),
        "has_roadmap_word": _contains(text, ("roadmap", "로드맵")),
        "has_explain_word": _contains(text, ("explain", "describe", "설명", "알려줘")),
        "has_answer_only_signal": _contains(text, ("just answer", "respond only", "도구 없이", "답만")),
        "requires_tool_signal": _contains(text, ("file", "run", "edit", "search", "파일", "실행", "수정", "검색")),
        "history_empty_or_short": float(history_len <= 2),
    })
    return out
