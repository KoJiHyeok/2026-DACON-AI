"""Contract tests for duplicated encoder serialization helpers."""
from __future__ import annotations

import ast
import copy
import difflib
import re
import textwrap
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = ROOT / "submit" / "script.py"
TARGET_PATHS = (
    REFERENCE_PATH,
    ROOT / "colab" / "encoder_v2_s42_repro.py",
    ROOT / "colab" / "mdeberta_finetune.py",
    ROOT / "colab" / "encoder_e5_holdout85_maxhist.py",
)
FUNCTION_NAMES = ("_bucket", "serialize")
SERIALIZE_SOURCE_PATTERNS = {
    "query cap [:800]": "[:800]",
    "result_summary cap [:120]": "[:120]",
    "user content cap [:200]": "[:200]",
    "open_files cap [:5]": "open_files[:5]",
    "recent history first": "reversed(hist[-max_hist:])",
}


def _read_source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _find_function(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(_read_source(path), filename=str(path))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert matches, f"{path.relative_to(ROOT)} is missing function {name!r}"
    assert len(matches) == 1, f"{path.relative_to(ROOT)} has {len(matches)} functions named {name!r}"
    return matches[0]


def _strip_docstring(node: ast.FunctionDef) -> ast.FunctionDef:
    clean = copy.deepcopy(node)
    if (
        clean.body
        and isinstance(clean.body[0], ast.Expr)
        and isinstance(clean.body[0].value, ast.Constant)
        and isinstance(clean.body[0].value.value, str)
    ):
        clean.body.pop(0)
    return clean


def _function_dump(path: Path, name: str) -> str:
    return ast.dump(_strip_docstring(_find_function(path, name)), annotate_fields=True)


def _dumps_match(expected: str, actual: str) -> bool:
    return expected == actual


def _wrapped_dump_lines(dump: str) -> list[str]:
    return textwrap.wrap(dump, width=120, break_long_words=False, break_on_hyphens=False)


def _dump_diff(expected: str, actual: str) -> str:
    lines = difflib.unified_diff(
        _wrapped_dump_lines(expected),
        _wrapped_dump_lines(actual),
        fromfile="submit/script.py",
        tofile="candidate",
        lineterm="",
    )
    return "\n".join(list(lines)[:80])


def _function_source(path: Path, name: str) -> str:
    source_lines = _read_source(path).splitlines()
    node = _find_function(path, name)
    assert node.end_lineno is not None, f"{path.relative_to(ROOT)}:{name} is missing end_lineno"
    return "\n".join(source_lines[node.lineno - 1 : node.end_lineno])


def _normalized_source(source: str) -> str:
    return re.sub(r"\s+", "", source)


def _mutated_serialize_dump(path: Path) -> str:
    node = _strip_docstring(_find_function(path, "serialize"))
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and child.value == 120:
            child.value = 121
            return ast.dump(node, annotate_fields=True)
    raise AssertionError("reference serialize() has no Constant(value=120) to mutate")


@pytest.mark.parametrize("path", TARGET_PATHS[1:], ids=lambda p: str(p.relative_to(ROOT)))
@pytest.mark.parametrize("name", FUNCTION_NAMES)
def test_serialize_helpers_match_submit_reference_ast(path: Path, name: str):
    expected = _function_dump(REFERENCE_PATH, name)
    actual = _function_dump(path, name)

    assert _dumps_match(expected, actual), (
        f"{path.relative_to(ROOT)}:{name} AST differs from submit/script.py:{name}\n"
        + _dump_diff(expected, actual)
    )


@pytest.mark.parametrize("path", TARGET_PATHS, ids=lambda p: str(p.relative_to(ROOT)))
def test_serialize_keeps_required_char_cap_patterns(path: Path):
    serialize_source = _normalized_source(_function_source(path, "serialize"))
    missing = [label for label, pattern in SERIALIZE_SOURCE_PATTERNS.items() if pattern not in serialize_source]

    assert not missing, f"{path.relative_to(ROOT)}:serialize missing source guard(s): {', '.join(missing)}"


def test_serialize_ast_comparison_detects_drift():
    expected = _function_dump(REFERENCE_PATH, "serialize")
    mutated = _mutated_serialize_dump(REFERENCE_PATH)

    assert mutated != expected
    assert not _dumps_match(expected, mutated)
