"""The project's prose keeps the writing rules a machine can check: no em dash anywhere, no dash between clauses, and outside the decision records no path of one machine.

Prose is the Markdown outside code blocks and code spans, and the docstrings
and comments of the Python files.
"""

import ast
import io
import re
import sys
import tokenize
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent
# Folders a tool writes, which hold copies of the project's files.
_BUILT = {"mutants", "build", "dist", "site", "__pycache__"}
# Where the machines measurements were taken on are named: the decision records.
_DECISIONS = _ROOT / "docs" / "decisions"

_EM_DASH = re.compile("\u2014")
_CLAUSE_DASH = re.compile(r"\w\s(?:-|--|–)\s\w")
_MACHINE_PATH = re.compile(r"(?:/home/|/tmp/|/Users/|[A-Z]:\\\\)")
_CODE_SPAN = re.compile(r"``?[^`]*``?")


def _prose_files():
    for path in sorted(_ROOT.rglob("*")):
        parts = path.relative_to(_ROOT).parts
        if path.is_file() and path.suffix in (".py", ".md") \
                and not any(part.startswith(".") or part in _BUILT for part in parts):
            yield path


def _python_prose(text: str):
    """The lines of ``text``'s comments and docstrings, with their numbers."""
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            yield token.start[0], token.string
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node, clean=False)
            if docstring:
                for offset, line in enumerate(docstring.splitlines()):
                    yield node.body[0].lineno + offset, line


def _markdown_prose(text: str):
    """The lines of ``text`` outside its code blocks, with their numbers."""
    fenced = False
    for number, line in enumerate(text.splitlines(), 1):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced:
            yield number, line


def _breaks(path: Path) -> list[str]:
    text = path.read_text()
    lines = _python_prose(text) if path.suffix == ".py" else _markdown_prose(text)
    # Nowhere, not even in code.
    found = [f"{path.relative_to(_ROOT)}:{number}: an em dash"
             for number, line in enumerate(text.splitlines(), 1) if _EM_DASH.search(line)]
    for number, line in lines:
        prose = _CODE_SPAN.sub("", line)
        where = f"{path.relative_to(_ROOT)}:{number}"
        if _CLAUSE_DASH.search(prose):
            found.append(f"{where}: a dash between clauses")
        if _DECISIONS not in path.parents:
            if _MACHINE_PATH.search(prose):
                found.append(f"{where}: a path of one machine")
    return found


@pytest.mark.parametrize("path", list(_prose_files()),
                         ids=[str(p.relative_to(_ROOT)) for p in _prose_files()])
def test_the_prose_keeps_the_writing_rules(path):
    assert _breaks(path) == []


def test_each_break_of_the_writing_rules_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(sys.modules[__name__], "_ROOT", tmp_path)
    sample = tmp_path / "notes.md"
    sample.write_text("One clause \u2014 another.\nOne - two.\nAt /home/someone it ran.\n"
                      "```\nx = a - b  # code, left alone\n```\nIn `a - b` too.\n")
    found = _breaks(sample)
    assert [line.split(": ", 1)[1] for line in found] == [
        "an em dash", "a dash between clauses", "a path of one machine"]
