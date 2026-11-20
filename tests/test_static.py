"""The library passes its linter and its type checker, as ``pyproject.toml`` configures them.

Both are development dependencies (``pip install -e ".[dev]"``); where one is
not installed, its test is skipped.
"""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent


def _run(module: str, *arguments: str) -> subprocess.CompletedProcess:
    if importlib.util.find_spec(module) is None:
        pytest.skip(f"{module} is not installed; pip install -e \".[dev]\"")
    return subprocess.run([sys.executable, "-m", module, *arguments], cwd=_ROOT,
                          capture_output=True, text=True, check=False)


@pytest.mark.time_limit_s(300)
def test_the_linter_finds_nothing():
    checked = _run("ruff", "check", "--no-cache", "llama_chat", "tests", "benchmarks")
    assert checked.returncode == 0, checked.stdout + checked.stderr


@pytest.mark.time_limit_s(600)
def test_the_type_checker_finds_nothing():
    checked = _run("mypy", "--cache-dir", os.devnull, "llama_chat")
    assert checked.returncode == 0, checked.stdout + checked.stderr
