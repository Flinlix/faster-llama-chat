"""The project's contract and what it names agree: every file a rule is verified by exists, the rules are numbered in order, and every rule a file cites by number is one of them.

The contract is ``AGENTS.md`` at the project's root.
"""

import re
from pathlib import Path


_ROOT = Path(__file__).parent.parent
_CONTRACT = _ROOT / "AGENTS.md"
# Folders a tool writes, which hold copies of the project's files.
_BUILT = {"mutants", "build", "dist", "site", "__pycache__"}


def _rules() -> dict[int, str]:
    """Each rule of the contract by its number, with its text up to the next."""
    text = _CONTRACT.read_text()
    section = text.split("## Rules", 1)[1].split("\n## ", 1)[0]
    found = list(re.finditer(r"^(\d+)\. \*\*", section, re.M))
    return {int(m.group(1)): section[m.start():(found[i + 1].start() if i + 1 < len(found) else None)]
            for i, m in enumerate(found)}


def test_the_rules_are_numbered_from_one_without_a_gap():
    assert list(_rules()) == list(range(1, len(_rules()) + 1))


def test_every_file_a_rule_is_verified_by_exists():
    for number, text in _rules().items():
        named = re.findall(r"`((?:tests|llama_chat)/[\w/.-]+)`", text)
        assert named, f"rule {number} names no file it is verified by"
        for path in named:
            assert (_ROOT / path).exists(), f"rule {number} names {path}, which does not exist"


def test_every_rule_a_file_cites_by_number_is_one_of_the_contracts():
    highest = max(_rules())
    for path in sorted(_ROOT.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".md") or path == _CONTRACT \
                or any(part.startswith(".") or part in _BUILT for part in path.relative_to(_ROOT).parts):
            continue
        for cited in re.findall(r"\b[Rr]ule (\d+)\b", path.read_text(errors="replace")):
            assert 1 <= int(cited) <= highest, \
                f"{path.relative_to(_ROOT)} cites rule {cited}; the contract has {highest}"
