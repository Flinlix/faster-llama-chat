"""A mutation run that also tests again what the tests edited, added or removed since the last run reach.

Run from the project's root as ``python -m tests.mutation_run``; options
such as ``--max-children`` are handed to each ``mutmut run``. The mutation tool tests a mutant again only
where the library function it is in changed. This run first finds the tests
whose code changed since the last run, which counts every test where code the
tests share changed, has the tool record what those tests reach now, and
clears the results of every mutant in a function they reached before or reach
now. The first run records the tests as they are.
"""

from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).parent.parent
_TESTS = _ROOT / "tests"
_RESULTS = _ROOT / "mutants"
_STATS = _RESULTS / "mutmut-stats.json"
_FINGERPRINTS = _RESULTS / "mutmut-test-fingerprints.json"
# Test code the tests share: every file in the tests folder but a test module, and the library's testing package.
_SHARED = (_TESTS, _ROOT / "llama_chat" / "testing")
_MUTANT_MARK = "__mutmut_"


def _digest(text: str | bytes) -> str:
    return hashlib.sha256(text.encode() if isinstance(text, str) else text).hexdigest()[:16]


def _is_test_module(path: Path) -> bool:
    return path.parent == _TESTS and path.name.startswith("test_") and path.suffix == ".py"


def _shared_digest() -> str:
    files = sorted(path for folder in _SHARED for path in folder.rglob("*")
                   if path.is_file() and path != Path(__file__) and not _is_test_module(path)
                   and not any(part.startswith(".") or part == "__pycache__" for part in path.parts))
    return _digest("".join(f"{path.relative_to(_ROOT)}:{_digest(path.read_bytes())}" for path in files))


def _test_fingerprints() -> dict[str, str]:
    """Each test function, ``file::name`` or ``file::Class::name``, mapped to a digest of its code, of its module's other code and of the shared test code."""
    shared = _shared_digest()
    fingerprints = {}
    for path in sorted(_TESTS.glob("test_*.py")):
        tests: dict[str, str] = {}
        rest = []
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test"):
                tests[node.name] = ast.dump(node)
            elif isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
                for member in node.body:
                    if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)) and member.name.startswith("test"):
                        tests[f"{node.name}::{member.name}"] = ast.dump(member)
                    else:
                        rest.append(ast.dump(member))
            else:
                rest.append(ast.dump(node))
        module = _digest(shared + "".join(rest))
        name = path.relative_to(_ROOT).as_posix()
        fingerprints.update({f"{name}::{test}": _digest(module + code) for test, code in tests.items()})
    return fingerprints


def _function_of(test_id: str) -> str:
    """The test function a collected test's id belongs to, its parameters left out."""
    return test_id.partition("[")[0]


def _reach(stats: dict, test_functions: set[str]) -> set[str]:
    """The library functions a test of ``test_functions`` reaches, as the tool names them."""
    return {name for name, tests in stats["tests_by_mangled_function_name"].items()
            if any(_function_of(test) in test_functions for test in tests)}


def _clear_results(reached: set[str] | None) -> int:
    """Clear the result of every mutant in a function of ``reached``, or of every mutant where ``None``, so that the next run tests it."""
    cleared = 0
    for results_file in _RESULTS.rglob("*.py.meta"):
        data = json.loads(results_file.read_text())
        results = data["exit_code_by_key"]
        keys = [key for key, result in results.items()
                if result is not None and (reached is None or key.partition(_MUTANT_MARK)[0] in reached)]
        for key in keys:
            results[key] = None
        if keys:
            results_file.write_text(json.dumps(data, indent=4))
            cleared += len(keys)
    return cleared


def _forget(stats: dict, test_functions: set[str]) -> None:
    """Drop what the tests of ``test_functions`` reached, so that the tool collects it again as for a new test."""
    for test in [test for test in stats["duration_by_test"] if _function_of(test) in test_functions]:
        del stats["duration_by_test"][test]
    for name, tests in stats["tests_by_mangled_function_name"].items():
        stats["tests_by_mangled_function_name"][name] = [test for test in tests if _function_of(test) not in test_functions]


def _cheapest_mutant(stats: dict) -> str:
    """A mutant whose function the fewest tests reach, for a run that only collects what tests reach."""
    reached = {name: len(tests) for name, tests in stats["tests_by_mangled_function_name"].items() if tests}
    keys = [key for results_file in _RESULTS.rglob("*.py.meta") for key in json.loads(results_file.read_text())["exit_code_by_key"]
            if key.partition(_MUTANT_MARK)[0] in reached]
    if not keys:
        raise SystemExit("no mutant is reached by any test")
    return min(keys, key=lambda key: reached[key.partition(_MUTANT_MARK)[0]])


def _mutmut_run(*arguments: str) -> None:
    subprocess.run([sys.executable, "-m", "mutmut", "run", *arguments], cwd=_ROOT, check=True)


def main(arguments: list[str]) -> None:
    fingerprints = _test_fingerprints()
    if _STATS.exists() and _FINGERPRINTS.exists():
        before = json.loads(_FINGERPRINTS.read_text())
        changed = {test for test in before.keys() | fingerprints.keys() if before.get(test) != fingerprints.get(test)}
        if changed:
            # Cleared before what they reached is forgotten, so that a run stopped here loses nothing.
            stats = json.loads(_STATS.read_text())
            cleared = _clear_results(_reach(stats, changed))
            _forget(stats, changed)
            if any(stats["tests_by_mangled_function_name"].values()):
                _STATS.write_text(json.dumps(stats))
                print(f"{len(changed)} test functions changed; collecting what they reach now")
                _mutmut_run(_cheapest_mutant(stats), *arguments)
                cleared += _clear_results(_reach(json.loads(_STATS.read_text()), changed))
            else:
                # Every test changed: the tool collects what each one reaches, as on its first run.
                cleared += _clear_results(None)
                _STATS.unlink()
            print(f"{cleared} mutants to test again")
    _FINGERPRINTS.write_text(json.dumps(fingerprints, indent=1))
    _mutmut_run(*arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
