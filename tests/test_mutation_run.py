"""The mutation run's script tests again exactly what the tests changed since the last run reach."""

import json

import pytest

from . import mutation_run


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A project of two test modules and one shared helper, the script pointed at it."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "helper.py").write_text("VALUE = 1\n")
    (tests / "test_a.py").write_text("SETUP = 1\n\ndef test_one():\n    assert 1\n\ndef test_two():\n    assert 2\n")
    (tests / "test_b.py").write_text("class TestGroup:\n    def test_three(self):\n        assert 3\n")
    (tmp_path / "mutants").mkdir()
    monkeypatch.setattr(mutation_run, "_ROOT", tmp_path)
    monkeypatch.setattr(mutation_run, "_TESTS", tests)
    monkeypatch.setattr(mutation_run, "_SHARED", (tests,))
    monkeypatch.setattr(mutation_run, "_RESULTS", tmp_path / "mutants")
    return tests


def _changed(before, after):
    return {test for test in before.keys() | after.keys() if before.get(test) != after.get(test)}


def test_a_test_function_is_named_as_the_tool_names_it(project):
    assert set(mutation_run._test_fingerprints()) == {
        "tests/test_a.py::test_one", "tests/test_a.py::test_two", "tests/test_b.py::TestGroup::test_three"}


def test_an_edit_counts_for_the_tests_it_can_change_and_a_comment_for_none(project):
    before = mutation_run._test_fingerprints()
    test_a = project / "test_a.py"
    test_a.write_text(test_a.read_text().replace("assert 2", "assert 2  # a comment"))
    assert _changed(before, mutation_run._test_fingerprints()) == set()
    test_a.write_text(test_a.read_text().replace("assert 2", "assert 22"))
    assert _changed(before, mutation_run._test_fingerprints()) == {"tests/test_a.py::test_two"}
    test_a.write_text(test_a.read_text().replace("SETUP = 1", "SETUP = 2"))
    assert _changed(before, mutation_run._test_fingerprints()) == {"tests/test_a.py::test_one", "tests/test_a.py::test_two"}
    (project / "helper.py").write_text("VALUE = 2\n")
    assert _changed(before, mutation_run._test_fingerprints()) == set(before)


def test_what_changed_tests_reached_is_tested_again_and_collected_again(project):
    stats = {"duration_by_test": {"tests/test_a.py::test_one[x]": 1.0, "tests/test_a.py::test_two": 1.0},
             "tests_by_mangled_function_name": {"pkg.mod.x_f": ["tests/test_a.py::test_one[x]"],
                                                "pkg.mod.x_g": ["tests/test_a.py::test_two"]}}
    results_file = project.parent / "mutants" / "mod.py.meta"
    results_file.write_text(json.dumps({"exit_code_by_key": {
        "pkg.mod.x_f__mutmut_1": 1, "pkg.mod.x_f__mutmut_2": 0, "pkg.mod.x_g__mutmut_1": 0, "pkg.mod.x_g__mutmut_2": None}}))
    changed = {"tests/test_a.py::test_one"}

    assert mutation_run._clear_results(mutation_run._reach(stats, changed)) == 2
    assert json.loads(results_file.read_text())["exit_code_by_key"] == {
        "pkg.mod.x_f__mutmut_1": None, "pkg.mod.x_f__mutmut_2": None, "pkg.mod.x_g__mutmut_1": 0, "pkg.mod.x_g__mutmut_2": None}
    assert mutation_run._cheapest_mutant(stats) == "pkg.mod.x_f__mutmut_1"
    mutation_run._forget(stats, changed)
    assert stats == {"duration_by_test": {"tests/test_a.py::test_two": 1.0},
                     "tests_by_mangled_function_name": {"pkg.mod.x_f": [], "pkg.mod.x_g": ["tests/test_a.py::test_two"]}}
