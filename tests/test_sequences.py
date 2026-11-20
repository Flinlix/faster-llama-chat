"""Random sequences of actions, drawn and shrunk by Hypothesis, on every template that calls tools, every kind of cache and both tokenizers: every state they reach keeps the rules.

After every action: the cache is the template's render (rule 1), a refused or
failed action changed nothing (rule 6), the conversation begins again the same
(rule 10), and a question asked then is answered (rule 11). A failing sequence
is shrunk to the shortest one that still fails and printed step by step, with
the blob that replays it.
"""

import json
import os
from pathlib import Path

import pytest
from hypothesis import HealthCheck, assume, note, settings, strategies as st
from hypothesis.stateful import (RuleBasedStateMachine, initialize, invariant, precondition, rule,
                                 run_state_machine_as_test)

from llama_cpp_conversation.addon_files import template_fingerprint
from tests.sequence_run import KINDS, SequenceRun, _known

# Each run draws its choices from the test's own source, so that nothing is saved between
# runs and every failure is printed with what replays it.
_SHARED = dict(database=None, deadline=None, print_blob=True,
               suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large,
                                      HealthCheck.filter_too_much])
# A run of the suite: the same sequences every time. A slow run: new ones every time.
_EVERY_RUN = settings(max_examples=40, stateful_step_count=12, derandomize=True, **_SHARED)
_SLOW_RUN = settings(max_examples=int(os.environ.get("LLAMA_CPP_CONVERSATION_SLOW_EXAMPLES", "1500")),
                     stateful_step_count=24, **_SHARED)


class _Draw:
    """The ``random()``, ``choice()`` and ``randint()`` a sequence draws with, answered by Hypothesis so that it can shrink them."""

    def __init__(self, data) -> None:
        self._data = data

    def random(self) -> float:
        return self._data.draw(st.floats(0, 1, exclude_max=True))

    def choice(self, options):
        return self._data.draw(st.sampled_from(list(options)))

    def randint(self, low: int, high: int) -> int:
        return self._data.draw(st.integers(low, high))


class Sequences(RuleBasedStateMachine):
    """One conversation on one drawn setup; each rule an action, the rules checked after each."""

    run: SequenceRun | None = None
    begun = False
    steps = 0

    @initialize(data=st.data())
    def set_up(self, data):
        self.run = SequenceRun(_Draw(data), note)
        self.begun = self.run.first_begin()
        assume(self.begun)              # declarations longer than the context drawn
        self.run.check("the first begin")

    def _act(self, data, kind: str) -> None:
        self.run.draw = _Draw(data)
        self.steps += 1
        note(f"step {self.steps}: {kind}")
        self.run.act(kind, self.steps)

    for _kind in KINDS:
        def _rule(self, data, kind=_kind):
            self._act(data, kind)
        _rule.__name__ = f"act_{_kind}"
        locals()[_rule.__name__] = precondition(lambda self: self.begun)(rule(data=st.data())(_rule))
    del _kind, _rule

    @precondition(lambda self: self.begun)
    @rule()
    def ask_a_question(self):
        self.run.question_answered(f"step {self.steps}")

    @invariant()
    def keeps_the_rules(self):
        if self.begun:
            self.run.check(f"step {self.steps}")

    def teardown(self):
        if self.run is not None:
            if self.begun:
                self.run.question_answered("the end")
                self.run.check("the end")
            self.run.wrapper.close()


@pytest.mark.time_limit_s(900)
def test_every_sequence_drawn_keeps_the_rules():
    run_state_machine_as_test(Sequences, settings=_EVERY_RUN)


@pytest.mark.slow
@pytest.mark.time_limit_s(3 * 3600)
def test_many_new_sequences_keep_the_rules():
    run_state_machine_as_test(Sequences, settings=_SLOW_RUN)


# ----- sequences once found failing, replayed every run ---------------------------
_FOUND = sorted((Path(__file__).parent / "found_sequences").glob("*.json"))


class _Recorded:
    """The values a sequence once drew, given back in the order it drew them: a choice by its place among the options, the template by its fingerprint, so that the record names neither."""

    def __init__(self, values: list) -> None:
        self._values = iter(values)

    def random(self) -> float:
        return next(self._values)

    def randint(self, low: int, high: int) -> int:
        return next(self._values)

    def choice(self, options):
        options, drawn = list(options), next(self._values)
        if "template" in drawn:
            return next(name for name in options
                        if template_fingerprint(_known(name)["chat_template"]) == drawn["template"])
        return options[drawn["option"]]


@pytest.mark.parametrize("path", _FOUND, ids=[path.stem for path in _FOUND])
def test_each_sequence_once_found_failing_keeps_the_rules(path):
    """A failing sequence the random runs found, kept as its drawn values and its steps, as Hypothesis keeps none on disk; it replays only while ``sequence_run`` draws in the same order."""
    record = json.loads(path.read_text())
    run, steps = None, 0
    for step in record["steps"]:
        if step == "set_up":
            run = SequenceRun(_Recorded(record["draws"]), lambda _text: None)
            assert run.first_begin()
            run.check("the first begin")
        elif step.startswith("act_"):
            steps += 1
            run.act(step[len("act_"):], steps)
        elif step == "keeps_the_rules":
            run.check(f"step {steps}")
        elif step == "ask_a_question":
            run.question_answered(f"step {steps}")
    run.question_answered("the end")
    run.check("the end")
    run.wrapper.close()
