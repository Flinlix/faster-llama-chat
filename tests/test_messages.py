"""Pure bookkeeping tests for the message table (no model required)."""

import pytest

from llama_chat.messages import Message, MessageTable, fit_newest_first


def _msg(role, n):
    return Message(role, role[0] * n, list(range(n)))


def test_append_assigns_contiguous_positions():
    t = MessageTable()
    t.append(_msg("system", 3))
    t.append(_msg("user", 4))
    t.append(_msg("assistant", 5))
    assert [(m.pos_start, m.pos_end) for m in t.messages] == [(0, 3), (3, 7), (7, 12)]
    assert t.total == 12


def test_evict_oldest_keeps_system_and_renumbers():
    t = MessageTable()
    t.append(_msg("system", 3))
    t.append(_msg("user", 4))      # positions [3, 7)
    t.append(_msg("assistant", 5))  # positions [7, 12)

    ev = t.evict_oldest()
    assert (ev.removed_start, ev.removed_end, ev.old_total) == (3, 7, 12)
    assert ev.shift_delta == 4
    # System preserved, assistant shifted down to close the gap.
    roles = [(m.role, m.pos_start, m.pos_end) for m in t.messages]
    assert roles == [("system", 0, 3), ("assistant", 3, 8)]
    assert t.total == 8


def test_evict_raises_when_only_system_remains():
    t = MessageTable()
    t.append(_msg("system", 3))
    with pytest.raises(IndexError):
        t.evict_oldest()


def test_n_evictable():
    t = MessageTable()
    assert t.n_evictable == 0
    t.append(_msg("system", 1))
    assert t.n_evictable == 0
    t.append(_msg("user", 1))
    t.append(_msg("assistant", 1))
    assert t.n_evictable == 2


def test_fit_newest_first():
    # system=2, budget=10 -> 8 left. messages oldest..newest: [5,4,3]
    # newest-first: 3 (->5 used 3), 4 (->9), 5 would exceed -> stop. keep 2.
    assert fit_newest_first([5, 4, 3], system_tokens=2, budget=10) == 2
    # everything fits
    assert fit_newest_first([1, 1, 1], system_tokens=2, budget=10) == 3
    # nothing fits
    assert fit_newest_first([20], system_tokens=2, budget=10) == 0
    # empty history
    assert fit_newest_first([], system_tokens=2, budget=10) == 0
