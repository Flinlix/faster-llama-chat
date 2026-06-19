"""Pure bookkeeping tests for the message table (no model required)."""

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


def test_evict_oldest_until_single_message_renumbers():
    t = MessageTable()
    t.append(_msg("system", 3))
    t.append(_msg("user", 4))      # positions [3, 7)
    t.append(_msg("assistant", 5))  # positions [7, 12)

    ev, count = t.evict_oldest_until(lambda: t.total <= 8)
    assert count == 1
    assert (ev.remove_start, ev.remove_end, ev.old_total) == (3, 7, 12)
    assert ev.shift_delta == 4
    # System preserved, assistant shifted down to close the gap.
    roles = [(m.role, m.pos_start, m.pos_end) for m in t.messages]
    assert roles == [("system", 0, 3), ("assistant", 3, 8)]
    assert t.total == 8


def test_evict_oldest_until_coalesces():
    t = MessageTable()
    t.append(_msg("system", 3))
    t.append(_msg("user", 4))       # positions [3, 7)
    t.append(_msg("assistant", 5))  # positions [7, 12)
    t.append(_msg("user", 6))       # positions [12, 18)

    # Force dropping the two oldest evictable messages (4 + 5 tokens).
    ev, count = t.evict_oldest_until(lambda: t.total <= 9)
    assert count == 2
    # One combined edit spanning exactly the dropped block [3, 12).
    assert (ev.remove_start, ev.remove_end, ev.old_total) == (3, 12, 18)
    assert ev.shift_delta == 9
    # System preserved, survivor shifted down to close the gap.
    roles = [(m.role, m.pos_start, m.pos_end) for m in t.messages]
    assert roles == [("system", 0, 3), ("user", 3, 9)]
    assert t.total == 9


def test_evict_oldest_until_noop_when_already_fits():
    t = MessageTable()
    t.append(_msg("system", 3))
    t.append(_msg("user", 4))
    ev, count = t.evict_oldest_until(lambda: True)
    assert ev is None and count == 0
    assert t.total == 7


def test_evict_oldest_until_stops_when_only_system_remains():
    t = MessageTable()
    t.append(_msg("system", 3))
    ev, count = t.evict_oldest_until(lambda: False)  # can never fit
    assert ev is None and count == 0
    assert [m.role for m in t.messages] == ["system"]


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
