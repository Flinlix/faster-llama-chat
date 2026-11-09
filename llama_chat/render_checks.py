"""What a template's renders allow a conversation to save: each a pure function of a render callable.

``render(messages, generation_prompt)`` returns the template's render, or
``None`` where it refuses the shape. Each check renders probe conversations of
its own, with the conversation's system message, tools and reasoning already
bound into ``render``.
"""

from __future__ import annotations

from collections.abc import Callable

Render = Callable[[list[dict], bool], "str | None"]

# Stands where the next question will go in the render made ahead of it.
STAND_IN = "QQSTANDINQQ"
# The exchanges a template must write the same in a window as in the whole render for windows to be used.
_WINDOW_PROBES = tuple(
    [{"role": "user", "content": f"Question {n}, asked plainly."},
     {"role": "assistant", "content": f"Reply {n}, given plainly."}] for n in range(5))
_SPLICE_BASE = [{"role": "user", "content": "Hello."}, {"role": "assistant", "content": "Hi."}]


def question_splices(render: Render, system: list[dict], question: str) -> bool:
    """Whether the template writes ``question`` exactly where it writes the stand-in, after one exchange."""
    ahead = render(system + _SPLICE_BASE + [{"role": "user", "content": STAND_IN}], True)
    if ahead is None or ahead.count(STAND_IN) != 1:
        return False
    return render(system + _SPLICE_BASE + [{"role": "user", "content": question}],
                  True) == ahead.replace(STAND_IN, question)


def windows_are_exact(render: Render, system: list[dict], tag_after) -> bool:
    """Whether this template writes the newest two exchanges, and the oldest two kept after a drop, as the whole render does.

    Checked on five exchanges with the conversation's system message, tools
    and reasoning: the window of the newest two from the first tag after the
    older one's reply, the history before that tag as the prompt for the
    newest reply wrote it, and after dropping the oldest exchange, the oldest
    two kept up to the first tag after the first one's reply and everything
    after that tag as before.

    Args:
        tag_after: ``(text, at)`` to where the first tag at or after ``at``
            begins, ``None`` where there is none.
    """
    stand_in = [{"role": "user", "content": STAND_IN}]
    ex = [list(e) for e in _WINDOW_PROBES]
    whole = render(system + sum(ex, []) + stand_in, True)
    window = render(system + ex[3] + ex[4] + stand_in, True)
    prompt = render(system + sum(ex[:4], []) + ex[4][:1], True)
    dropped = render(system + sum(ex[1:], []) + stand_in, True)
    head = render(system + ex[1] + ex[2] + stand_in, True)
    if None in (whole, window, prompt, dropped, head):
        return False
    older, first = ex[3][1]["content"], ex[1][1]["content"]

    def after(text, content):
        found = text.find(content)
        return None if found < 0 else tag_after(text, found + len(content))

    at = [after(text, content) for text, content in
          ((whole, older), (window, older), (dropped, first), (head, first), (whole, first))]
    if None in at:
        return False
    tail, in_window, drop, in_head, kept = at
    return (whole[tail:] == window[in_window:] and prompt[:tail] == whole[:tail]
            and dropped[:drop] == head[:in_head] and dropped[drop:] == whole[kept:])


def ensure_history_stays_written(render: Render, system: list[dict]) -> None:
    """Refuse a template that writes an older exchange otherwise once another follows.

    Such a conversation could only be kept by decoding it again each turn,
    which the rolling memory does not do. Changes within the newest exchange,
    such as a past reply written without its thought or a system prompt moving
    to the newest question, are what the cache follows.

    Raises:
        ValueError: If the template rewrites anything before the newest
            exchange, or refuses the probe that shows whether it does.
    """
    exchanges = [[{"role": "user", "content": f"QQASK{n}QQ"},
                  {"role": "assistant", "content": f"QQREPLY{n}QQ"}] for n in range(3)]
    two = render(system + exchanges[0] + exchanges[1], False)
    three = render(system + exchanges[0] + exchanges[1] + exchanges[2], False)
    if two is None or three is None or "QQREPLY0QQ" not in two:
        raise ValueError("this model's template refuses a plain conversation of three "
                         "exchanges, so whether it keeps older exchanges as written cannot "
                         "be told")
    kept = two.index("QQREPLY0QQ") + len("QQREPLY0QQ")
    if two[:kept] != three[:kept]:
        raise ValueError("this model's template writes an older exchange otherwise once "
                         "another follows, which a rolling memory cannot keep")


def writes_two_user_turns(render: Render) -> bool:
    """Whether the template writes two user turns in a row."""
    twice = [{"role": "user", "content": "QQONEQQ"}, {"role": "user", "content": "QQTWOQQ"}]
    return render(twice, False) is not None
