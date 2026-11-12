"""Filtering a reply's text: stop strings, thoughts and channel headers."""

import pytest

from llama_chat.reply_text import ReplyText
from llama_chat.template import Reasoning


# ----- stop-string scanning ------------------------------------------------
def test_the_earliest_of_several_needles_is_found():
    from llama_chat.reply_text import earliest

    assert earliest("abcSTOPdef", ["STOP"]) == 3
    assert earliest("aXbYc", ["Y", "X"]) == 1          # earliest of several
    assert earliest("abc", ["STOP"]) is None
    assert earliest("abc", ["", "b"]) == 1             # empty strings ignored
    # A match strictly before the offset is skipped; one spanning or after it is found.
    assert earliest("STOPxxSTOP", ["STOP"], search_from=1) == 6
    assert earliest("xxSTOP", ["STOP"], search_from=2) == 2
    assert earliest("xxSTOP", ["STOP"], search_from=3) is None


# ----- the thought ---------------------------------------------------------
def test_the_thought_count_never_exceeds_the_pieces_fed():
    import random
    tags = Reasoning("<thought>", "</thought>", "", "")
    text = "a<thought>x</thought>b<thought>y</thought>c<tho"
    rng = random.Random(7)
    for _ in range(2000):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 12)))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
        reply = ReplyText(tags, "", "", [])
        shown = "".join(reply.feed(p) for p in pieces) + reply.finish("end_of_turn")
        assert shown == reply.answer == "abc<tho"
        assert reply.thought == "xy"
        assert 0 < reply.thought_pieces <= len(pieces)


# ----- a channel's header --------------------------------------------------
def test_a_channel_header_cut_short_is_not_answer():
    reply = ReplyText(None, "<|route|>final<|body|>", "", [])
    assert reply.feed("<|route|>final") == "" and reply.finish("token_limit") == ""


@pytest.mark.parametrize("opening", ["</reason>", "</xx:reason>"])
def test_whitespace_after_a_channel_header_is_not_answer(opening):
    reply = ReplyText(None, opening, "", [])
    assert reply.feed(opening) + reply.feed("\n\n") + reply.feed("Antwort.") == "Antwort."


def test_whitespace_opening_a_reply_without_a_thought_stays_the_answers():
    tags = Reasoning("<thought>", "</thought>", "", "")
    f = ReplyText(tags, "", "", [])
    assert f.feed(" ") + f.feed(" x") + f.finish("end_of_turn") == "  x"
    g = ReplyText(tags, "", "", [])
    assert g.feed(" ") + g.feed("<thought>a</thought>") + g.feed("b") \
        + g.finish("end_of_turn") == "b"
