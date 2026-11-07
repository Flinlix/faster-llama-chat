"""Filtering a reply's text: stop strings, thoughts and channel headers."""

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


# ----- the thought filter --------------------------------------------------
def test_the_thought_count_never_exceeds_the_pieces_fed():
    import random
    from llama_chat.reply_text import ThoughtFilter
    tags = Reasoning("<thought>", "</thought>", "", "")
    text = "a<thought>x</thought>b<thought>y</thought>c<tho"
    rng = random.Random(7)
    for _ in range(2000):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 12)))
        pieces = [text[i:j] for i, j in zip([0, *cuts], [*cuts, len(text)])]
        f = ThoughtFilter(tags)
        shown = "".join(f.feed(p) for p in pieces) + f.flush()
        assert shown == "abc<tho"
        assert 0 < f.thought_pieces <= len(pieces)


# ----- a channel's header --------------------------------------------------
def test_a_channel_header_cut_short_is_not_answer():
    from llama_chat.reply_text import AnswerOpening
    opening = AnswerOpening("<|channel|>final<|message|>")
    assert opening.feed("<|channel|>final") == "" and opening.flush(complete=False) == ""


def test_whitespace_after_a_channel_header_is_not_answer():
    from llama_chat.reply_text import AnswerOpening
    opening = AnswerOpening("</mm:think>")
    assert opening.feed("</mm:think>") + opening.feed("\n\n") + opening.feed("Antwort.") \
        == "Antwort."


def test_whitespace_opening_a_reply_without_a_thought_stays_the_answers():
    from llama_chat.reply_text import ThoughtFilter
    f = ThoughtFilter(Reasoning("<thought>", "</thought>", "", ""))
    assert f.feed(" ") + f.feed(" x") + f.flush() == "  x"
    g = ThoughtFilter(Reasoning("<thought>", "</thought>", "", ""))
    assert g.feed(" ") + g.feed("<thought>a</thought>") + g.feed("b") + g.flush() == "b"
