"""Filtering a reply's text: stop strings, thoughts and channel headers."""

import re
import random
import pytest

from llama_chat.reply_text import ReplyText, reply_header
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
        reply = ReplyText(tags, "", None, "", [])
        shown = "".join(reply.feed(p) for p in pieces) + reply.finish("end_of_turn")
        assert shown == reply.answer == "abc<tho"
        assert reply.thought == "xy"
        assert 0 < reply.thought_pieces <= len(pieces)


def test_the_turns_end_closes_a_thought_the_generated_text_is_in_though_none_of_it_was_fed():
    tags = Reasoning("<|ch|>think<|msg|>", "<|stop|><|ch|>final<|msg|>", "")
    texts = {1: b"<|ch|>think<|msg|>", 2: b"Kurz.", 3: b"<|stop|>", 4: b"<|ch|>final<|msg|>"}
    generated = [1, 2]
    ends = ReplyText(tags, "", None, "", []).turn_end_test(3, "<|stop|>", generated,
                                                     texts.__getitem__, lambda token: token != 2)
    assert not ends(3)
    generated += [3, 4, 2]
    assert ends(3)


def _fed_as_tokens(reply, tokens, tag_spans):
    """Feed ``tokens``, each a text and whether it is one of the model's tags, as a generation does."""
    shown, at = [], 0
    for text, is_tag in tokens:
        if is_tag:
            tag_spans.append((at, at + len(text)))
        at += len(text)
        shown.append(reply.feed(text))
    return "".join(shown) + reply.finish("end_of_turn")


@pytest.mark.parametrize("written_as_tag", [True, False])
def test_after_answer_text_only_the_models_own_tag_opens_a_thought(written_as_tag):
    tags = Reasoning("<thought>", "</thought>", "", "")
    spans = []
    reply = ReplyText(tags, "", None, "", [], spans)
    shown = _fed_as_tokens(reply, [("<thought>", True), ("plan", False), ("</thought>", True),
                                   ("Write ", False), ("<thought>", written_as_tag),
                                   (" to start one.", False)], spans)
    if written_as_tag:
        assert (shown, reply.thought) == ("Write ", "plan to start one.")
    else:
        assert (shown, reply.thought) == ("Write <thought> to start one.", "plan")


@pytest.mark.parametrize("tags", [None, Reasoning("<thought>", "</thought>", "", "")])
def test_a_tag_the_model_writes_in_its_answer_is_not_shown_and_its_letters_are(tags):
    spans = []
    reply = ReplyText(tags, "<|text|>", None, "", [], spans)
    shown = _fed_as_tokens(reply, [("<|text|>", True), ("Rain later.", False), ("</thought>", True),
                                   ("<|text|>", True), (" Spell <|text|> so.", False)], spans)
    assert shown == reply.answer == "Rain later. Spell <|text|> so."


# ----- a channel's header --------------------------------------------------
def test_a_channel_header_cut_short_is_not_answer():
    reply = ReplyText(None, "<|route|>final<|body|>", None, "", [])
    assert reply.feed("<|route|>final") == "" and reply.finish("token_limit") == ""


@pytest.mark.parametrize("opening", ["</reason>", "</xx:reason>"])
def test_whitespace_after_a_channel_header_is_not_answer(opening):
    reply = ReplyText(None, opening, None, "", [])
    assert reply.feed(opening) + reply.feed("\n\n") + reply.feed("Antwort.") == "Antwort."


def test_whitespace_opening_a_reply_without_a_thought_stays_the_answers():
    tags = Reasoning("<thought>", "</thought>", "", "")
    f = ReplyText(tags, "", None, "", [])
    assert f.feed(" ") + f.feed(" x") + f.finish("end_of_turn") == "  x"
    g = ReplyText(tags, "", None, "", [])
    assert g.feed(" ") + g.feed("<thought>a</thought>") + g.feed("b") \
        + g.finish("end_of_turn") == "b"


def test_a_stop_string_ends_the_answer_before_it_even_where_a_later_piece_holds_it_twice():
    from llama_chat.reply_text import _AnswerStops
    stops = _AnswerStops(["\n"])
    stops.feed("Hello")
    stops.feed("\n\nmore")
    assert stops.text == "Hello"


def _stream_cut_anywhere_but_inside_a_tag(tokens, rng):
    """``tokens`` joined and cut into pieces at random, never inside a tag, with where each tag stands."""
    text, spans, cuts, at = "", [], set(), 0
    for spelled, is_tag in tokens:
        if is_tag:
            spans.append((at, at + len(spelled)))
            cuts |= {at, at + len(spelled)}
        else:
            cuts |= {i for i in range(at + 1, at + len(spelled)) if rng.random() < 0.3}
        text += spelled
        at += len(spelled)
    if rng.random() < 0.3:
        cuts = set(rng.sample(sorted(cuts), len(cuts) // 3))   # a merging cut: tags inside pieces
    ends = sorted(cuts - {0, len(text)})
    return [text[i:j] for i, j in zip([0, *ends], [*ends, len(text)])], spans


@pytest.mark.parametrize("opening, closing", [("<thought>", "</thought>"), ("¶", "§")])
@pytest.mark.parametrize("stop_reason", ["end_of_turn", "token_limit"])
@pytest.mark.parametrize("start", ["", "Hi"])
def test_the_answer_and_thought_do_not_depend_on_how_the_stream_is_cut_into_pieces(
        opening, closing, stop_reason, start):
    """Letters of a tag, a tag right after them, and whitespace before a thought, wherever the cuts fall."""
    tags = Reasoning(opening, closing, "")
    tokens = [(start, False), (" ", False), ("\n", False), (opening, True), ("plan", False),
              (closing, True), ("Write " + opening[:-1], False), (opening, True), ("more", False),
              (closing, True), (f"Spell {opening} twice: {opening}", False), (opening, True),
              (f"aside {opening} here", False), (closing, True), ("done", False), (" ", False),
              (opening, True), ("brief", False), (closing, True), (f"end {opening}", False)]
    answer = (f"{start} \n" if start else "") \
        + f"Write {opening[:-1]}Spell {opening} twice: {opening}done end {opening}"
    rng = random.Random(11)
    for _ in range(300):
        pieces, spans = _stream_cut_anywhere_but_inside_a_tag([t for t in tokens if t[0]], rng)
        known_spans, fed, shown = [], 0, []
        reply = ReplyText(tags, "", None, "", [], known_spans)
        for piece in pieces:
            known_spans.extend(s for s in spans if fed <= s[0] < fed + len(piece))
            fed += len(piece)
            shown.append(reply.feed(piece))
        shown.append(reply.finish(stop_reason))
        assert "".join(shown) == reply.answer == answer, pieces
        assert reply.thought == f"planmoreaside {opening} herebrief", pieces


@pytest.mark.parametrize("stop_reason, answer", [("end_of_turn", " \n "), ("token_limit", ""),
                                                 ("interrupted", "")])
def test_a_reply_of_whitespace_alone_keeps_all_of_it_where_it_ended_of_its_own_accord(
        stop_reason, answer):
    tags = Reasoning("<thought>", "</thought>", "")
    reply = ReplyText(tags, "", None, "", [])
    assert reply.feed(" ") + reply.feed("\n") + reply.feed(" ") + reply.finish(stop_reason) == answer
    reply = ReplyText(tags, "", None, "", [])
    assert reply.feed(" ") + reply.feed("\n") + reply.feed("x") + reply.finish(stop_reason) == " \nx"


def test_an_answer_opening_written_in_tags_and_text_after_a_possible_thought_is_not_shown():
    tags = Reasoning("<|ch|>think<|msg|>", "<|stop_ch|>", "")
    opening = "<|ch|>final<|msg|>"
    header = reply_header(opening, tags, ("<|ch|>", "<|msg|>", "<|stop_ch|>"))
    spans = []
    reply = ReplyText(tags, opening, header, "", [], spans)
    shown = _fed_as_tokens(reply, [("<|ch|>", True), ("final", False), ("<|msg|>", True),
                                   ("Antwort.", False)], spans)
    assert shown == reply.answer == "Antwort."


def test_the_thought_count_is_the_number_of_pieces_that_hold_part_of_a_thought():
    tags = Reasoning("<thought>", "</thought>", "", "")
    text = "a<thought>x</thought>b<thought>y</thought>c<tho"
    spans = [(match.start(), match.end()) for match in re.finditer("<thought>.*?</thought>", text)]
    rng = random.Random(11)
    for _ in range(2000):
        cuts = sorted(rng.sample(range(1, len(text)), rng.randint(1, 12)))
        bounds = list(zip([0, *cuts], [*cuts, len(text)]))
        reply = ReplyText(tags, "", None, "", [])
        for i, j in bounds:
            reply.feed(text[i:j])
        reply.finish("end_of_turn")
        holding = sum(any(i < stop and start < j for start, stop in spans) for i, j in bounds)
        assert reply.thought_pieces == holding, bounds
