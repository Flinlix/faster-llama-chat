"""Text into tokens as a render of the whole conversation tokenizes it, and the model's own tags found in a text."""

from __future__ import annotations

import re

from .cache_context import ModelReader


def split_at_tags(reader: ModelReader, text: str) -> list[tuple[str, bool]] | None:
    """``text`` as the model's tags and the plain text between them, each marked whether a tag.

    ``None`` where a tag the tokenizer finds cannot be located in the text.
    """
    pieces, at = [], 0
    for token in reader.tokenize(text, parse_special=True):
        if not reader.is_special(token):
            continue
        tag = reader.token_text(token)
        start = text.find(tag, at)
        if not tag or start < 0:
            return None
        if start > at:
            pieces.append((text[at:start], False))
        pieces.append((tag, True))
        at = start + len(tag)
    if at < len(text):
        pieces.append((text[at:], False))
    return pieces


def first_special_token(reader: ModelReader, text: str) -> int | None:
    """The first of the model's own tags in ``text``, or ``None``.

    The first tag, not the first token: a recovered tool tag carries fixed
    lead-in text.
    """
    return next((token for token in reader.tokenize(text, add_special=False, parse_special=True)
                 if reader.is_special(token)), None)


def tag_texts(reader: ModelReader) -> tuple[str, ...]:
    """The texts of every one of the model's own tags."""
    return tuple(text for text in (reader.token_text(token) for token in reader.special_tokens())
                 if text)


def tokenize_pieces(reader: ModelReader, pieces: list[tuple[str, bool]], *,
                    add_special: bool = False,
                    tag_tokens: dict[str, int] | None = None) -> list[int]:
    """``pieces`` as one render tokenizes them.

    The model's own tags are tokenized as tags, and the plain text between two
    tags as one run with special-token parsing off, whether it is the
    template's or the caller's. See the README, "Special tokens in message
    content".

    Args:
        reader: The model's tokenizer and tags.
        pieces: The text in order, each marked whether it is the caller's own,
            which is never read as a tag, rather than the template's.
        add_special: Attach the model's sequence opener to the first token.
        tag_tokens: Tags written as one given token wherever the template's
            text holds them, such as the tags around a generated call.
    """
    tokens: list[int] = []
    plain = ""                  # the text between two tags, own and template alike
    opens = add_special

    def tokenize(text: str, parse_special: bool) -> None:
        nonlocal opens
        tokens.extend(reader.tokenize(text, add_special=opens, parse_special=parse_special))
        opens = False

    def end_plain() -> None:
        nonlocal plain
        if plain:
            tokenize(plain, parse_special=False)
        plain = ""

    given = re.compile("(" + "|".join(map(re.escape, tag_tokens)) + ")") if tag_tokens else None
    for text, own in pieces:
        if own:
            plain += text
            continue
        for part in (given.split(text) if given else [text]):
            if not part:
                continue
            if tag_tokens and part in tag_tokens:
                end_plain()
                if opens:
                    tokenize("", parse_special=True)    # the sequence opener alone
                tokens.append(tag_tokens[part])
                continue
            split = split_at_tags(reader, part)
            if split is None:           # a tag the tokenizer finds but cannot place
                end_plain()
                tokenize(part, parse_special=True)
                continue
            for piece, is_tag in split:
                if is_tag:
                    end_plain()
                    tokenize(piece, parse_special=True)
                else:
                    plain += piece
    end_plain()
    if opens:
        tokenize("", parse_special=True)        # nothing else: the opener alone
    return tokens


class MemoTokenizer:
    """A model's tokenizer that remembers, up to a bound, what it tokenized."""

    _BOUND = 20000                  # entries kept before the memory starts over

    def __init__(self, reader: ModelReader) -> None:
        self._reader = reader
        self._known: dict = {}

    def tokenize(self, text, add_special=False, parse_special=True):
        key = (text, add_special, parse_special)
        if key not in self._known:
            if len(self._known) > self._BOUND:
                self._known.clear()
            self._known[key] = self._reader.tokenize(text, add_special=add_special,
                                                     parse_special=parse_special)
        return self._known[key]

    def token_text(self, token):
        return self._reader.token_text(token)

    def is_special(self, token):
        return self._reader.is_special(token)
