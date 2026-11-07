"""Text into tokens as a render of the whole conversation tokenizes it."""

from __future__ import annotations

import re

from .cache_context import ModelReader


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
            split = reader.split_at_tags(part)
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


def tokenize_message(reader: ModelReader, prefix: str, content: str, suffix: str,
                     add_special: bool = False) -> list[int]:
    """One templated message: the template's ``prefix`` and ``suffix`` around the caller's ``content``."""
    return tokenize_pieces(reader, [(prefix, False), (content, True), (suffix, False)],
                           add_special=add_special)
