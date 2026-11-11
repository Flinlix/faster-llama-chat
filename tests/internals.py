"""What tests read of a wrapper, or set in it, that no public member offers, here and nowhere else.

Each function reads or sets one thing in plain values, so that an internal
change that keeps the behaviour changes this module alone.
"""

import dataclasses

from llama_chat.resolved_template import ResolvedTemplate


# ----- the tokens the wrapper holds the cache to be ----------------------------------
def held_tokens(wrapper) -> list[int]:
    """The tokens the wrapper holds the cache to be; ``total_tokens`` is their count."""
    return list(wrapper._conversation._memory.tokens)


def overwrite_held_tokens(wrapper, tokens: list[int]) -> None:
    """Have the wrapper hold the cache to be ``tokens``, by no change it would notice."""
    wrapper._conversation._memory.tokens[:] = tokens


def leave_cache_behind(wrapper, tokens: list[int]) -> None:
    """Have the wrapper hold the cache to be ``tokens``, the cache left for later as an action may leave it."""
    wrapper._conversation._memory.replace(tokens)


def bring_cache_back(wrapper) -> None:
    """Bring the cache back where a failed decode, edit or generation left it unknown, as the next action would."""
    memory = wrapper._conversation._memory
    if not memory.is_known:
        memory.repair()


# ----- the wrapper and the conversation as it is rendered ------------------------------
def context_of(wrapper):
    """The cache context the wrapper runs on, the one it built where it was given none."""
    return wrapper._context


def is_closed(wrapper) -> bool:
    return wrapper._closed


def is_begun(wrapper) -> bool:
    return wrapper._conversation.begun


def rendered_messages(wrapper) -> list[dict]:
    """The messages the conversation is rendered from, the system prompt's included."""
    return [dict(message) for message in wrapper._conversation.messages]


def declarations(wrapper) -> list[dict] | None:
    """The tools declared to the conversation, as a render is given them; ``None`` for none."""
    return [spec.as_declaration() for spec in wrapper._conversation.tools] or None


def reasons(wrapper) -> bool:
    """Whether the conversation was begun reasoning."""
    return wrapper._conversation.reasoning


def kept_turns(wrapper) -> list[tuple[list[int], str]]:
    """Each turn kept as generated: its tokens, and the text of the message it ends with."""
    return [(list(kept.tokens), (kept.last.get("content") or "").strip())
            for kept in wrapper._conversation._state.kept]


def kept_turn_places(wrapper) -> list[tuple[int, int] | None]:
    """Where the template writes each kept turn, in each render that splits, with the generation prompt and without."""
    conversation = wrapper._conversation
    messages = conversation.messages
    places = []
    for generation_prompt in (False, True):
        split = conversation._rendering.split(messages, generation_prompt)
        if split is not None:
            places += [conversation._rendering.place(kept, messages, split)
                       for kept in conversation._state.kept]
    return places


def renders_in_windows(wrapper) -> bool:
    """Whether windows of the conversation stand for its whole render."""
    return wrapper._conversation._state.windows


def last_spliced_question(wrapper) -> str | None:
    """The question last decoded in the stand-in's place, ``None`` where none was."""
    spliced_at = wrapper._conversation._state.spliced_at
    return None if spliced_at is None else spliced_at[2].messages[-1]["content"]


def prompt_text(wrapper) -> str:
    """The template text the newest generation followed."""
    return wrapper._conversation.prompt_text


def pending_calls_held(wrapper) -> list:
    """The calls the wrapper itself holds as owed, not the copies ``pending_calls`` hands out."""
    return wrapper._conversation.pending_calls


# ----- a template read otherwise than it writes ---------------------------------------
def read_fragments_as(monkeypatch, **changes) -> None:
    """Have every template's fragments read with ``changes``, as no template can make them read."""
    read = ResolvedTemplate._resolve_fragments
    monkeypatch.setattr(ResolvedTemplate, "_resolve_fragments",
                        lambda self: dataclasses.replace(read(self), **changes))
