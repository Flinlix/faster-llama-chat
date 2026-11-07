"""Cache contexts over real chat templates, for tests: a renderer, tags as a vocabulary holds them, and a cache.

The renderer is the one a model is loaded with; only the tokenizer is a
stand-in, one token per character and one per tag, so what is compared is the
text.
"""

import re

from llama_chat.addon_files import addon_file
from llama_chat.cache_context import GenerationAccumulator
from llama_chat.model_template import ModelTemplate


class TemplateContext:
    """A cache context reduced to rendering through one real template."""

    def __init__(self, fixture: dict) -> None:
        self._model = ModelTemplate(fixture["chat_template"], fixture["eos_token"])
        self._template = fixture["chat_template"]

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None, **variables):
        return self._model.render(messages, tools, generation_prompt=generation_prompt,
                                  reasoning=reasoning, **variables)

    def chat_template_text(self):
        return self._template

    def addon_call(self):
        # The shipped add-on file that names this template, as a model's load finds it.
        return addon_file(None, self._template)["call"]

    def tokenize(self, text, add_special=False, parse_special=True):
        return [ord(c) for c in text]

    def tags(self):
        # As a vocabulary holds them: every tag-shaped text, and what each closer closes.
        found = set(TAG.findall(self._template))
        return tuple(sorted(found | {"<" + tag[2:] for tag in found if tag.startswith("</")}))

    def extract_fragments(self):
        return self._model.extract_fragments()

    def retokenized(self, tokens):
        return list(tokens)

    def split_at_tags(self, text):
        pieces, at = [], 0
        for found in TAG.finditer(text):
            if found.start() > at:
                pieces.append((text[at:found.start()], False))
            pieces.append((found.group(), True))
            at = found.end()
        return pieces + ([(text[at:], False)] if at < len(text) else [])

    def tokenizes_to_tags(self, text):
        pieces = self.split_at_tags(text.strip())
        return bool(pieces) and all(is_tag for _, is_tag in pieces)

    def first_special_token(self, text):
        return None             # no vocabulary, so no call tag resolves

    def token_text(self, token):
        return ""


TAG = re.compile(r"<\|[^<>\s]*?\|?>|<[^<>\s|]+\|>|<[/a-z_]+(?::[a-z0-9_]+)?>|\[[A-Z_]+\]|<｜[^｜]+｜>|\]~!?[a-z][\[\]]|\[[a-z]~\[")


class TaggedContext(TemplateContext):
    """The same, with tags single tokens.

    A stand-in for a vocabulary that holds every tag-shaped text, such as one
    closing each argument of a call, and the tags that close those, spelled as
    tags are paired (``<x>`` and ``</x>``, ``<|x>`` and ``<x|>``, ``start`` and
    ``end``). The earliest of them in a text is its first tag.
    """

    def __init__(self, fixture: dict) -> None:
        super().__init__(fixture)
        self._tags: list[str] = []

    def first_special_token(self, text):
        # The earliest tag in the text, as a vocabulary finds it.
        known = [(text.index(tag), tag) for tag in self._tags if tag in text]
        found = TAG.search(text)
        if found is not None and found.group() not in self._tags:
            known.append((found.start(), found.group()))
        if not known:
            return None
        tag = min(known)[1]
        if tag not in self._tags:
            self._tags += [tag] + [closer for closer in closers(tag)
                                   if closer not in self._tags]
        return self._id(tag)

    def _id(self, tag):
        return -1 - self._tags.index(tag)

    def token_text(self, token):
        return self._tags[-1 - token] if token < 0 else chr(token)


class TagTokenizingContext(TaggedContext):
    """The same, with each tag tokenized as the one token it resolves to."""

    def tokenize(self, text, add_special=False, parse_special=True):
        tokens, pos = [], 0
        for found in TAG.finditer(text):
            tokens += [ord(c) for c in text[pos:found.start()]]
            tokens.append(self.first_special_token(found.group()))
            pos = found.end()
        return tokens + [ord(c) for c in text[pos:]]


def closers(tag: str) -> list[str]:
    """The tags that would close ``tag``."""
    inner = tag.strip("<>[]")
    return [f"</{inner}>", f"<{inner.lstrip('|')}|>", tag.replace("start", "end"),
            tag.replace("<｜", "</｜", 1)]


class ConversationContext(TagTokenizingContext):
    """The same, with a cache a wrapper writes a whole conversation into.

    Each generation writes the next reply of ``replies``, one token per
    character, and first hands ``before_reply`` the cache's text, so a test can
    compare it with the template's render of the conversation at that moment.
    """

    def __init__(self, fixture: dict, replies: list[str], can_shift: bool = True) -> None:
        super().__init__(fixture)
        self.cache: list[int] = []
        self.can_shift = can_shift
        self.can_cut_tail = can_shift
        self.context_size = 1 << 20
        self.resets = 0             # how often the whole cache was emptied
        self.edits: list[tuple[str, int]] = []   # ("cut", position) and ("shift", tokens), in order
        self._replies = list(replies)
        self.before_reply = None

    def text_of(self, tokens) -> str:
        return "".join(self.token_text(token) for token in tokens)

    def reset(self):
        self.cache = []
        self.resets += 1

    def warmup(self):
        pass

    def prefill(self, token_ids, start_pos, want_logits):
        assert start_pos == len(self.cache), (start_pos, len(self.cache))
        self.cache.extend(token_ids)

    def save_state(self):
        return list(self.cache)

    def restore_state(self, state, pos):
        assert len(state) == pos
        self.cache = list(state)

    def remove_from(self, pos):
        assert self.can_cut_tail and 0 <= pos <= len(self.cache)
        self.edits.append(("cut", pos))
        del self.cache[pos:]

    def apply_eviction(self, ev):
        assert self.can_shift and len(self.cache) == ev.old_total
        self.edits.append(("shift", ev.shift_delta))
        del self.cache[ev.remove_start:ev.remove_end]

    def generate(self, start_pos, max_generated_tokens, out=None, cancel=None,
                 sampling=None, ends_turn=None, boundary=None):
        assert start_pos == len(self.cache)
        if self.before_reply is not None:
            self.before_reply(self.text_of(self.cache))
        generated = out if out is not None else GenerationAccumulator()
        if isinstance(self._replies[0], ScriptedCall):
            return (yield from self._generate_call(generated, boundary))
        reply = self._replies.pop(0)[:max(0, max_generated_tokens)]
        for ch in reply:
            self.cache.append(ord(ch))
            generated.token_ids.append(ord(ch))
            generated.text += ch
            yield ch
        generated.stop_reason = "end_of_turn"
        return generated

    def _generate_call(self, generated, boundary):
        """Write the scripted call as the model would: its opening, its body, its closing."""
        call = self._replies.pop(0)
        assert boundary is not None and (boundary.tool_tokens or boundary.call_texts), \
            "a call was scripted where calls are withheld"
        if boundary.tool_tokens:
            opening, closing = ([token] for token in boundary.tool_tokens)
        else:
            opening, closing = ([ord(c) for c in text] for text in boundary.call_texts)
        for token in opening + [ord(c) for c in call.body] + closing:
            self.cache.append(token)
            generated.token_ids.append(token)
        generated.stop_reason, generated.tool_call = "tool_call", call.body
        return generated
        yield

    def close(self):
        pass


class ScriptedCall:
    """A call the conversation context writes in place of a reply, ``body`` between its tags."""

    def __init__(self, body: str) -> None:
        self.body = body


class HonestTagContext(ConversationContext):
    """The same, tokenizing a tag as its token only where special-token parsing is on.

    Text tokenized with parsing off, as a caller's own is, keeps a tag it
    contains as its characters, so a tag typed into a message cannot become
    the model's own.
    """

    def tokenize(self, text, add_special=False, parse_special=True):
        if not parse_special:
            return [ord(c) for c in text]
        return super().tokenize(text, add_special, parse_special)
