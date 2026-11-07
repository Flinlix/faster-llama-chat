"""Cache contexts over real chat templates, for tests: a renderer, tags as a vocabulary holds them, and a cache.

The renderer is the one a model is loaded with; only the tokenizer is a
stand-in, one token per character and one per tag, so what is compared is the
text.
"""

import re

from llama_chat._backend import Backend
from llama_chat.addon_files import addon_file
from llama_chat.context import GenerationAccumulator
from llama_chat.template_extract import reasoning_switch


class TemplateContext:
    """A cache context reduced to rendering through one real template."""

    def __init__(self, fixture: dict) -> None:
        backend = Backend()
        self._template = fixture["chat_template"]
        self._renderers = {prompted: backend.template_renderer(
            fixture["chat_template"], fixture["eos_token"], prompted)
            for prompted in (False, True)}
        self._switch = reasoning_switch(lambda messages, generation_prompt=False, **variables:
                                        self._renderers[generation_prompt](
                                            messages=messages, **variables).prompt)

    def render_with_model_template(self, messages, tools=None, *,
                                   generation_prompt=False, reasoning=None, **variables):
        try:
            return self._renderers[generation_prompt](
                messages=messages, tools=tools, **self._switch(reasoning), **variables).prompt
        except Exception:
            return None

    def chat_template_text(self):
        return self._template

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
        from llama_chat.template_extract import extract_fragments
        return extract_fragments(self.render_with_model_template)

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
        self._replies = list(replies)
        self.before_reply = None

    def text_of(self, tokens) -> str:
        return "".join(self.token_text(token) for token in tokens)

    def reset(self):
        self.cache = []

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
        del self.cache[pos:]

    def apply_eviction(self, ev):
        assert self.can_shift and len(self.cache) == ev.old_total
        del self.cache[ev.remove_start:ev.remove_end]

    def generate(self, start_pos, max_generated_tokens, stop, out=None, cancel=None,
                 tool_tokens=None, sampling=None, ends_turn=None, call_texts=None,
                 turn_end_text=None, call_opens_as_text=False):
        assert start_pos == len(self.cache)
        if self.before_reply is not None:
            self.before_reply(self.text_of(self.cache))
        generated = out if out is not None else GenerationAccumulator()
        reply = self._replies.pop(0)[:max(0, max_generated_tokens)]
        for ch in reply:
            self.cache.append(ord(ch))
            generated.token_ids.append(ord(ch))
            generated.text += ch
            yield ch
        generated.stop_reason = "end_of_turn"
        return generated

    def close(self):
        pass
