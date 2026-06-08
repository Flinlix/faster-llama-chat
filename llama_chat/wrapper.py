"""ChatWrapper — the public API: ``begin``, ``inject``, ``request``.

The wrapper's goal is to prefill as little as possible. ``begin`` resets and
prefills the system prompt plus whatever recent history fits; ``inject`` prefills
a single message with no generation; ``request`` prefills only the new request
text and then generates. When the cache crosses the eviction threshold the oldest
non-system messages are removed and the survivors shifted down to close the gap,
so their KV is reused without re-prefilling.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from .config import Config
from .context import GenerationAccumulator, KVContext
from .messages import Message, MessageTable, fit_newest_first
from .template import TemplateFormatter


class ContextOverflowError(RuntimeError):
    """A request would leave fewer than ``min_answer_tokens`` free for the reply."""


@dataclass
class Turn:
    """What a ``request`` produced.

    Attributes:
        text: The assistant's reply (trimmed at any stop string).
        n_prefilled: Tokens decoded for the request prompt (request text + the
            assistant-open tag).
        n_generated: Tokens sampled.
        n_evicted: Messages dropped to make room for this turn.
        stop_reason: Why generation stopped (``"eog"``/``"stop"``/``"length"``).
    """

    text: str
    n_prefilled: int
    n_generated: int
    n_evicted: int
    stop_reason: str


class ChatWrapper:
    """Stateful single-conversation chat over llama.cpp with KV reuse."""

    def __init__(self, config: Config = Config(), context: KVContext | None = None) -> None:
        # ``context`` is injectable for testing only.
        self._ctx = context if context is not None else KVContext(config)
        self._cfg = config
        self._validate_template(self._cfg)
        self._fmt = TemplateFormatter(self._cfg)
        if self._cfg.validate_against_model_template:
            self._check_fragments_match_model()
        self._table = MessageTable()
        # Serializes all cache-mutating actions: one KV sequence is shared across
        # turns, so concurrent decodes (e.g. a threaded HTTP server) would corrupt
        # it. Reentrant so request() can hold it across the stream() it drains.
        self._lock = threading.RLock()

    # ----- introspection -------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self._table.total

    def snapshot(self) -> list[dict]:
        """Return the current cache layout (for inspection and tests)."""
        return [
            {
                "role": m.role,
                "n_tokens": m.n_tokens,
                "pos_start": m.pos_start,
                "pos_end": m.pos_end,
            }
            for m in self._table.messages
        ]

    # ----- action: begin -------------------------------------------------
    def begin(self, system_prompt: str, messages: list[tuple[str, str]] | None = None) -> None:
        """Reset the conversation: prefill the system prompt and recent history.

        Args:
            system_prompt: The system prompt (always kept, never evicted).
            messages: Optional ``(role, text)`` history, oldest first. Only the
                most recent messages that fit under the threshold alongside the
                system prompt are prefilled; older excess is dropped.
        """
        messages = messages or []
        with self._lock:
            self._ctx.reset()
            self._table.reset()

            sys_tokens = self._ctx.tokenize(self._fmt.fragment("system", system_prompt), add_special=True)
            hist_tokens = [
                self._ctx.tokenize(self._fmt.fragment(role, text), add_special=False)
                for role, text in messages
            ]

            kept = fit_newest_first(
                [len(t) for t in hist_tokens], len(sys_tokens), self._cfg.threshold_tokens
            )
            keep_from = len(messages) - kept
            kept_messages = messages[keep_from:]
            kept_tokens = hist_tokens[keep_from:]

            self._check_template_equivalence(system_prompt, kept_messages, sys_tokens, kept_tokens)

            # One decode for the whole conversation == cheapest possible prefill.
            all_tokens = list(sys_tokens)
            for toks in kept_tokens:
                all_tokens.extend(toks)
            self._ctx.prefill(all_tokens, start_pos=0, want_logits=False)

            self._table.append(Message("system", system_prompt, sys_tokens))
            for (role, text), toks in zip(kept_messages, kept_tokens):
                self._table.append(Message(role, text, toks))

    # ----- action: inject ------------------------------------------------
    def inject(self, text: str, role: str = "user") -> int:
        """Prefill one message as context without generating.

        Evicts the oldest messages until the new message fits under the
        threshold, then prefills it.

        Returns:
            The number of messages evicted to make room.
        """
        with self._lock:
            tokens = self._ctx.tokenize(self._fmt.fragment(role, text), add_special=False)
            evicted = self._evict_until(lambda: self._table.total + len(tokens) <= self._cfg.threshold_tokens)
            tokens = self._fit_or_raise(
                tokens, budget=self._cfg.threshold_tokens - self._table.total, what="injected message"
            )

            self._ctx.prefill(tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message(role, text, tokens))
            return evicted

    # ----- action: request / stream --------------------------------------
    def stream(
        self, text: str, *, max_tokens: int | None = None, stop: list[str] | None = None
    ):
        """Add a user request and stream the reply token-by-token.

        Yields visible text deltas as they are generated. The generator's ``return``
        value is the :class:`Turn` summary, available via ``StopIteration.value``
        or by calling :meth:`request`, which wraps this method.

        Barge-in is safe: if the consumer stops early (``gen.close()``), the
        ``finally`` block still terminates the assistant turn and records exactly
        the tokens that reached the cache, so the next turn's cache stays valid.

        Args:
            text: The user request text.
            max_tokens: Override for the per-turn generation cap.
            stop: Extra stop strings for this request (added to the config's).

        Yields:
            Reply text deltas.

        Returns:
            A :class:`Turn` describing the reply and the work performed.
        """
        with self._lock:
            user_tokens = self._ctx.tokenize(self._fmt.fragment("user", text), add_special=False)
            open_tokens = self._ctx.tokenize(self._fmt.assistant_open(), add_special=False)
            close_tokens = self._ctx.tokenize(self._fmt.assistant_close(), add_special=False)
            n_prompt = len(user_tokens) + len(open_tokens)

            # Best-effort: drop the oldest until we're back under threshold.
            evicted = self._evict_until(
                lambda: self._table.total + n_prompt <= self._cfg.threshold_tokens
            )
            # Hard wall: the prompt plus its turn closer must fit in the context.
            user_tokens = self._fit_or_raise(
                user_tokens,
                budget=self._cfg.n_ctx - len(open_tokens) - len(close_tokens) - self._table.total,
                what="request",
            )
            n_prompt = len(user_tokens) + len(open_tokens)

            # Refuse before mutating the cache if too little room remains to reply.
            free = self._cfg.n_ctx - self._table.total - n_prompt - len(close_tokens)
            if free < self._cfg.min_answer_tokens:
                raise ContextOverflowError(
                    f"only {free} tokens free for the reply "
                    f"(min_answer_tokens={self._cfg.min_answer_tokens}); shorten "
                    f"the request or lower threshold_pct"
                )

            self._ctx.prefill(user_tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message("user", text, user_tokens))

            gen_start = self._table.total + len(open_tokens)
            self._ctx.prefill(open_tokens, start_pos=self._table.total, want_logits=True)

            budget = self._cfg.n_ctx - gen_start - len(close_tokens)
            cap = max_tokens if max_tokens is not None else self._cfg.max_tokens
            n_predict_max = max(0, min(cap, budget))
            stops = list(self._cfg.stop) + list(stop or [])

            gen = GenerationAccumulator()
            try:
                yield from self._ctx.generate(gen_start, n_predict_max, stops, out=gen)
            finally:
                # Runs on normal completion *and* on early close (barge-in).
                close_start = gen_start + len(gen.token_ids)
                self._ctx.prefill(close_tokens, start_pos=close_start, want_logits=False)
                assistant_tokens = open_tokens + gen.token_ids + close_tokens
                self._table.append(Message("assistant", gen.text, assistant_tokens))

            return Turn(
                text=gen.text,
                n_prefilled=n_prompt,
                n_generated=len(gen.token_ids),
                n_evicted=evicted,
                stop_reason=gen.stop_reason,
            )

    def request(
        self, text: str, *, max_tokens: int | None = None, stop: list[str] | None = None
    ) -> Turn:
        """Add a user request and generate a reply, reusing all existing context.

        Convenience wrapper that drains :meth:`stream` and returns its summary.

        Returns:
            A :class:`Turn` describing the reply and the work performed.
        """
        gen = self.stream(text, max_tokens=max_tokens, stop=stop)
        try:
            while True:
                next(gen)
        except StopIteration as done:
            return done.value

    def close(self) -> None:
        self._ctx.close()

    # ----- internals -----------------------------------------------------
    def _validate_template(self, config: Config) -> None:
        """Ensure the turn-terminator is a real special token of this model.

        A template whose terminator splits into plain-text pieces (e.g. ChatML
        tags on a Gemma model) never lets the model emit end-of-generation, so it
        runs to the token cap every turn. Detecting that here lets users fix their
        template instead of silently generating bad outputs.
        """
        # FakeContext (tests) has no special-token vocabulary; skip gracefully.
        if not hasattr(self._ctx, "tokenizes_to_special"):
            return
        terminator = config.assistant_suffix.strip()
        if terminator and not self._ctx.tokenizes_to_special(terminator):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not tokenize to a special token, so "
                "generation would never stop. Set the assistant_prefix/"
                "assistant_suffix (and other *_prefix/*_suffix) fields on Config "
                "to match the model's trained template."
            )

    def _check_fragments_match_model(self) -> None:
        """Assert the user/assistant fragments match the model's trained template.

        The hand-configured ``*_prefix``/``*_suffix`` tags must reproduce the tags
        the model saw in training, or incremental prefill writes tokens it never
        learned. We render a fixed user/assistant probe through the model's own
        chat template (from the GGUF) and compare its tokenization to the fragment
        render. The system fragment is *not* probed: chat templates handle a
        system role inconsistently (Gemma rejects it, others fold it into the
        first user turn), so there is no portable ground truth for it.
        """
        # FakeContext (tests) and GGUFs without a template have nothing to check.
        if not hasattr(self._ctx, "render_with_model_template"):
            return
        # Surrounding whitespace makes the probe also catch a wrong `trim_content`
        # setting: a template that applies `| trim` collapses it, a verbatim one
        # keeps it, and the two renders diverge unless the flag matches.
        probe = [
            {"role": "user", "content": " ping "},
            {"role": "assistant", "content": " pong "},
        ]
        rendered = self._ctx.render_with_model_template(probe)
        if rendered is None:
            return
        model_tokens = self._ctx.tokenize(rendered, add_special=False)
        fragment_text = "".join(self._fmt.fragment(m["role"], m["content"]) for m in probe)
        fragment_tokens = self._ctx.tokenize(fragment_text, add_special=False)
        if model_tokens != fragment_tokens:
            raise ValueError(
                "chat template fragments do not match the model's trained "
                "template:\n"
                f"  model renders:    {rendered!r}\n"
                f"  fragments render: {fragment_text!r}\n"
                "Set the user_prefix/user_suffix/assistant_prefix/assistant_suffix "
                "(and the matching system_* fields) on Config to the tags shown in "
                "the model's render above, or pass "
                "validate_against_model_template=False to skip this check."
            )

    def _evict_until(self, fits) -> int:
        """Evict the oldest non-system messages until ``fits()`` or none remain.

        Two strategies, chosen by the cache's capabilities:

        * **shift** (default) — each eviction removes the oldest message's span
          and shifts the survivors down in place, reusing their KV for free.
          Applied incrementally as messages are dropped.
        * **rebuild** (caches that can't shift, e.g. compact SWA / recurrent) —
          drop the oldest from the bookkeeping only, then re-prefill the whole
          surviving conversation once. Correct, but not free; this is the expensive
          fallback for models where in-place shifting is unsupported.
        """
        if self._ctx.can_shift:
            evicted = 0
            while not fits() and self._table.n_evictable > 0:
                self._ctx.apply_eviction(self._table.evict_oldest())
                evicted += 1
            return evicted

        # Rebuild path: decide survivors first (bookkeeping only), then re-prefill.
        evicted = 0
        while not fits() and self._table.n_evictable > 0:
            self._table.evict_oldest()  # renumbers the table; cache untouched yet
            evicted += 1
        if evicted:
            self._rebuild_cache()
        return evicted

    def _rebuild_cache(self) -> None:
        """Re-prefill the cache from the surviving messages (no in-place shift)."""
        self._ctx.reset()
        all_tokens: list[int] = []
        for m in self._table.messages:
            all_tokens.extend(m.token_ids)
        self._ctx.prefill(all_tokens, start_pos=0, want_logits=False)

    def _fit_or_raise(self, tokens: list[int], budget: int, what: str) -> list[int]:
        """Enforce the size policy when a message still doesn't fit."""
        if len(tokens) <= budget:
            return tokens
        if self._cfg.oversize_policy == "truncate" and budget > 0:
            return tokens[:budget]
        raise ValueError(
            f"{what} needs {len(tokens)} tokens but only {budget} fit "
            f"(oversize_policy='{self._cfg.oversize_policy}')"
        )

    def _check_template_equivalence(self, system, kept_messages, sys_tokens, kept_tokens) -> None:
        """Assert per-message prefill yields the same tokens as a one-shot render.

        If this fails, the template tokenizes differently across message
        boundaries and incremental ``inject``/``request`` prefill would diverge
        from a full render — better to fail loudly at ``begin`` than corrupt the
        cache silently later.
        """
        whole = self._ctx.tokenize(self._fmt.full_conversation(system, kept_messages), add_special=True)
        piecewise = list(sys_tokens)
        for toks in kept_tokens:
            piecewise.extend(toks)
        if whole != piecewise:
            raise ValueError(
                "chat template is not safe for per-message prefill: tokenization "
                "differs across message boundaries. Adjust the template fragments "
                "on Config so each fragment tokenizes independently."
            )
