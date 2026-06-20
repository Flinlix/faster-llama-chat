"""ChatWrapper - the public API: ``begin``, ``inject``, ``request``, ``stream``.

The wrapper's goal is to prefill as little as possible. ``begin`` resets and
prefills the system prompt plus whatever recent history fits (and on the first
call warms the generation graph so the first reply has no cold-start latency);
``inject`` prefills a single message with no generation; ``request`` and
``stream`` prefill only the new request text and then generate. When the cache
crosses the eviction threshold the oldest non-system messages are removed and
the survivors shifted down to close the gap, so their KV is reused without
re-prefilling.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from dataclasses import dataclass

from .config import Config
from .context import GenerationAccumulator, KVContext
from .messages import Message, MessageTable, fit_newest_first
from .template import Fragments, TemplateFormatter


class ContextOverflowError(RuntimeError):
    """A request would leave fewer than ``min_reply_tokens`` free for the reply."""


@dataclass
class Turn:
    """What a ``request`` produced.

    Attributes:
        text: The assistant's reply (trimmed at any stop string).
        n_prefilled: Tokens decoded for the request prompt (request text + the
            assistant-open tag).
        n_generated: Tokens sampled.
        n_evicted: Messages evicted after this turn to restore the cache to
            ``threshold_tokens`` (the reply is appended first, then older turns
            are trimmed).
        stop_reason: Why generation stopped (``"eog"``/``"stop"``/``"length"``).
    """

    text: str
    n_prefilled: int
    n_generated: int
    n_evicted: int
    stop_reason: str


class ChatWrapper:
    """Stateful single-conversation chat over llama.cpp with KV reuse."""

    def __init__(
        self,
        config: Config | None = None,
        context: KVContext | None = None,
        *,
        fragments: Fragments | None = None,
        on_message: Callable[[str, str], None] | None = None,
        **kwargs,
    ) -> None:
        """Load the model, resolve the chat template, and validate it.

        Args:
            config: Full configuration. Mutually exclusive with ``**kwargs``.
            context: Pre-built cache context; injectable for testing only.
            fragments: Explicit chat-template fragments, for a model that ships
                no embedded template (otherwise they are recovered from the
                model's own GGUF chat template).
            on_message: Invoked with ``(role, text)`` for every message except
                the system prompt and begin-replay - i.e. each genuine new turn
                (user, assistant) and each inject. Runs under the lock, so it
                must be cheap and must not call back into the wrapper.
            **kwargs: :class:`~llama_chat.config.Config` field overrides, used
                to build the config when ``config`` is not given.

        Raises:
            TypeError: If both ``config`` and config ``**kwargs`` are passed.
            ValueError: If the chat template cannot be recovered or fails
                validation against the model.
        """
        if config is None:
            config = Config(**kwargs)
        elif kwargs:
            raise TypeError(
                "pass either a Config or Config field overrides, not both "
                f"(got config and {sorted(kwargs)})"
            )
        self._on_message = on_message
        owns_ctx = context is None
        self._ctx = KVContext(config) if owns_ctx else context
        self._cfg = config
        try:
            # The template comes from the model itself; pass ``fragments`` explicitly
            # only for a model that ships no embedded chat template.
            self._frags = fragments if fragments is not None else self._resolve_fragments()
            self._validate_template(self._frags)
            special = self._ctx.special_token_texts() if hasattr(self._ctx, "special_token_texts") else []
            self._fmt = TemplateFormatter(self._frags, special_tokens=special)
            self._check_fragments_match_model()
        except BaseException:
            # Template validation is a designed failure path; don't leak the
            # model/context created above.
            if owns_ctx:
                self._ctx.close()
            raise
        self._table = MessageTable()
        # The first ``begin`` warms the generation graph once (see ``warmup``).
        self._warmed = False
        self._closed = False
        # Serializes all cache-mutating actions: one KV sequence is shared across
        # turns, so concurrent decodes (e.g. a threaded HTTP server) would corrupt
        # it. ``stream`` acquires it once inside the generator body and holds it
        # until the turn completes (or the stream is closed).
        self._lock = threading.Lock()

    # ----- introspection -------------------------------------------------
    @property
    def total_tokens(self) -> int:
        return self._table.total

    def snapshot(self) -> list[dict]:
        """Return the current cache layout (for inspection and tests).

        Safe to call from any thread, including while another thread is
        mid-turn: the table guards its own reads, so the layout is never torn.
        """
        return self._table.snapshot_rows()

    # ----- action: begin -------------------------------------------------
    def begin(self, system_prompt: str, messages: list[tuple[str, str]] | None = None) -> None:
        """Reset the conversation: prefill the system prompt and recent history.

        The first call also warms the model's single-token generation graph (a
        throwaway decode that is immediately cleared), so the first ``request``
        streams without paying the one-time graph-capture/allocation cost.

        Args:
            system_prompt: The system prompt (always kept, never evicted).
            messages: Optional ``(role, text)`` history, oldest first. Only the
                most recent messages that fit under the threshold alongside the
                system prompt are prefilled; older excess is dropped.

        Raises:
            ValueError: If the system prompt alone exceeds ``threshold_tokens``
                (nothing could ever be evicted to make room for a turn), or if
                the template fails the per-message equivalence check. The
                previous conversation is left intact in either case.
        """
        messages = messages or []
        with self._lock:
            self._ensure_open()
            # Tokenize and validate first: a failed begin must not destroy the
            # conversation it was about to replace.
            sys_tokens = self._ctx.tokenize(self._fmt.fragment("system", system_prompt), add_special=True)
            if len(sys_tokens) > self._cfg.threshold_tokens:
                raise ValueError(
                    f"system prompt needs {len(sys_tokens)} tokens but the "
                    f"eviction threshold is {self._cfg.threshold_tokens}; "
                    "shorten the prompt or raise context_size/eviction_threshold"
                )
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

            self._ctx.reset()
            self._table.reset()
            # First conversation: warm the batch-1 generation graph so the first
            # reply has no cold-start latency. warmup self-clears the cache.
            if not self._warmed:
                self._ctx.warmup()
                self._warmed = True

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

        Raises:
            ValueError: If the message cannot fit even after evicting every
                non-system message (and truncation is disabled or impossible).
                Raised *before* anything is evicted, so a rejected message
                leaves the conversation untouched.
        """
        with self._lock:
            self._ensure_open()
            tokens = self._ctx.tokenize(self._fmt.fragment(role, text), add_special=False)
            # Enforce the size policy against the best case (everything but the
            # system prompt evicted) before evicting anything: a rejected
            # message must not destroy the history it could never fit beside.
            # After this, eviction below is guaranteed to make the message fit.
            floor = self._table.messages[0].n_tokens if self._table.has_system else 0
            tokens = self._fit_or_raise(
                tokens, budget=self._cfg.threshold_tokens - floor,
                what="injected message", role=role,
            )
            evicted = self._evict_until(lambda: self._table.total + len(tokens) <= self._cfg.threshold_tokens)

            self._ctx.prefill(tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message(role, text, tokens))
            if self._on_message:
                self._on_message(role, text)
            return evicted

    # ----- action: request / stream --------------------------------------
    def stream(
        self, text: str, *, max_tokens: int | None = None, stop: list[str] | None = None
    ) -> Generator[str, None, Turn]:
        """Add a user request and stream the reply token-by-token.

        Yields visible text deltas as they are generated. The generator's ``return``
        value is the :class:`Turn` summary, available via ``StopIteration.value``
        or by calling :meth:`request`, which wraps this method.

        The generator is lazy: nothing happens (including the
        :class:`ContextOverflowError` headroom check) until the first iteration.

        Barge-in is safe: if the consumer stops early (``gen.close()``), the
        ``finally`` block still terminates the assistant turn and records exactly
        the tokens that reached the cache, so the next turn's cache stays valid.

        Note:
            ``text`` is sanitized before tokenization: any of the model's
            special-token pieces (e.g. ``<end_of_turn>``) are stripped from the
            content, so untrusted input cannot forge turn boundaries (see the
            README's chat-template section).

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
            self._ensure_open()
            user_tokens = self._ctx.tokenize(self._fmt.fragment("user", text), add_special=False)
            open_tokens = self._ctx.tokenize(self._fmt.assistant_open(), add_special=False)
            close_tokens = self._ctx.tokenize(self._fmt.assistant_close(), add_special=False)

            # Hard wall: the prompt plus its turn closer must fit in the context.
            user_tokens = self._fit_or_raise(
                user_tokens,
                budget=self._cfg.context_size - len(open_tokens) - len(close_tokens) - self._table.total,
                what="request", role="user",
            )
            n_prompt = len(user_tokens) + len(open_tokens)

            # Refuse before mutating the cache if too little room remains to reply.
            free = self._cfg.context_size - self._table.total - n_prompt - len(close_tokens)
            if free < self._cfg.min_reply_tokens:
                raise ContextOverflowError(
                    f"only {free} tokens free for the reply "
                    f"(min_reply_tokens={self._cfg.min_reply_tokens}); shorten "
                    f"the request or lower eviction_threshold"
                )

            self._ctx.prefill(user_tokens, start_pos=self._table.total, want_logits=False)
            self._table.append(Message("user", text, user_tokens))
            if self._on_message:
                self._on_message("user", text)

            gen_start = self._table.total + len(open_tokens)
            self._ctx.prefill(open_tokens, start_pos=self._table.total, want_logits=True)

            budget = self._cfg.context_size - gen_start - len(close_tokens)
            cap = max_tokens if max_tokens is not None else self._cfg.max_tokens
            n_predict_max = max(0, min(cap, budget))
            stops = list(self._cfg.stop_strings) + list(stop or [])

            gen = GenerationAccumulator()
            n_evicted = 0
            try:
                yield from self._ctx.generate(gen_start, n_predict_max, stops, out=gen)
            finally:
                # Runs on normal completion *and* on early close (barge-in).
                close_start = gen_start + len(gen.token_ids)
                self._ctx.prefill(close_tokens, start_pos=close_start, want_logits=False)
                assistant_tokens = open_tokens + gen.token_ids + close_tokens
                self._table.append(Message("assistant", gen.text, assistant_tokens))
                if self._on_message:
                    self._on_message("assistant", gen.text)
                # Trim back under threshold after the reply, so the cache rests
                # below threshold for the next turn (and on barge-in too).
                n_evicted = self._evict_until(
                    lambda: self._table.total <= self._cfg.threshold_tokens
                )

            return Turn(
                text=gen.text,
                n_prefilled=n_prompt,
                n_generated=len(gen.token_ids),
                n_evicted=n_evicted,
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
        """Free the underlying context and model. Idempotent and thread-safe.

        Waits for any in-flight turn (the lock serializes against it), then
        closes the context - including one injected at construction. Every
        action after close raises ``RuntimeError``.
        """
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._ctx.close()

    def __enter__(self) -> ChatWrapper:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ----- internals -----------------------------------------------------
    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("ChatWrapper is closed")

    def _resolve_fragments(self) -> Fragments:
        """Recover the template fragments from the model's own chat template."""
        # (FakeContext (tests) supplies its own fragments via extract_fragments.)
        if not hasattr(self._ctx, "extract_fragments"):
            raise ValueError("context cannot supply template fragments")
        frags = self._ctx.extract_fragments()
        if frags is None:
            raise ValueError(
                "the model ships no chat template, so the fragments cannot be "
                "derived; pass them explicitly with ChatWrapper(fragments=...)"
            )
        return frags

    def _validate_template(self, frags: Fragments) -> None:
        """Ensure the turn-terminator is a real special token of this model.

        A template whose terminator splits into plain-text pieces (e.g. ChatML
        tags on a Gemma model) never lets the model emit end-of-generation, so it
        runs to the token cap every turn. Detecting that here surfaces a bad
        template instead of silently generating bad outputs.
        """
        # FakeContext (tests) has no special-token vocabulary; skip gracefully.
        if not hasattr(self._ctx, "tokenizes_to_special"):
            return
        terminator = frags.assistant_suffix.strip()
        if terminator and not self._ctx.tokenizes_to_special(terminator):
            raise ValueError(
                "chat template is not valid for this model: its turn-terminator "
                f"{terminator!r} does not tokenize to a special token, so "
                "generation would never stop. The model's embedded chat template "
                "is malformed; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _check_fragments_match_model(self) -> None:
        """Assert the user/assistant fragments match the model's trained template.

        The fragments in use must reproduce the tags the model saw in training, or
        incremental prefill writes tokens it never learned. We render a fixed
        user/assistant probe through the model's own chat template (from the GGUF)
        and compare its tokenization to the fragment render - a self-consistency
        check on the recovered (or supplied) fragments. The system fragment is
        *not* probed: chat templates handle a system role inconsistently (Gemma
        rejects it, others fold it into the first user turn), so there is no
        portable ground truth for it.
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
                "The recovered fragments do not round-trip through the model's "
                "template; supply correct tags with ChatWrapper(fragments=...)."
            )

    def _evict_until(self, fits) -> int:
        """Evict the oldest non-system messages until ``fits()`` or none remain.

        Two strategies, chosen by the cache's capabilities:

        * **shift** (default) - remove the dropped span and shift the survivors
          down in place, reusing their KV for free. The contiguous block of
          victims collapses into a single remove-and-shift.
        * **rebuild** (caches that can't shift, e.g. compact SWA / recurrent) -
          drop the oldest from the bookkeeping only, then re-prefill the whole
          surviving conversation once. Correct, but not free; this is the expensive
          fallback for models where in-place shifting is unsupported.
        """
        if self._ctx.can_shift:
            eviction, evicted = self._table.evict_oldest_until(fits)
            if eviction is not None:
                self._ctx.apply_eviction(eviction)  # one seq_rm + one seq_add
            return evicted

        # Rebuild path: decide survivors first (bookkeeping only), then re-prefill.
        _, evicted = self._table.evict_oldest_until(fits)
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

    def _fit_or_raise(self, tokens: list[int], budget: int, what: str, role: str) -> list[int]:
        """Enforce the size policy when a message still doesn't fit.

        Truncation drops content tokens from the end but re-appends the role's
        turn-terminator tag, so a clipped message still closes its turn and the
        cache never holds a malformed fragment.
        """
        if len(tokens) <= budget:
            return tokens
        if self._cfg.oversize_policy == "truncate":
            suffix = self._ctx.tokenize(self._fmt.suffix(role), add_special=False)
            if budget > len(suffix):
                return tokens[: budget - len(suffix)] + suffix
        raise ValueError(
            f"{what} needs {len(tokens)} tokens but only {budget} fit "
            f"(oversize_policy='{self._cfg.oversize_policy}')"
        )

    def _check_template_equivalence(
        self,
        system: str,
        kept_messages: list[tuple[str, str]],
        sys_tokens: list[int],
        kept_tokens: list[list[int]],
    ) -> None:
        """Assert per-message prefill yields the same tokens as a one-shot render.

        If this fails, the template tokenizes differently across message
        boundaries and incremental ``inject``/``request`` prefill would diverge
        from a full render - better to fail loudly at ``begin`` than corrupt the
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
