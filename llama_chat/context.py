"""KV-cache operations: tokenize, prefill, generate, evict-and-shift.

This is the only module that drives llama.cpp directly. It owns the model,
context and sampler, and translates the wrapper's intent into the low-level
calls resolved by :class:`~llama_chat._backend.Backend`. It deliberately knows
nothing about *which* messages to keep — that policy lives in the wrapper and
:mod:`~llama_chat.messages`.
"""

from __future__ import annotations

import codecs
from dataclasses import dataclass, field

from ._backend import Backend
from .config import KV_CACHE_GGML_TYPES, Config
from .messages import Eviction

SEQ = 0  # single active conversation -> single sequence id


class _TextBuffer:
    """Accumulated decoded text with a cursor tracking what has been emitted."""

    def __init__(self) -> None:
        self._buf = ""
        self._emitted = 0

    def append(self, text: str) -> None:
        self._buf += text

    def emit(self, upto: int) -> str | None:
        if upto > self._emitted:
            delta = self._buf[self._emitted:upto]
            self._emitted = upto
            return delta
        return None

    @property
    def text(self) -> str:
        return self._buf

    def __len__(self) -> int:
        return len(self._buf)


@dataclass
class GenerationAccumulator:
    """Live accumulator for one generation, updated as tokens stream in.

    It is filled incrementally by :meth:`KVContext.generate` so that a caller
    which abandons the stream early (barge-in) can still read exactly what
    reached the cache: every token in ``token_ids`` was decoded before its text
    was yielded, so ``token_ids`` and the KV cache never disagree.

    Attributes:
        token_ids: Every token decoded into the cache (the cache truth).
        text: Visible reply text so far, trimmed at the stop string if one hit.
        stop_reason: ``"eog"``, ``"stop"`` or ``"length"``.
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    stop_reason: str = "length"


class KVContext:
    """Owns the llama.cpp model/context and performs all cache edits."""

    def __init__(self, config: Config) -> None:
        self._cfg = config
        self._b = Backend()
        self._model = self._b.load_model(config.model_path, config.n_gpu_layers)
        kv_type = KV_CACHE_GGML_TYPES.get(config.kv_cache_type)
        self._ctx = self._b.new_context(
            self._model, config.n_ctx, config.n_threads, config.n_batch,
            flash_attn=config.flash_attn, type_k=kv_type, type_v=kv_type)
        self._vocab = self._b.vocab(self._model)
        self._n_batch = config.n_batch
        self._can_shift = self._b.can_shift(self._ctx)
        self._sampler = self._build_sampler(config)
        self._model_formatter = self._build_model_formatter()

    @property
    def can_shift(self) -> bool:
        """Whether eviction can shift survivors in place vs. rebuild the cache."""
        return self._can_shift

    # ----- tokenization --------------------------------------------------
    def tokenize(self, text: str, add_special: bool = False) -> list[int]:
        return self._b.tokenize(self._vocab, text, add_special)

    def detokenize(self, token_ids: list[int]) -> str:
        return "".join(self._b.token_to_piece(self._vocab, t) for t in token_ids)

    def tokenizes_to_special(self, text: str) -> bool:
        """True if ``text`` tokenizes to exactly one control/special token.

        Used to validate that a configured turn-terminator tag is a real special
        token of *this* model, rather than being silently split into plain-text
        pieces (which would stop generation from ever terminating).
        """
        toks = self.tokenize(text, add_special=False)
        return len(toks) == 1 and self._b.is_special(self._vocab, toks[0])

    def render_with_model_template(self, messages: list[dict]) -> str | None:
        """Render ``messages`` with the model's own GGUF chat template.

        Returns ``None`` when the model ships no ``tokenizer.chat_template`` or
        the template rejects this message shape (e.g. Gemma refusing a system
        role) — callers treat that as "nothing to validate against". BOS is
        suppressed and no generation prompt is added so the render aligns with a
        fragment prefill (``add_special=False``).
        """
        if self._model_formatter is None:
            return None
        try:
            return self._model_formatter(messages=messages).prompt
        except Exception:
            return None

    def _build_model_formatter(self):
        template = self._b.metadata_value(self._model, "tokenizer.chat_template")
        if not template:
            return None
        from llama_cpp.llama_chat_format import Jinja2ChatFormatter
        return Jinja2ChatFormatter(
            template=template,
            eos_token=self._b.eos_text(self._vocab),
            bos_token="",
            add_generation_prompt=False,
        )

    # ----- cache edits ---------------------------------------------------
    def reset(self) -> None:
        self._b.kv_clear(self._ctx)

    def prefill(self, token_ids: list[int], start_pos: int, want_logits: bool) -> None:
        self._b.decode(self._ctx, token_ids, start_pos, SEQ, want_logits, self._n_batch)

    def apply_eviction(self, ev: Eviction) -> None:
        """Remove an evicted message's tokens, then shift survivors to close the gap."""
        self._b.kv_seq_rm(self._ctx, SEQ, ev.remove_start, ev.remove_end)
        if ev.remove_end < ev.old_total:
            self._b.kv_seq_add(
                self._ctx, SEQ, ev.remove_end, ev.old_total, -ev.shift_delta
            )

    # ----- generation ----------------------------------------------------
    def generate(
        self, start_pos: int, n_predict_max: int, stop: list[str],
        out: GenerationAccumulator | None = None,
    ):
        """Stream a reply, yielding text deltas as tokens are sampled.

        Generation begins from the logits of the token most recently prefilled
        (the caller must have prefilled the generation prompt with
        ``want_logits=True``). Each sampled token is decoded into the cache at the
        next free position **before** its text is yielded, so the cache and
        ``out.token_ids`` never diverge even if the consumer abandons the stream
        early (barge-in).

        Bytes are run through an incremental UTF-8 decoder, so a codepoint split
        across two tokens is never emitted as a replacement character. When
        ``stop`` strings are configured, the trailing few characters are held back
        until they are known not to begin a stop string.

        Args:
            start_pos: Cache position of the first generated token.
            n_predict_max: Hard cap on tokens to generate.
            stop: Stop strings; generation ends just before the earliest match.
            out: Accumulator to fill (token_ids/text/stop_reason). Pass one in to
                observe progress after barge-in; a fresh one is used otherwise.

        Yields:
            Visible reply text deltas; their concatenation equals ``out.text``.
        """
        acc = out if out is not None else GenerationAccumulator()
        pos = start_pos
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        max_stop = max((len(s) for s in stop if s), default=0)
        tbuf = _TextBuffer()

        for _ in range(max(0, n_predict_max)):
            tok = self._b.lc.llama_sampler_sample(self._sampler, self._ctx, -1)
            if self._b.is_eog(self._vocab, tok):
                acc.stop_reason = "eog"
                break
            # Commit to the cache first, then record, then surface text: this
            # ordering is what keeps token_ids == cache across an early close.
            self._b.decode(self._ctx, [tok], pos, SEQ, True, self._n_batch)
            pos += 1
            acc.token_ids.append(tok)
            tbuf.append(decoder.decode(self._b.token_to_piece_bytes(self._vocab, tok)))

            if max_stop:
                hit = _stop_hit(tbuf.text, stop)
                if hit is not None:
                    delta = tbuf.emit(hit)
                    if delta:
                        acc.text += delta
                        yield delta
                    acc.stop_reason = "stop"
                    break
                delta = tbuf.emit(len(tbuf) - (max_stop - 1))  # hold back possible prefix
                if delta:
                    acc.text += delta
                    yield delta
            else:
                delta = tbuf.emit(len(tbuf))
                if delta:
                    acc.text += delta
                    yield delta
        else:
            acc.stop_reason = "length"

        # Flush the held-back tail (and any dangling bytes) unless a stop string
        # truncated the reply.
        if acc.stop_reason != "stop":
            tbuf.append(decoder.decode(b"", final=True))
            delta = tbuf.emit(len(tbuf))
            if delta:
                acc.text += delta
                yield delta

    # ----- lifecycle -----------------------------------------------------
    def close(self) -> None:
        try:
            self._b.lc.llama_sampler_free(self._sampler)
        finally:
            self._b.free_context(self._ctx)
            self._b.free_model(self._model)

    # ----- internals -----------------------------------------------------
    def _build_sampler(self, cfg: Config):
        lc = self._b.lc
        params = lc.llama_sampler_chain_default_params()
        chain = lc.llama_sampler_chain_init(params)
        add = lc.llama_sampler_chain_add
        if cfg.repeat_penalty and cfg.repeat_penalty != 1.0:
            try:
                add(chain, lc.llama_sampler_init_penalties(
                    cfg.repeat_last_n, cfg.repeat_penalty, 0.0, 0.0))
            except TypeError:
                pass  # older signature; skip penalties rather than crash
        if cfg.temperature <= 0.0:
            add(chain, lc.llama_sampler_init_greedy())
            return chain
        if cfg.top_k > 0:
            add(chain, lc.llama_sampler_init_top_k(cfg.top_k))
        add(chain, lc.llama_sampler_init_top_p(cfg.top_p, 1))
        add(chain, lc.llama_sampler_init_temp(cfg.temperature))
        add(chain, lc.llama_sampler_init_dist(cfg.seed))
        return chain


def _stop_hit(text: str, stop: list[str]) -> int | None:
    """Return the index where the earliest stop string begins, or ``None``."""
    earliest: int | None = None
    for s in stop:
        if not s:
            continue
        i = text.find(s)
        if i != -1 and (earliest is None or i < earliest):
            earliest = i
    return earliest
