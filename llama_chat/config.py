"""Runtime configuration: the only thing that differs between deployment targets."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

# The quantizations a caller may name. Which type id each one is worth is
# llama.cpp's own numbering, resolved where llama.cpp is named.
KV_CACHE_TYPES = ("f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")

# The one type here that is not quantized, and so the only one usable without
# flash attention. It is kv_cache_type's default.
UNQUANTIZED_KV_CACHE_TYPE = "f16"


class InvalidSettingError(ValueError):
    """A setting the configuration refuses.

    Attributes:
        field: The name of the setting, as :class:`Config` and
            :class:`Sampling` spell it.
    """

    def __init__(self, field: str, message: str) -> None:
        super().__init__(message)
        self.field = field


@dataclass(frozen=True)
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

    The five model sampling fields, ``temperature`` to
    ``repetition_penalty``, left ``None`` take the model's own when it loads:
    what its GGUF states, else its add-on file's, else llama.cpp's default. See
    the README, "Configuration".

    Attributes:
        model_path: Path to the GGUF model file. Loading refuses without one.
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding,
            which leaves ``top_k``, ``top_p``, ``min_p`` and ``seed`` unused.
        top_k: Top-k sampling cutoff (``<= 0`` disables).
        top_p: Nucleus sampling cutoff.
        min_p: Drop tokens whose probability is below ``min_p`` times the top
            token's probability. ``0`` disables.
        repetition_penalty: Penalty on any token in the penalty window
            (``1.0`` disables).
        presence_penalty: Flat penalty on any token in the penalty window
            (``0`` disables).
        frequency_penalty: Penalty scaled by how often a token occurs in the
            penalty window (``0`` disables).
        penalty_window_tokens: How far back the three penalties look, within
            the tokens of the generation being sampled: each generation starts
            with an empty window, and a turn resumed after a tool result is a
            generation of its own.
        seed: Sampling RNG seed in ``[0, 2**32 - 2]``, which every generation
            starts from. ``None`` draws a fresh one per generation.
        context_size: The hard limit in tokens; generation never crosses it.
        evict_above_fraction: Fraction of ``context_size`` above which the
            oldest messages are evicted. Must be in ``(0, 1]`` and must come
            to at least one token.
        max_history_exchanges: The most exchanges to keep, a second eviction
            trigger OR'd with ``evict_above_fraction``. An exchange is one
            user message and everything after it up to the next, the unit an
            eviction drops; what a history holds before its first user message
            is one more. ``None`` leaves the token threshold as the only
            trigger; otherwise at least ``1``.
        max_reply_tokens: Cap on tokens per generation, a thought included.
            ``None`` sets none, so a reply may run until the context is full;
            generation never crosses ``context_size`` either way.
        min_reply_tokens: Refuse a request with
            :class:`~llama_chat.reply.ContextOverflowError` when fewer than
            this many tokens would be left for the reply. ``0`` disables it,
            and it must stay below ``context_size``, which no request can leave
            entirely free.
        gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only build,
            ``-1`` for all of them.
        kv_cache_on_gpu: ``True`` keeps each layer's KV cache where that
            layer runs, in VRAM for the layers ``gpu_layers`` offloads;
            ``False`` keeps all of it in host RAM. Ignored on a CPU-only build.
        decode_threads: CPU threads for decode. ``None`` lets llama.cpp decide.
        prefill_threads: CPU threads for prefill. ``None`` means as many as
            ``decode_threads``.
        max_batch_tokens: Most tokens handed to the engine in one decode call; a longer
            prefill is chunked to this.
        flash_attention: ``True`` or ``False`` forces flash attention on or
            off. ``None`` leaves it to the engine, except under a quantized
            ``kv_cache_type``, which requires it and turns it on.
        kv_cache_type: KV-cache quantization, one of ``KV_CACHE_TYPES``.
            Every one of them but ``UNQUANTIZED_KV_CACHE_TYPE``, the default,
            requires flash attention, so is refused with
            ``flash_attention=False``.
    """

    # Model

    model_path: str | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repetition_penalty: float | None = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_window_tokens: int = 64
    seed: int | None = None

    # Context management

    context_size: int = 4096
    evict_above_fraction: float = 0.75
    max_history_exchanges: int | None = None
    max_reply_tokens: int | None = None
    min_reply_tokens: int = 32

    # Backend / hardware

    gpu_layers: int = 0
    kv_cache_on_gpu: bool = True
    decode_threads: int | None = None
    prefill_threads: int | None = None
    max_batch_tokens: int = 2048
    flash_attention: bool | None = None
    kv_cache_type: str = UNQUANTIZED_KV_CACHE_TYPE

    def __post_init__(self) -> None:
        if not 0.0 < self.evict_above_fraction <= 1.0:
            raise InvalidSettingError("evict_above_fraction",
                                      "evict_above_fraction must be in (0, 1]")
        if self.context_size <= 0:
            raise InvalidSettingError("context_size", "context_size must be positive")
        if self.evict_above_tokens < 1:
            raise InvalidSettingError(
                "evict_above_fraction",
                f"evict_above_fraction ({self.evict_above_fraction}) of "
                f"context_size ({self.context_size}) rounds down to no tokens, "
                "so not even a system prompt would ever fit")
        if self.max_batch_tokens <= 0:
            raise InvalidSettingError("max_batch_tokens", "max_batch_tokens must be positive")
        ensure_reply_cap(self.max_reply_tokens)
        if self.min_reply_tokens < 0:
            raise InvalidSettingError("min_reply_tokens", "min_reply_tokens must be >= 0")
        if self.min_reply_tokens >= self.context_size:
            raise InvalidSettingError(
                "min_reply_tokens",
                f"min_reply_tokens ({self.min_reply_tokens}) must be below "
                f"context_size ({self.context_size}); no request could leave "
                "that many tokens free, so every one of them would be refused")
        if self.max_history_exchanges is not None and self.max_history_exchanges < 1:
            raise InvalidSettingError(
                "max_history_exchanges",
                "max_history_exchanges must be None or >= 1 (use None for no exchange limit)")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise InvalidSettingError("repetition_penalty",
                                      "repetition_penalty must be > 0 (1.0 disables)")
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            raise InvalidSettingError("top_p", "top_p must be in (0, 1]")
        if self.min_p is not None and not 0.0 <= self.min_p <= 1.0:
            raise InvalidSettingError("min_p", "min_p must be in [0, 1]")
        if self.seed is not None and not 0 <= self.seed <= 0xFFFFFFFE:
            raise InvalidSettingError(
                "seed",
                "seed must be None or an int in [0, 2**32 - 2] (use None for a random seed)")
        if self.kv_cache_type not in KV_CACHE_TYPES:
            raise InvalidSettingError(
                "kv_cache_type", f"kv_cache_type must be one of {sorted(KV_CACHE_TYPES)}")
        if self.kv_cache_type != UNQUANTIZED_KV_CACHE_TYPE and self.flash_attention is False:
            raise InvalidSettingError(
                "kv_cache_type",
                f"kv_cache_type={self.kv_cache_type!r} is quantized and requires flash "
                f"attention, which flash_attention=False turns off (only "
                f"{UNQUANTIZED_KV_CACHE_TYPE!r} works without it)")

    @property
    def evict_above_tokens(self) -> int:
        """``evict_above_fraction`` as an absolute token count."""
        return int(self.evict_above_fraction * self.context_size)


def ensure_reply_cap(max_reply_tokens: int | None) -> None:
    """Refuse a cap on a reply's tokens that allows none.

    Raises:
        InvalidSettingError: If ``max_reply_tokens`` is not ``None`` or positive.
    """
    if max_reply_tokens is not None and max_reply_tokens <= 0:
        raise InvalidSettingError(
            "max_reply_tokens", "max_reply_tokens must be None or positive (use None for no cap)")


@dataclass(frozen=True)
class Sampling:
    """Sampling settings for one request, overriding the :class:`Config`'s.

    Every field left ``None`` keeps the value the wrapper was built with, so an
    empty ``Sampling`` changes nothing. A setting is refused where
    :class:`Config` refuses it, by the same checks.

    Attributes:
        temperature: As :attr:`Config.temperature`.
        top_p: As :attr:`Config.top_p`.
        seed: As :attr:`Config.seed`. A request cannot ask for a fresh seed
            where the config fixes one; leave it ``None`` to keep the config's.
        presence_penalty: As :attr:`Config.presence_penalty`.
        frequency_penalty: As :attr:`Config.frequency_penalty`.
    """

    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None

    def __post_init__(self) -> None:
        with_request_sampling(Config(), self)


def with_request_sampling(config: Config, sampling: Sampling | None) -> Config:
    """``config`` with the settings ``sampling`` sets in place of its own; ``config`` itself without one.

    Raises:
        InvalidSettingError: If a setting is one :class:`Config` refuses.
    """
    if sampling is None:
        return config
    return replace(config, **{f.name: getattr(sampling, f.name) for f in fields(sampling)
                              if getattr(sampling, f.name) is not None})
