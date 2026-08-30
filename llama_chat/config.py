"""Runtime configuration: the only thing that differs between deployment targets."""

from __future__ import annotations

from dataclasses import dataclass, field

# Quantization names -> ggml type ids (ggml.h, `enum ggml_type`).
KV_CACHE_GGML_TYPES = {
    "f16": 1,
    "q8_0": 8,
    "q5_1": 7,
    "q5_0": 6,
    "q4_1": 3,
    "q4_0": 2,
}

# The one type here that is not quantized, and so the only one usable without
# flash attention. Naming it is the same as leaving kv_cache_type unset.
UNQUANTIZED_KV_CACHE_TYPE = "f16"

LOG_LEVELS = ("debug", "info", "warn", "error", "none")


@dataclass
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

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
        penalty_window_tokens: How far back the three penalties look.
        seed: Sampling RNG seed in ``[0, 2**32 - 2]``. ``None`` draws a fresh
            one per run.
        context_size: The hard limit in tokens; generation never crosses it.
        evict_above_fraction: Fraction of ``context_size`` above which the
            oldest messages are evicted. Must be in ``(0, 1]``.
        max_history_messages: The most non-system messages to keep, a second
            eviction trigger OR'd with ``evict_above_fraction``. ``None``
            leaves the token threshold as the only trigger.
        max_reply_tokens: Cap on tokens generated per turn, capped again so the
            total stays inside ``context_size``.
        oversize_policy: What happens to a message that cannot fit even with
            everything but the system prompt evicted: ``"reject"`` raises,
            ``"truncate"`` clips the content and keeps the turn-terminator tag.
        min_reply_tokens: Refuse a request with
            :class:`~llama_chat.wrapper.ContextOverflowError` when fewer than
            this many tokens would be left for the reply. ``0`` disables it.
        stop_strings: Strings that end generation.
        gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only build,
            ``-1`` for all of them.
        kv_cache_on_gpu: ``True`` places the KV cache in VRAM, ``False`` keeps
            it in host RAM. Ignored on a CPU-only build.
        decode_threads: CPU threads for decode. ``None`` lets llama.cpp decide.
        prefill_threads: CPU threads for prefill. ``None`` means as many as
            ``decode_threads``.
        max_batch_tokens: Most tokens handed to one ``llama_decode``; a longer
            prefill is chunked to this.
        flash_attention: Enable flash attention. Required by a quantized
            ``kv_cache_type``.
        kv_cache_type: KV-cache quantization, one of ``KV_CACHE_GGML_TYPES``,
            or ``None`` for llama.cpp's own default. Every one of them but
            ``UNQUANTIZED_KV_CACHE_TYPE`` requires ``flash_attention=True``.
        log_level: llama.cpp's own verbosity, one of ``LOG_LEVELS``. Applied
            process-wide before the model loads.
    """

    # Model

    model_path: str | None = None
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    penalty_window_tokens: int = 64
    seed: int | None = None

    # Context management

    context_size: int = 4096
    evict_above_fraction: float = 0.75
    max_history_messages: int | None = None
    max_reply_tokens: int = 1024
    oversize_policy: str = "reject"
    min_reply_tokens: int = 32

    # Generation

    stop_strings: list[str] = field(default_factory=list)

    # Backend / hardware

    gpu_layers: int = 0
    kv_cache_on_gpu: bool = False
    decode_threads: int | None = None
    prefill_threads: int | None = None
    max_batch_tokens: int = 512
    flash_attention: bool = False
    kv_cache_type: str | None = None

    # Logging

    log_level: str = "error"

    def __post_init__(self) -> None:
        if not 0.0 < self.evict_above_fraction <= 1.0:
            raise ValueError("evict_above_fraction must be in (0, 1]")
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")
        if self.max_batch_tokens <= 0:
            raise ValueError("max_batch_tokens must be positive")
        if self.max_reply_tokens <= 0:
            raise ValueError("max_reply_tokens must be positive")
        if self.min_reply_tokens < 0:
            raise ValueError("min_reply_tokens must be >= 0")
        if self.max_history_messages is not None and self.max_history_messages < 1:
            raise ValueError(
                "max_history_messages must be None or >= 1 "
                "(use None for no message limit)"
            )
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0 (1.0 disables)")
        if self.seed is not None and not 0 <= self.seed <= 0xFFFFFFFE:
            raise ValueError(
                "seed must be None or an int in [0, 2**32 - 2] "
                "(use None for a random seed)"
            )
        if self.oversize_policy not in ("reject", "truncate"):
            raise ValueError("oversize_policy must be 'reject' or 'truncate'")
        if self.log_level not in LOG_LEVELS:
            raise ValueError(f"log_level must be one of {list(LOG_LEVELS)}")
        if self.kv_cache_type is not None:
            if self.kv_cache_type not in KV_CACHE_GGML_TYPES:
                raise ValueError(
                    f"kv_cache_type must be None or one of {sorted(KV_CACHE_GGML_TYPES)}"
                )
            if self.kv_cache_type != UNQUANTIZED_KV_CACHE_TYPE \
                    and not self.flash_attention:
                raise ValueError(
                    f"kv_cache_type={self.kv_cache_type!r} is quantized and "
                    f"requires flash_attention=True "
                    f"(only {UNQUANTIZED_KV_CACHE_TYPE!r} works without it)"
                )

    @property
    def evict_above_tokens(self) -> int:
        """``evict_above_fraction`` as an absolute token count."""
        return int(self.evict_above_fraction * self.context_size)
