"""Configuration for the chat wrapper.

The ``Config`` object is the only thing that differs between deployment targets:
the model (and its chat-template fragments), the context size, and the eviction
threshold. The wrapper logic itself is hardware agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    """Runtime configuration for :class:`~llama_chat.wrapper.ChatWrapper`.

    The chat template is expressed as per-role *fragments*: the literal text
    wrapped around each message's content, so one turn renders as
    ``prefix + text + suffix`` (the combination is done by
    :class:`~llama_chat.template.TemplateFormatter`).

    Attributes:
        model_path: Path to the GGUF model file.
        n_ctx: General context size in tokens (the hard wall — generation must
            never cross it).
        threshold_pct: Fraction of ``n_ctx`` above which prefilled content
            triggers eviction of the oldest messages. Must be in ``(0, 1]``.
        n_gpu_layers: Layers to offload to GPU. ``0`` for a CPU-only build;
            ``-1`` to offload everything on the GPU.
        n_threads: CPU threads for decode (``None`` lets llama.cpp decide).
        n_batch: Maximum tokens submitted to a single ``llama_decode`` call;
            longer prefills are chunked to this size.
        seed: RNG seed for sampling.
        max_tokens: Upper bound on generated tokens per ``request`` (further
            capped so total never exceeds ``n_ctx``).
        temperature: Sampling temperature; ``<= 0`` selects greedy decoding.
        top_k: Top-k sampling cutoff (``<= 0`` disables).
        top_p: Nucleus sampling cutoff.
        repeat_penalty: Repetition penalty applied over ``repeat_last_n`` tokens.
        repeat_last_n: Window for the repetition penalty.
        stop: Additional stop strings that end generation.
        oversize_policy: What to do when a single message cannot fit under the
            threshold even after evicting everything but the system prompt:
            ``"reject"`` raises, ``"truncate"`` clips the message to fit.
        system_prefix, system_suffix: Wrap a system turn.
        user_prefix, user_suffix: Wrap a user turn.
        assistant_prefix, assistant_suffix: Wrap an assistant turn.
        verbose: Pass through of llama.cpp's logging.
    """

    # Gemma 4

    model_path: str = "models/gemma-4-E2B-it-Q4_K_M.gguf"
    n_ctx: int = 4096
    threshold_pct: float = 0.75

    # Backend / hardware
    n_gpu_layers: int = 0
    n_threads: int | None = None
    n_batch: int = 512
    seed: int = 0

    # Generation defaults (overridable per request)
    max_tokens: int = 512
    temperature: float = 1.0
    top_k: int = 64
    top_p: float = 0.95
    repeat_penalty: float = 1.0 # 1.0 means no penalty
    repeat_last_n: int = 64
    stop: list[str] = field(default_factory=list)

    # Policy
    oversize_policy: str = "reject"

    # Chat template
    system_prefix: str = "<|turn>system\n"
    system_suffix: str = "<turn|>\n"
    user_prefix: str = "<|turn>user\n"
    user_suffix: str = "<turn|>\n"
    assistant_prefix: str = "<|turn>model\n"
    assistant_suffix: str = "<turn|>\n"

    verbose: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.threshold_pct <= 1.0:
            raise ValueError("threshold_pct must be in (0, 1]")
        if self.n_ctx <= 0:
            raise ValueError("n_ctx must be positive")
        if self.oversize_policy not in ("reject", "truncate"):
            raise ValueError("oversize_policy must be 'reject' or 'truncate'")

    @property
    def threshold_tokens(self) -> int:
        """Eviction trigger expressed as an absolute token count."""
        return int(self.threshold_pct * self.n_ctx)
