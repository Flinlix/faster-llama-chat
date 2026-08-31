"""End-to-end demo of the KV-cache manager against a real GGUF model.

The model file is given on the command line; the rest of the config defaults
to a CPU run and can be changed to match your hardware.

Install the package first (see install.sh or README), then run:

    python examples/demo.py /path/to/model.gguf                  # CPU
    python examples/demo.py /path/to/model.gguf --gpu-layers -1  # GPU

The demo proves the core claim by logging how many tokens are *prefilled* per
action: after ``begin`` primes the conversation, each ``generate`` prefills only
its own new prompt, never the surviving history. The eviction stress run uses
a deliberately tiny context to show the oldest messages being dropped while the
system prompt survives and the total never crosses ``context_size``.
"""

from __future__ import annotations

import argparse
import sys

from llama_chat import ChatWrapper, Config, KVContext


class CountingContext(KVContext):
    """KVContext that records prefill volume so the demo can report reuse."""

    def __init__(self, config: Config, on_load_progress=None) -> None:
        super().__init__(config, on_load_progress=on_load_progress)
        self.prefilled = 0

    def prefill(self, token_ids, start_pos, want_logits):
        self.prefilled += len(token_ids)
        super().prefill(token_ids, start_pos, want_logits)


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def load_bar(frac: float, width: int = 40) -> None:
    """Render a single-line model-load progress bar to stderr (no trailing newline)."""
    filled = int(frac * width)
    bar = "#" * filled + "-" * (width - filled)
    print(f"\rloading model [{bar}] {frac * 100:5.1f}%", end="", file=sys.stderr, flush=True)


def show(w: ChatWrapper) -> None:
    rows = w.snapshot()
    print(f"  cache: {w.total_tokens} tokens across {len(rows)} messages")
    for r in rows:
        print(f"    [{r['pos_start']:>5},{r['pos_end']:>5})  {r['role']:<10} ({r['n_tokens']} tok)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", help="path to a .gguf model")
    ap.add_argument("--context-size", type=int, default=2048)
    ap.add_argument("--gpu-layers", type=int, default=0)
    ap.add_argument("--evict-above-fraction", type=float, default=0.8)
    args = ap.parse_args()

    # ---- Normal conversation: prove only-new-tokens prefill ----
    banner("CONVERSATION (reuse): only new tokens are prefilled")
    cfg = Config(
        model_path=args.model,
        context_size=args.context_size,
        evict_above_fraction=args.evict_above_fraction,
        gpu_layers=args.gpu_layers,
        max_reply_tokens=64,
    )
    ctx = CountingContext(cfg, on_load_progress=load_bar)
    load_bar(1.0)           # snap to 100% (the loader may report just short) ...
    print(file=sys.stderr)  # ... then terminate the bar's line
    w = ChatWrapper(cfg, context=ctx)

    w.begin(
        "You are a concise, helpful assistant.",
        [("user", "Remember my name is Felix."), ("assistant", "Got it, Felix.")],
    )
    print(f"begin: prefilled {ctx.prefilled} tokens (system + history)")
    show(w)

    w.inject("Context: the user is building a llama.cpp wrapper in Python.")
    print(f"\nafter inject: total prefilled so far {ctx.prefilled}")
    show(w)

    for q in ["What's my name?", "What am I building?", "Suggest one optimization."]:
        before = ctx.prefilled
        reply = w.generate(q)
        print(f"\n> {q}")
        print(f"  reply: {reply.text.strip()[:200]}")
        print(
            f"  prefilled {ctx.prefilled - before} new tokens "
            f"(prompt={reply.n_prefilled_tokens}, "
            f"generated={reply.n_generated_tokens}, "
            f"evicted={reply.n_evicted_messages}, stop={reply.stop_reason})"
        )
    show(w)
    w.close()

    # ---- Eviction stress: tiny context, watch the eviction ----
    banner("EVICTION STRESS: a tiny context cuts the oldest messages, system kept")
    small = Config(
        model_path=args.model,
        context_size=512,
        evict_above_fraction=0.5,
        gpu_layers=args.gpu_layers,
        max_reply_tokens=32,
    )
    sw = ChatWrapper(small, context=KVContext(small))
    sw.begin("You are a terse assistant. Keep replies under 10 words.")
    for i in range(10):
        reply = sw.generate(f"Give me fact number {i} about the planet Mars, briefly.")
        roles = [r["role"] for r in sw.snapshot()]
        assert roles[0] == "system", "system prompt was evicted!"
        assert sw.total_tokens <= small.context_size, "context overflow!"
        print(
            f"turn {i:>2}: total={sw.total_tokens:>4}/{small.context_size} "
            f"messages={len(roles):>2} evicted={reply.n_evicted_messages}"
        )
    show(sw)
    sw.close()
    print("\nOK: system prompt survived every eviction; context never overflowed.")


if __name__ == "__main__":
    main()
