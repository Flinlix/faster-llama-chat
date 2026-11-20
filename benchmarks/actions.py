"""Time every public action of the wrapper on one model file, and each part inside it.

Usage: python benchmarks/actions.py MODEL_PATH OUT_JSON [--gpu-layers N] [--context N] [--threads N]

Each action is timed whole (wall clock), and inside it: the engine's own calls
(prefill, sampling, decoding a generated token, state saves and restores, shifts,
cuts, tokenizing), the template's renders, and the rest, which is the library's
own Python work. A reply also records the time to its first text and its rate.
"""

import argparse
import cProfile
import io
import json
import pstats
import threading
import time
from collections import defaultdict
from pathlib import Path

from llama_chat import ChatWrapper, Config, ToolRegistry
from llama_chat import context as context_module
from llama_chat import model_template

parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
parser.add_argument("model_path")
parser.add_argument("out_json")
parser.add_argument("--gpu-layers", type=int, default=-1, help="-1: every layer on the GPU; 0: none")
parser.add_argument("--context", type=int, default=8192, help="context size in tokens")
parser.add_argument("--threads", type=int, default=None, help="decode and prefill threads")
arguments = parser.parse_args()
MODEL, GPU_LAYERS, OUT, THREADS = (arguments.model_path, arguments.gpu_layers, arguments.out_json,
                                   arguments.threads)

# ----- timers ------------------------------------------------------------------------
_parts = defaultdict(float)       # seconds per part within the action being measured
_counts = defaultdict(int)
_tokens = defaultdict(int)        # tokens per engine part
_depth = threading.local()


def _timed(name, function, counts_tokens=False):
    def timed(*args, **kwargs):
        outer = getattr(_depth, "on", False)
        if outer:
            return function(*args, **kwargs)
        _depth.on = True
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _parts[name] += time.perf_counter() - started
            _counts[name] += 1
            if counts_tokens and len(args) > 1 and isinstance(args[1], list):
                _tokens[name] += len(args[1])
            _depth.on = False
    return timed


def _timed_generator(name, function):
    def timed(*args, **kwargs):
        steps = function(*args, **kwargs)
        try:
            while True:
                started = time.perf_counter()
                try:
                    token = next(steps)
                except StopIteration:
                    return
                finally:
                    _parts[name] += time.perf_counter() - started
                _counts[name] += 1
                yield token
        finally:
            steps.close()
    return timed


K = context_module.KVContext
for name in ("prefill", "decode_generated"):
    setattr(K, name, _timed(f"engine.{name}", getattr(K, name), counts_tokens=True))
for name in ("save_state", "restore_state", "apply_eviction", "remove_from", "reset", "tokenize",
             "token_text", "token_bytes", "is_special"):
    setattr(K, name, _timed(f"engine.{name}", getattr(K, name)))
K.sampled_tokens = _timed_generator("engine.sample", K.sampled_tokens)
model_template.ModelTemplate.render = _timed("template.render", model_template.ModelTemplate.render)

results = []


def measure(action, run, **facts):
    """Run ``run`` as one measured action; what it returns is merged into its record."""
    for part in (_parts, _counts, _tokens):
        part.clear()
    started = time.perf_counter()
    extra = run() or {}
    wall = time.perf_counter() - started
    engine = sum(v for k, v in _parts.items() if k.startswith("engine."))
    render = _parts.get("template.render", 0.0)
    record = {"action": action, "wall_s": round(wall, 4), "engine_s": round(engine, 4),
              "render_s": round(render, 4), "python_s": round(wall - engine - render, 4),
              "parts_s": {k: round(v, 4) for k, v in sorted(_parts.items())},
              "calls": dict(_counts), "tokens": dict(_tokens), **facts, **extra}
    results.append(record)
    print(json.dumps(record), flush=True)
    return record


def streamed(stream):
    """Read ``stream``; its first-text time, its end time, its text and its Reply."""
    started = time.perf_counter()
    first = None
    text = []
    for step in stream:
        if isinstance(step, str) and step and first is None:
            first = time.perf_counter() - started
        if isinstance(step, str):
            text.append(step)
    total = time.perf_counter() - started
    reply = stream.reply
    gen = reply.n_generated_tokens if reply else 0
    rate = gen / (total - first) if first is not None and total > first and gen > 1 else None
    return {"ttft_s": None if first is None else round(first, 4), "reply_s": round(total, 4),
            "generated": gen, "prefilled": reply.n_prefilled_tokens if reply else None,
            "prompt_tokens": reply.n_prompt_tokens if reply else None,
            "gen_tok_s": None if rate is None else round(rate, 2),
            "stop": reply.stop_reason if reply else None,
            "evicted": reply.n_evicted_messages if reply else None,
            "text": "".join(text)[:80]}


registry = ToolRegistry()


@registry.tool
def get_weather(city: str) -> str:
    """The current weather in a city."""
    return f"Sunny, 21 degrees in {city}."


@registry.tool
def get_time() -> str:
    """The current local time."""
    return "14:05"


SYSTEM = "You are a helpful, concise assistant. Answer in two or three sentences."
QUESTIONS = ["What is the capital of France?", "Why is the sky blue?",
             "Name three uses of copper.", "How does a refrigerator work?",
             "What is a prime number?", "Give me a short tip for sleeping better."]

config = Config(model_path=MODEL, context_size=arguments.context, gpu_layers=GPU_LAYERS,
                max_reply_tokens=160,
                **({"decode_threads": THREADS, "prefill_threads": THREADS} if THREADS else {}))
box = {}
measure("load (ChatWrapper)", lambda: box.update(w=ChatWrapper(config)))
w = box["w"]
caps = w.capabilities(tools=True)
print("capabilities", caps, "can_shift", w.can_shift, flush=True)


def opened(system: str) -> list[dict]:
    """A conversation's opening: the system prompt, where the template takes one."""
    takes = w.capabilities(tools=False).can_take_system_prompt
    return [{"role": "system", "content": system}] if takes else []


measure("begin (system prompt)", lambda: {"report": str(w.begin(opened(SYSTEM)))})

for i, q in enumerate(QUESTIONS):
    measure(f"reply {i + 1} (question)", lambda q=q: streamed(w.reply(q)))
    measure(f"work after reply {i + 1} (wait_idle)", lambda: {"idle": w.wait_idle(30)})

measure("inject (user note)", lambda: {"report": str(w.inject("Please keep answers short."))})
measure("reply (to injected note)", lambda: streamed(w.reply()))
measure("truncate_last_reply (half)", lambda: w.truncate_last_reply(
    w.conversation()["messages"][-1]["content"][:len(w.conversation()["messages"][-1]["content"]) // 2]))
measure("reply after truncate", lambda: streamed(w.reply("Continue, please.")))
measure("conversation()", lambda: {"messages": len(w.conversation()["messages"])})
saved = w.conversation()
measure("begin (same conversation again)", lambda: {"report": str(w.begin(**saved))})
measure("begin (another conversation)", lambda: {"report": str(w.begin(
    [*opened("You answer in German."),
     {"role": "user", "content": "Hallo"}, {"role": "assistant", "content": "Hallo! Wie kann ich helfen?"}]))})
measure("begin (restore saved conversation)", lambda: {"report": str(w.begin(**saved))})
measure("reply after restore", lambda: streamed(w.reply("One more question: what is 7 times 8?")))

if caps.can_reason:
    measure("begin (reasoning on)", lambda: {"report": str(w.begin(**{**w.conversation(), "reasoning": True, "reasoning_level": None}))})
    for i, q in enumerate(["Is 221 a prime number?", "What is 15% of 80?"]):
        measure(f"reply reasoning {i + 1}", lambda q=q: streamed(w.reply(q, max_reply_tokens=600)))
        measure(f"work after reasoning reply {i + 1}", lambda: {"idle": w.wait_idle(30)})
    measure("begin (reasoning off)", lambda: {"report": str(w.begin(**{**w.conversation(), "reasoning": False, "reasoning_level": None}))})

if caps.can_call_tools:
    measure("begin (tools declared)", lambda: {"report": str(w.begin(**{**w.conversation(), "tools": registry.specs()}))})
    measure("reply with registry (tool runs)", lambda: streamed(w.reply("What is the weather in Paris right now?", tool_call_policy=registry)))
    measure("work after tool reply", lambda: {"idle": w.wait_idle(30)})

    def yielded():
        stream = w.reply("And what time is it?", tool_call_policy="yield")
        out = streamed(stream)
        out["calls"] = [c.name for c in (stream.reply.tool_calls if stream.reply else [])]
        return out
    measure("reply with yield (call handed over)", yielded)
    if w.pending_calls:
        measure("inject_tool_result", lambda: {"reports": [str(w.inject_tool_result(call.id, "14:05"))
                                                           for call in list(w.pending_calls)]})
        measure("reply after result", lambda: streamed(w.reply(tool_call_policy="yield")))
    measure("reply with tools declared, no call", lambda: streamed(w.reply("Say hello in one word.", tool_call_policy=registry)))




def cancelled():
    cancel = threading.Event()
    stream = w.reply("Write a long story about a dragon.", cancel=cancel, max_reply_tokens=400)
    n = 0
    set_at = None
    for _ in stream:
        n += 1
        if n == 10:
            set_at = time.perf_counter()
            cancel.set()
    return {"cancel_to_end_s": round(time.perf_counter() - set_at, 4) if set_at else None,
            "stop": stream.reply.stop_reason if stream.reply else None}


measure("reply cancelled after 10 steps", cancelled)

# A long conversation: the room fills and evictions come.
measure("begin (long run)", lambda: {"report": str(w.begin(opened(SYSTEM)))})
for i in range(40):
    # The tokens held are read once the reply has run.
    measure(f"long run reply {i + 1}", lambda i=i: {**streamed(w.reply(
        f"Question {i + 1}: tell me one fact about the number {i + 7}, in about eighty words.",
        max_reply_tokens=140)), "total_tokens": w.total_tokens})

# A profile of a few plain replies, for the library's own Python hot spots.
profiler = cProfile.Profile()
measure("begin (profiled run)", lambda: {"report": str(w.begin(opened(SYSTEM)))})
profiler.enable()
for q in QUESTIONS[:3]:
    streamed(w.reply(q))
    w.wait_idle(30)
profiler.disable()
listing = io.StringIO()
pstats.Stats(profiler, stream=listing).sort_stats("tottime").print_stats(40)

measure("close", lambda: w.close())
Path(OUT).write_text(json.dumps({"model": Path(MODEL).name, "gpu_layers": GPU_LAYERS,
                                 "context": arguments.context, "threads": THREADS,
                                 "results": results, "profile": listing.getvalue()}, indent=1))
print("written", OUT)
