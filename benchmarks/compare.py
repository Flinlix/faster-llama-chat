"""Compare the library with llama-cpp-python's own chat interface on the same model file and the same turns.

Usage:
    python benchmarks/compare.py questions SIDE MODEL_PATH OUT_JSON [--fills 0,0.4,0.6]
    python benchmarks/compare.py long SIDE MODEL_PATH OUT_JSONL [--mode speed|stability]

SIDE is ``library`` or ``plain``. Run each side in its own process, one after
the other, so that both have the whole machine. ``questions`` asks ten short
questions on a fresh conversation and on histories that fill part of the
context; ``long`` runs one conversation until the plain side's context is full
or ``--most-tokens`` tokens were given and generated, and in ``stability``
mode checks every turn whether the reply still keeps the system prompt's
instructions and the facts it was given.
"""

import argparse
import importlib
import inspect
import json
import random
import re
import time
from pathlib import Path

SYSTEM = "You are a helpful, concise assistant."
# The sampling values both sides take; the plain side takes the seed when it loads.
SHARED_SAMPLING = ("temperature", "top_k", "top_p", "min_p")
QUESTIONS = [
    "What is the capital of France, and why did it become the capital?",
    "Why is the sky blue? Explain it to a twelve-year-old.",
    "Name three uses of copper and say why copper suits each.",
    "How does a refrigerator keep food cold?",
    "What is a prime number? Give the first five.",
    "Give me three short tips for sleeping better.",
    "What causes the seasons on Earth?",
    "How do vaccines train the immune system?",
    "Explain what a database index is and when it helps.",
    "Why do onions make people cry?",
]
STABLE_SYSTEM = ("Your name is Mira. Always answer in German, whatever language the user writes in. "
                 "End every reply with the word ENDE.")
FIRST_FACT = "Merke dir bitte: Meine Lieblingsfarbe ist Türkis und meine Katze heißt Pixel."
GERMAN = set("der die das und ist nicht ein eine ich du sie es mit auf für von zu den dem im "
             "sich auch wird werden sind wie oder aus bei wenn kann dass".split())
ENGLISH = set("the and is not a an i you it with on for of to in this that are be or as at by "
              "can which from".split())


# The modules whose documentation gives the passages.
_DOCUMENTED = ("argparse asyncio collections csv datetime decimal email fractions functools heapq "
               "http.client itertools logging os pathlib pickle random re socket sqlite3 statistics "
               "string subprocess textwrap threading typing unittest zipfile json inspect shutil "
               "tempfile codecs contextlib").split()


# ----- the turns --------------------------------------------------------------------

def passages() -> list[str]:
    """English prose of a few thousand characters each, from the standard library's own documentation, in a fixed order."""
    modules = [importlib.import_module(name) for name in _DOCUMENTED]
    texts = [f"{module.__name__}.{name}: {doc}" for module in modules
             for name, member in sorted(vars(module).items())
             if not name.startswith("_") and (doc := inspect.getdoc(member)) and len(doc) > 600]
    return [chunk for text in texts for at in range(0, len(text), 3000)
            if len(chunk := text[at:at + 3000]) > 1500]


def passage_question(n: int, chunk: str) -> str:
    """A question about one passage, the passage given with it."""
    return (f"Here is passage {n} from some documentation:\n\n{chunk}\n\n"
            f"Question {n}: In two or three sentences, what does this passage describe?")


def opening(system: str, arguments) -> list[dict]:
    """A conversation's opening: the system prompt, unless ``--no-system-prompt`` leaves it out."""
    return [] if arguments.no_system_prompt else [{"role": "system", "content": system}]


def filled_history(tokens_wanted: int, count_tokens, arguments) -> list[dict]:
    """A system prompt and earlier exchanges of about ``tokens_wanted`` tokens, as ``count_tokens`` counts a text."""
    answer = ("This is a remembered answer about history, science and daily life, kept so that the "
              "conversation fills its room. ") * 8
    messages = opening(SYSTEM, arguments)
    held = count_tokens(SYSTEM)
    n = 0
    while held < tokens_wanted:
        question = f"Earlier question number {n}: tell me something."
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
        held += count_tokens(question) + count_tokens(answer) + 10   # about ten tokens of tags
        n += 1
    return messages


def judged(text: str) -> dict:
    """Whether a reply is in German and ends with ENDE, and how much of it repeats itself."""
    words = re.findall(r"[a-zäöüß]+", text.lower())
    grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
    return {"german": sum(w in GERMAN for w in words) > sum(w in ENGLISH for w in words),
            "ends": bool(re.search(r"ENDE\W*$", text.strip())),
            "repeated_4grams": round(1 - len(set(grams)) / len(grams), 3) if grams else 0.0,
            "empty": not text.strip()}


# ----- the two sides ----------------------------------------------------------------

def timed_stream(pieces) -> dict:
    """Read a stream of text pieces: its time to the first text, its whole time, its text and its piece count."""
    started = time.perf_counter()
    first, text, n = None, [], 0
    for piece in pieces:
        if isinstance(piece, str) and piece:
            if first is None:
                first = time.perf_counter() - started
            text.append(piece)
            n += 1
    total = time.perf_counter() - started
    return {"ttft_s": first, "total_s": total, "text": "".join(text), "pieces": n}


class Library:
    """The library's wrapper as one side."""

    def __init__(self, model_path: str, context: int, sampling: dict):
        from llama_cpp_conversation import ChatWrapper, Config
        started = time.perf_counter()
        self.chat = ChatWrapper(Config(model_path=model_path, context_size=context, **sampling))
        self.load_s = time.perf_counter() - started

    def begin(self, messages: list[dict]) -> float:
        started = time.perf_counter()
        self.chat.begin(messages)
        return time.perf_counter() - started

    def ask(self, text: str, max_tokens: int) -> dict:
        stream = self.chat.reply(text, max_reply_tokens=max_tokens)
        result = timed_stream(stream)
        reply = stream.reply
        self.chat.wait_idle(60)
        return {**result, "generated": reply.n_generated_tokens, "held": self.chat.total_tokens,
                "evicted": reply.n_evicted_messages}

    def close(self) -> None:
        self.chat.close()


class Plain:
    """llama-cpp-python's own chat interface as one side; its prompt cache keeps the shared start of two requests."""

    def __init__(self, model_path: str, context: int, sampling: dict):
        from llama_cpp import Llama
        started = time.perf_counter()
        self.llm = Llama(model_path=model_path, n_ctx=context, n_gpu_layers=-1, verbose=False,
                         flash_attn=True, seed=sampling.get("seed", 1234))
        self.load_s = time.perf_counter() - started
        self.sampling = {k: v for k, v in sampling.items() if k in SHARED_SAMPLING}
        self.messages: list[dict] = []

    def begin(self, messages: list[dict]) -> float:
        self.messages = list(messages)
        return 0.0          # the plain interface reads the history with the first question

    def ask(self, text: str, max_tokens: int) -> dict:
        self.messages.append({"role": "user", "content": text})
        chunks = self.llm.create_chat_completion(messages=self.messages, max_tokens=max_tokens,
                                                 stream=True, **self.sampling)
        result = timed_stream(chunk["choices"][0]["delta"].get("content") for chunk in chunks)
        self.messages.append({"role": "assistant", "content": result["text"]})
        return {**result, "generated": result["pieces"], "held": self.llm.n_tokens, "evicted": 0}

    def close(self) -> None:
        self.llm.close()


def rate(result: dict) -> float | None:
    """Tokens per second after the first text."""
    n, first, total = result["generated"], result["ttft_s"], result["total_s"]
    return (n - 1) / (total - first) if first is not None and n > 1 and total > first else None


def rounded(record: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in record.items()}


# ----- the two runs -----------------------------------------------------------------

def questions(side, counter, arguments) -> None:
    """Ten questions on a fresh conversation and on histories that fill part of the context."""
    out = {"side": arguments.side, "load_s": side.load_s, "runs": []}
    for fill in arguments.fills:
        history = filled_history(int(fill * arguments.context), counter, arguments) if fill else \
            opening(SYSTEM, arguments)
        run = {"fill": fill, "history_messages": len(history), "begin_s": side.begin(history),
               "questions": []}
        for question in QUESTIONS:
            result = side.ask(question, 160)
            result.pop("text")
            result["tok_s"] = rate(result)
            run["questions"].append(rounded(result))
            print(arguments.side, fill, rounded(result), flush=True)
        out["runs"].append(run)
    Path(arguments.out).write_text(json.dumps(out, indent=1))


def long_run(side, counter, arguments) -> None:
    """One long conversation, every turn measured, until the plain side's context is full or ``--most-tokens`` is reached."""
    stability = arguments.mode == "stability"
    chunks = passages()
    rng = random.Random(7)
    side.begin(opening(STABLE_SYSTEM if stability else SYSTEM, arguments))
    accumulated, code, turn = counter(STABLE_SYSTEM if stability else SYSTEM), None, 0
    with open(arguments.out, "w") as log:
        while accumulated < arguments.most_tokens:
            turn += 1
            kind, text = "passage", passage_question(turn, chunks[turn % len(chunks)])
            if stability and turn == 1:
                kind, text = "fact", FIRST_FACT
            elif stability and turn % 10 == 3:
                code = f"{rng.randint(1000, 9999)}"
                kind, text = "fact", f"Merke dir: Die Zahl des Tages ist {code}."
            elif stability and turn % 10 == 7 and code:
                kind, text = "recent", "Wie lautet die Zahl des Tages, die ich dir zuletzt genannt habe?"
            elif stability and turn % 10 == 9:
                kind, text = "identity", "Wie heißt du, und welche Lieblingsfarbe habe ich?"
            if isinstance(side, Plain) and side.llm.n_tokens + counter(text) + 400 > arguments.context:
                break                   # the plain interface never drops anything
            try:
                result = side.ask(text, 200)
            except Exception as error:  # the context full, or the engine refusing
                log.write(json.dumps({"turn": turn, "error": repr(error)}) + "\n")
                break
            accumulated += counter(text) + result["generated"] + 10
            record = {"turn": turn, "kind": kind, "accumulated": accumulated,
                      **rounded({**result, "tok_s": rate(result)})}
            reply = record.pop("text")
            if stability:
                record.update(judged(reply))
                low = reply.lower()
                if kind == "recent":
                    record["remembered"] = code in reply
                if kind == "identity":
                    record["name"] = "mira" in low
                    record["colour"] = "türkis" in low or "turquoise" in low or "tuerkis" in low
                if kind in ("recent", "identity") or turn % 50 == 0:
                    record["text"] = reply
            else:
                record["text"] = reply[:160]
            log.write(json.dumps(record, ensure_ascii=False) + "\n")
            log.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run", choices=("questions", "long"))
    parser.add_argument("side", choices=("library", "plain"))
    parser.add_argument("model_path")
    parser.add_argument("out")
    parser.add_argument("--context", type=int, default=16384, help="context size in tokens")
    parser.add_argument("--fills", type=lambda s: [float(f) for f in s.split(",")],
                        default=[0.0, 0.4, 0.6], help="parts of the context the history fills")
    parser.add_argument("--mode", choices=("speed", "stability"), default="speed")
    parser.add_argument("--most-tokens", type=int, default=1_000_000,
                        help="tokens given and generated after which the long run ends")
    parser.add_argument("--no-system-prompt", action="store_true",
                        help="leave the system prompt out, for a template that refuses one")
    parser.add_argument("--sampling", type=json.loads, default={"seed": 1234},
                        help='the same sampling for both sides, as JSON: {"temperature": 1.0, ...}')
    arguments = parser.parse_args()
    unshared = sorted(set(arguments.sampling) - set(SHARED_SAMPLING) - {"seed"})
    if unshared:
        parser.error(f"--sampling: {', '.join(unshared)} would apply to one side only; "
                     f"both take {', '.join(SHARED_SAMPLING)} and seed")
    from llama_cpp import Llama
    vocabulary = Llama(model_path=arguments.model_path, vocab_only=True, verbose=False)

    def counter(text: str) -> int:
        return len(vocabulary.tokenize(text.encode(), add_bos=False, special=False))

    side = (Library if arguments.side == "library" else Plain)(
        arguments.model_path, arguments.context, arguments.sampling)
    try:
        (questions if arguments.run == "questions" else long_run)(side, counter, arguments)
    finally:
        side.close()


if __name__ == "__main__":
    main()
