# Benchmarks

Scripts that measure the library on a model file of your choice. They need
the package installed and a GGUF file; nothing is downloaded. The results are
JSON files; the measurements the design relies on, with the machine they were
taken on, are recorded in the [decision records](../docs/decisions/).

| Script | What it measures |
|---|---|
| `actions.py` | Every public action of the wrapper once or more: its wall time, and inside it the engine's calls (prefill, sampling, token decode, state save and restore, shift, cut, tokenizing), the template's renders and the library's own Python work. A reply also records its time to the first text and its tokens per second. Ends with a profile of a few plain replies. |
| `compare.py questions` | Ten short questions, on a fresh conversation and on histories that fill 40 % and 60 % of the context, for the library and for llama-cpp-python's own chat interface on the same file. |
| `compare.py long` | One long conversation, every turn measured, for both sides. `--mode stability` checks every turn whether the reply still follows the system prompt's three instructions and remembers the facts it was given. |

## Running them

```bash
python benchmarks/actions.py path/to/model.gguf actions.json --gpu-layers -1
python benchmarks/actions.py path/to/model.gguf actions-cpu.json --gpu-layers 0 --threads 8

python benchmarks/compare.py questions library path/to/model.gguf library.json
python benchmarks/compare.py questions plain path/to/model.gguf plain.json
python benchmarks/compare.py long library path/to/model.gguf long-library.jsonl --mode stability
```

Run one script at a time, and each side of a comparison in its own process,
so that every run has the machine to itself. Give both sides the same
sampling with `--sampling '{"temperature": 1.0, "top_k": 64, "seed": 1234}'`;
it takes `temperature`, `top_k`, `top_p`, `min_p` and `seed`, the values both
sides take, and refuses any other. Without it, each side uses its own
defaults. The plain side counts streamed
pieces as tokens, which is close but not exact. For a template that refuses a
system prompt, pass `--no-system-prompt`; `actions.py` leaves it out on its own.
