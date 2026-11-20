# Verify

```bash
python -m pytest -q                                  # no model needed
python -m pytest -q -n auto                          # the same, on every processor core
python -m pytest -q -m slow                          # many new random sequences of actions
python -m pytest -q -m gpu                           # real model files, named as below
python examples/demo.py /path/to/model.gguf          # against a real model
```

The suite also runs the linter and the type checker, and checks the writing
rules of the prose and that every file the contract names exists. Every test
has a time limit; past it the run stops and prints where each thread stood.
`LLAMA_CHAT_CHECK_EVERY_ACTION=1` runs the suite with `check_cache` on in every
wrapper, the tests that count the library's work left out.

The random sequences are drawn by Hypothesis: a failing one is cut down to the
shortest that still fails and printed step by step, with what replays it. The
suite draws the same ones every run; `-m slow` draws new ones,
`LLAMA_CHAT_SLOW_EXAMPLES` of them, 1500 by default. Nothing of either is kept
on disk by Hypothesis; a sequence once found failing is kept in
`tests/found_sequences/` instead, as what it drew and its steps, and replayed
every run.

The tests on real model files read their paths from `LLAMA_CHAT_TEST_MODELS`,
separated as the system separates paths, and offload `LLAMA_CHAT_TEST_GPU_LAYERS`
layers, all by default. Each runs a long conversation with `check_cache` on and
bounds how much the process grows; `LLAMA_CHAT_TEST_GPU_MEMORY_COMMAND`, a shell
command that prints the MiB the process `{pid}` holds on the GPU, bounds GPU
memory too; the decision record
[Faults show themselves](../decisions/0048-faults-that-show-themselves.md) holds one.

`./install.sh --dev` switches on `hooks/pre-push`, which refuses a push the
suite and a shorter slow run do not pass. Mutation testing,
`python -m tests.mutation_run` and then `mutmut results`, changes the code on
purpose and lists each change no test noticed; it runs for hours, and writes its
copies and results to `mutants/`. A later run tests again what changed since: a
changed library function, and whatever a test edited, added or removed since
reaches, all of it where code the tests share changed. The tests that
read the source text or trace functions by name are left out of it, since they
would read the tool's rewritten copy (`[tool.mutmut]` in `pyproject.toml`).

The tests run every chat template in `tests/templates/` with a stand-in
tokenizer, one token per character and one per tag, and again with one that
merges a space with the letter or digit after it; each test file's docstring
says what it checks. The demo loads a real model with its own tokenizer and
prints how many tokens each action decodes; add `--gpu-layers -1` to run it on
the GPU.
