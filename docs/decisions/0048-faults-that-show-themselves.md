# 0048. Faults show themselves: every loop ends, every test is timed, and the cache can be checked against a fresh begin

Date: 2026-11-15

## Context

The owner asked to make the project as error proof as possible and to enforce
as many rules as a machine can. Random action sequences had found a reply
that never ended, on a turn a reply cap cut inside a call.

## Decision

- **Every loop has a variant.** Each `while` loop gives a `LoopVariant` a
  measure at the start of every pass, a whole number of at least 0 or a tuple
  of them, which must be smaller than the pass before; a pass that is not
  raises `InternalError` naming the loop. `tests/test_loops.py` reads the
  source and fails for a loop without one, except three forms that end by
  construction: a loop over an iterator's steps, an endless generator whose
  every pass yields, and a work list popped from a finite value.
- **Every test has a time limit**, 60 s by default, set by `faulthandler`,
  which stops the process and prints every thread's stack.
- **`check_cache`**, a setting off by default, compares the cache with a fresh
  begin of the same conversation on a context that reads the model's tokens
  and decodes nothing: right before a generation, where they must be equal,
  and once the work after a reply has ended, where they may differ in length
  but in no shared token. What it finds after a reply is raised by the next
  action. `LLAMA_CPP_CONVERSATION_CHECK_EVERY_ACTION=1` turns it on for every wrapper of
  the suite.
- **Random sequences by Hypothesis**, whose database is off and whose home is
  the null device, so nothing is kept on disk; a failure prints the blob that
  replays it.
- **The contract and the prose are tested**: every file a rule names exists,
  rules are numbered without a gap, and no em dash, dash between clauses or
  path of one machine appears.
- **Lint and type checks are tests** (ruff with E, F, W and B; mypy with
  optional values checked). `DrawnSampling`, every value set, is the only
  sampling a generation takes.
- **Real model files** are named by the environment: `LLAMA_CPP_CONVERSATION_TEST_MODELS`,
  and `LLAMA_CPP_CONVERSATION_TEST_GPU_MEMORY_COMMAND`, a shell command that prints the
  MiB the process `{pid}` holds on the GPU. On an Nvidia card:
  `LLAMA_CPP_CONVERSATION_TEST_GPU_MEMORY_COMMAND='nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits | awk -F", " "\$1=={pid}{print \$2}"'`.
  Each runs 120 turns with `check_cache` on, cuts and begins again, and bounds
  growth after 30 turns to 64 MiB of RAM and 16 MiB of GPU memory.
- **Before each push**, `hooks/pre-push` refuses a working tree that differs
  from the commit, since the tests read the working tree, then runs the suite,
  200 new sequences and the strict site build; `install.sh --dev` sets it as
  the checkout's hooks folder.
- **Mutation testing** by mutmut leaves out the cache-context protocol (the
  library reads its method names) and the engine modules (only a real model
  runs them).

## Consequences

- With every wrapper checked the suite raised no alarm; a test shows a cache
  holding one wrong token is caught.
- Run on Gemma 3 1B and Llama 3.2 3B on the processor, the real-model tests
  passed in 12 minutes. On the GPU runs, process RAM grew 51 MB over the first
  40 turns and 11 MB over the next 160, GPU memory stayed flat, and closing a
  wrapper gave both back.
- Cost of a variant: a method call and a comparison per pass.

## Alternatives rejected

- A cache check between any two actions: between actions the library leaves
  tokens stale on purpose; 78 alarms in the suite were all such states.
- The pytest-timeout package: a dependency, and the stack is what finds a
  hang.
- A TLA+ model of the thread hand-over: it needs Java and would drift from the
  code; `tests/test_thread_orders.py` runs every order instead.
- A prose check for measured numbers: a measurement cannot be told from a
  documented default or a test value by its form.
