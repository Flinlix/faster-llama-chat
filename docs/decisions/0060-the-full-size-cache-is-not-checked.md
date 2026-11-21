# 0060. The full-size cache keeps the gap llama.cpp leaves in its window layers; a kept conversation keeps every position

Date: 2026-11-21

## Context

- When llama.cpp looks for room for new tokens, it reuses any cell its window
  layers no longer attend over, in the full-size cache too (`find_slot`, read
  in the source of binding 0.3.37). After an eviction, the window layers of
  Gemma 4 E2B held 30 to 250 positions beyond the window (measured on
  2026-11-21, a context of 4,096 tokens). A cut further back, or a shift that
  removes tokens near the end, then leaves them without positions the next
  token attends over, and the engine reports no error. A check of the window
  after every turn in the long real-model test failed on the full-size cache
  with binding 0.3.37: Gemma 4 26B A4B at turn 45, after a shortened reply;
  gpt-oss 20B at turn 92 and, once cuts were checked, at turn 100, after a
  shift.
- The gap is the oldest part of the window, only in the window layers, about
  as long as the cut reaches past what they still hold: usually 10 to 200
  tokens (estimated from the measured margin). A reply after such a gap
  matched a fresh begin in the one case compared.
- Saving one sequence's state leaves out the positions its window layers no
  longer attend over (`state_write`, read in the source of bindings 0.3.36 and
  0.3.37). A kept conversation that was restored and then cut back lacked
  positions 82 to 108 (Gemma 4 E2B, measured).

## Decision

- The full-size cache is not checked: a cut or shift after an eviction may
  leave the window layers a gap, which is documented, not repaired. Only the
  compact cache checks its cuts (0059).
- A kept conversation is saved and restored for every sequence (sequence id
  `-1`), so the save keeps every position; the library decodes one sequence
  only.
- The long real-model test checks the window after every turn on the compact
  cache, and a second test switches away from a conversation and back.

## Consequences

- No action on the full-size cache reads the whole conversation again because
  of the gap, which keeps a slow processor's answers prompt.
- The model answers from a cache whose window layers may miss a stretch, so it
  does not answer exactly as after a fresh begin.
- A kept conversation of about 3,000 tokens takes 52.4 MiB instead of 23.5 MiB
  (Gemma 4 E2B, measured), so the same bound keeps about half as many.

## Alternatives rejected

- Checking every cut and shift on the full-size cache as well, and reading the
  conversation again where the window falls short: built and measured on
  2026-11-21, correct, but a shortened reply after an eviction would read the
  whole conversation again, minutes on a Raspberry Pi 5.
- Marking every decoded cell with a second sequence, which llama.cpp never
  reuses: the window layers then held every position (measured), but an
  engine created for two sequences read 1,809 tokens in 18.0 s instead of
  6.1 s on 4 CPU threads (Gemma 4 E2B, measured twice), whether or not any cell
  was marked.
- Keeping the smaller state of a kept conversation: the switch back would
  leave the window layers short.
