# 0052. The conversation and the wrapper hand each job to a collaborator; their actions stay where they are

Date: 2026-11-20

## Context

`rendered_cache.py` had grown to about 2,600 lines and `wrapper.py` to about
1,650, each class doing several jobs at once. A reader had to know the whole
file to change one part of it.

## Decision

- Each job that reads and writes a known part of the conversation's state is
  a collaborator object in a module of its own, as `CacheSync`, `Eviction` and
  `Windows` already were:
  - `cache_targets.CacheTargets`: what the cache should hold for a list of
    messages, from a window of the newest of them where one stands for the
    whole, and its length.
  - `cache_holding.CacheHolding`: bringing the cache to a target, the anchors
    and the states kept, and the render decoded ahead for the next question.
  - `opening_checks.OpeningChecks`: what a template's renders allow for one
    opening, kept for the openings worked out last, and the room a short
    question needs on the longest date.
  - `message_admission.MessageAdmission`: making way for a new message and
    checking that it is written and fits.
  - `kept_turns.KeptTurns`: the turns kept as generated, which a begin keeps
    and which are given up.
  - `reply_rounds.ReplyRounds`: a reply's generations, round by round, and
    the results of the calls they made.
- A collaborator is given the shared `ConversationState`, the rendering and
  the cache, and a callable for each answer only its owner can give (the
  template in use, whether a turn is open). It owns no part of the
  conversation's state, so an action put back puts back everything. Two
  exceptions hold no state an action changes: `OpeningChecks` keeps caches of
  checks already worked out, and `ReplyRounds.answers_waiting`, the answers
  the engine failed to write, moved there from the wrapper unchanged.
- Moved code keeps its text; only the names it is reached by change. A
  check that only some templates need stays where the call is made, so that
  rule 5 holds as before.
- The actions themselves (begin, ask, prompt, generate and record, results,
  truncate, the work after a reply), the undo path, publishing, reply
  generation and the lock stay in `RenderedCache` and `ChatWrapper`.

## Consequences

- `rendered_cache.py` is about 1,750 lines and `wrapper.py` about 1,220,
  most of the latter the public interface's documentation.
- The plain-reply list of the rule 5 test names functions by their new
  modules.
- Each module logs under its own name, so the lines about cut or joined
  questions and closed turns now come from `llama_chat.message_admission`,
  and the tool timeout from `llama_chat.reply_rounds`. The names a caller is
  told to use, `llama_chat` and `llama_chat.engine`, are unchanged.

## Alternatives rejected

- Moving each action out as well: an action combines most collaborators in
  one transaction, so a collaborator for it would be built from ten or more
  others, or call back into its owner.
- Moving reply generation into a module above the conversation: it reads and
  writes about twenty of the conversation's members, which would all have to
  become public.
- Moving the lock and the unread-reply watch out of the wrapper: about sixty
  test lines reach those parts of the threads directly, for a gain of about
  250 lines.
