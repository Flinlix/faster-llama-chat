# Keeping a conversation

The cache holds only the recent window that fits; `conversation()` returns what
it holds as `begin` takes it, the system prompt, tool turns, the thoughts the
template keeps, the declared tools and the reasoning switch included, so a
conversation saved from it and begun again from it reads in as it was:

```python
saved = chat.conversation()
...
chat.begin(**saved)
```

A tool declared there is declared again, not registered: a request whose calls
are to run is still handed its `ToolRegistry`.

**Exact restore, and another model.** The messages are in no model's own
format: a call is its name and arguments, a thought plain text. A turn the
model generated with a call is kept in the cache as the tokens it generated,
which no template writes back exactly, so `conversation()` also gives each
such turn as `generated_turns`: its tokens, and its own messages as the
conversation holds them, a thought and a call the model wrote unreadably
included; `generated_turns()` gives them alone. Given back to `begin` on the
same model, each turn is found by its Chat Completions messages and the
question before them, a thought or an empty text aside, wherever it now stands
among them, so generated turns kept beside messages you add to or drop from
still serve them; it is put back as the conversation held it and restored
token for token. On another model, or without them, every turn is written as
that model's template writes it from the messages alone. `generated_turns` is
JSON-ready and only meaningful to the model that wrote it; reasoning on or off
and the declared tools carry over to another model, its other settings come
from it. It is trusted input: its tokens go into the cache as they stand, the
model's own tags included, and nothing checks them against the messages'
text, so a `generated_turns` someone edited can put any turn, a system turn
included, in front of the model. Give back only what `conversation()` gave,
kept where nobody else can change it; where a saved conversation passes
through hands you do not trust, begin again from its messages alone.

To keep everything, older turns included, keep each message as you send it
and each reply as you read it. A request whose generation fails before it says
anything changes nothing: its question is taken back, and the older exchanges
it pushed out come back. So does a request whose reply the template refuses to
write, and one whose reply produced nothing, where the template writes no
empty reply: one the model ended at once, or one closed for lack of room after
its calls, which takes the turn's calls and results back with it. Where the
template writes an empty reply and two questions in a row, a reply to a
question that has no answer, such as one that only thought, is taken back
alone: its question stays, the next
question follows it, and its thought is not kept.
