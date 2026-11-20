# Streams, cancelling and threads

A `ReplyStream` yields text deltas; its `reply` is the `Reply` once the stream
has ended of its own accord, and `finish()` reads it to its end and returns
that. Each token reaches the cache before the stream yields the next text or
ends, so closing a stream early records exactly what the cache holds. Where the
model runs on the processor alone (`gpu_layers=0`, or a build without GPU
support), a token's text is yielded first and the token decoded right after,
so each word shows one decode step sooner. An abandoned stream holds
the wrapper's lock until it is collected: read it to its end, close it, or use
it in a `with`. To stop a generation from another thread, pass
`cancel=threading.Event()` and set it; the reply ends within one token with
`stop_reason == "interrupted"`. Set before the reply's first word, while its
question is decoded too, which stops within one part of the decode (512 tokens,
or `micro_batch_tokens` where that is more), it ends the request
with nothing changed: no question is added and no older exchange is dropped.
After a tool's result, set before the next first word, it leaves the turn
open with its results, for `reply()` to go on with. Set while registered
tools run, it runs no later call of that round: the calls that ran keep their
results, the rest stay owed. It also stops, within one such part, the decode the
reply before left for the next action, and the wait for a reply another
thread holds, which then ends at once with nothing changed. A tool that reads
the stream of the reply running it is refused with an `OutOfOrderError`, told
to the model as the tool's failure.

Actions that change the cache are serialized by one lock. `close()`, or
`with ChatWrapper(...) as chat:`, waits for a running generation, frees the
model and makes later actions raise, while `conversation()` and what says
where it stands still answer. A reply holds the lock while it is read.
An action on the thread that reads it ends that reply first, as closing its
stream does, with a warning logged, and then goes on. A `begin` refused by
the checks that read its arguments alone (the declarations, the message forms,
what the model can take, tags and tool names) ends nothing; any other refusal,
such as a history the template refuses, one that does not fit, or a
`truncate_last_reply` text that is no prefix of the reply, has ended it
already. A reply handed to
another thread is read by that thread from then on. An action from another
thread waits for the reply, and where nobody reads the reply on for
`unread_reply_timeout_s`, it ends that reply as closing its stream does (a call
already handed over stays owed, and a question then closes its turn) and goes
on.
A tool the reply runs may not act on the wrapper: such an action is refused
with an `OutOfOrderError`, which the model is told as the tool's failure, and
changes nothing, a `close()` included. A reply asked for and not yet read when
the wrapper closes raises that it is closed once it is read. A
tool runs on a thread of its own, and one that has not returned within
`tool_timeout_s` is answered with a `tool_timeout` failure, and what it returns
later is dropped; with `tool_timeout_s=None` it runs on the reply's thread.

## From asyncio

Every action blocks its thread until it is done, and waits while another
action holds the lock, so an event loop calls them through a thread of its
own. A whole reply:

```python
import asyncio

async def ask(chat, question: str) -> str:
    reply = await asyncio.to_thread(lambda: chat.reply(question).finish())
    return reply.text
```

A reply streamed to an asynchronous reader, stopped when the reader goes away:
the stream is made and read on one worker thread, which hands each text to
the event loop, and leaving the loop sets the reply's `cancel`, which ends it
within one token.

```python
import asyncio
import threading

async def stream(chat, question: str):
    loop = asyncio.get_running_loop()
    pieces: asyncio.Queue = asyncio.Queue()
    cancel = threading.Event()

    def generate():
        try:
            with chat.reply(question, cancel=cancel) as reply:
                for text in reply:
                    loop.call_soon_threadsafe(pieces.put_nowait, text)
        finally:
            loop.call_soon_threadsafe(pieces.put_nowait, None)

    worker = asyncio.ensure_future(asyncio.to_thread(generate))
    try:
        while (text := await pieces.get()) is not None:
            yield text
    finally:
        cancel.set()
        await worker              # raises what the reply raised
```
