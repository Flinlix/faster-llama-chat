# faster-llama-chat

--8<-- "README.md:lead"

## Why

--8<-- "README.md:why"

## Compared with llama.cpp's context shift

Both use the same cache operations; the policy differs.

| | llama.cpp's server | faster-llama-chat |
|------|------|------|
| what is removed | a fixed token count from `n_keep`, which can cut inside a message or a tag | whole exchanges, from a user turn to the next one, so the cache always holds turns the template writes |
| the system prompt | `--keep N`, a hand-counted offset | pinned, never evicted |
| when | when the context is full | above a threshold (`evict_above_fraction`), or above a count of exchanges (`max_history_exchanges`); a request that would not fit evicts first |
| caches that cannot shift | context shift off or unreliable | detected at load; such a cache is cut, restored to a saved state or decoded again instead, and an eviction there drops well below the threshold and below `max_history_exchanges`, so that this happens only every few turns |
| bookkeeping | none at message level | the conversation as the messages its template is given; the cache compared with the template's render of them |

## Quick start

--8<-- "README.md:quick-start"

How a reply streams, how to stop one, and how the wrapper behaves across
threads: [Streams, cancelling and threads](guide/streams-and-threads.md).
