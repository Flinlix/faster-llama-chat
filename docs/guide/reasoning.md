# Reasoning

```python
chat.begin([{"role": "system", "content": "You are a helpful assistant."}], reasoning=True)
reply = chat.reply("Why is the sky blue?").finish()
reply.text                  # the answer alone
reply.thought               # what the model thought before it, without its tags
reply.n_reasoning_tokens    # about how many tokens the thought took
```

Reasoning is chosen per conversation, at `begin`, and changed by beginning
again with the conversation, because a template may switch it in the system
turn. `reasoning=False`, the default, asks for none; `reasoning=True` for the
template's own way of reasoning; and `reasoning=True` with an effort level,
such as `reasoning_level="low"`, for that level, where the template knows it.
A level without `reasoning=True` is refused, naming `reasoning_level`.
`capabilities(tools=...).can_reason` says whether `begin` accepts
`reasoning=True`, which it does where the template marks a thought and its
prompt with reasoning on leaves that thought open, and `reasoning_levels` lists
the levels `reasoning_level` takes. They are the levels the template tells apart,
read from the template when the model loads, each one writing the prompt
reasoning on writes; where the template writes any level it is given as it is,
the levels are those the model's [add-on file](add-on-files.md) names. A level
the template does not know is refused, naming `reasoning_level`. Reasoning
off is the template's own off: `can_turn_off_reasoning` in the same record says whether the
template has one. Where it has none, as on a template that always reasons, a
conversation begun without reasoning is written as the template writes it,
and the model may reason in every reply all the same. Where a template's
switch cannot be read from the template alone, the model's
[add-on file](add-on-files.md) names the template variables that turn it on.
Every
stream yields the answer alone, with or without reasoning, and a header a
template writes for another reader is kept out of it too. Once the answer has
begun, a thought opens only where the model writes its own thought tag, not
where the answer spells the tag's letters, and any other tag of the model's
that it writes inside its answer is left out of the stream. Stop strings are
matched against the answer only. A finished reply is kept as the template
writes a past reply, without its thought where the template leaves one out,
and with it, handed over in the form the template reads one in, where the
template keeps it. An answer that holds the thought tag's text is kept as
the part of it the template writes as text, and as an answer with no text
where that is nothing. Where that differs from what was generated, the cache is brought to that form
right after the reply, while it is read. A turn that called tools keeps the
tokens it was generated as. A thought the template keeps comes back from
`conversation()`: a reply's as `{"role": "assistant", "content": text,
"reasoning_content": thought}`, which `begin` takes as well, a reply that
called tools included.
