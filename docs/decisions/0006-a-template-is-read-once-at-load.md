# 0006. A template is read once at load into one record, and refused where reading it would need a guess

Date: 2026-11-19

## Context

The library needs a few facts about a template that the render alone does not
give: where a reply ends, the thought tags, the call format, what the template
does to message text, whether windows of the conversation stand for its whole
render. These were once worked out in loose fields, some again on every
reply, and some by rules about angle brackets that held only by the texts'
chance. Where a fact could not be read, a fallback guessed it.

## Decision

- Everything a load reads from a template is one frozen `TemplateProperties`
  record, built at the end of the load in a module of its own. Everything
  after the load reads properties only from the record. Each test template's
  file states the expected value of each property, and
  `tests/test_real_templates.py` compares the record with it field by field.
- The load check tests what a generation reads, in each reasoning mode: that
  the generation prompt ends the render of a question, and that the closing
  after a reply follows a past reply's text.
- The closing after a reply must tokenize into the model's own tags; a reply
  ends at the first of them. A closing that opens with plain text is refused.
  An empty closing is accepted only where the tag that opens the next turn
  ends a generation, which is a fact of the vocabulary (`ends_generation`).
- The user and assistant tags are read from a conversation of two user turns,
  because a first turn may carry text written once and a newest reply what
  only the newest carries. What a template writes once after the last message
  belongs to no turn. An empty system prompt is no system message.
- Tags are the model's own tokens. Where two template texts share a beginning,
  the shared part stops where one of the model's tags ends, with no other tag
  running across that point, or where a word ends at whitespace; it never stops
  inside a tag. No bracket character is written into the code for this. The
  tag list a context gives is checked where the model is adopted: a probe
  render is tokenized, and a tag the tokenizer reads that the list leaves out
  refuses the context, naming it.
- What the template does to a message's text is learned in one render with
  every tag text and every character an escaping filter changes, each between
  numbered marks, read in one pass.
- A system prompt is accepted wherever no later question moves it, read from
  two renders that differ only in the oldest question. One that moves with the
  newest question is refused.
- Render checks run at `begin`, at a change of tools and at a change of date:
  an older exchange must stay written as it was once another follows
  (otherwise keeping it would mean decoding it again each turn), windows must
  stand for the whole render (0026), and whether a past tool turn stays
  written is read.
- A template whose reply cannot be split off a render is refused at load, with
  that cause; "the model ships no chat template" is said only where there is
  no template text. A render that raises during a probe is a refusal of that
  probe.
- The model's tags are read from the vocabulary in one pass of the backend.

## Consequences

- Measured on an i9-9900K: reading the content rewrites took gemma-3-1b's
  load (6,419 tag texts) from 0.40 s to 0.0034 s, and the load from 1.0 s to
  0.6 s. The vocabulary scan went from 0.096 s to 0.080 s.
- On GLM-4.7 Flash with llama.cpp, `<|user|>`, `<|observation|>` and
  `<|endoftext|>` end a generation and `<|assistant|>` does not, so its empty
  closing is accepted.
- Gemma 3, which writes the system text at the head of the first user turn,
  is accepted; Ministral 2410, which writes it before the newest question, is
  refused.

## Alternatives rejected

- Guessing the assistant tags as the user tags where no reply could be split
  off: a guess, which the rule against approximating rules out.
- Refusing a system prompt wherever the system tags equal the user tags: it
  refused Gemma 3, whose cache holds the system text correctly.
- Reading a tag's end as `>` or `]`: two test template families end their tags
  otherwise.
- Parsing the names a template reads with a plain `jinja2.Environment`: it
  failed on 7 of 52 test templates.
- A template-wide splice probe of three questions at `begin`: it added no
  guarantee beyond checking each question on its own text.
