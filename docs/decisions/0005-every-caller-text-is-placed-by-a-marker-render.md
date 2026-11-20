# 0005. Every text of the caller's is placed by a second render with a marker in its place

Date: 2026-11-19

## Context

The cache must know where each text of the caller's sits in the render, so
that the text is tokenized as text and a tag typed into it stays letters.
Searching the render with `find` took the first match: a message whose text
was template text, such as `<|im_end|>`, made the template's own tag plain
text and the typed one a real turn end, on 49 of 52 test templates. Counting
closings found nothing on templates that write no reply closing.

## Decision

- Each render is made a second time with a marker (`QQ<n>MQQ`, letters and
  digits that trimming, escaping and quoting leave whole) in place of every
  text of the caller's: a content, a thought, a call's name and string
  arguments, and every non-empty `description` of a declaration at any depth.
- The real render must be the marked one with each marker replaced by its
  text as the template writes it: as a content of its role, trimmed, with the
  rewrites read at load applied, or inside a JSON string as
  `json.dumps(text, ensure_ascii=False)` writes it. Failing those, the text is
  what lies before the template's next text, where exactly one place leaves
  the whole render matching. That search is memoized per part and position,
  so it runs in polynomial time.
- A template that writes anything else of a text makes the renders differ,
  and the conversation is refused.
- Empty replies and empty descriptions carry no marker; empty user and tool
  contents carry one.
- A call's end is anchored after its last string argument, else after its
  name.
- A tag in a tool's name, a parameter's name, a choice value or a title is
  refused, since the template reads those as written and decides by them what
  it writes.
- Where texts repeat and a template compares them for equality, a failed
  split is tried once more with one marker per kind, role and text.
- Each text is tokenized with special-token parsing off.

## Consequences

- A typed tag stays text in every message, thought, argument and description,
  on every test template. Tool descriptions, which often come from third
  parties, cannot inject the model's own tokens.
- One more render per render, reused within an action.
- Measured on the stand-in: a MiniCPM 5 history of 11 past calls whose
  arguments it wraps in CDATA took 2.9 s to begin and 3.6 s per reply before
  the memoized search, 0.02 s after; 40 calls take 0.2 s.

## Alternatives rejected

- Searching the render for the text: wrong wherever a text equals template
  text.
- Marking names, types and enum values too: a template reads them to decide
  what it writes, so a marker would change the render around them. They are
  refused where they hold a tag instead.
- The ASCII-escaped and the HTML-escaped JSON forms: over the whole suite they
  matched nothing the remaining form did not.
