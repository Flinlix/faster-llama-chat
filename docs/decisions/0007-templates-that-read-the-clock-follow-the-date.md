# 0007. A template that writes the date follows the clock

Date: 2026-11-19

## Context

Some templates write the current date into the conversation's opening. Once
the date changes, the render changes, and so must the cache, while every
other template must not pay for the check (0003).

## Decision

- Only a template whose text names `strftime_now` is taken to read the clock;
  a template whose text cannot be read is taken to read it.
- The opening's render notes each form it wrote the clock in and what it wrote.
  The opening is rendered again only once one of those forms reads otherwise
  now. Then the conversation is rendered whole once and decoded again from the
  date on, and the render checks run again (0006).
- `begin` measures the room a short question needs on the day whose date and
  time, in each form the template writes the clock in, take the most tokens,
  plus each month's last day where the template names the months itself, and
  remembers the result for the same opening and year. A date the template
  refuses is skipped.
- A question waiting for its reply is cut again where the date has grown and
  the conversation no longer fits with it; a question cut before keeps its
  one cut mark.
- A template that refuses certain dates outright is left as it is: the library
  does not work around faults in templates.

## Consequences

- The model always reads today's date.
- A system prompt that fits a short question on every day of the year is the
  one `begin` accepts, so the date cannot later leave no room for a question.
- The longest date takes 13 renders.

## Alternatives rejected

- Freezing the date per conversation: exact without the decode, but the model
  reads yesterday's date until the next `begin`.
- Resolving the template again at each `begin`: nothing the template is read
  for holds the date.
- Rendering the opening on every action to see whether the date moved:
  replaced by noting the forms the render wrote.
