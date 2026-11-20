# 0050. Each decision is one record; the dated log is an archive

Date: 2026-11-20

## Context

The reasons behind the code were kept in one dated log, `design_history.md`,
that grew to about 6,900 lines. A later entry often revised an earlier one, so
the decision as it stands had to be pieced together from several entries. The
entries also carried the identifiers of review pages, which mean nothing to a
reader outside those reviews.

## Decision

- Each decision that stands is one record in `docs/decisions/`, numbered in
  order of foundation: its context, the decision, its consequences and the
  alternatives it rejected. The index is `docs/decisions/README.md`.
- A decision made or revised gets a new record, or its record is updated, in
  the same commit as the code. A record states the decision as it stands, not
  the history of how it was reached.
- The code points to a record by its file name.
- The old log is kept unchanged in `docs/decisions/archive/` and is no longer
  added to.
- Measurements, with the machine they were taken on, and model names may
  stand in the records: rule 3 and the prose test exempt the folder.

## Consequences

- A reader finds the reason for a line one file away, without reading the
  revisions that led to it.
- Records carry no review identifiers or round bookkeeping.

## Alternatives rejected

- Keeping the dated log and adding to it: each decision stays spread across
  entries.
- Deleting the log: the reasoning behind superseded choices, and the
  measurements of the time, would be lost.
