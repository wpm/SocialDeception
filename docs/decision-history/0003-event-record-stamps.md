# ADR-0003: An event record stamps its time as arrived, sent or due

**Status:** Superseded by [ADR-0007](0007-reinforcement-learning-vocabulary.md)
**Date:** 2026-09-18
**Deciders:** Bill McNeill
**Amends:** [ADR-0002](0002-jsonl-trajectory-format.md)
**Superseded by:** [ADR-0007](0007-reinforcement-learning-vocabulary.md), whose observation, action and control records carry `created` and `received` times in place of these stamps

## Context

ADR-0002 gives every event record one `time` field. What that field means
depends on the event. For an input it is when the router put the event on the
agent's inbox. For a message the agent itself sent it is when the loop handed
the message to the router. For a `Think` it is the deadline the think was
scheduled for. A reader has to compare the event's `sender` with the record's
`agent` to know which, and nothing in the file says so. Documentation saying so
would be overlooked.

## Decision

**The event record's `time` field is replaced by three mutually exclusive
fields, exactly one of which appears on every event record: `arrived`, `sent`
or `due`.**

| Field | On | Meaning |
|-------|----|---------|
| `arrived` | an input: a control event, or a message from another agent | when the router put it on this agent's inbox |
| `sent` | a message this agent sent | when the loop handed it to the router |
| `due` | a `Think` | the deadline it was scheduled for, at or before the moment the loop noticed it |

The value is a timestamp as ADR-0002 defines it: nanoseconds since the
episode's clock started, as a bare integer.

In Rust the three fields are one enum, `Stamp`, flattened into the record, so
that a record with none or two of them cannot be constructed.

ADR-0002's sample becomes:

```json
{"type":"event","agent":"a","seq":0,"arrived":10,"event":{"kind":"control","control":"start"}}
{"type":"event","agent":"a","seq":1,"arrived":20,"event":{"kind":"message","sender":"b","recipients":["a"],"payload":{"Step":6}}}
{"type":"event","agent":"a","seq":2,"sent":40,"event":{"kind":"message","sender":"a","recipients":["b"],"payload":{"Step":3}}}
{"type":"cycle","agent":"a","t_start":30,"t_stop":50,"inputs":[0,1],"outputs":[2]}
```

Everything else in ADR-0002 stands. Where it says `time`, read `arrived` or
`due` for an input and `sent` for an output.

## Alternatives considered

### Keep `time` and document its three meanings

Rejected because the meaning would live in prose a reader of the file can
skip. The schema should carry it.

### Keep `time` and add a `direction` field

An `"in"` or `"out"` field alongside `time` says which case applies, but does
not separate a think's scheduled time from an observed arrival, and every
reader would carry two fields to interpret one moment.

### Stamp a think with when it was observed

Every stamp would then be a crossing of the agent's boundary. Rejected because
the scheduled deadline would appear nowhere in the log, and a think handled
late after a slow pass would look as if it had fired on time.

## Consequences

- **A reader that wants one moment per event coalesces three keys.** That is
  the cost of a schema that names the moment.
- **The stamp is redundant with the event's direction**, on purpose. A checker
  can assert that `arrived` appears only on a control event or on a message
  whose sender is not `agent`, `sent` only on a message whose sender is
  `agent`, and `due` only on a think.
- **ADR-0002's invariant** that every event a cycle lists as an input has an
  event record with an earlier `time` now reads with `arrived` or `due` in
  place of `time`.
- **Field names remain a contract with the Python side.** This change is made
  before any Python reads a file.
