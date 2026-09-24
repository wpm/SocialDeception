# ADR-0002: Trajectories are JSON Lines of event and cycle records

**Status:** Accepted
**Date:** 2026-09-18
**Deciders:** Bill McNeill
**Amended by:** [ADR-0003](0003-event-record-stamps.md), which replaces the event record's `time` field, and by [ADR-0007](0007-reinforcement-learning-vocabulary.md),
which replaces the event record with observation, action, dropped, control and reward records

## Context

ADR-0001 settles that the fold is the log: each agent's own event sequence is
its trajectory, recorded by the agent loop as it folds and sent over an
in-process channel to a writer. It deliberately defers what the writer actually
writes. This record decides that.

Two things have settled since ADR-0001 and shape the answer. First, an agent
handles a *batch* of events per pass: it drains its whole inbox at the start,
whether woken by an arrival or by its deadline. Second, analysis happens in
Python. The Rust side produces trajectories and never reads one back.

## Decision

**A trajectory is a JSON Lines file: one JSON object per line, each object one
of two record types, distinguished by a `type` field.**

### Two record types

ADR-0001 put two times on every event — when it arrived and when it was
handled. With batch handling those two times no longer belong to the same
thing, so they split across two records:

| Record | `type` | Fields |
|--------|--------|--------|
| event | `event` | `agent`, `seq`, `time`, `event` |
| cycle | `cycle` | `agent`, `t_start`, `t_stop`, `inputs`, `outputs` |

An **event record** is one event in one agent's trajectory: the agent it
belongs to, that agent's sequence number for it, when it arrived on the
agent's channel, and the event itself. A **cycle record** is one pass of the
agent's loop: the handling window and, by sequence number, the events that
were in the drain and the events the fold emitted.

How long an agent was busy — the gap ADR-0001 cares about — is recoverable
from an event's arrival time and its cycle's `t_start`. The cycle's input list
additionally records exactly what the agent saw on that pass, which is what
makes a cycle replayable, and it is why there is no per-recipient delivery
record: in a single process, arrival time is send time plus scheduler jitter
and carries nothing the cycle does not already say.

### Sequence numbers cover outputs too

A cycle names its outputs by sequence number, so the agent's sequence space
covers everything the loop recorded, inputs and outputs alike. A message the
agent sends gets an event record on the sender's side, with `time` being when
the loop handed it to the router, and a separate event record on each
recipient's side when it arrives there. Sequence numbers are per agent and
meaningful only together with the agent id.

### Times

Times are nanoseconds since the episode's clock was started, as bare integers.
They come from one monotonic clock per episode, so they are comparable across
agents with no skew to correct, and they are paired with the sequence number
because two events can read the same instant and a training record cannot
tolerate a tie it has no way to break.

### Events

An event is an internally tagged object whose `kind` is `message`, `control`
or `think` — the three variants of ADR-0001's `Event` enum. A message carries
its `sender`, its `recipients` as a sorted list so that two logs of the same
run compare equal, and an environment-specific `payload` the runtime does not
interpret. A control event carries the instruction in a `control` field. A
think event carries nothing but its kind.

Written out, a pass in which agent `a` receives a start and a message and
replies to `b` looks like this:

```json
{"type":"event","agent":"a","seq":0,"time":10,"event":{"kind":"control","control":"start"}}
{"type":"event","agent":"a","seq":1,"time":20,"event":{"kind":"message","sender":"b","recipients":["a"],"payload":{"Step":6}}}
{"type":"event","agent":"a","seq":2,"time":40,"event":{"kind":"message","sender":"a","recipients":["b"],"payload":{"Step":3}}}
{"type":"cycle","agent":"a","t_start":30,"t_stop":50,"inputs":[0,1],"outputs":[2]}
```

### The writer

One writer thread per episode owns the output file and receives records from
every agent over an unbounded in-process channel. It runs until every sender
has been dropped, then flushes and exits. A write failure stops the thread and
drops its receiver, so that the next agent to record something fails at the
point of sending, and the failure is reported again when the writer is joined.
A trajectory that is silently incomplete is worse than one that is loudly
broken.

## Alternatives considered

### One record per event carrying both times

ADR-0001's original shape. It cannot express batch handling: the events in
one drain share a handling window, and putting that window on each of them
loses which events were handled together. The cycle record keeps that.

### A per-recipient delivery record

A record per delivery, written by the router, saying when each copy of a
message was put on each recipient's channel. Rejected because the cycle's
input list already says what each agent saw, and in a single process the
delivery time adds only scheduler jitter to the send time.

### A binary or columnar format

Parquet or similar would be smaller and faster to load. Rejected for now
because trajectories are small, the readers are Python scripts that have not
been written yet, and a format a person can read with `head` is worth more at
this stage than one a machine can read faster. Nothing in the record
structure prevents a later conversion.

### Reading logs back in Rust

Not done. Only `Serialize` is required of a payload; a `Deserialize` bound is
added when something on the Rust side needs to read a log, and not before.

## Consequences

- **A file is one stream for the whole episode**, not one file per agent.
  Per-agent trajectories are recovered by filtering on `agent`; a reader that
  wants one agent must do that filter.
- **Every field name here is a contract with the Python side.** Renaming one
  is a breaking change to every analysis script, and should get an ADR.
- **The trajectory invariants a checker can enforce** without knowing the
  environment follow from this format: every event a cycle lists as an input
  has an event record with an earlier `time`; per-agent sequence numbers are
  contiguous and strictly increasing; no cycle's input list is empty; no
  message has its sender among its recipients.
- **Emitted messages are recorded twice**, once by the sender and once by each
  recipient. That is redundancy, not waste: it is what lets a sender's
  trajectory be read on its own, and the recipient-side record is the one
  that carries an arrival time.
