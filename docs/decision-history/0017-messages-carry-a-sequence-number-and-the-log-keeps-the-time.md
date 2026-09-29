# ADR-0017: Messages carry a sequence number, and the log keeps the time

**Status:** Accepted
**Date:** 2026-09-28
**Deciders:** Bill McNeill
**Amends:** [ADR-0002](0002-jsonl-trajectory-format.md),
[ADR-0007](0007-reinforcement-learning-vocabulary.md)

## Context

ADR-0007 put a `created` time on every message and gave every observation a
`received` time, with `Created` and `Received` traits to keep them apart in
the types and a `latency()` that subtracts one from the other. The times come
from an episode `Clock` whose origin is fixed at the start, and the clock is
threaded through `Wiring`, the environment `Adapter` and `TimerSource`.
`clock.rs` is 206 lines.

Two assumptions sit under that design, and neither holds.

**That an agent should know how old a message is.** Latency is part of what
an agent observes, but the latency it can observe is the one it perceives:
the times at which its observations arrive, relative to each other and to
itself. A message's creation time is a fact about the sender's clock. Once
[ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)
makes arrival mean arrival — a perception thread logs each message as it
comes in, whatever the handler is doing — arrival time is what the agent
has, and it is enough. The handler is told when each observation arrived,
and a policy that wants the current time reads it.

**That the log should be easy to read as written.** An episode's log is
always post-processed, and the framework has no opinion about how. What it
writes is a record of exactly what happened, not a presentation.

The `created` time also does a second job: it is half of how an observation
record is joined to the action record it came from (sender and creation
time). Removing it needs a replacement for that.

## Decision

**A message carries its sender, its recipients, a per-sender sequence number,
and its payload, and no time. Times exist only in the log: each record carries
the instant it describes, and the log writer turns that into nanoseconds
since the episode's origin.**

### Sequence numbers

When the runtime turns a `Send` action into a message, it gives the message
the next number in its sender's sequence. One send to five recipients is one message and one number. The sender's action record and each recipient's observation record
carry `(sender, seq)`, so a single action joins to all of its observations
without reference to time, and two messages sent in the same instant cannot
tie.

A reminder takes its number when it is set, and the message it becomes at
its deadline carries the same number, so setting a reminder and receiving it
join like any other send and receipt.

A relayed message is a new message with the relaying actor's own sequence
number. Its `Envelope` carries the original `(sender, seq)`, so a relay joins
back to the action it relays the same way. Following one utterance relayed by
a moderator takes both keys:

| Record | Written by | Outer `(from, seq)` | Envelope `(from, seq)` |
|---|---|---|---|
| action: alice → moderator | alice | (alice, 7) | — |
| observation | moderator | (alice, 7) | — |
| action: moderator → bob, carol | moderator | (moderator, 42) | (alice, 7) |
| observation | bob | (moderator, 42) | (alice, 7) |
| observation | carol | (moderator, 42) | (alice, 7) |

### An envelope has one shape everywhere

An `Envelope` sits inside a game's payload, where the framework never looks.
So that a parser can follow relays without knowing any game's payload
format, an envelope serializes the same way in every application, wherever
in a payload it appears:

```json
{"envelope":{"from":"alice","seq":7,"payload":{...}}}
```

The single key `envelope` marks it, and `from`, `seq` and `payload` are the
original message's. The framework still reads nothing inside a payload; it
only fixes how its own type is written, so that any tool can find relays by
shape.

### Where times come from

An actor reads `Instant::now()` at the moment something happens — a message
arriving, a message sent, a reminder coming due, a policy call starting or
ending — and puts that `Instant` on the log record. It does not convert it.

**The episode has one clock.** Before it creates the log writer or starts any
actor, the `Episode` captures a single `Instant`, the episode's origin, and
wraps it in a `Clock`:

```rust
#[derive(Clone, Copy)]
pub struct Clock { origin: Instant }

impl Clock {
    pub fn origin(self) -> Instant;
    pub fn offset(self, at: Instant) -> Duration;   // since the origin; zero if earlier
}
```

Every actor and the log writer get a copy of the same `Clock`. The writer
writes each record's instant as `clock.offset(t)` in whole nanoseconds. An
actor gets its copy through its `start` hook
([ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)),
so anything it measures from the origin, such as the `[m:ss]` stamps in a
language-model player's prompt, is on the same timeline as the log, and a
prompt line can be matched to its log record by time. Actors are started by
the environment's `Start` command, so their `start` hooks run a little after
the origin; that does not matter, because every offset is measured from the
shared origin and not from when a hook ran.

`Clock` is only the origin. Times are `Instant`s everywhere; there is no
timestamp type, and a clock never appears on a message.

The origin is not a convenience for reading. Rust's `Instant` is monotonic
and has no epoch, so it can only be written as an offset from another
`Instant`. The one time that can be written absolutely, `SystemTime`, is the
wall clock, which can be adjusted in the middle of an episode and can go
backwards. So the log uses monotonic offsets within the episode, plus one
wall-clock anchor.

### The header

The log's first line is an `episode` record carrying the wall-clock start as
Unix nanoseconds. It is the only wall-clock time in the log, and it is there
so that post-processing can line episodes up against each other, or against
a model provider's logs, if it ever needs to.

```json
{"type":"episode","start_unix_ns":1790630400000000000}
{"type":"observation","actor":"bob","t":1200345,"from":"moderator","seq":0,"payload":{...}}
{"type":"cycle","actor":"bob","t_start":1200501,"t_stop":3901120034,"from":"moderator","seq":0}
{"type":"action","actor":"bob","t":3901122877,"seq":0,"recipients":["moderator"],"payload":{...}}
```

An `observation` record is a message coming in, written by its receiver; an
`action` record is a message going out, written by its sender. The two names
mean incoming and outgoing and nothing more. The log does not pair them into
anything; that is the parser's job.

A cycle is one call of `policy` or `step`: the observation it was called
with, named by its `(from, seq)`, and when the call started and ended. A
`start` call is a cycle with no observation. Actions a lazy iterator yields
during the call may be logged before the cycle record, since each is logged
as it is sent; they belong to the cycle whose window contains them. The record types for undelivered and unsent messages, which
ADR-0016 brings back, carry the same fields as the observation and action
records they would have been. A reminder still held at a `Stop` is an
undelivered record.

### Line order

Records arrive at the writer in whatever order the channel delivers them from
different threads, so line order in the file carries no meaning. Order is
given by `t`, and by `seq` within a sender.

### The module is `log`

`trajectory.rs` becomes `log.rs`, and `LogRecord` becomes `Record`. A
trajectory, in the reinforcement learning sense, is one agent's observations
and actions paired up, which is what a parser builds from the log; the
runtime writes a record of what happened and builds nothing. The writer,
its `Sink`s and `JsonLines` stay as they are, apart from the writer holding
the episode's `Clock`: it converts each record's instant to an offset before
handing the record to its sinks, so a live view and the file show the same numbers.
The `Woken` mark on cycle records goes, since there are no timeouts to
record.

### What is removed

From `clock.rs`: `Timestamp`, `Created`, `Received`, `latency()` and
`Episode::with_clock`; and the `created` and `received` fields on messages,
observations and controls. `Clock` stays, reduced to the episode's origin and
an `offset` from it, and is passed only to the log writer and to each
actor's `start` hook.

## Consequences

**A recipient cannot tell how long a message was in transit.** In-process,
with a perception thread that never waits, that interval is microseconds, and
the part of it that ever mattered — the time a message sat unread while its
agent thought — no longer exists.

**Messages are smaller and have no clock in them**, so a game's payload types
and tests say nothing about time unless the game is about time.

**Reading a log by eye needs a tool.** Offsets in nanoseconds from an origin
are exact and unfriendly. That is post-processing's job, which is where the
framework already puts everything else it has no opinion about.

**The join between an observation and its action changes shape**, from
sender and creation time to sender and sequence number. Anything that reads
logs, including Werewolf's transcript, joins on `(from, seq)`, and a relay
joins through its envelope's `(from, seq)`.

## Alternatives considered

### Keep a `sent_at` in the message envelope

Would let a policy compute a message's age, and keep offline features and
online observations the same if age ever became a feature. Rejected because
the agent's perceived latency is arrival time, which it already has, and
because under ADR-0016 there is no longer a delay between sending and
arriving worth knowing about.

### Log raw wall-clock nanoseconds

No origin, nothing to convert. Rejected because the wall clock is not
monotonic, and a record of what happened must not have time run backwards.

### Let the writer capture its own origin

The first draft of this record: the writer captured an `Instant` when it was
created, and actors read `Instant::now()` with no origin of their own.
Rejected because actors need the origin too. A language-model player stamps
its prompt with times since the start of the game, and if each player
measured from its own start those stamps would be offset from one another,
and from the log, by each actor's start latency.
