# ADR-0008: A cycle handles one observation

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** Bill McNeill
**Amends:** [ADR-0002](0002-jsonl-trajectory-format.md),
[ADR-0007](0007-reinforcement-learning-vocabulary.md)

## Context

Since ADR-0001 an agent has handled a *batch*: it drains its whole queue at
the start of a cycle and hands the handler everything it found. ADR-0002
records that as settled rather than deciding it, and builds the two-record
trajectory format on top of it — the cycle record exists partly to say which
observations shared a handling window. No ADR ever states the case for
batching. It arrived with the first implementation and nothing since has
asked whether it earns its place.

Two things say it does not.

**The grouping is the scheduler's, not the domain's.** What arrives together
depends on when threads happen to run. The same game, from the same seed,
produces a different number of cycles on every run — 128, 126 and 123 in
three consecutive runs of the seed-26 fixture, with identical play. A handler
that read meaning into what appeared together would be reading the scheduler.
So a batch cannot be how a game expresses that several messages belong
together: it is not reliable enough to carry the meaning, and nothing in the
codebase tries. Every handler — `Seat`, `Moderator`, `Collatz` — immediately
loops over the slice one element at a time, and
`independent_observations_fold_in_any_order` asserts that the order within a
batch carries no information either.

**Reinforcement learning has no batch of observations.** An observation is
what an agent conditions on at a decision point. A handler given *n* of them
is being asked to make one decision from *n* observations, which the
formalism has no name for. Where a set of messages genuinely is meaningful as
a set, the set is the observation, and assembling it is the game's business:
it knows which messages belong together and a queue does not.

## Decision

**A cycle handles at most one observation, and a timeout is not an
observation.**

```rust
pub trait Handler<D: Domain> {
    fn start(&mut self) -> Vec<Action<D>> { Vec::new() }
    fn handle(&mut self, observation: &Observation<D>, cancel: &Cancel) -> Vec<Action<D>>;
    fn timeout(&mut self, _cancel: &Cancel) -> Vec<Action<D>> { Vec::new() }
}
```

The loop still pops controls before events, and still pops them all: a
control is out-of-domain and the queue order is the whole point of there
being two queues. What changes is the event side. A cycle takes **one** event
and leaves the rest, so an agent with a full queue runs a cycle per event
rather than one cycle for all of them.

A cycle that observed nothing and whose deadline passed calls `timeout`. It
is a separate method because waking on a deadline is not observing anything,
and saying so with an empty slice made "no observation" a kind of
observation. The default implementation does nothing, which is what an agent
without a timeout wants and what every agent in the tree wants today.

**A deadline is not exclusive of an observation.** A deadline that passes
while an event is waiting joins that event's cycle, and that cycle observes
the event and calls `handle` like any other: the observation is what the
agent decides from, and the deadline only says when it decided. So exactly
one hook runs per cycle — `handle` if an observation was popped, `timeout`
if none was and the deadline fired, and neither for a cycle that popped only
controls — and `handle` wins when both are true.

The cycle record's `woken` mark follows the same rule: it says `timeout`
whenever the deadline was what the loop noticed, whether or not the cycle
also observed something. It records when the cycle ran, not what it decided
from, which is what makes it the right thing to measure the next deadline
against.

### What the trajectory says

A cycle record's `inputs` no longer lists a set of observations that shared a
window. It names at most one observation, plus whatever controls the cycle
popped. `t_start` and `t_stop` therefore bracket one decision, which is what
makes the gap between them deliberation time rather than the time to work
through an arbitrary pile.

Nothing else about the format changes. Every record type keeps its fields,
and the join between an observation and the action it came from — sender and
creation time — is untouched.

## Consequences

An agent is stale by at most one decision rather than by however much arrived
while it was busy, and what it was thinking about when it decided is exactly
one thing. A trajectory read for training no longer has to discount groupings
that were the scheduler's doing.

Cycles become more numerous and smaller. That costs a cycle record per event
where a busy agent used to write one for several, and a channel send per
cycle rather than per batch. Both are cheap next to a handler that calls a
language model, which is the case the design is for.

A game that needs several messages taken together must say so in its own
types — one payload carrying them — rather than hoping they arrive in one
drain. That is a constraint, and the right one: it makes the grouping a claim
the game makes rather than an accident the runtime allows.

The change is confined to the agent loop and the handlers. The two-queue
split, the control-before-event order, `Cancel` and the dropped actions of a
preempted cycle are all as ADR-0007 left them.

## Alternatives considered

### Keep the batch

Rejected on the two grounds above: the grouping is unreliable, so no game can
depend on it, and no game does. Retaining a parameter shaped for a use nobody
has invites a future handler to read meaning into it that is not there.

### `Option<&Observation>` for the timeout

One entry point, with `None` meaning the deadline fired. Rejected because it
has the same flaw as the empty slice in a smaller size: it still asks the
handler to pattern-match its way from "an observation" to "not an
observation", and every handler would open with the same unwrapping. A method
that is called only on a timeout says it once, in its name.

### A batch that the game reassembles

Keep the drain, and give games a way to declare which messages to coalesce.
Rejected as the same mechanism twice: a game that can say which messages
belong together can put them in one payload, and then the runtime needs no
notion of coalescing at all.
