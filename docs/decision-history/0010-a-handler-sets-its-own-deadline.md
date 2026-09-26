# ADR-0010: A handler sets its own deadline

**Status:** Proposed
**Date:** 2026-09-26
**Deciders:** Bill McNeill
**Amends:** [ADR-0008](0008-one-observation-per-cycle.md)

## Context

An agent's deadline today is a fixed interval, given to it when it is wired:
a cycle calls `Handler::timeout` when the interval passes with nothing
waiting, and the next deadline is one interval after that cycle. Nothing in
the tree needs more. The moderator is purely reactive, and a player only
answers requests.

Two things about to be built need more.

**The moderator runs clocks.** Werewolf's phases become timed sessions
([ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md)). A session
closes when its hard limit passes, or when its members have been quiet for a
set time after their last change of mind. The moment a session should close
moves every time somebody points, and a night has three sessions with three
clocks. The moderator knows exactly when it next needs to wake. A fixed
interval does not.

**A player decides when to speak.** A talking player waits a random delay
after hearing something before it considers answering, and composes an
utterance in the cycle after it announces that it is typing
([ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)). Both
are "wake me at this instant", computed by the player from what it has just
observed.

Polling would do it: give every such agent a short fixed interval and let
the handler compare the time against its own deadlines on each tick. That
writes a cycle record per tick per agent, ten a second at a 100 ms interval,
and makes the trajectory mostly a record of an agent checking its watch.

## Decision

**After every cycle, and after its start hook, the loop asks the handler for
its next deadline: an absolute `Timestamp` on the agent's clock, or none.**

```rust
pub trait Handler<D: Domain> {
    // start, handle and timeout as before
    fn deadline(&self) -> Option<Timestamp> { /* the fixed interval, as today */ }
}
```

The default keeps today's behavior, so an agent wired with an interval and
no opinion of its own behaves exactly as it does now. A handler that
overrides it owns its schedule completely, and returning `None` means it
only reacts.

The rest of ADR-0008 is unchanged. A deadline that passes while an event is
waiting still joins that event's cycle, which calls `handle`, and the cycle
record still marks it as woken by the deadline. A handler that keeps its own
deadlines therefore checks them in `handle` as well as in `timeout`: a due
deadline is a fact about the time, not about which method is running.

A handler reads the time from the same clock the loop stamps records with.
That keeps deadlines and stamps in one timeline, and it keeps tests on a fake
clock deterministic.

## Consequences

The moderator can run any number of session clocks by returning the
earliest of them, and a player can schedule its own next thought without
addressing a message to itself, which the router forbids.

A deadline that has already passed when it is returned fires at once. That
is how a handler asks for "the next cycle, whatever else happens", which
[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md) uses to
compose speech in the cycle after announcing it.

A handler that returns a deadline in the past on every cycle spins. That is a
bug in the handler, in the same way as a handler that blocks: the loop does
not guard against it.

## Alternatives considered

### Poll on a short fixed interval

Rejected for the trajectory noise above, and because the interval puts a
floor on how precisely a clock can be kept. A quiet period measured on a
100 ms tick is up to 100 ms late, and that lateness is invisible in the
record.

### Return the deadline with the actions

Every hook would return a pair, actions and next deadline. Rejected because
most handlers never set one, and every one of them would have to return a
value it does not care about. A separate method that defaults does the same
job and costs nothing where it is not used.

### A timer agent the moderator sends requests to

A separate agent that sends "your time is up" events. Rejected because the
wake-up is not an event in the domain: it would appear in trajectories as an
observation the moderator conditioned on, and it would put a second thread's
scheduling between a deadline and the cycle that acts on it.
