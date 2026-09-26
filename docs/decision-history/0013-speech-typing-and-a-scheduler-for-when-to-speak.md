# ADR-0013: Speech and typing are events, and a scheduler decides when to speak

**Status:** Accepted
**Date:** 2026-09-26
**Deciders:** Bill McNeill
**Amends:** [ADR-0005](0005-policy-separates-decisions-from-rules.md)
**Depends on:** [ADR-0010](0010-a-handler-sets-its-own-deadline.md),
[ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md),
[ADR-0012](0012-a-dead-player-stops.md)

## Context

Werewolf's players are about to talk, and there is no turn-taking to lean
on. Seven players hear the same line at the same moment, and in the simplest
design every one of them asks its model whether to answer. Humans have the
same problem and solve much of it by seeing someone draw breath. In a chat,
the equivalent is seeing that someone is typing.

Two parts of the framework shape how that can work.

**A handler's actions go out when it returns**
([ADR-0009](0009-one-queue-and-no-cancellation.md)). A player that decides to
speak and calls its model in the same cycle sends "I am typing" together with
what it typed, which tells nobody anything. The announcement has to leave in
one cycle and the generation has to happen in a later one.

**Deciding when to speak is a different problem from deciding what to say,
and probably a harder one.** The closest prior work, Eckhaus et al.,
[*Time to Talk*](https://arxiv.org/abs/2506.05309) (EMNLP Findings 2025),
splits an agent playing asynchronous Mafia into a scheduler that decides
when and a generator that decides what, and found that its timing matched
the humans' more closely than its content did. The first scheduler here will
be crude on purpose, and it has to be replaceable without touching the part
that writes the words.

## Decision

### Three events

- `TypingStarted { typing }`: the player has begun composing.
- `Say { typing, text }`: what it composed.
- `TypingEnded { typing }`: it has stopped composing, whether or not it said
  anything.

`typing` is a per-player counter, so a sender and a typing id together name
one attempt to speak. A player is one thread and composes one thing at a
time, so its typing events alternate. Read together, they give three cases:

| Seen | Meaning |
|---|---|
| started, say, ended | it spoke |
| started, ended | it gave up |
| started only | it was cut off, by the end of the day or of the episode |

### Who hears it

Speech is public and happens only during the day. `TypingStarted`, `Say` and
`TypingEnded` are addressed to every living player and copied to the
moderator. They are not routed through the moderator. A player works out who
is living from its own `Knowledge`, and one that has just died is dealt with
by [ADR-0012](0012-a-dead-player-stops.md). A player's `Knowledge` records
what it heard, in the order it observed it.

### Speaking takes two cycles

In the cycle where a player decides to speak, it returns `TypingStarted` and
sets its next deadline to now, using
[ADR-0010](0010-a-handler-sets-its-own-deadline.md). The next cycle, whether
the deadline or a waiting event wakes it, calls the model and returns `Say`
and `TypingEnded`, or `TypingEnded` alone if the model passes.

### A scheduler decides when

When a player considers speaking is a **strategy object**, separate from the
policy, chosen in configuration:

```rust
pub trait Scheduler {
    /// Called with each observation; returns when the player should next
    /// consider acting on the conversation, or none.
    fn observed(&mut self, observation: &Observation<WerewolfDomain>, now: Timestamp)
        -> Option<Timestamp>;
}
```

The first implementation waits a random delay, drawn from the player's own
seeded generator, after each thing it hears, then lets the policy decide
whether to say something, point, or pass. The policy never knows which
scheduler is running. A fast gating model, a bidding scheme or whatever the
reading turns up replaces the scheduler and nothing else.

### Text only at the renderer

`Say` carries text today. Nothing outside `Say` and the renderer that turns
a player's `Knowledge` into a prompt may assume it, so that a later
experiment can put something other than language on the wire.

## Consequences

**Typing can stop a player from starting, but not make it yield.** A player
only sees someone else's `TypingStarted` between cycles. Once it is
composing it cannot notice another speaker until it is done, so collisions
will happen, within the window of one model call. That is realistic, and
measuring it comes before building anything smarter. Stopping mid-sentence
would need the interruption ADR-0009 removed, and would come with a case of
its own.

**A talkative day can outrun a player.** Every cycle handles one
observation ([ADR-0008](0008-one-observation-per-cycle.md)), and a cycle that
calls a model takes seconds. If speech arrives faster than a player can
work through it, the player answers an old conversation. The fix, if the
first version shows it, is to make the model call on a side thread and
deliver the result back to the player as a later observation, which also
lets it keep listening while it composes. That would be its own ADR.

**Every consideration is a model call.** With a random-delay scheduler, the
cost of the day grows with how much is said. The scheduler is where a cheap
gate in front of the model belongs, when one is built.

**Seven identical players will behave alike.** Random delays drawn from each
player's own generator, and different prompts per player, are the first
defense against everyone answering at once.

## Alternatives considered

### Send `TypingStarted` from inside the handler

Not possible under ADR-0009: nothing leaves a cycle until the handler
returns.

### A compose event the player addresses to itself

Rejected. The router forbids an agent addressing itself, and a player's
thoughts should not travel the router and appear in trajectories as
observations. A deadline already means "wake me", which is all the second
cycle needs.

### A central arbiter that gives out the floor

In *Werewolf Arena* (Bailis et al., 2024) each agent bids for the right to
speak next and the highest bid wins. Rejected because an arbiter turns the
game back into turns, which is what this framework is built to avoid. A bid
is still a good shape for what a scheduler might compute; it just decides
for one player, not for the table.

### Put the timing inside the policy

Rejected because it couples the part expected to be research-hard to the
part that writes the words. Swapping one would mean rewriting the other, and
a user who only wants to change prompts would be exposed to the scheduling.

### Run the model call on a side thread now

Deferred, not rejected. It is the likely fix for falling behind, but the
two-cycle version is simpler, and whether it is enough is an empirical
question the first version will answer.
