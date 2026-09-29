# ADR-0018: In Werewolf, all speech goes through the moderator

**Status:** Accepted
**Date:** 2026-09-28
**Deciders:** Bill McNeill
**Amends:** [ADR-0004](0004-moderator-agent-runs-the-game.md),
[ADR-0005](0005-policy-separates-decisions-from-rules.md),
[ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md),
[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)
**Depends on:** [ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md),
[ADR-0017](0017-messages-carry-a-sequence-number-and-the-log-keeps-the-time.md)

## Context

[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md) addresses
a player's `TypingStarted`, `Say` and `TypingEnded` to every living player,
copied to the moderator, and not routed through the moderator. Each player
works out who is living from its own `Knowledge`.

That spreads one rule — who hears what — across every player. The moderator
is the only participant that knows authoritatively who is alive at a given
moment, and under
[ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md) it is
also the one that stops a player at its death, with a stop that now takes
effect ahead of anything queued.

It also strains the vocabulary. In the multi-agent reinforcement learning
picture the framework follows, agents do not talk to each other: an agent
sends an action to the environment, and the environment sends observations
to agents. Direct speech made a player's `Say` an observation for other
players without the environment having produced it.

## Decision

**A Werewolf player addresses every action to the moderator and nobody else.
The moderator decides who observes it and sends it on.**

- During the day, the moderator sends each player's `TypingStarted`, `Say`
  and `TypingEnded` to every living player except the speaker, who already
  knows what it said.
- The message the moderator sends on is its own message, with its own
  sequence number. Its payload carries an `Envelope` of the speaker's
  message ([ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)),
  so a listener knows who spoke, and the log can join the relay to the
  speaker's action by the envelope's sender and sequence number.
- Players no longer compute recipients for speech. A player's `Knowledge`
  still records what it heard, in the order it observed it.
- The moderator relays with the same determinism rule as everything else it
  does ([ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)):
  it applies its rules after each observation, never waiting for a lull, so the
  point that reaches a majority ends the day even if a switch arrived right
  behind it.
- A `debug_assert!` in the player's code checks that every action is
  addressed to the moderator alone. Players cooperate with the moderator and
  are not assumed to cheat, so this catches mistakes rather than enforcing
  rules.

This is Werewolf's decision, not the framework's. A message may still name
any recipient set, and another game may have its actors talk directly.

## Consequences

**Who hears what is decided in one place**, the moderator's `step`, alongside
the rest of the game's rules: the living, the pack at night, and nobody once
dead.

**Every utterance is logged twice**: once as the speaker's action to the
moderator, and once as the moderator's action to the listeners, which each
listener logs as an observation. Transcript code joins them through the
envelope.

**Speech waits on the moderator.** Relaying adds a hop through the
moderator's perception and handler threads. In-process the hop is
microseconds, and a `step` that relays does no slow work, so a burst of
speech is relayed as fast as it arrives. If the moderator ever makes a slow
call in `step`, speech waits for it; that would be the time to give relaying
its own path.

## What else changes in Werewolf

These follow from [ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)
and [ADR-0017](0017-messages-carry-a-sequence-number-and-the-log-keeps-the-time.md)
rather than from relaying, and are recorded here because they are
Werewolf's.

**Werewolf's `Policy` becomes `Strategy`.** Werewolf already has a
`Policy` trait, `choose(view) -> Option<ActorId>`: the strategy that
[ADR-0005](0005-policy-separates-decisions-from-rules.md) separates from a
role's rules. The framework's `Policy` is the function a user of the
framework implements, and keeps the name. Werewolf's is renamed `Strategy`,
which is the game-theory word for what it is: a choice from the action space
a role permits.

**A player plays a role, choosing with a strategy, from what it knows.**
Today Werewolf's agent is called `Seat`, because the name `Player` was
taken by a trait: the role's rules plus access to `Knowledge`, implemented
by four structs, `Villager`, `Werewolf`, `Seer` and `Doctor`. Those structs
differ in nothing but the `Role` value their `Knowledge` already holds:
each is `{ knowledge: Knowledge }`, every `action_space` is
`base_action_space(&self.knowledge, kind)`, the one role-specific rule (the
doctor may not protect the same player twice running) is decided inside
that function from the role, and which sessions a role joins is
`Role::asked_in`, on the enum.

So the four structs and the `Player` trait go, the rules move onto `Role`,
and the agent takes the ordinary name:

| Concept | Name | What it is |
|---|---|---|
| the participant; the framework's `Agent` | `Player` (was `Seat`) | knowledge, strategy, and the moderator it sends to |
| the rules it is dealt | `Role` | an enum with `asked_in` and `action_space` |
| how it chooses | `Strategy` (was Werewolf's `Policy`) | a choice from the action space its role permits |
| what it has learned | `Knowledge` | its fold of what it observed, including its `Role` |
| the environment | `Moderator` | the framework's `Environment` |

```rust
impl Role {
    fn asked_in(self, phase: Phase) -> Option<SessionKind>;
    fn action_space(self, knowledge: &Knowledge, kind: SessionKind) -> Vec<ActorId>;
}

pub struct Player<S: Strategy> { knowledge: Knowledge, strategy: S, moderator: ActorId }
```

[ADR-0005](0005-policy-separates-decisions-from-rules.md)'s separation is
kept and made plainer: the rules are all on `Role`, the choice is all in
`Strategy`, and `Player` joins them with what it knows. A future role that
needs state of its own would be a reason to give roles types again; none
does today.

**`RequestKind` becomes `SessionKind`.** The `Request` message it was named
for was removed by
[ADR-0014](0014-events-carry-information-controls-carry-instruction.md); what
it names now is the kind of pointing session a phase opens.

**`Player` implements `Policy`, and `Moderator` implements `Step`.** `Player`
folds each observation into `Knowledge` and returns its point, if any, as an
action. The moderator's effects become `Effect`s returned from `step`, and
its opening move comes from `start`.

**Scripted players do not batch.** A `Player` with a scripted strategy decides
after each observation and never waits for a lull, so a seeded game still decides
each phase the same way. Its random generator is drawn from once per
decision, never once per call.

**Pointing sessions time points by arrival.** A session stamps each point
with `observation.at`, not with the time `step` runs, and closes on a
reminder the moderator sets for itself at the session's limit and again on
each restart of the quiet period. A reminder that arrives for a limit that
has since moved is ignored. `Timestamp` becomes `Instant` throughout
`game.rs`.

**The transcript joins on sequence numbers.** `transcript.rs` is Werewolf's
log parser. Its joins move from creation times to `(from, seq)`, and a
relayed utterance joins to the speaker's action through its envelope.

## Alternatives considered

### Keep direct speech with the moderator copied

The status quo. Rejected because it leaves "who is living" to each player's
bookkeeping, and because it is the one place Werewolf's messages do not fit
the action/observation vocabulary.
