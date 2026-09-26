# ADR-0011: Werewolf phases are timed pointing sessions

**Status:** Accepted
**Date:** 2026-09-26
**Deciders:** Bill McNeill
**Amends:** [ADR-0004](0004-moderator-agent-runs-the-game.md),
[ADR-0005](0005-policy-separates-decisions-from-rules.md)
**Depends on:** [ADR-0010](0010-a-handler-sets-its-own-deadline.md)

## Context

In the rules of [ADR-0004](0004-moderator-agent-runs-the-game.md) a phase is
one round of requests. The moderator asks every player who acts in the phase
for one move, and the phase resolves at the moment the last one answers.
Nobody sees anybody else's choice until the tally, nobody can change their
mind, and the phase takes exactly as long as its slowest player.

That was right for a uniform-random policy, which answers instantly and has
nothing to negotiate. It is wrong for players backed by language models,
which are what Werewolf is for. In the live game, werewolves agree on a
victim without a word, by pointing and re-pointing until the pack settles,
and the village votes by raising hands while the argument goes on, changing
them as it turns. A moderator who waited for the slowest player would wait
for a model that has timed out, and a phase that ends at the last answer has
no room for a change of mind.

So a phase needs a clock, and a vote needs to be something a player can
revise.

## Decision

**A phase is made of sessions. In a session, players point at a target
whenever they like, a player's most recent point is its vote, and the
session closes on a clock.**

### Sessions

A session has:

- **members**, who may point;
- **observers**, who see every point as it is made;
- a **quiet period** and a **hard limit**, both set in configuration.

A point is an event naming one target from the member's action space, which
is the one [ADR-0005](0005-policy-separates-decisions-from-rules.md) already
defines for the corresponding request, including the doctor's constraints. A
member may point any number of times while the session is open. Pointing
nowhere is how a member abstains, so `Abstain` is no longer a move.

A night session closes when every member has pointed and no point has
changed for the quiet period, or when its hard limit passes, whichever comes
first. Any change of mind restarts the quiet period. A point that arrives
after its session has closed is ignored: it lost a race and is not a bug.

### Night

A night is three sessions running at once, each with its own quiet period
and hard limit, so that a slow role cannot spend another role's time.

| Session | Members | Observers |
|---|---|---|
| Pack | living werewolves | the pack and the moderator |
| Seer | the living seer | the moderator |
| Doctor | the living doctor | the moderator |

The seer is told what it found when its own session closes. Nobody speaks at
night, so learning early gives it nothing to act on until day, and a seer
devoured that night still learns what it learned, as before.

The night resolves once every session has closed. The victim is the
plurality of the pack's latest points, and a first-place tie is broken by the
moderator's seeded generator as before. If no wolf pointed, nobody is
devoured. The doctor's protection then applies, and a death, or nobody's
death, is announced to the living.

### Day

A day is one session. Every living player is a member, and every living
player and the moderator observe it, so the tally is public as it forms.

A day has no quiet period. It closes the moment a **majority of the living
players**, more than half of them and not merely of those who have pointed,
point at the same player. That player is lynched. The point that makes the
majority is the *hammer*, and casting it is a move in its own right: it ends
the argument. If the hard limit passes without a majority, nobody is
lynched.

Talk during the day is
[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md).

### Stalemate

A day that can end without a lynch removes what guaranteed termination: in
ADR-0004 the day always eliminated someone. A doctor who keeps saving and a
village that keeps running out the clock could go on forever. So a **day
cap**, set in configuration, ends a game that reaches it as a stalemate,
which is a new kind of outcome. ADR-0004's round cap, which that record kept
only as a guard, becomes the rule.

A stalemate pays **−1 to every player**, the same as losing. Stalling can
then never beat losing, and a side that is ahead has every reason to finish.

### Clocks

The moderator keeps every session's clock and wakes on the earliest of them,
using [ADR-0010](0010-a-handler-sets-its-own-deadline.md). `Game` stays a
pure state machine: the moderator tells it when a session's time is up, so
the rules remain testable with a scripted sequence of points and deadlines
and no threads.

## Consequences

**Determinism survives for policies that do not deliberate, with a
narrower statement.** A random player points once, as soon as its session
opens, and never changes its mind. A night session then closes only after
all its members have pointed, so its tally does not depend on the order the
points arrived in. A day closes at the first majority, and since nobody
changes their point, no later point could have made a different one, so
arrival order changes *when* the hammer falls but never *who* is lynched.
What does vary is which late points arrive before a session closes, and
their order in the record. ADR-0005's guarantee becomes: for a fixed
configuration and seed, every phase's outcome is identical on every run,
namely each death, each finding and the winner, but the sequence of points
is not. This holds as long as every random point lands inside its hard
limit, which at limits measured in seconds is a matter of microseconds of
thread latency. Model-backed games are not reproducible, and nothing here
tries to make them so.

**The golden tests change.** They pin the old rules and will be rewritten
against the new ones.

**The theoretical baselines assume no cap.** Win rates reported against them
also have to report the stalemate rate. If it is ever large, the cap is
shaping the result.

**A phase takes wall-clock time.** A game of random players used to take
milliseconds. With the example's limits it takes as long as its quiet
periods, which for a batch of random games is a cost to keep in mind when
choosing them.

## Alternatives considered

### Keep one-shot requests, with a deadline

The smallest change: ask as now, and resolve on the last answer or the
limit. Rejected because it rules out negotiation. The pack's pointing is how
wolves agree without talking, and the village's is how an argument turns
into a vote.

### One night session for every role

Rejected because it ties the seer's and the doctor's time to the pack's. A
seer who decided at once would wait for an arguing pack, and a pack that
argued long would spend the seer's limit.

### A fixed delay after the last first point

Close a night session a fixed time after every member has pointed once.
Rejected because a change of mind near the end of the delay races the
close. With a quiet period that restarts on every change, the session closes
only once the members have actually settled.

### Lynch the plurality when the day runs out

Rejected. Forcing a lynch on a weak plurality at the buzzer would make
running out the clock a way to get somebody killed, and it would make the
hammer matter less. A village that cannot agree loses its day.

### A stalemate that pays zero

Rejected because a side that is losing would prefer a stalemate to a loss,
and would have a reason to stall.

### Tell the seer at the end of the night

Rejected because it gains nothing. The seer cannot act before day either
way, and waiting only couples its result to the other sessions.

## See also

[`docs/visualization/timed-rounds.html`](../visualization/timed-rounds.html)
animates one night and one day under these rules.
