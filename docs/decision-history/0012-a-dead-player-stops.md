# ADR-0012: A dead player stops

**Status:** Proposed
**Date:** 2026-09-26
**Deciders:** Bill McNeill
**Amends:** [ADR-0004](0004-moderator-agent-runs-the-game.md),
[ADR-0007](0007-reinforcement-learning-vocabulary.md)

## Context

A dead Werewolf player's agent keeps running until the episode ends. ADR-0004
kept it silent by never addressing it: the moderator sends a dead player the
news of its own death, then nothing until the final `Outcome`, which was
broadcast so that a dead werewolf could observe that its side won.

Two things have changed.

**The outcome no longer has to reach the dead.** Since ADR-0007, a reward is
logged by the environment rather than sent, so a dead player is paid whether
or not it hears the result. The reason for the one broadcast is gone, and
ADR-0007 already withdrew it.

**Players will talk to each other directly.** Speech goes from player to
player, not through the moderator
([ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)). A
speaker works out who is alive from its own `Knowledge`, which can be a
cycle behind the moderator's. So somebody will eventually speak to a player
who has just died, and silence toward the dead can no longer be arranged by
the one agent that knows who they are.

A dead player that is still running is also a thread doing nothing useful,
and a model-backed one that observes the day's talk is a thread spending
money to do it.

## Decision

**When a player dies, the moderator announces the death to the living and to
the victim, and stops the victim's agent. From then on the dead player hears
nothing.**

The victim's `Stop` is sent in the same cycle as the announcement. With one
queue per agent ([ADR-0009](0009-one-queue-and-no-cancellation.md)), the
victim observes its own death before it reaches its `Stop`. Anything queued
between them it also observes, but a player that knows it is dead takes no
further action: players cooperate with the moderator in enforcing the rules.

**A `Stop` to some agents while others run is delivered at once.** ADR-0007
holds every `Stop` until nothing is in flight, so that an agent hears
everything said to it before it stops. That is right for the end of an
episode and wrong here: during a day there is always talk in flight, so a
held `Stop` might never go out, and the point of this one is that the victim
does *not* hear what comes after. The hold stays for the end of the episode.

**An event addressed to an agent the environment has stopped is dropped, not
an error.** The router currently fails the episode with `QueueClosed` when a
recipient's queue is gone, because that means something broke. A queue
closed because the environment stopped its agent is a different case, and
the router records the difference. Delivering to the other recipients
continues as normal. `QueueClosed` stays an error for a queue that closed
any other way.

The outcome stays as ADR-0007 left it: announced to the living, with a
reward logged for every player, living and dead.

## Consequences

A dead player's trajectory ends with the observation of its own death and
its stop, which is where the game ended for it.

An event sent to a dead player now leaves a trace in the sender's trajectory
and none in the recipient's: an action with a recipient who never observed
it. Anything that joins the two sides of an event, including replay, has to
allow for that. It is correct for RL: the recipient did not observe it.

At the end of the episode the environment stops only the agents still
running. An agent already stopped is not sent a second `Stop`.

Stopping one agent in the middle of an episode is new to the shutdown logic,
which until now stopped everyone together. It is a small change, but it
touches the invariant the episode uses to detect a stall: an agent that has
stopped is no longer one the episode waits on.

## Alternatives considered

### Keep dead agents running, and never address them

The current approach. Rejected because it depends on every sender knowing
who is dead at the moment it sends, which a speaker working from its own
`Knowledge` does not. It would also keep a thread, and possibly a model,
running for a player out of the game.

### Let the router filter by a set of dead players

Rejected because the router would then know a game rule. Whether an agent
has stopped is a fact about the framework; whether a player is dead is a
fact about Werewolf. The environment translates one into the other by
stopping the agent.

### Tell the dead the outcome

Rejected because nothing needs it. The reward is logged either way, and an
observation a dead player cannot act on only lengthens its trajectory.
