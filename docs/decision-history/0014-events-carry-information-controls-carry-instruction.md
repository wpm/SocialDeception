# ADR-0014: Events carry information, controls carry instruction

**Status:** Accepted
**Date:** 2026-09-27
**Deciders:** Bill McNeill
**Amends:** [ADR-0004](0004-moderator-agent-runs-the-game.md),
[ADR-0005](0005-policy-separates-decisions-from-rules.md),
[ADR-0007](0007-reinforcement-learning-vocabulary.md),
[ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md)

## Context

ADR-0007 named the framework's terms: an event from the environment to an
agent is that agent's *observation*, and an event from an agent is its
*action*. The names were adopted; the discipline behind them was not
enforced, and Werewolf drifted in two directions at once.

**Messages that inform nobody.** Under
[ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md) a pointing
session closes with a tally of its members' latest points, sent to those
members. For a pack that is the whole point: a wolf sees where the others
landed. For a session of one it is a message telling a player the single
thing it has just said. Six of the seed-26 fixture's thirteen tallies were
of that kind. A tally was also being read back out of the trajectory by
checks that wanted to know what a session decided — using a message to
players as a place to keep a record, when every point was already in the
trajectory as an observation the moderator made.

**Messages that inform nobody, in the other direction.** The moderator
issues each player a `Request` at the start of a phase: an event, typed as
an observation, carrying `{id, round, kind}`. A player already knows the
round and the phase, from the `PhaseBegan` it has just observed, and it
knows its own role. So it already knows what it will be asked.
`Knowledge::observe` says as much by ignoring a request entirely, and
`roles::base_action_space` says it louder: it computes the action space
from the player's own knowledge and *asserts* that the request's kind is
the one that player's role is asked in that phase. The request tells the
player what the player already knows, and the assertion checks that it
does.

Worse is what the request does to the shape of the system. To issue one,
the moderator reads each player's role, works out what that player would
do tonight, and tells it to do that. A werewolf never decides to hunt: it
is a reflex arc, pulled by a request it answers. In the live game nobody
waits to be asked. The moderator says that night has fallen, and the
werewolves start pointing because they are werewolves and it is night.

These are the same mistake in two directions: treating the event stream as
a place to put whatever needs to move between agents, rather than as the
record of what each agent knew and did.

## Decision

**An event from an environment to an agent is an observation and nothing
else: a fact about the world that the agent conditions on. An event from
an agent is an action and nothing else. An agent that needs state keeps
it; the event trajectory is not its memory. When an environment must make
an agent do something that transfers no information, that is a control.**

Four rules, in the framework and in every domain built on it.

### The heuristic

**An episode's event trajectory should read as a narrative.** Read the
events in order: each one should be a sentence in the story of what
happened, and together they should be the whole of it.

Werewolf's night reads as one when every line earns its place:

> Night 1 begins, with seven living. Carol protects alice. Dave points
> at alice and erin at bob. Grace investigates alice. The pack is split,
> dave to alice and erin to bob. Grace learns that alice is Village.
> Nobody dies.

Each sentence is somebody learning something or doing something, and the
reader who has watched the roles being dealt needs nothing else to follow
it.

This is the quickest way to apply the rules below, because a narrative
and a training signal want the same thing: what an agent learned, what it
did, in the order it happened.

### 1. Every event to an agent informs it

An event an environment sends is something the recipient did not already
know and can condition on. A message whose content the recipient could
have derived from what it has already observed is not an observation; it
is noise in the trajectory, tokens in a prompt, and a fact stated twice
that can be stated inconsistently.

The test is not "is this useful to send?" but "does the recipient learn
something?" A tally to a session of two informs both; a tally to a session
of one informs nobody and is not sent.

### 2. Every event from an agent is an action

What an agent emits is what it chose to do. Nothing an agent sends is
bookkeeping, and nothing it sends is addressed to itself.

### 3. State lives in the agent that needs it, not in the trajectory

An environment that must remember something remembers it. It does not
narrate it so that it, or a reader, can recover it later. The trajectory
is the record of what agents observed and did; it is external, and it is
not any agent's working memory.

A reader that wants to know what a session decided reads the points the
agents sent, which are actions, and the requests they answer. It does not
need a summary narrated for its benefit, and one sent for that reason
would be an environment writing notes to itself in public.

### 4. Instruction without information is a control

`Start` and `Stop` exist because an episode must sometimes make an agent
*do* something rather than *know* something. That is what a control is
for, and a domain that needs its own is not thereby licensed to send an
event instead.

Werewolf, as it turns out, needs none. A request was instruction with no
information, and the information the player needed — that a phase had
begun — it already had.

## What changes in Werewolf

**`Request` goes.** A player decides to act from what it observes. On
`PhaseBegan` a living player consults its own role: a werewolf points at
night, a seer investigates, a doctor protects, and every living player
points by day. This is [ADR-0005](0005-policy-separates-decisions-from-rules.md)'s
separation carried one step earlier — a role already computes the action
space the rules permit it, and now it also decides whether the rules ask
anything of it at all.

**The moderator stops simulating its players.** It opens each phase's
sessions, runs their clocks, and accepts the points that arrive. It no
longer reads each player's role to work out whom to prod.

**A point names its session rather than echoing a request id.** Both sides
derive the session from the round and the kind of act, which they each
know, so there is nothing to correlate and no id to mint.

**The moderator checks a point when it arrives.** It knew who it had asked;
now it checks a point's sender against the role and the phase, which is
the same rule applied on receipt rather than in advance. A point from
somebody the rules do not ask remains a bug in a player and still panics.

**A session of one is sent no tally**, and the checks that read a tally to
learn what a session decided read the points instead.

## Consequences

**The trajectory is a cleaner training signal.** Every observation in it
is something the agent did not know. A policy trained on it is not
learning to ignore messages, and a model-backed player
([ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)) does
not spend tokens reading them.

**Players get harder and environments get simpler,** which is the right
direction. Deciding when to act is the agent's job. The environment's job
is the rules of the world and the clocks.

**M3 gets easier.** A model-backed player that decides when to act from
what it observes is what
[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)'s
scheduler already describes. The request was scaffolding that would have
had to come out.

**Existing trajectories do not read.** A trajectory with `Request` records
is not one this code can replay, and the seed-26 fixtures are regenerated.

**This is a rule about every domain, not about Werewolf.** Collatz
satisfies it already: its environment sends steps, which are facts, and
its agents send steps back. A future domain gets the same test.

## Alternatives considered

### Make a request a domain-specific control

Keep the request and move it off the event stream: give `Domain` an
associated control type beside `Payload` and `Reward`, so a request is
logged as a control and the loop hands it to a hook of its own.

Rejected because it keeps the reflex arc. The problem with a request is
not where it is written down but that the moderator is deciding for the
player. Once a player decides for itself there is nothing left for the
request to carry, and a mechanism for domain controls would exist with no
user. If a domain ever needs instruction without information, this is the
shape to reach for — but Werewolf is not that domain.

### Leave the request and document that it informs nobody

Cheapest, and it is what the code already half-does: `Knowledge::observe`
ignores a request and `base_action_space` asserts its kind is redundant.
Rejected because a record that has to explain why part of it is
meaningless is a record that should be smaller, and because the reflex arc
survives either way.

### Have the moderator ask only players it cannot predict

Keep requests, but skip the ones whose answer the moderator could work out
— which is all of them, since the moderator computes each action space
already. Rejected as a reductio: noticing that every request is
predictable is the argument for removing requests, not for removing some
of them.

### Let the environment narrate whatever a reader might want

The rejected half of rule 3, and the status quo for tallies. It is
convenient: a summary in the stream is easier to read back than a join
over actions. Rejected because it makes the trajectory the environment's
memory, and because a summary can disagree with the events it summarizes,
at which point a reader has two answers and no way to choose.
