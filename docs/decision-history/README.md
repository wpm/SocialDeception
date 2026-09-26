# Decisions

This directory holds **Architecture Decision Records (ADRs)** for Social
Deception — short documents that each capture one significant decision: the
context that forced it, the choice made, the alternatives weighed, and the
consequences accepted.

An ADR is a point-in-time record, not living documentation. Once accepted, an
ADR is not rewritten when the world changes; instead a new ADR supersedes it
and the old one is marked `Superseded`. The trail of records is the value — it
tells a future reader *why* the system is the way it is, including the roads
not taken.

For the canonical description of the practice, see Michael Nygard's original
article, [Documenting Architecture Decisions](https://cognitect.com/blog/2011/11/15/documenting-architecture-decisions),
and the community hub at [adr.github.io](https://adr.github.io/).

## Conventions

- One decision per file, named `NNNN-short-title.md` with a zero-padded
  sequence number (`0001-...`, `0002-...`). Numbers are never reused.
- Status is one of `Proposed`, `Accepted`, `Deprecated`, or `Superseded`.
- When a decision replaces an earlier one, set the old ADR's status to
  `Superseded` and link the two.
- Each record should stand on its own: a reader who has seen no other document
  should be able to follow it. Links are for depth, not for meaning.
- Keep them short. An ADR that needs many pages is usually several decisions.

The numbering starts over here. This repository is a fresh start, and records
from the earlier attempt at the project are not part of its history.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-single-process-thread-per-agent.md) | A single process per episode, one thread per agent | Accepted |
| [0002](0002-jsonl-trajectory-format.md) | Trajectories are JSON Lines of event and cycle records | Accepted |
| [0003](0003-event-record-stamps.md) | An event record stamps its time as arrived, sent or due | Superseded by [0007](0007-reinforcement-learning-vocabulary.md) |
| [0004](0004-moderator-agent-runs-the-game.md) | A moderator agent runs the game, and the seam with the episode is construction | Accepted |
| [0005](0005-policy-separates-decisions-from-rules.md) | A policy separates decisions from rules | Accepted |
| [0006](0006-knowledge-holds-what-the-agent-did-in-secret.md) | Knowledge holds what the agent did in secret | Accepted |
| [0007](0007-reinforcement-learning-vocabulary.md) | The framework speaks the vocabulary of reinforcement learning | Accepted |
| [0008](0008-one-observation-per-cycle.md) | A cycle handles one observation | Accepted |
| [0009](0009-one-queue-and-no-cancellation.md) | One queue, and an agent that does not know it is being stopped | Accepted |
| [0010](0010-a-handler-sets-its-own-deadline.md) | A handler sets its own deadline | Accepted |
| [0011](0011-werewolf-phases-are-timed-pointing-sessions.md) | Werewolf phases are timed pointing sessions | Accepted |
| [0012](0012-a-dead-player-stops.md) | A dead player stops | Accepted |
| [0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md) | Speech and typing are events, and a scheduler decides when to speak | Accepted |
