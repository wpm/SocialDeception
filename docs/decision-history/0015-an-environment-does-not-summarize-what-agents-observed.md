# ADR-0015: An environment does not summarize what its agents already observed

**Status:** Accepted
**Date:** 2026-09-28
**Deciders:** Bill McNeill
**Amends:** [ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md),
[ADR-0014](0014-events-carry-information-controls-carry-instruction.md)

## Context

Under [ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md) a
pointing session closed with a **tally**: a narration of every member's
latest target, addressed to the session's observers — the living by day,
the pack at night — and naming, for a day, the *hammer* whose point
completed the majority.

[ADR-0014](0014-events-carry-information-controls-carry-instruction.md)
took the first bite out of that. It held that an event from the
environment to an agent is that agent's *observation*, and that a message
which informs its recipient of nothing is not one. On that ground it
withdrew the tally from a session of **one** member, which had been
telling a lone seer, doctor or wolf the single thing that player had just
said. It kept the tally for a session of more than one, on the reasoning
that "a tally to a session of two informs both": a wolf could not
otherwise see where its packmate had landed.

That reasoning has since expired. A point now goes to the moderator and
to nobody else, and the moderator **forwards** each point it accepts to the players the point names as its observers. A point that
arrives after its session has closed is dropped and forwarded to nobody.
The audience of a day point is every other living player; the audience of
a `Devour` is the rest of the living pack. Those are exactly a tally's
recipients.

So by the time a session closes, each of its observers has already watched
it converge, point by point, in the order the moderator accepted them —
and each knows its own latest point because it made it. The tally had
become a message whose every word its recipient had already heard.

Three further things were true of it:

- **The hammer was recoverable.** A day ends on the point that completes a
  majority, and that point is forwarded before the `Eliminated { cause:
  Lynched }` it causes. The hammer is the sender of the last `Nominate`
  the moderator passed on before the lynching.
- **It was doing a job it should not have had.** The doc comment said a
  tally "marks the close of the session it belongs to, which is how its
  members and the transcript know a point arriving later is late." That is
  trajectory-as-state, which ADR-0014 rule 3 rules out: *state lives in
  the agent that needs it, not in the trajectory.* The moderator already
  keeps its sessions and drops late points itself. No player acts on when
  a session closed.
- **It was read as a record.** The integration checks asked a tally what a
  session had decided — using a message to players as a place to keep a
  record, the same habit ADR-0014 named and only half cured.

ADR-0014 listed "let the environment narrate whatever a reader might want"
among its rejected alternatives, and noted that the tally was the status
quo it was rejecting. This finishes that job.

## Decision

**An environment does not send an agent a summary of what that agent has
already observed.**

Concretely:

1. `Narration::Tally` is deleted. No narration is sent when a session
   closes. A night session's close produces the seer's `Investigated` and
   nothing else; a day's close produces the `Eliminated` that a majority
   called for, or a `NoLynch`.
2. A session's close is the **moderator's own business**. What its members
   learn is the session's *outcome*, and that is announced anyway:
   `Eliminated`, `NoDeath`, `NoLynch`, `Investigated`.
3. The **hammer is not narrated**. The moderator uses it internally to
   name the player who dies. A reader of the trajectory recovers it as the
   sender of the last forwarded `Nominate` before the lynching, which is
   the same thing every living player saw.
4. A player that wants the history of a phase **keeps it itself**.
   `Knowledge` archives each phase's points when the next phase begins,
   into `Knowledge::history`, built from the points the agent observed and
   the points it made. It receives no summary, and remains a pure fold
   over observations and its own actions.
5. A reader that wants to know what a session decided reads the
   **points**, not a narration. A forwarded point is the trajectory's
   record that the moderator accepted it.

### Whether a last-second point counted

A player is **not** told whether a point it sent in the last moments of a
session was counted, and that is by design rather than an omission. An
agent has its observations and its actions; the moderator's bookkeeping is
neither. A player that pointed and saw no forward of its own — it never
would, since a point is not forwarded back to its author — and then
observed the phase end learned exactly what the rules give it: the
outcome. Telling it "your point was too late" would be telling it
something about the environment's internal state, which is the thing this
record is against.

## Consequences

- **Fewer messages, and every one of them informative.** The seed-26
  fixture loses thirteen narrations and every player's observation stream
  becomes strictly things it did not already know.
- **`Knowledge` gains a field and loses one.** `tallies: Vec<Heard>`
  becomes `history: Vec<Phased>`, and `Knowledge::acted` now folds the
  agent's own point into `points` — the one entry no forward can supply,
  since a point is never forwarded to its author. The state is still a
  pure function of observations and actions.
- **The transcript reads the hammer from the trajectory's actions.** It
  renders the same summary line as before, and on the regenerated seed-26
  fixture it reports the same hammer. The reader keeps a little more state
  — the latest nomination passed on — which is the cost of taking a fact
  from the primary record instead of from a message that restated it.
- **The integration checks read points.** "A night tally goes to the
  werewolves who cast it" becomes "a `Devour` is passed on to the living
  pack alone", and "an elimination is of a player the tally names most"
  becomes "of a player the phase's *counted* points name most", where
  counted means forwarded. A point with no audience — a lone werewolf's
  `Devour` names nobody to see it — is never forwarded, so for those the
  check reads the points the moderator heard, which is as much as the
  trajectory says.
- **Seed-26 fixtures are regenerated.** The game they record is
  unchanged: the same deaths, the same findings, the same winner in the
  same four rounds, and the same hammer. Only what was said about it is
  smaller.
- **A reader wanting a vote breakdown must count.** Nothing hands one a
  per-session total any more. That is the point: the environment records
  what happened and a reader does its own arithmetic, rather than the
  environment narrating a derived quantity so that a reader need not.
- **The next such message should not be written.** The test this record
  leaves is a question to ask of any narration: *could its recipient have
  derived this from what it has already observed?* If yes, it is not an
  observation, and it does not go in the trajectory.

## Alternatives considered

**Keep the tally as a close marker.** Rejected: it is trajectory-as-state
by ADR-0014 rule 3, and no player acts on it. The moderator that needs to
know when a session closed already knows.

**Keep only the hammer, dropping the votes.** Rejected: the hammer is as
derivable as the votes, from the same forwards. A narration that exists to
save a reader one lookup is the rejected alternative ADR-0014 already
named.

**Tell a player whether its late point counted.** Rejected above: it is a
fact about the moderator's bookkeeping, not about the game the player is
in.

**Keep the tally for the pack alone.** Rejected: the pack is precisely the
case forwarding already covers. A `Devour` names the rest of the living
pack as its observers, so a wolf sees each of its packmates' points as
they are accepted — which is more than a tally gave it, and sooner.
