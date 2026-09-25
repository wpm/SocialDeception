# ADR-0009: One queue, and an agent that does not know it is being stopped

**Status:** Accepted
**Date:** 2026-09-25
**Deciders:** Bill McNeill
**Amends:** [ADR-0007](0007-reinforcement-learning-vocabulary.md)

## Context

ADR-0007 gave each agent two queues and rejected carrying controls on the
event queue, in one sentence:

> Rejected because a control queued behind a batch of events waits for them,
> and a `Stop` behind a slow cycle waits for the cycle.

Neither half survives.

**There is no batch.** ADR-0008 made a cycle handle one observation, so
"queued behind a batch" describes nothing. What is left is a `Stop` queued
behind *n* events, which the episode does not produce: it holds a `Stop`
until its in-flight count reads zero, which is to say until every event it
routed has been handled. A `Stop` behind unhandled events happens only when
the episode has already failed and is abandoning the run.

**A `Stop` behind a slow cycle is not the message loop's problem.** A
handler that blocks for thirty seconds is a defect wherever it appears, and
the place to fix it is inside the handler — in whatever it is blocking on —
not by teaching the loop to interrupt it. Cooperative cancellation put the
cost of that defect in every agent's loop, for every cycle, forever, to
mitigate one nobody has yet written.

And the mitigation has no users. `Cancel` is passed to every handler in the
tree and read by none of them: `Seat`, `Moderator` and Collatz all ignore
it. The only code that calls `is_cancelled` or `receiver` is the tests
written to prove the machinery works.

The machinery is not small. `cancel.rs` is 438 lines, `Cancel` appears in
ten files, and it brings `ControlSender`, `Arm`, `Trip`, `Signal` and the
`Never` type with it. It also brings the `dropped` record: a cycle a `Stop`
preempts sends nothing, and its actions are logged as `dropped` instead. No
game has ever produced one — the seed-26 fixture has zero, and 40 seeds were
scanned without finding any — because an agent is only stopped once it is
idle.

## Decision

**An agent has one queue, and it does not know it is being stopped.**

A `Stop` is delivered like anything else and is handled when the agent
reaches it. An agent runs until then, as though it were killed at that
moment rather than asked to wind down: it is not told that a stop is coming,
it cannot act on the knowledge, and no handler is offered a way to.

This removes:

- the control queue, and with it `ControlSender`, `Arm`, `Trip` and `Signal`;
- `Cancel`, `Never`, and the `cancel` parameter on `Handler::handle`,
  `Environment::handle` and `Werewolf`'s `Policy::choose`;
- the `dropped` record and everything that reads it, since nothing is
  preempted;
- the post-handler control check, and the one place a cycle's inputs did not
  all share its `t_start`.

`Control` stays what it is — out-of-domain data on the wire, `Start` and
`Stop`, logged as a `control` record and acted on by the loop rather than
the handler. What changes is how it travels and what it can interrupt,
which is nothing.

One thing moves rather than goes. The post-handler drain was also where a
late `Start` was caught, so deleting it deletes that check, and the check
has to be rebuilt somewhere. It becomes a flag on the loop: an agent
records that it has started, and a second `Start` fails against that rather
than against its position in a cycle. This is stronger than what it
replaces, which is the point. The old check fired only for a `Start` that
arrived while a handler was running, because that is the only path it sat
on; one that arrived between cycles ran the start hook a second time and
said nothing. With one queue there is no "during a cycle" for a positional
check to key on, and the rule was never about timing anyway: an agent is
started once, before anything is addressed to it.

### What an agent's queue carries

One channel carrying both kinds. That is an enum on the wire again, which
ADR-0007's `Delivery` was and which this record reinstates deliberately:
ADR-0007's objection was to an `Event` that *meant* three unlike things at
once — a message, an instruction, a timer wake-up — not to a transport that
carries two clearly separate ones. `Observation` and `Action` stay what
ADR-0007 made them, and a control is still not an observation.

## Consequences

The loop gets smaller and the guarantees get simpler. Everything in a
cycle's `inputs` shares its `t_start` again, with no exception to state or
check. A cycle either observed one event or was woken by its deadline, and
what it returns is always sent.

**Shutdown becomes messier, and that is accepted.** An agent may observe and
answer events that were queued ahead of a `Stop`, which on the abandon-ship
path means it does work for an episode that has already failed. Keeping a
trajectory tidy through a failure is the environment's job: it decides when
to stop whom, and it can stop an agent cleanly by stopping it when nothing
is in flight, which is what the episode already does on every path but
failure. If agent shutdown needs to be tidier later, that is a problem to
solve then, with the case in hand.

**A blocking handler now blocks its episode.** Nothing interrupts it, so a
handler that waits thirty seconds delays its own stop by thirty seconds.
This is the cost this record accepts, and the reason it is acceptable is
that the fix belongs where the blocking is: a model-backed policy should
bound its own call, stream its response, and give up on its own deadline.
A policy that can do that needs nothing from the loop; one that cannot is
not made correct by the loop cancelling it.

## Alternatives considered

### Keep the two queues

Rejected because both of ADR-0007's reasons are gone and no third has
appeared. A split kept for a reason that no longer holds is a split the next
reader must reconstruct an argument for.

### Keep `Cancel`, drop the second queue

The trip could hang off the single sender. Rejected because cancellation is
the part with no users: keeping the mechanism and removing the thing that
made it awkward is backwards. If a future handler needs to be interruptible,
it will come with the case that says what interruption should mean, which is
better evidence than this record has.

### Run the handler on a worker thread

ADR-0007 already rejected this, and its reasoning stands: it costs a thread
per agent and still cannot stop the work, since an abandoned model call runs
and bills to completion. Noted here only so that removing cooperative
cancellation is not read as an argument for it.
