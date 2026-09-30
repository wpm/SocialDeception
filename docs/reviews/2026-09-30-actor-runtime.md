# Review: the actor runtime, 2026-09-30

A walkthrough of the M2 actor runtime with the code in front of us, at the
point where milestone M2 was finished and about to land. Nothing here was
changed as a result: the reading happened after the branch was complete, and
the findings are for afterwards.

**This is not decision history.** An ADR records a decision made and the
reasons that forced it; what follows is mostly open questions, and several of
them are the same question seen from different sides. It is filed here so the
reasoning survives until each part is cooked enough to become an issue, and
then — for the ones that change the architecture — an ADR of its own.

## What held up

Most of what we looked at needed no defense, and two things are worth
recording as settled rather than leaving them to be re-litigated.

**Only the environment can command, and the compiler says so.** An agent's
`Policy` returns `Action`, which has exactly two variants, `Send` and
`Remind`. There is no `Command` and no `Reward`, so an agent cannot express
either — it is not rejected at runtime, there is no value it could construct
to try. `Router::command`'s `NotTheEnvironment` check is a backstop for
direct router callers, which the router's own module documentation already
says. The primary guarantee is structural.

**`Episode` does not need typestate.** The suggestion was that building and
running are two phases and should be two types: you add agents, then you run,
and a running episode can be neither rerun nor re-rostered. Both invariants
already hold, by ownership rather than by marker type. `run(self)` consumes
the episode, so there is no second call to make; and `add` takes `&mut self`,
which cannot be obtained once `self` is gone. The test for whether typestate
earns its keep is naming the program that wrongly compiles, and we could not
find one. Note the asymmetry with `UnstartedActor`, which *is* typestate-ish:
both of its states have values. An episode's second state has no value at
all, just a stack frame inside `run`, so there is nothing to give a type to.

## One problem, seen three ways

The largest finding is not three findings. `Seat`'s visibility, a raw control
sender that escapes, and a name nobody can say naturally are one thing.

### The name does not fit the layer

The runtime's vocabulary is domain-neutral throughout — `Actor`, `Episode`,
`Router`, `Message`, `Control`, `Observation`, `Action`. `Seat` is the one
name that imports a game's picture into a layer that has no tables, only a
pair of channel ends. As the reviewer put it: a decent domain term for
Werewolf, but not at the shared protocol level. The cost is not only
stylistic — two layers now use one word for unrelated things, so a grep for
`seat` mixes them.

Werewolf keeps it. Players really are seated at a table: `setup.rs` has a
`seat()` function and `mod.rs` speaks of seating every player. Only the type
and its uses in `router.rs`, `episode.rs` and `thread.rs` are in scope, which
was 15 of the 51 hits a repository-wide search returned when this was written
— before this document added its own.

### The smell was really about visibility

The reviewer kept wanting to say *adds agents and an environment to the
episode*, never *adds seats*. That is diagnostic rather than stylistic. A
caller never passes a `Seat` in and never sees one: the API is
`new(sinks, environment_id, environment)` and `add(id, handler)`. The seat is
*derived* from what was added, in one loop inside `run`, and is not an
ingredient anybody supplies.

`Seat` looks like a local variable that was promoted to a public type. In the
running system it is constructed in exactly one place, the struct literal in
`Episode::run`; `Router::seats` is private and `seat()` is a private method,
so nothing reads one back out. It is `pub` only to shape one argument of
`Router::new`.

### And the visibility is where a control sender escapes

`UnstartedActor::control()` and `Actor::control()` are both `pub` and hand
out a `Sender<Control>`. That sender is checked by neither guarantee above:
not by the types, since it is not an `Action`, and not by the router, since
it does not go through one. `UnstartedActor::new` is `pub` too, so this is
reachable from outside the crate.

`UnstartedActor::control()` has **zero** non-test callers — `episode.rs`
takes the sender from `new`'s return tuple instead — and its documentation
claims it is "how an episode stops an actor it has not started yet", which
nothing does. `Actor::control()` has one caller, feeding the teardown `Stop`
in `join()`, which is legitimate: teardown is not an in-world command and
should not be subject to the environment check.

### So: narrow first, rename second

Make `Seat` `pub(crate)`, or have `Router::new` take the `(id, inbox,
control)` triples and build the map itself. Narrow both `control()`
accessors, and check whether the `UnstartedActor.control` field can go
entirely — `Actor` needs its copy for teardown, the unstarted form appears
not to. One change closes the encapsulation hole and removes the vocabulary
pressure at the same time, because nobody outside the crate would say the
word. Rename only what survives.

Precedent is [#141](https://github.com/wpm/SocialDeception/pull/141), which
narrowed `Writer::spawn`, `Writer::create`, `EpisodeRecord::of` and
`Record::elapsed` to `pub(crate)`, and deleted `Writer::create` once
narrowing had made it dead.

### The idea that does not work, and why

Making the agent/environment difference a type parameter on the *sender* —
`Message` against `Message + Control` — cannot be done. Commands and messages
do not share a channel: a `Seat` holds two, and they are separate for a
load-bearing reason. `Perception::sense` does `controls.try_recv()` *before*
its `select!`, which is what lets a `Stop` jump the queue of whatever is
backed up in the inbox. Union them onto one channel and a `Stop` queues
behind a slow handler's backlog, destroying the property the `bounded(0)`
rendezvous exists to guarantee.

The shape that would work is two capability handles over one router: an agent
port exposing only `send`, an environment port adding `command` and `reward`.
An agent's thread would hold the former and have no `command` method at all.
Worth weighing against its value first, though — it converts the backstop,
not the primary guarantee, and it buys nothing while `Seat`'s fields are
`pub`, which is this section's first half.

## The reward type parameter is residue

`Episode<W, P>` carries a reward type that no field mentions, so it needs
`PhantomData<fn() -> W>` to stay legal — `fn() -> W` rather than `W` so the
episode claims it might produce a reward without owning or dropping one.

`W` is erased almost immediately. `pay()` converts the reward with
`serde_json::to_value` and `RewardRecord.value` is a `serde_json::Value`, so
nothing downstream — log, writer, sinks — is generic in it. It survives one
function call. Both users instantiate it as `i32`.

The shape is not a considered choice but a leftover: ADR-0007 bundled
`Payload` and `Reward` into a `Domain` trait, and
[#132](https://github.com/wpm/SocialDeception/pull/132) deleted `Domain`,
leaving the two parameters threaded individually.

**Preferred: make it `serde_json::Number`.** ADR-0007 says a reward is
numeric — "rewards may be integers or reals, so the reward type is a
parameter" — and carries "no arithmetic bound" because it is "only carried
and serialized". `Number` is exactly that union, and `pay()` already converts
to `Value` one line later, so this makes the destination explicit rather than
adding a step. It removes the parameter *and* the concept: `Episode<P>`,
`Effect<P>`, `Step<P>`, and the `PhantomData` with its variance puzzle goes
with them.

It costs a little ergonomics — `Effect::reward(who, 1)` needs
`Number::from(1)` unless the constructor takes `impl Into<Number>`, which
recovers it — and it gives up a type-level guarantee, since `Number` admits a
float where `W = i32` made that a compile error. Given the ADR's own framing
that guarantee is worth little, but it is not nothing.

**Fallback: an associated type on `Step`.** `type Reward: Serialize` also
removes `W` from `Episode` and puts the reward type where it is decided, the
environment's alone — which is what the `PhantomData` comment already says.
It keeps per-game precision, at the price of wordier definition sites, and
`spawn_environment::<W, _, _>`'s turbofish becomes `H::Reward`, an inference
change to watch with `impl Trait` in return position.

Choose `Number` unless per-game reward precision turns out to matter.

## The cycle record lost its job

A `CycleRecord` was load-bearing when a handler took a **batch**: it was the
only record saying which observations were folded into one call. One
observation per call (ADR-0008) makes `observed` a single optional key, and
that job is gone.

Nothing in the library consumes one. The transcript reader discards
`control`, `cycle` and `episode` outright, on the grounds that a cycle "says
nothing about the game". Every other mention in `src/` is the write path, the
type, the re-export, or a test. So it serves the test suite and offline
analysis; no functionality depends on it.

What it still uniquely carries is the **deliberation window**. `t_start` is
when the observation *arrived*, not when the call began, so the window covers
queuing as well as thinking, and nothing else in the log holds that
quantity. It is the evidence for ADR-0016's claim that perceiving never waits
on deciding: overlapping windows for one actor are the observable signature
of the two threads, and `check_cycles_bracket_their_observations` is the
assertion that rests on it. Remove the record and the claim becomes
untestable from a log.

The reviewer's position is that an agent wanting to log its own thinking can
do so itself, and that no generic mechanism is needed for it. **That needs no
new API**: an action addressed to an empty recipient set is already logged and
delivered to nobody, so an agent can leave a trace by sending to nobody with
a payload that explains itself — which is richer than two bare timestamps,
since it can say which model and how many tokens.

If the record is slimmed rather than removed, `observed` is the weakest field
now: the same `(from, seq)` is on the observation record, so the join can be
made from that side. The window is the part that carries information nothing
else does. The record is really a *deliberation* record now rather than a
cycle in the batch sense, and a rename may be hiding in that.

## Controls should be logged twice, not moved

Today the **receiving** actor's perception thread writes the `ControlRecord`
when it sees the control, so the timestamp is when it was observed and the
record proves receipt.

The gap: an actor that dies, hangs, or is starved never writes one, so a log
cannot tell *never told* from *told and it never arrived* — which is exactly
the hang the episode's time limit exists to catch. The suggestion was that
the episode log starts and stops itself, independently of the agents they are
sent to. The precedent holds: `RewardRecord`'s own documentation notes "the
agent rewarded, which is not the agent that wrote it", so a record whose
writer is not its subject is already established here.

**Do it as two records rather than by moving the one.** The episode records
the command as sent; the actor goes on recording it as seen. A sent-with-no-
seen pair is then the diagnostic, directly visible, and volume is bounded at
two controls per actor. Two records rather than two times on one record,
because ADR-0017 deliberately removed the second time from a record.

Replacing receipt-logging would weaken two invariants. `check_the_join` and
`nothing_follows_a_stop` both rest on the `Stop` record marking where an
actor's records end; written at send time it can land *before* records the
actor wrote afterwards, since the actor runs until it sees the control, which
inverts the invariant. And log order is per agent only, with `seq` a
message's number rather than a record index, so an episode-written control
belongs to no agent's sequence and its position among that agent's records
would mean nothing.

## Names to settle

Three renames, all cosmetic in effect and none urgent, but each pointing at
something real about the layer it sits in.

| Now | Proposed | Why |
|-----|----------|-----|
| `Seat` | `ActorEndpoint`, or nothing | A game's word at the protocol layer; narrow it first and the question may not arise |
| `UnstartedActor` | `ActorChannels` | It is not a builder |
| `handler` thread | `decision` thread | `handler` is a callback noun beside a domain verb |

**`UnstartedActor` is not a builder** and should not be renamed as though it
were. It has no setters, and `spawn_agent` takes it plus six separate
arguments — handler, router, records, clock, timer, done — so none of the
configuration a builder would accumulate lives in it. It holds one actor's
channel endpoints across a two-phase construction that a dependency cycle
forces: every inbox and control channel first, so the router is complete
before anybody can send; then the router; only then the threads. `new`
returns the struct plus two senders, receiving ends staying inside for the
thread and sending ends going into the roster. That is closer to typestate
than to builder, and the transition consumes it.

**Not `action` for the second thread.** `Action` is already a public type at
the crate root, with `ActionRecord` beside it, and ADR-0007 pins it as the RL
term for what an agent emits; "action thread" would read both as the thread
that produces actions and as a thread that is one. Prefer `decision`, which
is this project's own word: ADR-0016 is titled "Actors perceive on one thread
and **decide** on another", and `docs/visualization/perceive-and-decide.html`
carries `.perceive` and `.decide` classes already. It also fits the
observe–decide–act triple, where the decision is the call and the action is
what it yields, which keeps `Action` for what it means.

Two cautions for whoever does these. `Seat` and `UnstartedActor` are the two
halves of the same channels — one the sending ends held by the router, the
other the receiving ends held by the actor — so their names have to stay
tellable apart and should be chosen together; `ActorEndpoint` and
`ActorChannels` may be too close for that. And "handler" names two different
things in the tree, the thread and the policy object registered with `add`;
only the thread sense should change, since `Policy` is already the object's
real name.

## What to do next

Nothing here is scheduled. In rough order of value:

1. **Narrow `Seat` and the `control()` accessors.** The only item with a
   correctness argument behind it, and it settles a naming question for free.
2. **Drop the reward type parameter.** A parameter off the most-used public
   type in the crate, plus a `PhantomData` and a variance puzzle.
3. **Log controls as sent and seen.** Closes a real diagnostic gap for the
   hangs the time limit is there to catch.
4. **Decide the cycle record's future.** A product question about what a
   training pipeline needs, not a correctness one.
5. **The renames.** Last, and cheapest to defer.

Each of the first three probably earns an ADR when it is decided. The renames
would not.
