# ADR-0005: A policy separates decisions from rules

**Status:** Accepted
**Date:** 2026-09-18
**Deciders:** Bill McNeill
**Refined by:** [ADR-0006](0006-knowledge-holds-what-the-agent-did-in-secret.md),
which adds the agent's own secret actions to what `Knowledge` folds
**Amended by:** [ADR-0007](0007-reinforcement-learning-vocabulary.md), which renames the action type `Move`: an
`Action` is now the message an agent sends, here a `Response` carrying a `Move`

## Context

Werewolf is played by agents, each of which is asked at a decision point to
pick an action: whom to devour, whom to investigate, whom to protect, whom to
nominate. The first version makes every decision with a uniform-random
baseline; later versions will make them with a language model, and later
still with a learned policy trained on the trajectories the baseline and the
model leave behind.

Three properties are wanted at once, and they pull against each other.

- **The rules must be unbypassable.** A language model will name dead
  players, nominate itself, and emit strings that are not player names at
  all. Nothing it does may put the game in a state the rules forbid.
- **Interesting behavior must stay available.** A werewolf that sacrifices a
  packmate to build credibility with the village is playing well. Encoding
  "never eat a packmate" as a rule designs that play out of the game before
  any model has been asked.
- **The baseline must play sensibly, not merely within the rules.** The
  experiment measures a model's win rate against the baseline. A baseline
  that eats its own packmates at random is a strawman, and the number the
  experiment produces then means nothing.

The name *policy* is borrowed from reinforcement learning on purpose.

## Decision

**A role type computes the action space — the actions the rules permit right
now — and nothing else. A `Policy` picks an action from it. Every strategic
judgment lives inside the policy. `RandomPolicy` carries its own heuristic,
private to it, so that the baseline is a fair opponent; that heuristic is
deliberately not offered to any other policy.**

### The vocabulary, stated once and used throughout the code

The code names the reinforcement-learning triple this way and introduces no
synonyms.

| RL | Here | Kind |
|----|------|------|
| observation | an observation of a `Message` — everything that arrives on an agent's receiver | enum |
| state | `Knowledge` — a sufficient statistic of the observation history | struct |
| action | the response an agent sends, carrying one `Move` drawn from the action space available now | struct |

The set of permitted moves is the **action space**. The role computes it,
as `Player::action_space` and `View::action_space`. A move not in it is
*outside the action space*. **No identifier is called `legal`.**

`Move` is the *type* of the choice an action carries. The action space is a
*value* of type `Vec<Move>`: the subset available for one decision. The
space depends on the state; the type does not.

A request invokes `Policy::choose` with the `Knowledge` and the action
space, and the chosen `Move` becomes the response.

### The message kinds are named for what they do

So that the word `action` names one thing only, the three message kinds are
named for their function rather than for their content.

| Message | Direction | Is |
|---------|-----------|----|
| `Narration` | moderator → a chosen set of players | a true statement the recipients now observe |
| `Request` | moderator → one player | a decision point: the moment the policy is invoked |
| `Response` | player → moderator | the reply, echoing the request's id and carrying one `Move` |

Narration is addressed, not broadcast;
[ADR-0004](0004-moderator-agent-runs-the-game.md) records why. A player
observes many narrations but acts only on a request, which is what gives an
agent in a turnless runtime its decision points. Dialogue, when it arrives,
is a fourth kind: player-to-player speech.

### `Knowledge` holds only true statements

`Knowledge` is not an observation; it is the state a policy conditions on,
folded from every observation so far. Nor is it a belief. "Belief" is
reserved for information that could be false, and nothing in `Knowledge` can
be: every observation a player receives is a true statement from the
moderator. The state is *incomplete* about the hidden world and never
*incorrect* about it. A villager does not know who the werewolves are, but
nothing it does know is wrong. The name is meant in its ordinary sense, in
which knowing something entails its being so.

That boundary will move. Dialogue may be false — that is what a social
deception game is — and whatever holds players' assertions, and the
conclusions drawn from them, will be beliefs and will want a type of its
own. The line is already drawn: `Knowledge` folds only moderator narration,
and inference is the policy's business rather than the state's. A villager
hearing a day tally stores the tally, which is true, and draws whatever
conclusions it likes in its policy, where being wrong costs it nothing
structural.

### There is no `observation_space` function

The action space is small and is enumerated at every decision, which is why
it is computed and returned. The observation space is combinatorial:
`Event<Message>` ranges over arbitrary agent sets and tallies, and nothing
would ever enumerate it. It earns a name in the documentation and no code at
all.

The observation space is per role. A villager can never observe a night
`Tally`, an `Investigated`, or a non-empty pack; those lie outside a
villager's observation space. "The realized observations stay inside each
role's observation space" is the precise statement of what routing enforces,
and it is what the hidden-information invariants are to assert.

### The action space is what a policy is a distribution over

"Never target yourself" is a rule and shapes the action space. "Never eat a
packmate" is strategy and lives nowhere in the rules. The rules define the
game and are enforced at the boundary; strategy is the thing under study and
is enforced by nobody.

A policy is a conditional distribution over the action space given
`Knowledge`. That is the functional form a learned policy takes, and two
things follow from it. They are the strongest justification for the
decision.

- **The action space must be identical for every policy.** Trajectories from
  one policy are training data for another only if they share a support. A
  baseline whose action space were a hand-written subset would produce data
  in which "eat a packmate" lies outside the support the data can speak to at
  all, so no amount of it could teach a learner what that move is worth.
- **A prior belongs in the policy, not in the action space.** In the policy —
  as initialization, as shaped reward, as the baseline's sampling — a prior
  can be weakened, ablated or learned away. In the action space it cannot be
  moved, and it silently becomes part of the definition of the game.

Therefore the action space is an **ordered** `Vec<Move>`: targets in sorted
agent order, `Abstain` last where it is permitted. An index into it is a
stable action label, which is what a learner needs and what a constrained
decode over a model's output needs.

### The baseline's heuristic, stated so it is reproducible

`RandomPolicy` chooses uniformly from a set of candidates narrowed from the
action space as follows.

- A werewolf's candidates exclude its living packmates for `Devour` and
  `Nominate`.
- A seer's candidates for `Investigate` exclude targets it has already seen.
- `Abstain` is excluded wherever anything else is available.
- Otherwise the candidates are the whole action space.
- A narrowing that would empty the set falls back to the whole action space.

Deliberately absent: a seer using its findings when nominating, and a
villager reading the tallies. The uniform baseline is meant to leave the
seer's information unused, which is what makes its win rate a function of
the role counts alone.

### `Policy::choose` is infallible

A language model will time out, refuse, or emit garbage, and there is no
correct thing for the *rules* to do about that. The decision belongs to the
policy that failed: it owns its retries and holds a `RandomPolicy` fallback.
A role validates the returned move against the action space and panics on
a violation, because that is a policy bug and not a condition the game can
continue from.

A policy may block. [ADR-0001](0001-single-process-thread-per-agent.md)
grants every agent its own thread precisely so that a handler waiting on a
model provider blocks only itself.

### Determinism

Each agent holds its own `ChaCha8Rng`, seeded from the master seed mixed with
the agent's id. Per-agent seeding means an agent's actions depend on its own
history and not on thread scheduling. The moderator's tie-breaking generator
is seeded with a distinct label.

Stated precisely: for a fixed configuration and seed, the **logical
transcript** — role assignment, every request, response, tally, elimination
and the outcome — is identical on every run. Wall-clock timestamps and the
interleaving of different agents' records in the trajectory are not, and
cannot be, because the agents are threads.

## Alternatives considered

### A `sensible` subset published beside the action space

An earlier draft of this design had each role publish a non-empty `sensible`
subset alongside the action space, for policies to prefer. Rejected for
three reasons. It would have exactly one consumer forever, since a
language-model policy makes its own strategic judgments; a mechanism with
one caller, built into the type system as though it were general, is the
premature generalization this project restarted to avoid. Handing it to a
model would contaminate the measurement, making the win rate partly the
model's play and partly the list's, with no clean ablation between them. And
it bought nothing: the stated justification was that strategic filters need
private knowledge, but that knowledge is in `Knowledge`, which is in the
`View`, so any policy can compute those filters itself.

### Strategic filtering inside the role

The role could simply omit strategically poor actions from the action space.
Rejected for the two reasons under *The action space is what a policy is a
distribution over*: a prior fixed in the action space cannot be ablated, and
the resulting trajectories have no support for the omitted moves.

### A fallible `choose`

`Policy::choose` could return an error for the rules to handle. Rejected
because the game cannot skip a decision, and substituting an action is a
policy decision that belongs in the policy.

### `StdRng` instead of `ChaCha8Rng`

`StdRng`'s algorithm may change between `rand` releases. A seed that stops
meaning the same thing after an upgrade is not a reproducible experiment.

## Consequences

- **Measuring the baseline needs no model.** Running many episodes to do so
  is a small later addition, out of scope for the Werewolf epic.
- **Roles stay small.** A role is its state and its action space, nothing
  more.
- **Each new policy writes its own strategy from scratch.** A second
  hand-written policy would duplicate some of `RandomPolicy`'s filtering.
  Accepted, because two callers is the point at which extracting something
  is justified and one is not.
- **`Policy` is Werewolf-specific on purpose.** It becomes a parameterized
  notion when several games exist. Generalizing from one example is how the
  previous attempt at this project lost a year.
- **Testing determinism means projecting out timestamps and record
  interleaving** before comparing two runs of one configuration and seed.
