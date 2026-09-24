# Social Deception

Social deception games

## The vocabulary

An episode is a fixed roster of agents, each a thread, talking over
in-process channels without turn-taking. The framework names the parts as
reinforcement learning does, because that is what the trajectories it
writes are read in (see
[ADR-0007](docs/decision-history/0007-reinforcement-learning-vocabulary.md)).

An **agent** runs a loop whose one turn is a **cycle**: it pops its two
queues, folds what it popped into its own state, and sends what its handler
returns. What travels between agents is an **event** — sender, recipients,
creation time and a payload the game defines. The same event is an
**action** of the agent that sent it and an **observation** of each agent
that pops it, which is what lets one agent's trajectory be joined to
another's.

One agent per episode is the **environment**: it alone starts and stops the
others, and it alone decides what an agent's behavior was worth. Those two
powers travel differently. A **control** — start or stop — goes on a queue
of its own, so that it is acted on ahead of anything already waiting, and
it preempts the cycle it interrupts. A **reward** does not travel at all:
nothing in a running episode reads it, so the environment writes it
straight to the trajectory, where training picks it up. Werewolf's
environment is the moderator, and it pays +1 to every player on the winning
faction and −1 to every player on the losing one, living and dead alike.

## Playing Werewolf

The `werewolf` binary plays one episode of Werewolf from a TOML
configuration, with every player an agent and every decision made by a
uniform random policy, and writes the trajectory beside the configuration
that reproduces it. The example at `examples/werewolf.toml` is a
seven-player game:

    cargo run --bin werewolf -- play examples/werewolf.toml

That writes `werewolf.jsonl`, the trajectory, and `werewolf.jsonl.toml`,
the effective configuration, at the repository root, where both are
ignored by git. To read the game back:

    cargo run --bin werewolf -- replay werewolf.jsonl

That renders the logical game the trajectory records — the deal, each
round's moves and deaths, and who won — and ends with the reward every
player was paid, which is the only place a reward is ever shown, since it
was never said to anybody.

and to play it again from what it left behind, whatever flags produced it:

    cargo run --bin werewolf -- play werewolf.jsonl.toml --trajectory rerun.jsonl

`--seed` overrides the configuration's seed. `--help` on either command
lists the rest.

## Developing

After cloning, run this once so that the checked-in pre-commit hook runs
before every commit:

    git config core.hooksPath .githooks

The hook runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
and `cargo test`, plus `cargo deny check` and `typos` when those tools are
installed. CI runs the same checks on every push and pull request.
