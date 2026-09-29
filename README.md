# Social Deception

Social deception games

## The vocabulary

An episode is a fixed roster of participants talking over in-process
channels without turn-taking. The framework names the parts as
reinforcement learning does, because that is the vocabulary its log is
read in (see
[ADR-0007](docs/decision-history/0007-reinforcement-learning-vocabulary.md)).

Every participant is an **actor**: two threads, one that perceives and one
that decides (see
[ADR-0016](docs/decision-history/0016-actors-perceive-on-one-thread-and-decide-on-another.md)).
The perception thread receives, stamps what arrives with the instant it
arrived, logs it and forwards it. The handler thread calls one application
function per **observation** and sends what it returns as it is returned.
The point of the second thread is that perceiving never waits on deciding:
an actor's sense of time is when its observations arrive, and an actor that
stopped listening during a multi-second model call would hear everything
said during it as arriving at once.

What travels between actors is a **message** — sender, recipients, a
per-sender sequence number and a payload the game defines. The same message
is an **action** of the actor that sent it and an observation of each actor
that receives it, which is what lets one actor's records be joined to
another's, and a trajectory built from the log. An actor that wants a
message from itself sets a **reminder**: a payload and a deadline, delivered
back to it as an ordinary message when the deadline arrives. That is the
only way a message reaches its own sender, and nothing cancels one — a
handler that has changed its mind ignores a stale reminder.

One actor per episode is the **environment**: it alone starts and stops the
others, and it alone decides what an agent's behavior was worth. An agent
returns actions and an environment returns **effects**, which are an action,
a control, or a reward; an agent cannot command or reward because its return
type has no variant for either. A **control** — start or stop — travels on a
channel of its own, and a stop takes effect the moment the perception thread
sees it, ahead of anything queued behind it. Whatever was queued is logged
as undelivered, which is why an environment that wants a last word heard
sets a reminder and stops everybody when it arrives. A **reward** does not
travel at all: nothing in a running episode reads it, so the environment
writes it straight to the log, where training picks it up. Werewolf's
environment is the moderator, and it pays +1 to every player on the winning
faction and -1 to every player on the losing one, living and dead alike.

## Playing Werewolf

The `werewolf` binary plays one episode of Werewolf from a TOML
configuration, with every player an agent and every decision made by a
uniform random strategy, and writes the log beside the configuration
that reproduces it. The example at `examples/werewolf.toml` is a
seven-player game:

    cargo run -- play examples/werewolf.toml

That narrates the game as it happens, a line per thing that is said —
the time, who said it, who heard it, and what it was — and then prints
how it ended. `--quiet` leaves the narration out and prints the summary
alone. The narration is a live view rather than the record of the game:
it shows each message once, from the side of whoever sent it, in the order
the players' threads produced them. `replay`, below, is the reproducible
reading.

It also writes `werewolf.jsonl`, the log, and `werewolf.jsonl.toml`,
the effective configuration, at the repository root, where both are
ignored by git. To read the game back:

    cargo run -- replay werewolf.jsonl

That renders the logical game the log records — the deal, each
round's moves and deaths, and who won — and ends with the reward every
player was paid, which is the only place a reward is ever shown, since it
was never said to anybody.

and to play it again from what it left behind, whatever flags produced it:

    cargo run -- play werewolf.jsonl.toml --trajectory rerun.jsonl

`--seed` overrides the configuration's seed. `--help` on either command
lists the rest.

## Developing

After cloning, run this once so that the checked-in pre-push hook runs
before every push:

    git config core.hooksPath .githooks

The hook runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
and `cargo test`, plus `cargo deny check` and `typos` when those tools are
installed. CI runs the same checks on every push and pull request.
