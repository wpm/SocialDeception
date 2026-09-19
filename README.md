# Social Deception

Social deception games

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
