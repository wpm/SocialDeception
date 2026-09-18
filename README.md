# Social Deception

Social deception games

## Developing

After cloning, run this once so that the checked-in pre-commit hook runs
before every commit:

    git config core.hooksPath .githooks

The hook runs `cargo fmt --check`, `cargo clippy --all-targets -- -D warnings`
and `cargo test`, plus `cargo deny check` and `typos` when those tools are
installed. CI runs the same checks on every push and pull request.
