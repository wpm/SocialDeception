//! Werewolf, end to end.
//!
//! The fixture trajectory under `tests/fixtures` is what the transcript
//! reader and the `werewolf replay` command are tested against. It is
//! checked here against the invariants every trajectory satisfies, so that it
//! cannot rot into something the runtime would never have written.

mod support;

use std::fs;

/// A seven-player game played to a village win, with its effective config
/// and its expected rendering beside it.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/werewolf.jsonl");

#[test]
fn the_fixture_is_a_trajectory_the_runtime_could_have_written() {
    let lines = support::parse(&fs::read(FIXTURE).unwrap());
    support::check(&lines);
}
