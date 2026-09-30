//! Helpers shared by the integration tests: a [`TempDir`] to write a log in,
//! reading a log file back, and the accessors that name the parts of a log
//! record.
//!
//! The invariants themselves live with the thing they describe. [`actor`] holds
//! the ones any episode's log satisfies whatever the game; [`werewolf`] holds
//! the ones a game of Werewolf satisfies on top of them. What is here is what
//! more than one of them reaches for.

pub mod actor;
pub mod collatz_actor;
mod temp;
pub mod werewolf;

use serde_json::Value;
pub use temp::TempDir;

/// Parses a log file into one JSON value per line.
///
/// # Panics
///
/// If the bytes are not UTF-8, the text does not end with a newline, or any
/// line is not a JSON value.
pub fn parse(bytes: &[u8]) -> Vec<Value> {
    let text = std::str::from_utf8(bytes).expect("a log is UTF-8");
    assert!(text.ends_with('\n'), "a log ends with a newline");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("every line is a JSON value"))
        .collect()
}

/// The agent a record belongs to.
pub fn agent(line: &Value) -> &str {
    line["agent"]
        .as_str()
        .expect("every record names its agent")
}

/// A message record's sequence number.
pub fn seq(line: &Value) -> u64 {
    line["seq"]
        .as_u64()
        .expect("every message record has a sequence number")
}

/// The time a record carries under `key`, as nanoseconds since the episode's
/// origin.
///
/// # Panics
///
/// If the record has no such time.
pub fn time(line: &Value, key: &str) -> u64 {
    line[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{line} has no {key}"))
}
