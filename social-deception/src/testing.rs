//! Helpers shared by the crate's unit tests.

use std::collections::BTreeSet;

use serde_json::Value;

use crate::event::AgentId;

/// Parses a trajectory file into one JSON value per line.
///
/// # Panics
///
/// If the bytes are not UTF-8, the text does not end with a newline, or any
/// line is not a JSON value.
pub(crate) fn parse_lines(bytes: &[u8]) -> Vec<Value> {
    let text = std::str::from_utf8(bytes).unwrap();
    assert!(text.ends_with('\n'), "file must end with a newline");
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// Serializes a value to a JSON value, for asserting on its shape.
///
/// # Panics
///
/// If the value cannot be serialized.
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap()
}

/// An agent id from a name.
pub(crate) fn id(name: &str) -> AgentId {
    AgentId::new(name)
}

/// A set of agent ids from names.
pub(crate) fn ids(names: &[&str]) -> BTreeSet<AgentId> {
    names.iter().map(|name| id(name)).collect()
}
