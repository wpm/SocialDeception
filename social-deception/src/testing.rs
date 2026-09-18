//! Helpers shared by the crate's unit tests.

use serde_json::Value;

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
