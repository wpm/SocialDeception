//! Helpers shared by the integration tests: the [`collatz`] environment,
//! a [`TempDir`] to write a trajectory in, reading a trajectory file back,
//! checking the invariants every trajectory satisfies whatever the
//! environment, and, in [`werewolf`], the invariants a trajectory of
//! Werewolf satisfies on top of them.
//!
//! The checks here are properties of the log, not of any environment. They
//! are meant to run unchanged against episodes where no independent check on
//! the content is available.

pub mod collatz;
mod temp;
pub mod werewolf;

use std::collections::HashMap;

use serde_json::Value;
pub use temp::TempDir;

/// Parses a trajectory file into one JSON value per line.
///
/// # Panics
///
/// If the bytes are not UTF-8, the text does not end with a newline, or any
/// line is not a JSON value.
pub fn parse(bytes: &[u8]) -> Vec<Value> {
    let text = std::str::from_utf8(bytes).expect("a trajectory is UTF-8");
    assert!(text.ends_with('\n'), "a trajectory ends with a newline");
    text.lines()
        .map(|line| serde_json::from_str(line).expect("every line is a JSON value"))
        .collect()
}

/// Asserts the invariants every trajectory satisfies:
///
/// - every event a cycle lists as an input has an event record whose
///   arrival, or the deadline it was due at, is no later than the cycle's
///   `t_start`;
/// - every event a cycle lists as an output has an event record sent within
///   the cycle's handling window;
/// - a cycle's inputs followed by its outputs are exactly the event records
///   its agent wrote since its previous cycle, and every event record
///   belongs to some cycle;
/// - per-agent sequence numbers are contiguous and strictly increasing from
///   zero, in file order;
/// - no cycle's input list is empty;
/// - no message has its sender among its recipients, and a message that
///   arrived at an agent lists that agent among them;
/// - an event record's stamp matches its direction: `arrived` on a control
///   event or a message from another agent, `sent` on a message from the
///   agent itself, `due` on a think.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value]) {
    let records = records(lines);
    for line in lines {
        match line["type"].as_str() {
            Some("event") => check_event(line),
            Some("cycle") => check_cycle(line, &records),
            other => panic!("unknown record type {other:?} in {line}"),
        }
    }
    check_sequence_numbers(lines);
    check_grouping(lines);
}

/// The event records of `lines`, by agent and sequence number.
pub fn records(lines: &[Value]) -> HashMap<(&str, u64), &Value> {
    lines
        .iter()
        .filter(|line| line["type"] == "event")
        .map(|line| ((agent(line), seq(line)), line))
        .collect()
}

/// The agent a record belongs to.
pub fn agent(line: &Value) -> &str {
    line["agent"]
        .as_str()
        .expect("every record names its agent")
}

/// An event record's sequence number.
pub fn seq(line: &Value) -> u64 {
    line["seq"]
        .as_u64()
        .expect("every event record has a sequence number")
}

/// The sequence numbers a cycle record lists under `key`.
pub fn seqs<'a>(cycle: &'a Value, key: &str) -> impl Iterator<Item = u64> + 'a {
    cycle[key]
        .as_array()
        .expect("a cycle lists its inputs and outputs")
        .iter()
        .map(|seq| seq.as_u64().expect("a sequence number is an integer"))
}

fn time(line: &Value, key: &str) -> u64 {
    line[key]
        .as_u64()
        .unwrap_or_else(|| panic!("{line} has no {key}"))
}

fn check_event(line: &Value) {
    let event = &line["event"];
    let kind = event["kind"].as_str().expect("an event has a kind");
    let stamps: Vec<&str> = ["arrived", "sent", "due"]
        .into_iter()
        .filter(|stamp| !line[stamp].is_null())
        .collect();
    let expected = match kind {
        "message" if event["sender"] == agent(line) => "sent",
        "control" | "message" => "arrived",
        "think" => "due",
        other => panic!("unknown event kind {other:?} in {line}"),
    };
    assert_eq!(
        stamps,
        [expected],
        "the stamp matches the direction: {line}"
    );
    if kind == "message" {
        let recipients = event["recipients"]
            .as_array()
            .expect("a message lists its recipients");
        assert!(!recipients.is_empty(), "a message has recipients: {line}");
        assert!(
            !recipients.contains(&event["sender"]),
            "no message has its sender among its recipients: {line}"
        );
        if expected == "arrived" {
            assert!(
                recipients.contains(&Value::from(agent(line))),
                "a message arrives only at its recipients: {line}"
            );
        }
    }
}

fn check_cycle(cycle: &Value, records: &HashMap<(&str, u64), &Value>) {
    let (t_start, t_stop) = (time(cycle, "t_start"), time(cycle, "t_stop"));
    assert!(
        t_start <= t_stop,
        "a handling window runs forwards: {cycle}"
    );
    let record = |seq: u64| {
        records
            .get(&(agent(cycle), seq))
            .unwrap_or_else(|| panic!("{cycle} lists seq {seq}, which has no event record"))
    };

    let mut inputs = seqs(cycle, "inputs").peekable();
    assert!(
        inputs.peek().is_some(),
        "no cycle's input list is empty: {cycle}"
    );
    for record in inputs.map(record) {
        let stamp = ["arrived", "due"]
            .into_iter()
            .find(|stamp| !record[stamp].is_null())
            .unwrap_or_else(|| {
                panic!("an input is an arrival or a think, not an output: {record} in {cycle}")
            });
        assert!(
            time(record, stamp) <= t_start,
            "an input is on the inbox, or due, before its cycle starts: {record} in {cycle}"
        );
    }
    for record in seqs(cycle, "outputs").map(record) {
        let sent = time(record, "sent");
        assert!(
            t_start <= sent && sent <= t_stop,
            "an output is sent within its cycle's window: {record} in {cycle}"
        );
    }
}

fn check_sequence_numbers(lines: &[Value]) {
    let mut next: HashMap<&str, u64> = HashMap::new();
    for line in lines.iter().filter(|line| line["type"] == "event") {
        let expected = next.entry(agent(line)).or_insert(0);
        assert_eq!(
            seq(line),
            *expected,
            "sequence numbers are contiguous from zero: {line}"
        );
        *expected += 1;
    }
}

fn check_grouping(lines: &[Value]) {
    let mut pending: HashMap<&str, Vec<u64>> = HashMap::new();
    for line in lines {
        let pending = pending.entry(agent(line)).or_default();
        if line["type"] == "event" {
            pending.push(seq(line));
        } else {
            let listed: Vec<u64> = seqs(line, "inputs").chain(seqs(line, "outputs")).collect();
            assert_eq!(
                listed,
                std::mem::take(pending),
                "a cycle lists exactly the records since its agent's previous cycle: {line}"
            );
        }
    }
    for (agent, pending) in pending {
        assert!(
            pending.is_empty(),
            "every record belongs to a cycle, but {agent} left {pending:?} after its last"
        );
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A well-formed trajectory: agent `a` receives a start and a message
    /// from `b` in one pass and replies, then receives a stop.
    fn good() -> Vec<Value> {
        vec![
            json!({"type": "event", "agent": "a", "seq": 0, "arrived": 10,
                   "event": {"kind": "control", "control": "start"}}),
            json!({"type": "event", "agent": "a", "seq": 1, "arrived": 20,
                   "event": {"kind": "message", "sender": "b", "recipients": ["a"],
                             "payload": {"Step": 6}}}),
            json!({"type": "event", "agent": "a", "seq": 2, "sent": 40,
                   "event": {"kind": "message", "sender": "a", "recipients": ["b"],
                             "payload": {"Step": 3}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50,
                   "inputs": [0, 1], "outputs": [2]}),
            json!({"type": "event", "agent": "a", "seq": 3, "due": 55,
                   "event": {"kind": "think"}}),
            json!({"type": "cycle", "agent": "a", "t_start": 60, "t_stop": 70,
                   "inputs": [3], "outputs": []}),
        ]
    }

    /// The good trajectory with one edit applied to line `index`.
    fn edited(index: usize, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        let mut lines = good();
        edit(&mut lines[index]);
        lines
    }

    #[test]
    fn a_good_trajectory_passes() {
        check(&good());
        let text = good()
            .iter()
            .fold(String::new(), |text, line| text + &line.to_string() + "\n");
        assert_eq!(parse(text.as_bytes()), good());
    }

    #[test]
    #[should_panic(expected = "before its cycle starts")]
    fn an_input_arriving_after_its_cycle_starts_is_caught() {
        check(&edited(1, |line| line["arrived"] = json!(31)));
    }

    #[test]
    #[should_panic(expected = "before its cycle starts")]
    fn a_think_due_after_its_cycle_starts_is_caught() {
        check(&edited(4, |line| line["due"] = json!(61)));
    }

    #[test]
    #[should_panic(expected = "within its cycle's window")]
    fn an_output_sent_outside_its_cycle_is_caught() {
        check(&edited(2, |line| line["sent"] = json!(51)));
    }

    #[test]
    #[should_panic(expected = "contiguous from zero")]
    fn a_gap_in_sequence_numbers_is_caught() {
        let mut lines = good();
        lines[4]["seq"] = json!(4);
        lines[5]["inputs"] = json!([4]);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "input list is empty")]
    fn an_empty_drain_is_caught() {
        check(&edited(5, |line| line["inputs"] = json!([])));
    }

    #[test]
    #[should_panic(expected = "arrives only at its recipients")]
    fn a_message_delivered_to_a_non_recipient_is_caught() {
        check(&edited(1, |line| {
            line["event"]["recipients"] = json!(["c"]);
        }));
    }

    #[test]
    #[should_panic(expected = "not an output")]
    fn a_cycle_that_lists_an_output_as_an_input_is_caught() {
        check(&edited(3, |line| {
            line["inputs"] = json!([0, 1, 2]);
            line["outputs"] = json!([]);
        }));
    }

    #[test]
    #[should_panic(expected = "sender among its recipients")]
    fn a_loopback_is_caught() {
        check(&edited(2, |line| {
            line["event"]["recipients"] = json!(["a", "b"]);
        }));
    }

    #[test]
    #[should_panic(expected = "stamp matches the direction")]
    fn a_sent_stamp_on_an_arrival_is_caught() {
        check(&edited(1, |line| {
            line["sent"] = line["arrived"].take();
        }));
    }

    #[test]
    #[should_panic(expected = "records since its agent's previous cycle")]
    fn a_cycle_that_skips_a_record_is_caught() {
        check(&edited(3, |line| line["inputs"] = json!([0])));
    }

    #[test]
    #[should_panic(expected = "every record belongs to a cycle")]
    fn a_record_after_the_last_cycle_is_caught() {
        let mut lines = good();
        lines.truncate(5);
        check(&lines);
    }
}
