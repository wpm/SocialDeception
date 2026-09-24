//! Helpers shared by the integration tests: the [`collatz`] environment,
//! a [`TempDir`] to write a trajectory in, reading a trajectory file back,
//! checking the invariants every trajectory satisfies whatever the game,
//! and, in [`werewolf`], the invariants a trajectory of Werewolf satisfies
//! on top of them.
//!
//! The checks here are properties of the log, not of any game. They are
//! meant to run unchanged against episodes where no independent check on the
//! content is available.

pub mod collatz;
mod temp;
pub mod werewolf;

use std::collections::{BTreeMap, HashMap, HashSet};

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
/// - every observation and control was received at or after it was created,
///   and at exactly the `t_start` of the cycle that lists it;
/// - every action was created within the window of the cycle that lists it;
/// - a cycle's inputs followed by its outputs are exactly the records its
///   agent wrote since its previous cycle, and every record belongs to some
///   cycle;
/// - per-agent sequence numbers are contiguous and strictly increasing from
///   zero, in file order;
/// - a cycle woken by the queue has at least one input, and one woken by the
///   timeout has no observations at all;
/// - no event has its sender among its recipients; an observation lists the
///   agent that recorded it among the recipients, and an action names it as
///   the sender;
/// - every observation joins exactly one action, by sender and creation
///   time, and every action has one matching observation per recipient.
///
/// The last is the one that makes a trajectory a single object rather than a
/// pile of per-agent logs: an observation and the action that produced it
/// are the same event seen from its two ends, and nothing but the sender and
/// the creation time links them.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value]) {
    let records = records(lines);
    for line in lines {
        match line["type"].as_str() {
            Some("observation" | "action" | "control") => check_record(line),
            Some("cycle") => check_cycle(line, &records),
            other => panic!("unknown record type {other:?} in {line}"),
        }
    }
    check_sequence_numbers(lines);
    check_grouping(lines);
    check_the_join(lines);
}

/// The non-cycle records of `lines`, by agent and sequence number.
pub fn records(lines: &[Value]) -> HashMap<(&str, u64), &Value> {
    lines
        .iter()
        .filter(|line| line["type"] != "cycle")
        .map(|line| ((agent(line), seq(line)), line))
        .collect()
}

/// The agent a record belongs to.
pub fn agent(line: &Value) -> &str {
    line["agent"]
        .as_str()
        .expect("every record names its agent")
}

/// A record's sequence number.
pub fn seq(line: &Value) -> u64 {
    line["seq"]
        .as_u64()
        .expect("every non-cycle record has a sequence number")
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

/// What a record says about the event it carries: its sender, its
/// recipients and its payload, which together with the creation time are
/// what an observation and its action must agree on.
fn event(line: &Value) -> (&str, Vec<&Value>, &Value) {
    let event = &line["event"];
    let sender = event["sender"].as_str().expect("an event names its sender");
    let recipients: Vec<&Value> = event["recipients"]
        .as_array()
        .expect("an event lists its recipients")
        .iter()
        .collect();
    (sender, recipients, &event["payload"])
}

fn check_record(line: &Value) {
    match line["type"].as_str() {
        Some("control") => {
            assert!(
                time(line, "created") <= time(line, "received"),
                "nothing is received before it was created: {line}"
            );
        }
        Some("observation") => {
            assert!(
                time(line, "created") <= time(line, "received"),
                "nothing is received before it was created: {line}"
            );
            let (sender, recipients, _) = event(line);
            check_recipients(line, sender, &recipients);
            assert!(
                recipients.contains(&&Value::from(agent(line))),
                "an event is observed only by its recipients: {line}"
            );
            assert_ne!(
                sender,
                agent(line),
                "an agent does not observe what it sent: {line}"
            );
        }
        Some("action") => {
            // An action has no `received`: its sender knows only when it
            // sent it, and when each recipient got it is in that
            // recipient's own observation record.
            assert!(
                line["received"].is_null(),
                "an action records no receipt: {line}"
            );
            let (sender, recipients, _) = event(line);
            check_recipients(line, sender, &recipients);
            assert_eq!(
                sender,
                agent(line),
                "an action names the agent that took it as its sender: {line}"
            );
        }
        other => panic!("unknown record type {other:?} in {line}"),
    }
}

fn check_recipients(line: &Value, sender: &str, recipients: &[&Value]) {
    assert!(!recipients.is_empty(), "an event has recipients: {line}");
    assert!(
        !recipients.contains(&&Value::from(sender)),
        "no event has its sender among its recipients: {line}"
    );
}

fn check_cycle(cycle: &Value, records: &HashMap<(&str, u64), &Value>) {
    let (t_start, t_stop) = (time(cycle, "t_start"), time(cycle, "t_stop"));
    assert!(
        t_start <= t_stop,
        "a handling window runs forwards: {cycle}"
    );
    let woken = cycle["woken"].as_str().expect("a cycle says what woke it");
    assert!(
        woken == "queue" || woken == "timeout",
        "a cycle is woken by the queue or the timeout: {cycle}"
    );
    let record = |seq: u64| {
        records
            .get(&(agent(cycle), seq))
            .unwrap_or_else(|| panic!("{cycle} lists seq {seq}, which has no record"))
    };

    let inputs: Vec<&Value> = seqs(cycle, "inputs").map(|seq| *record(seq)).collect();
    let observations = inputs
        .iter()
        .filter(|input| input["type"] == "observation")
        .count();
    if woken == "timeout" {
        assert_eq!(
            observations, 0,
            "a cycle woken by the timeout has no observations: {cycle}"
        );
    } else {
        assert!(
            !inputs.is_empty(),
            "a cycle woken by the queue popped something: {cycle}"
        );
    }
    for input in inputs {
        assert!(
            input["type"] != "action",
            "an input is something popped, not an output: {input} in {cycle}"
        );
        assert_eq!(
            time(input, "received"),
            t_start,
            "everything a cycle popped was popped at its start: {input} in {cycle}"
        );
    }
    for record in seqs(cycle, "outputs").map(record) {
        assert_eq!(
            record["type"], "action",
            "an output is an action: {record} in {cycle}"
        );
        let created = time(record, "created");
        assert!(
            t_start <= created && created <= t_stop,
            "an action is created within its cycle's window: {record} in {cycle}"
        );
    }
}

fn check_sequence_numbers(lines: &[Value]) {
    let mut next: HashMap<&str, u64> = HashMap::new();
    for line in lines.iter().filter(|line| line["type"] != "cycle") {
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
        if line["type"] == "cycle" {
            let listed: Vec<u64> = seqs(line, "inputs").chain(seqs(line, "outputs")).collect();
            assert_eq!(
                listed,
                std::mem::take(pending),
                "a cycle lists exactly the records since its agent's previous cycle: {line}"
            );
        } else {
            pending.push(seq(line));
        }
    }
    for (agent, pending) in pending {
        assert!(
            pending.is_empty(),
            "every record belongs to a cycle, but {agent} left {pending:?} after its last"
        );
    }
}

/// Every observation is somebody's action, and every action is observed by
/// each of its recipients. The join is on the sender and the creation time,
/// which is all a reader has: nothing carries an identifier for an event.
fn check_the_join(lines: &[Value]) {
    let mut actions: BTreeMap<(&str, u64), &Value> = BTreeMap::new();
    for line in lines.iter().filter(|line| line["type"] == "action") {
        let key = (agent(line), time(line, "created"));
        assert!(
            actions.insert(key, line).is_none(),
            "an agent takes at most one action per instant, or no observation could \
             name which: {line}"
        );
    }
    let mut observed: HashSet<((&str, u64), &str)> = HashSet::new();
    for line in lines.iter().filter(|line| line["type"] == "observation") {
        let (sender, recipients, payload) = event(line);
        let key = (sender, time(line, "created"));
        let action = actions.get(&key).unwrap_or_else(|| {
            panic!("every observation joins an action by sender and creation time: {line}")
        });
        let (_, sent_to, sent) = event(action);
        assert_eq!(
            (&recipients, payload),
            (&sent_to, sent),
            "an observation and its action are the same event: {line} against {action}"
        );
        assert!(
            observed.insert((key, agent(line))),
            "an agent observes an event once: {line}"
        );
    }
    for (key, action) in &actions {
        let (_, recipients, _) = event(action);
        for who in recipients {
            let who = who.as_str().expect("a recipient is an agent id");
            assert!(
                observed.contains(&(*key, who)),
                "every recipient of an action observes it, but {who} did not: {action}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A well-formed trajectory: agent `a` pops a start and an event from
    /// `b` in one cycle and replies, then runs a cycle on its timeout, then
    /// pops a stop. `b`'s side is here too, because the join is between
    /// agents and cannot be checked from one alone.
    fn good() -> Vec<Value> {
        vec![
            json!({"type": "control", "agent": "a", "seq": 0, "created": 10, "received": 30,
                   "control": "start"}),
            json!({"type": "observation", "agent": "a", "seq": 1, "created": 20, "received": 30,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "seq": 2, "created": 40,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "inputs": [0, 1], "outputs": [2]}),
            json!({"type": "cycle", "agent": "a", "t_start": 60, "t_stop": 70,
                   "woken": "timeout", "inputs": [], "outputs": []}),
            json!({"type": "control", "agent": "a", "seq": 3, "created": 75, "received": 80,
                   "control": "stop"}),
            json!({"type": "cycle", "agent": "a", "t_start": 80, "t_stop": 81, "woken": "queue",
                   "inputs": [3], "outputs": []}),
            json!({"type": "action", "agent": "b", "seq": 0, "created": 20,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 15, "t_stop": 25, "woken": "timeout",
                   "inputs": [], "outputs": [0]}),
            json!({"type": "observation", "agent": "b", "seq": 1, "created": 40, "received": 45,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 45, "t_stop": 46, "woken": "queue",
                   "inputs": [1], "outputs": []}),
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
    #[should_panic(expected = "received before it was created")]
    fn an_observation_received_before_it_was_sent_is_caught() {
        check(&edited(1, |line| line["created"] = json!(31)));
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn an_input_received_at_other_than_its_cycles_start_is_caught() {
        let mut lines = good();
        lines[0]["received"] = json!(29);
        lines[1]["received"] = json!(29);
        lines[3]["t_start"] = json!(30);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "created within its cycle's window")]
    fn an_action_created_outside_its_cycle_is_caught() {
        check(&edited(2, |line| line["created"] = json!(51)));
    }

    #[test]
    #[should_panic(expected = "woken by the timeout has no observations")]
    fn a_timeout_cycle_with_an_observation_is_caught() {
        let mut lines = good();
        // `b`'s observation of `a`'s reply, filed under the cycle that b
        // ran on its timeout.
        lines[10]["woken"] = json!("timeout");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "woken by the queue popped something")]
    fn a_queue_cycle_that_popped_nothing_is_caught() {
        check(&edited(4, |line| line["woken"] = json!("queue")));
    }

    #[test]
    #[should_panic(expected = "woken by the queue or the timeout")]
    fn a_cycle_woken_by_something_else_is_caught() {
        check(&edited(4, |line| line["woken"] = json!("thinking")));
    }

    #[test]
    #[should_panic(expected = "contiguous from zero")]
    fn a_gap_in_sequence_numbers_is_caught() {
        let mut lines = good();
        lines[5]["seq"] = json!(4);
        lines[6]["inputs"] = json!([4]);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "observed only by its recipients")]
    fn an_event_delivered_to_a_non_recipient_is_caught() {
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
    #[should_panic(expected = "names the agent that took it as its sender")]
    fn an_action_recorded_by_somebody_other_than_its_sender_is_caught() {
        check(&edited(2, |line| line["agent"] = json!("c")));
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
        lines.remove(6);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "joins an action by sender and creation time")]
    fn an_observation_of_something_nobody_sent_is_caught() {
        check(&edited(1, |line| line["created"] = json!(21)));
    }

    #[test]
    #[should_panic(expected = "the same event")]
    fn an_observation_that_disagrees_with_its_action_is_caught() {
        check(&edited(1, |line| {
            line["event"]["payload"] = json!({"Step": 99});
        }));
    }

    #[test]
    #[should_panic(expected = "every recipient of an action observes it")]
    fn an_action_nobody_received_is_caught() {
        let mut lines = good();
        // `a`'s reply never reaches `b`, whose last cycle then popped
        // nothing at all.
        lines.remove(9);
        lines[9]["inputs"] = json!([]);
        lines[9]["woken"] = json!("timeout");
        check(&lines);
    }
}
