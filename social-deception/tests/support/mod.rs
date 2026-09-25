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
///   and at exactly the `t_start` of the cycle that lists it, with no
///   exception: everything a cycle pops, it pops at its start (ADR-0009);
/// - every action was created within the window of the cycle that lists it;
/// - a cycle's inputs followed by its outputs are exactly the records its
///   agent wrote since its previous cycle, and every record belongs to some
///   cycle;
/// - per-agent sequence numbers are contiguous and strictly increasing from
///   zero, in file order;
/// - a cycle lists **at most one** observation among its inputs, and a
///   cycle woken by the queue has at least one input;
/// - no event has its sender among its recipients; an observation lists the
///   agent that recorded it among the recipients, and an action names it as
///   the sender;
/// - nothing follows an agent's `Stop` in its trajectory but the end of the
///   cycle that popped it;
/// - a `reward` names an agent and a value, carries no sequence number and
///   no receipt, and belongs to no cycle;
/// - every observation joins exactly one action, by sender and creation
///   time, and every action has one matching observation per recipient.
///
/// A reward is outside almost all of it, and deliberately. It belongs to
/// the agent it names and was written by the environment, so it is in no
/// cycle of that agent's, has no sequence number to be contiguous with, and
/// arrives in the file wherever the writer took it. What is left to check
/// is its shape, which is what the reward clause above says, and the one
/// ordering that does hold: it precedes the `Stop` that ends the trajectory
/// it belongs to, since a reward assigned after an agent has been told to
/// stop would be scoring an episode that was already over.
///
/// The last is the one that makes a trajectory a single object rather than a
/// pile of per-agent logs: an observation and the action that produced it
/// are the same event seen from its two ends, and nothing but the sender and
/// the creation time links them. Every action a handler returns is sent, so
/// every action record has an other side and none is exempt from the join.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value]) {
    let records = records(lines);
    // Each cycle's run: the sequence numbers its agent wrote since its
    // previous cycle. `check_grouping` proves the run is exactly the
    // cycle's inputs followed by its outputs.
    let mut runs: HashMap<&str, Vec<u64>> = HashMap::new();
    for line in lines {
        match line["type"].as_str() {
            Some(kind @ ("observation" | "action" | "control")) => {
                check_record(line, kind);
                runs.entry(agent(line)).or_default().push(seq(line));
            }
            // A reward is in no cycle and in no run: it was written by the
            // environment, about somebody else, outside that agent's loop
            // entirely.
            Some("reward") => check_reward(line),
            Some("cycle") => {
                // The run is taken so that it does not outlive its cycle;
                // `check_grouping` is what proves the two agree.
                runs.remove(agent(line));
                check_cycle(line, &records);
            }
            other => panic!("unknown record type {other:?} in {line}"),
        }
    }
    check_sequence_numbers(lines);
    check_grouping(lines);
    check_nothing_follows_a_stop(lines);
    check_rewards_precede_their_stop(lines);
    check_the_join(lines);
}

/// A reward names the agent it belongs to, says when it was logged and what
/// it is worth, and carries neither a sequence number nor a receipt.
///
/// The two absences are the point. A reward has no `seq` because sequence
/// numbers are the agent loop's to assign and this record was not written by
/// that loop, and no `received` because a reward is logged and never sent,
/// so nobody ever received it.
fn check_reward(line: &Value) {
    agent(line);
    time(line, "created");
    assert!(
        line["value"].is_number(),
        "a reward says what it is worth: {line}"
    );
    assert!(
        line["seq"].is_null(),
        "a reward carries no sequence number: {line}"
    );
    assert!(
        line["received"].is_null(),
        "a reward is logged, never sent, so nobody received it: {line}"
    );
    assert!(line["event"].is_null(), "a reward carries no event: {line}");
}

/// Every reward precedes the `Stop` of the agent it belongs to.
///
/// An agent's trajectory ends at its `Stop`, so a reward after one would be
/// scoring an episode that was already over for that agent. The check is on
/// the `created` stamps and not on file order, because a reward is written
/// by the environment's thread and its line lands wherever the writer took
/// it.
fn check_rewards_precede_their_stop(lines: &[Value]) {
    let stopped: HashMap<&str, u64> = lines
        .iter()
        .filter(|line| line["type"] == "control" && line["control"] == "stop")
        .map(|line| (agent(line), time(line, "created")))
        .collect();
    for line in lines.iter().filter(|line| line["type"] == "reward") {
        let Some(stop) = stopped.get(agent(line)) else {
            continue;
        };
        assert!(
            time(line, "created") <= *stop,
            "a reward is logged before the stop that ends its agent's trajectory: {line}"
        );
    }
}

/// The numbered records of `lines`, by agent and sequence number: every
/// record but a cycle, which has no number, and a reward, which has none
/// either and belongs to an agent other than the one that wrote it.
pub fn records(lines: &[Value]) -> HashMap<(&str, u64), &Value> {
    lines
        .iter()
        .filter(|line| numbered(line))
        .map(|line| ((agent(line), seq(line)), line))
        .collect()
}

/// Whether a record carries a sequence number: everything its agent's own
/// loop wrote, which is everything but a cycle and a reward.
pub fn numbered(line: &Value) -> bool {
    line["type"] != "cycle" && line["type"] != "reward"
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

fn check_record(line: &Value, kind: &str) {
    // Everything that was popped says when, and nothing is popped before it
    // was created. An action is the exception: it was never received.
    if line["received"].is_null() {
        assert!(
            kind == "action",
            "only an action records no receipt: {line}"
        );
    } else {
        assert!(
            time(line, "created") <= time(line, "received"),
            "nothing is received before it was created: {line}"
        );
    }
    match kind {
        "control" => {}
        "observation" => {
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
        "action" => {
            // An action has no `received`: its sender knows only when it
            // sent it, and when each recipient got it is in that
            // recipient's own observation record. The check above says so.
            let (sender, recipients, _) = event(line);
            check_recipients(line, sender, &recipients);
            assert_eq!(
                sender,
                agent(line),
                "an action names the agent that took it as its sender: {line}"
            );
        }
        // `check` matched the kind before calling; there is no other.
        _ => unreachable!("check_record was handed a {kind} record: {line}"),
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
    // A cycle is one decision, so it observed one thing or nothing at all
    // (ADR-0008). This is what makes `t_start` and `t_stop` bracket a single
    // decision rather than the time to work through an arbitrary pile, and
    // so what lets a reader take the gap between them for deliberation.
    //
    // It is asserted of every cycle, whatever woke it. `timeout` says the
    // deadline had passed when the cycle began, not that the cycle observed
    // nothing: a deadline that passes while an event is waiting joins that
    // event's cycle. What no cycle does is observe twice.
    assert!(
        observations <= 1,
        "a cycle handles at most one observation, but this one lists \
         {observations}: {cycle}"
    );
    if woken == "queue" {
        assert!(
            !inputs.is_empty(),
            "a cycle woken by the queue popped something: {cycle}"
        );
    }
    // Everything a cycle popped, it popped at the cycle's start: the
    // controls and the observation alike, with no exception (ADR-0009).
    // That is what makes the gap between an input's `received` and an
    // output's `created` the agent's deliberation.
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
    for line in lines.iter().filter(|line| numbered(line)) {
        let expected = next.entry(agent(line)).or_insert(0);
        assert_eq!(
            seq(line),
            *expected,
            "sequence numbers are contiguous from zero: {line}"
        );
        *expected += 1;
    }
}

/// A cycle's inputs followed by its outputs are exactly the records its
/// agent wrote since its previous cycle.
///
/// Exactly, with nothing taken out: every record an agent writes within a
/// cycle is either something it popped or something it sent, since
/// everything a handler returns is sent (ADR-0009). That is the grouping
/// every reader of a trajectory relies on.
fn check_grouping(lines: &[Value]) {
    let mut pending: HashMap<&str, Vec<u64>> = HashMap::new();
    for line in lines {
        // A reward belongs to no cycle of the agent it names: the
        // environment wrote it, outside that agent's loop entirely.
        if line["type"] == "reward" {
            continue;
        }
        let agent = agent(line);
        let pending = pending.entry(agent).or_default();
        if line["type"] == "cycle" {
            let listed: Vec<u64> = seqs(line, "inputs").chain(seqs(line, "outputs")).collect();
            let written: Vec<u64> = std::mem::take(pending);
            assert_eq!(
                listed, written,
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

/// Nothing follows an agent's `Stop` in its trajectory but the end of the
/// cycle that popped it.
///
/// A `Stop` is the last thing an agent ever pops, so its trajectory ends
/// there: one cycle record to close the cycle, and nothing after it. An
/// agent that wrote anything more either kept running after it was told to
/// stop or was told twice, and the log would be claiming both.
fn check_nothing_follows_a_stop(lines: &[Value]) {
    let mut stopped: HashSet<&str> = HashSet::new();
    let mut closed: HashSet<&str> = HashSet::new();
    for line in lines {
        // A reward is not the agent's own record and is not placed by file
        // order: the environment writes it on its own thread, so it can
        // land after the agent's last cycle. That it was *logged* before
        // the stop is `check_rewards_precede_their_stop`'s business, on the
        // stamps, which is where the claim can actually be made.
        if line["type"] == "reward" {
            continue;
        }
        let agent = agent(line);
        assert!(
            !closed.contains(agent),
            "an agent's trajectory ends with the cycle that popped its stop: {line}"
        );
        if line["type"] == "cycle" {
            if stopped.contains(agent) {
                closed.insert(agent);
            }
        } else if line["type"] == "control" && line["control"] == "stop" {
            assert!(stopped.insert(agent), "an agent is stopped once: {line}");
        }
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

    /// A trajectory in which `a` reached a `Stop` that had been queued
    /// behind an event: it observed the event, answered it, and popped the
    /// stop in the cycle after (ADR-0009). `b`'s side is here for the join.
    ///
    /// This is the shape the episode produces only when it is abandoning a
    /// run that has already failed; on every other path it holds a stop
    /// back until nothing is in flight, and the stop arrives to an empty
    /// queue.
    fn stopped_behind_an_event() -> Vec<Value> {
        vec![
            json!({"type": "control", "agent": "a", "seq": 0, "created": 10, "received": 30,
                   "control": "start"}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 31, "woken": "queue",
                   "inputs": [0], "outputs": []}),
            json!({"type": "observation", "agent": "a", "seq": 1, "created": 40, "received": 45,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "seq": 2, "created": 70,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 45, "t_stop": 72, "woken": "queue",
                   "inputs": [1], "outputs": [2]}),
            json!({"type": "control", "agent": "a", "seq": 3, "created": 60, "received": 80,
                   "control": "stop"}),
            json!({"type": "cycle", "agent": "a", "t_start": 80, "t_stop": 81, "woken": "queue",
                   "inputs": [3], "outputs": []}),
            json!({"type": "action", "agent": "b", "seq": 0, "created": 40,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 35, "t_stop": 41,
                   "woken": "timeout", "inputs": [], "outputs": [0]}),
            json!({"type": "observation", "agent": "b", "seq": 1, "created": 70, "received": 75,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 75, "t_stop": 76, "woken": "queue",
                   "inputs": [1], "outputs": []}),
        ]
    }

    #[test]
    fn a_stop_reached_behind_an_event_passes() {
        // The answer to the event was sent, not withheld, and `b` observed
        // it: an agent stopped this way did the work queued ahead of the
        // stop, and the trajectory says so on both sides.
        check(&stopped_behind_an_event());
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn a_control_popped_after_its_cycles_start_is_caught() {
        // Nothing a cycle pops is popped later than its start any more.
        // The old exception — a stop that preempted the cycle it landed in
        // — is gone with preemption itself (ADR-0009), so a control stamped
        // later than `t_start` is simply an input that was not popped when
        // it claims to have been.
        let mut lines = stopped_behind_an_event();
        lines[5]["received"] = json!(81);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn an_ordinary_control_popped_late_is_still_caught() {
        check(&edited(0, |line| line["received"] = json!(31)));
    }

    #[test]
    #[should_panic(expected = "ends with the cycle that popped its stop")]
    fn a_cycle_after_an_agents_stop_is_caught() {
        // A whole cycle after the one that popped the stop: the agent kept
        // running after it was told to stop.
        let mut lines = stopped_behind_an_event();
        lines.insert(
            7,
            json!({"type": "cycle", "agent": "a", "t_start": 90, "t_stop": 91,
                   "woken": "timeout", "inputs": [], "outputs": []}),
        );
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "an agent is stopped once")]
    fn an_agent_stopped_twice_is_caught() {
        // Both stops are popped in the same cycle, so nothing follows the
        // first one but the cycle it belongs to, and it is being told twice
        // that the check has left to catch. The rest of the trajectory is
        // dropped, since a trajectory that ends at the stop is what the
        // other check already asserts.
        let mut lines = stopped_behind_an_event()[..2].to_vec();
        lines[0]["control"] = json!("stop");
        lines.insert(
            1,
            json!({"type": "control", "agent": "a", "seq": 1, "created": 11, "received": 30,
                   "control": "stop"}),
        );
        lines[2]["inputs"] = json!([0, 1]);
        check(&lines);
    }

    /// The good trajectory with a reward for `a`, logged before its stop,
    /// as an environment would have written it.
    fn rewarded() -> Vec<Value> {
        let mut lines = good();
        lines.push(json!({"type": "reward", "agent": "a", "created": 74, "value": 1}));
        lines
    }

    #[test]
    fn a_reward_passes_and_belongs_to_no_cycle() {
        // It carries no sequence number and closes no cycle, and its line
        // sits after the cycle that ended `a`'s trajectory, because the
        // environment wrote it on its own thread.
        check(&rewarded());
    }

    #[test]
    #[should_panic(expected = "carries no sequence number")]
    fn a_reward_with_a_sequence_number_is_caught() {
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last]["seq"] = json!(4);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "nobody received it")]
    fn a_reward_that_claims_to_have_been_received_is_caught() {
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last]["received"] = json!(75);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "says what it is worth")]
    fn a_reward_without_a_value_is_caught() {
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last].as_object_mut().unwrap().remove("value");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "before the stop that ends its agent's trajectory")]
    fn a_reward_logged_after_its_agents_stop_is_caught() {
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last]["created"] = json!(76);
        check(&lines);
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
    fn a_timeout_cycle_that_also_observed_passes() {
        // `timeout` says the deadline had passed when the cycle began, not
        // that the cycle observed nothing. A deadline that passes while an
        // event is waiting joins that event's cycle, which observes it and
        // calls `handle` like any other, so this is a shape the loop really
        // produces and the checker must accept.
        let mut lines = good();
        lines[10]["woken"] = json!("timeout");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "at most one observation")]
    fn a_cycle_that_observed_twice_is_caught() {
        // A cycle is one decision, so it conditions on one observation or
        // on none. Two would be the batch ADR-0008 removed, and a reader
        // taking the window for deliberation would be reading the time to
        // work through a pile.
        let mut lines = good();
        lines.insert(
            2,
            json!({"type": "observation", "agent": "a", "seq": 2, "created": 20, "received": 30,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 8}}}),
        );
        // Renumber the rest of `a`'s records around the extra one.
        lines[3]["seq"] = json!(3);
        lines[4]["inputs"] = json!([0, 1, 2]);
        lines[4]["outputs"] = json!([3]);
        lines[6]["seq"] = json!(4);
        lines[7]["inputs"] = json!([4]);
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
