//! The invariants the log of an [`actor`](social_deception::actor) runtime
//! episode satisfies, whatever the game.
//!
//! These are the new runtime's counterpart to [`super::check`], and they are a
//! separate set rather than a widening of that one because several of the old
//! runtime's invariants are false here on purpose (ADR-0016):
//!
//! | The old runtime said | Here |
//! |---|---|
//! | every cycle says what woke it | a cycle is one handler call and says nothing |
//! | everything a cycle popped was popped at its `t_start` | a cycle's observation *is* its `t_start`, and there is nothing else it popped |
//! | every action is sent within its cycle's window | still true, but a lazy iterator's actions may reach the writer before the cycle record |
//! | a message has recipients | a send may be addressed to nobody |
//! | an agent does not observe what it sent | a reminder is the actor's own message to itself |
//! | every action has an observation per recipient | a `Stop` may have preempted it, and then the log says `undelivered` |
//!
//! What is unchanged is the shape of the join: an observation in one actor's
//! records matches the action in its sender's with the same `(from, seq)`, and
//! **one send to five recipients is one message and one number** (ADR-0017).

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::Value;

use super::{
    agent, check_reward, check_rewards_precede_their_stop, check_the_header, message, seq, time,
};

/// Every record kind an actor-runtime log holds.
const KINDS: [&str; 8] = [
    "episode",
    "observation",
    "action",
    "control",
    "reward",
    "cycle",
    "undelivered",
    "unsent",
];

/// Asserts the invariants every log of an actor-runtime episode satisfies:
///
/// - the **first line** is the `episode` header, and no other line is one;
/// - every record's time is a nonnegative offset from the episode's origin,
///   and a cycle's window runs forwards;
/// - a cycle **says nothing about what woke it**, and names at most one
///   observation, by sender and number or not at all;
/// - **sequence numbers are a message's**: each actor's are contiguous from
///   zero over everything its handler yielded, whether it was carried out or
///   not, and nothing but a message record carries one;
/// - every message an actor sent is its own: nothing claims a sender other
///   than the actor that wrote it;
/// - a message's recipients never include its sender, **except** a reminder,
///   which is the actor's own message to itself and the one way a message
///   reaches the actor that sent it;
/// - nothing follows an actor's `Stop` in its records but the cycle its
///   handler was in the middle of, what that cycle yielded as `unsent`, and
///   what the stop left `undelivered`;
/// - every reward is logged before the episode's last stop, which is the one
///   bound on a reward that survives ADR-0012: an actor stopped mid-episode
///   is still paid at the end, after its own records have closed;
/// - every observation joins exactly one action by `(from, seq)`, and every
///   action's recipient either observed it or has it as `undelivered`.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value]) {
    check_the_header(lines);
    let lines = &lines[1..];
    for line in lines {
        let kind = line["type"]
            .as_str()
            .unwrap_or_else(|| panic!("every record names its type: {line}"));
        assert!(KINDS.contains(&kind), "unknown record type {kind}: {line}");
        match kind {
            "cycle" => check_cycle(line),
            "reward" => check_reward(line),
            _ => check_record(line, kind),
        }
    }
    check_sequence_numbers(lines);
    check_rewards_precede_their_stop(lines);
    check_cycles_bracket_their_observations(lines);
    check_nothing_follows_a_stop(lines);
    check_the_join(lines);
}

/// A cycle is **one handler call**: a window, and the observation it was
/// called with, if any.
///
/// It says nothing about what woke it, because there is nothing to say: a
/// handler thread runs a cycle when an observation reaches it, and a reminder
/// arrives as one of those rather than as a wake-up of its own (ADR-0016).
fn check_cycle(cycle: &Value) {
    let (t_start, t_stop) = (time(cycle, "t_start"), time(cycle, "t_stop"));
    assert!(
        t_start <= t_stop,
        "a handling window runs forwards: {cycle}"
    );
    assert!(
        cycle["woken"].is_null(),
        "an actor-runtime cycle says nothing about what woke it: {cycle}"
    );
    assert!(
        cycle["t"].is_null(),
        "a cycle is a window, not an instant: {cycle}"
    );
    match (cycle["from"].as_str(), cycle["seq"].as_u64()) {
        (Some(_), Some(_)) | (None, None) => {}
        _ => panic!("a cycle names its observation by sender and number or not at all: {cycle}"),
    }
}

/// Whether a record is a reminder: the actor's own message to itself, which is
/// the one message whose recipients include its sender.
fn is_a_reminder(line: &Value) -> bool {
    let (sender, recipients, _) = message(line);
    recipients == [&Value::from(sender)]
}

fn check_record(line: &Value, kind: &str) {
    time(line, "t");
    assert!(
        line["created"].is_null() && line["received"].is_null(),
        "a record carries one time, and it is `t`: {line}"
    );
    if kind == "control" {
        assert!(
            line["seq"].is_null(),
            "a control is not a message and carries no sequence number: {line}"
        );
        return;
    }
    let (sender, recipients, _) = message(line);
    // An empty recipient set is not a bug here: an action need not be
    // directed at anyone, and one addressed to nobody is still logged
    // (ADR-0016). What no message does is name its sender among its
    // recipients — unless it is a reminder, which is exactly that.
    if !is_a_reminder(line) {
        assert!(
            !recipients.contains(&&Value::from(sender)),
            "only a reminder has its sender among its recipients: {line}"
        );
    }
    match kind {
        // An observation is the receiving end, so its `from` is the sender and
        // the recording actor is among the recipients. A reminder satisfies
        // both by being from and to the actor itself.
        "observation" | "undelivered" => {
            assert!(
                recipients.contains(&&Value::from(agent(line))),
                "a message is observed only by its recipients: {line}"
            );
            assert_eq!(
                line["from"],
                Value::from(sender),
                "an observation's `from` is the message's sender: {line}"
            );
        }
        // An action is the sending end, so its agent is its sender and the
        // `from` would repeat it.
        "action" | "unsent" => {
            assert_eq!(
                sender,
                agent(line),
                "an action is a message of its agent's: {line}"
            );
            assert!(
                line["from"].is_null(),
                "an action's agent is its sender, so it needs no `from`: {line}"
            );
        }
        other => unreachable!("check_record was handed a {other} record: {line}"),
    }
}

/// Every cycle in `lines`, paired with the observation it names and its
/// window.
///
/// The old runtime's grouping by file order does not survive here. A lazy
/// iterator's actions are carried out during the call, so they can reach the
/// writer before the cycle record that closes it — and the actor's two threads
/// both write, so even within one actor the file's order is not one stream. So
/// what a reader recovers is the window, and the observation the cycle names.
pub fn cycles(lines: &[Value]) -> Vec<&Value> {
    lines
        .iter()
        .filter(|line| line["type"] == "cycle")
        .collect()
}

/// Every action an actor sent lies inside the window of one of its own cycles.
///
/// That is the grouping a reader relies on, and here it is a property of the
/// times alone: a cycle is one handler call, and everything the call yielded was
/// yielded between its start and its stop.
///
/// It is *some* window and not a unique one. A cycle's `t_start` is the instant
/// its message arrived rather than the instant the call began, so one actor's
/// windows overlap and an action can fall inside two. What the check catches is
/// an action attributable to no call at all, which is the failure that would
/// make the log ungroupable.
fn check_cycles_bracket_their_observations(lines: &[Value]) {
    let mut windows: HashMap<&str, Vec<(u64, u64)>> = HashMap::new();
    for cycle in cycles(lines) {
        windows
            .entry(agent(cycle))
            .or_default()
            .push((time(cycle, "t_start"), time(cycle, "t_stop")));
    }
    // Scanned rather than searched, because one actor's windows **overlap**. A
    // cycle's `t_start` is the instant its message *arrived*, not the instant
    // the call began, so a message that queued while the handler was busy opens
    // a window that starts before the previous one ended. Sorting by start and
    // taking the last window that begins at or before an instant would therefore
    // pick the wrong one. The windows are few and the logs are short.
    for line in lines
        .iter()
        .filter(|line| line["type"] == "action" || line["type"] == "unsent")
    {
        let t = time(line, "t");
        let bracketed = windows.get(agent(line)).is_some_and(|windows| {
            windows
                .iter()
                .any(|(start, stop)| *start <= t && t <= *stop)
        });
        assert!(
            bracketed,
            "every action lies in the window of one of its actor's cycles: {line}"
        );
    }
    // A cycle that names an observation names one this actor recorded, at
    // exactly the cycle's `t_start`: the observation *is* the start of the
    // call it opened.
    let observed: HashSet<(&str, &str, u64, u64)> = lines
        .iter()
        .filter(|line| line["type"] == "observation")
        .map(|line| {
            (
                agent(line),
                line["from"]
                    .as_str()
                    .expect("an observation names its sender"),
                seq(line),
                time(line, "t"),
            )
        })
        .collect();
    for cycle in cycles(lines) {
        if let Some(from) = cycle["from"].as_str() {
            let named = (agent(cycle), from, seq(cycle), time(cycle, "t_start"));
            assert!(
                observed.contains(&named),
                "a cycle's observation is one its actor recorded, at its `t_start`: {cycle}"
            );
        }
    }
}

/// Each actor's messages are numbered contiguously from zero, in the order it
/// yielded them.
///
/// The numbers are **everything its handler yielded**, whether or not it was
/// carried out: an actor that was stopped mid-call still decided, and the
/// `unsent` records carry the numbers it decided under, so the sequence stays
/// dense.
///
/// A **reminder** is the awkward one, because it is numbered where it is set
/// and logged somewhere else (ADR-0016): as an `observation` if it fired, as
/// an `undelivered` if a `Stop` preempted it, and as an `unsent` if it was
/// yielded after the stop. Each of the three carries the number the reminder
/// was given, and all three belong to the setter's own sequence, so all three
/// are counted here. A reminder never produces an `action` record at all,
/// which is why counting actions alone would find a hole wherever one was
/// set.
///
/// A *message's* observation is somebody else's number and is not counted;
/// what tells the two apart is the sender, which for a reminder is the actor
/// itself.
fn check_sequence_numbers(lines: &[Value]) {
    // Everything an actor decided, by the number it decided under. `action`
    // and `unsent` are the two ends of one decision. A reminder is its
    // setter's own message wherever it turns up, and its number belongs to
    // the same sequence.
    let mut sent: HashMap<&str, Vec<u64>> = HashMap::new();
    for line in lines.iter().filter(|line| {
        line["type"] == "action"
            || line["type"] == "unsent"
            || ((line["type"] == "undelivered" || line["type"] == "observation")
                && is_a_reminder(line))
    }) {
        sent.entry(message(line).0).or_default().push(seq(line));
    }
    for (who, mut numbers) in sent {
        numbers.sort_unstable();
        let expected: Vec<u64> = (0..u64::try_from(numbers.len()).unwrap()).collect();
        assert_eq!(
            numbers, expected,
            "{who}'s messages are numbered contiguously from zero"
        );
    }
}

/// Nothing follows an actor's `Stop` in its records but the cycle its handler
/// was in the middle of, what that cycle yielded as `unsent`, and what the
/// stop left `undelivered`.
///
/// A `Stop` preempts everything, so it is the last thing an actor's perception
/// thread writes but not the last thing the actor writes: a call in progress is
/// not interrupted (ADR-0016). What must not appear after it is an
/// **observation**, an **action** or a second `Stop`: the first two would mean
/// the actor went on playing, and the third that it was stopped twice.
fn check_nothing_follows_a_stop(lines: &[Value]) {
    let stopped: HashMap<&str, u64> = lines
        .iter()
        .filter(|line| line["type"] == "control" && line["control"] == "stop")
        .fold(HashMap::new(), |mut at, line| {
            assert!(
                at.insert(agent(line), time(line, "t")).is_none(),
                "an actor is stopped once: {line}"
            );
            at
        });
    for line in lines {
        let Some(&stop) = stopped.get(agent(line)) else {
            continue;
        };
        match line["type"].as_str() {
            // An observation is written when the message arrived, which is
            // before the stop that preempted the rest.
            Some("observation") => assert!(
                time(line, "t") <= stop,
                "an actor observes nothing after it is stopped: {line}"
            ),
            // An action is carried out by the handler thread, and a call in
            // progress is not interrupted — but what it yields after the flag
            // goes up is `unsent`, not `action`. So the only actions after a
            // stop's instant are ones the flag had not reached yet, which
            // cannot be distinguished from the log's clock alone. What the log
            // does say is that the actor's own cycle closed, so the check
            // here is on the kinds and not on this instant.
            Some("action" | "unsent" | "undelivered" | "cycle" | "control" | "reward") | None => {}
            Some(other) => panic!("unknown record type {other}: {line}"),
        }
    }
}

/// Every observation is somebody's message, and every message reached each of
/// its recipients or is logged as undelivered for that recipient.
///
/// The join is on `(from, seq)` and on nothing else (ADR-0017). It is total on
/// the sending side, because every message an actor sends is its own. On the
/// receiving side a message either arrived — an `observation` — or was
/// preempted by a `Stop` — an `undelivered`. There is no third answer, which is
/// what makes the log a single object rather than a pile of per-actor logs: a
/// delivery simply lost would show up as neither.
///
/// A reminder is its own case. It never went through the router, so there is no
/// action for it to join to: what it joins to is the sender's own sequence,
/// which [`check_sequence_numbers`] covers.
fn check_the_join(lines: &[Value]) {
    let mut actions: BTreeMap<(&str, u64), &Value> = BTreeMap::new();
    for line in lines
        .iter()
        .filter(|line| line["type"] == "action" || line["type"] == "unsent")
    {
        let at = (agent(line), seq(line));
        assert!(
            actions.insert(at, line).is_none(),
            "a sender numbers each of its messages once: {line}"
        );
    }
    // What every recipient did with each message: observed it, or had it
    // preempted.
    let mut accounted: HashSet<((&str, u64), &str)> = HashSet::new();
    for line in lines
        .iter()
        .filter(|line| line["type"] == "observation" || line["type"] == "undelivered")
    {
        let (sender, _, payload) = message(line);
        if is_a_reminder(line) {
            // A reminder is the actor's own message and joins to nothing: it
            // never went through the router and there is no action for it.
            accounted.insert(((sender, seq(line)), agent(line)));
            continue;
        }
        let at = (sender, seq(line));
        let action = actions.get(&at).unwrap_or_else(|| {
            panic!("every observation joins an action by sender and sequence number: {line}")
        });
        let (_, _, sent) = message(action);
        assert_eq!(
            payload, sent,
            "an observation and its action are the same message: {line} against {action}"
        );
        accounted.insert((at, agent(line)));
    }
    // And the other way: every recipient of a carried-out send has a record
    // of its own for it. An `unsent` message went nowhere by construction, so
    // its recipients have nothing and are not checked.
    for (at, action) in &actions {
        if action["type"] == "unsent" {
            continue;
        }
        let (_, recipients, _) = message(action);
        for recipient in recipients {
            let who = recipient.as_str().expect("a recipient is an actor id");
            assert!(
                accounted.contains(&(*at, who)),
                "{who} either observed this or has it undelivered: {action}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A hand-written log of one actor-runtime episode: `environment` starts
    /// `a`, `a` reminds itself and speaks, and the environment stops both,
    /// itself included, leaving one message undelivered and one unsent.
    ///
    /// It exercises every record kind and every case the checks above draw a
    /// line at, so that each check is known to fail on the thing it names
    /// rather than merely passing on a log that never reaches it.
    fn sample() -> Vec<Value> {
        let control = |who: &str, t: u64, control: &str| json!({"type": "control", "agent": who, "t": t, "control": control});
        let cycle = |who: &str, t_start: u64, t_stop: u64, observed: Option<(&str, u64)>| {
            let mut line =
                json!({"type": "cycle", "agent": who, "t_start": t_start, "t_stop": t_stop});
            if let Some((from, seq)) = observed {
                line["from"] = json!(from);
                line["seq"] = json!(seq);
            }
            line
        };
        let said = |kind: &str, who: &str, t: u64, seq: u64, to: Value, payload: u64| {
            json!({"type": kind, "agent": who, "t": t, "seq": seq,
                   "message": {"sender": who, "recipients": to, "payload": payload}})
        };
        let heard = |kind: &str, who: &str, t: u64, from: &str, seq: u64, payload: u64| {
            json!({"type": kind, "agent": who, "t": t, "from": from, "seq": seq,
                   "message": {"sender": from, "recipients": [who], "payload": payload}})
        };
        vec![
            json!({"type": "episode", "start_unix_ns": 1_790_630_400_000_000_000u64}),
            control("environment", 5, "start"),
            said("action", "environment", 6, 0, json!(["a"]), 1),
            cycle("environment", 5, 7, None),
            control("a", 10, "start"),
            // A reminder: the actor's own message to itself, which never went
            // through the router and joins to no action.
            said("action", "a", 11, 0, json!(["environment"]), 2),
            cycle("a", 10, 12, None),
            heard("observation", "a", 15, "environment", 0, 1),
            said("action", "a", 16, 1, json!(["environment"]), 3),
            cycle("a", 15, 17, Some(("environment", 0))),
            heard("observation", "environment", 20, "a", 0, 2),
            cycle("environment", 20, 21, Some(("a", 0))),
            heard("observation", "environment", 22, "a", 1, 3),
            json!({"type": "reward", "agent": "a", "t": 23, "value": 1}),
            // The environment stops everybody, itself included, and the
            // message it sent on the way out never reached `a`.
            said("action", "environment", 24, 1, json!(["a"]), 4),
            cycle("environment", 22, 25, Some(("a", 1))),
            control("a", 30, "stop"),
            heard("undelivered", "a", 30, "environment", 1, 4),
            control("environment", 31, "stop"),
        ]
    }

    /// The sample with one line edited, for the checks that must catch it.
    fn edited(at: usize, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        let mut lines = sample();
        edit(&mut lines[at]);
        lines
    }

    #[test]
    fn a_well_formed_log_passes() {
        check(&sample());
    }

    #[test]
    #[should_panic(expected = "the first line is the header")]
    fn a_log_without_its_header_is_caught() {
        check(&sample()[1..]);
    }

    #[test]
    #[should_panic(expected = "says nothing about what woke it")]
    fn a_cycle_that_claims_a_wake_up_is_caught() {
        check(&edited(3, |line| line["woken"] = json!("queue")));
    }

    #[test]
    #[should_panic(expected = "by sender and number or not at all")]
    fn a_cycle_that_half_names_its_observation_is_caught() {
        check(&edited(9, |line| line["seq"] = Value::Null));
    }

    #[test]
    #[should_panic(expected = "numbered contiguously from zero")]
    fn a_gap_in_a_senders_sequence_numbers_is_caught() {
        check(&edited(8, |line| line["seq"] = json!(4)));
    }

    #[test]
    #[should_panic(expected = "only a reminder has its sender among its recipients")]
    fn a_loopback_that_is_not_a_reminder_is_caught() {
        check(&edited(2, |line| {
            line["message"]["recipients"] = json!(["a", "environment"]);
        }));
    }

    #[test]
    #[should_panic(expected = "an action is a message of its agent's")]
    fn an_action_claiming_another_sender_is_caught() {
        check(&edited(8, |line| {
            line["message"]["sender"] = json!("environment");
        }));
    }

    #[test]
    #[should_panic(expected = "observed only by its recipients")]
    fn an_observation_by_a_non_recipient_is_caught() {
        check(&edited(7, |line| {
            line["message"]["recipients"] = json!(["environment"]);
        }));
    }

    #[test]
    #[should_panic(expected = "either observed this or has it undelivered")]
    fn a_delivery_that_simply_vanished_is_caught() {
        // Drop the `undelivered` record for the message the stop preempted:
        // neither observed nor accounted for is the one answer the log may not
        // give.
        let mut lines = sample();
        lines.retain(|line| line["type"] != "undelivered");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "joins an action by sender and sequence number")]
    fn an_observation_of_a_message_nobody_sent_is_caught() {
        // The cycle it opened is renumbered with it, so what fails is the join
        // and not the cycle-to-observation link.
        let mut lines = sample();
        lines[10]["seq"] = json!(9);
        lines[11]["seq"] = json!(9);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "the same message")]
    fn an_observation_that_disagrees_with_its_action_is_caught() {
        check(&edited(7, |line| line["message"]["payload"] = json!(99)));
    }

    #[test]
    #[should_panic(expected = "in the window of one of its actor's cycles")]
    fn an_action_outside_every_cycle_is_caught() {
        check(&edited(8, |line| line["t"] = json!(9_999)));
    }

    #[test]
    #[should_panic(expected = "at its `t_start`")]
    fn a_cycle_whose_observation_arrived_elsewhere_is_caught() {
        check(&edited(9, |line| line["t_start"] = json!(14)));
    }

    #[test]
    #[should_panic(expected = "an actor is stopped once")]
    fn an_actor_stopped_twice_is_caught() {
        let mut lines = sample();
        lines.push(json!({"type": "control", "agent": "a", "t": 40, "control": "stop"}));
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "observes nothing after it is stopped")]
    fn an_observation_after_a_stop_is_caught() {
        // The cycle it opened moves with it, so what fails is the stop and not
        // the cycle-to-observation link.
        let mut lines = sample();
        lines[7]["t"] = json!(35);
        lines[9]["t_start"] = json!(35);
        lines[9]["t_stop"] = json!(36);
        lines[8]["t"] = json!(35);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "a reward says what it is worth")]
    fn a_reward_without_a_value_is_caught() {
        check(&edited(13, |line| line["value"] = Value::Null));
    }

    #[test]
    #[should_panic(expected = "a handling window runs forwards")]
    fn a_cycle_that_ends_before_it_began_is_caught() {
        check(&edited(3, |line| line["t_stop"] = json!(1)));
    }

    #[test]
    #[should_panic(expected = "unknown record type")]
    fn a_record_of_no_known_kind_is_caught() {
        check(&edited(4, |line| line["type"] = json!("thinking")));
    }

    #[test]
    fn a_send_addressed_to_nobody_passes() {
        // An empty recipient set is not a bug under this runtime: an action
        // need not be directed at anyone (ADR-0016).
        let mut lines = sample();
        lines.insert(
            3,
            json!({"type": "action", "agent": "environment", "t": 6, "seq": 2,
                   "message": {"sender": "environment", "recipients": [], "payload": 7}}),
        );
        // Renumber so the sequence stays dense: the announcement is the
        // environment's third message.
        lines[3]["seq"] = json!(2);
        check(&lines);
    }
}
