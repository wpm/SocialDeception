//! Helpers shared by the integration tests: the [`collatz`] environment, a
//! [`TempDir`] to write a log in, reading a log file back, checking the
//! invariants every log satisfies whatever the game, and, in [`werewolf`],
//! the invariants the log of a game of Werewolf satisfies on top of them.
//!
//! The checks here are properties of the log, not of any game. They are
//! meant to run unchanged against episodes where no independent check on the
//! content is available.
//!
//! # Two runtimes, two sets of invariants
//!
//! [`check`] is the old runtime's. Several of its invariants are false under
//! the [`actor`](social_deception::actor) runtime on purpose — a reminder is
//! an actor's own message to itself, a cycle says nothing about what woke it,
//! a send may be addressed to nobody — so that runtime's invariants are in
//! [`actor`], and its Collatz ring is in [`collatz_actor`]. Werewolf runs on
//! the old runtime until issue #116, so [`werewolf`] builds on [`check`].

pub mod actor;
pub mod collatz;
pub mod collatz_actor;
mod temp;
pub mod werewolf;

use std::collections::{BTreeMap, HashMap, HashSet};

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

/// Asserts the invariants every log satisfies:
///
/// - the **first line** is the `episode` header, carrying a wall-clock start
///   and nothing else, and no other line is one;
/// - every other record's `t` is a nonnegative offset from the episode's
///   origin, and everything a cycle popped was popped at exactly its
///   `t_start`, with no exception: everything a cycle pops, it pops at its
///   start (ADR-0009);
/// - every action was sent within the window of the cycle it belongs to;
/// - **sequence numbers are a message's, not a record's**: each sender's are
///   contiguous from zero over the messages it sent, and nothing but a
///   message record carries one;
/// - a cycle names at most one observation, and a cycle woken by the queue
///   popped something;
/// - every action is a message of its own agent's: nothing claims a sender
///   other than the actor that sent it, and an agent passing another's
///   message on carries it in an `Envelope` inside its own payload
///   (ADR-0017);
/// - no message has its sender among its recipients; an observation lists the
///   agent that recorded it among the recipients, and its `from` is the
///   message's sender;
/// - nothing follows an agent's `Stop` in its records but the end of the
///   cycle that popped it;
/// - a `reward` names an agent and a value, carries no sequence number and no
///   message, and belongs to no cycle;
/// - every observation joins exactly one action **by `(from, seq)`**, and
///   every action has one matching observation per recipient.
///
/// A reward is outside almost all of it, and deliberately. It belongs to
/// the agent it names and was written by the environment, so it is in no
/// cycle of that agent's and arrives in the file wherever the writer took
/// it. What is left to check is its shape, which is what the reward clause
/// above says, and the one ordering that does hold: it is logged before the
/// episode's last `Stop`, since a reward assigned after the run had finished
/// for everybody would be scoring an episode that no longer existed. It need
/// not precede the stop of its *own* agent: an agent stopped mid-episode, as
/// a dead Werewolf player is (ADR-0012), is paid at the end like everybody
/// else, which a logged reward allows and a sent one would not.
///
/// The last is the one that makes the log a single object rather than a
/// pile of per-agent logs: an observation and the action that produced it are
/// the same message seen from its two ends, and `(from, seq)` is what links
/// them (ADR-0017). One send to five recipients is one message and one
/// number, so one action joins to all of its observations at once. Every
/// action a handler returns is sent, so every action record has an other
/// side and none is exempt from the join.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value]) {
    check_the_header(lines);
    let lines = &lines[1..];
    for line in lines {
        match line["type"].as_str() {
            Some(kind @ ("observation" | "action" | "control")) => check_record(line, kind),
            // A reward is in no cycle: it was written by the environment,
            // about somebody else, outside that agent's loop entirely.
            Some("reward") => check_reward(line),
            Some("cycle") => check_cycle(line),
            other => panic!("unknown record type {other:?} in {line}"),
        }
    }
    check_sequence_numbers(lines);
    check_cycles_bracket_their_records(lines);
    check_nothing_follows_a_stop(lines);
    check_rewards_precede_their_stop(lines);
    check_the_join(lines);
}

/// The log opens with the `episode` header and never mentions it again.
///
/// The header's `start_unix_ns` is the one wall-clock time anywhere in the
/// log, so it is checked for its presence and its type and never for its
/// value: it is different on every run by construction (ADR-0017).
pub(crate) fn check_the_header(lines: &[Value]) {
    let header = lines.first().expect("a log has at least its header");
    assert_eq!(header["type"], "episode", "the first line is the header");
    assert!(
        header["start_unix_ns"].is_u64(),
        "the header anchors the episode to the wall clock: {header}"
    );
    assert!(
        header["agent"].is_null(),
        "the header is nobody's record: {header}"
    );
    for line in &lines[1..] {
        assert_ne!(line["type"], "episode", "one header per log: {line}");
    }
}

/// A reward names the agent it belongs to, says when it was logged and what
/// it is worth, and carries neither a sequence number nor a message.
///
/// The two absences are the point. A reward has no `seq` because a sequence
/// number is a message's and a reward is not a message, and no message
/// because it is logged rather than said.
pub(crate) fn check_reward(line: &Value) {
    agent(line);
    time(line, "t");
    assert!(
        line["value"].is_number(),
        "a reward says what it is worth: {line}"
    );
    assert!(
        line["seq"].is_null(),
        "a reward is not a message and carries no sequence number: {line}"
    );
    assert!(
        line["message"].is_null(),
        "a reward carries no message: {line}"
    );
}

/// Every reward is logged before the episode ends, which is the last `Stop`
/// in the run.
///
/// A reward may *not* precede the stop of the agent it belongs to. An
/// environment may stop one agent while the others run on — Werewolf stops
/// a player in the cycle its death is announced (ADR-0012) — and rewards
/// are handed out when the episode ends, so an agent that left early is
/// paid after its own records have closed. That is sound because a reward
/// is logged rather than sent (ADR-0007): the agent does not have to be
/// there to receive it, and its value is the episode's to decide once the
/// episode is over.
///
/// What still holds is the outer bound. A reward logged after the last stop
/// would be scoring a run that had finished for everybody, with no episode
/// left to have produced it. The check is on the times and not on file
/// order, because a reward is written by the environment's thread and its
/// line lands wherever the writer took it.
fn check_rewards_precede_their_stop(lines: &[Value]) {
    let Some(end) = lines
        .iter()
        .filter(|line| line["type"] == "control" && line["control"] == "stop")
        .map(|line| time(line, "t"))
        .max()
    else {
        return;
    };
    for line in lines.iter().filter(|line| line["type"] == "reward") {
        assert!(
            time(line, "t") <= end,
            "a reward is logged before the episode's last stop: {line}"
        );
    }
}

/// The `episode` header every log opens with, as a hand-written log needs
/// one.
///
/// The anchor is a fixed moment no check reads: it is the one wall-clock time
/// in a log, and what it is worth is that post-processing can line episodes
/// up by it, which no test here does.
#[must_use]
pub fn header() -> Value {
    serde_json::json!({"type": "episode", "start_unix_ns": 1_790_630_400_000_000_000u64})
}

/// Every cycle in `lines`, paired with the records of its agent's that it
/// closes, in the order they were written.
///
/// Line order carries no meaning across agents (ADR-0017), but one agent's
/// records still reach the writer in the order it wrote them, and a cycle
/// record is written after every record of that cycle. So the records of one
/// agent between two of its cycle records belong to the later one, and that
/// is the grouping a reader recovers. The cycle's window is what confirms it,
/// and `check` asserts that.
///
/// A reward is in nobody's cycle: it was written by the environment, about
/// another agent, so it is left out. So is the header, which is nobody's
/// record at all.
///
/// # Panics
///
/// If any agent's records run past its last cycle. Every record an agent
/// writes belongs to some cycle of its own — everything it popped or sent,
/// since everything a handler returns is sent (ADR-0009) — so a leftover is a
/// record no cycle closed, and the grouping this returns would be missing it
/// rather than merely ordering it oddly.
pub fn cycles(lines: &[Value]) -> Vec<(&Value, Vec<&Value>)> {
    let mut pending: HashMap<&str, Vec<&Value>> = HashMap::new();
    let mut grouped = Vec::new();
    for line in lines {
        if line["type"] == "reward" || line["type"] == "episode" {
            continue;
        }
        let run = pending.entry(agent(line)).or_default();
        if line["type"] == "cycle" {
            grouped.push((line, std::mem::take(run)));
        } else {
            run.push(line);
        }
    }
    for (agent, run) in pending {
        assert!(
            run.is_empty(),
            "every record belongs to a cycle, but {agent} left {run:?} after its last"
        );
    }
    grouped
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

/// What a record says about the message it carries: its sender, its
/// recipients and its payload, which together with the key are what an
/// observation and its action must agree on.
pub(crate) fn message(line: &Value) -> (&str, Vec<&Value>, &Value) {
    let message = &line["message"];
    let sender = message["sender"]
        .as_str()
        .expect("a message names its sender");
    let recipients: Vec<&Value> = message["recipients"]
        .as_array()
        .expect("a message lists its recipients")
        .iter()
        .collect();
    (sender, recipients, &message["payload"])
}

fn check_record(line: &Value, kind: &str) {
    // Every record says when the thing it describes happened, as an offset
    // from the episode's origin, and no record carries a second time: the
    // sender's clock reading is not something a message travels with
    // (ADR-0017).
    time(line, "t");
    assert!(
        line["created"].is_null() && line["received"].is_null(),
        "a record carries one time, and it is `t`: {line}"
    );
    match kind {
        "control" => assert!(
            line["seq"].is_null(),
            "a control is not a message and carries no sequence number: {line}"
        ),
        "observation" => {
            let (sender, recipients, _) = message(line);
            check_recipients(line, sender, &recipients);
            assert!(
                recipients.contains(&&Value::from(agent(line))),
                "a message is observed only by its recipients: {line}"
            );
            assert_ne!(
                sender,
                agent(line),
                "an agent does not observe what it sent: {line}"
            );
            assert_eq!(
                line["from"],
                Value::from(sender),
                "an observation's `from` is the message's sender: {line}"
            );
        }
        "action" => {
            // An action records only when it was sent. When each recipient
            // got it is in that recipient's own observation record.
            let (sender, recipients, _) = message(line);
            check_recipients(line, sender, &recipients);
            // Every message an agent sends is its own: there is no way to
            // name another sender, so no record claims one (ADR-0017). An
            // agent passing another's message on sends a message of its own
            // carrying an `Envelope` of what it received, which is inside the
            // payload, where the framework never looks.
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
        // `check` matched the kind before calling; there is no other.
        _ => unreachable!("check_record was handed a {kind} record: {line}"),
    }
}

fn check_recipients(line: &Value, sender: &str, recipients: &[&Value]) {
    assert!(!recipients.is_empty(), "a message has recipients: {line}");
    assert!(
        !recipients.contains(&&Value::from(sender)),
        "no message has its sender among its recipients: {line}"
    );
}

fn check_cycle(cycle: &Value) {
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
    assert!(
        cycle["t"].is_null(),
        "a cycle is a window, not an instant: {cycle}"
    );
    // A cycle names at most one observation because there is one field to
    // name it in, so the shape of the record is the invariant: what is left
    // to check is that a cycle that names one names it whole.
    match (cycle["from"].as_str(), cycle["seq"].as_u64()) {
        (Some(_), Some(_)) | (None, None) => {}
        _ => panic!("a cycle names its observation by sender and number or not at all: {cycle}"),
    }
}

/// A cycle's own records lie inside it: everything it popped at exactly its
/// `t_start`, and everything it sent between `t_start` and `t_stop`.
///
/// This is the grouping a reader of the log relies on, and it is now a
/// property of the times rather than of lists the cycle carries. [`cycles`]
/// recovers the grouping from file order; the windows here are what confirm
/// it, so the two together are the claim that a cycle's records are its own.
fn check_cycles_bracket_their_records(lines: &[Value]) {
    for (cycle, records) in cycles(lines) {
        let (t_start, t_stop) = (time(cycle, "t_start"), time(cycle, "t_stop"));
        let mut observations = 0;
        for record in &records {
            if record["type"] == "action" {
                assert!(
                    t_start <= time(record, "t") && time(record, "t") <= t_stop,
                    "an action is sent within its cycle's window: {record} in {cycle}"
                );
                continue;
            }
            if record["type"] == "observation" {
                observations += 1;
            }
            assert_eq!(
                time(record, "t"),
                t_start,
                "everything a cycle popped was popped at its start: {record} in {cycle}"
            );
        }
        // A cycle is one decision, so it observed one thing or nothing at
        // all (ADR-0008). This is what makes `t_start` and `t_stop` bracket
        // a single decision rather than the time to work through an
        // arbitrary pile, and so what lets a reader take the gap between
        // them for deliberation.
        //
        // It is asserted of every cycle, whatever woke it. `timeout` says
        // the deadline had passed when the cycle began, not that the cycle
        // observed nothing: a deadline that passes while a message is
        // waiting joins that message's cycle. What no cycle does is observe
        // twice.
        assert!(
            observations <= 1,
            "a cycle handles at most one observation, but this one wrote \
             {observations}: {cycle}"
        );
        assert_eq!(
            observations == 1,
            !cycle["from"].is_null(),
            "a cycle names the observation it was called with, and only that: {cycle}"
        );
        if cycle["woken"] == "queue" {
            assert!(
                !records.is_empty(),
                "a cycle woken by the queue popped something: {cycle}"
            );
        }
    }
}

/// Each sender's messages are numbered contiguously from zero, in the order
/// it sent them.
///
/// The numbers are **the messages'**, so this counts each sender's own
/// action records and nothing else: a control, a reward and a cycle carry
/// none, and an observation carries its *sender's* number, which that
/// sender's own records are where the counting happens. Every message an
/// agent sends is its own, relays included, so nothing is skipped and the
/// numbers are dense over everything an agent sent (ADR-0017).
fn check_sequence_numbers(lines: &[Value]) {
    let mut next: HashMap<&str, u64> = HashMap::new();
    for line in lines.iter().filter(|line| line["type"] == "action") {
        // `check_record` has already held the two to be the same, so the
        // record's agent is the sender its numbers belong to.
        let expected = next.entry(agent(line)).or_insert(0);
        assert_eq!(
            seq(line),
            *expected,
            "a sender's messages are numbered contiguously from zero: {line}"
        );
        *expected += 1;
    }
}

/// Nothing follows an agent's `Stop` in its records but the end of the
/// cycle that popped it.
///
/// A `Stop` is the last thing an agent ever pops, so its records end
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
        // times, which is where the claim can actually be made.
        if line["type"] == "reward" {
            continue;
        }
        let agent = agent(line);
        assert!(
            !closed.contains(agent),
            "an agent's records end with the cycle that popped its stop: {line}"
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

/// Every observation is somebody's message, and every message is observed by
/// each of its recipients that was still running. The join is on `(from,
/// seq)` and on nothing else (ADR-0017).
///
/// The join is no longer total on the recipient side, and ADR-0012 is why.
/// An environment may stop one agent while the rest run on, and a message
/// addressed to an agent that has stopped is dropped for that recipient and
/// delivered to the others. So an action may name a recipient with no
/// matching observation anywhere: a trace in the sender's records and
/// none in the recipient's. That is correct for reinforcement learning —
/// the recipient did not observe it, and its records should not pretend
/// otherwise — and anything joining the two sides of a message, replay
/// included, has to allow for it.
///
/// What cannot be allowed is using that as a blanket excuse, because then
/// the check would pass for a delivery that was simply lost. A missing
/// observation is accepted only for a recipient this run actually stopped,
/// and only for a message sent after the last message that recipient did
/// observe. Up to that instant the agent was demonstrably taking delivery,
/// so a gap there is a real failure and still fails here.
fn check_the_join(lines: &[Value]) {
    // Which agents were stopped at all: being stopped is what licenses a
    // missing observation.
    let stopped: HashSet<&str> = lines
        .iter()
        .filter(|line| line["type"] == "control" && line["control"] == "stop")
        .map(agent)
        .collect();
    // Keyed by the message: who sent it and which of theirs it is, which is
    // exactly what an observation names. Every action is its agent's own
    // message, so every one is under this key and the join is total on the
    // sending side (ADR-0017).
    let mut actions: BTreeMap<(&str, u64), &Value> = BTreeMap::new();
    for line in lines.iter().filter(|line| line["type"] == "action") {
        let at = (agent(line), seq(line));
        assert!(
            actions.insert(at, line).is_none(),
            "a sender numbers each of its messages once, or no observation could \
             name which: {line}"
        );
    }
    // The latest message each agent observed, **by when it was sent**, which
    // bounds what that agent may legitimately have missed.
    //
    // The bound cannot be read off the stop's own time. A stop and the
    // messages around it are one cycle's work: the environment returns a
    // batch of effects together and the episode applies that batch's
    // controls before routing its messages, so a message sent earlier in
    // the batch than the stop is still dropped for the agent the batch
    // stopped. What the log does show without guesswork is the last message
    // the agent actually observed, and every message it missed was sent after
    // that one.
    //
    // It is the *send* time, not the arrival, because that is what the other
    // side of the comparison is: an action record says when its sender sent
    // it. Arrival is later than sending by however long the message sat on a
    // queue, so comparing a send against an arrival would call a message
    // legitimately dropped only if it was sent after another had been popped,
    // which under load is a different claim and a false failure.
    let mut last_observed: HashMap<&str, u64> = HashMap::new();
    let mut observed: HashSet<((&str, u64), &str)> = HashSet::new();
    for line in lines.iter().filter(|line| line["type"] == "observation") {
        let (sender, recipients, payload) = message(line);
        let at = (sender, seq(line));
        let action = actions.get(&at).unwrap_or_else(|| {
            panic!("every observation joins an action by sender and sequence number: {line}")
        });
        let (_, sent_to, sent) = message(action);
        assert_eq!(
            payload, sent,
            "an observation and its action are the same message: {line} against {action}"
        );
        // The join above is what ties an observation to the agent that
        // really sent it: it is keyed on the sender and the number, against
        // the messages each agent sent, so an observation naming a sender
        // that sent no such message panics there. A relay is no exception —
        // it is the relaying agent's own message under its own key, and what
        // it passes on is inside its payload.
        assert_eq!(
            recipients, sent_to,
            "an observation and its action name the same recipients: \
             {line} against {action}"
        );
        assert!(
            observed.insert((at, agent(line))),
            "an agent observes a message once: {line}"
        );
        let sent = time(action, "t");
        last_observed
            .entry(agent(line))
            .and_modify(|latest| *latest = (*latest).max(sent))
            .or_insert(sent);
    }
    for (at, action) in &actions {
        let (_, recipients, _) = message(action);
        let sent = time(action, "t");
        for who in recipients {
            let who = who.as_str().expect("a recipient is an actor id");
            if observed.contains(&(*at, who)) {
                continue;
            }
            // Unobserved: allowed only from the sender's cycle that
            // stopped this recipient onward, which is the one way a
            // message legitimately reaches nobody (ADR-0012).
            let dropped =
                stopped.contains(who) && sent > last_observed.get(who).copied().unwrap_or(0);
            assert!(
                dropped,
                "every recipient of a message observes it unless it had stopped, but {who} \
                 did not: {action}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// A well-formed log: agent `a` pops a start and a message from
    /// `b` in one cycle and replies, then runs a cycle on its timeout, then
    /// pops a stop. `b`'s side is here too, because the join is between
    /// agents and cannot be checked from one alone.
    fn good() -> Vec<Value> {
        vec![
            header(),
            json!({"type": "control", "agent": "a", "t": 30, "control": "start"}),
            json!({"type": "observation", "agent": "a", "t": 30, "from": "b", "seq": 0,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "t": 40, "seq": 0,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "from": "b", "seq": 0}),
            json!({"type": "cycle", "agent": "a", "t_start": 60, "t_stop": 70,
                   "woken": "timeout"}),
            json!({"type": "control", "agent": "a", "t": 80, "control": "stop"}),
            json!({"type": "cycle", "agent": "a", "t_start": 80, "t_stop": 81,
                   "woken": "queue"}),
            json!({"type": "action", "agent": "b", "t": 20, "seq": 0,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 15, "t_stop": 25,
                   "woken": "timeout"}),
            json!({"type": "observation", "agent": "b", "t": 45, "from": "a", "seq": 0,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 45, "t_stop": 46, "woken": "queue",
                   "from": "a", "seq": 0}),
        ]
    }

    /// A log in which `r` relays `b`'s message to `c`.
    ///
    /// `b` addresses `r` alone; `r` sends a message of **its own**, numbered
    /// among `r`'s, whose payload carries an envelope naming `b` and `b`'s
    /// number (ADR-0017). So `c`'s observation joins `r`'s action on
    /// `(r, 0)`, and following the envelope from there reaches `b`'s own
    /// action on `(b, 0)`. Nothing claims a sender it does not have.
    fn relayed() -> Vec<Value> {
        let envelope = json!({"Relayed": {"envelope": {
            "from": "b", "seq": 0, "payload": {"Step": 6},
        }}});
        vec![
            header(),
            json!({"type": "control", "agent": "b", "t": 10, "control": "start"}),
            json!({"type": "action", "agent": "b", "t": 20, "seq": 0,
                   "message": {"sender": "b", "recipients": ["r"], "payload": {"Step": 6}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 10, "t_stop": 25,
                   "woken": "queue"}),
            json!({"type": "cycle", "agent": "b", "t_start": 26, "t_stop": 27,
                   "woken": "timeout"}),
            json!({"type": "control", "agent": "b", "t": 95, "control": "stop"}),
            json!({"type": "cycle", "agent": "b", "t_start": 95, "t_stop": 96,
                   "woken": "queue"}),
            json!({"type": "control", "agent": "r", "t": 10, "control": "start"}),
            json!({"type": "cycle", "agent": "r", "t_start": 10, "t_stop": 11,
                   "woken": "queue"}),
            json!({"type": "observation", "agent": "r", "t": 30, "from": "b", "seq": 0,
                   "message": {"sender": "b", "recipients": ["r"], "payload": {"Step": 6}}}),
            // The relay: `r`'s own message, under `r`'s own number zero,
            // carrying `b`'s in the envelope.
            json!({"type": "action", "agent": "r", "t": 35, "seq": 0,
                   "message": {"sender": "r", "recipients": ["c"], "payload": envelope}}),
            json!({"type": "cycle", "agent": "r", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "from": "b", "seq": 0}),
            json!({"type": "control", "agent": "r", "t": 95, "control": "stop"}),
            json!({"type": "cycle", "agent": "r", "t_start": 95, "t_stop": 96,
                   "woken": "queue"}),
            json!({"type": "control", "agent": "c", "t": 10, "control": "start"}),
            json!({"type": "cycle", "agent": "c", "t_start": 10, "t_stop": 11,
                   "woken": "queue"}),
            json!({"type": "observation", "agent": "c", "t": 60, "from": "r", "seq": 0,
                   "message": {"sender": "r", "recipients": ["c"], "payload": envelope}}),
            json!({"type": "cycle", "agent": "c", "t_start": 60, "t_stop": 61, "woken": "queue",
                   "from": "r", "seq": 0}),
            json!({"type": "control", "agent": "c", "t": 95, "control": "stop"}),
            json!({"type": "cycle", "agent": "c", "t_start": 95, "t_stop": 96,
                   "woken": "queue"}),
        ]
    }

    /// The good log with one edit applied to line `index`.
    fn edited(index: usize, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        let mut lines = good();
        edit(&mut lines[index]);
        lines
    }

    /// A log in which `a` reached a `Stop` that had been queued
    /// behind a message: it observed the message, answered it, and popped the
    /// stop in the cycle after (ADR-0009). `b`'s side is here for the join.
    ///
    /// This is the shape the episode produces only when it is abandoning a
    /// run that has already failed; on every other path it holds a stop
    /// back until nothing is in flight, and the stop arrives to an empty
    /// queue.
    fn stopped_behind_a_message() -> Vec<Value> {
        vec![
            header(),
            json!({"type": "control", "agent": "a", "t": 30, "control": "start"}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 31,
                   "woken": "queue"}),
            json!({"type": "observation", "agent": "a", "t": 45, "from": "b", "seq": 0,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "t": 70, "seq": 0,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 45, "t_stop": 72, "woken": "queue",
                   "from": "b", "seq": 0}),
            json!({"type": "control", "agent": "a", "t": 80, "control": "stop"}),
            json!({"type": "cycle", "agent": "a", "t_start": 80, "t_stop": 81,
                   "woken": "queue"}),
            json!({"type": "action", "agent": "b", "t": 40, "seq": 0,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 35, "t_stop": 41,
                   "woken": "timeout"}),
            json!({"type": "observation", "agent": "b", "t": 75, "from": "a", "seq": 0,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}}),
            json!({"type": "cycle", "agent": "b", "t_start": 75, "t_stop": 76, "woken": "queue",
                   "from": "a", "seq": 0}),
        ]
    }

    #[test]
    fn a_stop_reached_behind_a_message_passes() {
        // The answer to the message was sent, not withheld, and `b` observed
        // it: an agent stopped this way did the work queued ahead of the
        // stop, and the log says so on both sides.
        check(&stopped_behind_a_message());
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn a_control_popped_after_its_cycles_start_is_caught() {
        // Nothing a cycle pops is popped later than its start any more.
        // The old exception — a stop that preempted the cycle it landed in
        // — is gone with preemption itself (ADR-0009), so a control stamped
        // later than `t_start` is simply an input that was not popped when
        // it claims to have been.
        let mut lines = stopped_behind_a_message();
        lines[6]["t"] = json!(81);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn an_ordinary_control_popped_late_is_still_caught() {
        check(&edited(1, |line| line["t"] = json!(31)));
    }

    #[test]
    #[should_panic(expected = "end with the cycle that popped its stop")]
    fn a_cycle_after_an_agents_stop_is_caught() {
        // A whole cycle after the one that popped the stop: the agent kept
        // running after it was told to stop.
        let mut lines = stopped_behind_a_message();
        lines.insert(
            8,
            json!({"type": "cycle", "agent": "a", "t_start": 90, "t_stop": 91,
                   "woken": "timeout"}),
        );
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "an agent is stopped once")]
    fn an_agent_stopped_twice_is_caught() {
        // Both stops are popped in the same cycle, so nothing follows the
        // first one but the cycle it belongs to, and it is being told twice
        // that the check has left to catch. The rest of the log is
        // dropped, since a log that ends at the stop is what the
        // other check already asserts.
        let mut lines = stopped_behind_a_message()[..3].to_vec();
        lines[1]["control"] = json!("stop");
        lines.insert(
            2,
            json!({"type": "control", "agent": "a", "t": 30, "control": "stop"}),
        );
        check(&lines);
    }

    /// The good log with a reward for `a`, logged before its stop,
    /// as an environment would have written it.
    fn rewarded() -> Vec<Value> {
        let mut lines = good();
        lines.push(json!({"type": "reward", "agent": "a", "t": 74, "value": 1}));
        lines
    }

    #[test]
    fn a_reward_passes_and_belongs_to_no_cycle() {
        // It carries no sequence number and closes no cycle, and its line
        // sits after the cycle that ended `a`'s records, because the
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
    #[should_panic(expected = "says what it is worth")]
    fn a_reward_without_a_value_is_caught() {
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last].as_object_mut().unwrap().remove("value");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "before the episode's last stop")]
    fn a_reward_logged_after_the_episode_is_caught() {
        // The only stop in `good()` is `a`'s, at 80, so that is where the
        // episode ends and 81 is past it.
        let mut lines = rewarded();
        let last = lines.len() - 1;
        lines[last]["t"] = json!(81);
        check(&lines);
    }

    #[test]
    fn a_reward_after_its_own_agents_stop_is_not_a_bug() {
        // An environment may stop one agent while the rest run on, and
        // pay it when the episode ends (ADR-0012). A reward logged after
        // its own agent's stop but before the run finished is therefore
        // correct, not a fault: the reward is logged rather than sent
        // (ADR-0007), so the agent need not be there to take it.
        //
        // `b` is never stopped in `good()`, so `a` may be paid after its
        // own stop at 80 while the episode is still going.
        let mut lines = good();
        lines.push(json!({"type": "control", "agent": "b", "t": 95, "control": "stop"}));
        lines.push(
            json!({"type": "cycle", "agent": "b", "t_start": 95, "t_stop": 96,
                          "woken": "queue"}),
        );
        lines.push(json!({"type": "reward", "agent": "a", "t": 85, "value": 1}));
        check(&lines);
    }

    #[test]
    fn a_good_log_passes() {
        check(&good());
        let text = good()
            .iter()
            .fold(String::new(), |text, line| text + &line.to_string() + "\n");
        assert_eq!(parse(text.as_bytes()), good());
    }

    #[test]
    #[should_panic(expected = "the first line is the header")]
    fn a_log_without_its_header_is_caught() {
        check(&good()[1..]);
    }

    #[test]
    #[should_panic(expected = "one header per log")]
    fn a_second_header_is_caught() {
        let mut lines = good();
        lines.push(header());
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "anchors the episode to the wall clock")]
    fn a_header_without_a_wall_clock_start_is_caught() {
        let mut lines = good();
        lines[0].as_object_mut().unwrap().remove("start_unix_ns");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "one time, and it is `t`")]
    fn a_record_that_keeps_a_creation_time_is_caught() {
        // No message carries when its sender sent it (ADR-0017), so a
        // record with a second time is a record from the old format.
        check(&edited(2, |line| line["created"] = json!(20)));
    }

    #[test]
    #[should_panic(expected = "popped at its start")]
    fn an_input_popped_at_other_than_its_cycles_start_is_caught() {
        let mut lines = good();
        lines[1]["t"] = json!(29);
        lines[2]["t"] = json!(29);
        lines[4]["t_start"] = json!(30);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "sent within its cycle's window")]
    fn an_action_sent_outside_its_cycle_is_caught() {
        check(&edited(3, |line| line["t"] = json!(51)));
    }

    #[test]
    fn a_timeout_cycle_that_also_observed_passes() {
        // `timeout` says the deadline had passed when the cycle began, not
        // that the cycle observed nothing. A deadline that passes while a
        // message is waiting joins that message's cycle, which observes it and
        // calls `handle` like any other, so this is a shape the loop really
        // produces and the checker must accept.
        let mut lines = good();
        lines[11]["woken"] = json!("timeout");
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
            3,
            json!({"type": "observation", "agent": "a", "t": 30, "from": "b", "seq": 1,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 8}}}),
        );
        // And `b` really sent it, so the join is not what catches this.
        lines.push(json!({"type": "action", "agent": "b", "t": 21, "seq": 1,
               "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 8}}}));
        lines.push(
            json!({"type": "cycle", "agent": "b", "t_start": 21, "t_stop": 22,
                          "woken": "timeout"}),
        );
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "woken by the queue popped something")]
    fn a_queue_cycle_that_popped_nothing_is_caught() {
        check(&edited(5, |line| line["woken"] = json!("queue")));
    }

    #[test]
    #[should_panic(expected = "woken by the queue or the timeout")]
    fn a_cycle_woken_by_something_else_is_caught() {
        check(&edited(5, |line| line["woken"] = json!("thinking")));
    }

    #[test]
    #[should_panic(expected = "numbered contiguously from zero")]
    fn a_gap_in_a_senders_sequence_numbers_is_caught() {
        // Two messages of one sender carry consecutive numbers, so a jump is
        // a missing message. `b`'s only action is numbered zero.
        let mut lines = good();
        lines[8]["seq"] = json!(1);
        lines[2]["seq"] = json!(1);
        lines[4]["seq"] = json!(1);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "observed only by its recipients")]
    fn a_message_delivered_to_a_non_recipient_is_caught() {
        check(&edited(2, |line| {
            line["message"]["recipients"] = json!(["c"]);
        }));
    }

    #[test]
    #[should_panic(expected = "sender among its recipients")]
    fn a_loopback_is_caught() {
        check(&edited(3, |line| {
            line["message"]["recipients"] = json!(["a", "b"]);
        }));
    }

    #[test]
    #[should_panic(expected = "every record belongs to a cycle")]
    fn an_action_recorded_by_somebody_other_than_its_sender_is_caught() {
        // Moving `a`'s action into `c`'s records, where no cycle of `c`'s
        // closes it. `check_record` catches the sender first, so the record
        // is relabeled as `c`'s message too: what is left is a record no
        // cycle of its agent's closes.
        check(&edited(3, |line| {
            line["agent"] = json!("c");
            line["message"]["sender"] = json!("c");
        }));
    }

    #[test]
    #[should_panic(expected = "every record belongs to a cycle")]
    fn a_record_after_the_last_cycle_is_caught() {
        let mut lines = good();
        lines.remove(7);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "joins an action by sender and sequence number")]
    fn an_observation_of_something_nobody_sent_is_caught() {
        let mut lines = good();
        lines[2]["seq"] = json!(1);
        lines[4]["seq"] = json!(1);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "the same message")]
    fn an_observation_that_disagrees_with_its_action_is_caught() {
        check(&edited(2, |line| {
            line["message"]["payload"] = json!({"Step": 99});
        }));
    }

    #[test]
    fn a_relay_is_the_relaying_agents_own_message_and_joins_like_any_other() {
        // The relay is `r`'s message under `r`'s number zero, so `c`'s
        // observation of it joins on `(r, 0)` like every other observation,
        // and the envelope inside it is where `b` is named. Nothing about the
        // log's own join treats a relay specially any more (ADR-0017).
        check(&relayed());
    }

    #[test]
    #[should_panic(expected = "an action is a message of its agent's")]
    fn an_action_claiming_a_sender_other_than_its_agent_is_caught() {
        // The impersonation ADR-0017 removes. `r` writes an action claiming
        // `b` as its sender, which is what a relay used to look like and what
        // nothing may look like now.
        let mut lines = relayed();
        let relay = lines
            .iter()
            .position(|line| line["type"] == "action" && line["agent"] == "r")
            .expect("the fixture has the relay");
        lines[relay]["message"]["sender"] = json!("b");
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "no message has its sender among its recipients")]
    fn a_relay_back_to_the_agent_that_relayed_it_is_caught() {
        // A relay is `r`'s own message, so naming `r` among its recipients
        // would be `r` observing what it had just sent.
        let mut lines = relayed();
        let relay = lines
            .iter()
            .position(|line| line["type"] == "action" && line["agent"] == "r")
            .expect("the fixture has the relay");
        lines[relay]["message"]["recipients"] = json!(["r"]);
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "every recipient of a message observes it")]
    fn a_message_lost_to_a_stopped_agent_before_its_last_observation_is_caught() {
        // Being stopped excuses only what came after. `c` was stopped, and
        // the relay it observed was sent at 35, so a message to `c` sent
        // before that and never observed is a real lost delivery: up to 35
        // the agent was demonstrably taking delivery, and `stopped` is not a
        // blanket excuse.
        //
        // Both sides of that comparison are *send* times, which is why
        // `last_observed` is read off the action a recipient observed rather
        // than off when the recipient popped it: an arrival is later than its
        // send by however long the message waited, and comparing across the
        // two would excuse a message sent inside that window.
        let mut lines = relayed();
        let cycle = lines
            .iter()
            .position(|line| line["type"] == "cycle" && line["agent"] == "b")
            .expect("the fixture has b's first cycle");
        lines.insert(
            cycle,
            json!({"type": "action", "agent": "b", "t": 21, "seq": 1,
                   "message": {"sender": "b", "recipients": ["c"],
                               "payload": {"Step": 9}}}),
        );
        check(&lines);
    }

    #[test]
    #[should_panic(expected = "every recipient of a message observes it")]
    fn a_message_nobody_received_is_caught() {
        let mut lines = good();
        // `a`'s reply never reaches `b`, whose last cycle then popped
        // nothing at all.
        lines.remove(10);
        lines[10].as_object_mut().unwrap().remove("from");
        lines[10].as_object_mut().unwrap().remove("seq");
        lines[10]["woken"] = json!("timeout");
        check(&lines);
    }
}
