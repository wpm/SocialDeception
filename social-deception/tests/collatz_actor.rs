//! Runs the Collatz ring end to end on the
//! actor runtime.
//!
//! Each test constructs an [`Episode`] with a [`Referee`], runs it to the
//! environment's `Stop` with the log going to a file, reads the file back, and
//! asserts on both the outcome and the log. The outcome is checked against
//! Collatz sequences computed here, not by the library, so that the two cannot
//! share a bug. The log is checked against the invariants in
//! [`support::actor`], which know nothing about Collatz.
//!
//! This is the runtime's end-to-end test: every value at every step is known in
//! advance, so any difference between the log and the independently computed
//! sequence is a runtime bug.

mod support;

use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::time::Duration;

use serde_json::Value;
use social_deception::Episode;
use support::TempDir;
use support::collatz_actor::{Collatz, Referee, Step};

/// The environment of every ring here, which starts the agents and stops
/// everybody once every chain has reached 1.
const ENVIRONMENT: &str = "environment";

/// How long an episode here is given. Generous: these rings are microseconds of
/// work, so the limit is only a backstop against a runtime bug that would
/// otherwise hang the test suite.
const LIMIT: Duration = Duration::from_secs(60);

/// A ring of agents: each passes to the next in the list, and the last to the
/// first. Each agent opens a chain from every starting number listed for it.
type Ring<'a> = [(&'a str, &'a [u64])];

/// One step an actor sent: when it sent it, and which step it was.
type Sent = (u64, (u64, u64));

/// The Collatz function, computed here rather than borrowed from the
/// environment so that the check and the thing checked cannot share a bug.
fn successor(n: u64) -> u64 {
    assert!(
        n > 0,
        "the Collatz function is defined on positive integers"
    );
    if n % 2 == 0 { n / 2 } else { 3 * n + 1 }
}

/// The Collatz sequence from `start` down to 1.
fn sequence(start: u64) -> Vec<u64> {
    let mut values = vec![start];
    let mut n = start;
    while n != 1 {
        n = successor(n);
        values.push(n);
    }
    values
}

/// The agent that position `i` of `ring` passes to: the next in the list, or
/// the first after the last.
fn passes_to<'a>(ring: &Ring<'a>, i: usize) -> &'a str {
    ring[(i + 1) % ring.len()].0
}

/// The chains a ring opens, each named by its start, with the sequence each one
/// should pass.
fn expected_chains(ring: &Ring) -> BTreeMap<u64, Vec<u64>> {
    ring.iter()
        .flat_map(|(_, opens)| opens.iter().copied())
        .map(|start| (start, sequence(start)))
        .collect()
}

/// Runs one episode over `ring` and returns the log it wrote, read back from
/// disk.
///
/// The log is checked against the invariants in [`support::actor`] before it is
/// returned, so that a malformed log fails by the name of the invariant it
/// breaks rather than by a lookup that misses in the Collatz checks below.
///
/// **There is no clock here.** The episode starts its own, before the writer
/// and before any actor, which is what makes the shared origin structural
/// rather than something a caller can get wrong (ADR-0017).
fn run(ring: &Ring) -> Vec<Value> {
    let dir = TempDir::new();
    let file = dir.join("collatz.jsonl");
    let environment = Referee::new(
        ENVIRONMENT,
        ring.iter().map(|(name, _)| *name),
        ring.iter().flat_map(|(_, opens)| opens.iter().copied()),
    );
    let mut episode: Episode<i32, Step> =
        Episode::to_file(&file, ENVIRONMENT, environment).unwrap();
    episode = episode.within(LIMIT);
    for (i, (name, opens)) in ring.iter().enumerate() {
        let agent = opens.iter().fold(
            Collatz::new(passes_to(ring, i), ENVIRONMENT),
            |agent, &start| agent.opening(start),
        );
        episode.add(*name, agent).unwrap();
    }
    episode.run().unwrap();
    let lines = support::parse(&fs::read(&file).unwrap());
    support::actor::check(&lines);
    lines
}

/// The step a record's message carries, as the chain's name and the value, or
/// `None` for a `Finished`, a control or a cycle.
fn step(record: &Value) -> Option<(u64, u64)> {
    let step = &record["message"]["payload"]["Pass"];
    Some((step["chain"].as_u64()?, step["value"].as_u64()?))
}

/// The chain a record reports finished, or `None` for anything else.
fn finished(record: &Value) -> Option<u64> {
    record["message"]["payload"]["Finished"]["chain"].as_u64()
}

/// The records of `lines` of the given type.
fn of<'a>(lines: &'a [Value], kind: &'a str) -> impl Iterator<Item = &'a Value> {
    lines.iter().filter(move |line| line["type"] == kind)
}

/// The chains reported finished, in ascending order, with the recipient of
/// every report checked to be the environment.
fn reports(lines: &[Value]) -> Vec<u64> {
    let mut chains: Vec<u64> = of(lines, "action")
        .filter_map(|line| {
            let chain = finished(line)?;
            assert_eq!(
                line["message"]["recipients"].as_array().unwrap().as_slice(),
                [Value::from(ENVIRONMENT)],
                "a report goes to the environment and nobody else: {line}"
            );
            Some(chain)
        })
        .collect();
    chains.sort_unstable();
    chains
}

/// Every chain in `lines`, by name, each in the order it was passed.
///
/// A chain is followed hop by hop, from the observation of a step to the action
/// that carries the next one, and the two are tied by the **cycle** that
/// bracketed them: a cycle is one handler call, so the action a step called for
/// is the one its own cycle sent. The order recovered is therefore causal and
/// owes nothing to the file's order, which under this runtime is two threads
/// per actor interleaved as the channel delivered them.
fn chains(lines: &[Value]) -> BTreeMap<u64, Vec<u64>> {
    // Each cycle by the observation it names, which is what an action is
    // attributed to.
    let mut calls: HashMap<(&str, &str, u64), (u64, u64, &str)> = HashMap::new();
    for cycle in support::actor::cycles(lines) {
        let Some(from) = cycle["from"].as_str() else {
            continue;
        };
        calls.insert(
            (support::agent(cycle), from, support::seq(cycle)),
            (
                support::time(cycle, "t_start"),
                support::time(cycle, "t_stop"),
                support::agent(cycle),
            ),
        );
    }
    // Where each step arrived: the actor that observed it and the window of
    // the call it opened.
    let mut arrivals: HashMap<(u64, u64), (&str, u64, u64)> = HashMap::new();
    for record in of(lines, "observation") {
        let Some(at) = step(record) else { continue };
        let who = support::agent(record);
        let from = record["from"]
            .as_str()
            .expect("an observation names its sender");
        let (t_start, t_stop, _) = calls
            .get(&(who, from, support::seq(record)))
            .unwrap_or_else(|| panic!("every observation opened a cycle: {record}"));
        assert!(
            arrivals.insert(at, (who, *t_start, *t_stop)).is_none(),
            "a chain never carries the same value twice: {at:?}"
        );
    }
    // The steps each actor sent, with the instant of each, so that a step can
    // be attributed to the call whose window holds it.
    let mut sent: HashMap<&str, Vec<Sent>> = HashMap::new();
    for record in of(lines, "action") {
        if let Some(at) = step(record) {
            sent.entry(support::agent(record))
                .or_default()
                .push((support::time(record, "t"), at));
        }
    }
    // What each call sent of a given chain.
    let outputs = |who: &str, window: (u64, u64), chain: u64| -> Vec<u64> {
        sent.get(who)
            .into_iter()
            .flatten()
            .filter(|(t, _)| window.0 <= *t && *t <= window.1)
            .filter(|(_, (name, _))| *name == chain)
            .map(|(_, (_, value))| *value)
            .collect()
    };

    let names: Vec<u64> = arrivals
        .keys()
        .filter(|(chain, value)| chain == value)
        .map(|(chain, _)| *chain)
        .collect();
    let mut chains = BTreeMap::new();
    for chain in names {
        let mut values = vec![chain];
        loop {
            let current = *values.last().unwrap();
            let (who, t_start, t_stop) = arrivals[&(chain, current)];
            let next = outputs(who, (t_start, t_stop), chain);
            if current == 1 {
                assert!(
                    next.is_empty(),
                    "nothing of chain {chain} is sent after a 1, but {who} sent {next:?}"
                );
                break;
            }
            assert_eq!(
                next.len(),
                1,
                "one step of chain {chain} in, one out, but {who} sent {next:?}"
            );
            values.push(next[0]);
        }
        chains.insert(chain, values);
    }
    chains
}

/// Every step sent by anyone, in ascending order.
fn steps_sent(lines: &[Value]) -> Vec<(u64, u64)> {
    let mut steps: Vec<(u64, u64)> = of(lines, "action").filter_map(step).collect();
    steps.sort_unstable();
    steps
}

/// Asserts everything the outcome of an episode over `ring` must satisfy: every
/// chain the ring opened was passed in exactly its Collatz sequence, every step
/// went to the next agent in the ring, and nothing was sent that belongs to no
/// chain.
fn check_outcome(lines: &[Value], ring: &Ring) {
    let expected = expected_chains(ring);
    assert_eq!(chains(lines), expected);
    let next: HashMap<&str, &str> = ring
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, passes_to(ring, i)))
        .collect();
    for line in of(lines, "action").filter(|line| step(line).is_some()) {
        let recipients = line["message"]["recipients"]
            .as_array()
            .expect("a message lists its recipients");
        assert_eq!(
            recipients.as_slice(),
            [Value::from(next[support::agent(line)])],
            "a step goes to the next agent in the ring: {line}"
        );
    }
    // Exactly the chains the ring opened were reported finished, one report
    // each: that is what tells the environment to stop everybody, so it is what
    // the episode ending at all depends on.
    let mut opened: Vec<u64> = expected.keys().copied().collect();
    opened.sort_unstable();
    assert_eq!(reports(lines), opened);
    let mut steps: Vec<(u64, u64)> = expected
        .iter()
        .flat_map(|(&chain, values)| values.iter().map(move |&value| (chain, value)))
        .collect();
    steps.sort_unstable();
    assert_eq!(steps_sent(lines), steps);
}

#[test]
fn a_chain_passed_around_a_ring_is_the_collatz_sequence() {
    let ring: &Ring = &[("a", &[27]), ("b", &[]), ("c", &[])];
    let lines = run(ring);
    check_outcome(&lines, ring);
    // `check_outcome` trusts `sequence`; one well-known length pins that down.
    assert_eq!(
        chains(&lines)[&27].len(),
        112,
        "27 takes 111 steps to reach 1"
    );
}

#[test]
fn a_chain_from_one_is_over_at_once() {
    let ring: &Ring = &[("a", &[1]), ("b", &[])];
    let lines = run(ring);
    assert_eq!(chains(&lines)[&1], [1]);
    check_outcome(&lines, ring);
}

#[test]
fn several_chains_at_once_are_each_the_collatz_sequence() {
    // 6 and 7 merge at 10 and every chain here ends 4, 2, 1, so a step's value
    // alone does not say which chain it belongs to; the chain's name on each
    // step is what lets every chain be followed separately.
    let ring: &Ring = &[("a", &[6, 7]), ("b", &[27]), ("c", &[]), ("d", &[97, 871])];
    let lines = run(ring);
    check_outcome(&lines, ring);
}

#[test]
fn chains_that_share_a_value_stay_apart() {
    // 3 is one hop from 6, so from the second hop on the two chains carry the
    // same values, in step, around a ring of two.
    let ring: &Ring = &[("a", &[6, 3]), ("b", &[])];
    let lines = run(ring);
    check_outcome(&lines, ring);
}

#[test]
fn a_ring_that_rewards_nobody_logs_no_rewards() {
    // A reward is a game's verdict on a player, and a ring passing numbers
    // around has nothing to win. The runtime offers the environment an
    // `Effect::Reward` and this one never returns any, so no reward record
    // exists: nothing writes one on an environment's behalf, and an agent has
    // no way to ask for one.
    let ring: &Ring = &[("a", &[6, 7]), ("b", &[]), ("c", &[3])];
    let lines = run(ring);
    assert_eq!(
        of(&lines, "reward").count(),
        0,
        "the Collatz ring rewards nobody"
    );
    check_outcome(&lines, ring);
}

#[test]
fn a_ring_that_opens_nothing_is_started_and_stopped_and_says_nothing() {
    let ring: &Ring = &[("a", &[]), ("b", &[])];
    let lines = run(ring);
    assert!(chains(&lines).is_empty());
    assert_eq!(
        of(&lines, "control").count(),
        6,
        "a start and a stop for each of two agents and the environment"
    );
    assert_eq!(
        of(&lines, "action").count(),
        0,
        "with no chain to pass, nobody has anything to say"
    );
    check_outcome(&lines, ring);
}

#[test]
fn every_actors_start_and_the_log_share_one_origin() {
    // The shared origin is enforced by construction — the episode starts the
    // clock, so there is no second one to hand anybody — and what that buys is
    // this: every offset in the log is measured from the one origin, so the
    // header's anchor is the moment offset zero refers to and no record sits
    // before it. A second origin would show up as a record whose offset was
    // clamped to zero where a real one belongs.
    let ring: &Ring = &[("a", &[27]), ("b", &[])];
    let lines = run(ring);
    assert!(
        lines[0]["start_unix_ns"].as_u64().unwrap() > 0,
        "the header anchors the episode: {}",
        lines[0]
    );
    // Every offset is strictly after the origin, which is what one origin
    // means here. `Clock::offset` saturates an instant before the origin to
    // zero rather than reporting a time that ran backwards, so an actor
    // measuring from a *second*, later origin is exactly what a zero offset
    // would be evidence of: the episode captures its clock before it builds a
    // single channel, so nothing any actor does can land on it.
    //
    // What is deliberately *not* asserted is which record comes first. Three
    // actors' perception threads write to one writer, and which of them the
    // operating system runs first is not the log's business — even the
    // environment's own `Start`, which the episode sends before anything else,
    // is *recorded* by a thread that may be descheduled. Line order carries no
    // meaning (ADR-0017), and neither does the minimum.
    for line in &lines[1..] {
        let times: &[&str] = if line["type"] == "cycle" {
            &["t_start", "t_stop"]
        } else {
            &["t"]
        };
        for key in times {
            assert!(
                support::time(line, key) > 0,
                "every offset is measured from the one origin the episode \
                 captured before anything could happen: {line}"
            );
        }
    }
}
