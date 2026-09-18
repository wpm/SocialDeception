//! Runs the Collatz environment end to end.
//!
//! Each test constructs an episode, runs it to quiescence with the trajectory
//! going to a file, reads the file back, and asserts on both the outcome and
//! the log. The outcome is checked against Collatz sequences computed here,
//! not by the library, so that the two cannot share a bug. The log is
//! checked against the invariants in [`support`], which know nothing about
//! Collatz.

mod support;

use std::collections::HashMap;
use std::fs;
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;
use social_deception::collatz::Collatz;
use social_deception::{Episode, Writer};

/// A ring of agents: each passes to the next in the list, and the last to
/// the first. Each agent opens a chain from every starting number listed for
/// it.
type Ring<'a> = [(&'a str, &'a [u64])];

/// The Collatz sequence from `start` down to 1.
fn sequence(start: u64) -> Vec<u64> {
    let mut values = vec![start];
    let mut n = start;
    while n != 1 {
        n = if n % 2 == 0 { n / 2 } else { 3 * n + 1 };
        values.push(n);
    }
    values
}

/// Runs one episode over `ring` and returns the trajectory it wrote, read
/// back from disk.
fn run(ring: &Ring) -> Vec<Value> {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "social-deception-collatz-{}-{}.jsonl",
        process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    let (records, writer) = Writer::create(&path).unwrap();
    let mut episode = Episode::new(records);
    for (i, (name, opens)) in ring.iter().enumerate() {
        let to = ring[(i + 1) % ring.len()].0;
        let agent = opens
            .iter()
            .fold(Collatz::new(to), |agent, &start| agent.opening(start));
        episode.add(*name, agent).unwrap();
    }
    episode.run().unwrap();
    writer.join().unwrap();
    let bytes = fs::read(&path).unwrap();
    fs::remove_file(&path).unwrap();
    support::parse(&bytes)
}

/// The value a message record carries, or `None` for a control or a think.
fn value(record: &Value) -> Option<u64> {
    record["event"]["payload"]["Step"].as_u64()
}

/// The event records of `lines`, by agent and sequence number.
fn records(lines: &[Value]) -> HashMap<(&str, u64), &Value> {
    lines
        .iter()
        .filter(|line| line["type"] == "event")
        .map(|line| ((support::agent(line), support::seq(line)), line))
        .collect()
}

fn cycles(lines: &[Value]) -> impl Iterator<Item = &Value> {
    lines.iter().filter(|line| line["type"] == "cycle")
}

/// Asserts that every cycle sent exactly what the rule says its inputs call
/// for, in order: the agent's own chains on `Start`, the next value for each
/// value above 1, and nothing for a 1, a stop or a think.
fn every_hop_follows_the_rule(lines: &[Value], ring: &Ring) {
    let opens: HashMap<&str, &[u64]> = ring.iter().copied().collect();
    let records = records(lines);
    for cycle in cycles(lines) {
        let agent = support::agent(cycle);
        let record = |seq| records[&(agent, seq)];
        let expected: Vec<u64> = support::seqs(cycle, "inputs")
            .map(record)
            .flat_map(
                |input| match (input["event"]["control"].as_str(), value(input)) {
                    (Some("start"), _) => opens[agent].to_vec(),
                    (_, Some(1) | None) => Vec::new(),
                    (_, Some(n)) => vec![if n % 2 == 0 { n / 2 } else { 3 * n + 1 }],
                },
            )
            .collect();
        let sent: Vec<u64> = support::seqs(cycle, "outputs")
            .map(record)
            .map(|output| value(output).expect("an agent only ever sends a step"))
            .collect();
        assert_eq!(sent, expected, "the outputs of {cycle}");
    }
}

/// The one chain in `lines`, in the order it was passed, starting from the
/// chain `opener` opened on `Start`.
///
/// Each hop is followed from the record of the value's arrival to the
/// output of the cycle that handled it, so the order recovered is causal and
/// owes nothing to timestamps.
fn chain(lines: &[Value], opener: &str) -> Vec<u64> {
    let records = records(lines);
    let mut arrivals: HashMap<u64, (&str, u64)> = HashMap::new();
    for (&(agent, seq), record) in &records {
        if let Some(n) = value(record).filter(|_| !record["arrived"].is_null()) {
            assert!(
                arrivals.insert(n, (agent, seq)).is_none(),
                "a single chain never carries the same value twice: {n}"
            );
        }
    }
    let handled_by: HashMap<(&str, u64), &Value> = cycles(lines)
        .flat_map(|cycle| {
            support::seqs(cycle, "inputs").map(move |seq| ((support::agent(cycle), seq), cycle))
        })
        .collect();
    let outputs = |cycle: &Value| -> Vec<u64> {
        support::seqs(cycle, "outputs")
            .map(|seq| value(records[&(support::agent(cycle), seq)]).unwrap())
            .collect()
    };

    let opening = cycles(lines)
        .filter(|cycle| support::agent(cycle) == opener)
        .find(|cycle| {
            support::seqs(cycle, "inputs")
                .any(|seq| records[&(opener, seq)]["event"]["control"] == "start")
        })
        .expect("the opener handled a start");
    let mut values = outputs(opening);
    assert_eq!(values.len(), 1, "the opener opened one chain on start");
    loop {
        let current = *values.last().unwrap();
        let handled = handled_by[&arrivals[&current]];
        let next = outputs(handled);
        if current == 1 {
            assert!(next.is_empty(), "nothing is sent after a 1: {handled}");
            return values;
        }
        assert_eq!(next.len(), 1, "one value in, one value out: {handled}");
        values.push(next[0]);
    }
}

/// Every value sent by anyone, in ascending order.
fn values_sent(lines: &[Value]) -> Vec<u64> {
    let mut values: Vec<u64> = lines
        .iter()
        .filter(|line| !line["sent"].is_null())
        .map(|line| value(line).expect("an agent only ever sends a step"))
        .collect();
    values.sort_unstable();
    values
}

#[test]
fn a_chain_passed_around_a_ring_is_the_collatz_sequence() {
    let ring: &Ring = &[("a", &[27]), ("b", &[]), ("c", &[])];
    let lines = run(ring);
    let chain = chain(&lines, "a");
    assert_eq!(chain, sequence(27));
    assert_eq!(chain.len(), 112, "27 takes 111 steps to reach 1");
    every_hop_follows_the_rule(&lines, ring);
    support::check(&lines);
}

#[test]
fn a_chain_from_one_is_over_at_once() {
    let ring: &Ring = &[("a", &[1]), ("b", &[])];
    let lines = run(ring);
    assert_eq!(chain(&lines, "a"), [1]);
    every_hop_follows_the_rule(&lines, ring);
    support::check(&lines);
}

#[test]
fn several_chains_at_once_all_reach_one() {
    // 6 and 7 merge at 10, so a chain cannot be told from another by its
    // values alone; what can be checked is that every hop is right and that
    // the values sent are exactly the union of the chains, with every 1
    // reached exactly once.
    let ring: &Ring = &[("a", &[6, 7]), ("b", &[27]), ("c", &[]), ("d", &[97, 871])];
    let lines = run(ring);
    every_hop_follows_the_rule(&lines, ring);
    let mut expected: Vec<u64> = ring
        .iter()
        .flat_map(|(_, opens)| opens.iter().copied())
        .flat_map(sequence)
        .collect();
    expected.sort_unstable();
    assert_eq!(values_sent(&lines), expected);
    support::check(&lines);
}

#[test]
fn a_ring_that_opens_nothing_goes_quiescent_at_once() {
    let ring: &Ring = &[("a", &[]), ("b", &[])];
    let lines = run(ring);
    assert!(values_sent(&lines).is_empty());
    let controls: Vec<&Value> = lines
        .iter()
        .filter(|line| line["event"]["kind"] == "control")
        .collect();
    assert_eq!(
        controls.len(),
        4,
        "a start and a stop for each of two agents"
    );
    every_hop_follows_the_rule(&lines, ring);
    support::check(&lines);
}
