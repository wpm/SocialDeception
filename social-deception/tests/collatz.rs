//! Runs the Collatz environment end to end.
//!
//! Each test constructs an episode, runs it to quiescence with the trajectory
//! going to a file, reads the file back, and asserts on both the outcome and
//! the log. The outcome is checked against Collatz sequences computed here,
//! not by the library, so that the two cannot share a bug. The log is
//! checked against the invariants in [`support`], which know nothing about
//! Collatz.

mod support;

use std::collections::{BTreeMap, HashMap};
use std::fs;

use serde_json::Value;
use social_deception::{Episode, Writer};
use support::TempDir;
use support::collatz::Collatz;

/// A ring of agents: each passes to the next in the list, and the last to
/// the first. Each agent opens a chain from every starting number listed for
/// it.
type Ring<'a> = [(&'a str, &'a [u64])];

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

/// The agent that position `i` of `ring` passes to: the next in the list,
/// or the first after the last.
fn passes_to<'a>(ring: &Ring<'a>, i: usize) -> &'a str {
    ring[(i + 1) % ring.len()].0
}

/// The chains a ring opens, each named by its start, with the sequence each
/// one should pass.
fn expected_chains(ring: &Ring) -> BTreeMap<u64, Vec<u64>> {
    ring.iter()
        .flat_map(|(_, opens)| opens.iter().copied())
        .map(|start| (start, sequence(start)))
        .collect()
}

/// Runs one episode over `ring` and returns the trajectory it wrote, read
/// back from disk.
///
/// The log is checked against the invariants in [`support`] before it is
/// returned, so that a malformed log fails by the name of the invariant it
/// breaks rather than by a lookup that misses in the Collatz checks below.
fn run(ring: &Ring) -> Vec<Value> {
    let dir = TempDir::new();
    let file = dir.join("collatz.jsonl");
    let (records, writer) = Writer::create(&file).unwrap();
    let mut episode = Episode::new(records);
    for (i, (name, opens)) in ring.iter().enumerate() {
        let agent = opens
            .iter()
            .fold(Collatz::new(passes_to(ring, i)), |agent, &start| {
                agent.opening(start)
            });
        episode.add(*name, agent).unwrap();
    }
    episode.run().unwrap();
    writer.join().unwrap();
    let lines = support::parse(&fs::read(&file).unwrap());
    support::check(&lines);
    lines
}

/// The step a record's event carries, as the chain's name and the value, or
/// `None` for a control or a cycle.
fn step(record: &Value) -> Option<(u64, u64)> {
    let step = &record["event"]["payload"]["Step"];
    Some((step["chain"].as_u64()?, step["value"].as_u64()?))
}

/// The records of `lines` of the given type.
fn of<'a>(lines: &'a [Value], kind: &'a str) -> impl Iterator<Item = &'a Value> {
    lines.iter().filter(move |line| line["type"] == kind)
}

fn cycles(lines: &[Value]) -> impl Iterator<Item = &Value> {
    of(lines, "cycle")
}

/// The steps a cycle sent, in order.
fn outputs(cycle: &Value, records: &HashMap<(&str, u64), &Value>) -> Vec<(u64, u64)> {
    support::seqs(cycle, "outputs")
        .map(|seq| records[&(support::agent(cycle), seq)])
        .map(|output| step(output).expect("an agent only ever sends a step"))
        .collect()
}

/// Asserts that every cycle sent exactly what the rule says its inputs call
/// for, in order: the agent's own chains when it pops its start, each at its
/// starting value; the next value of the same chain for each value above 1;
/// and nothing for a 1 or a stop.
///
/// The chains an agent opens are sent in the cycle that popped its start,
/// because that is the cycle in which the loop calls the start hook, and
/// they come first in it, ahead of anything that cycle also observed.
fn every_hop_follows_the_rule(lines: &[Value], ring: &Ring) {
    let opens: HashMap<&str, &[u64]> = ring.iter().copied().collect();
    let records = support::records(lines);
    for cycle in cycles(lines) {
        let agent = support::agent(cycle);
        let inputs: Vec<&Value> = support::seqs(cycle, "inputs")
            .map(|seq| records[&(agent, seq)])
            .collect();
        let opened: Vec<(u64, u64)> = if inputs.iter().any(|input| input["control"] == "start") {
            opens[agent].iter().map(|&s| (s, s)).collect()
        } else {
            Vec::new()
        };
        let expected: Vec<(u64, u64)> = opened
            .into_iter()
            .chain(inputs.iter().flat_map(|input| match step(input) {
                Some((_, 1)) | None => Vec::new(),
                Some((chain, n)) => vec![(chain, successor(n))],
            }))
            .collect();
        assert_eq!(outputs(cycle, &records), expected, "the outputs of {cycle}");
    }
}

/// Every chain in `lines`, by name, each in the order it was passed.
///
/// A chain is followed hop by hop from the observation of a step to the
/// output of the cycle that handled it, so the order recovered is causal
/// and owes nothing to timestamps. A chain begins at the step that carries
/// its name as its value, which is what its opener sends when it starts.
fn chains(lines: &[Value]) -> BTreeMap<u64, Vec<u64>> {
    let records = support::records(lines);
    let mut arrivals: HashMap<(u64, u64), (&str, u64)> = HashMap::new();
    for (&(agent, seq), record) in &records {
        if record["type"] != "observation" {
            continue;
        }
        if let Some(step) = step(record) {
            assert!(
                arrivals.insert(step, (agent, seq)).is_none(),
                "a chain never carries the same value twice: {step:?}"
            );
        }
    }
    let handled_by: HashMap<(&str, u64), &Value> = cycles(lines)
        .flat_map(|cycle| {
            support::seqs(cycle, "inputs").map(move |seq| ((support::agent(cycle), seq), cycle))
        })
        .collect();
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
            let handled = handled_by[&arrivals[&(chain, current)]];
            let next: Vec<u64> = outputs(handled, &records)
                .into_iter()
                .filter(|(name, _)| *name == chain)
                .map(|(_, value)| value)
                .collect();
            if current == 1 {
                assert!(next.is_empty(), "nothing is sent after a 1: {handled}");
                break;
            }
            assert_eq!(next.len(), 1, "one step of a chain in, one out: {handled}");
            values.push(next[0]);
        }
        chains.insert(chain, values);
    }
    chains
}

/// Every step sent by anyone, in ascending order.
fn steps_sent(lines: &[Value]) -> Vec<(u64, u64)> {
    let mut steps: Vec<(u64, u64)> = of(lines, "action")
        .map(|line| step(line).expect("an agent only ever sends a step"))
        .collect();
    steps.sort_unstable();
    steps
}

/// Asserts everything the outcome of an episode over `ring` must satisfy:
/// every chain the ring opened was passed in exactly its Collatz sequence,
/// every hop obeyed the rule, every step went to the next agent in the
/// ring, and nothing was sent that belongs to no chain.
fn check_outcome(lines: &[Value], ring: &Ring) {
    let expected = expected_chains(ring);
    assert_eq!(chains(lines), expected);
    every_hop_follows_the_rule(lines, ring);
    let next: HashMap<&str, &str> = ring
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, passes_to(ring, i)))
        .collect();
    for line in of(lines, "action") {
        let recipients = line["event"]["recipients"]
            .as_array()
            .expect("an event lists its recipients");
        assert_eq!(
            recipients.as_slice(),
            [Value::from(next[support::agent(line)])],
            "a step goes to the next agent in the ring: {line}"
        );
    }
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
    // 6 and 7 merge at 10 and every chain here ends 4, 2, 1, so a step's
    // value alone does not say which chain it belongs to; the chain's name
    // on each step is what lets every chain be followed separately.
    let ring: &Ring = &[("a", &[6, 7]), ("b", &[27]), ("c", &[]), ("d", &[97, 871])];
    let lines = run(ring);
    check_outcome(&lines, ring);
}

#[test]
fn chains_that_share_a_value_stay_apart() {
    // 3 is one hop from 6, so from the second hop on the two chains carry
    // the same values, in step, around a ring of two.
    let ring: &Ring = &[("a", &[6, 3]), ("b", &[])];
    let lines = run(ring);
    check_outcome(&lines, ring);
}

/// A hand-written trajectory of a ring of two in which `a` opens 4 and 2,
/// `b` drains both opening steps in one cycle, and `a` drains both replies in
/// one cycle. The runtime makes such cycles likely but not certain, so the
/// case is pinned down here rather than hoped for in a live episode.
fn mixed_drains() -> Vec<Value> {
    let control = |agent: &str, seq: u64, created: u64, received: u64, control: &str| {
        serde_json::json!({"type": "control", "agent": agent, "seq": seq, "created": created,
                           "received": received, "control": control})
    };
    // An action and the observation of it carry the same `created`: they are
    // the same event from its two ends, and that is what joins them.
    let action = |agent: &str, seq: u64, created: u64, chain: u64, value: u64| {
        let other = if agent == "a" { "b" } else { "a" };
        serde_json::json!({"type": "action", "agent": agent, "seq": seq, "created": created,
                           "event": {"sender": agent, "recipients": [other],
                                     "payload": {"Step": {"chain": chain, "value": value}}}})
    };
    let observation =
        |agent: &str, seq: u64, created: u64, received: u64, chain: u64, value: u64| {
            let other = if agent == "a" { "b" } else { "a" };
            serde_json::json!({"type": "observation", "agent": agent, "seq": seq,
                               "created": created, "received": received,
                               "event": {"sender": other, "recipients": [agent],
                                         "payload": {"Step": {"chain": chain, "value": value}}}})
        };
    let cycle = |agent: &str, t_start: u64, t_stop: u64, inputs: &[u64], outputs: &[u64]| {
        serde_json::json!({"type": "cycle", "agent": agent, "t_start": t_start, "t_stop": t_stop,
                           "woken": "queue", "inputs": inputs, "outputs": outputs})
    };
    vec![
        control("a", 0, 10, 15, "start"),
        action("a", 1, 20, 4, 4),
        action("a", 2, 21, 2, 2),
        cycle("a", 15, 25, &[0], &[1, 2]),
        control("b", 0, 10, 30, "start"),
        observation("b", 1, 20, 30, 4, 4),
        observation("b", 2, 21, 30, 2, 2),
        action("b", 3, 40, 4, 2),
        action("b", 4, 41, 2, 1),
        cycle("b", 30, 45, &[0, 1, 2], &[3, 4]),
        observation("a", 3, 40, 50, 4, 2),
        observation("a", 4, 41, 50, 2, 1),
        action("a", 5, 60, 4, 1),
        cycle("a", 50, 65, &[3, 4], &[5]),
        observation("b", 5, 60, 70, 4, 1),
        cycle("b", 70, 75, &[5], &[]),
        control("a", 6, 80, 85, "stop"),
        cycle("a", 85, 86, &[6], &[]),
        control("b", 6, 80, 85, "stop"),
        cycle("b", 85, 86, &[6], &[]),
    ]
}

#[test]
fn chains_are_told_apart_within_one_drain() {
    let lines = mixed_drains();
    support::check(&lines);
    let ring: &Ring = &[("a", &[4, 2]), ("b", &[])];
    check_outcome(&lines, ring);
    let chains = chains(&lines);
    assert_eq!(chains[&4], [4, 2, 1]);
    assert_eq!(chains[&2], [2, 1]);
}

#[test]
#[should_panic(expected = "the outputs of")]
fn a_step_sent_on_the_wrong_chain_is_caught() {
    let mut lines = mixed_drains();
    // b's reply to chain 4's step is filed under chain 2.
    lines[7]["event"]["payload"]["Step"]["chain"] = serde_json::json!(2);
    let ring: &Ring = &[("a", &[4, 2]), ("b", &[])];
    every_hop_follows_the_rule(&lines, ring);
}

#[test]
#[should_panic(expected = "next agent in the ring")]
fn a_step_sent_to_the_wrong_agent_is_caught() {
    let mut lines = mixed_drains();
    // a's opening step for chain 4 is addressed to c, who is not in the ring.
    lines[1]["event"]["recipients"] = serde_json::json!(["c"]);
    let ring: &Ring = &[("a", &[4, 2]), ("b", &[])];
    check_outcome(&lines, ring);
}

#[test]
fn a_ring_that_opens_nothing_goes_quiescent_at_once() {
    let ring: &Ring = &[("a", &[]), ("b", &[])];
    let lines = run(ring);
    assert!(chains(&lines).is_empty());
    let controls: Vec<&Value> = of(&lines, "control").collect();
    assert_eq!(
        controls.len(),
        4,
        "a start and a stop for each of two agents"
    );
    check_outcome(&lines, ring);
}
