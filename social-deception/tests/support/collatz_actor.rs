//! The Collatz ring on the actor runtime: actors
//! pass a Collatz chain around a ring.
//!
//! The Collatz function takes a positive integer `n` to `n / 2` if `n` is even
//! and to `3n + 1` if it is odd. Iterated from any starting number anyone has
//! tried, it reaches 1. A chain is the sequence of values from a starting
//! number down to 1.
//!
//! Each agent in the ring passes to one other agent. On observing a value it
//! computes the next one and sends it on; on observing 1 it tells the
//! [`Referee`] that the chain is [`Finished`](Step) and sends nothing else. An
//! agent may also open chains of its own when it is started, which is what its
//! `start` hook is for.
//!
//! The ring is purely reactive, so nothing in it could end an episode. The
//! environment is what does: it starts every agent, holds the multiset of
//! chains that will be opened, strikes one off for each `Finished`, and
//! **stops every actor, itself included**, once none is left. A multiset and
//! not a set, because two chains opened from the same number share a name and
//! cannot be told apart; what can be counted is how many are outstanding.
//!
//! A chain is named by its starting number, and every step carries the name of
//! the chain it belongs to. Chains from different starting numbers merge, so
//! without the name a value in the log could not be attributed to a chain once
//! it had.
//!
//! Every value at every step is known in advance, so any difference between
//! the log an episode writes and the sequence computed independently is a bug
//! in the runtime, not a model being unpredictable. That is what this ring is
//! for: it is a second payload type beside Werewolf's, and a second
//! environment beside the moderator, which is what keeps the runtime honest
//! about being generic, and the runtime's end-to-end test.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use social_deception::{
    Action, ActorId, Clock, Control, Effect, Observation, Policy, Step as Stepping,
};

/// The Collatz function: `n / 2` for even `n`, `3n + 1` for odd `n`.
///
/// # Panics
///
/// If `n` is 0, which is not in the function's domain and would map to
/// itself forever, or if `3n + 1` does not fit in a `u64`.
pub fn next(n: u64) -> u64 {
    assert!(
        n > 0,
        "the Collatz function is defined on positive integers"
    );
    if n % 2 == 0 {
        n / 2
    } else {
        n.checked_mul(3)
            .and_then(|m| m.checked_add(1))
            .expect("3n + 1 does not fit in a u64")
    }
}

/// What Collatz actors say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum Step {
    /// One step of a chain, from one agent of the ring to the next. The
    /// recipient takes the next step.
    Pass {
        /// The chain this step belongs to: its starting number.
        chain: u64,
        /// The chain's current value.
        value: u64,
    },
    /// A chain has reached 1, from the agent that observed it to the
    /// environment. Nothing else is ever addressed to the environment.
    Finished {
        /// The chain that is over: its starting number.
        chain: u64,
    },
}

/// One agent of the Collatz ring.
///
/// It passes every value it receives one step on to a single other agent,
/// tells the environment when a chain reaches 1, and opens chains of its own
/// when the episode starts. It keeps no state between calls: the chain's name
/// and value travel with the message, which makes it a reactive policy,
/// π(a | o), of the plainest sort.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collatz {
    to: ActorId,
    environment: ActorId,
    opens: Vec<u64>,
}

impl Collatz {
    /// An agent that passes every chain it receives on to `to`, reports a
    /// finished chain to `environment`, and opens none of its own.
    pub fn new(to: impl Into<ActorId>, environment: impl Into<ActorId>) -> Self {
        Self {
            to: to.into(),
            environment: environment.into(),
            opens: Vec::new(),
        }
    }

    /// The same agent, also opening a chain from `start` when the episode
    /// starts. An agent opens its chains in the order this was called.
    ///
    /// # Panics
    ///
    /// If `start` is 0, which is not in the Collatz function's domain.
    #[must_use]
    pub fn opening(mut self, start: u64) -> Self {
        assert!(start > 0, "a Collatz chain starts at a positive integer");
        self.opens.push(start);
        self
    }

    /// The agent this one passes to.
    #[must_use]
    pub const fn to(&self) -> &ActorId {
        &self.to
    }

    /// The starting numbers of the chains this agent opens, in order.
    #[must_use]
    pub fn opens(&self) -> &[u64] {
        &self.opens
    }

    /// One step, addressed to the agent this one passes to.
    fn pass(&self, step: Step) -> Action<Step> {
        Action::to([self.to.clone()], step)
    }
}

impl Policy<Step> for Collatz {
    /// Opens this agent's own chains, each at its starting value.
    fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<Step>> {
        self.opens
            .iter()
            .map(|&start| {
                self.pass(Step::Pass {
                    chain: start,
                    value: start,
                })
            })
            .collect::<Vec<_>>()
    }

    /// What one observation calls for sending. It depends only on the step
    /// observed.
    ///
    /// # Panics
    ///
    /// If the environment sends an agent a `Finished`, which is a message that
    /// only ever travels the other way.
    fn policy(&mut self, observation: Observation<Step>) -> impl IntoIterator<Item = Action<Step>> {
        match observation.message.payload {
            // A chain that has reached 1 is over, and the environment is the
            // one that needs to know.
            Step::Pass { chain, value: 1 } => {
                vec![Action::to(
                    [self.environment.clone()],
                    Step::Finished { chain },
                )]
            }
            Step::Pass { chain, value } => vec![self.pass(Step::Pass {
                chain,
                value: next(value),
            })],
            Step::Finished { chain } => {
                panic!("an agent of the ring was told chain {chain} finished")
            }
        }
    }
}

/// The environment of a Collatz ring: it starts every agent and stops
/// everybody, itself included, once every chain has reached 1.
///
/// It knows in advance every chain that will be opened, as a multiset of
/// starting numbers, because two chains opened from the same number share a
/// name. Each `Finished` strikes one off; when none is left the ring has
/// nothing more to do and the episode is over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Referee {
    id: ActorId,
    agents: BTreeSet<ActorId>,
    /// How many chains of each starting number are still running.
    outstanding: BTreeMap<u64, usize>,
    /// Whether the ring has been stopped, so that it is stopped once.
    stopped: bool,
}

impl Referee {
    /// An environment seated under `id`, over `agents`, expecting exactly the
    /// chains `opens` names, one entry per chain opened.
    pub fn new<A, I>(id: impl Into<ActorId>, agents: A, opens: I) -> Self
    where
        A: IntoIterator,
        A::Item: Into<ActorId>,
        I: IntoIterator<Item = u64>,
    {
        let mut outstanding: BTreeMap<u64, usize> = BTreeMap::new();
        for chain in opens {
            *outstanding.entry(chain).or_default() += 1;
        }
        Self {
            id: id.into(),
            agents: agents.into_iter().map(Into::into).collect(),
            outstanding,
            stopped: false,
        }
    }

    /// Strikes one chain named `chain` off the outstanding ones.
    ///
    /// # Panics
    ///
    /// If no chain of that name is outstanding, which means the ring finished a
    /// chain nobody opened, or finished one twice.
    fn finished(&mut self, chain: u64) {
        let remaining = self
            .outstanding
            .get_mut(&chain)
            .unwrap_or_else(|| panic!("chain {chain} finished, but none was outstanding"));
        *remaining -= 1;
        if *remaining == 0 {
            self.outstanding.remove(&chain);
        }
    }

    /// The one command that ends the episode — every actor stopped, this
    /// environment included — or nothing if chains are still running or the
    /// ring has already been stopped.
    ///
    /// **The environment stops itself.** That is how an episode of the
    /// actor runtime ends: the episode holds no
    /// view of what is in flight and stops nobody of its own accord until its
    /// time limit (ADR-0016).
    fn stop_if_done(&mut self) -> Vec<Effect<i32, Step>> {
        if self.outstanding.is_empty() && !self.stopped {
            self.stopped = true;
            let mut everybody: Vec<ActorId> = self.agents.iter().cloned().collect();
            everybody.push(self.id.clone());
            vec![Effect::command(everybody, Control::Stop)]
        } else {
            Vec::new()
        }
    }
}

impl Stepping<i32, Step> for Referee {
    /// Starts every agent of the ring, which is when they open their chains.
    ///
    /// A ring that opens nothing is over before it begins, and is stopped in
    /// the same call it is started.
    fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Effect<i32, Step>> {
        let mut effects = vec![Effect::command(
            self.agents.iter().cloned().collect::<Vec<_>>(),
            Control::Start,
        )];
        effects.extend(self.stop_if_done());
        effects
    }

    /// Strikes each finished chain off, and stops everybody once none is left.
    ///
    /// # Panics
    ///
    /// If an agent sends the environment a step, which is a message that only
    /// ever travels around the ring.
    fn step(
        &mut self,
        observation: Observation<Step>,
    ) -> impl IntoIterator<Item = Effect<i32, Step>> {
        match observation.message.payload {
            Step::Finished { chain } => self.finished(chain),
            Step::Pass { chain, value } => panic!(
                "{} sent the environment step {value} of chain {chain}",
                observation.message.sender
            ),
        }
        self.stop_if_done()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use social_deception::Message;

    use super::*;

    /// The environment every agent in these tests reports to.
    const ENVIRONMENT: &str = "environment";

    /// A message from `sender` to `recipient`, as the recipient observes it.
    ///
    /// The times play no part in what an actor does with a step, so one
    /// stand-in serves every test here.
    fn observation(sender: &str, recipient: &str, payload: Step) -> Observation<Step> {
        Observation {
            at: Instant::now(),
            message: Message::new(sender, [recipient], 0, payload),
        }
    }

    /// A step of chain `chain` observed by `a`.
    fn observing(sender: &str, chain: u64, value: u64) -> Observation<Step> {
        observation(sender, "a", Step::Pass { chain, value })
    }

    /// A step of chain 27 observed by `a`.
    fn step(sender: &str, value: u64) -> Observation<Step> {
        observing(sender, 27, value)
    }

    /// What an agent does with one observation, which is all a call has.
    ///
    /// **A policy is testable with no threads.** Call it with an observation
    /// and compare the actions it returns; there is no channel, no context and
    /// no fake runtime anywhere in this module.
    fn sent(agent: &mut Collatz, observation: Observation<Step>) -> Vec<Action<Step>> {
        agent.policy(observation).into_iter().collect()
    }

    fn to_b(chain: u64, value: u64) -> Action<Step> {
        Action::to(["b"], Step::Pass { chain, value })
    }

    fn finished(chain: u64) -> Action<Step> {
        Action::to([ENVIRONMENT], Step::Finished { chain })
    }

    fn agent() -> Collatz {
        Collatz::new("b", ENVIRONMENT)
    }

    #[test]
    fn a_policy_is_testable_with_no_threads() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, step("c", 6)), [to_b(27, 3)]);
        assert_eq!(sent(&mut agent, step("c", 3)), [to_b(27, 10)]);
    }

    #[test]
    fn one_ends_the_chain_and_the_environment_is_told() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, step("c", 1)), [finished(27)]);
    }

    #[test]
    fn a_step_keeps_its_chain() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, observing("c", 7, 10)), [to_b(7, 5)]);
    }

    #[test]
    fn chains_are_opened_at_the_start_in_order_and_named_by_their_start() {
        let mut opener = agent().opening(6).opening(7);
        assert_eq!(opener.opens(), [6, 7]);
        assert_eq!(opener.to(), &ActorId::new("b"));
        assert_eq!(
            opener.start(Clock::start()).into_iter().collect::<Vec<_>>(),
            [to_b(6, 6), to_b(7, 7)]
        );
        // An agent that opens nothing opens nothing.
        assert!(agent().start(Clock::start()).into_iter().next().is_none());
    }

    #[test]
    fn each_call_answers_its_one_observation() {
        // One observation per call, so three steps are three calls, and each
        // answer is the one its own step called for.
        let mut agent = agent();
        assert_eq!(sent(&mut agent, step("c", 8)), [to_b(27, 4)]);
        assert_eq!(sent(&mut agent, step("c", 1)), [finished(27)]);
        assert_eq!(sent(&mut agent, step("c", 3)), [to_b(27, 10)]);
    }

    #[test]
    #[should_panic(expected = "positive integer")]
    fn a_chain_cannot_start_at_zero() {
        let _ = agent().opening(0);
    }

    #[test]
    #[should_panic(expected = "told chain 6 finished")]
    fn an_agent_told_a_chain_finished_panics() {
        let mut agent = agent();
        let _ = sent(
            &mut agent,
            observation(ENVIRONMENT, "a", Step::Finished { chain: 6 }),
        );
    }

    /// An environment over `a` and `b` expecting the given chains.
    fn environment(opens: [u64; 2]) -> Referee {
        Referee::new(ENVIRONMENT, ["a", "b"], opens)
    }

    /// An agent's report that a chain has finished, as the environment observes
    /// it.
    fn reports(who: &str, chain: u64) -> Observation<Step> {
        observation(who, ENVIRONMENT, Step::Finished { chain })
    }

    fn started() -> Effect<i32, Step> {
        Effect::command(["a", "b"], Control::Start)
    }

    /// The command that ends an episode: everybody, the environment included.
    fn ended() -> Effect<i32, Step> {
        Effect::command(["a", "b", ENVIRONMENT], Control::Stop)
    }

    fn folded(environment: &mut Referee, observation: Observation<Step>) -> Vec<Effect<i32, Step>> {
        environment.step(observation).into_iter().collect()
    }

    fn opened(environment: &mut Referee) -> Vec<Effect<i32, Step>> {
        environment.start(Clock::start()).into_iter().collect()
    }

    #[test]
    fn the_environment_starts_the_ring_and_stops_everybody_when_every_chain_is_over() {
        let mut environment = environment([6, 7]);
        assert_eq!(opened(&mut environment), [started()]);
        assert!(folded(&mut environment, reports("a", 6)).is_empty());
        assert_eq!(folded(&mut environment, reports("b", 7)), [ended()]);
    }

    #[test]
    fn two_chains_of_the_same_name_are_counted_not_merged() {
        // Both are called 6, so only the count tells them apart: one report
        // leaves one running.
        let mut environment = environment([6, 6]);
        opened(&mut environment);
        assert!(folded(&mut environment, reports("a", 6)).is_empty());
        assert_eq!(folded(&mut environment, reports("a", 6)), [ended()]);
    }

    #[test]
    fn a_ring_that_opens_nothing_is_started_and_stopped_at_once() {
        let mut environment = Referee::new(ENVIRONMENT, ["a", "b"], []);
        assert_eq!(opened(&mut environment), [started(), ended()]);
    }

    #[test]
    #[should_panic(expected = "chain 9 finished, but none was outstanding")]
    fn a_chain_nobody_opened_panics() {
        let mut environment = environment([6, 7]);
        opened(&mut environment);
        let _ = folded(&mut environment, reports("a", 9));
    }

    #[test]
    #[should_panic(expected = "a sent the environment step 3 of chain 6")]
    fn a_step_sent_to_the_environment_panics() {
        let mut environment = environment([6, 7]);
        opened(&mut environment);
        let _ = folded(
            &mut environment,
            observation("a", ENVIRONMENT, Step::Pass { chain: 6, value: 3 }),
        );
    }

    #[test]
    fn next_halves_even_and_triples_plus_one_odd() {
        assert_eq!(next(6), 3);
        assert_eq!(next(3), 10);
        assert_eq!(next(2), 1);
        assert_eq!(next(1), 4);
    }

    #[test]
    #[should_panic(expected = "defined on positive integers")]
    fn next_rejects_zero() {
        let _ = next(0);
    }

    #[test]
    #[should_panic(expected = "does not fit in a u64")]
    fn next_is_loud_about_overflow() {
        let _ = next(u64::MAX);
    }
}
