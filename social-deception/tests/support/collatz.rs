//! The Collatz environment: agents pass a Collatz chain around a ring.
//!
//! The Collatz function takes a positive integer `n` to `n / 2` if `n` is
//! even and to `3n + 1` if it is odd. Iterated from any starting number
//! anyone has tried, it reaches 1. A chain is the sequence of values from a
//! starting number down to 1.
//!
//! Each agent in the ring passes to one other agent. On observing a value it
//! computes the next one and sends it on; on observing 1 it tells the
//! [`CollatzEnvironment`] that the chain is [`Finished`](CollatzPayload) and
//! sends nothing else. An agent may also open chains of its own when it is
//! started, which is what its start hook is for.
//!
//! The ring is purely reactive, so nothing in it could end an episode. The
//! environment is what does: it starts every agent, holds the multiset of
//! chains that will be opened, strikes one off for each `Finished`, and
//! stops every agent once none is left. A multiset and not a set, because
//! two chains opened from the same number share a name and cannot be told
//! apart; what can be counted is how many are outstanding.
//!
//! A chain is named by its starting number, and every step carries the name
//! of the chain it belongs to. Chains from different starting numbers merge,
//! so without the name a value in the log could not be attributed to a
//! chain once it had.
//!
//! Every value at every step is known in advance, so any difference between
//! the trajectory an episode writes and the sequence computed independently
//! is a bug in the runtime, not a model being unpredictable. That is what
//! this ring is for: it is a second [`Domain`] beside Werewolf's, and a
//! second [`Environment`] beside the moderator, which is what keeps the
//! runtime honest about being generic, and the runtime's end-to-end test.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use social_deception::{
    Action, AgentId, Cancel, Control, Domain, Effect, Environment, Handler, Observation,
};

/// The Collatz environment as a [`Domain`].
///
/// Its rewards are integers. Nothing assigns one yet: the type is named
/// because a domain names both of a game's types, and a ring passing numbers
/// around has no notion of winning to score.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CollatzDomain;

impl Domain for CollatzDomain {
    type Payload = CollatzPayload;
    type Reward = i32;
}

/// What Collatz agents say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum CollatzPayload {
    /// One step of a chain, from one agent of the ring to the next. The
    /// recipient takes the next step.
    Step {
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

/// One agent of the Collatz ring.
///
/// It passes every value it receives one step on to a single other agent,
/// tells the environment when a chain reaches 1, and opens chains of its own
/// when the episode starts. It keeps no state between cycles: the chain's
/// name and value travel with the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collatz {
    to: AgentId,
    environment: AgentId,
    opens: Vec<u64>,
}

impl Collatz {
    /// An agent that passes every chain it receives on to `to`, reports a
    /// finished chain to `environment`, and opens none of its own.
    pub fn new(to: impl Into<AgentId>, environment: impl Into<AgentId>) -> Self {
        Self {
            to: to.into(),
            environment: environment.into(),
            opens: Vec::new(),
        }
    }

    /// The same agent, also opening a chain from `start` when the episode
    /// starts. An agent opens its chains in the order this was called.
    ///
    /// The chain is named by `start`. Two chains opened from the same
    /// number, by this agent or by two, share a name and cannot be told
    /// apart in the trajectory.
    ///
    /// # Panics
    ///
    /// If `start` is 0, which is not in the Collatz function's domain.
    pub fn opening(mut self, start: u64) -> Self {
        assert!(start > 0, "a Collatz chain starts at a positive integer");
        self.opens.push(start);
        self
    }

    /// The agent this one passes to.
    pub fn to(&self) -> &AgentId {
        &self.to
    }

    /// The starting numbers of the chains this agent opens, in order.
    pub fn opens(&self) -> &[u64] {
        &self.opens
    }

    /// What one observation calls for sending. It depends only on the step
    /// observed: an agent keeps no state between cycles, because the chain's
    /// name and value travel with the message.
    ///
    /// # Panics
    ///
    /// If the environment sends an agent a `Finished`, which is a message
    /// that only ever travels the other way.
    fn reply(&self, observation: &Observation<CollatzDomain>) -> Vec<Action<CollatzDomain>> {
        match observation.event.payload {
            // A chain that has reached 1 is over, and the environment is
            // the one that needs to know.
            CollatzPayload::Step { chain, value: 1 } => vec![Action::to(
                [self.environment.clone()],
                CollatzPayload::Finished { chain },
            )],
            CollatzPayload::Step { chain, value } => vec![self.pass(CollatzPayload::Step {
                chain,
                value: next(value),
            })],
            CollatzPayload::Finished { chain } => {
                panic!("an agent of the ring was told chain {chain} finished")
            }
        }
    }

    /// One step, addressed to the agent this one passes to.
    fn pass(&self, step: CollatzPayload) -> Action<CollatzDomain> {
        Action::to([self.to.clone()], step)
    }
}

impl Handler<CollatzDomain> for Collatz {
    /// Opens this agent's own chains, each at its starting value.
    fn start(&mut self) -> Vec<Action<CollatzDomain>> {
        self.opens
            .iter()
            .map(|&start| {
                self.pass(CollatzPayload::Step {
                    chain: start,
                    value: start,
                })
            })
            .collect()
    }

    /// Ignores `cancel`: computing the next value of a chain cannot block,
    /// so there is nothing a preemption could interrupt.
    fn handle(
        &mut self,
        observation: &Observation<CollatzDomain>,
        _: &Cancel,
    ) -> Vec<Action<CollatzDomain>> {
        self.reply(observation)
    }
}

/// The environment of a Collatz ring: it starts every agent and stops them
/// all once every chain has reached 1.
///
/// It knows in advance every chain that will be opened, as a multiset of
/// starting numbers, because two chains opened from the same number share a
/// name. Each `Finished` strikes one off; when none is left the ring has
/// nothing more to do and the episode is over.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CollatzEnvironment {
    agents: BTreeSet<AgentId>,
    /// How many chains of each starting number are still running.
    outstanding: BTreeMap<u64, usize>,
    /// Whether the ring has been stopped, so that it is stopped once.
    ///
    /// The loop calls the handler in the same cycle it calls the start
    /// hook, so an environment whose work is already done when it starts
    /// would otherwise ask for the stop twice in one cycle, and an agent
    /// told to stop twice is an agent whose trajectory says it was running
    /// after it had stopped.
    stopped: bool,
}

impl CollatzEnvironment {
    /// An environment over `agents`, expecting exactly the chains `opens`
    /// names, one entry per chain opened.
    pub fn new<A, I>(agents: A, opens: I) -> Self
    where
        A: IntoIterator,
        A::Item: Into<AgentId>,
        I: IntoIterator<Item = u64>,
    {
        let mut outstanding: BTreeMap<u64, usize> = BTreeMap::new();
        for chain in opens {
            *outstanding.entry(chain).or_default() += 1;
        }
        Self {
            agents: agents.into_iter().map(Into::into).collect(),
            outstanding,
            stopped: false,
        }
    }

    /// Whether every chain the environment was expecting has finished.
    fn done(&self) -> bool {
        self.outstanding.is_empty()
    }

    /// Strikes one chain named `chain` off the outstanding ones.
    ///
    /// # Panics
    ///
    /// If no chain of that name is outstanding, which means the ring
    /// finished a chain nobody opened, or finished one twice.
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

    /// The one control that ends the episode, or nothing if chains are
    /// still running or the ring has already been stopped.
    fn stop_if_done(&mut self) -> Vec<Effect<CollatzDomain>> {
        if self.done() && !self.stopped {
            self.stopped = true;
            vec![Effect::control(self.agents.clone(), Control::Stop)]
        } else {
            Vec::new()
        }
    }
}

impl Environment<CollatzDomain> for CollatzEnvironment {
    /// Starts every agent of the ring, which is when they open their chains.
    ///
    /// A ring that opens nothing is over before it begins, and is stopped in
    /// the same cycle it is started.
    fn start(&mut self) -> Vec<Effect<CollatzDomain>> {
        let mut effects = vec![Effect::control(self.agents.clone(), Control::Start)];
        effects.extend(self.stop_if_done());
        effects
    }

    /// Strikes each finished chain off, and stops the ring once none is
    /// left.
    ///
    /// Ignores `cancel`: striking a chain off a map cannot block.
    ///
    /// # Panics
    ///
    /// If an agent sends the environment a step, which is a message that
    /// only ever travels around the ring.
    fn handle(
        &mut self,
        observation: &Observation<CollatzDomain>,
        _: &Cancel,
    ) -> Vec<Effect<CollatzDomain>> {
        match observation.event.payload {
            CollatzPayload::Finished { chain } => self.finished(chain),
            CollatzPayload::Step { chain, value } => panic!(
                "{} sent the environment step {value} of chain {chain}",
                observation.event.sender
            ),
        }
        self.stop_if_done()
    }
}

#[cfg(test)]
mod tests {
    use social_deception::{Event, Timestamp};

    use super::*;

    /// The environment every agent in these tests reports to.
    const ENVIRONMENT: &str = "environment";

    /// A step of chain `chain` observed by `a`. The times play no part in
    /// what an agent does with it, so one stand-in serves every test here.
    fn observing(sender: &str, chain: u64, value: u64) -> Observation<CollatzDomain> {
        observation(sender, "a", CollatzPayload::Step { chain, value })
    }

    /// An event from `sender` to `recipient`, as the recipient observes it.
    fn observation(
        sender: &str,
        recipient: &str,
        payload: CollatzPayload,
    ) -> Observation<CollatzDomain> {
        Observation {
            event: Event::new(sender, [recipient], Timestamp::default(), payload),
            received: Timestamp::default(),
        }
    }

    /// A step of chain 27 observed by `a`.
    fn step(sender: &str, value: u64) -> Observation<CollatzDomain> {
        observing(sender, 27, value)
    }

    /// What an agent does with one cycle's observation, which is all a
    /// cycle has.
    fn sent(
        agent: &mut Collatz,
        observation: &Observation<CollatzDomain>,
    ) -> Vec<Action<CollatzDomain>> {
        agent.handle(observation, &Cancel::cancelled())
    }

    fn to_b(chain: u64, value: u64) -> Action<CollatzDomain> {
        Action::to(["b"], CollatzPayload::Step { chain, value })
    }

    fn finished(chain: u64) -> Action<CollatzDomain> {
        Action::to([ENVIRONMENT], CollatzPayload::Finished { chain })
    }

    fn agent() -> Collatz {
        Collatz::new("b", ENVIRONMENT)
    }

    #[test]
    fn next_halves_even_and_triples_plus_one_odd() {
        assert_eq!(next(6), 3);
        assert_eq!(next(3), 10);
        assert_eq!(next(2), 1);
        assert_eq!(next(1), 4);
    }

    #[test]
    #[should_panic(expected = "positive integers")]
    fn next_rejects_zero() {
        let _ = next(0);
    }

    #[test]
    #[should_panic(expected = "does not fit")]
    fn next_is_loud_about_overflow() {
        let _ = next(u64::MAX);
    }

    #[test]
    fn a_value_is_passed_on_one_step_further() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, &step("c", 6)), [to_b(27, 3)]);
        assert_eq!(sent(&mut agent, &step("c", 3)), [to_b(27, 10)]);
    }

    #[test]
    fn one_ends_the_chain_and_the_environment_is_told() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, &step("c", 1)), [finished(27)]);
    }

    #[test]
    fn a_step_keeps_its_chain() {
        let mut agent = agent();
        assert_eq!(sent(&mut agent, &observing("c", 7, 10)), [to_b(7, 5)]);
    }

    #[test]
    fn chains_are_opened_at_the_start_in_order_and_named_by_their_start() {
        let mut opener = agent().opening(6).opening(7);
        assert_eq!(opener.opens(), [6, 7]);
        assert_eq!(opener.to(), &AgentId::new("b"));
        assert_eq!(opener.start(), [to_b(6, 6), to_b(7, 7)]);
        // An agent that opens nothing opens nothing.
        assert!(agent().start().is_empty());
    }

    #[test]
    fn each_cycle_answers_its_one_observation() {
        // One observation per cycle, so three steps are three cycles, and
        // each answer is the one its own step called for.
        let mut agent = agent();
        assert_eq!(sent(&mut agent, &step("c", 8)), [to_b(27, 4)]);
        assert_eq!(sent(&mut agent, &step("c", 1)), [finished(27)]);
        assert_eq!(sent(&mut agent, &step("c", 3)), [to_b(27, 10)]);
    }

    #[test]
    fn a_timeout_sends_nothing() {
        // What a timeout cycle looks like from inside the handler: the loop
        // calls `timeout`, not `handle`, and this agent has nothing to do on
        // a deadline.
        let mut agent = agent().opening(5);
        assert!(agent.timeout(&Cancel::cancelled()).is_empty());
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
            &observation(ENVIRONMENT, "a", CollatzPayload::Finished { chain: 6 }),
        );
    }

    /// An environment over `a` and `b` expecting the given chains.
    fn environment(opens: [u64; 2]) -> CollatzEnvironment {
        CollatzEnvironment::new(["a", "b"], opens)
    }

    /// An agent's report that a chain has finished, as the environment
    /// observes it.
    fn reports(who: &str, chain: u64) -> Observation<CollatzDomain> {
        observation(who, ENVIRONMENT, CollatzPayload::Finished { chain })
    }

    fn start(agents: [&str; 2]) -> Effect<CollatzDomain> {
        Effect::control(agents, Control::Start)
    }

    fn stop(agents: [&str; 2]) -> Effect<CollatzDomain> {
        Effect::control(agents, Control::Stop)
    }

    fn folded(
        environment: &mut CollatzEnvironment,
        observation: &Observation<CollatzDomain>,
    ) -> Vec<Effect<CollatzDomain>> {
        environment.handle(observation, &Cancel::cancelled())
    }

    #[test]
    fn the_environment_starts_the_ring_and_stops_it_when_every_chain_is_over() {
        let mut environment = environment([6, 7]);
        assert_eq!(environment.start(), [start(["a", "b"])]);
        assert!(folded(&mut environment, &reports("a", 6)).is_empty());
        assert_eq!(
            folded(&mut environment, &reports("b", 7)),
            [stop(["a", "b"])]
        );
    }

    #[test]
    fn two_chains_of_the_same_name_are_counted_not_merged() {
        // Both are called 6, so only the count tells them apart: one report
        // leaves one running.
        let mut environment = environment([6, 6]);
        environment.start();
        assert!(folded(&mut environment, &reports("a", 6)).is_empty());
        assert_eq!(
            folded(&mut environment, &reports("a", 6)),
            [stop(["a", "b"])]
        );
    }

    #[test]
    fn a_ring_that_opens_nothing_is_started_and_stopped_at_once() {
        let mut environment = CollatzEnvironment::new(["a", "b"], []);
        assert_eq!(environment.start(), [start(["a", "b"]), stop(["a", "b"])]);
    }

    #[test]
    #[should_panic(expected = "chain 9 finished, but none was outstanding")]
    fn a_chain_nobody_opened_panics() {
        let mut environment = environment([6, 7]);
        environment.start();
        let _ = folded(&mut environment, &reports("a", 9));
    }

    #[test]
    #[should_panic(expected = "a sent the environment step 3 of chain 6")]
    fn a_step_sent_to_the_environment_panics() {
        let mut environment = environment([6, 7]);
        environment.start();
        let _ = folded(
            &mut environment,
            &observation(
                "a",
                ENVIRONMENT,
                CollatzPayload::Step { chain: 6, value: 3 },
            ),
        );
    }
}
