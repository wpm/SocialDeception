//! The Collatz environment: agents pass a Collatz chain around a ring.
//!
//! The Collatz function takes a positive integer `n` to `n / 2` if `n` is
//! even and to `3n + 1` if it is odd. Iterated from any starting number
//! anyone has tried, it reaches 1. A chain is the sequence of values from a
//! starting number down to 1.
//!
//! Each agent in the environment passes to one other agent. On observing a
//! value it computes the next one and sends it on; on observing 1 it sends
//! nothing. An agent may also open chains of its own when it is started,
//! which is what its start hook is for. Collatz agents have no timeout, so
//! once every chain has reached 1 nothing is in flight and the episode ends
//! on its own.
//!
//! A chain is named by its starting number, and every step carries the name
//! of the chain it belongs to. Chains from different starting numbers merge,
//! so without the name a value in the log could not be attributed to a
//! chain once it had.
//!
//! Every value at every step is known in advance, so any difference between
//! the trajectory an episode writes and the sequence computed independently
//! is a bug in the runtime, not a model being unpredictable. That is what the
//! environment is for: it is a second [`Domain`] beside Werewolf's, which is
//! what keeps the runtime honest about being generic, and the runtime's
//! end-to-end test.

use serde::Serialize;
use social_deception::{Action, AgentId, Domain, Handler, Observation};

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

/// What Collatz agents say to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum CollatzPayload {
    /// One step of a chain. The recipient takes the next.
    Step {
        /// The chain this step belongs to: its starting number.
        chain: u64,
        /// The chain's current value.
        value: u64,
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

/// One agent of the Collatz environment.
///
/// It passes every value it receives one step on to a single other agent,
/// and opens chains of its own when the episode starts. It keeps no state
/// between cycles: the chain's name and value travel with the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Collatz {
    to: AgentId,
    opens: Vec<u64>,
}

impl Collatz {
    /// An agent that passes every chain it receives on to `to` and opens
    /// none of its own.
    pub fn new(to: impl Into<AgentId>) -> Self {
        Self {
            to: to.into(),
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

    /// The steps one observation calls for sending. It depends only on the
    /// step observed: an agent keeps no state between cycles, because the
    /// chain's name and value travel with the message.
    fn reply(observation: &Observation<CollatzDomain>) -> Vec<CollatzPayload> {
        let CollatzPayload::Step { chain, value } = observation.event.payload;
        match value {
            // A chain that has reached 1 is over.
            1 => Vec::new(),
            value => vec![CollatzPayload::Step {
                chain,
                value: next(value),
            }],
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

    fn handle(
        &mut self,
        observations: &[Observation<CollatzDomain>],
    ) -> Vec<Action<CollatzDomain>> {
        observations
            .iter()
            .flat_map(Self::reply)
            .map(|step| self.pass(step))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use social_deception::{Event, Timestamp};

    use super::*;

    /// A step of chain `chain` observed by `a`. The times play no part in
    /// what an agent does with it, so one stand-in serves every test here.
    fn observing(sender: &str, chain: u64, value: u64) -> Observation<CollatzDomain> {
        Observation {
            event: Event::new(
                sender,
                ["a"],
                Timestamp::default(),
                CollatzPayload::Step { chain, value },
            ),
            received: Timestamp::default(),
        }
    }

    /// A step of chain 27 observed by `a`.
    fn step(sender: &str, value: u64) -> Observation<CollatzDomain> {
        observing(sender, 27, value)
    }

    fn sent<const N: usize>(
        agent: &mut Collatz,
        observations: &[Observation<CollatzDomain>; N],
    ) -> Vec<Action<CollatzDomain>> {
        agent.handle(observations)
    }

    fn to_b(chain: u64, value: u64) -> Action<CollatzDomain> {
        Action::to(["b"], CollatzPayload::Step { chain, value })
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
        let mut agent = Collatz::new("b");
        assert_eq!(sent(&mut agent, &[step("c", 6)]), [to_b(27, 3)]);
        assert_eq!(sent(&mut agent, &[step("c", 3)]), [to_b(27, 10)]);
    }

    #[test]
    fn one_ends_the_chain() {
        let mut agent = Collatz::new("b");
        assert!(sent(&mut agent, &[step("c", 1)]).is_empty());
    }

    #[test]
    fn a_step_keeps_its_chain() {
        let mut agent = Collatz::new("b");
        assert_eq!(sent(&mut agent, &[observing("c", 7, 10)]), [to_b(7, 5)]);
    }

    #[test]
    fn chains_are_opened_at_the_start_in_order_and_named_by_their_start() {
        let mut agent = Collatz::new("b").opening(6).opening(7);
        assert_eq!(agent.opens(), [6, 7]);
        assert_eq!(agent.to(), &AgentId::new("b"));
        assert_eq!(agent.start(), [to_b(6, 6), to_b(7, 7)]);
        // An agent that opens nothing opens nothing.
        assert!(Collatz::new("b").start().is_empty());
    }

    #[test]
    fn a_cycle_is_answered_in_order() {
        let mut agent = Collatz::new("b");
        assert_eq!(
            sent(&mut agent, &[step("c", 8), step("c", 1), step("c", 3)]),
            [to_b(27, 4), to_b(27, 10)]
        );
    }

    #[test]
    fn a_cycle_with_no_observations_sends_nothing() {
        // What a timeout cycle looks like from inside the handler. A start
        // is not among these: the loop calls the start hook instead of
        // handing the handler a control.
        let mut agent = Collatz::new("b").opening(5);
        assert!(agent.handle(&[]).is_empty());
    }

    #[test]
    #[should_panic(expected = "positive integer")]
    fn a_chain_cannot_start_at_zero() {
        let _ = Collatz::new("b").opening(0);
    }
}
