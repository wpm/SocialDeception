//! The Collatz environment: agents pass a Collatz chain around a ring.
//!
//! The Collatz function takes a positive integer `n` to `n / 2` if `n` is
//! even and to `3n + 1` if it is odd. Iterated from any starting number
//! anyone has tried, it reaches 1. A chain is the sequence of values from a
//! starting number down to 1.
//!
//! Each agent in the environment passes to one other agent. On receiving a
//! value it computes the next one and sends it on; on receiving 1 it sends
//! nothing. An agent may also open chains of its own when the episode starts.
//! Collatz agents never think unprompted, so once every chain has reached 1
//! nothing is in flight and the episode ends on its own.
//!
//! A chain is named by its starting number, and every step carries the name
//! of the chain it belongs to. Chains from different starting numbers merge,
//! so without the name a value in the log could not be attributed to a
//! chain once it had.
//!
//! Every value at every step is known in advance, so any difference between
//! the trajectory an episode writes and the sequence computed independently
//! is a bug in the runtime, not a model being unpredictable. That is what the
//! environment is for: it is the first instantiation of the generic
//! [`Event`], and the runtime's end-to-end test.

use serde::Serialize;
use social_deception::{AgentId, Control, Event, Handler, Outgoing};

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

    /// The steps one event calls for sending.
    fn reply(&self, event: &Event<CollatzPayload>) -> Vec<CollatzPayload> {
        match event {
            Event::Control(Control::Start) => self
                .opens
                .iter()
                .map(|&start| CollatzPayload::Step {
                    chain: start,
                    value: start,
                })
                .collect(),
            Event::Message {
                payload: CollatzPayload::Step { chain, value },
                ..
            } => match *value {
                // A chain that has reached 1 is over.
                1 => Vec::new(),
                value => vec![CollatzPayload::Step {
                    chain: *chain,
                    value: next(value),
                }],
            },
            Event::Control(Control::Stop) | Event::Think => Vec::new(),
        }
    }
}

impl Handler<CollatzPayload> for Collatz {
    fn handle(&mut self, events: &[Event<CollatzPayload>]) -> Vec<Outgoing<CollatzPayload>> {
        events
            .iter()
            .flat_map(|event| self.reply(event))
            .map(|step| Outgoing::to([self.to.clone()], step))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A step of chain 27 arriving at `a`.
    fn step(sender: &str, value: u64) -> Event<CollatzPayload> {
        Event::message(sender, ["a"], CollatzPayload::Step { chain: 27, value })
    }

    fn sent(
        agent: &mut Collatz,
        events: &[Event<CollatzPayload>],
    ) -> Vec<Outgoing<CollatzPayload>> {
        agent.handle(events)
    }

    fn to_b(chain: u64, value: u64) -> Outgoing<CollatzPayload> {
        Outgoing::to(["b"], CollatzPayload::Step { chain, value })
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
        let step = Event::message(
            "c",
            ["a"],
            CollatzPayload::Step {
                chain: 7,
                value: 10,
            },
        );
        assert_eq!(sent(&mut agent, &[step]), [to_b(7, 5)]);
    }

    #[test]
    fn chains_are_opened_on_start_in_order_and_named_by_their_start() {
        let mut agent = Collatz::new("b").opening(6).opening(7);
        assert_eq!(agent.opens(), [6, 7]);
        assert_eq!(agent.to(), &AgentId::new("b"));
        assert_eq!(
            sent(&mut agent, &[Event::Control(Control::Start)]),
            [to_b(6, 6), to_b(7, 7)]
        );
        assert!(
            Collatz::new("b")
                .handle(&[Event::Control(Control::Start)])
                .is_empty()
        );
    }

    #[test]
    fn a_batch_is_answered_in_order() {
        let mut agent = Collatz::new("b").opening(5);
        let batch = [
            Event::Control(Control::Start),
            step("c", 8),
            step("c", 1),
            step("c", 3),
        ];
        assert_eq!(
            sent(&mut agent, &batch),
            [to_b(5, 5), to_b(27, 4), to_b(27, 10)]
        );
    }

    #[test]
    fn stop_and_think_send_nothing() {
        let mut agent = Collatz::new("b").opening(5);
        assert!(sent(&mut agent, &[Event::Control(Control::Stop), Event::Think]).is_empty());
    }

    #[test]
    #[should_panic(expected = "positive integer")]
    fn a_chain_cannot_start_at_zero() {
        let _ = Collatz::new("b").opening(0);
    }
}
