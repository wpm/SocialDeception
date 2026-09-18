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
//! Every value at every step is known in advance, so any difference between
//! the trajectory an episode writes and the sequence computed independently
//! is a bug in the runtime, not a model being unpredictable. That is what the
//! environment is for: it is the first instantiation of the generic
//! [`Event`], and the runtime's end-to-end test.

use serde::Serialize;

use crate::agent::{Handler, Outgoing};
use crate::event::{AgentId, Control, Event};

/// What Collatz agents say to each other.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
pub enum CollatzPayload {
    /// The chain's current value. The recipient takes the next step.
    Step(u64),
}

/// The Collatz function: `n / 2` for even `n`, `3n + 1` for odd `n`.
///
/// # Panics
///
/// If `n` is 0, which is not in the function's domain and would map to
/// itself forever, or if `3n + 1` does not fit in a `u64`.
#[must_use]
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
/// between passes: the chain's value travels with the message.
///
/// # Example
///
/// A ring of two, one of which opens a chain from 6:
///
/// ```
/// use social_deception::collatz::Collatz;
/// use social_deception::{Episode, Writer};
///
/// let (records, writer) = Writer::spawn(Vec::new());
/// let mut episode = Episode::new(records);
/// episode.add("a", Collatz::new("b").opening(6)).unwrap();
/// episode.add("b", Collatz::new("a")).unwrap();
/// episode.run().unwrap();
///
/// // 6, 3, 10, 5, 16, 8, 4, 2, 1: nine values, each sent once.
/// let trajectory = String::from_utf8(writer.join().unwrap()).unwrap();
/// assert_eq!(trajectory.matches("\"sent\"").count(), 9);
/// ```
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
    pub fn to(&self) -> &AgentId {
        &self.to
    }

    /// The starting numbers of the chains this agent opens, in order.
    #[must_use]
    pub fn opens(&self) -> &[u64] {
        &self.opens
    }

    /// The values one event calls for sending.
    fn reply(&self, event: &Event<CollatzPayload>) -> Vec<u64> {
        match event {
            Event::Control(Control::Start) => self.opens.clone(),
            Event::Message {
                payload: CollatzPayload::Step(n),
                ..
            } => match *n {
                // A chain that has reached 1 is over.
                1 => Vec::new(),
                n => vec![next(n)],
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
            .map(|value| Outgoing::to([self.to.clone()], CollatzPayload::Step(value)))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(sender: &str, n: u64) -> Event<CollatzPayload> {
        Event::message(sender, ["a"], CollatzPayload::Step(n))
    }

    fn sent(
        agent: &mut Collatz,
        events: &[Event<CollatzPayload>],
    ) -> Vec<Outgoing<CollatzPayload>> {
        agent.handle(events)
    }

    fn to_b(n: u64) -> Outgoing<CollatzPayload> {
        Outgoing::to(["b"], CollatzPayload::Step(n))
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
        assert_eq!(sent(&mut agent, &[step("c", 6)]), [to_b(3)]);
        assert_eq!(sent(&mut agent, &[step("c", 3)]), [to_b(10)]);
    }

    #[test]
    fn one_ends_the_chain() {
        let mut agent = Collatz::new("b");
        assert!(sent(&mut agent, &[step("c", 1)]).is_empty());
    }

    #[test]
    fn chains_are_opened_on_start_in_order() {
        let mut agent = Collatz::new("b").opening(6).opening(7);
        assert_eq!(agent.opens(), [6, 7]);
        assert_eq!(agent.to(), &AgentId::new("b"));
        assert_eq!(
            sent(&mut agent, &[Event::Control(Control::Start)]),
            [to_b(6), to_b(7)]
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
        assert_eq!(sent(&mut agent, &batch), [to_b(5), to_b(4), to_b(10)]);
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
