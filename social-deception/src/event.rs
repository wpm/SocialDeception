//! Everything that can happen to an agent.
//!
//! ADR-0001: every agent has exactly one receiver, and in-world messages from
//! other agents, control messages from the runtime, and the agent's own
//! internal "think" wake-ups all arrive on it as variants of one [`Event`]
//! enum. The runtime is agnostic about what a message says: the payload type
//! `P` belongs to the environment (Collatz, later Werewolf), and the runtime
//! only requires of it what the [`Payload`] trait states.

use std::collections::BTreeSet;
use std::fmt;

use serde::Serialize;

/// What the runtime requires of an environment's message payload.
///
/// `Serialize` because payloads are written to the trajectory; `Send` and
/// `'static` because they cross thread boundaries; `Clone` because one message
/// addressed to several agents is copied onto each recipient's channel.
///
/// The trait is a name for the bound, nothing more: it is implemented
/// automatically for every type that satisfies it.
pub trait Payload: Serialize + Send + Clone + 'static {}

impl<P: Serialize + Send + Clone + 'static> Payload for P {}

/// The name of an agent within an episode.
///
/// Agent ids are strings, per ADR-0001. They are what application code
/// addresses; it never touches the transport behind them.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct AgentId(String);

impl AgentId {
    /// Creates an agent id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for AgentId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for AgentId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for AgentId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

/// A runtime instruction to an agent.
///
/// Start and stop are the whole vocabulary for now; ADR-0001 says "start,
/// stop, and their kin", and the kin get added when something needs them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(tag = "control", rename_all = "snake_case")]
pub enum Control {
    /// The episode has started; the agent may begin acting.
    Start,
    /// The episode is over; the agent's loop exits after this event.
    Stop,
}

/// Something that happened to an agent.
///
/// This is the one type that ever arrives on an agent's receiver. The three
/// variants are the three sources of events named in ADR-0001, and there is
/// no fourth.
///
/// Serialises as an internally tagged object whose `kind` field names the
/// variant, so that a trajectory reader can dispatch on it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event<P> {
    /// Something said in the world.
    Message {
        /// The agent that sent it.
        sender: AgentId,
        /// The agents it was addressed to, in canonical order so that two logs
        /// of the same run compare equal.
        ///
        /// The set never contains the sender. `Think` is the only way an agent
        /// acts without external input, and it comes from the agent's own
        /// deadline rather than from the router. The router is where that
        /// invariant is enforced.
        recipients: BTreeSet<AgentId>,
        /// What was said. Its meaning belongs to the environment.
        payload: P,
    },
    /// A runtime instruction.
    Control(Control),
    /// The agent's own internal prompting to reconsider, produced by its
    /// receive deadline firing rather than by anything another agent did.
    Think,
}

impl<P> Event<P> {
    /// Creates a `Message` event.
    pub fn message<I, A>(sender: impl Into<AgentId>, recipients: I, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<AgentId>,
    {
        Self::Message {
            sender: sender.into(),
            recipients: recipients.into_iter().map(Into::into).collect(),
            payload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    enum TestPayload {
        Step(u64),
    }

    fn json<T: Serialize>(value: &T) -> serde_json::Value {
        serde_json::to_value(value).unwrap()
    }

    #[test]
    fn message_serialises_with_sorted_recipients() {
        let event = Event::message("a", ["c", "b"], TestPayload::Step(7));
        assert_eq!(
            json(&event),
            serde_json::json!({
                "kind": "message",
                "sender": "a",
                "recipients": ["b", "c"],
                "payload": {"Step": 7},
            })
        );
    }

    #[test]
    fn control_serialises_flat() {
        let event: Event<TestPayload> = Event::Control(Control::Stop);
        assert_eq!(
            json(&event),
            serde_json::json!({"kind": "control", "control": "stop"})
        );
    }

    #[test]
    fn think_serialises_as_its_tag_alone() {
        let event: Event<TestPayload> = Event::Think;
        assert_eq!(json(&event), serde_json::json!({"kind": "think"}));
    }

    #[test]
    fn agent_id_serialises_as_a_bare_string() {
        assert_eq!(json(&AgentId::new("alice")), serde_json::json!("alice"));
        assert_eq!(AgentId::new("alice").to_string(), "alice");
    }
}
