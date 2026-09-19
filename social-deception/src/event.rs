//! Everything that can happen to an agent.
//!
//! An agent has one receiver, and everything that arrives on it is an
//! [`Event`]: a message from another agent, a control instruction from the
//! runtime, or the agent's own think wake-up. The message payload type `P`
//! belongs to the environment; the runtime requires of it only what the
//! [`Payload`] trait states.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// What the runtime requires of an environment's message payload:
/// `Serialize`, `Send`, `Clone` and `'static`.
///
/// The trait is a name for that bound, nothing more: it is implemented
/// automatically for every type that satisfies it.
pub trait Payload: Serialize + Send + Clone + 'static {}

impl<P: Serialize + Send + Clone + 'static> Payload for P {}

/// The name of an agent within an episode.
///
/// Agent ids are strings. Application code addresses agents by id.
///
/// An id serializes as its bare string, and deserializes from one that is
/// not empty: an empty id names nobody, so a file that carries one is
/// malformed wherever the empty string appears, and every reader gets that
/// check from the type rather than writing its own.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct AgentId(String);

impl<'de> Deserialize<'de> for AgentId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id = String::deserialize(deserializer)?;
        if id.is_empty() {
            return Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Str(&id),
                &"a non-empty agent id",
            ));
        }
        Ok(Self(id))
    }
}

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
/// This is the one type that ever arrives on an agent's receiver.
///
/// Serializes as an internally tagged object whose `kind` field names the
/// variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Event<P> {
    /// Something said in the world.
    Message {
        /// The agent that sent it.
        sender: AgentId,
        /// The agents it was addressed to, in canonical order.
        ///
        /// The set never contains the sender; the router enforces that.
        recipients: BTreeSet<AgentId>,
        /// What was said. Its meaning belongs to the environment.
        payload: P,
    },
    /// A runtime instruction.
    Control(Control),
    /// The agent's own internal prompting to reconsider, produced by its
    /// receive deadline firing.
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

    #[test]
    fn agent_id_deserialises_from_a_bare_string() {
        let id: AgentId = serde_json::from_value(serde_json::json!("alice")).unwrap();
        assert_eq!(id, AgentId::new("alice"));
    }

    #[test]
    fn an_empty_agent_id_does_not_deserialise() {
        let error = serde_json::from_value::<AgentId>(serde_json::json!("")).unwrap_err();
        assert!(error.to_string().contains("non-empty agent id"), "{error}");
        // Inside a collection too, since that is where a reader meets it.
        let error =
            serde_json::from_value::<Vec<AgentId>>(serde_json::json!(["alice", ""])).unwrap_err();
        assert!(error.to_string().contains("non-empty agent id"), "{error}");
        // A number is not an id either.
        assert!(serde_json::from_value::<AgentId>(serde_json::json!(7)).is_err());
    }
}
