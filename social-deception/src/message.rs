//! What travels on the wire: [`Message`], [`Control`] and the [`Delivery`]
//! that carries either of them, and the [`Payload`] a game's messages
//! carry.
//!
//! Two kinds of thing reach an agent, and the distinction is the one
//! ADR-0007 draws. A [`Message`] is *in-domain* data: something an agent said
//! to other agents, carrying a payload whose meaning belongs entirely to the
//! game. A [`Control`] is *out-of-domain*: an instruction about the episode
//! rather than a move within it. Handlers see the first and never the
//! second, because a handler plays the game and the loop runs the episode.
//!
//! Both travel on one queue, so the thing actually sent is a [`Delivery`],
//! which is one or the other (ADR-0009). The distinction survives the
//! transport rather than being erased by it: the enum has two variants and
//! the loop matches on them, so a control is never mistaken for something a
//! handler should see.
//!
//! A `Message` is a struct rather than an enum because there is now only one
//! thing it can be. It was an enum when it also had to carry controls and
//! timer wake-ups; a wake-up is neither in-domain nor out-of-domain nor
//! anything that traveled, so it is gone, and a deadline now calls the
//! handler's own [`timeout`](crate::Handler::timeout) rather than pretending
//! to be something observed.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

/// What the runtime requires of a game's message payload:
/// `Serialize`, `Send`, `Clone` and `'static`.
///
/// The trait is a name for that bound, nothing more: it is implemented
/// automatically for every type that satisfies it.
pub trait Payload: Serialize + Send + Clone + 'static {}

impl<P: Serialize + Send + Clone + 'static> Payload for P {}

/// The name of an actor within an episode.
///
/// An actor is a thread-backed participant with an inbox; [`Agent`] and
/// [`Environment`] are the two roles one plays (ADR-0016), and both are
/// named by an id of this type.
///
/// Actor ids are strings. Application code addresses actors by id.
///
/// [`Agent`]: crate::Agent
/// [`Environment`]: crate::Environment
///
/// An id serializes as its bare string, and deserializes from one that is
/// not empty: an empty id names nobody, so a file that carries one is
/// malformed wherever the empty string appears, and every reader gets that
/// check from the type rather than writing its own.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ActorId(String);

impl<'de> Deserialize<'de> for ActorId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id = String::deserialize(deserializer)?;
        if id.is_empty() {
            return Err(serde::de::Error::invalid_value(
                serde::de::Unexpected::Str(&id),
                &"a non-empty actor id",
            ));
        }
        Ok(Self(id))
    }
}

impl ActorId {
    /// Creates an actor id.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for ActorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for ActorId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for ActorId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

/// An out-of-domain instruction to an agent about the episode itself.
///
/// A control is not a move in the game and no handler ever sees one. The
/// loop logs it and acts on it: `Start` makes it call the handler's
/// [`start`](crate::Handler::start) hook, `Stop` makes it exit after the
/// cycle that popped it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    /// The episode has started; the agent may begin acting.
    Start,
    /// The episode is over; the agent's loop exits after this cycle.
    Stop,
}

/// In-domain data on the wire: what one agent said to others, and which of
/// its sender's messages it is.
///
/// The same value is an [`Action`](crate::Action) of its sender and an
/// [`Observation`](crate::Observation) of each of its recipients; on the
/// wire it is only a message. The sender and the sequence number are stamped
/// by the loop as it sends, never by the handler, which is why the value a
/// handler returns is an `Action` and not this.
///
/// **A message carries no time** (ADR-0017). The times are in the log: an
/// observation record says when this agent received it and the sender's
/// action record says when it was sent, and `(sender, seq)` is what joins
/// the two. What an agent perceives of latency is when its observations
/// arrived relative to each other, which is what its own records hold; when
/// somebody else's clock read is a fact about that clock and not about the
/// message.
///
/// It has no `Serialize` of its own: a message goes into the log through the
/// record that carries it, which writes it without the `seq` that record
/// already carries at the top level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message<P: Payload> {
    /// The agent that sent it.
    pub sender: ActorId,
    /// The agents it was addressed to, in canonical order.
    ///
    /// The set never contains the sender; the router enforces that.
    pub recipients: BTreeSet<ActorId>,
    /// Which of its sender's messages this is, counting from zero.
    ///
    /// The number is the sender's, so it means nothing without
    /// [`sender`](Self::sender): two agents both have a message 0. **One
    /// send to five recipients is one message and one number**, so the
    /// sender's action record and each recipient's observation record carry
    /// the same pair and one action joins to all of its observations
    /// (ADR-0017).
    pub seq: u64,
    /// What was said. Its meaning belongs to the game.
    pub payload: P,
}

impl<P: Payload> Message<P> {
    /// A message from `sender` to `recipients`, the `seq`th that sender
    /// sent.
    pub fn new<I, A>(sender: impl Into<ActorId>, recipients: I, seq: u64, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self {
            sender: sender.into(),
            recipients: recipients.into_iter().map(Into::into).collect(),
            seq,
            payload,
        }
    }
}

/// One thing on an agent's queue: a message, or a control and when it was
/// sent.
///
/// An agent has **one** queue, and it carries both kinds, so the type the
/// queue carries has to be able to be either. That is an enum, and ADR-0009
/// reinstates the one ADR-0007 had for exactly this. ADR-0007's objection was
/// to a single type that *meant* three unlike things at once — in-domain
/// data, an instruction, a timer wake-up — which made every handler ask what
/// it had been given before it could act. This is not that. It is a transport
/// carrying two things that stay clearly separate: the loop matches on the
/// variant and nothing else ever holds a `Delivery`, so an [`Observation`] is
/// still only ever a message and a control is still never observed.
///
/// One queue rather than two because the reasons for two are gone
/// (ADR-0009). A control no longer preempts anything, so there is nothing
/// for it to reach the agent ahead of, and a cycle handles one observation
/// (ADR-0008), so there is no batch for it to be queued behind. What is
/// left is a FIFO whose order is the order things were sent, which is the
/// order an agent handles them in.
///
/// Neither variant carries a time. Nothing on the wire does (ADR-0017): a
/// control's record says when the agent popped it, which is the one instant
/// about it that agent knows.
///
/// [`Observation`]: crate::Observation
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Delivery<P: Payload> {
    /// In-domain data: what becomes the recipient's [`Observation`].
    ///
    /// [`Observation`]: crate::Observation
    Message(Message<P>),
    /// An out-of-domain instruction about the episode.
    Control(Control),
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
    fn a_message_holds_its_recipients_in_canonical_order() {
        // A message is written to the log by `log::Envelope`,
        // which is the only wire shape it has, so what is asserted here is
        // the set itself: the order the envelope will write.
        let message = Message::<TestPayload>::new("a", ["c", "b"], 0, TestPayload::Step(7));
        let recipients: Vec<&str> = message.recipients.iter().map(ActorId::as_str).collect();
        assert_eq!(recipients, ["b", "c"]);
    }

    #[test]
    fn a_message_knows_which_of_its_sender_s_it_is_and_no_time() {
        // The number is the sender's own, and it is the only thing besides
        // the sender that tells one of its messages from another: a message
        // carries no time at all (ADR-0017).
        let message = Message::<TestPayload>::new("a", ["b"], 7, TestPayload::Step(7));
        assert_eq!(message.seq, 7);
        assert_ne!(
            message,
            Message::new("a", ["b"], 8, TestPayload::Step(7)),
            "two messages of one sender differ by their sequence number"
        );
    }

    #[test]
    fn a_delivery_is_one_kind_or_the_other_and_says_which() {
        // The whole point of the enum: one queue carries both, and what
        // came off it is still unambiguously a message or a control.
        let message = Message::<TestPayload>::new("a", ["b"], 0, TestPayload::Step(7));
        let carried = Delivery::Message(message.clone());
        let Delivery::Message(back) = &carried else {
            panic!("a message delivery is a message: {carried:?}");
        };
        assert_eq!(back, &message);
        assert_eq!(carried, Delivery::Message(message));

        // A control is itself and nothing beside it: nothing on the wire
        // carries a time.
        let stop = Delivery::<TestPayload>::Control(Control::Stop);
        let Delivery::Control(control) = stop else {
            panic!("a control delivery is a control: {stop:?}");
        };
        assert_eq!(control, Control::Stop);
        assert_ne!(
            Delivery::<TestPayload>::Control(Control::Stop),
            Delivery::Control(Control::Start)
        );
    }

    #[test]
    fn a_control_serializes_as_its_name() {
        assert_eq!(json(&Control::Start), serde_json::json!("start"));
        assert_eq!(json(&Control::Stop), serde_json::json!("stop"));
    }

    #[test]
    fn actor_id_serializes_as_a_bare_string() {
        assert_eq!(json(&ActorId::new("alice")), serde_json::json!("alice"));
        assert_eq!(ActorId::new("alice").to_string(), "alice");
    }

    #[test]
    fn actor_id_deserializes_from_a_bare_string() {
        let id: ActorId = serde_json::from_value(serde_json::json!("alice")).unwrap();
        assert_eq!(id, ActorId::new("alice"));
    }

    #[test]
    fn an_empty_actor_id_does_not_deserialize() {
        let error = serde_json::from_value::<ActorId>(serde_json::json!("")).unwrap_err();
        assert!(error.to_string().contains("non-empty actor id"), "{error}");
        // Inside a collection too, since that is where a reader meets it.
        let error =
            serde_json::from_value::<Vec<ActorId>>(serde_json::json!(["alice", ""])).unwrap_err();
        assert!(error.to_string().contains("non-empty actor id"), "{error}");
        // A number is not an id either.
        assert!(serde_json::from_value::<ActorId>(serde_json::json!(7)).is_err());
    }
}
