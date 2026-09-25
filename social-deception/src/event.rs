//! What travels on the wire: [`Event`], [`Control`] and the [`Delivery`]
//! that carries either of them, and the [`Domain`] that names a game's
//! types.
//!
//! Two kinds of thing reach an agent, and the distinction is the one
//! ADR-0007 draws. An [`Event`] is *in-domain* data: something an agent said
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
//! An `Event` is a struct rather than an enum because there is now only one
//! thing it can be. It was an enum when it also had to carry controls and
//! timer wake-ups; a wake-up is neither in-domain nor out-of-domain nor
//! anything that traveled, so it is gone, and a deadline now calls the
//! handler's own [`timeout`](crate::Handler::timeout) rather than pretending
//! to be something observed.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use crate::clock::{Created, Timestamp};

/// What the runtime requires of a game's message payload:
/// `Serialize`, `Send`, `Clone` and `'static`.
///
/// The trait is a name for that bound, nothing more: it is implemented
/// automatically for every type that satisfies it.
pub trait Payload: Serialize + Send + Clone + 'static {}

impl<P: Serialize + Send + Clone + 'static> Payload for P {}

/// The types one game contributes to the runtime.
///
/// The runtime is generic over a `Domain` rather than over the payload and
/// the reward separately. The trait carries no behavior; it names a set of
/// types, as [`Payload`] names a bound. One trait rather than two type
/// parameters means a future per-game type is one more associated type here
/// instead of another parameter on every signature in the crate.
///
/// The reward type is only carried and serialized, never added up by the
/// runtime, so it needs no arithmetic bound. It is named here before
/// anything logs a reward, so that the generic parameter does not have to
/// change twice.
pub trait Domain: 'static {
    /// What this game's events carry.
    type Payload: Payload;
    /// The numeric type of this game's rewards. Integers for a game scored
    /// in wins and losses, reals for one scored more finely.
    type Reward: Serialize + Copy + Send + 'static;
}

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

/// In-domain data on the wire: what one agent said to others.
///
/// The same value is an [`Action`](crate::Action) of its sender and an
/// [`Observation`](crate::Observation) of each of its recipients; on the
/// wire it is only an event. The sender and the creation time are stamped by
/// the loop as it sends, never by the handler, which is why the value a
/// handler returns is an `Action` and not this.
/// `Debug`, `Clone`, equality and `Serialize` are implemented by hand
/// rather than derived, because a derive would demand each of them of `D`,
/// the marker type, when what actually has to have them is `D::Payload`.
pub struct Event<D: Domain> {
    /// The agent that sent it.
    pub sender: AgentId,
    /// The agents it was addressed to, in canonical order.
    ///
    /// The set never contains the sender; the router enforces that.
    pub recipients: BTreeSet<AgentId>,
    /// The instant the sender sent it.
    pub created: Timestamp,
    /// What was said. Its meaning belongs to the game.
    pub payload: D::Payload,
}

impl<D: Domain> Event<D> {
    /// An event from `sender` to `recipients`, created at `created`.
    pub fn new<I, A>(
        sender: impl Into<AgentId>,
        recipients: I,
        created: Timestamp,
        payload: D::Payload,
    ) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<AgentId>,
    {
        Self {
            sender: sender.into(),
            recipients: recipients.into_iter().map(Into::into).collect(),
            created,
            payload,
        }
    }
}

impl<D: Domain> Created for Event<D> {
    fn created(&self) -> Timestamp {
        self.created
    }
}

impl<D: Domain> fmt::Debug for Event<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Event")
            .field("sender", &self.sender)
            .field("recipients", &self.recipients)
            .field("created", &self.created)
            .field("payload", &self.payload)
            .finish()
    }
}

impl<D: Domain> Clone for Event<D> {
    fn clone(&self) -> Self {
        Self {
            sender: self.sender.clone(),
            recipients: self.recipients.clone(),
            created: self.created,
            payload: self.payload.clone(),
        }
    }
}

impl<D: Domain> PartialEq for Event<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.sender == other.sender
            && self.recipients == other.recipients
            && self.created == other.created
            && self.payload == other.payload
    }
}

impl<D: Domain> Eq for Event<D> where D::Payload: Eq {}

/// One thing on an agent's queue: an event, or a control and when it was
/// sent.
///
/// An agent has **one** queue, and it carries both kinds, so the queue's
/// message type has to be able to be either. That is an enum, and ADR-0009
/// reinstates the one ADR-0007 had for exactly this. ADR-0007's objection
/// was to an `Event` that *meant* three unlike things at once — a message,
/// an instruction, a timer wake-up — which made every handler ask what it
/// had been given before it could act. This is not that. It is a transport
/// carrying two things that stay clearly separate: the loop matches on the
/// variant and nothing else ever holds a `Delivery`, so an [`Observation`]
/// is still only ever an event and a control is still never observed.
///
/// One queue rather than two because the reasons for two are gone
/// (ADR-0009). A control no longer preempts anything, so there is nothing
/// for it to reach the agent ahead of, and a cycle handles one observation
/// (ADR-0008), so there is no batch for it to be queued behind. What is
/// left is a FIFO whose order is the order things were sent, which is the
/// order an agent handles them in.
///
/// A control carries its `created` here because nothing else does: an
/// [`Event`] has a field for the instant its sender made it and a
/// [`Control`] is a bare two-variant enum, so the stamp travels beside it.
/// `Debug`, `Clone` and equality are written out rather than derived, for
/// the reason [`Event`]'s are: a derive would ask them of `D`.
///
/// [`Observation`]: crate::Observation
pub enum Delivery<D: Domain> {
    /// In-domain data: what becomes the recipient's [`Observation`].
    ///
    /// [`Observation`]: crate::Observation
    Event(Event<D>),
    /// An out-of-domain instruction, and the instant the sender sent it.
    Control {
        /// What the agent is told.
        control: Control,
        /// When whoever sent it sent it. An [`Event`] carries its own; a
        /// [`Control`] has nowhere to put one, so it is here.
        created: Timestamp,
    },
}

impl<D: Domain> Delivery<D> {
    /// A control delivery stamped with `created`.
    #[must_use]
    pub const fn control(control: Control, created: Timestamp) -> Self {
        Self::Control { control, created }
    }
}

impl<D: Domain> fmt::Debug for Delivery<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Event(event) => f.debug_tuple("Event").field(event).finish(),
            Self::Control { control, created } => f
                .debug_struct("Control")
                .field("control", control)
                .field("created", created)
                .finish(),
        }
    }
}

impl<D: Domain> Clone for Delivery<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Event(event) => Self::Event(event.clone()),
            Self::Control { control, created } => Self::Control {
                control: *control,
                created: *created,
            },
        }
    }
}

impl<D: Domain> PartialEq for Delivery<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Event(mine), Self::Event(theirs)) => mine == theirs,
            (
                Self::Control { control, created },
                Self::Control {
                    control: other_control,
                    created: other_created,
                },
            ) => control == other_control && created == other_created,
            _ => false,
        }
    }
}

impl<D: Domain> Eq for Delivery<D> where D::Payload: Eq {}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    enum TestPayload {
        Step(u64),
    }

    /// A domain whose events carry a [`TestPayload`].
    struct TestDomain;

    impl Domain for TestDomain {
        type Payload = TestPayload;
        type Reward = i32;
    }

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    fn json<T: Serialize>(value: &T) -> serde_json::Value {
        serde_json::to_value(value).unwrap()
    }

    #[test]
    fn an_event_holds_its_recipients_in_canonical_order() {
        // An event is written to a trajectory by `trajectory::Envelope`,
        // which is the only wire shape it has, so what is asserted here is
        // the set itself: the order the envelope will write.
        let event = Event::<TestDomain>::new("a", ["c", "b"], at(40), TestPayload::Step(7));
        let recipients: Vec<&str> = event.recipients.iter().map(AgentId::as_str).collect();
        assert_eq!(recipients, ["b", "c"]);
    }

    #[test]
    fn an_event_knows_when_it_was_created() {
        let event = Event::<TestDomain>::new("a", ["b"], at(40), TestPayload::Step(7));
        assert_eq!(Created::created(&event), at(40));
    }

    #[test]
    fn a_delivery_is_one_kind_or_the_other_and_says_which() {
        // The whole point of the enum: one queue carries both, and what
        // came off it is still unambiguously an event or a control.
        let event = Event::<TestDomain>::new("a", ["b"], at(40), TestPayload::Step(7));
        let carried = Delivery::Event(event.clone());
        let Delivery::Event(back) = &carried else {
            panic!("an event delivery is an event: {carried:?}");
        };
        assert_eq!(back, &event);
        assert_eq!(carried, Delivery::Event(event));

        // A control has nowhere of its own to keep the instant it was sent,
        // so the delivery keeps it.
        let stop = Delivery::<TestDomain>::control(Control::Stop, at(10));
        let Delivery::Control { control, created } = stop else {
            panic!("a control delivery is a control: {stop:?}");
        };
        assert_eq!((control, created), (Control::Stop, at(10)));
        assert_ne!(
            Delivery::<TestDomain>::control(Control::Stop, at(10)),
            Delivery::control(Control::Start, at(10))
        );
    }

    #[test]
    fn a_control_serializes_as_its_name() {
        assert_eq!(json(&Control::Start), serde_json::json!("start"));
        assert_eq!(json(&Control::Stop), serde_json::json!("stop"));
    }

    #[test]
    fn agent_id_serializes_as_a_bare_string() {
        assert_eq!(json(&AgentId::new("alice")), serde_json::json!("alice"));
        assert_eq!(AgentId::new("alice").to_string(), "alice");
    }

    #[test]
    fn agent_id_deserializes_from_a_bare_string() {
        let id: AgentId = serde_json::from_value(serde_json::json!("alice")).unwrap();
        assert_eq!(id, AgentId::new("alice"));
    }

    #[test]
    fn an_empty_agent_id_does_not_deserialize() {
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
