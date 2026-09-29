//! What travels on the wire: [`Message`] and [`Control`], and the
//! [`Payload`] a game's messages carry.
//!
//! Two kinds of thing reach an actor, and the distinction is the one ADR-0007
//! draws. A [`Message`] is *in-domain* data: something an actor said to other
//! actors, carrying a payload whose meaning belongs entirely to the game. A
//! [`Control`] is *out-of-domain*: an instruction about the episode rather
//! than a move within it. Handlers see the first and never the second,
//! because a handler plays the game and the runtime runs the episode.
//!
//! The two travel on **separate channels**, an inbox and a control channel,
//! so the distinction is the transport's rather than a tag inside it
//! (ADR-0016). That is what lets a [`Control::Stop`] take effect ahead of
//! everything an actor has queued: the perception thread checks the control
//! channel before it looks at the inbox at all.
//!
//! A `Message` is a struct rather than an enum because there is only one
//! thing it can be. It was an enum when it also had to carry controls and
//! timer wake-ups; controls have their own channel, and a wake-up is now a
//! [`Reminder`](crate::Reminder) the actor set for itself, delivered back as
//! an ordinary message from itself, so there is no third kind of thing on the
//! wire.

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
/// An actor is a participant with an inbox and two threads of its own; agent
/// and environment are the two roles one plays, written as [`Policy`] and
/// [`Step`] (ADR-0016), and both are named by an id of this type.
///
/// Actor ids are strings. Application code addresses actors by id.
///
/// [`Policy`]: crate::Policy
/// [`Step`]: crate::Step
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

/// An out-of-domain instruction to an actor about the episode itself.
///
/// A control is not a move in the game and no handler ever sees one. It
/// travels on the actor's control channel rather than its inbox, and the
/// perception thread logs it and acts on it: `Start` calls the handler's
/// `start` hook, and `Stop` takes effect at once, ahead of anything in the
/// inbox (ADR-0016).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Control {
    /// The episode has started; the actor may begin acting.
    Start,
    /// The episode is over. The actor forwards nothing more, and its handler
    /// thread ends once the call in progress returns.
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

/// A received message's origin and payload, packaged so that an environment
/// can pass it on without losing who said it.
///
/// A relay is the relaying actor's **own** message, with its own sequence
/// number, and what it carries in its payload is one of these: the original
/// sender, the original sequence number, and what was said (ADR-0017). So a
/// recipient of a relay knows who spoke, and a reader of the log joins the
/// relay back to the speaker's own action record on the envelope's `(from,
/// seq)` while the outer record joins on the relayer's.
///
/// # One shape everywhere
///
/// An envelope sits inside a game's payload, where the framework never
/// looks. So that a parser can follow relays without knowing any game's
/// payload format, it serializes the same way in every application and
/// wherever in a payload it appears:
///
/// ```json
/// {"envelope":{"from":"alice","seq":7,"payload":{...}}}
/// ```
///
/// The single key `envelope` is what marks it. The framework still reads
/// nothing inside a payload; it only fixes how its own type is written, so
/// that any tool can find relays by shape.
///
/// Which payloads carry one is a game's decision, not the framework's: a
/// game that relays gives one of its payload variants an `Envelope`, and a
/// game whose actors talk directly has none anywhere.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Envelope<P> {
    /// The actor that sent the message being passed on.
    pub from: ActorId,
    /// Which of that actor's messages it was.
    pub seq: u64,
    /// What it said.
    pub payload: P,
}

impl<P> Envelope<P> {
    /// An envelope naming `from` and `seq` around `payload`.
    ///
    /// `payload` is whatever a game puts in an envelope, which need not be
    /// its whole payload type: a game whose payload is an enum relays one
    /// variant of it, and that variant's contents are what travels.
    pub fn new(from: impl Into<ActorId>, seq: u64, payload: P) -> Self {
        Self {
            from: from.into(),
            seq,
            payload,
        }
    }
}

/// The one shape an envelope has everywhere: a single `envelope` key around
/// `from`, `seq` and `payload` (ADR-0017).
///
/// Both directions go through this one type, so the shape is written down
/// once and reading cannot drift from writing. It is generic over how it
/// holds the origin and the payload so that writing can borrow them
/// (`Shape<&ActorId, &P>`) while reading owns them (`Shape<ActorId, P>`).
///
/// `deny_unknown_fields` is what makes the single key a check rather than a
/// convention: a payload that merely has a `from` and a `seq` of its own is
/// not an envelope, and neither is an envelope with anything else beside
/// them.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape<F, P> {
    envelope: Body<F, P>,
}

/// The body an envelope's one key holds.
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Body<F, P> {
    from: F,
    seq: u64,
    payload: P,
}

impl<P: Serialize> Serialize for Envelope<P> {
    /// Writes the shape above, borrowing rather than cloning.
    ///
    /// Written out rather than derived on `Envelope` itself because the shape
    /// is a promise to every reader of a log, whatever `P` is, and deriving
    /// it on the public type would leave the promise a consequence of which
    /// serde attributes happened to sit there.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Shape {
            envelope: Body {
                from: &self.from,
                seq: self.seq,
                payload: &self.payload,
            },
        }
        .serialize(serializer)
    }
}

impl<'de, P: Deserialize<'de>> Deserialize<'de> for Envelope<P> {
    /// Reads back exactly what [`serialize`](Envelope::serialize) writes, so
    /// that a game's payload type round trips through the log.
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let Shape {
            envelope: Body { from, seq, payload },
        } = Shape::<ActorId, P>::deserialize(deserializer)?;
        Ok(Self { from, seq, payload })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
    fn a_control_serializes_as_its_name() {
        assert_eq!(json(&Control::Start), serde_json::json!("start"));
        assert_eq!(json(&Control::Stop), serde_json::json!("stop"));
    }

    #[test]
    fn an_envelope_has_one_shape_whatever_it_carries() {
        // The promise ADR-0017 makes to every reader of a log: one
        // `envelope` key, and `from`, `seq` and `payload` under it. The
        // payload's own type is the game's business and changes nothing
        // about the envelope around it, so the shape is asserted over
        // payloads that serialize as unlike things: a struct variant, a bare
        // integer, a string and a null.
        let shape = |payload: serde_json::Value| serde_json::json!({"envelope": {"from": "alice", "seq": 7, "payload": payload}});
        assert_eq!(
            json(&Envelope::new("alice", 7, TestPayload::Step(3))),
            shape(serde_json::json!({"Step": 3}))
        );
        assert_eq!(
            json(&Envelope::new("alice", 7, 3_u64)),
            shape(serde_json::json!(3))
        );
        assert_eq!(
            json(&Envelope::new("alice", 7, "spoken")),
            shape(serde_json::json!("spoken"))
        );
        assert_eq!(
            json(&Envelope::<Option<u8>>::new("alice", 7, None)),
            shape(serde_json::Value::Null)
        );
    }

    #[test]
    fn an_envelope_says_who_spoke_and_nothing_of_the_relayers_own() {
        // The sender, the number and what was said. Whom alice addressed is
        // alice's action record's business, so the recipients are not in it;
        // neither is anything of the actor doing the relaying.
        let envelope = Envelope::new("alice", 7, TestPayload::Step(3));
        assert_eq!(envelope.from, ActorId::new("alice"));
        assert_eq!(envelope.seq, 7);
        assert_eq!(envelope.payload, TestPayload::Step(3));
        assert_eq!(
            json(&envelope),
            serde_json::json!(
                {"envelope": {"from": "alice", "seq": 7, "payload": {"Step": 3}}}
            )
        );
    }

    #[test]
    fn an_envelope_round_trips() {
        let envelope = Envelope::new("alice", 7, TestPayload::Step(3));
        let back: Envelope<TestPayload> = serde_json::from_value(json(&envelope)).unwrap();
        assert_eq!(back, envelope);
    }

    #[test]
    fn something_that_is_not_an_envelope_does_not_deserialize_as_one() {
        // The single key is what marks an envelope, so a payload that
        // happens to have a `from` and a `seq` at the top level is not one.
        let bare = serde_json::json!({"from": "alice", "seq": 7, "payload": {"Step": 3}});
        assert!(serde_json::from_value::<Envelope<TestPayload>>(bare).is_err());
        // Nor is an envelope with anything else beside its three fields, or
        // one missing any of them.
        let extra = serde_json::json!(
            {"envelope": {"from": "alice", "seq": 7, "payload": null, "to": "bob"}}
        );
        assert!(serde_json::from_value::<Envelope<TestPayload>>(extra).is_err());
        let short = serde_json::json!({"envelope": {"from": "alice", "seq": 7}});
        assert!(serde_json::from_value::<Envelope<TestPayload>>(short).is_err());
        // And an envelope names nobody unless its `from` is an actor id.
        let empty = serde_json::json!({"envelope": {"from": "", "seq": 7, "payload": {"Step": 3}}});
        assert!(serde_json::from_value::<Envelope<TestPayload>>(empty).is_err());
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
