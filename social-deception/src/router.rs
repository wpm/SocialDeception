//! The router: a map from agent id to that agent's sender, and nothing else.
//!
//! An episode's topology is fixed when it starts. The roster is known, it
//! does not change, and agents do not discover each other. A message carries
//! an explicit recipient set, and the router copies the event onto each
//! recipient's channel. Application code addresses agent ids and never
//! touches transport.
//!
//! The router validates at the boundary rather than trusting handlers. An
//! unknown agent id, an empty recipient set, and a sender in its own
//! recipient set are each rejected loudly. There is no loopback.
//!
//! Channels are unbounded. With bounded channels one agent slow to drain its
//! inbox would apply back-pressure through the router to every other agent in
//! the episode. A slow agent is a normal condition here and must not be able
//! to stall the world.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crossbeam_channel::Sender;

use crate::agent::Delivery;
use crate::clock::Clock;
use crate::event::{AgentId, Control, Event, Payload};

/// Why an event could not be routed.
///
/// The first three are invariants of the system: a handler that trips one
/// has a bug, and the episode fails rather than carrying on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// A sender or recipient that is not in the roster.
    UnknownAgent(AgentId),
    /// A message addressed to nobody.
    NoRecipients,
    /// A sender that addressed itself.
    Loopback(AgentId),
    /// A recipient whose inbox has been dropped, so the copy for it could
    /// not be delivered.
    InboxClosed(AgentId),
    /// Something other than a message. Control events go to everyone through
    /// [`Router::control`], and a think never leaves the agent that had it.
    NotAMessage,
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAgent(id) => write!(f, "no agent {id} in the roster"),
            Self::NoRecipients => f.write_str("a message must have at least one recipient"),
            Self::Loopback(id) => write!(f, "agent {id} addressed itself"),
            Self::InboxClosed(id) => write!(f, "the inbox of agent {id} is closed"),
            Self::NotAMessage => f.write_str("only messages are routed"),
        }
    }
}

impl Error for RouteError {}

/// The map from agent id to that agent's sender.
#[derive(Debug)]
pub struct Router<P> {
    inboxes: BTreeMap<AgentId, Sender<Delivery<P>>>,
    clock: Clock,
}

impl<P: Payload> Router<P> {
    /// A router over the given senders, stamping every delivery with
    /// `clock`.
    #[must_use]
    pub fn new(inboxes: BTreeMap<AgentId, Sender<Delivery<P>>>, clock: Clock) -> Self {
        Self { inboxes, clock }
    }

    /// The ids in the roster, in order.
    pub fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.inboxes.keys()
    }

    /// Copies a message onto the channel of each of its recipients and
    /// returns how many deliveries that was.
    ///
    /// # Errors
    ///
    /// - [`RouteError::NotAMessage`] for anything but [`Event::Message`];
    /// - [`RouteError::NoRecipients`] for an empty recipient set;
    /// - [`RouteError::Loopback`] if the sender is among the recipients;
    /// - [`RouteError::UnknownAgent`] if the sender or a recipient is not in
    ///   the roster;
    /// - [`RouteError::InboxClosed`] if a recipient's inbox has been dropped.
    ///   Recipients before it in the set have already received the event.
    ///
    /// Nothing is delivered when a validation fails.
    pub fn route(&self, event: &Event<P>) -> Result<usize, RouteError> {
        let Event::Message {
            sender, recipients, ..
        } = event
        else {
            return Err(RouteError::NotAMessage);
        };
        if recipients.is_empty() {
            return Err(RouteError::NoRecipients);
        }
        if recipients.contains(sender) {
            return Err(RouteError::Loopback(sender.clone()));
        }
        if let Some(unknown) = std::iter::once(sender)
            .chain(recipients)
            .find(|id| !self.inboxes.contains_key(*id))
        {
            return Err(RouteError::UnknownAgent(unknown.clone()));
        }
        self.deliver(recipients, event)
    }

    /// Delivers a control event to every agent in the roster and returns how
    /// many deliveries that was.
    ///
    /// # Errors
    ///
    /// [`RouteError::InboxClosed`] if some agent's inbox has been dropped.
    /// Agents before it in the roster have already received the event.
    pub fn control(&self, control: Control) -> Result<usize, RouteError> {
        self.deliver(self.inboxes.keys(), &Event::Control(control))
    }

    fn deliver<'a>(
        &self,
        recipients: impl IntoIterator<Item = &'a AgentId>,
        event: &Event<P>,
    ) -> Result<usize, RouteError> {
        let time = self.clock.now();
        let mut deliveries = 0;
        for id in recipients {
            let inbox = self
                .inboxes
                .get(id)
                .ok_or_else(|| RouteError::UnknownAgent(id.clone()))?;
            inbox
                .send(Delivery {
                    time,
                    event: event.clone(),
                })
                .map_err(|_| RouteError::InboxClosed(id.clone()))?;
            deliveries += 1;
        }
        Ok(deliveries)
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};

    use super::*;
    use crate::testing::id;

    fn world(names: &[&str]) -> (Router<u64>, BTreeMap<AgentId, Receiver<Delivery<u64>>>) {
        let mut senders = BTreeMap::new();
        let mut receivers = BTreeMap::new();
        for name in names {
            let (sender, receiver) = unbounded();
            senders.insert(AgentId::new(*name), sender);
            receivers.insert(AgentId::new(*name), receiver);
        }
        (Router::new(senders, Clock::start()), receivers)
    }

    #[test]
    fn a_message_is_copied_to_each_recipient_and_nobody_else() {
        let (router, inboxes) = world(&["a", "b", "c"]);
        let event = Event::message("a", ["b", "c"], 7);
        assert_eq!(router.route(&event), Ok(2));
        for name in ["b", "c"] {
            let delivery = inboxes[&id(name)].try_recv().unwrap();
            assert_eq!(delivery.event, event);
            assert!(delivery.time <= router.clock.now());
        }
        assert!(
            inboxes[&id("a")].try_recv().is_err(),
            "the sender gets no copy"
        );
    }

    #[test]
    fn an_unknown_recipient_is_rejected_and_nothing_is_delivered() {
        let (router, inboxes) = world(&["a", "b"]);
        let error = router
            .route(&Event::message("a", ["b", "nobody"], 1))
            .unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("nobody")));
        assert!(inboxes[&id("b")].try_recv().is_err());
    }

    #[test]
    fn an_unknown_sender_is_rejected() {
        let (router, _inboxes) = world(&["a", "b"]);
        let error = router
            .route(&Event::message("ghost", ["b"], 1))
            .unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("ghost")));
    }

    #[test]
    fn an_empty_recipient_set_is_rejected() {
        let (router, _inboxes) = world(&["a", "b"]);
        let error = router
            .route(&Event::message("a", Vec::<AgentId>::new(), 1))
            .unwrap_err();
        assert_eq!(error, RouteError::NoRecipients);
    }

    #[test]
    fn a_sender_in_its_own_recipient_set_is_rejected() {
        let (router, inboxes) = world(&["a", "b"]);
        let error = router
            .route(&Event::message("a", ["a", "b"], 1))
            .unwrap_err();
        assert_eq!(error, RouteError::Loopback(id("a")));
        assert!(inboxes[&id("b")].try_recv().is_err());
    }

    #[test]
    fn only_messages_are_routed() {
        let (router, _inboxes) = world(&["a"]);
        assert_eq!(router.route(&Event::Think), Err(RouteError::NotAMessage));
        assert_eq!(
            router.route(&Event::Control(Control::Start)),
            Err(RouteError::NotAMessage)
        );
    }

    #[test]
    fn control_goes_to_everyone() {
        let (router, inboxes) = world(&["a", "b", "c"]);
        assert_eq!(router.control(Control::Start), Ok(3));
        for inbox in inboxes.values() {
            assert_eq!(
                inbox.try_recv().unwrap().event,
                Event::Control(Control::Start)
            );
        }
        assert_eq!(
            router.ids().collect::<Vec<_>>(),
            [&id("a"), &id("b"), &id("c")]
        );
    }

    #[test]
    fn a_closed_inbox_is_reported() {
        let (router, mut inboxes) = world(&["a", "b"]);
        drop(inboxes.remove(&id("b")));
        assert_eq!(
            router.route(&Event::message("a", ["b"], 1)),
            Err(RouteError::InboxClosed(id("b")))
        );
        assert_eq!(
            router.control(Control::Stop),
            Err(RouteError::InboxClosed(id("b")))
        );
    }

    #[test]
    fn errors_explain_themselves() {
        assert_eq!(
            RouteError::UnknownAgent(id("z")).to_string(),
            "no agent z in the roster"
        );
        assert_eq!(
            RouteError::NoRecipients.to_string(),
            "a message must have at least one recipient"
        );
        assert_eq!(
            RouteError::Loopback(id("a")).to_string(),
            "agent a addressed itself"
        );
        assert_eq!(
            RouteError::InboxClosed(id("b")).to_string(),
            "the inbox of agent b is closed"
        );
        assert_eq!(
            RouteError::NotAMessage.to_string(),
            "only messages are routed"
        );
    }
}
