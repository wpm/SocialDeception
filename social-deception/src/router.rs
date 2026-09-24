//! The router: a map from agent id to that agent's sender, and nothing else.
//!
//! An episode's topology is fixed when it starts. The roster is known, it
//! does not change, and agents do not discover each other. An event carries
//! an explicit recipient set, and the router copies it onto each recipient's
//! channel. Application code addresses agent ids and never touches
//! transport.
//!
//! The router validates at the boundary rather than trusting handlers. An
//! unknown agent id, an empty recipient set, and a sender in its own
//! recipient set are each rejected loudly. There is no loopback. What it no
//! longer has to reject is a wake-up or a control arriving where an event
//! belongs: [`route`](Router::route) takes an [`Event`], and a control goes
//! through [`control`](Router::control), so the type says which is which.
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
use crate::event::{AgentId, Control, Domain, Event};

/// Why an event could not be routed.
///
/// Each is an invariant of the system: a handler that trips one has a bug,
/// and the episode fails rather than carrying on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// A sender or recipient that is not in the roster.
    UnknownAgent(AgentId),
    /// An event addressed to nobody.
    NoRecipients,
    /// A sender that addressed itself.
    Loopback(AgentId),
    /// A recipient whose inbox has been dropped, so the copy for it could
    /// not be delivered.
    InboxClosed(AgentId),
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAgent(id) => write!(f, "no agent {id} in the roster"),
            Self::NoRecipients => f.write_str("an event must have at least one recipient"),
            Self::Loopback(id) => write!(f, "agent {id} addressed itself"),
            Self::InboxClosed(id) => write!(f, "the inbox of agent {id} is closed"),
        }
    }
}

impl Error for RouteError {}

/// The map from agent id to that agent's sender.
#[derive(Debug)]
pub struct Router<D: Domain> {
    inboxes: BTreeMap<AgentId, Sender<Delivery<D>>>,
    clock: Clock,
}

impl<D: Domain> Router<D> {
    /// A router over the given senders, stamping every control it sends with
    /// `clock`.
    #[must_use]
    pub fn new(inboxes: BTreeMap<AgentId, Sender<Delivery<D>>>, clock: Clock) -> Self {
        Self { inboxes, clock }
    }

    /// The ids in the roster, in order.
    pub fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.inboxes.keys()
    }

    /// Copies an event onto the channel of each of its recipients and
    /// returns how many deliveries that was.
    ///
    /// # Errors
    ///
    /// - [`RouteError::NoRecipients`] for an empty recipient set;
    /// - [`RouteError::Loopback`] if the sender is among the recipients;
    /// - [`RouteError::UnknownAgent`] if the sender or a recipient is not in
    ///   the roster;
    /// - [`RouteError::InboxClosed`] if a recipient's inbox has been dropped.
    ///   Recipients before it in the set have already received the event.
    ///
    /// Nothing is delivered when a validation fails.
    pub fn route(&self, event: &Event<D>) -> Result<usize, RouteError> {
        let Event {
            sender, recipients, ..
        } = event;
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
        self.deliver(recipients, || Delivery::Event(event.clone()))
    }

    /// Delivers a control to every agent in the roster, stamped with the
    /// instant it was sent, and returns how many deliveries that was.
    ///
    /// # Errors
    ///
    /// [`RouteError::InboxClosed`] if some agent's inbox has been dropped.
    /// Agents before it in the roster have already received the control.
    pub fn control(&self, control: Control) -> Result<usize, RouteError> {
        self.deliver(self.inboxes.keys(), || {
            Delivery::control(self.clock, control)
        })
    }

    /// `delivery` is called once per recipient rather than cloned from one
    /// value, so routing to n agents makes exactly n deliveries and not
    /// n + 1: an event carries its recipients and its payload, so the
    /// spare copy was not a cheap one.
    fn deliver<'a>(
        &self,
        recipients: impl IntoIterator<Item = &'a AgentId>,
        delivery: impl Fn() -> Delivery<D>,
    ) -> Result<usize, RouteError> {
        let mut deliveries = 0;
        for id in recipients {
            let inbox = self
                .inboxes
                .get(id)
                .ok_or_else(|| RouteError::UnknownAgent(id.clone()))?;
            inbox
                .send(delivery())
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
    use crate::clock::Timestamp;
    use crate::testing::{TestDomain, TestPayload, id};

    type TestDelivery = Delivery<TestDomain>;

    fn world(
        names: &[&str],
    ) -> (
        Router<TestDomain>,
        BTreeMap<AgentId, Receiver<TestDelivery>>,
    ) {
        let mut senders = BTreeMap::new();
        let mut receivers = BTreeMap::new();
        for name in names {
            let (sender, receiver) = unbounded();
            senders.insert(AgentId::new(*name), sender);
            receivers.insert(AgentId::new(*name), receiver);
        }
        (Router::new(senders, Clock::start()), receivers)
    }

    /// An event from `sender` to `recipients`, created at a time the router
    /// neither reads nor changes.
    fn event<const N: usize>(sender: &str, recipients: [&str; N], n: u64) -> Event<TestDomain> {
        Event::new(
            sender,
            recipients,
            Timestamp::default(),
            TestPayload::Step(n),
        )
    }

    #[test]
    fn an_event_is_copied_to_each_recipient_and_nobody_else() {
        let (router, inboxes) = world(&["a", "b", "c"]);
        let sent = event("a", ["b", "c"], 7);
        assert_eq!(router.route(&sent), Ok(2));
        for name in ["b", "c"] {
            let delivery = inboxes[&id(name)].try_recv().unwrap();
            assert_eq!(delivery, Delivery::Event(sent.clone()));
        }
        assert!(
            inboxes[&id("a")].try_recv().is_err(),
            "the sender gets no copy"
        );
    }

    #[test]
    fn routing_leaves_the_senders_creation_time_alone() {
        // The event was created when its sender sent it; the router carries
        // it, and nothing about delivery changes when that was.
        let (router, inboxes) = world(&["a", "b"]);
        let created = Timestamp::from(std::time::Duration::from_nanos(40));
        let sent = Event::new("a", ["b"], created, TestPayload::Step(1));
        router.route(&sent).unwrap();
        let Ok(Delivery::Event(delivered)) = inboxes[&id("b")].try_recv() else {
            panic!("an event was delivered");
        };
        assert_eq!(delivered.created, created);
    }

    #[test]
    fn an_unknown_recipient_is_rejected_and_nothing_is_delivered() {
        let (router, inboxes) = world(&["a", "b"]);
        let error = router.route(&event("a", ["b", "nobody"], 1)).unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("nobody")));
        assert!(inboxes[&id("b")].try_recv().is_err());
    }

    #[test]
    fn an_unknown_sender_is_rejected() {
        let (router, _inboxes) = world(&["a", "b"]);
        let error = router.route(&event("ghost", ["b"], 1)).unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("ghost")));
    }

    #[test]
    fn an_empty_recipient_set_is_rejected() {
        let (router, _inboxes) = world(&["a", "b"]);
        let error = router.route(&event("a", [], 1)).unwrap_err();
        assert_eq!(error, RouteError::NoRecipients);
    }

    #[test]
    fn a_sender_in_its_own_recipient_set_is_rejected() {
        let (router, inboxes) = world(&["a", "b"]);
        let error = router.route(&event("a", ["a", "b"], 1)).unwrap_err();
        assert_eq!(error, RouteError::Loopback(id("a")));
        assert!(inboxes[&id("b")].try_recv().is_err());
    }

    #[test]
    fn a_control_goes_to_everyone_stamped_with_when_it_was_sent() {
        let (router, inboxes) = world(&["a", "b", "c"]);
        let before = router.clock.now();
        assert_eq!(router.control(Control::Start), Ok(3));
        let after = router.clock.now();
        for inbox in inboxes.values() {
            let Ok(TestDelivery::Control { control, created }) = inbox.try_recv() else {
                panic!("a control was delivered");
            };
            assert_eq!(control, Control::Start);
            assert!(before <= created && created <= after);
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
            router.route(&event("a", ["b"], 1)),
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
            "an event must have at least one recipient"
        );
        assert_eq!(
            RouteError::Loopback(id("a")).to_string(),
            "agent a addressed itself"
        );
        assert_eq!(
            RouteError::InboxClosed(id("b")).to_string(),
            "the inbox of agent b is closed"
        );
    }
}
