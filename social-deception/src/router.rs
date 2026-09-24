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
//! belongs: an event and a control travel on different channels, so the
//! router holds two senders per agent and the type says which is which.
//!
//! # A control preempts as it is sent
//!
//! An agent's control sender is a [`ControlSender`], not a plain channel
//! sender, so [`control`](Router::control) queues the control *and* trips
//! the recipient's current cycle in one step. The router does not know or
//! care that it is doing so; it is a property of the sending half it was
//! handed, which is what keeps a control on a queue and a cycle unaware of
//! it from being a state anything here can produce. See
//! [`cancel`](crate::cancel).
//!
//! Channels are unbounded. With bounded channels one agent slow to drain its
//! queues would apply back-pressure through the router to every other agent
//! in the episode. A slow agent is a normal condition here and must not be
//! able to stall the world.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crossbeam_channel::Sender;

use crate::cancel::ControlSender;
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
    /// A recipient whose queue has been dropped, so the copy for it could
    /// not be delivered.
    QueueClosed(AgentId),
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAgent(id) => write!(f, "no agent {id} in the roster"),
            Self::NoRecipients => f.write_str("an event must have at least one recipient"),
            Self::Loopback(id) => write!(f, "agent {id} addressed itself"),
            Self::QueueClosed(id) => write!(f, "the queue of agent {id} is closed"),
        }
    }
}

impl Error for RouteError {}

/// The two sending halves of one agent's queues.
///
/// They are held together because an agent is addressed as one thing, and
/// kept apart because what goes on them is: an [`Event`] is in-domain data
/// the handler will see, and a [`Control`] is an instruction to the loop
/// that preempts the cycle it lands in.
/// `Debug` and `Clone` are written out rather than derived, for the reason
/// [`Event`]'s are: a derive would ask them of `D`.
pub struct Queues<D: Domain> {
    /// Where the agent's events go.
    pub events: Sender<Event<D>>,
    /// Where the agent's controls go, tripping its current cycle as they
    /// land.
    pub controls: ControlSender,
}

impl<D: Domain> fmt::Debug for Queues<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queues")
            .field("controls", &self.controls)
            .finish_non_exhaustive()
    }
}

impl<D: Domain> Clone for Queues<D> {
    fn clone(&self) -> Self {
        Self {
            events: self.events.clone(),
            controls: self.controls.clone(),
        }
    }
}

/// The map from agent id to that agent's queues.
#[derive(Debug)]
pub struct Router<D: Domain> {
    queues: BTreeMap<AgentId, Queues<D>>,
    clock: Clock,
}

impl<D: Domain> Router<D> {
    /// A router over the given queues, stamping every control it sends with
    /// `clock`.
    #[must_use]
    pub fn new(queues: BTreeMap<AgentId, Queues<D>>, clock: Clock) -> Self {
        Self { queues, clock }
    }

    /// The ids in the roster, in order.
    pub fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.queues.keys()
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
    /// - [`RouteError::QueueClosed`] if a recipient's event queue has been
    ///   dropped.
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
        if !self.queues.contains_key(sender) {
            return Err(RouteError::UnknownAgent(sender.clone()));
        }
        // Every recipient is resolved before anything is sent, so an event
        // addressed to a stranger delivers to nobody rather than to the
        // agents that happened to be named before it. Resolving keeps the
        // queues it found, so each recipient is looked up once.
        let mut resolved = Vec::with_capacity(recipients.len());
        for id in recipients {
            resolved.push((id, self.queues_of(id)?));
        }
        let mut deliveries = 0;
        for (id, queues) in resolved {
            queues
                .events
                .send(event.clone())
                .map_err(|_| RouteError::QueueClosed(id.clone()))?;
            deliveries += 1;
        }
        Ok(deliveries)
    }

    /// Delivers a control to every agent in the roster, stamped with the
    /// instant it was sent, and returns how many deliveries that was.
    ///
    /// Each delivery trips the recipient's current cycle as it lands; see
    /// the [module documentation](self).
    ///
    /// # Errors
    ///
    /// [`RouteError::QueueClosed`] if some agent's control queue has been
    /// dropped. Agents before it in the roster have already received the
    /// control, and have already been tripped.
    pub fn control(&self, control: Control) -> Result<usize, RouteError> {
        let mut deliveries = 0;
        for (id, queues) in &self.queues {
            queues
                .controls
                .control(self.clock, control)
                .map_err(|_| RouteError::QueueClosed(id.clone()))?;
            deliveries += 1;
        }
        Ok(deliveries)
    }

    fn queues_of(&self, id: &AgentId) -> Result<&Queues<D>, RouteError> {
        self.queues
            .get(id)
            .ok_or_else(|| RouteError::UnknownAgent(id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};

    use super::*;
    use crate::cancel::{Arm, Cancel, Signal};
    use crate::clock::Timestamp;
    use crate::testing::{TestDomain, TestPayload, id};

    /// The receiving ends of one agent's two queues, and the arm an agent
    /// loop would hold. Nothing here runs a loop, so a cycle is armed only
    /// where a test asks for one, but the arm is kept either way so that the
    /// slot it shares lives as long as the sending half does, exactly as an
    /// agent's would.
    struct Ends {
        events: Receiver<Event<TestDomain>>,
        controls: Receiver<Signal>,
        arm: Arm,
    }

    fn world(names: &[&str]) -> (Router<TestDomain>, BTreeMap<AgentId, Ends>) {
        let mut queues = BTreeMap::new();
        let mut ends = BTreeMap::new();
        for name in names {
            let (sender, events) = unbounded();
            let (commander, controls, arm) = ControlSender::new();
            queues.insert(
                AgentId::new(*name),
                Queues {
                    events: sender,
                    controls: commander,
                },
            );
            ends.insert(
                AgentId::new(*name),
                Ends {
                    events,
                    controls,
                    arm,
                },
            );
        }
        (Router::new(queues, Clock::start()), ends)
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
        let (router, queues) = world(&["a", "b", "c"]);
        let sent = event("a", ["b", "c"], 7);
        assert_eq!(router.route(&sent), Ok(2));
        for name in ["b", "c"] {
            assert_eq!(queues[&id(name)].events.try_recv().unwrap(), sent);
            assert!(
                queues[&id(name)].controls.try_recv().is_err(),
                "an event goes on the event queue and nowhere else"
            );
        }
        assert!(
            queues[&id("a")].events.try_recv().is_err(),
            "the sender gets no copy"
        );
    }

    #[test]
    fn routing_leaves_the_senders_creation_time_alone() {
        // The event was created when its sender sent it; the router carries
        // it, and nothing about delivery changes when that was.
        let (router, queues) = world(&["a", "b"]);
        let created = Timestamp::from(std::time::Duration::from_nanos(40));
        let sent = Event::new("a", ["b"], created, TestPayload::Step(1));
        router.route(&sent).unwrap();
        let delivered = queues[&id("b")].events.try_recv().unwrap();
        assert_eq!(delivered.created, created);
    }

    #[test]
    fn an_unknown_recipient_is_rejected_and_nothing_is_delivered() {
        let (router, queues) = world(&["a", "b"]);
        let error = router.route(&event("a", ["b", "nobody"], 1)).unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("nobody")));
        assert!(queues[&id("b")].events.try_recv().is_err());
    }

    #[test]
    fn an_unknown_sender_is_rejected() {
        let (router, _queues) = world(&["a", "b"]);
        let error = router.route(&event("ghost", ["b"], 1)).unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("ghost")));
    }

    #[test]
    fn an_empty_recipient_set_is_rejected() {
        let (router, _queues) = world(&["a", "b"]);
        let error = router.route(&event("a", [], 1)).unwrap_err();
        assert_eq!(error, RouteError::NoRecipients);
    }

    #[test]
    fn a_sender_in_its_own_recipient_set_is_rejected() {
        let (router, queues) = world(&["a", "b"]);
        let error = router.route(&event("a", ["a", "b"], 1)).unwrap_err();
        assert_eq!(error, RouteError::Loopback(id("a")));
        assert!(queues[&id("b")].events.try_recv().is_err());
    }

    #[test]
    fn a_control_goes_to_everyone_stamped_with_when_it_was_sent() {
        let (router, queues) = world(&["a", "b", "c"]);
        let before = router.clock.now();
        assert_eq!(router.control(Control::Start), Ok(3));
        let after = router.clock.now();
        for ends in queues.values() {
            let delivered = ends.controls.try_recv().expect("a control was delivered");
            assert_eq!(delivered.control, Control::Start);
            assert!(before <= delivered.created && delivered.created <= after);
            assert!(
                ends.events.try_recv().is_err(),
                "a control goes on the control queue and nowhere else"
            );
        }
        assert_eq!(
            router.ids().collect::<Vec<_>>(),
            [&id("a"), &id("b"), &id("c")]
        );
    }

    #[test]
    fn a_closed_queue_is_reported() {
        let (router, mut queues) = world(&["a", "b"]);
        drop(queues.remove(&id("b")));
        assert_eq!(
            router.route(&event("a", ["b"], 1)),
            Err(RouteError::QueueClosed(id("b")))
        );
        assert_eq!(
            router.control(Control::Stop),
            Err(RouteError::QueueClosed(id("b")))
        );
    }

    #[test]
    fn a_control_trips_the_cycle_of_the_agent_it_goes_to_and_nobody_elses() {
        // This is the coupling the router itself knows nothing about: it
        // calls `send` on the sending half it was handed, and the preemption
        // is that half's doing.
        let (router, queues) = world(&["a", "b"]);
        let cycles: BTreeMap<&AgentId, _> = queues
            .iter()
            .map(|(who, ends)| (who, ends.arm.arm()))
            .collect();
        assert!(cycles.values().all(|cancel| !cancel.is_cancelled()));
        router.control(Control::Stop).unwrap();
        assert!(cycles.values().all(Cancel::is_cancelled));

        // Routing an event trips nobody: only a control preempts.
        let fresh: BTreeMap<&AgentId, _> = queues
            .iter()
            .map(|(who, ends)| (who, ends.arm.arm()))
            .collect();
        router.route(&event("a", ["b"], 1)).unwrap();
        assert!(fresh.values().all(|cancel| !cancel.is_cancelled()));
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
            RouteError::QueueClosed(id("b")).to_string(),
            "the queue of agent b is closed"
        );
    }
}
