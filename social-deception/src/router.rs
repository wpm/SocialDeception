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
//! recipient set are each rejected loudly. There is no loopback.
//!
//! # One sender per agent, carrying both kinds
//!
//! An agent has one queue (ADR-0009), so the router holds one sender per
//! agent and wraps what it sends in a [`Delivery`], which says which kind
//! it is. Nothing here reorders anything: a control the router sends takes
//! its place behind whatever was sent to that agent before it, and the
//! agent reaches it there. Getting a `Stop` to an agent with an empty queue
//! is the [`Episode`](crate::Episode)'s business, and it does it by holding
//! the stop until nothing is in flight.
//!
//! # Only the environment commands, and only the environment rewards
//!
//! A control is the [`Environment`](crate::Environment)'s to send, and no
//! ordinary agent's (ADR-0007). The types already say so — a
//! [`Handler`](crate::Handler) returns [`Action`](crate::Action)s and has
//! no way to name a control — so the router's check is a backstop against
//! a hole in the runtime rather than against a game's code:
//! [`command`](Router::command) refuses a control whose sender is not the
//! environment the router was built with, with
//! [`RouteError::NotTheEnvironment`].
//!
//! A reward is the environment's alone for the same reason, and though it
//! travels nowhere, whom it may name is the same question about the same
//! roster. [`rewardable`](Router::rewardable) answers it, and the episode
//! asks before letting a cycle's rewards stand.
//!
//! Channels are unbounded. With bounded channels one agent slow to drain its
//! queues would apply back-pressure through the router to every other agent
//! in the episode. A slow agent is a normal condition here and must not be
//! able to stall the world.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use crossbeam_channel::Sender;

use crate::clock::Clock;
use crate::event::{AgentId, Control, Delivery, Domain, Event};

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
    /// A control whose sender is not the episode's environment. Only the
    /// environment commands; see the [module documentation](self).
    NotTheEnvironment(AgentId),
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownAgent(id) => write!(f, "no agent {id} in the roster"),
            Self::NoRecipients => f.write_str("an event must have at least one recipient"),
            Self::Loopback(id) => write!(f, "agent {id} addressed itself"),
            Self::QueueClosed(id) => write!(f, "the queue of agent {id} is closed"),
            Self::NotTheEnvironment(id) => {
                write!(f, "agent {id} is not the environment and cannot command")
            }
        }
    }
}

impl Error for RouteError {}

/// The sending half of one agent's queue.
///
/// One sender, because an agent has one queue: what distinguishes an
/// [`Event`] from a [`Control`] is the [`Delivery`] variant they travel in
/// rather than which channel they were put on (ADR-0009). It is a struct of
/// one field so that the router's map says what it holds, and so that a
/// second thing an agent must be addressed by has somewhere to go.
/// `Debug` and `Clone` are written out rather than derived, for the reason
/// [`Event`]'s are: a derive would ask them of `D`.
pub struct Queues<D: Domain> {
    /// Where everything said to the agent goes.
    pub queue: Sender<Delivery<D>>,
}

impl<D: Domain> fmt::Debug for Queues<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queues").finish_non_exhaustive()
    }
}

impl<D: Domain> Clone for Queues<D> {
    fn clone(&self) -> Self {
        Self {
            queue: self.queue.clone(),
        }
    }
}

/// The map from agent id to that agent's queues.
#[derive(Debug)]
pub struct Router<D: Domain> {
    queues: BTreeMap<AgentId, Queues<D>>,
    environment: AgentId,
    clock: Clock,
}

impl<D: Domain> Router<D> {
    /// A router over the given queues, whose `environment` is the one agent
    /// allowed to [`command`](Router::command), stamping every control it
    /// sends with `clock`.
    #[must_use]
    pub fn new(queues: BTreeMap<AgentId, Queues<D>>, environment: AgentId, clock: Clock) -> Self {
        Self {
            queues,
            environment,
            clock,
        }
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
    /// - [`RouteError::QueueClosed`] if a recipient's queue has been
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
                .queue
                .send(Delivery::Event(event.clone()))
                .map_err(|_| RouteError::QueueClosed(id.clone()))?;
            deliveries += 1;
        }
        Ok(deliveries)
    }

    /// Delivers a control to each of `to`, stamped with the instant it was
    /// sent, and returns how many deliveries that was.
    ///
    /// This is the episode's own way of commanding, and it asks no
    /// questions about who is sending: the episode starts and stops the
    /// environment, and is not an agent. What an *agent* asks for goes
    /// through [`command`](Router::command).
    ///
    /// # A `Stop` behind events is reached behind them
    ///
    /// A control goes on the same queue as everything else, so an agent
    /// with three events waiting handles those three and then the stop.
    /// And an agent that pops a `Stop` leaves whatever is still behind it
    /// unpopped, because an agent that has stopped did not observe it. So a
    /// `Stop` sent while anything is in flight either waits for that work
    /// or swallows it, and which depends on where in the queue it landed.
    ///
    /// A caller that wants an orderly stop must therefore establish that
    /// nothing is in flight first, as [`Episode`](crate::Episode) does by
    /// holding a `Stop` back until its count of routed-and-unhandled
    /// deliveries reads zero (ADR-0007). The episode's other `Stop`, the
    /// one it sends to abandon an episode that has already failed, does
    /// not and cannot wait for that: the trajectory it leaves is a record
    /// of the failure, and the agent may answer events queued ahead of the
    /// stop before it reaches it (ADR-0009).
    ///
    /// # Errors
    ///
    /// [`RouteError::UnknownAgent`] for a recipient not in the roster, and
    /// [`RouteError::QueueClosed`] if a recipient's queue has been dropped.
    /// Recipients before it in the set have already received the control.
    pub fn control(&self, to: &BTreeSet<AgentId>, control: Control) -> Result<usize, RouteError> {
        let mut deliveries = 0;
        for id in to {
            self.queues_of(id)?
                .queue
                .send(Delivery::control(control, self.clock.now()))
                .map_err(|_| RouteError::QueueClosed(id.clone()))?;
            deliveries += 1;
        }
        Ok(deliveries)
    }

    /// Delivers a control an agent asked for, having checked that the agent
    /// is the environment.
    ///
    /// # Errors
    ///
    /// [`RouteError::NotTheEnvironment`] if `sender` is not the environment
    /// this router was built with, [`RouteError::Loopback`] if the
    /// environment addressed itself, and whatever
    /// [`control`](Router::control) returns. Nothing is delivered when a
    /// validation fails.
    pub fn command(
        &self,
        sender: &AgentId,
        to: &BTreeSet<AgentId>,
        control: Control,
    ) -> Result<usize, RouteError> {
        self.validate(sender, to)?;
        self.control(to, control)
    }

    /// Whether a control from `sender` to `to` would be accepted, without
    /// sending it.
    ///
    /// This is what [`command`](Router::command) checks before it delivers
    /// anything. It is public because a caller that has to hold a control
    /// back — as an [`Episode`](crate::Episode) holds a `Stop` until
    /// nothing is in flight — should still refuse a bad one where it was
    /// asked for, rather than long afterwards.
    ///
    /// # Errors
    ///
    /// [`RouteError::NotTheEnvironment`] if `sender` is not the
    /// environment, [`RouteError::Loopback`] if it addressed itself, and
    /// [`RouteError::UnknownAgent`] for a recipient not in the roster.
    pub fn validate(&self, sender: &AgentId, to: &BTreeSet<AgentId>) -> Result<(), RouteError> {
        if *sender != self.environment {
            return Err(RouteError::NotTheEnvironment(sender.clone()));
        }
        if to.contains(sender) {
            return Err(RouteError::Loopback(sender.clone()));
        }
        // Every recipient is resolved before anything is sent, for the
        // reason `route` resolves first: a control addressed to a stranger
        // reaches nobody rather than everybody named before it.
        for id in to {
            self.queues_of(id)?;
        }
        Ok(())
    }

    /// Whether a reward from `sender` to `agent` would be accepted.
    ///
    /// A reward is logged and never sent, so there is nothing here to
    /// deliver and nothing to count; what the router is being asked is only
    /// whether the environment named somebody it could be rewarding. The
    /// answers are the same as for a control, and for the same reasons:
    /// only the environment rewards, and it cannot reward itself, having no
    /// game to play and so nothing its behavior could be worth.
    ///
    /// # Errors
    ///
    /// [`RouteError::NotTheEnvironment`] if `sender` is not the
    /// environment, [`RouteError::Loopback`] if it rewarded itself, and
    /// [`RouteError::UnknownAgent`] if `agent` is not in the roster.
    pub fn rewardable(&self, sender: &AgentId, agent: &AgentId) -> Result<(), RouteError> {
        if *sender != self.environment {
            return Err(RouteError::NotTheEnvironment(sender.clone()));
        }
        if *agent == *sender {
            return Err(RouteError::Loopback(sender.clone()));
        }
        self.queues_of(agent).map(|_| ())
    }

    /// Every agent in the roster but the environment: whom an episode
    /// starts and stops through its environment.
    #[must_use]
    pub fn agents(&self) -> BTreeSet<AgentId> {
        self.queues
            .keys()
            .filter(|id| **id != self.environment)
            .cloned()
            .collect()
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
    use crate::clock::Timestamp;
    use crate::testing::{TestDomain, TestPayload, id};

    /// The receiving end of one agent's queue: what an agent's loop would
    /// be waiting on.
    type Ends = Receiver<Delivery<TestDomain>>;

    /// A world whose agents are `names` and whose environment is the first
    /// of them, which is the only one allowed to command.
    fn world(names: &[&str]) -> (Router<TestDomain>, BTreeMap<AgentId, Ends>) {
        let mut queues = BTreeMap::new();
        let mut ends = BTreeMap::new();
        for name in names {
            let (sender, receiver) = unbounded();
            queues.insert(AgentId::new(*name), Queues { queue: sender });
            ends.insert(AgentId::new(*name), receiver);
        }
        (
            Router::new(queues, AgentId::new(names[0]), Clock::start()),
            ends,
        )
    }

    /// The event of a delivery, or a panic saying what it was instead.
    fn as_event(delivery: &Delivery<TestDomain>) -> &Event<TestDomain> {
        match delivery {
            Delivery::Event(event) => event,
            other @ Delivery::Control { .. } => panic!("expected an event: {other:?}"),
        }
    }

    /// Every agent of a world, which is what the episode's own controls go
    /// to.
    fn all(names: &[&str]) -> BTreeSet<AgentId> {
        names.iter().map(|name| AgentId::new(*name)).collect()
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
            let delivered = queues[&id(name)].try_recv().unwrap();
            assert_eq!(delivered, Delivery::Event(sent.clone()));
            assert!(
                queues[&id(name)].try_recv().is_err(),
                "one copy each, and nothing else"
            );
        }
        assert!(
            queues[&id("a")].try_recv().is_err(),
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
        let delivered = queues[&id("b")].try_recv().unwrap();
        assert_eq!(as_event(&delivered).created, created);
    }

    #[test]
    fn an_unknown_recipient_is_rejected_and_nothing_is_delivered() {
        let (router, queues) = world(&["a", "b"]);
        let error = router.route(&event("a", ["b", "nobody"], 1)).unwrap_err();
        assert_eq!(error, RouteError::UnknownAgent(id("nobody")));
        assert!(queues[&id("b")].try_recv().is_err());
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
        assert!(queues[&id("b")].try_recv().is_err());
    }

    #[test]
    fn a_control_goes_to_everyone_stamped_with_when_it_was_sent() {
        let (router, queues) = world(&["a", "b", "c"]);
        let before = router.clock.now();
        assert_eq!(
            router.control(&all(&["a", "b", "c"]), Control::Start),
            Ok(3)
        );
        let after = router.clock.now();
        for ends in queues.values() {
            let delivered = ends.try_recv().expect("a control was delivered");
            let Delivery::Control { control, created } = delivered else {
                panic!("a control is delivered as a control: {delivered:?}");
            };
            assert_eq!(control, Control::Start);
            assert!(before <= created && created <= after);
            assert!(ends.try_recv().is_err(), "one control each, and no more");
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
            router.control(&all(&["a", "b"]), Control::Stop),
            Err(RouteError::QueueClosed(id("b")))
        );
    }

    #[test]
    fn only_the_environment_rewards_and_never_itself() {
        // The same three answers as for a control, and for the same
        // reasons, except that nothing is delivered either way: a reward
        // is logged, so all the router is asked is whether the environment
        // named somebody it could be rewarding.
        let (router, queues) = world(&["env", "a", "b"]);
        assert_eq!(router.rewardable(&id("env"), &id("a")), Ok(()));
        assert_eq!(
            router.rewardable(&id("a"), &id("b")),
            Err(RouteError::NotTheEnvironment(id("a"))),
            "an ordinary agent cannot reward"
        );
        assert_eq!(
            router.rewardable(&id("env"), &id("env")),
            Err(RouteError::Loopback(id("env"))),
            "the environment plays no game, so it has nothing to be worth"
        );
        assert_eq!(
            router.rewardable(&id("env"), &id("nobody")),
            Err(RouteError::UnknownAgent(id("nobody")))
        );
        for name in ["a", "b"] {
            assert!(
                queues[&id(name)].try_recv().is_err(),
                "checking a reward delivers nothing to {name}"
            );
        }
    }

    #[test]
    fn only_the_environment_commands() {
        // The backstop for a hole in the runtime: an ordinary agent's
        // handler has no way to name a control, so nothing in a game can
        // reach this, and the check is here so that a future one cannot.
        let (router, queues) = world(&["env", "a", "b"]);
        assert_eq!(
            router.command(&id("a"), &all(&["b"]), Control::Stop),
            Err(RouteError::NotTheEnvironment(id("a")))
        );
        assert!(
            queues[&id("b")].try_recv().is_err(),
            "a refused control reaches nobody"
        );
        assert_eq!(
            router.command(&id("env"), &all(&["a", "b"]), Control::Stop),
            Ok(2)
        );
        for name in ["a", "b"] {
            let delivered = queues[&id(name)].try_recv().unwrap();
            assert!(
                matches!(
                    delivered,
                    Delivery::Control {
                        control: Control::Stop,
                        ..
                    }
                ),
                "a stop was delivered to {name}: {delivered:?}"
            );
        }
        assert_eq!(router.agents(), all(&["a", "b"]));
    }

    #[test]
    fn an_environment_cannot_command_itself_or_a_stranger() {
        let (router, queues) = world(&["env", "a"]);
        assert_eq!(
            router.command(&id("env"), &all(&["env", "a"]), Control::Stop),
            Err(RouteError::Loopback(id("env")))
        );
        assert_eq!(
            router.command(&id("env"), &all(&["a", "nobody"]), Control::Stop),
            Err(RouteError::UnknownAgent(id("nobody")))
        );
        assert!(
            queues[&id("a")].try_recv().is_err(),
            "a refused control reaches nobody, not even the recipients it named first"
        );
    }

    #[test]
    fn events_and_controls_share_one_queue_in_the_order_they_were_sent() {
        // One queue per agent, so a control takes its place behind whatever
        // was sent to that agent before it (ADR-0009). The router does no
        // reordering: getting a stop to an agent with an empty queue is the
        // episode's business, and it does it by holding the stop back until
        // nothing is in flight.
        let (router, queues) = world(&["a", "b"]);
        router.route(&event("a", ["b"], 1)).unwrap();
        router.control(&all(&["b"]), Control::Stop).unwrap();
        router.route(&event("a", ["b"], 2)).unwrap();

        let delivered: Vec<Delivery<TestDomain>> = queues[&id("b")].try_iter().collect();
        assert_eq!(delivered.len(), 3);
        assert_eq!(as_event(&delivered[0]).payload, TestPayload::Step(1));
        assert!(
            matches!(
                delivered[1],
                Delivery::Control {
                    control: Control::Stop,
                    ..
                }
            ),
            "the stop is second, where it was sent: {delivered:?}"
        );
        assert_eq!(as_event(&delivered[2]).payload, TestPayload::Step(2));
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
        assert_eq!(
            RouteError::NotTheEnvironment(id("a")).to_string(),
            "agent a is not the environment and cannot command"
        );
    }
}
