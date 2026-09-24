//! An episode: one run of an environment with a fixed roster of agents, from
//! coordinated startup to coordinated shutdown.
//!
//! The episode constructs the roster, wires each agent's inbox, dispatch
//! channel and record channel, spawns one thread per agent, tells every
//! agent to start, routes what the agents send until the episode is over,
//! tells every agent to stop, and joins the threads. A [`Control`] is how it
//! says start and stop, and the episode stamps each one with the instant it
//! sent it, so that an agent's record of popping it says how long it waited.
//!
//! # Quiescence
//!
//! With no deadlines configured, an episode is over when no agent will ever
//! send again, which for a purely reactive environment is the natural end
//! of the run. The episode can tell, because it is the only component that
//! sees both halves of the work in progress: the deliveries it has routed
//! and the cycles the agents have reported. It keeps one count, of
//! deliveries routed and not yet reported handled. A cycle in progress is
//! exactly a set of deliveries taken off an inbox and not yet reported, and
//! an agent dispatches a cycle's deliveries and its outputs together, so the
//! count never reads zero while a cycle that might still send is under way.
//! When it reaches zero the episode is quiescent and shutdown begins.
//!
//! In-flight work is counted in **deliveries, not events**: one event
//! addressed to six agents is six handles that have not happened yet.
//!
//! An episode's agents have no timeout: an agent with one keeps waking on
//! its own, and the episode would never go quiescent.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use crossbeam_channel::{Receiver, Sender, select, unbounded};

use crate::agent::{self, Action, Agent, CycleDispatch, Handler, Observation, Wiring};
use crate::cancel::{Cancel, ControlSender};
use crate::clock::Clock;
use crate::event::{AgentId, Control, Domain};
use crate::router::{Queues, RouteError, Router};
use crate::trajectory::LogRecord;

/// Why an agent's thread did not end cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// The loop stopped because a channel it depends on went away.
    Error(agent::Error),
    /// The thread panicked, in the handler or elsewhere, with this message.
    Panicked(String),
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Error(error) => error.fmt(f),
            Self::Panicked(message) => write!(f, "panicked: {message}"),
        }
    }
}

/// Why an episode could not be built or did not run cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EpisodeError {
    /// An id was added to the roster twice.
    DuplicateAgent(AgentId),
    /// A control could not be delivered.
    Control(RouteError),
    /// The router refused an event this agent sent. That is a bug in the
    /// agent's handler, and the episode stops rather than carrying on
    /// without the event.
    Route {
        /// The agent that sent it.
        agent: AgentId,
        /// What was wrong with it.
        error: RouteError,
    },
    /// Some agents' threads did not end cleanly, and why.
    Agents(Vec<(AgentId, Failure)>),
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAgent(id) => write!(f, "agent {id} is in the roster twice"),
            Self::Control(error) => write!(f, "could not deliver a control: {error}"),
            Self::Route { agent, error } => {
                write!(
                    f,
                    "agent {agent} sent an event that could not be routed: {error}"
                )
            }
            Self::Agents(failures) => {
                f.write_str("agents failed:")?;
                for (id, failure) in failures {
                    write!(f, " [{id}: {failure}]")?;
                }
                Ok(())
            }
        }
    }
}

impl Error for EpisodeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Control(error) | Self::Route { error, .. } => Some(error),
            Self::DuplicateAgent(_) | Self::Agents(_) => None,
        }
    }
}

/// A roster of handlers and everything needed to run them once.
///
/// Build it with [`Episode::new`], add agents with [`Episode::add`], then
/// [`Episode::run`] it. The roster is fixed from the moment `run` starts.
pub struct Episode<D: Domain> {
    roster: BTreeMap<AgentId, Box<dyn Handler<D> + Send>>,
    records: Sender<LogRecord<D>>,
    clock: Clock,
}

impl<D: Domain> fmt::Debug for Episode<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Episode")
            .field("roster", &self.roster.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl<D: Domain> Episode<D> {
    /// An empty roster whose trajectory goes to `records`, timed by a
    /// [`Clock`] started now.
    #[must_use]
    pub fn new(records: Sender<LogRecord<D>>) -> Self {
        Self::with_clock(records, Clock::start())
    }

    /// An empty roster whose trajectory goes to `records`, timed by `clock`.
    #[must_use]
    pub fn with_clock(records: Sender<LogRecord<D>>, clock: Clock) -> Self {
        Self {
            roster: BTreeMap::new(),
            records,
            clock,
        }
    }

    /// Adds an agent to the roster.
    ///
    /// # Errors
    ///
    /// [`EpisodeError::DuplicateAgent`] if `id` is already in the roster.
    pub fn add(
        &mut self,
        id: impl Into<AgentId>,
        handler: impl Handler<D> + Send + 'static,
    ) -> Result<(), EpisodeError> {
        let id = id.into();
        if self.roster.contains_key(&id) {
            return Err(EpisodeError::DuplicateAgent(id));
        }
        self.roster.insert(id, Box::new(handler));
        Ok(())
    }

    /// The ids in the roster, in order.
    pub fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.roster.keys()
    }

    /// Runs the episode to completion: starts every agent, routes until the
    /// episode is quiescent, stops every agent, and joins every thread.
    ///
    /// The episode's own sender to the trajectory writer is dropped on the
    /// way, so once this returns the writer has no senders left and can be
    /// joined for the complete trajectory.
    ///
    /// # Errors
    ///
    /// If a control could not be delivered, an agent sent an event the
    /// router refused, or an agent's thread ended with an error or a
    /// panic. In every case the agents that were running have been stopped
    /// and joined before this returns, so the trajectory is complete up to
    /// the failure.
    pub fn run(self) -> Result<(), EpisodeError> {
        let Self {
            roster,
            records,
            clock,
        } = self;
        let ids: BTreeSet<AgentId> = roster.keys().cloned().collect();
        let (dispatch, dispatches) = unbounded();
        let (obituary, obituaries) = unbounded();
        let mut queues = BTreeMap::new();
        // Every queue stays open until every thread has been joined, so that
        // a message to an agent that has already stopped is delivered, and
        // never read, rather than failing its sender.
        let mut held = Vec::with_capacity(roster.len());
        let mut agents = Vec::with_capacity(roster.len());
        for (id, handler) in roster {
            let (sender, events) = unbounded();
            let (commander, controls, arm) = ControlSender::new();
            held.push((events.clone(), controls.clone()));
            queues.insert(
                id.clone(),
                Queues {
                    events: sender,
                    controls: commander,
                },
            );
            let wiring = Wiring {
                id: id.clone(),
                clock,
                events,
                controls,
                arm,
                dispatches: dispatch.clone(),
                records: records.clone(),
                timeout: None,
                peers: ids.iter().filter(|peer| **peer != id).cloned().collect(),
            };
            let watched = Watched {
                id,
                handler,
                obituary: obituary.clone(),
            };
            agents.push(Agent::spawn(wiring, watched, clock));
        }
        drop((dispatch, obituary, records));
        let router = Router::new(queues, clock);

        let outcome = router
            .control(Control::Start)
            .map_err(|error| Halt::Error(EpisodeError::Control(error)))
            .and_then(|in_flight| drive(&router, &dispatches, &obituaries, in_flight));
        let stopped = router.control(Control::Stop).map_err(EpisodeError::Control);
        let failures = join(agents);
        drop(held);

        match (outcome, stopped) {
            (Err(Departure), _) | (Ok(()), Ok(_)) if !failures.is_empty() => {
                Err(EpisodeError::Agents(failures))
            }
            (Err(Departure), _) | (Ok(()), Ok(_)) => Ok(()),
            (Err(Halt::Error(error)), _) | (Ok(()), Err(error)) => Err(error),
        }
    }
}

/// Why routing stopped before quiescence.
enum Halt {
    /// An agent's thread has ended, or every agent's has; joining them says
    /// why.
    Departure,
    /// A control event or a message could not be routed.
    Error(EpisodeError),
}

use Halt::Departure;

/// Routes what the agents send until nothing is in flight.
///
/// `in_flight` is the number of deliveries already made and not yet reported
/// handled. Each dispatch takes a cycle's deliveries off the count and puts the
/// deliveries its outputs cause onto it, in that order but as one step, so
/// the count reads zero only when no agent has anything left to handle or
/// send.
fn drive<D: Domain>(
    router: &Router<D>,
    dispatches: &Receiver<CycleDispatch<D>>,
    obituaries: &Receiver<AgentId>,
    mut in_flight: usize,
) -> Result<(), Halt> {
    while in_flight > 0 {
        let dispatch = select! {
            recv(dispatches) -> dispatch => dispatch.map_err(|_| Departure)?,
            recv(obituaries) -> _ => return Err(Departure),
        };
        in_flight = in_flight
            .checked_sub(dispatch.deliveries)
            .expect("an agent reported more deliveries than were routed to it");
        for event in &dispatch.sent {
            in_flight += router.route(event).map_err(|error| {
                Halt::Error(EpisodeError::Route {
                    agent: dispatch.agent.clone(),
                    error,
                })
            })?;
        }
    }
    Ok(())
}

/// An agent's handler, wrapped so that the episode hears when the agent's
/// thread ends.
///
/// The agent loop keeps its handler until it returns, so this is dropped,
/// and the obituary sent, exactly when the thread is ending: by a panic, by
/// an error, or, after shutdown, by the episode joining it and dropping the
/// handler it gets back. Before shutdown an obituary can only mean the
/// first two, and it is what keeps the episode from waiting forever for a
/// cycle that will never be reported.
struct Watched<D: Domain> {
    id: AgentId,
    handler: Box<dyn Handler<D> + Send>,
    obituary: Sender<AgentId>,
}

impl<D: Domain> Handler<D> for Watched<D> {
    fn start(&mut self) -> Vec<Action<D>> {
        self.handler.start()
    }

    fn handle(&mut self, observations: &[Observation<D>], cancel: &Cancel) -> Vec<Action<D>> {
        self.handler.handle(observations, cancel)
    }
}

impl<D: Domain> Drop for Watched<D> {
    fn drop(&mut self) {
        // After shutdown nobody is listening, and that is fine.
        let _ = self.obituary.send(self.id.clone());
    }
}

/// Joins every thread and collects the ones that did not end cleanly.
fn join<D: Domain>(agents: Vec<Agent<Watched<D>>>) -> Vec<(AgentId, Failure)> {
    agents
        .into_iter()
        .filter_map(|agent| {
            let id = agent.id().clone();
            match panic::catch_unwind(AssertUnwindSafe(|| agent.join())) {
                Ok(Ok(_handler)) => None,
                Ok(Err(error)) => Some((id, Failure::Error(error))),
                Err(payload) => Some((id, Failure::Panicked(panic_message(payload.as_ref())))),
            }
        })
        .collect()
}

fn panic_message(payload: &(dyn Any + Send)) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        (*message).to_string()
    } else if let Some(message) = payload.downcast_ref::<String>() {
        message.clone()
    } else {
        "(the panic payload is not a string)".to_string()
    }
}

#[cfg(test)]
mod tests {
    use serde::Serialize;
    use serde_json::Value;

    use super::*;
    use crate::testing::parse_lines;
    use crate::trajectory::Writer;

    /// A count, which is all these agents ever say to each other.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
    #[serde(transparent)]
    struct Count(u64);

    /// The domain of the counting games below.
    struct Counting;

    impl Domain for Counting {
        type Payload = Count;
        type Reward = i32;
    }

    /// The payloads of a batch of observations.
    fn counts(observations: &[Observation<Counting>]) -> Vec<u64> {
        observations
            .iter()
            .map(|observation| observation.event.payload.0)
            .collect()
    }

    /// Volleys a count back and forth with `partner` until it reaches
    /// `limit`, then goes quiet. The one that `serves` opens the rally.
    struct Rally {
        partner: AgentId,
        serves: bool,
        limit: u64,
    }

    impl Rally {
        fn to_partner(&self, n: u64) -> Action<Counting> {
            Action::to([self.partner.clone()], Count(n))
        }
    }

    impl Handler<Counting> for Rally {
        fn start(&mut self) -> Vec<Action<Counting>> {
            if self.serves {
                vec![self.to_partner(1)]
            } else {
                Vec::new()
            }
        }

        fn handle(
            &mut self,
            observations: &[Observation<Counting>],
            _: &Cancel,
        ) -> Vec<Action<Counting>> {
            counts(observations)
                .into_iter()
                .filter(|n| *n < self.limit)
                .map(|n| self.to_partner(n + 1))
                .collect()
        }
    }

    /// Broadcasts once when it starts, and says nothing more.
    struct Hub;

    impl Handler<Counting> for Hub {
        fn start(&mut self) -> Vec<Action<Counting>> {
            vec![Action::broadcast(Count(0))]
        }

        fn handle(&mut self, _: &[Observation<Counting>], _: &Cancel) -> Vec<Action<Counting>> {
            Vec::new()
        }
    }

    /// Replies once to whoever sends it anything.
    struct Spoke;

    impl Handler<Counting> for Spoke {
        fn handle(
            &mut self,
            observations: &[Observation<Counting>],
            _: &Cancel,
        ) -> Vec<Action<Counting>> {
            observations
                .iter()
                .map(|observation| Action::to([observation.event.sender.clone()], Count(1)))
                .collect()
        }
    }

    /// Sends one event to `to` when it starts, whoever that is.
    struct Addresses(&'static str);

    impl Handler<Counting> for Addresses {
        fn start(&mut self) -> Vec<Action<Counting>> {
            vec![Action::to([self.0], Count(1))]
        }

        fn handle(&mut self, _: &[Observation<Counting>], _: &Cancel) -> Vec<Action<Counting>> {
            Vec::new()
        }
    }

    struct Panics;

    impl Handler<Counting> for Panics {
        fn start(&mut self) -> Vec<Action<Counting>> {
            panic!("the handler is broken")
        }

        fn handle(&mut self, _: &[Observation<Counting>], _: &Cancel) -> Vec<Action<Counting>> {
            panic!("the handler is broken")
        }
    }

    fn of<'a>(lines: &'a [Value], agent: &str) -> impl Iterator<Item = &'a Value> {
        lines.iter().filter(move |line| line["agent"] == agent)
    }

    fn rally(limit: u64) -> (Episode<Counting>, Writer<Vec<u8>>) {
        let (records, writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        for (me, partner, serves) in [("a", "b", true), ("b", "a", false)] {
            episode
                .add(
                    me,
                    Rally {
                        partner: AgentId::new(partner),
                        serves,
                        limit,
                    },
                )
                .unwrap();
        }
        (episode, writer)
    }

    #[test]
    fn a_rally_runs_to_quiescence_and_the_writer_flushes() {
        let (episode, writer) = rally(20);
        assert_eq!(
            episode.ids().collect::<Vec<_>>(),
            [&AgentId::new("a"), &AgentId::new("b")]
        );
        episode.run().unwrap();

        // Every sender is gone, so the writer finishes on its own.
        let lines = parse_lines(&writer.join().unwrap());
        // The file interleaves agents in whatever order their records reached
        // the writer; only each agent's own order is promised.
        let sent = |agent: &str| -> Vec<u64> {
            of(&lines, agent)
                .filter(|line| line["type"] == "action")
                .map(|line| line["event"]["payload"].as_u64().unwrap())
                .collect()
        };
        assert_eq!(sent("a"), (1..=19).step_by(2).collect::<Vec<_>>());
        assert_eq!(sent("b"), (2..=20).step_by(2).collect::<Vec<_>>());
        for agent in ["a", "b"] {
            let seqs: Vec<u64> = of(&lines, agent)
                .filter(|line| line["type"] != "cycle")
                .map(|line| line["seq"].as_u64().unwrap())
                .collect();
            assert_eq!(seqs, (0..seqs.len() as u64).collect::<Vec<_>>());
            let controls: Vec<&Value> = of(&lines, agent)
                .filter(|line| line["type"] == "control")
                .map(|line| &line["control"])
                .collect();
            assert_eq!(controls, ["start", "stop"]);
            // No agent here has a timeout, so every cycle was woken by
            // something on the queue and none is empty.
            for line in of(&lines, agent).filter(|line| line["type"] == "cycle") {
                assert_eq!(line["woken"], "queue", "{line}");
                assert!(
                    !line["inputs"].as_array().unwrap().is_empty(),
                    "no cycle popped nothing: {line}"
                );
            }
            // Every action lies within the window of the cycle that sent it,
            // and every observation and control was received at a t_start.
            let starts: BTreeSet<u64> = of(&lines, agent)
                .filter(|line| line["type"] == "cycle")
                .map(|line| line["t_start"].as_u64().unwrap())
                .collect();
            for line in of(&lines, agent).filter(|line| line["type"] != "cycle") {
                if line["type"] == "action" {
                    continue;
                }
                let (created, received) = (
                    line["created"].as_u64().unwrap(),
                    line["received"].as_u64().unwrap(),
                );
                assert!(
                    created <= received,
                    "nothing arrives before it was sent: {line}"
                );
                assert!(
                    starts.contains(&received),
                    "everything popped is popped at a cycle's start: {line}"
                );
            }
        }
    }

    #[test]
    fn an_observation_carries_the_creation_time_of_the_action_that_sent_it() {
        // The join a training pipeline makes: an observation in one agent's
        // trajectory and the action in its sender's are the same event, and
        // nothing but the sender and the creation time links them.
        let (episode, writer) = rally(6);
        episode.run().unwrap();
        let lines = parse_lines(&writer.join().unwrap());

        let actions: BTreeSet<(String, u64)> = lines
            .iter()
            .filter(|line| line["type"] == "action")
            .map(|line| {
                (
                    line["event"]["sender"].as_str().unwrap().to_owned(),
                    line["created"].as_u64().unwrap(),
                )
            })
            .collect();
        let observations: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "observation")
            .collect();
        assert!(!observations.is_empty());
        for line in observations {
            let key = (
                line["event"]["sender"].as_str().unwrap().to_owned(),
                line["created"].as_u64().unwrap(),
            );
            assert!(
                actions.contains(&key),
                "every observation joins an action by sender and creation time: {line}"
            );
        }
    }

    #[test]
    fn in_flight_work_is_counted_in_deliveries_not_messages() {
        let (records, writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("hub", Hub).unwrap();
        for spoke in 1..=6 {
            episode.add(format!("spoke-{spoke}"), Spoke).unwrap();
        }
        episode.run().unwrap();

        // Had the broadcast counted as one delivery, the episode would have
        // gone quiescent after the first spoke's reply, and the hub would
        // not have handled all six.
        let lines = parse_lines(&writer.join().unwrap());
        let replies_seen_by_hub = of(&lines, "hub")
            .filter(|line| line["type"] == "observation")
            .count();
        assert_eq!(replies_seen_by_hub, 6);
        let broadcast = of(&lines, "hub")
            .find(|line| line["type"] == "action")
            .expect("the hub recorded its broadcast");
        assert_eq!(
            broadcast["event"]["recipients"].as_array().unwrap().len(),
            6,
            "a broadcast is recorded as everyone it went to"
        );
    }

    #[test]
    fn an_empty_roster_runs_and_writes_nothing() {
        let (records, writer) = Writer::spawn(Vec::new());
        Episode::<Counting>::new(records).run().unwrap();
        assert!(writer.join().unwrap().is_empty());
    }

    #[test]
    fn a_duplicate_id_is_rejected() {
        let (records, _writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("a", Hub).unwrap();
        assert_eq!(
            episode.add("a", Hub).unwrap_err(),
            EpisodeError::DuplicateAgent(AgentId::new("a"))
        );
        assert_eq!(episode.ids().count(), 1);
    }

    #[test]
    fn a_panicking_handler_fails_the_episode_without_hanging_it() {
        let (records, writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("a", Panics).unwrap();
        episode.add("b", Spoke).unwrap();
        let error = episode.run().unwrap_err();
        assert_eq!(
            error,
            EpisodeError::Agents(vec![(
                AgentId::new("a"),
                Failure::Panicked("the handler is broken".into())
            )])
        );
        assert!(error.to_string().contains("the handler is broken"));
        // The other agent was still stopped cleanly.
        let lines = parse_lines(&writer.join().unwrap());
        assert!(of(&lines, "b").any(|line| line["control"] == "stop"));
    }

    #[test]
    fn a_handler_that_addresses_itself_fails_the_episode() {
        let (records, _writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("a", Addresses("a")).unwrap();
        episode.add("b", Spoke).unwrap();
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Route {
                agent: AgentId::new("a"),
                error: RouteError::Loopback(AgentId::new("a")),
            }
        );
    }

    #[test]
    fn a_handler_that_addresses_a_stranger_fails_the_episode() {
        let (records, _writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("a", Addresses("nobody")).unwrap();
        episode.add("b", Spoke).unwrap();
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Route {
                agent: AgentId::new("a"),
                error: RouteError::UnknownAgent(AgentId::new("nobody")),
            }
        );
    }

    #[test]
    fn a_broadcast_from_a_roster_of_one_has_nobody_to_go_to() {
        let (records, _writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records);
        episode.add("alone", Hub).unwrap();
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Route {
                agent: AgentId::new("alone"),
                error: RouteError::NoRecipients,
            }
        );
    }

    #[test]
    fn a_vanished_writer_fails_the_episode_without_hanging_it() {
        // Nobody is reading the records, so every agent's first record fails
        // to send.
        let (records, nobody) = unbounded();
        let mut episode = Episode::new(records);
        episode.add("a", Spoke).unwrap();
        episode.add("b", Spoke).unwrap();
        drop(nobody);
        let error = episode.run().unwrap_err();
        let EpisodeError::Agents(failures) = error else {
            panic!("unexpected error: {error}");
        };
        assert!(!failures.is_empty());
        assert!(
            failures
                .iter()
                .all(|(_, failure)| *failure == Failure::Error(agent::Error::WriterClosed))
        );
    }

    #[test]
    fn errors_explain_themselves() {
        assert_eq!(
            EpisodeError::DuplicateAgent(AgentId::new("a")).to_string(),
            "agent a is in the roster twice"
        );
        assert_eq!(
            EpisodeError::Control(RouteError::InboxClosed(AgentId::new("a"))).to_string(),
            "could not deliver a control: the inbox of agent a is closed"
        );
        assert_eq!(
            EpisodeError::Route {
                agent: AgentId::new("a"),
                error: RouteError::NoRecipients
            }
            .to_string(),
            "agent a sent an event that could not be routed: an event must have at least one recipient"
        );
        assert_eq!(
            EpisodeError::Agents(vec![
                (
                    AgentId::new("a"),
                    Failure::Error(agent::Error::WriterClosed)
                ),
                (AgentId::new("b"), Failure::Panicked("boom".into())),
            ])
            .to_string(),
            "agents failed: [a: the trajectory writer has gone away] [b: panicked: boom]"
        );
        assert!(
            EpisodeError::Control(RouteError::NoRecipients)
                .source()
                .is_some()
        );
        let (records, _writer) = Writer::<Vec<u8>>::spawn::<Counting>(Vec::new());
        assert!(format!("{:?}", Episode::new(records)).starts_with("Episode"));
    }
}
