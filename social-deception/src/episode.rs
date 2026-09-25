//! An episode: one run of a game with a fixed roster of agents and one
//! [`Environment`], from coordinated startup to coordinated shutdown.
//!
//! The episode constructs the roster, wires each agent's queues, dispatch
//! channel and record channel, spawns one thread per agent, and then hands
//! the episode over: it starts the **environment** and nobody else, and
//! routes what everybody sends until the environment has stopped every
//! agent. A [`Control`] is how anyone says start and stop, and whoever
//! sends one stamps it with the instant it was sent, so that an agent's
//! record of popping it says how long it waited.
//!
//! # The environment controls the episode
//!
//! The environment is an agent like any other, with one power: its handler
//! returns [`Effect`](crate::Effect)s, and an effect may be a control
//! (ADR-0007). So the sequence is:
//!
//! 1. the episode delivers `Start` to the environment alone;
//! 2. the environment's `start` returns `Start` for the agents it wants
//!    playing, along with whatever it opens with;
//! 3. the game runs, the episode routing events and the controls the
//!    environment asks for, the second always after the first of the same
//!    cycle, so that an agent stopped in the same breath as it is spoken to
//!    hears the words first;
//! 4. the environment ends the episode by sending `Stop` to every agent.
//!    Once every agent has been sent one and its thread has ended, the
//!    episode sends `Stop` to the environment and joins it.
//!
//! An agent the environment never starts is spawned, blocked on its queues
//! and silent; it is stopped and joined with everyone else. That is a bug
//! in the environment, not a state the episode has to rescue.
//!
//! # Quiescence, which is now how a stall is detected
//!
//! The episode is the only component that sees both halves of the work in
//! progress: the deliveries it has routed and the cycles the agents have
//! reported. It keeps one count, of deliveries routed and not yet reported
//! handled. A cycle in progress is exactly a set of deliveries taken off a
//! queue and not yet reported, and an agent dispatches a cycle's deliveries
//! and its outputs together, so the count never reads zero while a cycle
//! that might still send is under way.
//!
//! The count reaching zero used to be how an episode ended. Now that the
//! environment ends it, a count of zero while some agent has not been sent
//! `Stop` means nobody will ever speak again and nobody has declared the
//! episode over: the run is **stalled**, and that is
//! [`EpisodeError::Stalled`]. The episode stops everyone and joins, so the
//! trajectory is complete up to the stall, and returns the error. In
//! Werewolf a stall is a player that did not answer a request.
//!
//! In-flight work is counted in **deliveries, not events**: one event
//! addressed to six agents is six handles that have not happened yet.
//!
//! An episode's agents have no timeout yet. The reason they could not have
//! one is gone — an agent that wakes on its own no longer keeps an episode
//! from ending, because quiescence no longer ends it — but giving them one
//! is another issue's.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::panic::{self, AssertUnwindSafe};

use crossbeam_channel::{Receiver, Sender, select, unbounded};

use crate::agent::{self, Action, Agent, CycleDispatch, Handler, Observation, Wiring};
use crate::clock::Clock;
use crate::environment::{Adapter, Commanded, Environment, Rewarded};
use crate::event::{AgentId, Control, Delivery, Domain};
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
///
/// An episode that ends in any of these but [`Stalled`](Self::Stalled) was
/// abandoned rather than finished: the episode stops whoever is left so
/// that the trajectory is complete up to the failure, and it cannot wait
/// for quiescence to do it, so that `Stop` lands behind events the agent
/// has not reached yet. Such a trajectory may therefore end with an agent
/// answering for an episode that had already failed, and with observations
/// nobody made — the events still behind the stop when it was popped
/// (ADR-0009). A `Stalled` episode is quiescent by definition, so its
/// shutdown is orderly.
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
    /// Nothing is in flight and the environment has not stopped these
    /// agents, so nobody will ever speak again and nobody has declared the
    /// episode over. See the [module documentation](self).
    Stalled {
        /// The agents still running when everything went quiet, in order.
        running: BTreeSet<AgentId>,
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
            Self::Stalled { running } => {
                f.write_str("the episode stalled with these agents still running:")?;
                for id in running {
                    write!(f, " {id}")?;
                }
                Ok(())
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
            Self::DuplicateAgent(_) | Self::Stalled { .. } | Self::Agents(_) => None,
        }
    }
}

/// A roster of handlers, an environment, and everything needed to run them
/// once.
///
/// Build it with [`Episode::new`], which takes the environment, add agents
/// with [`Episode::add`], then [`Episode::run`] it. The roster is fixed
/// from the moment `run` starts.
pub struct Episode<D: Domain> {
    roster: BTreeMap<AgentId, Box<dyn Handler<D> + Send>>,
    environment_id: AgentId,
    environment: Box<dyn Environment<D> + Send>,
    records: Sender<LogRecord<D>>,
    clock: Clock,
}

impl<D: Domain> fmt::Debug for Episode<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Episode")
            .field("environment", &self.environment_id)
            .field("roster", &self.roster.keys().collect::<Vec<_>>())
            .finish_non_exhaustive()
    }
}

impl<D: Domain> Episode<D> {
    /// An episode of `environment`, seated under `environment_id`, with no
    /// agents yet, whose trajectory goes to `records` and which is timed by
    /// a [`Clock`] started now.
    #[must_use]
    pub fn new(
        records: Sender<LogRecord<D>>,
        environment_id: impl Into<AgentId>,
        environment: impl Environment<D> + Send + 'static,
    ) -> Self {
        Self::with_clock(records, environment_id, environment, Clock::start())
    }

    /// The same, timed by `clock`.
    #[must_use]
    pub fn with_clock(
        records: Sender<LogRecord<D>>,
        environment_id: impl Into<AgentId>,
        environment: impl Environment<D> + Send + 'static,
        clock: Clock,
    ) -> Self {
        Self {
            roster: BTreeMap::new(),
            environment_id: environment_id.into(),
            environment: Box::new(environment),
            records,
            clock,
        }
    }

    /// Adds an agent to the roster.
    ///
    /// # Errors
    ///
    /// [`EpisodeError::DuplicateAgent`] if `id` is already in the roster,
    /// or is the environment's.
    pub fn add(
        &mut self,
        id: impl Into<AgentId>,
        handler: impl Handler<D> + Send + 'static,
    ) -> Result<(), EpisodeError> {
        let id = id.into();
        if id == self.environment_id || self.roster.contains_key(&id) {
            return Err(EpisodeError::DuplicateAgent(id));
        }
        self.roster.insert(id, Box::new(handler));
        Ok(())
    }

    /// The ids in the roster, the environment's among them, in order.
    pub fn ids(&self) -> impl Iterator<Item = &AgentId> {
        self.roster
            .keys()
            .chain([&self.environment_id])
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    /// The environment's id.
    #[must_use]
    pub const fn environment(&self) -> &AgentId {
        &self.environment_id
    }

    /// Runs the episode to completion: starts the environment, routes what
    /// everybody sends and the controls the environment asks for until every
    /// agent has been stopped, then stops the environment and joins every
    /// thread.
    ///
    /// The episode's own sender to the trajectory writer is dropped on the
    /// way, so once this returns the writer has no senders left and can be
    /// joined for the complete trajectory.
    ///
    /// # Errors
    ///
    /// If a control could not be delivered, an agent sent an event or asked
    /// for a control the router refused, the episode stalled
    /// ([`EpisodeError::Stalled`]), or a thread ended with an error or a
    /// panic. In every case the agents that were running have been stopped
    /// and joined before this returns, so the trajectory is complete up to
    /// the failure.
    pub fn run(self) -> Result<(), EpisodeError> {
        let Self {
            roster,
            environment_id,
            environment,
            records,
            clock,
        } = self;
        let ids: BTreeSet<AgentId> = roster
            .keys()
            .cloned()
            .chain([environment_id.clone()])
            .collect();
        let (dispatch, dispatches) = unbounded();
        let (obituary, obituaries) = unbounded();
        let (asked_for, commands) = unbounded();
        let (paid, rewards) = unbounded();
        let mut handlers: BTreeMap<AgentId, Box<dyn Handler<D> + Send>> = roster;
        handlers.insert(
            environment_id.clone(),
            Box::new(Adapter::new(
                environment,
                ids.iter()
                    .filter(|id| **id != environment_id)
                    .cloned()
                    .collect(),
                asked_for,
                paid,
                records.clone(),
                clock,
            )),
        );
        let Spawned {
            queues,
            held,
            agents,
        } = spawn(handlers, &ids, &dispatch, &obituary, &records, clock);
        drop((dispatch, obituary, records));
        let router = Router::new(queues, environment_id.clone(), clock);

        let environment_only = BTreeSet::from([environment_id.clone()]);
        let mut running = router.agents();
        let seat = Seat {
            id: environment_id.clone(),
            commanded: commands,
            rewarded: rewards,
        };
        let outcome = router
            .control(&environment_only, Control::Start)
            .map_err(|error| Halt::Error(EpisodeError::Control(error)))
            .and_then(|in_flight| {
                drive(
                    &router,
                    &seat,
                    &dispatches,
                    &obituaries,
                    in_flight,
                    &mut running,
                )
            });
        // Whatever happened, everybody is stopped, and nobody twice: an
        // agent the environment already stopped is not in `running`, and a
        // second `Stop` would put a second one in its trajectory, claiming
        // it was told to stop after it had stopped.
        //
        // The environment's own `Stop` comes last, once the agents it was
        // running have ended, because until then the episode is not over
        // for it: a cycle it is still in the middle of may yet have
        // something to say, and its trajectory should say so.
        let (environment_agent, agents) = split(agents, &environment_id);
        let stopped = router
            .control(&running, Control::Stop)
            .map_err(EpisodeError::Control);
        let mut failures = join(agents);
        let stopped = stopped.and_then(|_| {
            router
                .control(&environment_only, Control::Stop)
                .map_err(EpisodeError::Control)
        });
        failures.extend(join(environment_agent));
        failures.sort_by(|(one, _), (other, _)| one.cmp(other));
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

/// The running agents of an episode: its environment's thread and every
/// other.
type Threads<D> = Vec<Agent<Watched<D>>>;

/// What spawning an episode's threads leaves the episode holding.
struct Spawned<D: Domain> {
    /// Where to address each agent, which is what the router is built from.
    queues: BTreeMap<AgentId, Queues<D>>,
    /// A receiving half of every queue, kept alive until every thread has
    /// been joined, so that a message to an agent that has already stopped
    /// is delivered and never read rather than failing its sender.
    held: Vec<Receiver<Delivery<D>>>,
    /// The threads themselves.
    agents: Threads<D>,
}

/// Wires and spawns one thread per handler, in roster order.
fn spawn<D: Domain>(
    handlers: BTreeMap<AgentId, Box<dyn Handler<D> + Send>>,
    ids: &BTreeSet<AgentId>,
    dispatch: &Sender<CycleDispatch<D>>,
    obituary: &Sender<AgentId>,
    records: &Sender<LogRecord<D>>,
    clock: Clock,
) -> Spawned<D> {
    let mut queues = BTreeMap::new();
    let mut held = Vec::with_capacity(ids.len());
    let mut agents = Vec::with_capacity(ids.len());
    for (id, handler) in handlers {
        let (sender, queue) = unbounded();
        held.push(queue.clone());
        queues.insert(id.clone(), Queues { queue: sender });
        let wiring = Wiring {
            id: id.clone(),
            clock,
            queue,
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
    Spawned {
        queues,
        held,
        agents,
    }
}

/// Splits the environment's agent out of the roster's, so that the two can
/// be stopped and joined in their own order.
fn split<D: Domain>(agents: Threads<D>, environment: &AgentId) -> (Threads<D>, Threads<D>) {
    agents
        .into_iter()
        .partition(|agent| agent.id() == environment)
}

/// The environment's seat in the episode: who it is, and the two channels
/// it asks for things on.
///
/// The three travel together because they are one thing — what the episode
/// knows about its environment that it knows about nobody else — and
/// because a control and a reward are both drained at the same moment, just
/// after a dispatch of the environment's.
struct Seat {
    /// The environment's id.
    id: AgentId,
    /// The controls it has asked for and the episode has not yet issued.
    commanded: Receiver<Commanded>,
    /// The agents it has rewarded and the episode has not yet checked. The
    /// records are already written; see
    /// [`Adapter`](crate::environment::Adapter).
    rewarded: Receiver<Rewarded>,
}

/// Why routing stopped.
enum Halt {
    /// An agent's thread has ended, or every agent's has; joining them says
    /// why.
    Departure,
    /// A control event or a message could not be routed, or the episode
    /// stalled.
    Error(EpisodeError),
}

use Halt::Departure;

/// Routes what the agents send, and the controls the environment asks for,
/// until every agent has been stopped.
///
/// `in_flight` is the number of deliveries already made and not yet reported
/// handled. Each dispatch takes a cycle's deliveries off the count and puts
/// the deliveries its outputs cause onto it, in that order but as one step,
/// so the count reads zero only when no agent has anything left to handle or
/// send.
///
/// # When a control the environment asked for is issued
///
/// The environment queues a cycle's controls while its handler is running,
/// which is strictly before its loop sends the dispatch, so by the time a
/// dispatch of the environment's is in hand every control of that cycle is
/// already on `commanded` waiting to be drained (see
/// [`environment::Adapter`](crate::environment::Adapter)). What is left is
/// *when* to issue each, and the two controls want opposite answers.
///
/// A [`Start`](Control::Start) is issued **before** the cycle's events, so
/// that an agent logs its start before its first observation. The
/// environment that starts an agent and speaks to it in the same breath
/// means the start first.
///
/// A [`Stop`](Control::Stop) is issued **once nothing is in flight**, which
/// is to say once everything already said has been handled. It has to be,
/// and not merely after the cycle's events. An agent has one queue, so a
/// `Stop` sent early either waits behind work the agent has not reached or,
/// once popped, leaves the rest of that queue unobserved — an agent that
/// has stopped did not observe it (see [`agent`](crate::agent)) — and which
/// of the two depends on the scheduler. Holding the stop back until the
/// count reads zero means there is nothing for it to land behind, which
/// makes "an agent hears everything said to it before it is told to stop" a
/// guarantee rather than a hope, and it is what lets the moderator narrate
/// an outcome and end the episode in one cycle.
///
/// Nothing in flight, no stop waiting to be issued, and some agent still
/// running is a stall; see the [module documentation](self).
fn drive<D: Domain>(
    router: &Router<D>,
    environment: &Seat,
    dispatches: &Receiver<CycleDispatch<D>>,
    obituaries: &Receiver<AgentId>,
    mut in_flight: usize,
    running: &mut BTreeSet<AgentId>,
) -> Result<(), Halt> {
    let Seat {
        id: environment,
        commanded,
        rewarded,
    } = environment;
    let mut held: Vec<BTreeSet<AgentId>> = Vec::new();
    while !running.is_empty() {
        if in_flight == 0 {
            if held.is_empty() {
                return Err(Halt::Error(EpisodeError::Stalled {
                    running: running.clone(),
                }));
            }
            for to in held.drain(..) {
                in_flight += router
                    .command(environment, &to, Control::Stop)
                    .map_err(|error| {
                        Halt::Error(EpisodeError::Route {
                            agent: environment.clone(),
                            error,
                        })
                    })?;
                running.retain(|id| !to.contains(id));
            }
            continue;
        }
        let dispatch = select! {
            recv(dispatches) -> dispatch => dispatch.map_err(|_| Departure)?,
            recv(obituaries) -> _ => return Err(Departure),
        };
        in_flight = in_flight
            .checked_sub(dispatch.deliveries)
            .expect("an agent reported more deliveries than were routed to it");
        let refused = |error| {
            Halt::Error(EpisodeError::Route {
                agent: dispatch.agent.clone(),
                error,
            })
        };
        if dispatch.agent == *environment {
            // The rewards of this cycle, checked but not routed: a reward
            // is already in the trajectory and is not a delivery, so
            // nothing here adds to the in-flight count. What is left is
            // whether the environment named an agent it could reward.
            // Only a refusal arrives here: the adapter writes a reward it
            // accepts and says nothing. What comes is a name it would not
            // write, which the router turns into the error it would give
            // for addressing that name.
            while let Ok(Rewarded { agent }) = rewarded.try_recv() {
                router.rewardable(environment, &agent).map_err(refused)?;
            }
            while let Ok(Commanded { to, control }) = commanded.try_recv() {
                match control {
                    Control::Start => {
                        in_flight += router
                            .command(&dispatch.agent, &to, control)
                            .map_err(refused)?;
                    }
                    // Validated now, so that a stop addressed to a stranger
                    // fails the episode where it was asked for rather than
                    // once everything has gone quiet.
                    Control::Stop => {
                        router
                            .validate(&dispatch.agent, &to)
                            .map_err(refused)
                            .map(|()| held.push(to))?;
                    }
                }
            }
        }
        for event in &dispatch.sent {
            in_flight += router.route(event).map_err(refused)?;
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

    fn handle(&mut self, observation: &Observation<D>) -> Vec<Action<D>> {
        self.handler.handle(observation)
    }

    fn timeout(&mut self) -> Vec<Action<D>> {
        self.handler.timeout()
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
    use crate::environment::Effect;
    use crate::testing::parse_lines;
    use crate::trajectory::Writer;

    /// What the agents of the counting games below say.
    ///
    /// `Done` is how an agent tells the environment it has nothing more to
    /// send. Quiescence is no longer how an episode ends, so a purely
    /// reactive roster needs some way to say that it has finished, and
    /// saying so in the domain is the way a game does it; Collatz's
    /// `Finished` is the same idea.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
    enum Count {
        /// A number, which is all these agents ever say to each other.
        Say(u64),
        /// Nothing more is coming from this agent.
        Done,
    }

    use Count::{Done, Say};

    /// The domain of the counting games below.
    struct Counting;

    impl Domain for Counting {
        type Payload = Count;
        type Reward = i32;
    }

    /// The environment of the counting games: it starts the agents it was
    /// built with and stops them all once each has said [`Done`].
    ///
    /// It is the whole of what an environment must do — start the episode
    /// and end it — and nothing else, which is what makes it the stand-in
    /// for a game the runtime knows nothing about.
    struct Referee {
        agents: BTreeSet<AgentId>,
        working: BTreeSet<AgentId>,
    }

    impl Referee {
        /// A referee over these agents.
        fn over<const N: usize>(agents: [&str; N]) -> Self {
            let agents: BTreeSet<AgentId> = agents.map(AgentId::new).into();
            Self {
                working: agents.clone(),
                agents,
            }
        }
    }

    impl Environment<Counting> for Referee {
        fn start(&mut self) -> Vec<Effect<Counting>> {
            vec![Effect::control(self.agents.clone(), Control::Start)]
        }

        fn handle(&mut self, observation: &Observation<Counting>) -> Vec<Effect<Counting>> {
            if observation.event.payload == Done {
                self.working.remove(&observation.event.sender);
            }
            if self.working.is_empty() && !self.agents.is_empty() {
                let agents = std::mem::take(&mut self.agents);
                vec![Effect::control(agents, Control::Stop)]
            } else {
                Vec::new()
            }
        }
    }

    /// An environment that starts two agents, pays `to` whatever `value`
    /// is, and stops everybody, all in its opening cycle.
    ///
    /// It is the smallest thing that logs a reward: enough to see the
    /// record the runtime writes, and to point `to` at somebody who is not
    /// there.
    struct Paymaster {
        to: AgentId,
        value: i32,
    }

    impl Paymaster {
        fn paying(to: &str, value: i32) -> Self {
            Self {
                to: AgentId::new(to),
                value,
            }
        }
    }

    impl Environment<Counting> for Paymaster {
        fn start(&mut self) -> Vec<Effect<Counting>> {
            vec![
                Effect::control(["a", "b"], Control::Start),
                Effect::reward(self.to.clone(), self.value),
                Effect::control(["a", "b"], Control::Stop),
            ]
        }

        fn handle(&mut self, _: &Observation<Counting>) -> Vec<Effect<Counting>> {
            Vec::new()
        }
    }

    /// An episode of a [`Paymaster`] over two mute agents, and its writer.
    fn paid(to: &str, value: i32) -> (Episode<Counting>, Writer<Vec<u8>>) {
        let (records, writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records, REFEREE, Paymaster::paying(to, value));
        episode.add("a", Mute).unwrap();
        episode.add("b", Mute).unwrap();
        (episode, writer)
    }

    #[test]
    fn a_reward_is_logged_to_the_agent_it_names_and_routed_nowhere() {
        let (episode, writer) = paid("a", 7);
        episode.run().unwrap();
        let lines = parse_lines(&writer.join().unwrap());
        let rewards: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .collect();
        assert_eq!(rewards.len(), 1, "{lines:?}");
        let reward = rewards[0];
        assert_eq!(reward["agent"], "a", "the agent rewarded, not the payer");
        assert_eq!(reward["value"], 7);
        assert!(
            reward["seq"].is_null(),
            "a reward carries no sequence number: {reward}"
        );
        assert!(
            reward["received"].is_null(),
            "a reward is logged, never sent: {reward}"
        );
        assert!(reward["created"].is_u64(), "{reward}");
        // Nobody observed it: a reward is not a delivery, so it never
        // touched the in-flight count and never reached a queue.
        assert_eq!(
            lines
                .iter()
                .filter(|line| line["type"] == "observation")
                .count(),
            0
        );
        // And it was logged before the stop it precedes.
        let stop = lines
            .iter()
            .find(|line| {
                line["agent"] == "a" && line["type"] == "control" && line["control"] == "stop"
            })
            .expect("a was stopped");
        assert!(reward["created"].as_u64() <= stop["created"].as_u64());
    }

    #[test]
    fn a_reward_for_an_agent_not_in_the_roster_fails_the_episode() {
        // The router rejects it the way it rejects a control addressed to
        // a stranger: the episode stops and says whose bug it is.
        let (episode, _writer) = paid("nobody", 1);
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Route {
                agent: AgentId::new(REFEREE),
                error: RouteError::UnknownAgent(AgentId::new("nobody")),
            }
        );
    }

    #[test]
    fn an_environment_that_rewards_itself_fails_the_episode() {
        // The environment runs the game rather than playing it, so there
        // is nothing its own behavior could be worth.
        let (episode, _writer) = paid(REFEREE, 1);
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Route {
                agent: AgentId::new(REFEREE),
                error: RouteError::Loopback(AgentId::new(REFEREE)),
            }
        );
    }

    /// An environment that starts its agents and never stops them, so that
    /// whatever they do the episode ends by stalling.
    struct Absent<const N: usize>([&'static str; N]);

    impl<const N: usize> Environment<Counting> for Absent<N> {
        fn start(&mut self) -> Vec<Effect<Counting>> {
            vec![Effect::control(self.0, Control::Start)]
        }

        fn handle(&mut self, _: &Observation<Counting>) -> Vec<Effect<Counting>> {
            Vec::new()
        }
    }

    /// The name the counting games' environment goes by.
    const REFEREE: &str = "referee";

    /// The number an observation carries, if it is one: everything a
    /// counting agent hears that is not a [`Done`].
    fn count(observation: &Observation<Counting>) -> Option<u64> {
        match observation.event.payload {
            Say(n) => Some(n),
            Done => None,
        }
    }

    /// An agent's way of telling the environment it has finished.
    fn done() -> Action<Counting> {
        Action::to([REFEREE], Done)
    }

    /// Volleys a count back and forth with `partner` until it reaches
    /// `limit`, then tells the environment it is done. The one that
    /// `serves` opens the rally.
    ///
    /// Each side is done when it has seen the limit, whether it volleyed it
    /// or was volleyed it: the side that sends the limit hears nothing back,
    /// so waiting to be spoken to again would leave it running forever.
    struct Rally {
        partner: AgentId,
        serves: bool,
        limit: u64,
    }

    impl Rally {
        fn to_partner(&self, n: u64) -> Action<Counting> {
            Action::to([self.partner.clone()], Say(n))
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

        fn handle(&mut self, observation: &Observation<Counting>) -> Vec<Action<Counting>> {
            let Some(heard) = count(observation) else {
                return Vec::new();
            };
            let mut actions = Vec::new();
            if heard < self.limit {
                actions.push(self.to_partner(heard + 1));
            }
            if heard + 1 >= self.limit {
                actions.push(done());
            }
            actions
        }
    }

    /// Broadcasts once when it starts, then waits to hear from everybody it
    /// expects before declaring itself done.
    struct Hub {
        expects: usize,
        heard: usize,
    }

    impl Hub {
        const fn expecting(expects: usize) -> Self {
            Self { expects, heard: 0 }
        }
    }

    impl Handler<Counting> for Hub {
        fn start(&mut self) -> Vec<Action<Counting>> {
            vec![Action::broadcast(Say(0))]
        }

        fn handle(&mut self, observation: &Observation<Counting>) -> Vec<Action<Counting>> {
            if count(observation).is_some() {
                self.heard += 1;
            }
            if self.heard >= self.expects {
                vec![done()]
            } else {
                Vec::new()
            }
        }
    }

    /// Replies once to whoever sends it a number, and is then done.
    struct Spoke;

    impl Handler<Counting> for Spoke {
        fn handle(&mut self, observation: &Observation<Counting>) -> Vec<Action<Counting>> {
            if count(observation).is_none() {
                return Vec::new();
            }
            vec![
                Action::to([observation.event.sender.clone()], Say(1)),
                done(),
            ]
        }
    }

    /// Sends one event to `to` when it starts, whoever that is.
    struct Addresses(&'static str);

    impl Handler<Counting> for Addresses {
        fn start(&mut self) -> Vec<Action<Counting>> {
            vec![Action::to([self.0], Say(1))]
        }

        fn handle(&mut self, _: &Observation<Counting>) -> Vec<Action<Counting>> {
            Vec::new()
        }
    }

    /// Says nothing, ever, so an episode of it stalls.
    struct Mute;

    impl Handler<Counting> for Mute {
        fn handle(&mut self, _: &Observation<Counting>) -> Vec<Action<Counting>> {
            Vec::new()
        }
    }

    struct Panics;

    impl Handler<Counting> for Panics {
        fn start(&mut self) -> Vec<Action<Counting>> {
            panic!("the handler is broken")
        }

        fn handle(&mut self, _: &Observation<Counting>) -> Vec<Action<Counting>> {
            panic!("the handler is broken")
        }
    }

    fn of<'a>(lines: &'a [Value], agent: &str) -> impl Iterator<Item = &'a Value> {
        lines.iter().filter(move |line| line["agent"] == agent)
    }

    /// The controls in `lines`, in order. With an `agent`, that agent's;
    /// without one, the whole trajectory's. Every agent's reads
    /// `["start", "stop"]`: the environment starts it and stops it, and
    /// nothing else is a control.
    fn controls_of<'a>(lines: &'a [Value], agent: Option<&str>) -> Vec<&'a Value> {
        lines
            .iter()
            .filter(|line| agent.is_none_or(|agent| line["agent"] == agent))
            .filter(|line| line["type"] == "control")
            .map(|line| &line["control"])
            .collect()
    }

    /// An episode refereed over the named agents, whose trajectory goes to
    /// the writer returned beside it.
    fn refereed<const N: usize>(agents: [&str; N]) -> (Episode<Counting>, Writer<Vec<u8>>) {
        let (records, writer) = Writer::spawn(Vec::new());
        (
            Episode::new(records, REFEREE, Referee::over(agents)),
            writer,
        )
    }

    fn rally(limit: u64) -> (Episode<Counting>, Writer<Vec<u8>>) {
        let (mut episode, writer) = refereed(["a", "b"]);
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
    fn a_rally_runs_to_its_limit_and_the_writer_flushes() {
        let (episode, writer) = rally(20);
        assert_eq!(
            episode.ids().collect::<Vec<_>>(),
            [
                &AgentId::new("a"),
                &AgentId::new("b"),
                &AgentId::new(REFEREE)
            ]
        );
        assert_eq!(episode.environment(), &AgentId::new(REFEREE));
        episode.run().unwrap();

        // Every sender is gone, so the writer finishes on its own.
        let lines = parse_lines(&writer.join().unwrap());
        // The file interleaves agents in whatever order their records reached
        // the writer; only each agent's own order is promised.
        let volleyed = |agent: &str| -> Vec<u64> {
            of(&lines, agent)
                .filter(|line| line["type"] == "action")
                .filter_map(|line| line["event"]["payload"]["Say"].as_u64())
                .collect()
        };
        assert_eq!(volleyed("a"), (1..=19).step_by(2).collect::<Vec<_>>());
        assert_eq!(volleyed("b"), (2..=20).step_by(2).collect::<Vec<_>>());
        for agent in ["a", "b", REFEREE] {
            let seqs: Vec<u64> = of(&lines, agent)
                .filter(|line| line["type"] != "cycle")
                .map(|line| line["seq"].as_u64().unwrap())
                .collect();
            assert_eq!(seqs, (0..seqs.len() as u64).collect::<Vec<_>>());
            let controls = controls_of(&lines, Some(agent));
            assert_eq!(
                controls,
                ["start", "stop"],
                "every trajectory, the environment's included, begins with a start and \
                 ends with a stop"
            );
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
            // and every observation and control was received at a t_start,
            // with no exception: everything a cycle pops, it pops at its
            // start (ADR-0009).
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
                if line["control"] == "stop" {
                    continue;
                }
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
        let spokes: Vec<String> = (1..=6).map(|spoke| format!("spoke-{spoke}")).collect();
        let (records, writer) = Writer::spawn(Vec::new());
        let roster: BTreeSet<AgentId> = spokes
            .iter()
            .map(AgentId::new)
            .chain([AgentId::new("hub")])
            .collect();
        let mut episode = Episode::new(
            records,
            REFEREE,
            Referee {
                working: roster.clone(),
                agents: roster,
            },
        );
        episode.add("hub", Hub::expecting(6)).unwrap();
        for spoke in &spokes {
            episode.add(spoke.clone(), Spoke).unwrap();
        }
        episode.run().unwrap();

        // Had the broadcast counted as one delivery, the count could have
        // reached zero after the first spoke's reply, and the episode would
        // have called a run in progress a stall.
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
            7,
            "a broadcast is recorded as everyone it went to: the six spokes and \
             the environment"
        );
    }

    #[test]
    fn an_environment_with_no_agents_runs_and_writes_only_its_own_controls() {
        let (records, writer) = Writer::spawn(Vec::new());
        Episode::new(records, REFEREE, Referee::over([]))
            .run()
            .unwrap();
        let lines = parse_lines(&writer.join().unwrap());
        let controls = controls_of(&lines, None);
        assert_eq!(controls, ["start", "stop"]);
        assert!(lines.iter().all(|line| line["agent"] == REFEREE));
    }

    #[test]
    fn an_episode_whose_environment_never_stops_anyone_stalls() {
        // Nothing is in flight and no agent has been stopped: nobody will
        // ever speak again and nobody has declared the episode over.
        let (records, writer) = Writer::spawn(Vec::new());
        let mut episode = Episode::new(records, REFEREE, Absent(["a", "b"]));
        episode.add("a", Mute).unwrap();
        episode.add("b", Mute).unwrap();
        assert_eq!(
            episode.run().unwrap_err(),
            EpisodeError::Stalled {
                running: [AgentId::new("a"), AgentId::new("b")].into()
            }
        );
        // The trajectory is complete up to the stall: everybody was started,
        // and everybody, the environment last, was stopped and joined.
        let lines = parse_lines(&writer.join().unwrap());
        for agent in ["a", "b", REFEREE] {
            let controls = controls_of(&lines, Some(agent));
            assert_eq!(controls, ["start", "stop"], "{agent}");
        }
    }

    #[test]
    fn a_duplicate_id_is_rejected() {
        let (mut episode, _writer) = refereed(["a"]);
        episode.add("a", Spoke).unwrap();
        assert_eq!(
            episode.add("a", Spoke).unwrap_err(),
            EpisodeError::DuplicateAgent(AgentId::new("a"))
        );
        assert_eq!(
            episode.add(REFEREE, Spoke).unwrap_err(),
            EpisodeError::DuplicateAgent(AgentId::new(REFEREE)),
            "the environment is in the roster, under its own id"
        );
        assert_eq!(episode.ids().count(), 2);
    }

    #[test]
    fn a_panicking_handler_fails_the_episode_without_hanging_it() {
        let (mut episode, writer) = refereed(["a", "b"]);
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
        // The other agent, and the environment, were still stopped cleanly.
        let lines = parse_lines(&writer.join().unwrap());
        for agent in ["b", REFEREE] {
            assert!(
                of(&lines, agent).any(|line| line["control"] == "stop"),
                "{agent}"
            );
        }
    }

    #[test]
    fn a_handler_that_addresses_itself_fails_the_episode() {
        let (mut episode, _writer) = refereed(["a", "b"]);
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
        let (mut episode, _writer) = refereed(["a", "b"]);
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
    fn a_vanished_writer_fails_the_episode_without_hanging_it() {
        // Nobody is reading the records, so every agent's first record fails
        // to send.
        let (records, nobody) = unbounded();
        let mut episode = Episode::new(records, REFEREE, Referee::over(["a", "b"]));
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
            EpisodeError::Control(RouteError::QueueClosed(AgentId::new("a"))).to_string(),
            "could not deliver a control: the queue of agent a is closed"
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
            EpisodeError::Stalled {
                running: [AgentId::new("a"), AgentId::new("b")].into()
            }
            .to_string(),
            "the episode stalled with these agents still running: a b"
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
        assert!(
            EpisodeError::Stalled {
                running: BTreeSet::new()
            }
            .source()
            .is_none()
        );
        let (records, _writer) = Writer::<Vec<u8>>::spawn::<Counting>(Vec::new());
        assert!(
            format!("{:?}", Episode::new(records, REFEREE, Referee::over([])))
                .starts_with("Episode")
        );
    }
}
