//! The episode: the owner of one run.
//!
//! It owns the wiring, the threads, the log writer, a hard time limit, and
//! turning panics into errors. **Routing is not its business any more**: a
//! handler thread sends straight to its recipients' inboxes, so there is no
//! central loop and no count of deliveries in flight (ADR-0016).
//!
//! # One clock, and it cannot be two
//!
//! An episode's log and its actors must share one origin, or record offsets
//! refer to two different zeroes. Here that is **enforced by construction**,
//! two ways:
//!
//! - [`Episode::new`] takes the log's [`Sink`]s rather than a
//!   a [`Writer`] and a [`Clock`]. It starts the clock
//!   itself, spawns the writer with it, and hands a copy of the same clock to
//!   every actor's `start` hook. **There is no clock argument to get wrong.**
//! - The two functions that start an actor's threads are crate-private, so an
//!   episode is the only thing that can start one. A caller cannot assemble a
//!   roster of its own around a second origin.
//!
//! The order is the one ADR-0017 asks for: capture the clock, build every inbox
//! and control channel, build the [`Router`] from the senders, start the
//! writer, and only then start any actor, each with a copy of the same clock.
//!
//! [`Episode::clock`] is what makes the invariant testable from outside. It is
//! the one reason anything but the episode needs to see a clock at all.
//!
//! # Start and end
//!
//! The episode starts **only the environment**. The environment starts the
//! agents with a [`Command`](crate::Effect::Command) of [`Control::Start`],
//! and it **ends the episode by stopping every actor, itself included**.
//!
//! Every actor's threads report on a completion channel when they end, and
//! [`Episode::run`] waits on it with a hard time limit:
//!
//! - every actor reports after being sent `Stop`: join, and return `Ok`;
//! - the limit passes first: the episode sends `Stop` to every actor itself,
//!   since it holds every control sender, joins, and returns
//!   [`EpisodeError::Timeout`];
//! - an actor ends without having been sent `Stop`: stop the rest, join, and
//!   return [`EpisodeError::Departed`].
//!
//! Stall detection by quiescence is gone, because it depended on the episode
//! seeing every delivery. Under timed phases a stall can only be an
//! environment bug, and an episode that runs past its limit fails with a
//! timeout rather than hanging.
//!
//! # A roster can mix handler types
//!
//! [`Policy`] and [`Step`] return `impl IntoIterator` and so cannot be used as
//! `dyn`. So [`Episode::add`] **erases the handler's type** into a boxed
//! closure that, given what every actor shares, starts that actor, and `run`
//! calls each
//! closure. An LLM player and scripted players can share an episode without
//! generics reaching the episode.

use std::any::Any;
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::io;
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, unbounded};
use serde::Serialize;

use crate::clock::Clock;
use crate::contract::{Policy, Step};
use crate::log::{Policy as SinkPolicy, Record, Sink, Sinks, Writer};
use crate::message::{ActorId, Control, Payload};
use crate::router::{RouteError, Router, Seat};
use crate::thread::{self, Actor, ActorError, Ended, UnstartedActor};
use crate::timer::Timer;

/// Why an actor's threads did not end cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Failure {
    /// A thread stopped because something it depends on refused it or went
    /// away.
    Error(ActorError),
    /// A thread panicked, in the handler or elsewhere, with this message.
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
    /// An id was added to the roster twice, or is the environment's.
    DuplicateAgent(ActorId),
    /// A control the episode itself sent could not be delivered.
    Control(RouteError),
    /// The environment never stopped everybody within the episode's time
    /// limit.
    ///
    /// This replaces the old runtime's `Stalled`. The episode no longer sees
    /// what is in flight, so it cannot tell a quiet roster from a busy one;
    /// what it can do is refuse to run forever.
    Timeout {
        /// How long the episode was given.
        limit: Duration,
        /// The actors that had not reported when it ran out, in order.
        running: BTreeSet<ActorId>,
    },
    /// An actor's threads ended without having been sent a `Stop`.
    ///
    /// The episode was abandoned where it stood. Whether the thread that went
    /// away left a [`Failure`] behind is a separate question:
    /// [`Agents`](Self::Agents) reports that when it did, and this reports the
    /// departure when it did not.
    Departed,
    /// Some actors' threads did not end cleanly, and why.
    ///
    /// A send the router refused is one of these: it is a bug in the sending
    /// actor, so it fails that actor's thread and surfaces here.
    Agents(Vec<(ActorId, Failure)>),
}

impl fmt::Display for EpisodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateAgent(id) => write!(f, "actor {id} is in the roster twice"),
            Self::Control(error) => write!(f, "could not deliver a control: {error}"),
            Self::Timeout { limit, running } => {
                write!(
                    f,
                    "the episode ran past its limit of {limit:?} with these actors still running:"
                )?;
                for id in running {
                    write!(f, " {id}")?;
                }
                Ok(())
            }
            Self::Departed => f.write_str("an actor left while the episode was still running it"),
            Self::Agents(failures) => {
                f.write_str("actors failed:")?;
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
            Self::Control(error) => Some(error),
            Self::DuplicateAgent(_) | Self::Timeout { .. } | Self::Departed | Self::Agents(_) => {
                None
            }
        }
    }
}

/// What starting one actor takes, once the inboxes and the router exist: the
/// router, where records go, the episode's clock, and where the actor reports
/// when it ends.
///
/// It is the episode's alone, and every actor gets the same one: it exists so
/// that the closures below take one argument rather than four.
struct Shared<P: Payload> {
    router: Arc<Router<P>>,
    records: Sender<Record<P>>,
    clock: Clock,
    done: Sender<Ended>,
}

/// A handler whose type has been erased: a closure that, given what every
/// actor shares, starts that actor.
///
/// This is what lets one roster mix handler types. It returns a
/// [`Started`], which is the actor with its `join` erased too, since
/// the handler types differ and the episode does not want them back.
type Start<P> = Box<dyn FnOnce(UnstartedActor<P>, &Shared<P>) -> Started + Send>;

/// A running actor as the episode holds it: its id, its control sender, and a
/// closure that joins its threads.
///
/// The handler is dropped inside the closure rather than given back, because
/// the handler types in one roster differ and the episode has nothing to do
/// with any of them. An application that wants a handler back keeps its state
/// somewhere it can reach, which is what a channel or a shared counter is for.
struct Started {
    id: ActorId,
    control: Sender<Control>,
    join: Box<dyn FnOnce() -> Result<(), ActorError> + Send>,
}

/// A roster of handlers, an environment, and everything needed to run them
/// once.
///
/// Build it with [`Episode::new`], which takes the environment and the log's
/// sinks, add agents with [`Episode::add`], then [`Episode::run`] it. The
/// roster is fixed from the moment `run` starts.
pub struct Episode<W, P: Payload> {
    roster: BTreeMap<ActorId, Start<P>>,
    environment_id: ActorId,
    environment: Option<Start<P>>,
    clock: Clock,
    records: Sender<Record<P>>,
    writer: Writer,
    limit: Duration,
    /// The reward type is the environment's alone, and the environment has
    /// already been erased into a closure by the time it is stored, so nothing
    /// left in this struct mentions `W`.
    reward: std::marker::PhantomData<fn() -> W>,
}

impl<W, P: Payload> fmt::Debug for Episode<W, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Episode")
            .field("environment", &self.environment_id)
            .field("roster", &self.roster.keys().collect::<Vec<_>>())
            .field("limit", &self.limit)
            .finish_non_exhaustive()
    }
}

/// How long an episode runs before it stops everybody itself.
///
/// Generous, because the limit is a backstop against an environment that never
/// ends its episode and not a schedule anything is meant to meet: a handler
/// making model calls may legitimately take minutes. A run with a schedule of
/// its own sets its own with [`Episode::within`].
pub const DEFAULT_LIMIT: Duration = Duration::from_secs(600);

impl<W: Serialize + Send + 'static, P: Payload> Episode<W, P> {
    /// An episode of `environment`, seated under `environment_id`, with no
    /// agents yet, whose log goes to `sinks`.
    ///
    /// **The clock is started here**, before the writer and before any actor,
    /// and every one of them is given a copy of it. That is the whole of the
    /// shared-origin invariant, and it is not something a caller can get
    /// wrong: there is no clock argument.
    pub fn new(
        sinks: Sinks<P>,
        environment_id: impl Into<ActorId>,
        environment: impl Step<W, P> + Send + 'static,
    ) -> Self {
        // One origin for the episode, captured before anything that measures
        // time (ADR-0017).
        let clock = Clock::start();
        let (records, writer) = Writer::spawn(sinks, clock);
        let environment_id = environment_id.into();
        Self {
            roster: BTreeMap::new(),
            environment_id,
            environment: Some(Box::new(|unstarted, shared| {
                started(thread::spawn_environment::<W, _, _>(
                    unstarted,
                    environment,
                    Arc::clone(&shared.router),
                    shared.records.clone(),
                    shared.clock,
                    Timer::real(),
                    shared.done.clone(),
                ))
            })),
            clock,
            records,
            writer,
            limit: DEFAULT_LIMIT,
            reward: std::marker::PhantomData,
        }
    }

    /// An episode whose log is one JSON Lines file, created (or truncated) at
    /// `path`, as a run whose log is that file requires.
    ///
    /// # Errors
    ///
    /// Whatever `File::create` returns.
    pub fn to_file(
        path: impl AsRef<std::path::Path>,
        environment_id: impl Into<ActorId>,
        environment: impl Step<W, P> + Send + 'static,
    ) -> io::Result<Self> {
        let sink: Box<dyn Sink<P>> =
            Box::new(crate::log::JsonLines::new(std::fs::File::create(path)?));
        Ok(Self::new(
            vec![(sink, SinkPolicy::Required)],
            environment_id,
            environment,
        ))
    }

    /// The same episode with a time limit of `limit`.
    ///
    /// Past it the episode stops every actor itself and fails with
    /// [`EpisodeError::Timeout`]. See [`DEFAULT_LIMIT`].
    #[must_use]
    pub const fn within(mut self, limit: Duration) -> Self {
        self.limit = limit;
        self
    }

    /// Adds an agent to the roster, erasing its handler's type.
    ///
    /// # Errors
    ///
    /// [`EpisodeError::DuplicateAgent`] if `id` is already in the roster, or is
    /// the environment's.
    pub fn add<H: Policy<P> + Send + 'static>(
        &mut self,
        id: impl Into<ActorId>,
        handler: H,
    ) -> Result<(), EpisodeError> {
        let id = id.into();
        if id == self.environment_id || self.roster.contains_key(&id) {
            return Err(EpisodeError::DuplicateAgent(id));
        }
        self.roster.insert(
            id,
            Box::new(|unstarted, shared| {
                started(thread::spawn_agent(
                    unstarted,
                    handler,
                    Arc::clone(&shared.router),
                    shared.records.clone(),
                    shared.clock,
                    Timer::real(),
                    shared.done.clone(),
                ))
            }),
        );
        Ok(())
    }

    /// The ids in the roster, the environment's among them, in order.
    pub fn ids(&self) -> impl Iterator<Item = &ActorId> {
        self.roster
            .keys()
            .chain([&self.environment_id])
            .collect::<BTreeSet<_>>()
            .into_iter()
    }

    /// The environment's id.
    #[must_use]
    pub const fn environment(&self) -> &ActorId {
        &self.environment_id
    }

    /// The episode's clock: the origin every actor in it shares with the log
    /// writer.
    ///
    /// It is here so that the shared-origin invariant is testable from
    /// outside, which is the only reason anything but the episode needs to see
    /// a clock.
    #[must_use]
    pub const fn clock(&self) -> Clock {
        self.clock
    }

    /// The episode's time limit.
    #[must_use]
    pub const fn limit(&self) -> Duration {
        self.limit
    }

    /// Runs the episode: starts the environment, waits for every actor to
    /// report, joins every thread, and finishes the log.
    ///
    /// # Errors
    ///
    /// [`EpisodeError::Control`] if the environment could not be started,
    /// [`EpisodeError::Timeout`] if the environment never stopped everybody,
    /// [`EpisodeError::Departed`] if an actor left unbidden, and
    /// [`EpisodeError::Agents`] if a thread ended with an error or a panic. In
    /// every case every actor has been stopped and joined before this returns,
    /// so the log is complete up to the failure.
    ///
    /// The log's writer is joined here too, and an error from a required sink
    /// surfaces as the actors' [`ActorError::WriterClosed`].
    ///
    /// # Panics
    ///
    /// If a thread panicked, the panic is propagated once every other thread
    /// has been joined, so that a panicking handler does not leave an episode
    /// half torn down.
    pub fn run(self) -> Result<(), EpisodeError> {
        let Self {
            roster,
            environment_id,
            environment,
            clock,
            records,
            writer,
            limit,
            reward: _,
        } = self;
        // An episode keeps its environment until it runs, and this is the only
        // thing that takes it, so the option is `Some` here by construction.
        // It is an option at all because the closure has to move out of the
        // struct the rest of this destructuring is still using.
        let environment = environment.expect("an episode keeps its environment until it runs");

        // The wiring, in the one order that keeps the invariants: every inbox
        // and control channel first, so the router is complete before anybody
        // can send; then the router; and only then the threads.
        let mut unstarted = Vec::with_capacity(roster.len() + 1);
        let mut seats = BTreeMap::new();
        let mut starts: Vec<(ActorId, Start<P>)> = Vec::with_capacity(roster.len() + 1);
        for (id, start) in roster
            .into_iter()
            .chain([(environment_id.clone(), environment)])
        {
            let (channels, inbox, control) = UnstartedActor::new(id.clone());
            seats.insert(id.clone(), Seat { inbox, control });
            unstarted.push(channels);
            starts.push((id, start));
        }
        let router = Arc::new(Router::new(seats, environment_id.clone()));
        let (report, done) = unbounded();
        let shared = Shared {
            router: Arc::clone(&router),
            records,
            clock,
            done: report,
        };
        let mut actors: Vec<Started> = Vec::with_capacity(starts.len());
        for (channels, (_, start)) in unstarted.into_iter().zip(starts) {
            actors.push(start(channels, &shared));
        }
        // The episode's own senders go before it waits, so the writer has only
        // the actors' senders left and the completion channel only theirs. That
        // is what lets a disconnected channel mean "every thread has ended"
        // rather than "the episode is still holding one". Dropping the whole
        // `Shared` drops both, and the router clone with them.
        drop(shared);

        let outcome = router
            .command_as_episode(&[environment_id], Control::Start)
            .map_err(EpisodeError::Control)
            .and_then(|()| wait(&done, &actors, &router, limit));
        let failures = join(actors);
        // Every sender to the writer has gone with the threads that held them,
        // so the writer can be joined for the complete log. A failed required
        // sink shows up as every actor's `WriterClosed`, which the failures
        // below already report, so there is nothing to add here; what matters
        // is that the writer is joined before this returns.
        drop(writer.join());

        // A thread that did not end cleanly is the most specific thing the
        // episode can say, so it is said first: a timeout whose cause was a
        // panicking handler is reported as the panic.
        if !failures.is_empty() {
            return Err(EpisodeError::Agents(failures));
        }
        outcome
    }
}

/// Wraps a started actor in the erased form the episode holds.
fn started<H: Send + 'static>(actor: Actor<H>) -> Started {
    let (id, control) = (actor.id().clone(), actor.control().clone());
    Started {
        id,
        control,
        join: Box::new(move || actor.join().map(|_handler| ())),
    }
}

/// Stops everybody so that the log is complete up to whatever went wrong,
/// ignoring the actors that cannot be stopped because they already are.
///
/// An actor that has been stopped no longer has a control receiver —
/// `Perception::abandon` consumes itself and drops it — so a send to it is
/// refused. That is not a diagnosis and must not replace one: an episode
/// whose environment hung after some agent had died would otherwise come
/// back as `Control` ("could not deliver a control") rather than as the
/// `Timeout` naming who was still running, which is the whole reason this
/// milestone has a `Timeout` at all. [`join`] documents the same reasoning
/// for the same send; this is that treatment, applied where it was missing.
fn stop_everybody<P: Payload>(router: &Router<P>, everybody: &[ActorId]) {
    let _ = router.command_as_episode(everybody, Control::Stop);
}

/// Waits for every actor's threads to report, within `limit`.
///
/// Two threads per actor, so two reports per actor, and each report says
/// whether that actor had been stopped when its thread ended. Those three
/// outcomes are the three the episode distinguishes (ADR-0016):
///
/// - every actor reports having been stopped: the environment ended the
///   episode, and this returns `Ok`;
/// - an actor reports **without having been stopped**: it left unbidden, so the
///   episode stops the rest and returns [`EpisodeError::Departed`];
/// - the limit passes first: the episode sends `Stop` to every actor itself,
///   since it holds every control sender, and returns
///   [`EpisodeError::Timeout`].
fn wait<P: Payload>(
    done: &Receiver<Ended>,
    actors: &[Started],
    router: &Router<P>,
    limit: Duration,
) -> Result<(), EpisodeError> {
    let deadline = Instant::now() + limit;
    // Two threads apiece. An actor whose second thread has reported is
    // finished; until then the episode is still waiting on it.
    let mut outstanding: BTreeMap<ActorId, usize> =
        actors.iter().map(|actor| (actor.id.clone(), 2)).collect();
    let everybody: Vec<ActorId> = actors.iter().map(|actor| actor.id.clone()).collect();
    while !outstanding.is_empty() {
        match done.recv_deadline(deadline) {
            Ok(Ended { who, stopped }) => {
                if !stopped {
                    // Nobody told it to stop, so the episode was abandoned
                    // where it stood. The rest are stopped so that the log is
                    // complete up to the departure.
                    stop_everybody(router, &everybody);
                    return Err(EpisodeError::Departed);
                }
                if let Some(left) = outstanding.get_mut(&who) {
                    *left -= 1;
                    if *left == 0 {
                        outstanding.remove(&who);
                    }
                }
            }
            // Every actor still holds a sender, so a disconnected channel means
            // every thread has ended, reports and all.
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => {
                let running = outstanding.keys().cloned().collect();
                stop_everybody(router, &everybody);
                return Err(EpisodeError::Timeout { limit, running });
            }
        }
    }
    Ok(())
}

/// Joins every actor's threads and collects the ones that did not end cleanly.
///
/// Every actor is sent a `Stop` first, on the episode's own control sender for
/// it, and **before any of them is joined**. That is what makes joining
/// terminate at all: a perception thread waits on channels the [`Router`] holds
/// senders for, and the router is shared by every handler thread, so dropping
/// the episode's own senders frees nothing while any actor is alive. A `Stop`
/// is what ends a perception thread, so the episode sends one to everybody.
///
/// An actor already stopped no longer has a receiver, so the send fails and is
/// ignored: it cannot be stopped twice, because its perception thread has
/// already gone and there is nobody to write a second record.
fn join(actors: Vec<Started>) -> Vec<(ActorId, Failure)> {
    for actor in &actors {
        let _ = actor.control.send(Control::Stop);
    }
    let mut failures: Vec<(ActorId, Failure)> = actors
        .into_iter()
        .filter_map(|Started { id, control, join }| {
            drop(control);
            match panic::catch_unwind(AssertUnwindSafe(join)) {
                Ok(Ok(())) => None,
                Ok(Err(error)) => Some((id, Failure::Error(error))),
                Err(payload) => Some((id, Failure::Panicked(panic_message(payload.as_ref())))),
            }
        })
        .collect();
    failures.sort_by(|(one, _), (other, _)| one.cmp(other));
    failures
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
    use serde_json::Value;

    use super::*;
    use crate::contract::{Action, Effect, Observation};
    use crate::router::Seat;
    use crate::testing::{Shared, TestPayload, parse_lines, sinking};

    const ENVIRONMENT: &str = "environment";

    /// An episode whose log a test can read back.
    fn episode<W: Serialize + Send + 'static>(
        environment: impl Step<W, TestPayload> + Send + 'static,
    ) -> (Episode<W, TestPayload>, Shared) {
        let (sinks, log) = sinking();
        (Episode::new(sinks, ENVIRONMENT, environment), log)
    }

    /// An environment that starts everybody, waits for one message from each,
    /// and then stops everybody, itself included.
    struct Counting {
        agents: Vec<ActorId>,
        heard: usize,
    }

    impl Counting {
        fn new<const N: usize>(agents: [&str; N]) -> Self {
            Self {
                agents: agents.iter().map(|id| ActorId::new(*id)).collect(),
                heard: 0,
            }
        }

        /// Everybody, the environment included: what ends an episode.
        fn everybody(&self) -> Vec<ActorId> {
            let mut all = self.agents.clone();
            all.push(ActorId::new(ENVIRONMENT));
            all
        }
    }

    impl Step<i32, TestPayload> for Counting {
        fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
            let mut effects = vec![Effect::command(self.agents.clone(), Control::Start)];
            if self.agents.is_empty() {
                effects.push(Effect::command(self.everybody(), Control::Stop));
            }
            effects
        }

        fn step(
            &mut self,
            _observation: Observation<TestPayload>,
        ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
            self.heard += 1;
            if self.heard == self.agents.len() {
                vec![Effect::command(self.everybody(), Control::Stop)]
            } else {
                Vec::new()
            }
        }
    }

    /// An agent that greets the environment when it starts and says nothing
    /// else.
    struct Greeting;

    impl Policy<TestPayload> for Greeting {
        fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
            [Action::to([ENVIRONMENT], TestPayload::Step(0))]
        }

        fn policy(
            &mut self,
            _observation: Observation<TestPayload>,
        ) -> impl IntoIterator<Item = Action<TestPayload>> {
            []
        }
    }

    #[test]
    fn an_episode_runs_from_its_environments_start_to_its_stop() {
        let (mut episode, log) = episode(Counting::new(["a", "b"]));
        episode.add("a", Greeting).unwrap();
        episode.add("b", Greeting).unwrap();
        episode.run().unwrap();
        let lines = parse_lines(&log.bytes());
        assert_eq!(lines[0]["type"], "episode", "the header is first");
        let stops = lines
            .iter()
            .filter(|line| line["type"] == "control" && line["control"] == "stop")
            .count();
        assert_eq!(stops, 3, "the environment stops everybody, itself included");
    }

    #[test]
    fn every_actor_and_the_writer_hold_the_same_origin() {
        // The invariant is structural: `Episode::new` starts the clock, so
        // there is no second origin to hand anybody. What this checks is that
        // the one it started is the one the log's offsets are measured from,
        // by way of every logged instant being at or after it.
        /// An agent that reports the clock its start hook was given.
        struct Reporting(Sender<Clock>);

        impl Policy<TestPayload> for Reporting {
            fn start(&mut self, clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                self.0.send(clock).unwrap();
                [Action::to([ENVIRONMENT], TestPayload::Step(0))]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                []
            }
        }

        let (mut episode, log) = episode(Counting::new(["a", "b"]));
        let clock = episode.clock();
        let (told, clocks) = unbounded();
        episode.add("a", Reporting(told.clone())).unwrap();
        episode.add("b", Reporting(told)).unwrap();
        episode.run().unwrap();
        let reported: Vec<Clock> = clocks.try_iter().collect();
        assert_eq!(reported.len(), 2, "both agents started");
        for theirs in reported {
            assert_eq!(
                theirs.origin(),
                clock.origin(),
                "every actor's start hook is given the episode's clock"
            );
            assert_eq!(theirs.start_unix_ns(), clock.start_unix_ns());
        }
        // Every offset in the log is measured from that origin, so none can
        // be negative; `Elapsed` cannot hold a negative, so what a second
        // origin would produce is a zero where a real offset belongs. The
        // header's anchor is the clock's.
        let lines = parse_lines(&log.bytes());
        assert_eq!(
            lines[0]["start_unix_ns"].as_u64(),
            Some(clock.start_unix_ns()),
            "the log's anchor is the episode's clock"
        );
    }

    #[test]
    fn an_episode_with_no_agents_runs_and_logs_only_the_environments_own() {
        let (episode, log) = episode(Counting::new([]));
        episode.run().unwrap();
        let lines = parse_lines(&log.bytes());
        for line in &lines[1..] {
            assert_eq!(line["agent"], ENVIRONMENT, "{line}");
        }
    }

    #[test]
    fn a_roster_mixing_two_handler_types_runs() {
        /// A second handler type, which returns a `Vec` where [`Greeting`]
        /// returns an array, and keeps state besides. Two handler types in one
        /// roster is what the erasure in `add` buys.
        struct Chatty {
            said: usize,
        }

        impl Policy<TestPayload> for Chatty {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                self.said += 1;
                vec![Action::to([ENVIRONMENT], TestPayload::Step(1))]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                Vec::new()
            }
        }

        let (mut episode, log) = episode(Counting::new(["a", "b"]));
        episode.add("a", Greeting).unwrap();
        episode.add("b", Chatty { said: 0 }).unwrap();
        episode.run().unwrap();
        let lines = parse_lines(&log.bytes());
        let said: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "action")
            .collect();
        assert_eq!(said.len(), 2, "both handler types spoke: {lines:?}");
    }

    #[test]
    fn a_duplicate_id_is_rejected() {
        let (mut episode, _log) = episode(Counting::new(["a"]));
        episode.add("a", Greeting).unwrap();
        assert_eq!(
            episode.add("a", Greeting),
            Err(EpisodeError::DuplicateAgent(ActorId::new("a")))
        );
        assert_eq!(
            episode.add(ENVIRONMENT, Greeting),
            Err(EpisodeError::DuplicateAgent(ActorId::new(ENVIRONMENT)))
        );
    }

    #[test]
    fn the_roster_lists_every_actor_including_the_environment() {
        let (mut episode, _log) = episode(Counting::new(["b", "a"]));
        episode.add("b", Greeting).unwrap();
        episode.add("a", Greeting).unwrap();
        assert_eq!(
            episode.ids().cloned().collect::<Vec<_>>(),
            [
                ActorId::new("a"),
                ActorId::new("b"),
                ActorId::new(ENVIRONMENT)
            ]
        );
        assert_eq!(episode.environment(), &ActorId::new(ENVIRONMENT));
    }

    #[test]
    fn an_episode_whose_environment_never_stops_anyone_times_out() {
        /// An environment that starts everybody and never ends the episode.
        struct Forgetful;

        impl Step<i32, TestPayload> for Forgetful {
            fn start(
                &mut self,
                _clock: Clock,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                [Effect::command(["a"], Control::Start)]
            }

            fn step(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                []
            }
        }

        let (mut episode, log) = episode(Forgetful);
        episode.add("a", Greeting).unwrap();
        let limit = Duration::from_millis(200);
        let episode = episode.within(limit);
        assert_eq!(episode.limit(), limit);
        match episode.run() {
            Err(EpisodeError::Timeout { limit: given, .. }) => assert_eq!(given, limit),
            other => panic!("an episode that never ends times out: {other:?}"),
        }
        // The episode stopped everybody itself, so the log is complete.
        let lines = parse_lines(&log.bytes());
        let stops = lines
            .iter()
            .filter(|line| line["type"] == "control" && line["control"] == "stop")
            .count();
        assert_eq!(stops, 2, "the episode stopped both actors: {lines:?}");
    }

    #[test]
    fn a_timeout_is_still_a_timeout_when_somebody_was_already_stopped() {
        /// An environment that starts both agents, stops one of them the
        /// moment it hears anything, and then never ends the episode — the
        /// shape Werewolf has, where a player is stopped where it dies and
        /// the moderator may hang afterwards.
        struct StopsOneThenHangs {
            stopped: bool,
        }

        impl Step<i32, TestPayload> for StopsOneThenHangs {
            fn start(
                &mut self,
                _clock: Clock,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                [Effect::command(["a", "b"], Control::Start)]
            }

            fn step(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                // Stopping "a" closes its control channel, so the episode's
                // own `Stop` to everybody will be refused for it later.
                //
                // Once only: an environment that commands an actor it has
                // already stopped has its own send refused, which fails the
                // environment's thread — a separate hole in the same family
                // as this test's, and not the one under test here.
                let first = !self.stopped;
                self.stopped = true;
                first
                    .then(|| Effect::command(["a"], Control::Stop))
                    .into_iter()
            }
        }

        let (mut episode, _log) = episode(StopsOneThenHangs { stopped: false });
        episode.add("a", Greeting).unwrap();
        episode.add("b", Greeting).unwrap();
        let limit = Duration::from_millis(200);
        // The diagnosis must be the timeout and who was still running. Before
        // the fix the episode's own `Stop` to the already-stopped actor was
        // refused, the `?` turned that into `Control`, and the timeout — the
        // reason this milestone has a `Timeout` at all — was lost.
        match episode.within(limit).run() {
            Err(EpisodeError::Timeout {
                limit: given,
                running,
            }) => {
                assert_eq!(given, limit);
                assert!(
                    !running.is_empty(),
                    "the timeout names who was still running"
                );
            }
            other => panic!("a stopped actor must not mask the timeout: {other:?}"),
        }
    }

    #[test]
    fn an_actor_that_panics_mid_episode_is_reported_as_departed_or_failed() {
        /// An agent that panics the first time it is observed.
        struct Brittle;

        impl Policy<TestPayload> for Brittle {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                [Action::to([ENVIRONMENT], TestPayload::Step(0))]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                // A lazy iterator that panics on its first `next`. A bare
                // `panic!` in the body would give the method the return type
                // `!`, which is not an iterator, and a concrete `Vec` would
                // refine the trait's `impl IntoIterator`; this is an iterator
                // that happens never to yield.
                std::iter::from_fn(|| panic!("brittle gave up"))
            }
        }

        /// An environment that answers whoever greets it, which is what makes
        /// the brittle agent's `policy` run, and never stops anybody.
        struct Answering;

        impl Step<i32, TestPayload> for Answering {
            fn start(
                &mut self,
                _clock: Clock,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                [Effect::command(["a"], Control::Start)]
            }

            fn step(
                &mut self,
                observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
                [Effect::to(
                    [observation.message.sender],
                    TestPayload::Step(1),
                )]
            }
        }

        let (mut episode, _log) = episode(Answering);
        episode.add("a", Brittle).unwrap();
        let outcome = episode.within(Duration::from_secs(10)).run();
        match outcome {
            // The thread that panicked reported as it died, so the episode
            // learns of it either as a failure it joined or as a departure it
            // saw first; both are the episode being abandoned where it stood.
            Err(EpisodeError::Agents(failures)) => {
                assert!(
                    failures.iter().any(|(id, failure)| id == &ActorId::new("a")
                        && matches!(failure, Failure::Panicked(message) if message == "brittle gave up")),
                    "the panic is reported: {failures:?}"
                );
            }
            Err(EpisodeError::Departed) => {}
            other => panic!("a panicking handler fails the episode: {other:?}"),
        }
    }

    #[test]
    fn an_actor_that_ends_unbidden_is_a_departure() {
        // An actor's threads report as they end, and the report says whether
        // the actor had been stopped. A report from an actor nobody stopped is
        // a departure: the episode was abandoned where it stood, and waiting
        // out the time limit on an actor that is already gone would tell the
        // caller the wrong thing about the run.
        //
        // This is asserted of `wait` rather than of a whole episode because an
        // actor's threads only end of their own accord by panicking or by
        // failing, and then `run` reports the panic or the failure instead —
        // which is more specific and so the better answer. What `wait` decides
        // is the rule the issue states, and it is the rule under test.
        let (report, done) = unbounded();
        let (unstarted, inbox, control) = UnstartedActor::<TestPayload>::new("a");
        let seats = BTreeMap::from([(ActorId::new("a"), Seat { inbox, control })]);
        let router: Router<TestPayload> = Router::new(seats, ActorId::new("a"));
        let actors = vec![Started {
            id: ActorId::new("a"),
            control: unstarted.control().clone(),
            join: Box::new(|| Ok(())),
        }];
        report
            .send(Ended {
                who: ActorId::new("a"),
                stopped: false,
            })
            .unwrap();
        // A limit long enough that a timeout here would mean the departure went
        // unnoticed rather than that the machine was slow.
        assert_eq!(
            wait(&done, &actors, &router, Duration::from_secs(30)),
            Err(EpisodeError::Departed)
        );
        // And the actor was stopped on the way out, so its log closes.
        assert_eq!(
            unstarted.controls().try_recv(),
            Ok(Control::Stop),
            "the episode stops whoever is left"
        );
    }

    #[test]
    fn every_actor_reporting_a_stop_is_a_clean_run() {
        // The other side of the same rule: two reports per actor, both saying
        // it had been stopped, and `wait` is satisfied.
        let (report, done) = unbounded();
        let (unstarted, inbox, control) = UnstartedActor::<TestPayload>::new("a");
        let seats = BTreeMap::from([(ActorId::new("a"), Seat { inbox, control })]);
        let router: Router<TestPayload> = Router::new(seats, ActorId::new("a"));
        let actors = vec![Started {
            id: ActorId::new("a"),
            control: unstarted.control().clone(),
            join: Box::new(|| Ok(())),
        }];
        for _ in 0..2 {
            report
                .send(Ended {
                    who: ActorId::new("a"),
                    stopped: true,
                })
                .unwrap();
        }
        assert_eq!(
            wait(&done, &actors, &router, Duration::from_secs(30)),
            Ok(())
        );
    }

    #[test]
    fn errors_explain_themselves() {
        assert_eq!(
            EpisodeError::DuplicateAgent(ActorId::new("a")).to_string(),
            "actor a is in the roster twice"
        );
        assert_eq!(
            EpisodeError::Control(RouteError::Loopback(ActorId::new("a"))).to_string(),
            "could not deliver a control: actor a addressed itself"
        );
        assert_eq!(
            EpisodeError::Departed.to_string(),
            "an actor left while the episode was still running it"
        );
        let timed_out = EpisodeError::Timeout {
            limit: Duration::from_secs(1),
            running: BTreeSet::from([ActorId::new("a")]),
        };
        assert!(timed_out.to_string().ends_with(" a"), "{timed_out}");
        assert_eq!(
            EpisodeError::Agents(vec![(
                ActorId::new("a"),
                Failure::Panicked("oh no".to_string())
            )])
            .to_string(),
            "actors failed: [a: panicked: oh no]"
        );
        assert_eq!(
            Failure::Error(ActorError::WriterClosed).to_string(),
            "the log writer has gone away"
        );
    }
}
