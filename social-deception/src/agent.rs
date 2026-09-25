//! The agent: one thread, one queue, and a pop-fold-send loop.
//!
//! An agent's life is a fold over what arrives on its queue. Each cycle of
//! the loop:
//!
//! 1. waits until something arrives on the queue or its timeout fires;
//! 2. pops what is waiting: every control at the head of the queue and,
//!    behind them, **at most one** event, splitting them into
//!    [`Instruction`]s for the loop itself and one [`Observation`] for the
//!    handler;
//! 3. records each of them, stamped with the instant of the pop;
//! 4. hands the observation to the game's [`Handler`] and gets back the
//!    [`Action`]s to send;
//! 5. stamps each action with the agent's id and the instant of the send,
//!    records it, and sends the cycle's actions to the router as one
//!    [`CycleDispatch`];
//! 6. records the cycle.
//!
//! **A cycle handles one observation** (ADR-0008). An agent with a full
//! queue runs a cycle per event rather than one cycle for all of them, so it
//! is stale by at most one decision, and a cycle's window brackets exactly
//! one. What a drain found together was the scheduler's grouping and never
//! the game's; a game that means several messages together says so in one
//! payload.
//!
//! An event that arrives while the agent is busy waits its turn and is
//! picked up by a later cycle. So does a control: nothing an agent is told
//! reaches it before the things said to it first.
//!
//! Everything a cycle pops is stamped with one instant, so its observation
//! and every control it popped have the same `received`: the cycle's
//! `t_start`. There is no exception. That is what makes an agent's
//! deliberation recoverable from the log without a stamp for it, as ADR-0007
//! sets out: it is a sent action's `created` minus the `received` of the
//! observation in the same cycle, and the cycle record groups them.
//!
//! The handler returns what to send rather than sending it, so the loop sees
//! everything that goes out and the trajectory it records is authoritative.
//! It also means a handler cannot speak as anybody but itself: the loop is
//! what writes the sender.
//!
//! # One queue, and what that settles
//!
//! Events and controls arrive on one channel, as [`Delivery`]s, and the
//! loop waits on it alone (ADR-0009). A `Stop` is delivered like anything
//! else and is acted on when the agent reaches it: the agent is not told
//! that a stop is coming, it cannot act on the knowledge, and no handler is
//! offered a way to. Whatever a cycle's handler returns is always sent.
//!
//! ADR-0007 gave controls a queue of their own and a cancellation
//! trip-wire, for two reasons neither of which survived. A control queued
//! behind a *batch* of events waited for the batch — but ADR-0008 left no
//! batch. And a `Stop` behind a slow cycle waited for the cycle — but a
//! handler that blocks for thirty seconds is a defect wherever it appears,
//! and the place to fix it is inside the handler rather than in every
//! agent's loop forever.
//!
//! What the single queue keeps is the order things were sent in, which is
//! the order they are handled in. A `Stop` behind *n* events is reached
//! after those events, one cycle each. The episode does not produce that
//! arrangement on any path but failure: it holds a `Stop` back until its
//! in-flight count reads zero, which is to say until everything already
//! said has been handled (see [`episode`](crate::episode)). On the
//! abandon-ship path it cannot wait, and then an agent may answer events
//! queued ahead of the stop for an episode that has already failed. That is
//! accepted (ADR-0009): keeping a trajectory tidy through a failure is the
//! environment's job, since it decides when to stop whom.
//!
//! # Controls still come before events within a cycle
//!
//! One queue does not mean one thing per cycle. A cycle takes every control
//! at the head of the queue before it takes an event, so a cycle that finds
//! `[Start, event]` waiting pops the start, runs the start hook and then
//! observes the event, in that order, rather than observing first. And once
//! a `Stop` is in hand the event behind it is left where it is: this cycle
//! is the agent's last, and an event the agent never popped is one it never
//! observed, so it is never logged. A trajectory that said otherwise would
//! be claiming the agent saw something it did not.
//!
//! # The handler never sees a control
//!
//! A [`Control`] is out-of-domain, an instruction about the episode rather
//! than a move within the game, so the loop acts on it itself. [`Start`]
//! makes it call [`Handler::start`], whose opening actions are sent like any
//! others; [`Stop`] makes it exit after the cycle that popped it. Either way
//! the control is logged, so a reader sees it in the trajectory even though
//! no handler did.
//!
//! A `Start` that arrives after the agent has started is a bug in whoever
//! sent it, and the loop panics rather than record a start it did not act
//! on.
//!
//! [`Start`]: Control::Start
//! [`Stop`]: Control::Stop
//!
//! # Timeouts
//!
//! An agent may be given a timeout. From the cycle in which it pops
//! [`Control::Start`], a deadline is pending one interval ahead; when it
//! passes, the agent runs a cycle that calls [`Handler::timeout`] rather
//! than [`Handler::handle`], and the next deadline is one interval after
//! that cycle. A deadline is a separate method because waking on one is not
//! observing anything (ADR-0008); the default does nothing. The cycle record
//! says [`Woken::Timeout`], which is the only way to tell such a cycle from
//! one woken by an empty pop, since there is no wake-up object to record any
//! more.
//!
//! A cycle woken by the deadline that also found an event calls `handle`,
//! not `timeout`: it has an observation, and an observation is what a
//! handler decides from. The record still says the deadline woke it, which
//! is what moves the next one.
//!
//! Deadlines are absolute, and the wake channel for one is asked of the
//! [`TimerSource`] once and kept until it fires, so something arriving
//! before the deadline leaves the deadline where it was. An agent without an
//! interval blocks until something arrives.
//!
//! # Termination
//!
//! The loop exits when its queue closes, meaning every sender has been
//! dropped and nothing is left to pop, or after the cycle in which it popped
//! [`Control::Stop`], whichever comes first. Every record of that last cycle
//! has been sent to the writer before the thread returns.
//!
//! # Example
//!
//! An agent that echoes each message back to its sender, wired to a router
//! stand-in and a trajectory writer:
//!
//! ```
//! use crossbeam_channel::unbounded;
//! use social_deception::{
//!     Action, Agent, AgentId, Clock, Control, CycleDispatch, Delivery, Domain, Event, Handler,
//!     Observation, Wiring, Writer,
//! };
//!
//! struct Chat;
//!
//! impl Domain for Chat {
//!     type Payload = String;
//!     type Reward = i32;
//! }
//!
//! struct Echo;
//!
//! impl Handler<Chat> for Echo {
//!     fn handle(&mut self, observation: &Observation<Chat>) -> Vec<Action<Chat>> {
//!         vec![Action::to(
//!             [observation.event.sender.clone()],
//!             observation.event.payload.clone(),
//!         )]
//!     }
//! }
//!
//! let clock = Clock::start();
//! let (to_agent, queue) = unbounded();
//! let (dispatches, from_agent) = unbounded();
//! let (records, writer) = Writer::spawn::<Chat>(Vec::new());
//! let peers = [AgentId::new("caller")].into();
//! let wiring =
//!     Wiring { id: "echo".into(), clock, queue, dispatches, records, timeout: None, peers };
//! let agent = Agent::spawn(wiring, Echo, clock);
//!
//! let hello = Event::<Chat>::new("caller", ["echo"], clock.now(), String::from("hello"));
//! // One queue, so everything is said in the order it is to be handled: the
//! // start, the message, and then the stop the agent reaches after
//! // answering it.
//! to_agent.send(Delivery::control(Control::Start, clock.now())).unwrap();
//! to_agent.send(Delivery::Event(hello)).unwrap();
//! to_agent.send(Delivery::control(Control::Stop, clock.now())).unwrap();
//!
//! let echoed: CycleDispatch<Chat> = from_agent.iter().find(|r| !r.sent.is_empty()).unwrap();
//! agent.join().unwrap();
//! let sent = echoed.sent;
//! assert_eq!(sent.len(), 1);
//! assert_eq!(sent[0].sender, AgentId::new("echo"));
//! assert_eq!(sent[0].payload, "hello");
//! let trajectory = writer.join().unwrap();
//! assert!(!trajectory.is_empty());
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::panic;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError, never, select};

use crate::clock::{Clock, Created, Timestamp, Timestamped};
use crate::event::{AgentId, Control, Delivery, Domain, Event};
use crate::timer::TimerSource;
use crate::trajectory::{
    ActionRecord, ControlRecord, CycleRecord, LogRecord, ObservationRecord, Seq, Woken,
};

/// An event this agent has popped off its queue: what it observed, and when.
///
/// The event carries the instant its sender created it; this adds the
/// instant this agent received it. The gap between the two is the
/// observation's [`latency`](Timestamped::latency), the whole staleness of
/// what the agent is looking at.
/// `Debug`, `Clone` and equality are written out rather than derived, for
/// the reason [`Event`]'s are: a derive would ask them of `D`.
pub struct Observation<D: Domain> {
    /// The event.
    pub event: Event<D>,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
}

impl<D: Domain> Created for Observation<D> {
    fn created(&self) -> Timestamp {
        self.event.created
    }
}

impl<D: Domain> Timestamped for Observation<D> {
    fn received(&self) -> Timestamp {
        self.received
    }
}

impl<D: Domain> fmt::Debug for Observation<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Observation")
            .field("event", &self.event)
            .field("received", &self.received)
            .finish()
    }
}

impl<D: Domain> Clone for Observation<D> {
    fn clone(&self) -> Self {
        Self {
            event: self.event.clone(),
            received: self.received,
        }
    }
}

impl<D: Domain> PartialEq for Observation<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.event == other.event && self.received == other.received
    }
}

impl<D: Domain> Eq for Observation<D> where D::Payload: Eq {}

/// A control this agent has popped off its queue, and when.
///
/// Logged by the loop and never handed to a handler; it is a
/// [`Timestamped`] for the same reason an observation is, so that a control
/// that waited behind whatever was queued ahead of it says so.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Instruction {
    /// The control.
    pub control: Control,
    /// When the episode sent it.
    pub created: Timestamp,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
}

impl Created for Instruction {
    fn created(&self) -> Timestamp {
        self.created
    }
}

impl Timestamped for Instruction {
    fn received(&self) -> Timestamp {
        self.received
    }
}

/// What an agent sends: to whom, and what.
///
/// No sender and no creation time. An action has no creation time until it
/// is sent, and the handler cannot know that instant, so the loop stamps
/// both as it hands the action to the router; the stamped value is the
/// [`Event`] on the wire and what the `action` record logs.
/// `Debug`, `Clone` and equality are written out for the same reason
/// [`Event`]'s are.
pub struct Action<D: Domain> {
    /// The agents to send it to.
    pub recipients: Recipients,
    /// What to say.
    pub payload: D::Payload,
}

impl<D: Domain> Action<D> {
    /// An action addressed to the given recipients.
    pub fn to<I, A>(recipients: I, payload: D::Payload) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<AgentId>,
    {
        Self {
            recipients: Recipients::To(recipients.into_iter().map(Into::into).collect()),
            payload,
        }
    }

    /// An action addressed to every other agent in the roster.
    pub fn broadcast(payload: D::Payload) -> Self {
        Self {
            recipients: Recipients::Broadcast,
            payload,
        }
    }
}

impl<D: Domain> fmt::Debug for Action<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Action")
            .field("recipients", &self.recipients)
            .field("payload", &self.payload)
            .finish()
    }
}

impl<D: Domain> Clone for Action<D> {
    fn clone(&self) -> Self {
        Self {
            recipients: self.recipients.clone(),
            payload: self.payload.clone(),
        }
    }
}

impl<D: Domain> PartialEq for Action<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.recipients == other.recipients && self.payload == other.payload
    }
}

impl<D: Domain> Eq for Action<D> where D::Payload: Eq {}

/// Whom an action is for, as the handler states it.
///
/// The loop turns this into the explicit recipient set the sent [`Event`]
/// carries, so a broadcast is recorded as the agents it actually went to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recipients {
    /// Every other agent in the roster: the agent's [`Wiring::peers`].
    Broadcast,
    /// These agents and no others.
    To(BTreeSet<AgentId>),
}

/// A game's behavior for one agent.
///
/// The loop calls [`handle`](Handler::handle) once per cycle with the one
/// observation that cycle popped, and sends what comes back. Agent state
/// lives in the implementing type. A handler never sees a [`Control`]; see
/// the [module documentation](self).
pub trait Handler<D: Domain> {
    /// The agent's opening actions, called once when it is started.
    ///
    /// This is where an agent that acts before anybody has spoken to it does
    /// so. Most agents only react, and the default returns nothing.
    ///
    /// Opening actions are the agent's own, decided from nothing; a handler
    /// with work to do before it can name them has state to fold and belongs
    /// in `handle`.
    fn start(&mut self) -> Vec<Action<D>> {
        Vec::new()
    }

    /// Folds one observation into the agent's state and says what to send.
    ///
    /// One observation, because a cycle handles exactly one (ADR-0008): this
    /// is a decision point, and what an agent conditions on at a decision
    /// point is an observation, not a pile of them. A cycle that popped no
    /// observation — the opening `Start`, or the timeout — does not call
    /// this at all.
    ///
    /// Nothing interrupts it. An agent does not know it is being stopped
    /// (ADR-0009), so a handler is never asked to give up early and what
    /// this returns is always sent. A handler that blocks delays its own
    /// agent, and with it the end of its episode, for as long as it blocks;
    /// the place to bound that is inside the handler, in whatever it is
    /// blocking on.
    fn handle(&mut self, observation: &Observation<D>) -> Vec<Action<D>>;

    /// What the agent does when its deadline passes and nothing has arrived.
    ///
    /// Waking on a deadline is not observing anything, so it is its own
    /// method rather than a `handle` with nothing to hand over: an empty
    /// slice made "no observation" a kind of observation, and every handler
    /// would have had to unwrap its way back out of it (ADR-0008). The
    /// default does nothing, which is what an agent without a timeout wants
    /// and what every agent in the tree wants today.
    fn timeout(&mut self) -> Vec<Action<D>> {
        Vec::new()
    }
}

/// What one cycle of an agent's loop hands the router.
///
/// One dispatch is sent per cycle, after the cycle's actions have been
/// recorded and before its cycle record is written. Because the number of
/// deliveries consumed and the events produced arrive together, whoever
/// counts in-flight deliveries never sees a cycle's inputs settled before
/// its outputs exist.
/// `Debug`, `Clone` and equality are written out for the same reason
/// [`Event`]'s are.
pub struct CycleDispatch<D: Domain> {
    /// The agent whose cycle this was.
    pub agent: AgentId,
    /// How many deliveries the cycle took off its queue. A timeout is not a
    /// delivery.
    pub deliveries: usize,
    /// The events the cycle sent, stamped with this agent as sender, in the
    /// order the handler returned them.
    pub sent: Vec<Event<D>>,
}

/// Everything an agent's thread needs besides its handler and timer.
///
/// One queue, carrying both kinds of thing said to the agent; the
/// [module documentation](self) says why it is not two.
pub struct Wiring<D: Domain> {
    /// The agent's id: the sender on everything it emits and the `agent` on
    /// every record it writes.
    pub id: AgentId,
    /// The episode clock.
    pub clock: Clock,
    /// The agent's queue: events, which become [`Observation`]s when
    /// popped, and controls, which the loop acts on itself, in the order
    /// they were sent.
    pub queue: Receiver<Delivery<D>>,
    /// Where each cycle's dispatch goes.
    pub dispatches: Sender<CycleDispatch<D>>,
    /// Where the agent's trajectory goes.
    pub records: Sender<LogRecord<D>>,
    /// How long the agent waits before its deadline fires, or `None` for an
    /// agent that only ever reacts. A deadline that passes with nothing
    /// waiting runs a cycle that calls [`Handler::timeout`]; one that passes
    /// while an event waits joins that event's cycle instead.
    pub timeout: Option<Duration>,
    /// The other agents in the roster: what a broadcast goes to.
    pub peers: BTreeSet<AgentId>,
}

impl<D: Domain> fmt::Debug for CycleDispatch<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CycleDispatch")
            .field("agent", &self.agent)
            .field("deliveries", &self.deliveries)
            .field("sent", &self.sent)
            .finish()
    }
}

impl<D: Domain> Clone for CycleDispatch<D> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            deliveries: self.deliveries,
            sent: self.sent.clone(),
        }
    }
}

impl<D: Domain> PartialEq for CycleDispatch<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.agent == other.agent && self.deliveries == other.deliveries && self.sent == other.sent
    }
}

impl<D: Domain> Eq for CycleDispatch<D> where D::Payload: Eq {}

impl<D: Domain> fmt::Debug for Wiring<D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wiring")
            .field("id", &self.id)
            .field("timeout", &self.timeout)
            .field("peers", &self.peers)
            .finish_non_exhaustive()
    }
}

/// Why an agent's loop stopped before its queues closed or it was told to
/// stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    /// A record could not be sent: the trajectory writer has gone away.
    WriterClosed,
    /// A dispatch could not be sent: the router has gone away.
    RouterClosed,
    /// The timer source disconnected a wake channel the agent was waiting on.
    TimerClosed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WriterClosed => "the trajectory writer has gone away",
            Self::RouterClosed => "the router has gone away",
            Self::TimerClosed => "the timer source disconnected a wake channel",
        })
    }
}

impl std::error::Error for Error {}

/// A running agent: the handle to its thread.
///
/// Created by [`Agent::spawn`]; [`Agent::join`] waits for the thread to exit
/// and gives back the handler.
#[derive(Debug)]
pub struct Agent<H> {
    id: AgentId,
    thread: JoinHandle<Result<H, Error>>,
}

impl<H> Agent<H> {
    /// Starts an agent on its own thread.
    ///
    /// # Panics
    ///
    /// If the operating system refuses to create the thread.
    pub fn spawn<D, T>(wiring: Wiring<D>, handler: H, timer: T) -> Self
    where
        D: Domain,
        H: Handler<D> + Send + 'static,
        T: TimerSource + Send + 'static,
    {
        let id = wiring.id.clone();
        let agent = Loop {
            wiring,
            handler,
            timer,
            next_seq: 0,
            last_created: None,
            started: false,
            closed: false,
            pending: None,
        };
        let thread = thread::Builder::new()
            .name(id.to_string())
            .spawn(move || agent.run())
            .expect("failed to spawn agent thread");
        Self { id, thread }
    }

    /// The agent's id.
    #[must_use]
    pub fn id(&self) -> &AgentId {
        &self.id
    }

    /// Waits for the agent's thread to exit and gives back its handler.
    ///
    /// # Errors
    ///
    /// If the loop stopped because a channel it depends on went away; see
    /// [`Error`].
    ///
    /// # Panics
    ///
    /// If the handler panicked, the panic is propagated to the caller.
    pub fn join(self) -> Result<H, Error> {
        self.thread
            .join()
            .unwrap_or_else(|payload| panic::resume_unwind(payload))
    }
}

/// What ended a wait, and what it took off a queue doing so.
///
/// A `select!` receive is destructive, so whatever woke the loop is already
/// in hand and has to be carried into the cycle rather than left to the
/// drain to find.
enum Wake<D: Domain> {
    /// Something arrived on the queue, and here it is.
    Delivered(Delivery<D>),
    /// The pending deadline passed.
    Deadline,
    /// The queue closed, which the drain that follows confirms without
    /// throwing anything away.
    Closed,
    /// The wake channel disconnected.
    TimerGone,
}

/// What one cycle took off its queue, in the order the loop records it:
/// the controls at the head, then the one event behind them.
///
/// The controls are however many were waiting in a row and the event is at
/// most one. A control is out-of-domain and is the loop's own business, so
/// a cycle takes every one it finds before it looks for something to
/// observe; an event is an observation, and a cycle handles exactly one
/// (ADR-0008). What is left on the queue is the next cycle's.
///
/// `deliveries` counts everything taken off the queue, and is made where
/// the taking is rather than from what survives it. That matters on the
/// `Stop` path, where an event the wake-up had already taken is forgotten:
/// it was still delivered, and an episode that never hears of a delivery
/// waits forever for it (see [`episode`](crate::episode)).
struct Popped<D: Domain> {
    controls: Vec<Instruction>,
    event: Option<Event<D>>,
    deliveries: usize,
}

impl<D: Domain> Popped<D> {
    /// Whether the cycle popped nothing at all, which is how a wake-up that
    /// was only the queue closing is told from one that brought work.
    fn is_empty(&self) -> bool {
        self.controls.is_empty() && self.event.is_none()
    }
}

/// The state of an agent's thread.
struct Loop<D: Domain, H, T> {
    wiring: Wiring<D>,
    handler: H,
    timer: T,
    /// The next sequence number to assign.
    next_seq: u64,
    /// The last `created` this agent stamped onto an action. Per agent, not
    /// per cycle: the join an observation makes is on the sender and the
    /// instant over the whole trajectory, so two actions of one agent may
    /// not share an instant even across a cycle boundary.
    last_created: Option<Timestamp>,

    /// Whether the agent has been started. A second `Start` is a bug in
    /// whoever sent it; see [`cycle`](Self::cycle).
    started: bool,

    /// Whether the queue has closed. Sticky: learned from a pop that found
    /// it disconnected and true from then on. Asking again is not free —
    /// the only way to ask is `try_recv`, which would take the very item
    /// that makes the answer no — so the answer is kept.
    closed: bool,

    /// The earliest deadline and the wake channel asked for it, kept across
    /// cycles until it fires.
    pending: Option<(Timestamp, Receiver<Instant>)>,
}

impl<D, H, T> Loop<D, H, T>
where
    D: Domain,
    H: Handler<D>,
    T: TimerSource,
{
    fn run(mut self) -> Result<H, Error> {
        loop {
            let woke_with = self.wait();
            let woken_by_deadline = match woke_with {
                Wake::Deadline => true,
                Wake::TimerGone => return Err(Error::TimerClosed),
                Wake::Delivered(_) | Wake::Closed => false,
            };
            // Everything popped shares one instant, so the cycle's start is
            // taken before the pop rather than after it.
            let t_start = self.wiring.clock.now();
            let popped = self.drain(woke_with, t_start);
            if popped.is_empty() && !woken_by_deadline {
                // The queue closed and brought nothing with it. There is no
                // cycle to run and nothing left to wait for.
                break;
            }
            let timed_out = woken_by_deadline || self.deadline_passed()?;
            if timed_out {
                self.take_deadline();
            }
            let stop = self.cycle(t_start, popped, timed_out)?;
            if stop || self.closed {
                break;
            }
        }
        Ok(self.handler)
    }

    /// Blocks until something arrives on the queue or the pending deadline
    /// fires.
    ///
    /// A queue that has closed and emptied is a `select!` arm that is ready
    /// forever, which would spin, so a queue already known to be closed is
    /// swapped for one that is never ready. The loop never waits again after
    /// learning that, but the swap keeps the arm honest in the one turn
    /// between the two.
    fn wait(&self) -> Wake<D> {
        let (idle, spent) = (never(), never());
        let deadline = self.pending.as_ref().map_or(&idle, |(_, wake)| wake);
        let queue = if self.closed {
            &spent
        } else {
            &self.wiring.queue
        };
        select! {
            recv(queue) -> delivered => delivered.map_or(Wake::Closed, Wake::Delivered),
            recv(deadline) -> fired => if fired.is_ok() { Wake::Deadline } else { Wake::TimerGone },
        }
    }

    /// Takes the controls at the head of the queue and, behind them, **at
    /// most one** event, stamping each with `t_start`, and remembers the
    /// queue if it turned out to be closed.
    ///
    /// One queue, so one pass along it. The loop takes controls while
    /// controls are what it finds, because a control is out-of-domain and
    /// the loop's own business — a cycle that stopped at the first of a run
    /// of them would spend a cycle on each, recording nothing and asking
    /// the handler nothing. The first event ends the pass, because a cycle
    /// handles exactly one observation (ADR-0008) and the rest of the queue
    /// is the next cycle's.
    ///
    /// So a cycle that finds `[Start, event]` pops both and does the start
    /// first, while one that finds `[event, Stop]` pops only the event and
    /// reaches the stop next time round. The second is what ADR-0009
    /// accepts: with one queue a `Stop` behind events is handled after
    /// them, and the episode does not queue one that way except when it is
    /// abandoning a run that has already failed.
    ///
    /// Once a `Stop` is in hand the pass ends there. This cycle is the
    /// agent's last, and an event left on the queue is one the agent never
    /// popped, never observed and is never logged for: taking it only to
    /// observe it after the episode had ended would put a decision in the
    /// trajectory that nobody asked for. The one event the wake-up may
    /// already have taken ahead of a `Stop` cannot arise — the wake-up
    /// takes one thing and the pass stops at the stop it then finds — so
    /// the only thing forgotten is what was never taken.
    fn drain(&mut self, woke_with: Wake<D>, t_start: Timestamp) -> Popped<D> {
        let (mut controls, mut event, mut deliveries) = (Vec::new(), None, 0);
        let mut in_hand = match woke_with {
            Wake::Delivered(delivery) => Some(delivery),
            // A deadline or a closed queue brings nothing with it; what the
            // pass finds is the whole of the cycle.
            Wake::Deadline | Wake::Closed => None,
            Wake::TimerGone => unreachable!("the caller returns on a gone timer"),
        };
        loop {
            let delivery = match in_hand.take() {
                Some(delivery) => delivery,
                None if self.closed => break,
                None => match self.wiring.queue.try_recv() {
                    Ok(delivery) => delivery,
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        // Nothing is thrown away: the `try_recv` that
                        // reports the queue disconnected is the one that
                        // found it empty.
                        self.closed = true;
                        break;
                    }
                },
            };
            deliveries += 1;
            match delivery {
                Delivery::Control { control, created } => {
                    controls.push(Instruction {
                        control,
                        created,
                        received: t_start,
                    });
                    if control == Control::Stop {
                        break;
                    }
                }
                Delivery::Event(popped) => {
                    event = Some(popped);
                    break;
                }
            }
        }
        Popped {
            controls,
            event,
            deliveries,
        }
    }

    /// Whether the pending deadline has fired without being the reason for
    /// this wake-up.
    fn deadline_passed(&self) -> Result<bool, Error> {
        match &self.pending {
            None => Ok(false),
            Some((_, wake)) => match wake.try_recv() {
                Ok(_) => Ok(true),
                Err(TryRecvError::Empty) => Ok(false),
                Err(TryRecvError::Disconnected) => Err(Error::TimerClosed),
            },
        }
    }

    /// Retires the deadline that just fired.
    fn take_deadline(&mut self) {
        self.pending = None;
    }

    /// Schedules the next timeout one interval after `from`, if the agent
    /// has one at all.
    fn schedule_timeout(&mut self, from: Timestamp) {
        if let Some(every) = self.wiring.timeout {
            let deadline = from + every;
            self.pending = Some((deadline, self.timer.wake_at(deadline)));
        }
    }

    /// Runs one cycle: record what was popped, hand the observation to the
    /// handler, send and record what comes back, and close with the cycle
    /// record. Returns whether the cycle popped a stop.
    ///
    /// Whatever the handler returns is sent. Nothing arriving while it runs
    /// changes that: an agent does not know it is being stopped (ADR-0009),
    /// so there is no second look at the queue and no cycle whose outputs
    /// go nowhere.
    fn cycle(
        &mut self,
        t_start: Timestamp,
        popped: Popped<D>,
        timed_out: bool,
    ) -> Result<bool, Error> {
        let Popped {
            controls,
            event,
            deliveries,
        } = popped;
        let (mut inputs, mut started, mut stopped) = (Vec::new(), false, false);
        // Controls first, and in one pass, so that a `Start` at the head of
        // the queue has run the start hook before the event behind it is
        // observed.
        for instruction in &controls {
            match instruction.control {
                // An agent is started once, before anything is addressed to
                // it. A second `Start` cannot be honored — the start hook
                // has already run, and running it again would reopen an
                // agent that has been playing — and recording it anyway
                // would put a claim in the trajectory the loop did not act
                // on. Whoever sent it has a bug the trajectory must not
                // paper over.
                Control::Start => {
                    assert!(!self.started, "{} was started twice", self.wiring.id);
                    self.started = true;
                    started = true;
                }
                Control::Stop => stopped = true,
            }
            inputs.push(self.record_control(instruction)?);
        }
        let observation = event.map(|event| Observation {
            event,
            received: t_start,
        });
        if let Some(observation) = &observation {
            inputs.push(self.record_observation(observation)?);
        }
        // The next deadline is measured from this cycle, whether the agent
        // has just started or the last deadline is what woke it. Once, even
        // when both are true: asking the timer twice for the same instant
        // leaves the loop with one deadline either way, but the second ask
        // is on the record of any timer that keeps one.
        if started || timed_out {
            self.schedule_timeout(t_start);
        }

        // A start is the loop's own business: it calls the start hook, whose
        // opening actions go out ahead of whatever the cycle's observation
        // also produced, since the start was popped before it.
        let mut actions = if started {
            self.handler.start()
        } else {
            Vec::new()
        };
        // Exactly one of the two, or neither. An observation is what the
        // handler decides from; a deadline is not one, and says so by being
        // its own method; and a cycle that popped only controls — the
        // opening `Start`, or a `Stop` — asks the handler nothing, because
        // there is nothing it observed to ask about.
        if let Some(observation) = &observation {
            actions.extend(self.handler.handle(observation));
        } else if timed_out {
            actions.extend(self.handler.timeout());
        }

        let (mut sent, mut outputs) = (Vec::new(), Vec::new());
        for action in actions {
            let event = self.stamp(action);
            outputs.push(self.record_action(&event)?);
            sent.push(event);
        }

        let dispatch = CycleDispatch {
            agent: self.wiring.id.clone(),
            deliveries,
            sent,
        };
        self.wiring
            .dispatches
            .send(dispatch)
            .map_err(|_| Error::RouterClosed)?;
        let cycle = CycleRecord {
            agent: self.wiring.id.clone(),
            t_start,
            t_stop: self.wiring.clock.now(),
            woken: if timed_out {
                Woken::Timeout
            } else {
                Woken::Queue
            },
            inputs,
            outputs,
        };
        self.wiring
            .records
            .send(cycle.into())
            .map_err(|_| Error::WriterClosed)?;
        Ok(stopped)
    }

    /// Stamps one action with this agent as sender and the instant of the
    /// stamp, resolving a broadcast to the peers it actually goes to.
    ///
    /// The stamp is the action's `created` on the wire, and every action a
    /// handler returns is sent, so there is no other case.
    ///
    /// The stamp is always strictly later than the last one this agent
    /// handed out. That is not cosmetic. An observation names the action it
    /// came from by the sender and the creation time and by nothing else,
    /// since no event carries an identifier (ADR-0002), so two of an agent's
    /// actions sharing an instant would be two actions no observation could
    /// tell apart. A cycle that returns several actions stamps them within a
    /// few hundred nanoseconds of each other, which the clock's resolution
    /// does not always separate, and two cycles can run that close together
    /// too, so the guarantee is the agent's and not one cycle's.
    fn stamp(&mut self, action: Action<D>) -> Event<D> {
        let Action {
            recipients,
            payload,
        } = action;
        let now = self.wiring.clock.now();
        let created = match self.last_created {
            Some(previous) if now <= previous => previous + Duration::from_nanos(1),
            _ => now,
        };
        self.last_created = Some(created);
        Event {
            sender: self.wiring.id.clone(),
            recipients: match recipients {
                Recipients::Broadcast => self.wiring.peers.clone(),
                Recipients::To(recipients) => recipients,
            },
            created,
            payload,
        }
    }

    /// Takes the next sequence number.
    fn next_seq(&mut self) -> Seq {
        let seq = Seq(self.next_seq);
        self.next_seq += 1;
        seq
    }

    fn record_observation(&mut self, observation: &Observation<D>) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            ObservationRecord {
                agent: self.wiring.id.clone(),
                seq,
                created: observation.event.created,
                received: observation.received,
                event: observation.event.clone(),
            }
            .into(),
        )?;
        Ok(seq)
    }

    fn record_action(&mut self, event: &Event<D>) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            ActionRecord {
                agent: self.wiring.id.clone(),
                seq,
                created: event.created,
                event: event.clone(),
            }
            .into(),
        )?;
        Ok(seq)
    }

    fn record_control(&mut self, instruction: &Instruction) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            ControlRecord {
                agent: self.wiring.id.clone(),
                seq,
                created: instruction.created(),
                received: instruction.received(),
                control: instruction.control,
            }
            .into(),
        )?;
        Ok(seq)
    }

    fn send_record(&self, record: LogRecord<D>) -> Result<(), Error> {
        self.wiring
            .records
            .send(record)
            .map_err(|_| Error::WriterClosed)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossbeam_channel::unbounded;
    use serde_json::json;

    use super::*;
    use crate::testing::{TestDomain, TestPayload, parse_lines};
    use crate::timer::{ManualTimer, ManualTimerControl};
    use crate::trajectory::Writer;

    /// How long a test waits on a channel before giving up. A test only ever
    /// waits this long when it has already failed.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// A timeout interval. Its value never matters: the manual timer decides
    /// when deadlines fire.
    const EVERY: Duration = Duration::from_secs(1);

    type TestEvent = Event<TestDomain>;

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// A step from `sender` to `a`, created at `created`.
    fn step_at(sender: &str, n: u64, created: Timestamp) -> TestEvent {
        Event::new(sender, ["a"], created, TestPayload::Step(n))
    }

    fn step(sender: &str, n: u64) -> TestEvent {
        step_at(sender, n, Timestamp::default())
    }

    fn recv<T>(receiver: &Receiver<T>) -> T {
        receiver.recv_timeout(PATIENCE).expect("nothing arrived")
    }

    /// Remembers every observation it was given, in order, and replies to
    /// each with the next step, addressed to whoever sent it. It also counts
    /// the starts and the timeouts, so that a test can see that the loop,
    /// not the handler, acts on a control, and that a deadline reaches
    /// `timeout` rather than `handle`.
    ///
    /// One cycle is one observation, so `seen` is the sequence of
    /// observations across the whole run rather than a pile per cycle, which
    /// is exactly what ADR-0008 makes it.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Recorder {
        started: usize,
        timeouts: usize,
        seen: Vec<TestPayload>,
    }

    impl Handler<TestDomain> for Recorder {
        fn start(&mut self) -> Vec<Action<TestDomain>> {
            self.started += 1;
            Vec::new()
        }

        fn handle(&mut self, observation: &Observation<TestDomain>) -> Vec<Action<TestDomain>> {
            let TestPayload::Step(n) = observation.event.payload;
            self.seen.push(observation.event.payload.clone());
            vec![Action::to(
                [observation.event.sender.clone()],
                TestPayload::Step(n + 1),
            )]
        }

        fn timeout(&mut self) -> Vec<Action<TestDomain>> {
            self.timeouts += 1;
            Vec::new()
        }
    }

    /// A recorder that, on entering each cycle, tells the test it is busy and
    /// then waits to be released. That is how a test makes things arrive
    /// while the agent is provably mid-cycle.
    #[derive(Debug)]
    struct Gated {
        inner: Recorder,
        entered: Sender<()>,
        release: Receiver<()>,
    }

    impl Handler<TestDomain> for Gated {
        fn handle(&mut self, observation: &Observation<TestDomain>) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.handle(observation)
        }

        fn timeout(&mut self) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.timeout()
        }
    }

    /// Broadcasts a step when it starts.
    struct Town;

    impl Handler<TestDomain> for Town {
        fn start(&mut self) -> Vec<Action<TestDomain>> {
            vec![Action::broadcast(TestPayload::Step(0))]
        }

        fn handle(&mut self, _: &Observation<TestDomain>) -> Vec<Action<TestDomain>> {
            Vec::new()
        }
    }

    /// A panicking handler.
    struct Faulty;

    impl Handler<TestDomain> for Faulty {
        fn handle(&mut self, _: &Observation<TestDomain>) -> Vec<Action<TestDomain>> {
            panic!("handler bug");
        }

        fn timeout(&mut self) -> Vec<Action<TestDomain>> {
            panic!("handler bug");
        }
    }

    /// An agent and the test's end of every channel it is wired to.
    struct Rig<H> {
        agent: Agent<H>,
        clock: Clock,
        queue: Sender<Delivery<TestDomain>>,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
        timer: ManualTimerControl,
    }

    /// The channels of a rig, before the agent is spawned on them.
    struct Wires {
        wiring: Wiring<TestDomain>,
        queue: Sender<Delivery<TestDomain>>,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
    }

    impl Wires {
        fn control(&self, control: Control) {
            self.queue
                .send(Delivery::control(control, self.wiring.clock.now()))
                .unwrap();
        }

        fn send(&self, event: TestEvent) {
            self.queue.send(Delivery::Event(event)).unwrap();
        }
    }

    fn wires(timeout: Option<Duration>) -> Wires {
        let (queue, receiver) = unbounded();
        let (outbox, dispatches) = unbounded();
        let (recorder, records) = unbounded();
        let wiring = Wiring {
            id: AgentId::new("a"),
            clock: Clock::start(),
            queue: receiver,
            dispatches: outbox,
            records: recorder,
            timeout,
            peers: ["b", "c"].map(AgentId::new).into(),
        };
        Wires {
            wiring,
            queue,
            dispatches,
            records,
        }
    }

    fn rig<H: Handler<TestDomain> + Send + 'static>(
        handler: H,
        timeout: Option<Duration>,
    ) -> Rig<H> {
        let wires = wires(timeout);
        let (timer, control) = ManualTimer::new();
        let clock = wires.wiring.clock;
        Rig {
            agent: Agent::spawn(wires.wiring, handler, timer),
            clock,
            queue: wires.queue,
            dispatches: wires.dispatches,
            records: wires.records,
            timer: control,
        }
    }

    impl<H> Rig<H> {
        fn send(&self, event: TestEvent) {
            self.queue.send(Delivery::Event(event)).unwrap();
        }

        fn control(&self, control: Control) {
            self.queue
                .send(Delivery::control(control, self.clock.now()))
                .unwrap();
        }

        fn start(&self) {
            self.control(Control::Start);
        }

        fn stop(&self) {
            self.control(Control::Stop);
        }

        fn dispatch(&self) -> CycleDispatch<TestDomain> {
            recv(&self.dispatches)
        }

        /// The records of one cycle: everything it wrote, then its cycle
        /// record.
        fn cycle(&self) -> (Vec<LogRecord<TestDomain>>, CycleRecord) {
            let mut records = Vec::new();
            loop {
                match recv(&self.records) {
                    LogRecord::Cycle(cycle) => return (records, cycle),
                    record => records.push(record),
                }
            }
        }
    }

    fn gated() -> (Rig<Gated>, Receiver<()>, Sender<()>) {
        let (entered, busy) = unbounded();
        let (release, released) = unbounded();
        let handler = Gated {
            inner: Recorder::default(),
            entered,
            release: released,
        };
        (rig(handler, Some(EVERY)), busy, release)
    }

    fn steps<const N: usize>(ns: [u64; N]) -> Vec<TestPayload> {
        ns.map(TestPayload::Step).into()
    }

    /// What kind of record this is, for a test that cares about the order
    /// of the kinds rather than their contents.
    fn kind(record: &LogRecord<TestDomain>) -> &'static str {
        match record {
            LogRecord::Observation(_) => "observation",
            LogRecord::Action(_) => "action",
            LogRecord::Control(_) => "control",
            // An agent's loop never writes one: a reward is the
            // environment's, and it goes out through the adapter.
            LogRecord::Reward(_) => "reward",
            LogRecord::Cycle(_) => "cycle",
        }
    }

    /// The payloads of the action records among `records`, in order.
    fn acted(records: &[LogRecord<TestDomain>]) -> Vec<TestPayload> {
        records
            .iter()
            .filter_map(|record| match record {
                LogRecord::Action(record) => Some(record.event.payload.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_cycle_takes_one_event_and_leaves_the_rest() {
        // Three events are waiting before the agent is even spawned, so
        // nothing about the scheduler decides what it finds: one cycle
        // could have had all three. It takes them one at a time, in queue
        // order, and answers each on its own (ADR-0008).
        let wires = wires(None);
        wires.control(Control::Start);
        for event in [step("b", 6), step("c", 3), step("b", 5)] {
            wires.send(event);
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        drop(wires.queue);

        let handler = agent.join().unwrap();
        // The control was the loop's, so the handler saw three observations
        // and one start, not four events.
        assert_eq!(handler.started, 1);
        assert_eq!(handler.seen, steps([6, 3, 5]));
        // Every cycle that took an event answered it, and no cycle
        // answered two: one observation, one reply.
        let dispatches: Vec<_> = wires.dispatches.iter().collect();
        assert!(
            dispatches.iter().all(|dispatch| dispatch.sent.len() <= 1),
            "no cycle answers more than its one observation: {dispatches:?}"
        );
        let payloads: Vec<&TestPayload> = dispatches
            .iter()
            .flat_map(|dispatch| dispatch.sent.iter().map(|event| &event.payload))
            .collect();
        assert_eq!(payloads, steps([7, 4, 6]).iter().collect::<Vec<_>>());
        // Every delivery is reported exactly once, however the cycles fell:
        // the start and the three events.
        let deliveries: usize = dispatches.iter().map(|dispatch| dispatch.deliveries).sum();
        assert_eq!(deliveries, 4);
    }

    #[test]
    fn a_handler_never_sees_a_control_but_the_loop_acts_on_it() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        rig.dispatch();
        rig.send(step("b", 1));
        rig.dispatch();
        rig.stop();
        let handler = rig.agent.join().unwrap();
        // The start called `start` once; the stop ended the loop; and
        // neither reached `handle`, which saw only the one observation.
        assert_eq!(handler.started, 1);
        assert_eq!(handler.seen, steps([1]));
    }

    #[test]
    fn a_cycle_with_only_controls_calls_neither_entry_point() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        let handler = {
            rig.dispatch();
            rig.stop();
            rig.agent.join().unwrap()
        };
        // Both cycles held only a control. The start hook ran, the loop
        // exited on the stop, and neither cycle observed anything, so
        // neither `handle` nor `timeout` was called at all: there is no
        // decision to be asked for where there was nothing to decide from.
        assert_eq!(handler.started, 1);
        assert!(handler.seen.is_empty(), "{:?}", handler.seen);
        assert_eq!(handler.timeouts, 0);
    }

    #[test]
    fn an_event_arriving_mid_cycle_waits_for_a_cycle_of_its_own() {
        let (rig, busy, release) = gated();
        rig.start();
        // The start's cycle observes nothing, so it never enters the
        // handler and there is no gate to open for it. The first event is
        // what puts the agent inside `handle`.
        rig.send(step("b", 1));
        recv(&busy);
        // Now the agent is provably mid-cycle, deciding about the first.
        // The second arrives with nowhere to go but the queue.
        rig.send(step("b", 2));
        release.send(()).unwrap();
        // It gets a cycle of its own, which is the claim: the loop came
        // back round for it rather than folding it into the cycle already
        // running.
        recv(&busy);
        release.send(()).unwrap();
        // Both cycles have dispatched before the stop goes on the queue,
        // so this test is about what waits for the next cycle and nothing
        // else.
        rig.dispatch();
        rig.dispatch();
        rig.stop();

        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.inner.seen, steps([1, 2]));
    }

    #[test]
    fn exits_after_the_cycle_that_contained_stop() {
        let wires = wires(None);
        wires.control(Control::Start);
        wires.control(Control::Stop);
        wires.send(step("b", 1));
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        // The test still holds both senders, so the join returning at all is
        // the stop path. The handler was called once, with nothing: the
        // event was queued behind a stop and so was never popped.
        let handler = agent.join().unwrap();
        assert_eq!(handler.started, 1);
        assert!(handler.seen.is_empty(), "{:?}", handler.seen);
        drop(wires.queue);
    }

    #[test]
    fn exits_when_both_queues_close_without_a_cycle() {
        let rig = rig(Recorder::default(), Some(EVERY));
        drop(rig.queue);
        let handler = rig.agent.join().unwrap();
        assert!(handler.seen.is_empty());
        assert_eq!(handler.timeouts, 0);
        assert!(rig.dispatches.try_recv().is_err());
        assert!(rig.records.try_recv().is_err());
    }

    #[test]
    fn the_queue_closing_ends_the_loop_after_what_was_on_it() {
        // One queue, so there is one thing that can close, and closing it
        // is how an agent nobody ever stops comes to an end. What was
        // already on it is still handled first: the close is learned by the
        // pop that found the queue empty, which threw nothing away.
        let rig = rig(Recorder::default(), None);
        rig.control(Control::Start);
        rig.send(step("b", 1));
        let queue = rig.queue;
        drop(queue);
        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.started, 1);
        assert_eq!(handler.seen, steps([1]));
    }

    #[test]
    fn a_timeout_calls_the_timeout_hook_and_not_the_handler() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.start();
        assert_eq!(rig.dispatch().deliveries, 1);
        let (_, started) = rig.cycle();
        assert_eq!(started.woken, Woken::Queue);
        let first = recv(rig.timer.requests());
        assert_eq!(first, started.t_start + EVERY);

        rig.send(step("b", 1));
        assert_eq!(rig.dispatch().deliveries, 1);
        rig.cycle();

        rig.timer.fire().unwrap();
        assert_eq!(rig.dispatch().deliveries, 0);
        let (records, timed_out) = rig.cycle();
        assert!(
            records.is_empty(),
            "a timeout cycle records nothing it popped: {records:?}"
        );
        assert_eq!(timed_out.woken, Woken::Timeout);
        assert!(timed_out.inputs.is_empty() && timed_out.outputs.is_empty());

        // The message did not move the deadline: the only request between
        // the first and the one made after the timeout is none at all.
        let second = recv(rig.timer.requests());
        assert_eq!(second, timed_out.t_start + EVERY);
        assert!(second > first);

        rig.stop();
        let handler = rig.agent.join().unwrap();
        // The deadline reached `timeout`, once, and `handle` saw only the
        // one thing that was actually observed. A wake-up on a deadline is
        // not an observation, and the two entry points are what say so.
        assert_eq!(handler.seen, steps([1]));
        assert_eq!(handler.timeouts, 1);
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn a_deadline_that_passed_while_busy_joins_the_cycle_that_observed() {
        let (rig, busy, release) = gated();
        rig.start();
        // Hold the agent inside a cycle, so that the deadline and the second
        // event are both queued before it next waits. Racing the two onto an
        // idle agent would let it wake on whichever arrived first and run a
        // bare timeout cycle, and which one that was would be the
        // scheduler's answer rather than the claim's.
        rig.send(step("b", 1));
        recv(&busy);
        rig.timer.fire().unwrap();
        rig.send(step("b", 2));
        release.send(()).unwrap();

        // The next wait finds both. The deadline has passed and there is
        // something to observe, so the cycle observes it and calls `handle`:
        // what a handler decides from is the observation, and the deadline
        // only says when the cycle ran.
        recv(&busy);
        release.send(()).unwrap();
        // The stop comes only once both cycles are over and have
        // dispatched, so nothing about it bears on the claim.
        rig.dispatch();
        rig.dispatch();
        rig.stop();

        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.inner.seen, steps([1, 2]));
        assert_eq!(
            handler.inner.timeouts, 0,
            "the deadline joined an observing cycle, so `timeout` was never called"
        );
    }

    #[test]
    fn an_agent_without_an_interval_never_times_out() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        rig.dispatch();
        rig.send(step("b", 1));
        rig.dispatch();
        rig.stop();
        rig.agent.join().unwrap();
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn timing_out_starts_only_once_the_episode_has() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.send(step("b", 1));
        rig.dispatch();
        rig.start();
        rig.dispatch();
        // The first request is made only after the start was popped, so it
        // is the first thing on the channel either way; what the test can
        // check is which cycle it was measured from.
        let deadline = recv(rig.timer.requests());
        rig.cycle();
        let (_, started) = rig.cycle();
        assert_eq!(deadline, started.t_start + EVERY);
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_cycle_is_recorded_as_its_inputs_then_its_outputs_then_the_cycle() {
        let mut wires = wires(None);
        let (records, writer) = Writer::spawn(Vec::new());
        wires.wiring.records = records;
        wires
            .queue
            .send(Delivery::control(Control::Start, at(10)))
            .unwrap();
        wires.send(step_at("b", 6, at(20)));
        let clock = wires.wiring.clock;
        let agent = Agent::spawn(wires.wiring, Recorder::default(), clock);
        drop(wires.queue);
        agent.join().unwrap();
        let lines = parse_lines(&writer.join().unwrap());

        assert_eq!(lines.len(), 4);
        let t_start = lines[3]["t_start"].as_u64().unwrap();
        let t_stop = lines[3]["t_stop"].as_u64().unwrap();
        assert_eq!(
            lines[0],
            json!({"type": "control", "agent": "a", "seq": 0, "created": 10,
                   "received": t_start, "control": "start"})
        );
        assert_eq!(
            lines[1],
            json!({"type": "observation", "agent": "a", "seq": 1, "created": 20,
                   "received": t_start,
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}})
        );
        let created = lines[2]["created"].as_u64().unwrap();
        assert_eq!(
            lines[2],
            json!({"type": "action", "agent": "a", "seq": 2, "created": created,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}})
        );
        assert_eq!(
            lines[3],
            json!({"type": "cycle", "agent": "a", "t_start": t_start, "t_stop": t_stop,
                   "woken": "queue", "inputs": [0, 1], "outputs": [2]})
        );
        // Everything popped was popped at the start of the cycle, and
        // everything sent was sent within its window.
        assert!(20 < t_start);
        assert!(t_start <= created && created <= t_stop);
    }

    #[test]
    fn an_observation_carries_the_senders_creation_time_and_this_agents_receipt() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        rig.cycle();
        rig.send(step_at("b", 1, at(7)));
        let (records, cycle) = rig.cycle();
        let LogRecord::Observation(observation) = &records[0] else {
            panic!("the first record of the cycle is the observation: {records:?}");
        };
        assert_eq!(observation.created, at(7), "the sender's stamp is carried");
        assert_eq!(
            observation.received, cycle.t_start,
            "and the receipt is the pop, which is the cycle's start"
        );
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn sequence_numbers_run_on_across_cycles_and_cover_every_kind() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        let (first, cycle) = rig.cycle();
        assert!(matches!(first.as_slice(), [LogRecord::Control(_)]));
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(0)], vec![]));
        rig.send(step("b", 1));
        let (second, cycle) = rig.cycle();
        assert!(matches!(
            second.as_slice(),
            [LogRecord::Observation(_), LogRecord::Action(_)]
        ));
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(1)], vec![Seq(2)]));
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_broadcast_is_sent_and_recorded_as_every_peer() {
        let rig = rig(Town, None);
        rig.start();
        let sent = rig.dispatch().sent;
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].recipients, ["b", "c"].map(AgentId::new).into());
        let (records, cycle) = rig.cycle();
        let LogRecord::Action(action) = &records[1] else {
            panic!("the broadcast is recorded as an action: {records:?}");
        };
        assert_eq!(action.event.recipients, ["b", "c"].map(AgentId::new).into());
        assert_eq!(cycle.outputs, [Seq(1)]);
        rig.stop();
        rig.agent.join().unwrap();
    }

    /// Answers one observation with many actions at once, which is where
    /// two of an agent's stamps could collide.
    struct Chatters(usize);

    impl Handler<TestDomain> for Chatters {
        fn handle(&mut self, _: &Observation<TestDomain>) -> Vec<Action<TestDomain>> {
            (0..self.0)
                .map(|n| Action::to(["b"], TestPayload::Step(n as u64)))
                .collect()
        }
    }

    #[test]
    fn an_agents_actions_never_share_an_instant() {
        // An observation names the action it came from by the sender and
        // the creation time alone, so two of one agent's actions at one
        // instant would be two an observation could not tell apart. A cycle
        // that returns a hundred actions returns them far faster than the
        // clock's resolution.
        let rig = rig(Chatters(100), None);
        rig.start();
        rig.cycle();
        rig.send(step("b", 1));
        let (records, cycle) = rig.cycle();
        let created: Vec<Timestamp> = records
            .iter()
            .filter_map(|record| match record {
                LogRecord::Action(record) => Some(record.created),
                _ => None,
            })
            .collect();
        assert_eq!(created.len(), 100);
        assert!(
            created.windows(2).all(|pair| pair[0] < pair[1]),
            "every action of a cycle is stamped later than the one before it: {created:?}"
        );
        // And the stamps still lie inside the cycle's window, which is what
        // the trajectory checker asserts of an output.
        assert!(
            created
                .iter()
                .all(|at| cycle.t_start <= *at && *at <= cycle.t_stop)
        );
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_vanished_writer_is_an_error() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        drop(rig.records);
        assert_eq!(rig.agent.join(), Err(Error::WriterClosed));
    }

    #[test]
    fn a_vanished_router_is_an_error() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        drop(rig.dispatches);
        assert_eq!(rig.agent.join(), Err(Error::RouterClosed));
    }

    #[test]
    fn a_vanished_timer_is_an_error() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.start();
        rig.dispatch();
        recv(rig.timer.requests());
        drop(rig.timer);
        assert_eq!(rig.agent.join(), Err(Error::TimerClosed));
    }

    #[test]
    #[should_panic(expected = "handler bug")]
    fn a_handler_panic_reaches_whoever_joins() {
        let rig = rig(Faulty, None);
        rig.start();
        // An observation is what reaches the handler: the start's own cycle
        // calls the start hook and nothing else.
        rig.send(step("b", 1));
        let _ = rig.agent.join();
    }

    #[test]
    fn the_thread_carries_the_agents_id() {
        let rig = rig(Recorder::default(), None);
        assert_eq!(rig.agent.id(), &AgentId::new("a"));
        assert_eq!(rig.agent.thread.thread().name(), Some("a"));
        drop(rig.queue);
        rig.agent.join().unwrap();
    }

    #[test]
    fn an_observation_and_a_control_know_their_own_latency() {
        let observation = Observation::<TestDomain> {
            event: step_at("b", 1, at(40)),
            received: at(55),
        };
        assert_eq!(observation.created(), at(40));
        assert_eq!(observation.received(), at(55));
        assert_eq!(observation.latency(), Duration::from_nanos(15));

        let instruction = Instruction {
            control: Control::Start,
            created: at(10),
            received: at(12),
        };
        assert_eq!(instruction.latency(), Duration::from_nanos(2));
    }

    #[test]
    #[should_panic(expected = "was started twice")]
    fn an_agent_started_twice_is_a_bug_in_whoever_started_it() {
        // An agent is started once, before anything is addressed to it. A
        // second start cannot be honored — the start hook has already run,
        // and running it again would reopen an agent that has been playing
        // — and recording the agent as started anyway would put a claim in
        // the trajectory the loop did not act on.
        let wires = wires(None);
        wires.control(Control::Start);
        wires.send(step("b", 1));
        wires.control(Control::Start);
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        let _ = agent.join();
    }

    #[test]
    fn a_stop_queued_behind_many_events_is_reached_behind_them() {
        // Everything is queued before the agent is spawned, so the order is
        // the queue's doing and nothing else: twenty events went on the
        // wire first and the stop last. With one queue that is what the
        // agent works through — a cycle per event, then the stop — which is
        // the ordering ADR-0009 accepts in place of ADR-0007's two queues.
        //
        // The episode never queues a stop this way except when it is
        // abandoning a run that has already failed: it holds one back until
        // nothing is in flight.
        let wires = wires(None);
        wires.control(Control::Start);
        for n in 1..=20 {
            wires.send(step("b", n));
        }
        wires.control(Control::Stop);
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        let handler = agent.join().unwrap();

        // Every one of the twenty was observed, in queue order, and each
        // was answered: nothing was swallowed by the stop behind them.
        assert_eq!(handler.started, 1);
        assert_eq!(
            handler.seen,
            steps(std::array::from_fn::<u64, 20, _>(|i| i as u64 + 1))
        );
        let records: Vec<LogRecord<TestDomain>> = wires.records.try_iter().collect();
        let controls: Vec<Control> = records
            .iter()
            .filter_map(|record| match record {
                LogRecord::Control(record) => Some(record.control),
                _ => None,
            })
            .collect();
        assert_eq!(controls, [Control::Start, Control::Stop]);
        assert_eq!(
            acted(&records),
            steps(std::array::from_fn::<u64, 20, _>(|i| i as u64 + 2)),
            "every observation was answered and every answer was sent"
        );
        // Every delivery is reported exactly once, however the cycles fell:
        // the start, the twenty events and the stop. An episode that never
        // hears of one waits forever for it, and one that hears of a
        // delivery twice panics subtracting it.
        let dispatches: Vec<_> = wires.dispatches.try_iter().collect();
        let deliveries: usize = dispatches.iter().map(|dispatch| dispatch.deliveries).sum();
        assert_eq!(deliveries, 22);
    }

    #[test]
    fn a_stop_leaves_what_is_behind_it_unpopped_and_unlogged() {
        // The other half of the same fact. Once the stop is in hand the
        // cycle takes no event, and the events still queued are never
        // popped: an agent that has stopped did not observe them, and a
        // trajectory that logged them would be claiming it did.
        let wires = wires(None);
        wires.control(Control::Start);
        wires.control(Control::Stop);
        for n in 1..=5 {
            wires.send(step("b", n));
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        // The test still holds the sender, so the join returning at all is
        // the stop path rather than the queue closing.
        let handler = agent.join().unwrap();
        assert_eq!(handler.started, 1);
        assert!(handler.seen.is_empty(), "{:?}", handler.seen);
        let records: Vec<LogRecord<TestDomain>> = wires.records.try_iter().collect();
        assert!(
            !records
                .iter()
                .any(|record| matches!(record, LogRecord::Observation(_))),
            "an event never popped is never logged as an observation: {records:?}"
        );
        // Only what was taken is counted: the two controls, and none of the
        // five events left on the queue.
        let dispatches: Vec<_> = wires.dispatches.try_iter().collect();
        let deliveries: usize = dispatches.iter().map(|dispatch| dispatch.deliveries).sum();
        assert_eq!(deliveries, 2);
        drop(wires.queue);
    }

    #[test]
    fn a_cycle_takes_the_controls_at_the_head_before_the_event_behind_them() {
        // One queue still does not mean one thing per cycle. A start and an
        // event waiting together are one cycle: the start hook runs first,
        // and the event is observed after it, which is the order they were
        // sent in and the order the records are written in.
        let wires = wires(None);
        wires.control(Control::Start);
        wires.send(step("b", 1));
        let agent = Agent::spawn(wires.wiring, Town, Clock::start());
        drop(wires.queue);
        agent.join().unwrap();

        let records: Vec<LogRecord<TestDomain>> = wires.records.try_iter().collect();
        let kinds: Vec<&str> = records.iter().map(kind).collect();
        // `Town` broadcasts when it starts and says nothing to an
        // observation, so the opening action between them is the start
        // hook's, which places the start ahead of the observation without
        // the test having to read two stamps that may be equal.
        assert_eq!(
            kinds,
            ["control", "observation", "action", "cycle"],
            "the start, then what was behind it, then what the start said"
        );
    }

    #[test]
    fn whatever_the_handler_returns_is_sent_even_with_a_stop_waiting() {
        // The claim ADR-0009 makes about every cycle: nothing arriving
        // while the handler runs changes what becomes of what it returns.
        // The stop is queued while the agent is provably inside `handle`,
        // behind the event it is deciding about, and the answer goes out
        // all the same.
        let (rig, busy, release) = gated();
        rig.start();
        rig.send(step("b", 1));
        recv(&busy);
        rig.stop();
        release.send(()).unwrap();

        let (records, cycle) = rig.cycle();
        assert_eq!(
            acted(&records),
            steps([2]),
            "the answer was sent, not withheld: {records:?}"
        );
        assert_eq!(cycle.outputs.len(), 1, "and it is an output: {cycle:?}");
        let dispatch = rig
            .dispatches
            .iter()
            .find(|dispatch| !dispatch.sent.is_empty())
            .expect("the answer reached the router");
        assert_eq!(dispatch.sent.len(), 1);
        rig.agent.join().unwrap();
    }

    #[test]
    fn everything_a_cycle_pops_shares_its_start() {
        // There is no longer any exception. A control is popped at the top
        // of the cycle like the observation beside it, so every input's
        // `received` is the cycle's `t_start` and a reader needs no special
        // case for one of them.
        let rig = rig(Recorder::default(), None);
        rig.start();
        rig.send(step("b", 1));
        rig.stop();
        rig.agent.join().unwrap();

        let records: Vec<LogRecord<TestDomain>> = rig.records.try_iter().collect();
        let mut cycles = 0;
        let mut received: Vec<Timestamp> = Vec::new();
        for record in &records {
            match record {
                LogRecord::Control(record) => received.push(record.received),
                LogRecord::Observation(record) => received.push(record.received),
                LogRecord::Cycle(cycle) => {
                    cycles += 1;
                    assert!(
                        received.iter().all(|at| *at == cycle.t_start),
                        "every input of a cycle was popped at its start: \
                         {received:?} against {cycle:?}"
                    );
                    received.clear();
                }
                _ => {}
            }
        }
        assert!(cycles >= 2, "the start and the stop each ran a cycle");
        assert!(received.is_empty(), "every record belongs to some cycle");
    }
}
