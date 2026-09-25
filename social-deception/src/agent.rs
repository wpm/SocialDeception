//! The agent: one thread, two queues, and a pop-fold-send loop.
//!
//! An agent's life is a fold over what arrives on its queues. Each cycle of
//! the loop:
//!
//! 1. waits until something arrives on either queue or its timeout fires;
//! 2. arms a fresh [`Cancel`] for this cycle;
//! 3. pops everything waiting, **controls first**, splitting it into
//!    [`Instruction`]s for the loop itself and [`Observation`]s for the
//!    handler;
//! 4. records each of them, stamped with the instant of the pop;
//! 5. hands the observations and the cancel to the game's [`Handler`] and
//!    gets back the [`Action`]s to send;
//! 6. looks at the control queue again. If a [`Stop`] arrived while the
//!    handler was deciding, the cycle sends nothing; otherwise it stamps
//!    each action with the agent's id and the instant of the send, records
//!    it, and sends the cycle's actions to the router as one
//!    [`CycleDispatch`];
//! 7. records the cycle.
//!
//! An event that arrives while the agent is busy waits its turn and is
//! picked up at the start of the next cycle, so every agent is always stale
//! by exactly one handling window. A control does not wait: see below.
//!
//! Everything a cycle pops at its start is stamped with that instant, so
//! every observation in a cycle, and every control the cycle popped before
//! calling the handler, has the same `received`: the cycle's `t_start`. That
//! is what makes an agent's deliberation recoverable from the log without a
//! stamp for it, as ADR-0007 sets out: it is a sent action's `created` minus
//! the `received` of the observations in the same cycle, and the cycle
//! record groups them. The one exception is the `Stop` that preempted a
//! cycle, which was popped after the handler returned and says so.
//!
//! The handler returns what to send rather than sending it, so the loop sees
//! everything that goes out and the trajectory it records is authoritative.
//! It also means a handler cannot speak as anybody but itself: the loop is
//! what writes the sender.
//!
//! # Two queues, and why controls go first
//!
//! Events and controls arrive on separate channels, and the loop waits on
//! both. At the top of every cycle it takes the controls before the events,
//! so a [`Stop`] is acted on before anything queued behind it rather than
//! after. Events still waiting when the agent stops are never popped, so
//! they never become observations and are never logged: an agent that has
//! stopped did not see them, and a trajectory that said otherwise would be
//! claiming it did.
//!
//! One inbox carrying both would make a control wait for whatever is ahead
//! of it, which for a batch of events is an ordering accident and for a slow
//! cycle is an unbounded delay. ADR-0007 rejected it for that reason.
//!
//! # A cycle preempted by `Stop` sends nothing
//!
//! Two queues get a control past a queue of events; they do nothing about a
//! control that arrives while the handler is *running*. That is what
//! [`Cancel`] is for. The loop arms one per cycle, whoever queues a control
//! trips it in the same step (see [`cancel`](crate::cancel)), and a handler
//! that is blocked can wait on it alongside whatever it is blocked on.
//!
//! When the handler returns, the loop looks at the control queue before it
//! sends anything. If a `Stop` is waiting there:
//!
//! - **none of the cycle's actions is sent.** Each is stamped as it would
//!   have been and logged as a [`DroppedRecord`] instead of an
//!   [`ActionRecord`]; [`DroppedRecord`] says why that is the honest record.
//!   Nothing is routed, so the episode's in-flight count never sees them;
//! - the `Stop` is then popped and logged among the cycle's `inputs`, with
//!   its `received` the instant it was popped, which is after the handler
//!   returned;
//! - the cycle's `outputs` are empty, and the loop exits.
//!
//! A handler that ignores its `Cancel` is preempted just the same: what
//! decides is that a `Stop` was waiting when the handler returned, not how
//! the handler came to return. Ignoring the cancel only makes the `Stop`
//! wait for the handler, which ADR-0007 calls a bug in the policy rather
//! than in the runtime.
//!
//! [`Stop`] is the only control there is. What a future control that
//! preempts a cycle *without* ending the episode should do with that cycle's
//! actions is deliberately undecided.
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
//! [`Start`]: Control::Start
//! [`Stop`]: Control::Stop
//!
//! # Timeouts
//!
//! An agent may be given a timeout. From the cycle in which it pops
//! [`Control::Start`], a deadline is pending one interval ahead; when it
//! passes, the agent runs a cycle with **no observations** — the handler is
//! called with an empty slice — and the next deadline is one interval after
//! that cycle. The cycle record says [`Woken::Timeout`], which is the only
//! way to tell such a cycle from one woken by an empty pop, since there is
//! no wake-up object to record any more.
//!
//! Deadlines are absolute, and the wake channel for one is asked of the
//! [`TimerSource`] once and kept until it fires, so something arriving
//! before the deadline leaves the deadline where it was. An agent without an
//! interval blocks until something arrives.
//!
//! # Termination
//!
//! The loop exits when both its queues close, meaning every sender has been
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
//!     Action, Agent, AgentId, Cancel, Clock, Control, ControlSender, CycleDispatch, Domain,
//!     Event, Handler, Observation, Wiring, Writer,
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
//!     fn handle(&mut self, observations: &[Observation<Chat>], _: &Cancel) -> Vec<Action<Chat>> {
//!         observations
//!             .iter()
//!             .map(|observation| {
//!                 Action::to([observation.event.sender.clone()], observation.event.payload.clone())
//!             })
//!             .collect()
//!     }
//! }
//!
//! let clock = Clock::start();
//! let (to_agent, events) = unbounded();
//! let (commander, controls, arm) = ControlSender::new();
//! let (dispatches, from_agent) = unbounded();
//! let (records, writer) = Writer::spawn::<Chat>(Vec::new());
//! let peers = [AgentId::new("caller")].into();
//! let wiring =
//!     Wiring { id: "echo".into(), clock, events, controls, arm, dispatches, records,
//!              timeout: None, peers };
//! let agent = Agent::spawn(wiring, Echo, clock);
//!
//! let hello = Event::<Chat>::new("caller", ["echo"], clock.now(), String::from("hello"));
//! commander.control(clock, Control::Start).unwrap();
//! to_agent.send(hello.clone()).unwrap();
//! // Wait for the reply before stopping: a `Stop` that arrived first would
//! // preempt the cycle, and the echo would be dropped rather than sent.
//! let echoed: CycleDispatch<Chat> = from_agent.iter().find(|r| !r.sent.is_empty()).unwrap();
//! commander.control(clock, Control::Stop).unwrap();
//!
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

use crate::cancel::{Arm, Cancel, Signal};
use crate::clock::{Clock, Created, Timestamp, Timestamped};
use crate::event::{AgentId, Control, Domain, Event};
use crate::timer::TimerSource;
use crate::trajectory::{
    ActionRecord, ControlRecord, CycleRecord, DroppedRecord, LogRecord, ObservationRecord, Seq,
    Woken,
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
/// that waited behind a slow cycle says so.
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
/// The loop calls [`handle`](Handler::handle) once per cycle with everything
/// the agent observed, in pop order, and sends what comes back. Agent state
/// lives in the implementing type. A handler never sees a [`Control`]; see
/// the [module documentation](self).
pub trait Handler<D: Domain> {
    /// The agent's opening actions, called once when it is started.
    ///
    /// This is where an agent that acts before anybody has spoken to it does
    /// so. Most agents only react, and the default returns nothing.
    ///
    /// It takes no [`Cancel`]: opening actions are the agent's own, decided
    /// from nothing, and a handler with work to do before it can name them
    /// has state to fold and belongs in `handle`.
    fn start(&mut self) -> Vec<Action<D>> {
        Vec::new()
    }

    /// Folds one cycle's observations into the agent's state and says what
    /// to send.
    ///
    /// Empty when the cycle was woken by the timeout.
    ///
    /// `cancel` is this cycle's, and it trips when a control is queued for
    /// this agent while the handler is running. A handler that thinks in
    /// steps checks [`Cancel::is_cancelled`] between them; one that blocks
    /// waits on [`Cancel::receiver`] alongside whatever it is blocked on and
    /// gives up on whichever comes first. Honoring it is a courtesy to the
    /// episode, not a condition of correctness: a cycle preempted by
    /// [`Control::Stop`] sends nothing either way, and a handler that
    /// ignores the cancel only makes the stop wait for it.
    fn handle(&mut self, observations: &[Observation<D>], cancel: &Cancel) -> Vec<Action<D>>;
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
    /// How many deliveries the cycle took off its queues. A timeout is not a
    /// delivery.
    pub deliveries: usize,
    /// The events the cycle sent, stamped with this agent as sender, in the
    /// order the handler returned them.
    pub sent: Vec<Event<D>>,
}

/// Everything an agent's thread needs besides its handler and timer.
///
/// The two queues are separate channels and the loop waits on both; the
/// [module documentation](self) says why they are not one. The [`Arm`] is
/// the other end of the coupling: it belongs to the same agent as
/// `controls`, and arming it each cycle is what lets whoever queues a
/// control preempt the cycle in progress.
pub struct Wiring<D: Domain> {
    /// The agent's id: the sender on everything it emits and the `agent` on
    /// every record it writes.
    pub id: AgentId,
    /// The episode clock.
    pub clock: Clock,
    /// The event queue: in-domain data, which becomes an [`Observation`]
    /// when popped.
    pub events: Receiver<Event<D>>,
    /// The control queue: out-of-domain instructions, which the loop acts on
    /// itself.
    pub controls: Receiver<Signal>,
    /// Where each cycle's [`Cancel`] is armed, so that a control queued
    /// during the cycle trips it.
    pub arm: Arm,
    /// Where each cycle's dispatch goes.
    pub dispatches: Sender<CycleDispatch<D>>,
    /// Where the agent's trajectory goes.
    pub records: Sender<LogRecord<D>>,
    /// How long the agent waits before running a cycle with no observations,
    /// or `None` for an agent that only ever reacts.
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
            closed: Closed::default(),
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
    /// A control arrived, and here it is.
    Control(Signal),
    /// An event arrived, and here it is.
    Event(Event<D>),
    /// The pending deadline passed.
    Deadline,
    /// A queue closed. Which one is not said, because the loop learns that
    /// from the drain that follows, which is where a closure is noticed
    /// without anything being thrown away.
    Closed,
    /// The wake channel disconnected.
    TimerGone,
}

/// What one cycle took off its queues, in the order the loop records it:
/// controls first, then events.
///
/// `deliveries` is not `controls.len() + events.len()`. Everything taken off
/// a queue is a delivery the router made and the episode is waiting to hear
/// about, including an event this cycle took and then forgot because a
/// `Stop` came with it. Counting what survived rather than what was taken
/// would leave the episode waiting forever for a delivery that had already
/// happened, so the count is made where the taking is and carried from
/// there.
struct Popped<D: Domain> {
    controls: Vec<Signal>,
    events: Vec<Event<D>>,
    deliveries: usize,
}

impl<D: Domain> Popped<D> {
    /// Whether the cycle popped nothing at all, which is how a wake-up that
    /// was only a queue closing is told from one that brought work.
    fn is_empty(&self) -> bool {
        self.controls.is_empty() && self.events.is_empty()
    }
}

/// Which of an agent's queues are closed, and so can never produce anything
/// again.
///
/// The loop keeps this across waits because a closure is learned once, by a
/// drain that found a queue disconnected, and is true forever after. Asking
/// again is not free: the only way to ask is `try_recv`, which would take
/// the very item that makes the answer no.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
struct Closed {
    controls: bool,
    events: bool,
}

impl Closed {
    /// Whether both are closed, which is the only state in which nothing
    /// could ever reach the agent again.
    fn both(self) -> bool {
        self.controls && self.events
    }
}

/// Takes everything waiting on one queue, and says whether it turned out to
/// be closed. Nothing is thrown away: the `try_recv` that reports the queue
/// disconnected is the one that found it empty.
fn drain_queue<T>(queue: &Receiver<T>, taken: &mut Vec<T>) -> bool {
    loop {
        match queue.try_recv() {
            Ok(item) => taken.push(item),
            Err(TryRecvError::Empty) => return false,
            Err(TryRecvError::Disconnected) => return true,
        }
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

    /// Which queues have closed. Sticky: learned from a drain and true from
    /// then on.
    closed: Closed,

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
                Wake::Control(_) | Wake::Event(_) | Wake::Closed => false,
            };
            // The cancel is armed before the drain, so that a control
            // arriving from here on trips this cycle rather than one that is
            // already over. Anything that arrived between the wake-up and
            // the arming is found by the drain, which is the same cycle's
            // business either way.
            let cancel = self.wiring.arm.arm();
            let popped = self.drain(woke_with);
            if popped.is_empty() && !woken_by_deadline {
                // A queue closed and brought nothing with it. There is no
                // cycle to run; whether there is anything left to wait for
                // is the next wait's question.
                if self.closed.both() {
                    break;
                }
                continue;
            }
            let timed_out = woken_by_deadline || self.deadline_passed()?;
            if timed_out {
                self.take_deadline();
            }
            let t_start = self.wiring.clock.now();
            let stop = self.cycle(t_start, popped, timed_out, &cancel)?;
            if stop || self.closed.both() {
                break;
            }
        }
        Ok(self.handler)
    }

    /// Blocks until something arrives on either queue or the pending
    /// deadline fires.
    ///
    /// A queue that has closed and emptied is a `select!` arm that is ready
    /// forever, which would spin, so a queue already known to be closed is
    /// swapped for one that is never ready. Knowing is the point: the answer
    /// comes from a drain, which learns it without taking anything, and is
    /// kept, because the only way to ask a channel directly is to try to
    /// receive from it.
    fn wait(&self) -> Wake<D> {
        let (idle, quiet, spent) = (never(), never(), never());
        let deadline = self.pending.as_ref().map_or(&idle, |(_, wake)| wake);
        let controls = if self.closed.controls {
            &quiet
        } else {
            &self.wiring.controls
        };
        let events = if self.closed.events {
            &spent
        } else {
            &self.wiring.events
        };
        select! {
            recv(controls) -> control => control.map_or(Wake::Closed, Wake::Control),
            recv(events) -> event => event.map_or(Wake::Closed, Wake::Event),
            recv(deadline) -> fired => if fired.is_ok() { Wake::Deadline } else { Wake::TimerGone },
        }
    }

    /// Takes everything waiting on both queues, **controls first**, adding it
    /// to what the wake-up already had in hand, and remembers either queue
    /// that turned out to be closed.
    ///
    /// The order is the whole point of there being two queues: a `Stop` is
    /// popped ahead of every event waiting behind it, so it is acted on
    /// rather than queued. And once a `Stop` is in hand the event queue is
    /// not drained at all, because this cycle is the agent's last: an event
    /// left on the queue is one the agent never saw, and it is not an
    /// observation and is not logged. Taking it only to hand it to a handler
    /// whose output will be dropped would put a decision in the trajectory
    /// that went nowhere and was made after the episode had ended.
    fn drain(&mut self, woke_with: Wake<D>) -> Popped<D> {
        let (mut controls, mut events) = match woke_with {
            Wake::Control(control) => (vec![control], Vec::new()),
            Wake::Event(event) => (Vec::new(), vec![event]),
            // A deadline or a closed queue brings nothing with it; what the
            // drain finds is the whole of the cycle.
            Wake::Deadline | Wake::Closed => (Vec::new(), Vec::new()),
            Wake::TimerGone => unreachable!("the caller returns on a gone timer"),
        };
        if !self.closed.controls {
            self.closed.controls = drain_queue(&self.wiring.controls, &mut controls);
        }
        if controls
            .iter()
            .any(|signal| signal.control == Control::Stop)
        {
            // A `Stop` is in hand, so this cycle is the agent's last and no
            // event belongs in it. The queue is left alone, and the one
            // event the wake-up may have taken before the stop was seen goes
            // back the only way it can: it is forgotten. That is the same
            // fact as the ones still queued — an event the agent never
            // observed — and the trajectory says the same thing about both,
            // which is nothing. It was still delivered, though, so it is
            // still counted.
            let deliveries = controls.len() + events.len();
            events.clear();
            return Popped {
                controls,
                events,
                deliveries,
            };
        }
        if !self.closed.events {
            self.closed.events = drain_queue(&self.wiring.events, &mut events);
        }
        let deliveries = controls.len() + events.len();
        Popped {
            controls,
            events,
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

    /// Runs one cycle: record what was popped, hand the observations and the
    /// cancel to the handler, send and record what comes back unless a
    /// `Stop` arrived meanwhile, and close with the cycle record. Returns
    /// whether the cycle popped a stop.
    fn cycle(
        &mut self,
        t_start: Timestamp,
        popped: Popped<D>,
        timed_out: bool,
        cancel: &Cancel,
    ) -> Result<bool, Error> {
        let Popped {
            controls,
            events,
            mut deliveries,
        } = popped;
        let (mut inputs, mut started, mut stopped) = (Vec::new(), false, false);
        // Controls first, and in one pass, so that a `Stop` already waiting
        // is acted on ahead of every event behind it rather than after them.
        for Signal { control, created } in controls {
            match control {
                Control::Start => started = true,
                Control::Stop => stopped = true,
            }
            inputs.push(self.record_control(&Instruction {
                control,
                created,
                received: t_start,
            })?);
        }
        let mut observations = Vec::with_capacity(events.len());
        for event in events {
            let observation = Observation {
                event,
                received: t_start,
            };
            inputs.push(self.record_observation(&observation)?);
            observations.push(observation);
        }
        if timed_out {
            self.schedule_timeout(t_start);
        }

        // A start is the loop's own business: it calls the start hook, whose
        // opening actions go out with whatever the cycle's observations also
        // produced, in that order, since the start was popped before them.
        let mut actions = if started {
            self.schedule_timeout(t_start);
            self.handler.start()
        } else {
            Vec::new()
        };
        actions.extend(self.handler.handle(&observations, cancel));

        // The handler has returned; the question now is whether a `Stop`
        // arrived while it was deciding. It is asked of the queue and not of
        // the cancel, because the cancel says only that *some* control was
        // queued, and because a handler that ignored its cancel is preempted
        // just the same.
        let mut late = Vec::new();
        if !self.closed.controls {
            self.closed.controls = drain_queue(&self.wiring.controls, &mut late);
        }
        deliveries += late.len();
        let preempted = stopped || late.iter().any(|signal| signal.control == Control::Stop);

        let (mut sent, mut outputs) = (Vec::new(), Vec::new());
        for action in actions {
            let event = self.stamp(action);
            if preempted {
                // Not sent, so not an output of the cycle and not routed:
                // the episode's in-flight count never sees it.
                self.record_dropped(&event)?;
            } else {
                outputs.push(self.record_action(&event)?);
                sent.push(event);
            }
        }

        // The late controls are recorded after the dropped actions, with the
        // instant they were popped, which is after the handler returned.
        // That is the one place a cycle's inputs do not all share its
        // `t_start`, and the trajectory checker knows it.
        let received = self.wiring.clock.now();
        for Signal { control, created } in late {
            match control {
                Control::Stop => stopped = true,
                // The start hook ran before the handler did, so a start
                // arriving behind it cannot be honored: the agent would be
                // recorded as started with its opening actions never asked
                // for. An agent is started once, before anything is
                // addressed to it, and whoever sent this one twice or late
                // has a bug the trajectory must not paper over.
                Control::Start => {
                    panic!("{} was started during a cycle", self.wiring.id)
                }
            }
            inputs.push(self.record_control(&Instruction {
                control,
                created,
                received,
            })?);
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
    /// For an action that is sent, the stamp is its `created` on the wire.
    /// For one that is dropped, it is the instant the handler returned it,
    /// which is what its record carries: a dropped action is stamped exactly
    /// as it would have been, so that the two records differ only in what
    /// they say happened.
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

    fn record_dropped(&mut self, event: &Event<D>) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            DroppedRecord {
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
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use crossbeam_channel::unbounded;
    use serde_json::json;

    use super::*;
    use crate::cancel::ControlSender;
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

    /// The payloads of a batch of observations, which is what the handlers
    /// below remember and what most tests assert on.
    fn payloads(observations: &[Observation<TestDomain>]) -> Vec<TestPayload> {
        observations
            .iter()
            .map(|observation| observation.event.payload.clone())
            .collect()
    }

    fn recv<T>(receiver: &Receiver<T>) -> T {
        receiver.recv_timeout(PATIENCE).expect("nothing arrived")
    }

    /// Remembers every batch of observations it was given and replies to
    /// each with the next step, addressed to whoever sent it. It also
    /// remembers whether it was started, so that a test can see that the
    /// loop, not the handler, acts on a control.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Recorder {
        started: usize,
        batches: Vec<Vec<TestPayload>>,
    }

    impl Handler<TestDomain> for Recorder {
        fn start(&mut self) -> Vec<Action<TestDomain>> {
            self.started += 1;
            Vec::new()
        }

        fn handle(
            &mut self,
            observations: &[Observation<TestDomain>],
            _: &Cancel,
        ) -> Vec<Action<TestDomain>> {
            self.batches.push(payloads(observations));
            observations
                .iter()
                .map(|observation| {
                    let TestPayload::Step(n) = observation.event.payload;
                    Action::to([observation.event.sender.clone()], TestPayload::Step(n + 1))
                })
                .collect()
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
        fn handle(
            &mut self,
            observations: &[Observation<TestDomain>],
            cancel: &Cancel,
        ) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.handle(observations, cancel)
        }
    }

    /// Broadcasts a step when it starts.
    struct Town;

    impl Handler<TestDomain> for Town {
        fn start(&mut self) -> Vec<Action<TestDomain>> {
            vec![Action::broadcast(TestPayload::Step(0))]
        }

        fn handle(&mut self, _: &[Observation<TestDomain>], _: &Cancel) -> Vec<Action<TestDomain>> {
            Vec::new()
        }
    }

    /// A panicking handler.
    struct Faulty;

    impl Handler<TestDomain> for Faulty {
        fn handle(&mut self, _: &[Observation<TestDomain>], _: &Cancel) -> Vec<Action<TestDomain>> {
            panic!("handler bug");
        }
    }

    /// A handler that blocks on its cancel and nothing else, then answers
    /// every observation it was given. It tells the test when it entered the
    /// wait, so the test can put a control on the queue at a moment it knows
    /// the agent is inside `handle`.
    ///
    /// This is the shape ADR-0007 asks of a model-backed policy, with the
    /// model's answer left out: a wait on a result and on the cancel, which
    /// here has only the one arm because the result never comes.
    struct Waits {
        entered: Sender<()>,
        woke_cancelled: Sender<bool>,
    }

    impl Handler<TestDomain> for Waits {
        fn handle(
            &mut self,
            observations: &[Observation<TestDomain>],
            cancel: &Cancel,
        ) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            // Nothing is ever sent on this, so the cancel is the only thing
            // that can end the wait.
            let (_never, answer) = unbounded::<()>();
            select! {
                recv(answer) -> _ => {}
                recv(cancel.receiver()) -> _ => {}
            }
            self.woke_cancelled.send(cancel.is_cancelled()).unwrap();
            observations
                .iter()
                .map(|observation| {
                    let TestPayload::Step(n) = observation.event.payload;
                    Action::to([observation.event.sender.clone()], TestPayload::Step(n + 1))
                })
                .collect()
        }
    }

    /// A handler that never looks at its cancel: it waits to be released by
    /// the test and then answers, whatever has happened meanwhile.
    struct Deaf {
        entered: Sender<()>,
        release: Receiver<()>,
        cancelled_when_done: Sender<bool>,
    }

    impl Handler<TestDomain> for Deaf {
        fn handle(
            &mut self,
            observations: &[Observation<TestDomain>],
            cancel: &Cancel,
        ) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.cancelled_when_done
                .send(cancel.is_cancelled())
                .unwrap();
            observations
                .iter()
                .map(|observation| {
                    let TestPayload::Step(n) = observation.event.payload;
                    Action::to([observation.event.sender.clone()], TestPayload::Step(n + 1))
                })
                .collect()
        }
    }

    /// A handler that polls its cancel throughout a cycle and reports every
    /// answer it got, so that a test can say what was true for the whole of
    /// a cycle rather than only at its end.
    #[derive(Debug, Default)]
    struct Polls {
        answers: Vec<bool>,
        cycles: AtomicUsize,
    }

    impl Handler<TestDomain> for Polls {
        fn handle(
            &mut self,
            _: &[Observation<TestDomain>],
            cancel: &Cancel,
        ) -> Vec<Action<TestDomain>> {
            self.cycles.fetch_add(1, Ordering::Relaxed);
            for _ in 0..50 {
                self.answers.push(cancel.is_cancelled());
            }
            Vec::new()
        }
    }

    /// An agent and the test's end of every channel it is wired to.
    struct Rig<H> {
        agent: Agent<H>,
        clock: Clock,
        events: Sender<TestEvent>,
        controls: ControlSender,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
        timer: ManualTimerControl,
    }

    /// The channels of a rig, before the agent is spawned on them.
    struct Wires {
        wiring: Wiring<TestDomain>,
        events: Sender<TestEvent>,
        controls: ControlSender,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
    }

    impl Wires {
        fn control(&self, control: Control) {
            self.controls.control(self.wiring.clock, control).unwrap();
        }

        fn send(&self, event: TestEvent) {
            self.events.send(event).unwrap();
        }
    }

    fn wires(timeout: Option<Duration>) -> Wires {
        let (events, receiver) = unbounded();
        let (commander, controls, arm) = ControlSender::new();
        let (outbox, dispatches) = unbounded();
        let (recorder, records) = unbounded();
        let wiring = Wiring {
            id: AgentId::new("a"),
            clock: Clock::start(),
            events: receiver,
            controls,
            arm,
            dispatches: outbox,
            records: recorder,
            timeout,
            peers: ["b", "c"].map(AgentId::new).into(),
        };
        Wires {
            wiring,
            events,
            controls: commander,
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
            events: wires.events,
            controls: wires.controls,
            dispatches: wires.dispatches,
            records: wires.records,
            timer: control,
        }
    }

    impl<H> Rig<H> {
        fn send(&self, event: TestEvent) {
            self.events.send(event).unwrap();
        }

        fn control(&self, control: Control) {
            self.controls.control(self.clock, control).unwrap();
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

    /// The payloads of the dropped records among `records`, in order.
    fn dropped(records: &[LogRecord<TestDomain>]) -> Vec<TestPayload> {
        records
            .iter()
            .filter_map(|record| match record {
                LogRecord::Dropped(record) => Some(record.event.payload.clone()),
                _ => None,
            })
            .collect()
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
    fn a_cycle_pops_everything_waiting_on_both_queues() {
        let wires = wires(None);
        wires.control(Control::Start);
        for event in [step("b", 6), step("c", 3), step("b", 5)] {
            wires.send(event);
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        drop((wires.events, wires.controls));

        let handler = agent.join().unwrap();
        // The control was the loop's, so the handler saw three observations
        // and one start, not four events.
        assert_eq!(handler.started, 1);
        assert_eq!(handler.batches, [steps([6, 3, 5])]);
        let dispatches: Vec<_> = wires.dispatches.iter().collect();
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].deliveries, 4);
        let payloads: Vec<&TestPayload> = dispatches[0]
            .sent
            .iter()
            .map(|event| &event.payload)
            .collect();
        assert_eq!(payloads, steps([7, 4, 6]).iter().collect::<Vec<_>>());
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
        let seen: Vec<TestPayload> = handler.batches.into_iter().flatten().collect();
        assert_eq!(seen, steps([1]));
    }

    #[test]
    fn a_cycle_with_only_controls_calls_the_handler_with_nothing() {
        let rig = rig(Recorder::default(), None);
        rig.start();
        let handler_saw_empty = {
            rig.dispatch();
            rig.stop();
            rig.agent.join().unwrap()
        };
        // Both cycles held only a control, and the handler was still called
        // once per cycle, each time with no observations.
        assert_eq!(handler_saw_empty.batches, [Vec::new(), Vec::new()]);
    }

    #[test]
    fn events_arriving_mid_cycle_wait_for_the_next_cycle() {
        let (rig, busy, release) = gated();
        rig.start();
        recv(&busy);
        rig.send(step("b", 1));
        rig.send(step("b", 2));
        release.send(()).unwrap();
        recv(&busy);
        release.send(()).unwrap();
        // The stop waits for the cycle it would otherwise preempt to have
        // dispatched, so that this test is about what waits for the next
        // cycle and not about what a preemption drops.
        rig.dispatch();
        rig.dispatch();
        rig.stop();
        recv(&busy);
        release.send(()).unwrap();

        let handler = rig.agent.join().unwrap();
        assert_eq!(
            handler.inner.batches,
            [Vec::new(), steps([1, 2]), Vec::new()]
        );
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
        assert_eq!(handler.batches, [Vec::new()]);
        drop((wires.events, wires.controls));
    }

    #[test]
    fn exits_when_both_queues_close_without_a_cycle() {
        let rig = rig(Recorder::default(), Some(EVERY));
        drop((rig.events, rig.controls));
        let handler = rig.agent.join().unwrap();
        assert!(handler.batches.is_empty());
        assert!(rig.dispatches.try_recv().is_err());
        assert!(rig.records.try_recv().is_err());
    }

    #[test]
    fn one_queue_closing_does_not_end_the_loop() {
        // An agent whose event queue has gone still has to hear a `Stop`,
        // and one whose control queue has gone still has to handle what is
        // said to it.
        let deaf = rig(Recorder::default(), None);
        deaf.start();
        deaf.dispatch();
        let (events, controls, agent) = (deaf.events, deaf.controls, deaf.agent);
        drop(events);
        controls.control(deaf.clock, Control::Stop).unwrap();
        assert_eq!(agent.join().unwrap().started, 1);

        let mute = rig(Recorder::default(), None);
        mute.start();
        mute.dispatch();
        let (events, controls, agent) = (mute.events, mute.controls, mute.agent);
        drop(controls);
        events.send(step("b", 1)).unwrap();
        assert_eq!(recv(&mute.dispatches).deliveries, 1);
        drop(events);
        let handler = agent.join().unwrap();
        assert_eq!(handler.batches, [Vec::new(), steps([1])]);
    }

    #[test]
    fn a_timeout_runs_a_cycle_with_no_observations_when_the_deadline_passes() {
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
        // The handler was called on the timeout cycle too, with nothing.
        assert_eq!(
            handler.batches,
            [Vec::new(), steps([1]), Vec::new(), Vec::new()]
        );
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn a_deadline_that_passed_while_busy_is_handled_with_what_arrived() {
        let (rig, busy, release) = gated();
        rig.start();
        recv(&busy);
        rig.send(step("b", 1));
        // The fire is queued before the deadline is even armed; the manual
        // timer keeps it for the first wait.
        rig.timer.fire().unwrap();
        release.send(()).unwrap();
        recv(&busy);
        release.send(()).unwrap();
        // As above: the stop comes only once the cycle it could preempt is
        // over and has dispatched.
        rig.dispatch();
        rig.dispatch();
        rig.stop();
        recv(&busy);
        release.send(()).unwrap();

        let handler = rig.agent.join().unwrap();
        // The timeout did not add an observation; it joined the cycle that
        // the message woke, which the cycle record marks as a timeout.
        assert_eq!(handler.inner.batches, [Vec::new(), steps([1]), Vec::new()]);
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
            .controls
            .send(Signal {
                control: Control::Start,
                created: at(10),
            })
            .unwrap();
        wires.send(step_at("b", 6, at(20)));
        let clock = wires.wiring.clock;
        let agent = Agent::spawn(wires.wiring, Recorder::default(), clock);
        drop((wires.events, wires.controls));
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
        fn handle(&mut self, _: &[Observation<TestDomain>], _: &Cancel) -> Vec<Action<TestDomain>> {
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
        let _ = rig.agent.join();
    }

    #[test]
    fn the_thread_carries_the_agents_id() {
        let rig = rig(Recorder::default(), None);
        assert_eq!(rig.agent.id(), &AgentId::new("a"));
        assert_eq!(rig.agent.thread.thread().name(), Some("a"));
        drop((rig.events, rig.controls));
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

    /// A rig whose handler blocks on its cancel, with the channel it says
    /// it has entered `handle` on and the one it reports on.
    fn waiting() -> (Rig<Waits>, Receiver<()>, Receiver<bool>) {
        let (entered, inside) = unbounded();
        let (woke_cancelled, woke) = unbounded();
        let handler = Waits {
            entered,
            woke_cancelled,
        };
        (rig(handler, None), inside, woke)
    }

    #[test]
    fn a_handler_blocked_on_its_cancel_wakes_when_a_stop_is_queued() {
        let (rig, inside, woke) = waiting();
        rig.start();
        // The start's own cycle blocks too; release it by stopping nothing
        // and letting it see the queue empty. Instead, step past it: the
        // cancel of the start's cycle is tripped by the event that follows
        // only if that event were a control, which it is not, so the start
        // cycle would block forever. Send the stop straight away and let the
        // start cycle be the one that is preempted.
        recv(&inside);
        rig.stop();
        // The wait ends promptly, and the handler saw the cancel tripped.
        assert!(
            woke.recv_timeout(PATIENCE).expect("the handler woke"),
            "the handler woke because its cancel had tripped"
        );
        rig.agent.join().unwrap();
    }

    #[test]
    #[should_panic(expected = "was started during a cycle")]
    fn an_agent_started_during_a_cycle_is_a_bug_in_whoever_started_it() {
        // An agent is started once, before anything is addressed to it, so
        // the start hook runs before the handler. A start behind it cannot
        // be honored — the opening actions would never be asked for — and
        // recording the agent as started anyway would put a claim in the
        // trajectory the loop did not act on.
        let (rig, inside, _woke) = waiting();
        rig.start();
        recv(&inside);
        rig.control(Control::Start);
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_preempted_cycles_actions_are_dropped_and_none_reaches_the_router() {
        let (rig, inside, woke) = waiting();
        rig.start();
        recv(&inside);
        // Preempt the start's own cycle so the loop gets past it, then set
        // up the cycle the test is about.
        rig.stop();
        assert!(recv(&woke));

        let (records, cycle) = rig.cycle();
        assert!(
            dropped(&records).is_empty() && acted(&records).is_empty(),
            "the start cycle produced nothing to drop: {records:?}"
        );
        assert!(cycle.outputs.is_empty());
        let dispatch = rig.dispatch();
        assert!(dispatch.sent.is_empty());
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_stop_queued_behind_many_events_is_acted_on_first() {
        // Everything is queued before the agent is spawned, so there is one
        // cycle and the order within it is the queues' doing and nothing
        // else: twenty events went on the wire first and the stop last.
        let wires = wires(None);
        wires.control(Control::Start);
        for n in 1..=20 {
            wires.send(step("b", n));
        }
        wires.control(Control::Stop);
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        let handler = agent.join().unwrap();

        // One cycle, in which the handler saw nothing: the stop was popped
        // ahead of every event behind it, so the twenty events were never
        // popped at all.
        assert_eq!(handler.batches, [Vec::new()]);
        let records: Vec<LogRecord<TestDomain>> = wires.records.try_iter().collect();
        assert!(
            !records
                .iter()
                .any(|record| matches!(record, LogRecord::Observation(_))),
            "an event never popped is never logged as an observation: {records:?}"
        );
        let controls: Vec<Control> = records
            .iter()
            .filter_map(|record| match record {
                LogRecord::Control(record) => Some(record.control),
                _ => None,
            })
            .collect();
        assert_eq!(controls, [Control::Start, Control::Stop]);
        let dispatches: Vec<_> = wires.dispatches.try_iter().collect();
        assert_eq!(
            dispatches.len(),
            1,
            "one cycle, not one per queued event: {dispatches:?}"
        );
        // The two controls, and whichever single event the wake-up had
        // already taken before the stop was seen — that one is forgotten
        // rather than observed, but it was routed, so the cycle still
        // reports it. Which of the two happened is the scheduler's
        // business; that every routed delivery is reported exactly once is
        // not, because an episode that never hears of one waits forever.
        assert!(
            matches!(dispatches[0].deliveries, 2 | 3),
            "the cycle reports the controls it took and any event it dropped: {:?}",
            dispatches[0]
        );
        drop((wires.events, wires.controls));
    }

    #[test]
    fn a_handler_that_ignores_its_cancel_still_has_its_actions_dropped() {
        let (entered, inside) = unbounded();
        let (release, released) = unbounded();
        let (cancelled_when_done, seen) = unbounded();
        let rig = rig(
            Deaf {
                entered,
                release: released,
                cancelled_when_done,
            },
            None,
        );
        rig.start();
        recv(&inside);
        release.send(()).unwrap();
        assert!(!recv(&seen), "the start's cycle was never cancelled");
        rig.cycle();
        rig.dispatch();

        rig.send(step("b", 1));
        recv(&inside);
        // The stop arrives while the handler is deliberating, and the
        // handler pays it no attention at all.
        rig.stop();
        release.send(()).unwrap();
        assert!(recv(&seen), "the cancel had tripped, unread");

        let (records, cycle) = rig.cycle();
        assert_eq!(
            dropped(&records),
            steps([2]),
            "what the handler returned was dropped: {records:?}"
        );
        assert!(acted(&records).is_empty(), "and none of it was sent");
        assert!(
            cycle.outputs.is_empty(),
            "a preempted cycle has no outputs: {cycle:?}"
        );
        let dispatch = rig.dispatch();
        assert!(
            dispatch.sent.is_empty(),
            "nothing reached the router: {dispatch:?}"
        );
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_preempting_stop_is_logged_in_the_cycle_it_preempted_after_the_drops() {
        let (entered, inside) = unbounded();
        let (release, released) = unbounded();
        let (cancelled_when_done, seen) = unbounded();
        let rig = rig(
            Deaf {
                entered,
                release: released,
                cancelled_when_done,
            },
            None,
        );
        rig.start();
        recv(&inside);
        release.send(()).unwrap();
        recv(&seen);
        rig.cycle();
        rig.dispatch();

        rig.send(step("b", 1));
        recv(&inside);
        rig.stop();
        release.send(()).unwrap();
        recv(&seen);

        let (records, cycle) = rig.cycle();
        // The order on the wire: the observation popped at t_start, the
        // action that was dropped, then the stop, popped after the handler
        // returned and stamped with that later instant.
        let kinds: Vec<&str> = records
            .iter()
            .map(|record| match record {
                LogRecord::Observation(_) => "observation",
                LogRecord::Action(_) => "action",
                LogRecord::Dropped(_) => "dropped",
                LogRecord::Control(_) => "control",
                // An agent's loop never writes one: a reward is the
                // environment's, and it goes out through the adapter.
                LogRecord::Reward(_) => "reward",
                LogRecord::Cycle(_) => "cycle",
            })
            .collect();
        assert_eq!(kinds, ["observation", "dropped", "control"]);
        let (LogRecord::Dropped(gone), LogRecord::Control(stop)) = (&records[1], &records[2])
        else {
            unreachable!("the kinds were just checked");
        };
        assert_eq!(stop.control, Control::Stop);
        assert!(
            gone.created <= stop.received,
            "the stop was popped after the handler returned what was dropped"
        );
        assert!(
            cycle.t_start < stop.received,
            "which is later than the cycle's start: {cycle:?}"
        );
        assert_eq!(
            cycle.inputs,
            [Seq(1), Seq(3)],
            "the stop is among the cycle's inputs, after the observation"
        );
        assert_eq!(
            (gone.seq, stop.seq),
            (Seq(2), Seq(3)),
            "the dropped action took a sequence number between them"
        );
        assert!(cycle.outputs.is_empty());
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_cycle_no_control_interrupts_is_never_cancelled() {
        let rig = rig(Polls::default(), None);
        rig.start();
        rig.cycle();
        for n in 1..=5 {
            rig.send(step("b", n));
            rig.cycle();
        }
        rig.stop();
        let handler = rig.agent.join().unwrap();
        // Every poll of every cycle but the last said no. The last cycle is
        // the one that popped the stop, and its cancel was never tripped
        // either: the stop was already on the queue when the cycle began.
        //
        // How many cycles that took is the scheduler's business, not the
        // claim: a cycle drains whatever has arrived, so two steps landing
        // together are one cycle rather than two. What must hold is that
        // every cycle ran and none of them was cancelled.
        let cycles = handler.cycles.load(Ordering::Relaxed);
        assert!(
            (2..=7).contains(&cycles),
            "the start, at least one step and the stop each ran: {cycles}"
        );
        assert!(
            handler.answers.iter().all(|cancelled| !cancelled),
            "no poll of any cycle saw a cancel"
        );
    }
}
