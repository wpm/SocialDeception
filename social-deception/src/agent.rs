//! The agent: one thread, one queue, and a pop-fold-send loop.
//!
//! An agent's life is a fold over what arrives on its queue. Each cycle of
//! the loop:
//!
//! 1. waits until something arrives on the queue or its timeout fires;
//! 2. pops what is waiting: every control at the head of the queue and,
//!    behind them, **at most one** message, splitting them into
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
//! **A cycle handles one observation** (ADR-0008). An agent with a full queue
//! runs a cycle per message rather than one cycle for all of them, so it is
//! stale by at most one decision, and a cycle's window brackets exactly one.
//! What a drain found together was the scheduler's grouping and never the
//! game's; a game that means several messages together says so in one
//! payload.
//!
//! A message that arrives while the agent is busy waits its turn and is
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
//! everything that goes out and the log it records is authoritative.
//! It also means a handler cannot speak as anybody but itself: the loop is
//! what writes the sender.
//!
//! # One queue, and what that settles
//!
//! Messages and controls arrive on one channel, as [`Delivery`]s, and the
//! loop waits on it alone (ADR-0009). A `Stop` is delivered like anything
//! else and is acted on when the agent reaches it: the agent is not told
//! that a stop is coming, it cannot act on the knowledge, and no handler is
//! offered a way to. Whatever a cycle's handler returns is always sent.
//!
//! ADR-0007 gave controls a queue of their own and a cancellation
//! trip-wire, for two reasons neither of which survived. A control queued
//! behind a *batch* of messages waited for the batch — but ADR-0008 left no
//! batch. And a `Stop` behind a slow cycle waited for the cycle — but a
//! handler that blocks for thirty seconds is a defect wherever it appears,
//! and the place to fix it is inside the handler rather than in every
//! agent's loop forever.
//!
//! What the single queue keeps is the order things were sent in, which is
//! the order they are handled in. A `Stop` behind *n* messages is reached
//! after those messages, one cycle each. The episode does not produce that
//! arrangement on any path but failure: it holds a `Stop` back until its
//! in-flight count reads zero, which is to say until everything already
//! said has been handled (see [`episode`](crate::episode)). On the
//! abandon-ship path it cannot wait, and then an agent may answer messages
//! queued ahead of the stop for an episode that has already failed. That is
//! accepted (ADR-0009): keeping a log tidy through a failure is the
//! environment's job, since it decides when to stop whom.
//!
//! # Controls still come before messages within a cycle
//!
//! One queue does not mean one thing per cycle. A cycle takes every control
//! at the head of the queue before it takes a message, so a cycle that finds
//! `[Start, message]` waiting pops the start, runs the start hook and then
//! observes the message, in that order, rather than observing first. And once
//! a `Stop` is in hand the message behind it is left where it is: this cycle
//! is the agent's last, and a message the agent never popped is one it never
//! observed, so it is never logged. A log that said otherwise would
//! be claiming the agent saw something it did not.
//!
//! # The handler never sees a control
//!
//! A [`Control`] is out-of-domain, an instruction about the episode rather
//! than a move within the game, so the loop acts on it itself. [`Start`]
//! makes it call [`Handler::start`], whose opening actions are sent like any
//! others; [`Stop`] makes it exit after the cycle that popped it. Either way
//! the control is logged, so a reader sees it in the log even though
//! no handler did.
//!
//! A `Start` that arrives after the agent has started is a bug in whoever
//! sent it, and the loop panics rather than record a start it did not act
//! on.
//!
//! [`Start`]: Control::Start
//! [`Stop`]: Control::Stop
//!
//! # Deadlines
//!
//! **After the start hook and after every cycle, the loop asks the handler
//! when it next wants waking** (ADR-0010): [`Handler::deadline`] names an
//! absolute instant on the agent's clock, or `None`. When a deadline
//! passes, the agent runs a cycle that calls [`Handler::timeout`] rather
//! than [`Handler::handle`]. A deadline is a separate method because waking
//! on one is not observing anything (ADR-0008); the default does nothing.
//! The cycle record says [`Woken::Timeout`], which is the only way to tell
//! such a cycle from one woken by an empty pop, since there is no wake-up
//! object to record any more.
//!
//! A handler that names a deadline owns its schedule. The instant it gives
//! replaces whatever was pending, so a handler whose deadline moves — the
//! moderator's session clocks, which move every time somebody selects — is
//! re-armed where it moved to, and the wake channel is asked of the
//! [`TimerSource`] once per *distinct* deadline rather than once and kept.
//! A deadline already in the past fires at once, which is how a handler asks
//! for the next cycle whatever else happens; a handler that does that on
//! every cycle spins, which is its own bug and not one the loop guards
//! against.
//!
//! A handler that names none falls back on [`Wiring::timeout`], the fixed
//! interval given when the agent is wired: from the cycle in which it pops
//! [`Control::Start`] a deadline is pending one interval ahead, and the next
//! is one interval after the cycle the last one woke. Such a deadline
//! stays where it is while messages arrive, so being spoken to never pushes
//! it back.
//!
//! Where there is no interval to fall back on, `None` **withdraws** the
//! deadline the handler last named, and an agent left with none blocks
//! until something arrives. That is what lets a handler give a clock up as
//! well as move it: the moderator's night sessions close one at a time, and
//! the last of them leaves nothing to wake for.
//!
//! A cycle woken by the deadline that also found a message calls `handle`,
//! not `timeout`: it has an observation, and an observation is what a
//! handler decides from. The record still says the deadline woke it. A
//! handler that keeps its own deadlines therefore checks them in `handle`
//! too, against the observation's `received`.
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
//! stand-in and a log writer:
//!
//! ```
//! use std::sync::{Arc, Mutex};
//!
//! use crossbeam_channel::unbounded;
//! use social_deception::{
//!     Action, ActorId, Agent, Clock, Control, CycleDispatch, Delivery, Handler,
//!     JsonLines, Message, Observation, Sink, Wiring, Writer,
//! };
//! use social_deception::log::Policy;
//!
//! // What this game's agents say to each other: a line of chat.
//! type Chat = String;
//!
//! struct Echo;
//!
//! impl Handler<Chat> for Echo {
//!     fn handle(&mut self, observation: &Observation<Chat>) -> Vec<Action<Chat>> {
//!         vec![Action::to(
//!             [observation.message.sender.clone()],
//!             observation.message.payload.clone(),
//!         )]
//!     }
//! }
//!
//! let clock = Clock::start();
//! let (to_agent, queue) = unbounded();
//! let (dispatches, from_agent) = unbounded();
//! // The log goes to a `JsonLines` sink over a buffer this example
//! // can read back; a run writes one over a file instead.
//! let log = Arc::new(Mutex::new(Vec::new()));
//! let sink: Box<dyn Sink<Chat>> = Box::new(JsonLines::new(Recorded(log.clone())));
//! let (records, writer) = Writer::spawn(vec![(sink, Policy::Required)]);
//! let wiring =
//!     Wiring { id: "echo".into(), clock, queue, dispatches, records, timeout: None };
//! let agent = Agent::spawn(wiring, Echo, clock);
//!
//! let hello = Message::<Chat>::new("caller", ["echo"], clock.now(), String::from("hello"));
//! // One queue, so everything is said in the order it is to be handled: the
//! // start, the message, and then the stop the agent reaches after
//! // answering it.
//! to_agent.send(Delivery::control(Control::Start, clock.now())).unwrap();
//! to_agent.send(Delivery::Message(hello)).unwrap();
//! to_agent.send(Delivery::control(Control::Stop, clock.now())).unwrap();
//!
//! let echoed: CycleDispatch<Chat> = from_agent.iter().find(|r| !r.sent.is_empty()).unwrap();
//! agent.join().unwrap();
//! let sent = echoed.sent;
//! assert_eq!(sent.len(), 1);
//! assert_eq!(sent[0].sender, ActorId::new("echo"));
//! assert_eq!(sent[0].payload, "hello");
//! writer.join().unwrap();
//! assert!(!log.lock().unwrap().is_empty());
//!
//! /// A destination whose bytes stay readable after the sink has taken it.
//! struct Recorded(Arc<Mutex<Vec<u8>>>);
//!
//! impl std::io::Write for Recorded {
//!     fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
//!         self.0.lock().unwrap().extend_from_slice(buf);
//!         Ok(buf.len())
//!     }
//!
//!     fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
//! }
//! ```

use std::collections::BTreeSet;
use std::fmt;
use std::panic;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError, never, select};

use crate::clock::{Clock, Created, Received, Timestamp};
use crate::log::{ActionRecord, ControlRecord, CycleRecord, ObservationRecord, Record, Seq, Woken};
use crate::message::{ActorId, Control, Delivery, Message, Payload};
use crate::timer::TimerSource;

/// A message this agent has popped off its queue: what it observed, and when.
///
/// The message carries the instant its sender created it; this adds the
/// instant this agent received it. The gap between the two is the
/// observation's [`latency`](Received::latency), the whole staleness of
/// what the agent is looking at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation<P: Payload> {
    /// The message.
    pub message: Message<P>,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
}

impl<P: Payload> Created for Observation<P> {
    fn created(&self) -> Timestamp {
        self.message.created
    }
}

impl<P: Payload> Received for Observation<P> {
    fn received(&self) -> Timestamp {
        self.received
    }
}

/// A control this agent has popped off its queue, and when.
///
/// Logged by the loop and never handed to a handler; it implements
/// [`Received`] for the same reason an observation does, so that a control
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

impl Received for Instruction {
    fn received(&self) -> Timestamp {
        self.received
    }
}

/// What an agent sends: to whom, and what.
///
/// No sender and no creation time, unless the action is a relay. An action
/// has no creation time until it is sent, and the handler cannot know that
/// instant, so the loop stamps both as it hands the action to the router;
/// the stamped value is the [`Message`] on the wire and what the `action`
/// record logs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Action<P: Payload> {
    /// The agents to send it to, possibly none. The set never contains the
    /// sender; the router enforces that.
    pub recipients: BTreeSet<ActorId>,
    /// What to say.
    pub payload: P,
    /// Who really said it, and when, when this action is one agent
    /// passing on another's.
    ///
    /// `None` for the ordinary case: the action is the sender's own and
    /// the loop stamps it with the sender's own name and clock. `Some` is
    /// a **relay**, and the stamp keeps what is here instead, so the
    /// recipient sees the message the original actor would have sent it
    /// directly. See [`Action::relay`].
    pub origin: Option<Origin>,
}

/// Who first sent a relayed action, and when.
///
/// An agent that passes on another's action does not put its own name on
/// it. The pair here is what [`Message::sender`] and [`Message::created`]
/// become, so a relayed message is indistinguishable from a direct one and
/// the relay shows up only as the extra latency it costs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Origin {
    /// The agent whose action this is.
    pub sender: ActorId,
    /// The instant that agent created it.
    pub created: Timestamp,
}

impl<P: Payload> Action<P> {
    /// An action addressed to the given recipients.
    pub fn to<I, A>(recipients: I, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self {
            recipients: recipients.into_iter().map(Into::into).collect(),
            payload,
            origin: None,
        }
    }

    /// One agent passing on another's action, addressed to `recipients`.
    ///
    /// The message the recipients observe names `sender` and is stamped
    /// `created`, not the relaying agent and not the instant of the relay.
    /// What a recipient sees is therefore exactly what it would have seen
    /// had the original actor addressed it directly; the only trace of the
    /// relay is that the message arrives later than it was created.
    ///
    /// The relaying agent's own numbering is untouched: the strictly
    /// increasing stamp [`Handler`] actions get is per sender, and this
    /// action is not the relaying agent's to number. Two actions are told
    /// apart by their sender and creation time (ADR-0002), and both of
    /// those belong to the original actor here.
    pub fn relay<I, A>(
        sender: impl Into<ActorId>,
        created: Timestamp,
        recipients: I,
        payload: P,
    ) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self {
            recipients: recipients.into_iter().map(Into::into).collect(),
            payload,
            origin: Some(Origin {
                sender: sender.into(),
                created,
            }),
        }
    }
}

/// A game's behavior for one agent.
///
/// The loop calls [`handle`](Handler::handle) once per cycle with the one
/// observation that cycle popped, and sends what comes back. Agent state
/// lives in the implementing type. A handler never sees a [`Control`]; see
/// the [module documentation](self).
pub trait Handler<P: Payload> {
    /// The agent's opening actions, called once when it is started.
    ///
    /// This is where an agent that acts before anybody has spoken to it does
    /// so. Most agents only react, and the default returns nothing.
    ///
    /// Opening actions are the agent's own, decided from nothing; a handler
    /// with work to do before it can name them has state to fold and belongs
    /// in `handle`.
    fn start(&mut self, _now: Timestamp) -> Vec<Action<P>> {
        Vec::new()
    }

    /// Folds one observation into the agent's state and says what to send.
    ///
    /// One observation, because a cycle handles exactly one (ADR-0008): this
    /// is a decision point, and what an agent conditions on at a decision
    /// selection is an observation, not a pile of them. A cycle that popped
    /// no observation — the opening `Start`, or the timeout — does not call
    /// this at all.
    ///
    /// Nothing interrupts it. An agent does not know it is being stopped
    /// (ADR-0009), so a handler is never asked to give up early and what
    /// this returns is always sent. A handler that blocks delays its own
    /// agent, and with it the end of its episode, for as long as it blocks;
    /// the place to bound that is inside the handler, in whatever it is
    /// blocking on.
    fn handle(&mut self, observation: &Observation<P>) -> Vec<Action<P>>;

    /// What the agent does when its deadline passes and nothing has arrived.
    ///
    /// Waking on a deadline is not observing anything, so it is its own
    /// method rather than a `handle` with nothing to hand over: an empty
    /// slice made "no observation" a kind of observation, and every handler
    /// would have had to unwrap its way back out of it (ADR-0008). The
    /// default does nothing, which is what an agent without a timeout wants
    /// and what every agent in the tree wants today.
    fn timeout(&mut self, _now: Timestamp) -> Vec<Action<P>> {
        Vec::new()
    }

    /// The next instant this handler wants a cycle, whatever arrives before
    /// then, or `None` to fall back on the wired interval if there is one.
    ///
    /// The loop asks after the start hook and after every cycle, and arms a
    /// wake-up for whatever comes back; a deadline that differs from the one
    /// pending replaces it (ADR-0010). A handler that overrides this owns
    /// its schedule completely, and one that returns `None` throughout
    /// behaves exactly as it did before there was a deadline to set.
    ///
    /// The instant is absolute and on the agent's clock, the same clock the
    /// `now` of [`start`](Handler::start) and [`timeout`](Handler::timeout)
    /// and an observation's `received` are read from, so a handler may
    /// compare them directly.
    ///
    /// **A deadline already in the past fires at once**, which is how a
    /// handler asks for the next cycle whatever else happens. A handler that
    /// returns a past deadline on *every* cycle therefore spins; that is a
    /// bug in the handler, in the same way as a handler that blocks, and the
    /// loop does not guard against it.
    ///
    /// **Check the deadline in `handle` too.** A deadline that passes while
    /// a message is waiting joins that message's cycle, which calls `handle`
    /// rather than `timeout` (ADR-0008), so a handler that keeps deadlines
    /// compares them against `observation.received` as well. A due deadline
    /// is a fact about the time, not about which method is running.
    ///
    /// **`None` withdraws a deadline** for a handler whose agent was wired
    /// with no interval, which is every agent an [`Episode`] runs. Owning a
    /// schedule includes clearing it: a moderator whose last session has
    /// closed has no clock left to name, and the instant it named before
    /// must not outlive it. An agent that *was* wired with an interval
    /// falls back on that instead, since for it `None` means "no opinion"
    /// rather than "no deadline".
    ///
    /// [`Episode`]: crate::Episode
    fn deadline(&self) -> Option<Timestamp> {
        None
    }
}

/// What one cycle of an agent's loop hands the router.
///
/// One dispatch is sent per cycle, after the cycle's actions have been
/// recorded and before its cycle record is written. Because the number of
/// deliveries consumed and the messages produced arrive together, whoever
/// counts in-flight deliveries never sees a cycle's inputs settled before
/// its outputs exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleDispatch<P: Payload> {
    /// The agent whose cycle this was.
    pub agent: ActorId,
    /// How many deliveries the cycle took off its queue. A timeout is not a
    /// delivery.
    pub deliveries: usize,
    /// The messages the cycle sent, stamped with this agent as sender, in the
    /// order the handler returned them.
    pub sent: Vec<Message<P>>,
    /// Whether the agent is waiting on a deadline after this cycle.
    ///
    /// An agent that is has work of its own still to do: it will run
    /// another cycle when the instant arrives, whatever anybody says to
    /// it. The episode counts that as work outstanding, so a roster whose
    /// only pending thing is a clock is waiting rather than stalled (see
    /// [`episode`](crate::episode)).
    pub waking: bool,
}

/// Everything an agent's thread needs besides its handler and timer.
///
/// One queue, carrying both kinds of thing said to the agent; the
/// [module documentation](self) says why it is not two.
pub struct Wiring<P: Payload> {
    /// The agent's id: the sender on everything it emits and the `agent` on
    /// every record it writes.
    pub id: ActorId,
    /// The episode clock.
    pub clock: Clock,
    /// The agent's queue: messages, which become [`Observation`]s when
    /// popped, and controls, which the loop acts on itself, in the order
    /// they were sent.
    pub queue: Receiver<Delivery<P>>,
    /// Where each cycle's dispatch goes.
    pub dispatches: Sender<CycleDispatch<P>>,
    /// Where the agent's records go.
    pub records: Sender<Record<P>>,
    /// How long the agent waits before its deadline fires, or `None` for an
    /// agent that only ever reacts. A deadline that passes with nothing
    /// waiting runs a cycle that calls [`Handler::timeout`]; one that passes
    /// while a message waits joins that message's cycle instead.
    pub timeout: Option<Duration>,
}

impl<P: Payload> fmt::Debug for Wiring<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Wiring")
            .field("id", &self.id)
            .field("timeout", &self.timeout)
            .finish_non_exhaustive()
    }
}

/// Why an agent's loop stopped before its queues closed or it was told to
/// stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    /// A record could not be sent: the log writer has gone away.
    WriterClosed,
    /// A dispatch could not be sent: the router has gone away.
    RouterClosed,
    /// The timer source disconnected a wake channel the agent was waiting on.
    TimerClosed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::WriterClosed => "the log writer has gone away",
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
    id: ActorId,
    thread: JoinHandle<Result<H, Error>>,
}

impl<H> Agent<H> {
    /// Starts an agent on its own thread.
    ///
    /// # Panics
    ///
    /// If the operating system refuses to create the thread.
    pub fn spawn<P, T>(wiring: Wiring<P>, handler: H, timer: T) -> Self
    where
        P: Payload,
        H: Handler<P> + Send + 'static,
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
    pub fn id(&self) -> &ActorId {
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
enum Wake<P: Payload> {
    /// Something arrived on the queue, and here it is.
    Delivered(Delivery<P>),
    /// The pending deadline passed.
    Deadline,
    /// The queue closed, which the drain that follows confirms without
    /// throwing anything away.
    Closed,
    /// The wake channel disconnected.
    TimerGone,
}

/// What one cycle took off its queue, in the order the loop records it:
/// the controls at the head, then the one message behind them.
///
/// The controls are however many were waiting in a row and the message is at
/// most one. A control is out-of-domain and is the loop's own business, so
/// a cycle takes every one it finds before it looks for something to
/// observe; a message is an observation, and a cycle handles exactly one
/// (ADR-0008). What is left on the queue is the next cycle's.
///
/// `deliveries` counts everything taken off the queue, and is made where
/// the taking is rather than from what survives it. That matters on the
/// `Stop` path, where a message the wake-up had already taken is forgotten:
/// it was still delivered, and an episode that never hears of a delivery
/// waits forever for it (see [`episode`](crate::episode)).
struct Popped<P: Payload> {
    controls: Vec<Instruction>,
    message: Option<Message<P>>,
    deliveries: usize,
}

impl<P: Payload> Popped<P> {
    /// Whether the cycle popped nothing at all, which is how a wake-up that
    /// was only the queue closing is told from one that brought work.
    fn is_empty(&self) -> bool {
        self.controls.is_empty() && self.message.is_none()
    }
}

/// The state of an agent's thread.
struct Loop<P: Payload, H, T> {
    wiring: Wiring<P>,
    handler: H,
    timer: T,
    /// The next sequence number to assign.
    next_seq: u64,
    /// The last `created` this agent stamped onto an action. Per agent, not
    /// per cycle: the join an observation makes is on the sender and the
    /// instant over the whole log, so two actions of one agent may
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

impl<P, H, T> Loop<P, H, T>
where
    P: Payload,
    H: Handler<P>,
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
            let stopped = self.cycle(t_start, popped, timed_out)?;
            if stopped || self.closed {
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
    fn wait(&self) -> Wake<P> {
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
    /// most one** message, stamping each with `t_start`, and remembers the
    /// queue if it turned out to be closed.
    ///
    /// One queue, so one pass along it. The loop takes controls while
    /// controls are what it finds, because a control is out-of-domain and
    /// the loop's own business — a cycle that stopped at the first of a run
    /// of them would spend a cycle on each, recording nothing and asking
    /// the handler nothing. The first message ends the pass, because a cycle
    /// handles exactly one observation (ADR-0008) and the rest of the queue
    /// is the next cycle's.
    ///
    /// So a cycle that finds `[Start, message]` pops both and does the start
    /// first, while one that finds `[message, Stop]` pops only the message
    /// and reaches the stop next time round. The second is what ADR-0009
    /// accepts: with one queue a `Stop` behind messages is handled after
    /// them, and the episode does not queue one that way except when it is
    /// abandoning a run that has already failed.
    ///
    /// Once a `Stop` is in hand the pass ends there. This cycle is the
    /// agent's last, and a message left on the queue is one the agent never
    /// popped, never observed and is never logged for: taking it only to
    /// observe it after the episode had ended would put a decision in the
    /// log that nobody asked for. The one message the wake-up may
    /// already have taken ahead of a `Stop` cannot arise — the wake-up
    /// takes one thing and the pass stops at the stop it then finds — so
    /// the only thing forgotten is what was never taken.
    fn drain(&mut self, woke_with: Wake<P>, t_start: Timestamp) -> Popped<P> {
        let (mut controls, mut message, mut deliveries) = (Vec::new(), None, 0);
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
                Delivery::Message(popped) => {
                    message = Some(popped);
                    break;
                }
            }
        }
        Popped {
            controls,
            message,
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

    /// Settles the deadline to wait on after a cycle that began at `from`.
    ///
    /// The handler is asked first (ADR-0010). What it names is absolute and
    /// replaces whatever was pending, and a wake channel is asked for only
    /// when the instant differs from the one already armed, so a handler
    /// that keeps naming the same deadline waits on the same channel and a
    /// deadline that moves is re-armed where it moved to.
    ///
    /// A handler with no opinion falls back on the wired interval, measured
    /// from `from` and armed only by a cycle that is `due` one: the cycle
    /// that started the agent, or one whose deadline has just fired. That
    /// is narrower than "nothing is pending", and deliberately so. An agent
    /// can run cycles before its `Start` reaches it, because a message
    /// queued ahead of one is observed first, and those cycles have nothing
    /// pending either; an interval runs from the start, not from whatever
    /// the agent was doing beforehand. In between, the deadline stays where
    /// it is, so being spoken to never pushes it back.
    fn arm(&mut self, from: Timestamp, due: bool) {
        match (self.handler.deadline(), self.wiring.timeout) {
            // The handler named one. Re-armed only where it moved to, so a
            // handler that keeps naming the same instant waits on the
            // channel it already has.
            (Some(deadline), _) => {
                if self.pending.as_ref().is_none_or(|(at, _)| *at != deadline) {
                    self.pending = Some((deadline, self.timer.wake_at(deadline)));
                }
            }
            // A handler with an interval behind it and nothing to say
            // leaves the interval to it.
            (None, Some(every)) => {
                if due {
                    let deadline = from + every;
                    self.pending = Some((deadline, self.timer.wake_at(deadline)));
                }
            }
            // A handler with nothing behind it that names nothing has
            // withdrawn whatever it last named. Owning a schedule includes
            // clearing it: a session that has closed has no clock left, and
            // the deadline it kept must not outlive it.
            (None, None) => self.pending = None,
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
        popped: Popped<P>,
        timed_out: bool,
    ) -> Result<bool, Error> {
        let Popped {
            controls,
            message,
            deliveries,
        } = popped;
        let (mut inputs, mut started, mut stopped) = (Vec::new(), false, false);
        // Controls first, and in one pass, so that a `Start` at the head of
        // the queue has run the start hook before the message behind it is
        // observed.
        for instruction in &controls {
            match instruction.control {
                // An agent is started once, before anything is addressed to
                // it. A second `Start` cannot be honored — the start hook
                // has already run, and running it again would reopen an
                // agent that has been playing — and recording it anyway
                // would put a claim in the log the loop did not act on.
                // Whoever sent it has a bug the log must not paper over.
                Control::Start => {
                    assert!(!self.started, "{} was started twice", self.wiring.id);
                    self.started = true;
                    started = true;
                }
                Control::Stop => stopped = true,
            }
            inputs.push(self.record_control(instruction)?);
        }
        let observation = message.map(|message| Observation {
            message,
            received: t_start,
        });
        if let Some(observation) = &observation {
            inputs.push(self.record_observation(observation)?);
        }
        // A start is the loop's own business: it calls the start hook, whose
        // opening actions go out ahead of whatever the cycle's observation
        // also produced, since the start was popped before it.
        let mut actions = if started {
            self.handler.start(t_start)
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
            actions.extend(self.handler.timeout(t_start));
        }

        let (mut sent, mut outputs) = (Vec::new(), Vec::new());
        for action in actions {
            let message = self.stamp(action);
            outputs.push(self.record_action(&message)?);
            sent.push(message);
        }

        // The next deadline is settled before the dispatch goes out,
        // because the dispatch reports it. A stop ends the agent, so it
        // arms nothing and *drops* whatever was pending: an agent on its
        // way out is not waiting for anything, and a deadline left armed
        // would be reported as a wake-up the episode then waits on
        // forever.
        if stopped {
            self.pending = None;
        } else {
            self.arm(t_start, started || timed_out);
        }
        let dispatch = CycleDispatch {
            agent: self.wiring.id.clone(),
            deliveries,
            sent,
            waking: self.pending.is_some(),
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
    /// stamp.
    ///
    /// The stamp is the action's `created` on the wire, and every action a
    /// handler returns is sent, so there is no other case.
    ///
    /// The stamp is always strictly later than the last one this agent handed
    /// out. That is not cosmetic. An observation names the action it came
    /// from by the sender and the creation time and by nothing else, since no
    /// message carries an identifier (ADR-0002), so two of an agent's actions
    /// sharing an instant would be two actions no observation could tell
    /// apart. A cycle that returns several actions stamps them within a few
    /// hundred nanoseconds of each other, which the clock's resolution does
    /// not always separate, and two cycles can run that close together too,
    /// so the guarantee is the agent's and not one cycle's.
    ///
    /// # A relay is stamped with whose action it is
    ///
    /// An action carrying an [`Origin`] is one agent passing on another's,
    /// and it keeps the original sender and creation time
    /// ([`Action::relay`]). This agent's own numbering is left alone: the
    /// increasing-stamp guarantee is per sender, and a relayed action is
    /// not this agent's to number. Advancing `last_created` for one would
    /// push this agent's next real action past an instant it never used.
    fn stamp(&mut self, action: Action<P>) -> Message<P> {
        let Action {
            recipients,
            payload,
            origin,
        } = action;
        if let Some(Origin { sender, created }) = origin {
            return Message {
                sender,
                recipients,
                created,
                payload,
            };
        }
        let now = self.wiring.clock.now();
        let created = match self.last_created {
            Some(previous) if now <= previous => previous + Duration::from_nanos(1),
            _ => now,
        };
        self.last_created = Some(created);
        Message {
            sender: self.wiring.id.clone(),
            recipients,
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

    fn record_observation(&mut self, observation: &Observation<P>) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            ObservationRecord {
                agent: self.wiring.id.clone(),
                seq,
                created: observation.message.created,
                received: observation.received,
                message: observation.message.clone(),
            }
            .into(),
        )?;
        Ok(seq)
    }

    fn record_action(&mut self, message: &Message<P>) -> Result<Seq, Error> {
        let seq = self.next_seq();
        self.send_record(
            ActionRecord {
                agent: self.wiring.id.clone(),
                seq,
                created: message.created,
                message: message.clone(),
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

    fn send_record(&self, record: Record<P>) -> Result<(), Error> {
        self.wiring
            .records
            .send(record)
            .map_err(|_| Error::WriterClosed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use crossbeam_channel::{RecvTimeoutError, unbounded};
    use serde_json::json;

    use super::*;
    use crate::testing::{TestPayload, joined, parse_lines, recording};
    use crate::timer::{ManualTimer, ManualTimerControl};

    /// How long a test waits on a channel before giving up. A test only ever
    /// waits this long when it has already failed.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// A timeout interval. Its value never matters: the manual timer decides
    /// when deadlines fire.
    const EVERY: Duration = Duration::from_secs(1);

    type TestMessage = Message<TestPayload>;

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// A step from `sender` to `a`, created at `created`.
    fn step_at(sender: &str, n: u64, created: Timestamp) -> TestMessage {
        Message::new(sender, ["a"], created, TestPayload::Step(n))
    }

    fn step(sender: &str, n: u64) -> TestMessage {
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
        /// The `now` of every start and timeout, in order, so that a test
        /// can check the hooks are told the cycle's `t_start`.
        clock_readings: Vec<Timestamp>,
    }

    impl Handler<TestPayload> for Recorder {
        fn start(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.started += 1;
            self.clock_readings.push(now);
            Vec::new()
        }

        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            let TestPayload::Step(n) = observation.message.payload;
            self.seen.push(observation.message.payload.clone());
            vec![Action::to(
                [observation.message.sender.clone()],
                TestPayload::Step(n + 1),
            )]
        }

        fn timeout(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.timeouts += 1;
            self.clock_readings.push(now);
            Vec::new()
        }
    }

    /// A recorder that names its own deadline (ADR-0010).
    ///
    /// The deadline is whatever the test last set, shared so that the test
    /// can move it while the agent runs; the handler otherwise behaves
    /// exactly like the [`Recorder`] it wraps.
    ///
    /// The lock is this test's own business and not a pattern to copy. The
    /// loop asks for a deadline every cycle, so a real handler answers from
    /// a field it already holds; only a test needs another thread to be
    /// able to move the answer mid-run.
    #[derive(Debug)]
    struct Punctual {
        inner: Recorder,
        deadline: Arc<Mutex<Option<Timestamp>>>,
    }

    impl Punctual {
        /// The handler and the test's handle on its deadline.
        fn new(deadline: Option<Timestamp>) -> (Self, Arc<Mutex<Option<Timestamp>>>) {
            let deadline = Arc::new(Mutex::new(deadline));
            (
                Self {
                    inner: Recorder::default(),
                    deadline: deadline.clone(),
                },
                deadline,
            )
        }
    }

    impl Handler<TestPayload> for Punctual {
        fn start(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.inner.start(now)
        }

        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            self.inner.handle(observation)
        }

        fn timeout(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.inner.timeout(now)
        }

        fn deadline(&self) -> Option<Timestamp> {
            *self.deadline.lock().unwrap()
        }
    }

    /// A handler that, on entering each cycle, tells the test it is busy and
    /// then waits to be released. That is how a test makes things arrive
    /// while the agent is provably mid-cycle.
    ///
    /// Generic over what it wraps, so that the gate and the handler being
    /// gated are chosen separately: [`Recorder`] for a test that only needs
    /// to hold the agent, [`Punctual`] for one that also needs the handler
    /// to name a deadline. Every hook forwards, `deadline` included, so a
    /// gated handler schedules exactly as it would ungated.
    #[derive(Debug)]
    struct Gated<H> {
        inner: H,
        entered: Sender<()>,
        release: Receiver<()>,
    }

    impl<H: Handler<TestPayload>> Handler<TestPayload> for Gated<H> {
        fn start(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.inner.start(now)
        }

        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.handle(observation)
        }

        fn timeout(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.timeout(now)
        }

        fn deadline(&self) -> Option<Timestamp> {
            self.inner.deadline()
        }
    }

    /// A [`Punctual`] that withdraws its deadline the first time it fires,
    /// so that a past deadline runs exactly one cycle instead of spinning.
    struct Once {
        inner: Punctual,
        deadline: Arc<Mutex<Option<Timestamp>>>,
    }

    impl Handler<TestPayload> for Once {
        fn start(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            self.inner.start(now)
        }

        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            self.inner.handle(observation)
        }

        fn timeout(&mut self, now: Timestamp) -> Vec<Action<TestPayload>> {
            *self.deadline.lock().unwrap() = None;
            self.inner.timeout(now)
        }

        fn deadline(&self) -> Option<Timestamp> {
            self.inner.deadline()
        }
    }

    /// Says one step to the agents it was built with when it starts, and
    /// nothing after that.
    ///
    /// Built with nobody it is a soliloquist: an action need not be directed
    /// at anyone, and what one addressed to nobody is for is the log.
    struct Town(&'static [&'static str]);

    impl Handler<TestPayload> for Town {
        fn start(&mut self, _now: Timestamp) -> Vec<Action<TestPayload>> {
            vec![Action::to(self.0.iter().copied(), TestPayload::Step(0))]
        }

        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            Vec::new()
        }
    }

    /// Passes on whatever it observes, to `to`, as the agent that sent it.
    ///
    /// The stand-in for an environment that relays one agent's action to
    /// another: what it emits is not its own action but somebody else's,
    /// carried on.
    struct Relays(&'static str);

    impl Handler<TestPayload> for Relays {
        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            vec![Action::relay(
                observation.message.sender.clone(),
                observation.message.created,
                [self.0],
                observation.message.payload.clone(),
            )]
        }
    }

    /// Relays what it observes to `to`, then says something of its own.
    struct RelaysThenSpeaks(&'static str);

    impl Handler<TestPayload> for RelaysThenSpeaks {
        fn handle(&mut self, observation: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            vec![
                Action::relay(
                    observation.message.sender.clone(),
                    observation.message.created,
                    [self.0],
                    observation.message.payload.clone(),
                ),
                Action::to([self.0], TestPayload::Step(42)),
            ]
        }
    }

    /// A panicking handler.
    struct Faulty;

    impl Handler<TestPayload> for Faulty {
        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
            panic!("handler bug");
        }

        fn timeout(&mut self, _now: Timestamp) -> Vec<Action<TestPayload>> {
            panic!("handler bug");
        }
    }

    /// An agent and the test's end of every channel it is wired to.
    struct Rig<H> {
        agent: Agent<H>,
        clock: Clock,
        queue: Sender<Delivery<TestPayload>>,
        dispatches: Receiver<CycleDispatch<TestPayload>>,
        records: Receiver<Record<TestPayload>>,
        timer: ManualTimerControl,
    }

    /// The channels of a rig, before the agent is spawned on them.
    struct Wires {
        wiring: Wiring<TestPayload>,
        queue: Sender<Delivery<TestPayload>>,
        dispatches: Receiver<CycleDispatch<TestPayload>>,
        records: Receiver<Record<TestPayload>>,
    }

    impl Wires {
        fn control(&self, control: Control) {
            self.queue
                .send(Delivery::control(control, self.wiring.clock.now()))
                .unwrap();
        }

        fn send(&self, message: TestMessage) {
            self.queue.send(Delivery::Message(message)).unwrap();
        }
    }

    fn wires(timeout: Option<Duration>) -> Wires {
        let (queue, receiver) = unbounded();
        let (outbox, dispatches) = unbounded();
        let (recorder, records) = unbounded();
        let wiring = Wiring {
            id: ActorId::new("a"),
            clock: Clock::start(),
            queue: receiver,
            dispatches: outbox,
            records: recorder,
            timeout,
        };
        Wires {
            wiring,
            queue,
            dispatches,
            records,
        }
    }

    fn rig<H: Handler<TestPayload> + Send + 'static>(
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
        fn send(&self, message: TestMessage) {
            self.queue.send(Delivery::Message(message)).unwrap();
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

        fn dispatch(&self) -> CycleDispatch<TestPayload> {
            recv(&self.dispatches)
        }

        /// The records of one cycle: everything it wrote, then its cycle
        /// record.
        fn cycle(&self) -> (Vec<Record<TestPayload>>, CycleRecord) {
            let mut records = Vec::new();
            loop {
                match recv(&self.records) {
                    Record::Cycle(cycle) => return (records, cycle),
                    record => records.push(record),
                }
            }
        }
    }

    /// A rig whose handler can be held mid-cycle, and the two ends of the
    /// gate: the channel that says the agent has entered a cycle, and the
    /// one that lets it out again.
    fn gated_with<H: Handler<TestPayload> + Send + 'static>(
        inner: H,
        timeout: Option<Duration>,
    ) -> (Rig<Gated<H>>, Receiver<()>, Sender<()>) {
        let (entered, busy) = unbounded();
        let (release, released) = unbounded();
        let handler = Gated {
            inner,
            entered,
            release: released,
        };
        (rig(handler, timeout), busy, release)
    }

    fn gated() -> (Rig<Gated<Recorder>>, Receiver<()>, Sender<()>) {
        gated_with(Recorder::default(), Some(EVERY))
    }

    fn steps<const N: usize>(ns: [u64; N]) -> Vec<TestPayload> {
        ns.map(TestPayload::Step).into()
    }

    /// What kind of record this is, for a test that cares about the order
    /// of the kinds rather than their contents.
    fn kind(record: &Record<TestPayload>) -> &'static str {
        match record {
            Record::Observation(_) => "observation",
            Record::Action(_) => "action",
            Record::Control(_) => "control",
            // An agent's loop never writes one: a reward is the
            // environment's, and it goes out through the adapter.
            Record::Reward(_) => "reward",
            Record::Cycle(_) => "cycle",
        }
    }

    /// The payloads of the action records among `records`, in order.
    fn acted(records: &[Record<TestPayload>]) -> Vec<TestPayload> {
        records
            .iter()
            .filter_map(|record| match record {
                Record::Action(record) => Some(record.message.payload.clone()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_cycle_takes_one_message_and_leaves_the_rest() {
        // Three messages are waiting before the agent is even spawned, so
        // nothing about the scheduler decides what it finds: one cycle
        // could have had all three. It takes them one at a time, in queue
        // order, and answers each on its own (ADR-0008).
        let wires = wires(None);
        wires.control(Control::Start);
        for message in [step("b", 6), step("c", 3), step("b", 5)] {
            wires.send(message);
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        drop(wires.queue);

        let handler = agent.join().unwrap();
        // The control was the loop's, so the handler saw three observations
        // and one start, not four messages.
        assert_eq!(handler.started, 1);
        assert_eq!(handler.seen, steps([6, 3, 5]));
        // Every cycle that took a message answered it, and no cycle
        // answered two: one observation, one reply.
        let dispatches: Vec<_> = wires.dispatches.iter().collect();
        assert!(
            dispatches.iter().all(|dispatch| dispatch.sent.len() <= 1),
            "no cycle answers more than its one observation: {dispatches:?}"
        );
        let payloads: Vec<&TestPayload> = dispatches
            .iter()
            .flat_map(|dispatch| dispatch.sent.iter().map(|message| &message.payload))
            .collect();
        assert_eq!(payloads, steps([7, 4, 6]).iter().collect::<Vec<_>>());
        // Every delivery is reported exactly once, however the cycles fell:
        // the start and the three messages.
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
    fn a_message_arriving_mid_cycle_waits_for_a_cycle_of_its_own() {
        let (rig, busy, release) = gated();
        rig.start();
        // The start's cycle observes nothing, so it never enters the
        // handler and there is no gate to open for it. The first message is
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
        // message was queued behind a stop and so was never popped.
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
    fn a_handler_names_the_deadline_it_is_woken_at() {
        // Wired with an interval the handler must override: a handler that
        // names a deadline owns its schedule, and the interval never gets a
        // look in.
        let (handler, _deadline) = Punctual::new(Some(at(500)));
        let rig = rig(handler, Some(EVERY));
        rig.start();
        rig.dispatch();
        let (_, started) = rig.cycle();
        // Asked after the start hook, and armed at what it asked for.
        let asked = recv(rig.timer.requests());
        assert_eq!(asked, at(500));
        assert_ne!(asked, started.t_start + EVERY);

        rig.timer.fire().unwrap();
        assert_eq!(rig.dispatch().deliveries, 0);
        let (_, timed_out) = rig.cycle();
        assert_eq!(timed_out.woken, Woken::Timeout);
        assert!(timed_out.t_start > started.t_start);

        rig.stop();
        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.inner.timeouts, 1);
        // `now` is the cycle's `t_start`, on the same clock the records are
        // stamped from, so a handler's deadlines and the log share a
        // timeline.
        assert_eq!(
            handler.inner.clock_readings,
            [started.t_start, timed_out.t_start]
        );
    }

    #[test]
    fn moving_a_deadline_re_arms_the_timer_where_it_moved_to() {
        let (handler, deadline) = Punctual::new(Some(at(500)));
        let rig = rig(handler, None);
        rig.start();
        rig.dispatch();
        rig.cycle();
        assert_eq!(recv(rig.timer.requests()), at(500));

        // Later, then earlier. Each cycle asks the handler afresh, and each
        // distinct answer is a new wake channel at that instant.
        for moved in [at(900), at(100)] {
            *deadline.lock().unwrap() = Some(moved);
            rig.send(step("b", 1));
            rig.dispatch();
            rig.cycle();
            assert_eq!(recv(rig.timer.requests()), moved);
        }

        rig.stop();
        let handler = rig.agent.join().unwrap();
        // Only ever one deadline pending, so only the last one could have
        // woken anything, and nothing woke: the timer was never fired.
        assert_eq!(handler.inner.timeouts, 0);
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn a_handler_that_withdraws_its_deadline_is_not_woken_by_it() {
        // A handler owns its schedule, and owning it includes clearing it.
        // A session that closes has no clock left to name, so the deadline
        // it named must not survive it.
        let (handler, deadline) = Punctual::new(Some(at(500)));
        let rig = rig(handler, None);
        rig.start();
        rig.dispatch();
        rig.cycle();
        assert_eq!(recv(rig.timer.requests()), at(500));

        *deadline.lock().unwrap() = None;
        rig.send(step("b", 1));
        rig.dispatch();
        rig.cycle();

        // The withdrawn deadline is not waited on any more, so firing the
        // channel it was armed with wakes nothing. Fired with an empty
        // queue, so that a cycle could only be the deadline's: anything the
        // agent ran here it ran because it was still waiting on a deadline
        // it had given up.
        let _ = rig.timer.fire();
        assert_eq!(
            rig.dispatches.recv_timeout(Duration::from_millis(200)),
            Err(RecvTimeoutError::Timeout),
            "a withdrawn deadline still woke the agent"
        );

        // Still alive and still listening, so the silence was the deadline
        // being gone rather than the agent being gone.
        rig.send(step("b", 2));
        rig.dispatch();

        rig.stop();
        let handler = rig.agent.join().unwrap();
        assert_eq!(
            handler.inner.timeouts, 0,
            "a withdrawn deadline still woke the handler"
        );
        assert_eq!(handler.inner.seen, steps([1, 2]));
    }

    #[test]
    fn the_cycle_that_stops_an_agent_reports_no_deadline() {
        // An agent on its way out is not waiting for anything, whatever
        // it had armed before. The episode treats a cycle that reports a
        // pending deadline as work still to come and waits on it
        // (see `episode`), so a stopping cycle that reported one would
        // have the episode waiting on a thread that has ended.
        let (handler, _deadline) = Punctual::new(Some(at(500)));
        let rig = rig(handler, None);
        rig.start();
        let started = rig.dispatch();
        assert!(started.waking, "the handler named a deadline");

        rig.stop();
        let stopping = rig.dispatch();
        assert!(
            !stopping.waking,
            "the cycle that popped the stop still claimed a deadline"
        );
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_deadline_that_has_not_moved_is_not_asked_for_again() {
        // "Asked once per distinct deadline": a handler that keeps naming
        // the same instant waits on the channel it already has.
        let (handler, _deadline) = Punctual::new(Some(at(500)));
        let rig = rig(handler, None);
        rig.start();
        rig.dispatch();
        rig.cycle();
        assert_eq!(recv(rig.timer.requests()), at(500));

        for n in 1..=3 {
            rig.send(step("b", n));
            rig.dispatch();
            rig.cycle();
        }

        rig.stop();
        rig.agent.join().unwrap();
        assert!(
            rig.timer.requests().try_recv().is_err(),
            "the unchanged deadline was asked for again"
        );
    }

    #[test]
    fn a_handler_deadline_that_has_passed_runs_the_next_cycle_at_once() {
        // The clock starts at zero and only goes up, so a deadline of zero
        // is always in the past. The real clock is the timer source here:
        // nothing fires this but the instant itself having gone by.
        //
        // The handler withdraws the deadline from inside its own `timeout`,
        // so exactly one past-deadline cycle runs however the threads are
        // scheduled. Clearing it from the test instead would race the
        // agent, which spins until it sees the retraction.
        let (handler, deadline) = Punctual::new(Some(Timestamp::default()));
        let once = Once {
            inner: handler,
            deadline: deadline.clone(),
        };
        let wires = wires(None);
        let clock = wires.wiring.clock;
        let (queue, dispatches, records) = (wires.queue, wires.dispatches, wires.records);
        let agent = Agent::spawn(wires.wiring, once, clock);
        queue
            .send(Delivery::control(Control::Start, clock.now()))
            .unwrap();
        // The start cycle, then the cycle the past deadline woke, with
        // nothing having been said to the agent in between.
        assert_eq!(recv(&dispatches).deliveries, 1, "the start");
        assert_eq!(recv(&dispatches).deliveries, 0, "the deadline");

        queue
            .send(Delivery::control(Control::Stop, clock.now()))
            .unwrap();
        drop(queue);
        let handler = agent.join().unwrap();
        drop(records);
        assert_eq!(
            handler.inner.inner.timeouts, 1,
            "a deadline in the past fires once, and once only after it is withdrawn"
        );
    }

    #[test]
    fn a_handler_deadline_that_passed_while_a_message_waited_calls_handle() {
        // ADR-0008 is unchanged by ADR-0010: a deadline that passes while a
        // message is waiting joins that message's cycle, which observes. The
        // record still says the deadline woke it, which is why a handler
        // that keeps deadlines checks them in `handle` as well.
        let (handler, _deadline) = Punctual::new(Some(at(500)));
        let (rig, busy, release) = gated_with(handler, None);
        rig.start();
        // The start cycle first, and on its own. A cycle takes the controls
        // at the head of its queue and then one message, so a message sent
        // before the start was popped would be observed by that same cycle
        // and there would be one fewer cycle than this test counts.
        rig.dispatch();
        let (_, started) = rig.cycle();
        assert_eq!(started.woken, Woken::Queue);

        // Hold the agent inside a cycle, so that the deadline and the second
        // message are both queued before it next waits, for the reason the
        // wired-interval version of this test holds it.
        rig.send(step("b", 1));
        recv(&busy);
        rig.timer.fire().unwrap();
        rig.send(step("b", 2));
        release.send(()).unwrap();

        recv(&busy);
        release.send(()).unwrap();
        rig.dispatch();
        rig.dispatch();
        let (_, first) = rig.cycle();
        let (_, joined) = rig.cycle();
        assert_eq!(first.woken, Woken::Queue);
        assert_eq!(
            joined.woken,
            Woken::Timeout,
            "the cycle the deadline joined says the deadline woke it"
        );
        rig.stop();

        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.inner.inner.seen, steps([1, 2]));
        assert_eq!(
            handler.inner.inner.timeouts, 0,
            "the deadline joined an observing cycle, so `timeout` was never called"
        );
    }

    #[test]
    fn a_handler_with_no_deadline_leaves_the_wired_interval_alone() {
        // `deadline` returning `None` is the default, and an agent wired
        // with an interval behaves exactly as it did before ADR-0010.
        let (handler, _deadline) = Punctual::new(None);
        let rig = rig(handler, Some(EVERY));
        rig.start();
        rig.dispatch();
        let (_, started) = rig.cycle();
        assert_eq!(recv(rig.timer.requests()), started.t_start + EVERY);

        // Being spoken to does not move it.
        rig.send(step("b", 1));
        rig.dispatch();
        rig.cycle();
        assert!(rig.timer.requests().try_recv().is_err());

        rig.timer.fire().unwrap();
        rig.dispatch();
        let (_, timed_out) = rig.cycle();
        assert_eq!(timed_out.woken, Woken::Timeout);
        assert_eq!(recv(rig.timer.requests()), timed_out.t_start + EVERY);

        rig.stop();
        let handler = rig.agent.join().unwrap();
        assert_eq!(handler.inner.timeouts, 1);
    }

    #[test]
    fn a_deadline_that_passed_while_busy_joins_the_cycle_that_observed() {
        let (rig, busy, release) = gated();
        rig.start();
        // Hold the agent inside a cycle, so that the deadline and the second
        // message are both queued before it next waits. Racing the two onto an
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
        let (records, writer, bytes) = recording();
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
        let lines = parse_lines(&joined(writer, &bytes));

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
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}})
        );
        let created = lines[2]["created"].as_u64().unwrap();
        assert_eq!(
            lines[2],
            json!({"type": "action", "agent": "a", "seq": 2, "created": created,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 7}}})
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
        let Record::Observation(observation) = &records[0] else {
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
        assert!(matches!(first.as_slice(), [Record::Control(_)]));
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(0)], vec![]));
        rig.send(step("b", 1));
        let (second, cycle) = rig.cycle();
        assert!(matches!(
            second.as_slice(),
            [Record::Observation(_), Record::Action(_)]
        ));
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(1)], vec![Seq(2)]));
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_relayed_action_keeps_the_original_sender_and_creation_time() {
        // What `c` observes must be indistinguishable from what `b` would
        // have sent it directly: the relay costs latency and shows up
        // nowhere else.
        let rig = rig(Relays("c"), None);
        rig.start();
        rig.cycle();
        rig.dispatch();
        rig.send(step_at("b", 1, at(7)));
        let dispatch = rig.dispatch();
        assert_eq!(dispatch.sent.len(), 1);
        let sent = &dispatch.sent[0];
        assert_eq!(sent.sender, ActorId::new("b"), "whose action it is");
        assert_eq!(sent.created, at(7), "and when that agent made it");
        assert_eq!(sent.recipients, ["c"].map(ActorId::new).into());
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn relaying_does_not_advance_the_relaying_agents_own_stamps() {
        // The increasing-stamp guarantee is per sender. A relayed action
        // is not this agent's to number, so its own next action is stamped
        // from its clock and not pushed past an instant it never used.
        let rig = rig(RelaysThenSpeaks("c"), None);
        rig.start();
        rig.cycle();
        rig.dispatch();
        // Far enough ahead that the agent's own clock cannot have reached
        // it: if relaying advanced `last_created`, the agent's own action
        // would be dragged past this instant.
        let far = at(60_000_000_000);
        rig.send(step_at("b", 1, far));
        let dispatch = rig.dispatch();
        let [relayed, own] = dispatch.sent.as_slice() else {
            panic!("the cycle relays and then speaks: {:?}", dispatch.sent);
        };
        assert_eq!(relayed.sender, ActorId::new("b"));
        assert_eq!(relayed.created, far);
        assert_eq!(own.sender, ActorId::new("a"), "its own action is its own");
        assert!(
            own.created < far,
            "and is stamped from its own clock, not dragged past the relay: {:?}",
            own.created
        );
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn an_action_is_sent_and_recorded_as_the_agents_it_names() {
        let rig = rig(Town(&["b", "c"]), None);
        rig.start();
        let sent = rig.dispatch().sent;
        assert_eq!(sent.len(), 1);
        assert_eq!(sent[0].recipients, ["b", "c"].map(ActorId::new).into());
        let (records, cycle) = rig.cycle();
        let Record::Action(action) = &records[1] else {
            panic!("the action is recorded as an action: {records:?}");
        };
        assert_eq!(
            action.message.recipients,
            ["b", "c"].map(ActorId::new).into()
        );
        assert_eq!(cycle.outputs, [Seq(1)]);
        rig.stop();
        rig.agent.join().unwrap();
    }

    #[test]
    fn an_action_addressed_to_nobody_is_still_stamped_and_logged() {
        let rig = rig(Town(&[]), None);
        rig.start();
        let sent = rig.dispatch().sent;
        assert_eq!(sent.len(), 1, "the action is sent like any other");
        assert!(sent[0].recipients.is_empty());
        assert_eq!(sent[0].sender, ActorId::new("a"));
        let (records, cycle) = rig.cycle();
        let Record::Action(action) = &records[1] else {
            panic!("the action addressed to nobody is logged: {records:?}");
        };
        assert!(
            action.message.recipients.is_empty(),
            "and is logged as addressed to nobody"
        );
        assert_eq!(cycle.outputs, [Seq(1)]);
        rig.stop();
        rig.agent.join().unwrap();
    }

    /// Answers one observation with many actions at once, which is where
    /// two of an agent's stamps could collide.
    struct Chatters(usize);

    impl Handler<TestPayload> for Chatters {
        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Action<TestPayload>> {
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
                Record::Action(record) => Some(record.created),
                _ => None,
            })
            .collect();
        assert_eq!(created.len(), 100);
        assert!(
            created.windows(2).all(|pair| pair[0] < pair[1]),
            "every action of a cycle is stamped later than the one before it: {created:?}"
        );
        // And the stamps still lie inside the cycle's window, which is what
        // the log checker asserts of an output.
        assert!(
            created
                .iter()
                .all(|at| cycle.t_start <= *at && *at <= cycle.t_stop)
        );
        rig.stop();
        rig.agent.join().unwrap();
    }

    // Both of these drop a channel the loop depends on and expect the cycle
    // that follows to fail its send. The drop has to happen while the agent
    // is provably not about to send: it is otherwise a race, and the losing
    // interleaving is a hang rather than a failure, because a loop whose
    // queue is still open goes back to waiting. A `Gated` handler gives the
    // fixed point — it parks inside `handle`, before the cycle writes its
    // records or sends its dispatch — so the drop lands while the agent is
    // held there, and releasing it walks the cycle into the closed channel.

    #[test]
    fn a_vanished_writer_is_an_error() {
        let (rig, busy, release) = gated();
        rig.start();
        rig.send(step("b", 1));
        recv(&busy);
        drop(rig.records);
        release.send(()).unwrap();
        assert_eq!(rig.agent.join().err(), Some(Error::WriterClosed));
    }

    #[test]
    fn a_vanished_router_is_an_error() {
        let (rig, busy, release) = gated();
        rig.start();
        rig.send(step("b", 1));
        recv(&busy);
        drop(rig.dispatches);
        release.send(()).unwrap();
        assert_eq!(rig.agent.join().err(), Some(Error::RouterClosed));
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
        assert_eq!(rig.agent.id(), &ActorId::new("a"));
        assert_eq!(rig.agent.thread.thread().name(), Some("a"));
        drop(rig.queue);
        rig.agent.join().unwrap();
    }

    #[test]
    fn an_observation_and_a_control_know_their_own_latency() {
        let observation = Observation::<TestPayload> {
            message: step_at("b", 1, at(40)),
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
        // the log the loop did not act on.
        let wires = wires(None);
        wires.control(Control::Start);
        wires.send(step("b", 1));
        wires.control(Control::Start);
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        let _ = agent.join();
    }

    #[test]
    fn a_stop_queued_behind_many_messages_is_reached_behind_them() {
        // Everything is queued before the agent is spawned, so the order is
        // the queue's doing and nothing else: twenty messages went on the
        // wire first and the stop last. With one queue that is what the
        // agent works through — a cycle per message, then the stop — which is
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
        let records: Vec<Record<TestPayload>> = wires.records.try_iter().collect();
        let controls: Vec<Control> = records
            .iter()
            .filter_map(|record| match record {
                Record::Control(record) => Some(record.control),
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
        // the start, the twenty messages and the stop. An episode that never
        // hears of one waits forever for it, and one that hears of a
        // delivery twice panics subtracting it.
        let dispatches: Vec<_> = wires.dispatches.try_iter().collect();
        let deliveries: usize = dispatches.iter().map(|dispatch| dispatch.deliveries).sum();
        assert_eq!(deliveries, 22);
    }

    #[test]
    fn a_stop_leaves_what_is_behind_it_unpopped_and_unlogged() {
        // The other half of the same fact. Once the stop is in hand the
        // cycle takes no message, and the messages still queued are never
        // popped: an agent that has stopped did not observe them, and a
        // log that logged them would be claiming it did.
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
        let records: Vec<Record<TestPayload>> = wires.records.try_iter().collect();
        assert!(
            !records
                .iter()
                .any(|record| matches!(record, Record::Observation(_))),
            "a message never popped is never logged as an observation: {records:?}"
        );
        // Only what was taken is counted: the two controls, and none of the
        // five messages left on the queue.
        let dispatches: Vec<_> = wires.dispatches.try_iter().collect();
        let deliveries: usize = dispatches.iter().map(|dispatch| dispatch.deliveries).sum();
        assert_eq!(deliveries, 2);
        drop(wires.queue);
    }

    #[test]
    fn a_cycle_takes_the_controls_at_the_head_before_the_message_behind_them() {
        // One queue still does not mean one thing per cycle. A start and a
        // message waiting together are one cycle: the start hook runs first,
        // and the message is observed after it, which is the order they were
        // sent in and the order the records are written in.
        let wires = wires(None);
        wires.control(Control::Start);
        wires.send(step("b", 1));
        let agent = Agent::spawn(wires.wiring, Town(&["b", "c"]), Clock::start());
        drop(wires.queue);
        agent.join().unwrap();

        let records: Vec<Record<TestPayload>> = wires.records.try_iter().collect();
        let kinds: Vec<&str> = records.iter().map(kind).collect();
        // `Town` speaks when it starts and says nothing to an
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
        // behind the message it is deciding about, and the answer goes out
        // all the same.
        let (rig, busy, release) = gated();
        rig.start();
        // The start cycle first, and on its own. A cycle takes the controls
        // at the head of its queue and then one message, so a message sent
        // before the start was popped would share that cycle and the cycle
        // read below would be the wrong one.
        rig.cycle();
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

        let records: Vec<Record<TestPayload>> = rig.records.try_iter().collect();
        let mut cycles = 0;
        let mut received: Vec<Timestamp> = Vec::new();
        for record in &records {
            match record {
                Record::Control(record) => received.push(record.received),
                Record::Observation(record) => received.push(record.received),
                Record::Cycle(cycle) => {
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
