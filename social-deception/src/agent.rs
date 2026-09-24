//! The agent: one thread, one inbox, and a pop-fold-send loop.
//!
//! An agent's life is a fold over what arrives on its inbox. Each cycle of
//! the loop:
//!
//! 1. waits until something arrives or its timeout fires;
//! 2. pops everything waiting, whichever of the two woke it, splitting it
//!    into [`Observation`]s for the handler and [`Control`]s for the loop
//!    itself;
//! 3. records each of them, stamped with the instant of the pop;
//! 4. hands the observations to the game's [`Handler`] and gets back the
//!    [`Action`]s to send;
//! 5. stamps each action with the agent's id and the instant of the send,
//!    records it, sends the cycle's actions to the router as one
//!    [`CycleDispatch`], and records the cycle.
//!
//! Something that arrives while the agent is busy waits in the inbox and is
//! picked up at the start of the next cycle. Nothing is interrupted and
//! nothing is discarded, and every agent is always stale by exactly one
//! handling window.
//!
//! Everything a cycle pops is popped at its start, so every observation and
//! control in a cycle has the same `received`: the cycle's `t_start`. That
//! is what makes an agent's deliberation recoverable from the log without a
//! stamp for it, as ADR-0007 sets out: it is a sent action's `created` minus
//! the `received` of the observations in the same cycle, and the cycle
//! record groups them.
//!
//! The handler returns what to send rather than sending it, so the loop sees
//! everything that goes out and the trajectory it records is authoritative.
//! It also means a handler cannot speak as anybody but itself: the loop is
//! what writes the sender.
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
//! The loop exits when its inbox closes, meaning every sender has been
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
//!     fn handle(&mut self, observations: &[Observation<Chat>]) -> Vec<Action<Chat>> {
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
//! let (to_agent, inbox) = unbounded();
//! let (dispatches, from_agent) = unbounded();
//! let (records, writer) = Writer::spawn::<Chat>(Vec::new());
//! let peers = [AgentId::new("caller")].into();
//! let wiring = Wiring { id: "echo".into(), clock, inbox, dispatches, records, timeout: None, peers };
//! let agent = Agent::spawn(wiring, Echo, clock);
//!
//! let hello = Event::<Chat>::new("caller", ["echo"], clock.now(), String::from("hello"));
//! to_agent.send(Delivery::control(clock, Control::Start)).unwrap();
//! to_agent.send(Delivery::Event(hello.clone())).unwrap();
//! to_agent.send(Delivery::control(clock, Control::Stop)).unwrap();
//!
//! agent.join().unwrap();
//! let sent: Vec<Event<Chat>> = from_agent.iter().flat_map(|r: CycleDispatch<_>| r.sent).collect();
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
use crate::event::{AgentId, Control, Domain, Event};
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
    fn start(&mut self) -> Vec<Action<D>> {
        Vec::new()
    }

    /// Folds one cycle's observations into the agent's state and says what
    /// to send.
    ///
    /// Empty when the cycle was woken by the timeout.
    fn handle(&mut self, observations: &[Observation<D>]) -> Vec<Action<D>>;
}

/// Something on its way to an agent's inbox.
///
/// Until controls get a queue of their own, one inbox carries both kinds.
/// An event already knows when it was created; a control is stamped by
/// whoever sends it, which is the episode.
/// `Debug`, `Clone` and equality are written out for the same reason
/// [`Event`]'s are.
pub enum Delivery<D: Domain> {
    /// In-domain data, which becomes an [`Observation`] when popped.
    Event(Event<D>),
    /// An out-of-domain instruction, which the loop acts on itself.
    Control {
        /// The control.
        control: Control,
        /// When the episode sent it.
        created: Timestamp,
    },
}

impl<D: Domain> Delivery<D> {
    /// A control stamped with the clock's current time.
    #[must_use]
    pub fn control(clock: Clock, control: Control) -> Self {
        Self::Control {
            control,
            created: clock.now(),
        }
    }
}

impl<D: Domain> Created for Delivery<D> {
    fn created(&self) -> Timestamp {
        match self {
            Self::Event(event) => event.created,
            Self::Control { created, .. } => *created,
        }
    }
}

impl<D: Domain> fmt::Debug for Delivery<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Event(event) => f.debug_tuple("Event").field(event).finish(),
            Self::Control { control, created } => f
                .debug_struct("Control")
                .field("control", control)
                .field("created", created)
                .finish(),
        }
    }
}

impl<D: Domain> Clone for Delivery<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Event(event) => Self::Event(event.clone()),
            Self::Control { control, created } => Self::Control {
                control: *control,
                created: *created,
            },
        }
    }
}

impl<D: Domain> PartialEq for Delivery<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Event(a), Self::Event(b)) => a == b,
            (
                Self::Control { control, created },
                Self::Control {
                    control: other,
                    created: then,
                },
            ) => control == other && created == then,
            _ => false,
        }
    }
}

impl<D: Domain> Eq for Delivery<D> where D::Payload: Eq {}

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
    /// How many deliveries the cycle took off the inbox. A timeout is not a
    /// delivery.
    pub deliveries: usize,
    /// The events the cycle sent, stamped with this agent as sender, in the
    /// order the handler returned them.
    pub sent: Vec<Event<D>>,
}

/// Everything an agent's thread needs besides its handler and timer.
pub struct Wiring<D: Domain> {
    /// The agent's id: the sender on everything it emits and the `agent` on
    /// every record it writes.
    pub id: AgentId,
    /// The episode clock.
    pub clock: Clock,
    /// The agent's one receiver.
    pub inbox: Receiver<Delivery<D>>,
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

/// Why an agent's loop stopped before its inbox closed or it was told to
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

/// What ended a wait.
enum Wake<D: Domain> {
    /// Something arrived.
    Delivery(Delivery<D>),
    /// The pending deadline passed.
    Deadline,
    /// The inbox is closed and empty.
    Closed,
    /// The wake channel disconnected.
    TimerGone,
}

/// The state of an agent's thread.
struct Loop<D: Domain, H, T> {
    wiring: Wiring<D>,
    handler: H,
    timer: T,
    /// The next sequence number to assign.
    next_seq: u64,

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
            let (mut batch, mut closed) = (Vec::new(), false);
            let woken_by_deadline = match self.wait() {
                Wake::Delivery(delivery) => {
                    batch.push(delivery);
                    false
                }
                Wake::Deadline => true,
                Wake::Closed => break,
                Wake::TimerGone => return Err(Error::TimerClosed),
            };
            closed |= self.drain(&mut batch);
            let timed_out = woken_by_deadline || self.deadline_passed()?;
            if timed_out {
                self.take_deadline();
            }
            let t_start = self.wiring.clock.now();
            let stop = self.cycle(t_start, batch, timed_out)?;
            if stop || closed {
                break;
            }
        }
        Ok(self.handler)
    }

    /// Blocks until something arrives or the pending deadline fires.
    fn wait(&self) -> Wake<D> {
        let never = never();
        let wake = self.pending.as_ref().map_or(&never, |(_, wake)| wake);
        select! {
            recv(self.wiring.inbox) -> delivery => delivery.map_or(Wake::Closed, Wake::Delivery),
            recv(wake) -> fired => if fired.is_ok() { Wake::Deadline } else { Wake::TimerGone },
        }
    }

    /// Takes everything waiting in the inbox. Returns whether the inbox
    /// turned out to be closed.
    fn drain(&self, batch: &mut Vec<Delivery<D>>) -> bool {
        loop {
            match self.wiring.inbox.try_recv() {
                Ok(delivery) => batch.push(delivery),
                Err(TryRecvError::Empty) => return false,
                Err(TryRecvError::Disconnected) => return true,
            }
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

    /// Runs one cycle: record what was popped, hand the observations to the
    /// handler, send and record what comes back, and close with the cycle
    /// record. Returns whether the batch contained a stop.
    fn cycle(
        &mut self,
        t_start: Timestamp,
        batch: Vec<Delivery<D>>,
        timed_out: bool,
    ) -> Result<bool, Error> {
        let deliveries = batch.len();
        let (mut observations, mut inputs) = (Vec::with_capacity(deliveries), Vec::new());
        let (mut started, mut stopped) = (false, false);
        for delivery in batch {
            match delivery {
                Delivery::Control { control, created } => {
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
                Delivery::Event(event) => {
                    let observation = Observation {
                        event,
                        received: t_start,
                    };
                    inputs.push(self.record_observation(&observation)?);
                    observations.push(observation);
                }
            }
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
        actions.extend(self.handler.handle(&observations));

        let (mut sent, mut outputs) = (Vec::with_capacity(actions.len()), Vec::new());
        for Action {
            recipients,
            payload,
        } in actions
        {
            let recipients = match recipients {
                Recipients::Broadcast => self.wiring.peers.clone(),
                Recipients::To(recipients) => recipients,
            };
            let event = Event {
                sender: self.wiring.id.clone(),
                recipients,
                created: self.wiring.clock.now(),
                payload,
            };
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
    type TestDelivery = Delivery<TestDomain>;

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

        fn handle(&mut self, observations: &[Observation<TestDomain>]) -> Vec<Action<TestDomain>> {
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
        fn handle(&mut self, observations: &[Observation<TestDomain>]) -> Vec<Action<TestDomain>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.handle(observations)
        }
    }

    /// Broadcasts a step when it starts.
    struct Town;

    impl Handler<TestDomain> for Town {
        fn start(&mut self) -> Vec<Action<TestDomain>> {
            vec![Action::broadcast(TestPayload::Step(0))]
        }

        fn handle(&mut self, _: &[Observation<TestDomain>]) -> Vec<Action<TestDomain>> {
            Vec::new()
        }
    }

    /// A panicking handler.
    struct Faulty;

    impl Handler<TestDomain> for Faulty {
        fn handle(&mut self, _: &[Observation<TestDomain>]) -> Vec<Action<TestDomain>> {
            panic!("handler bug");
        }
    }

    /// An agent and the test's end of every channel it is wired to.
    struct Rig<H> {
        agent: Agent<H>,
        clock: Clock,
        inbox: Sender<TestDelivery>,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
        timer: ManualTimerControl,
    }

    /// The channels of a rig, before the agent is spawned on them.
    struct Wires {
        wiring: Wiring<TestDomain>,
        inbox: Sender<TestDelivery>,
        dispatches: Receiver<CycleDispatch<TestDomain>>,
        records: Receiver<LogRecord<TestDomain>>,
    }

    fn wires(timeout: Option<Duration>) -> Wires {
        let (inbox, receiver) = unbounded();
        let (outbox, dispatches) = unbounded();
        let (recorder, records) = unbounded();
        let wiring = Wiring {
            id: AgentId::new("a"),
            clock: Clock::start(),
            inbox: receiver,
            dispatches: outbox,
            records: recorder,
            timeout,
            peers: ["b", "c"].map(AgentId::new).into(),
        };
        Wires {
            wiring,
            inbox,
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
            inbox: wires.inbox,
            dispatches: wires.dispatches,
            records: wires.records,
            timer: control,
        }
    }

    impl<H> Rig<H> {
        fn send(&self, event: TestEvent) {
            self.inbox.send(Delivery::Event(event)).unwrap();
        }

        fn control(&self, control: Control) {
            self.inbox
                .send(Delivery::control(self.clock, control))
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

    #[test]
    fn a_cycle_pops_everything_waiting_in_the_inbox() {
        let wires = wires(None);
        wires
            .inbox
            .send(Delivery::control(wires.wiring.clock, Control::Start))
            .unwrap();
        for event in [step("b", 6), step("c", 3), step("b", 5)] {
            wires.inbox.send(Delivery::Event(event)).unwrap();
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        drop(wires.inbox);

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
        rig.send(step("b", 1));
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
        let clock = wires.wiring.clock;
        wires
            .inbox
            .send(Delivery::control(clock, Control::Start))
            .unwrap();
        wires
            .inbox
            .send(Delivery::control(clock, Control::Stop))
            .unwrap();
        wires.inbox.send(Delivery::Event(step("b", 1))).unwrap();
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        // The test still holds the inbox sender, so the join returning at all
        // is the stop path; and an event that arrived before the agent looked
        // is still handled, even after the stop.
        let handler = agent.join().unwrap();
        assert_eq!(handler.batches, [steps([1])]);
        drop(wires.inbox);
    }

    #[test]
    fn exits_when_the_inbox_closes_without_a_cycle() {
        let rig = rig(Recorder::default(), Some(EVERY));
        drop(rig.inbox);
        let handler = rig.agent.join().unwrap();
        assert!(handler.batches.is_empty());
        assert!(rig.dispatches.try_recv().is_err());
        assert!(rig.records.try_recv().is_err());
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
        rig.send(step("b", 1));
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
            .inbox
            .send(Delivery::Control {
                control: Control::Start,
                created: at(10),
            })
            .unwrap();
        wires
            .inbox
            .send(Delivery::Event(step_at("b", 6, at(20))))
            .unwrap();
        drop(wires.inbox);
        let clock = wires.wiring.clock;
        Agent::spawn(wires.wiring, Recorder::default(), clock)
            .join()
            .unwrap();
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
        drop(rig.inbox);
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

        // A delivery knows when it was created, whichever kind it is.
        assert_eq!(
            TestDelivery::Event(step_at("b", 1, at(40))).created(),
            at(40)
        );
        assert_eq!(
            TestDelivery::Control {
                control: Control::Stop,
                created: at(10)
            }
            .created(),
            at(10)
        );
    }
}
