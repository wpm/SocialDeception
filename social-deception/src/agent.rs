//! The agent: one thread, one inbox, and a drain-and-fold loop.
//!
//! An agent's life is a fold over the events that arrive on its inbox. Each
//! pass of the loop:
//!
//! 1. waits until something arrives or its deadline fires;
//! 2. drains the whole inbox, whichever of the two woke it, and adds a
//!    [`Event::Think`] to the end of the batch if the deadline has passed;
//! 3. records the batch, one event record per event;
//! 4. hands the batch to the environment's [`Handler`] and gets back what to
//!    send;
//! 5. records what came out, sends it to the router as one [`CycleReport`],
//!    and records the pass as a cycle.
//!
//! An event that arrives while the agent is busy waits in the inbox and is
//! picked up at the start of the next pass. Nothing is interrupted and
//! nothing is discarded, and every agent is always stale by exactly one
//! handling window.
//!
//! The handler returns what to send rather than sending it, so the loop sees
//! everything that goes out and the trajectory it records is authoritative.
//!
//! # Deadlines
//!
//! An agent may be given a think interval. From the pass in which it receives
//! [`Control::Start`], a deadline is pending one interval ahead; when it
//! passes, the agent wakes with a `Think`, and the next deadline is one
//! interval after the pass that handled it. Deadlines are absolute, and the
//! wake channel for one is asked of the [`TimerSource`] once and kept until it
//! fires, so a message arriving before the deadline leaves the deadline where
//! it was. An agent without an interval blocks until something arrives.
//!
//! # Termination
//!
//! The loop exits when its inbox closes, meaning every sender has been
//! dropped and nothing is left to drain, or after the pass in which it
//! received [`Control::Stop`], whichever comes first. Every record of that
//! last pass has been sent to the writer before the thread returns.
//!
//! # Example
//!
//! An agent that echoes each message back to its sender, wired to a router
//! stand-in and a trajectory writer:
//!
//! ```
//! use crossbeam_channel::unbounded;
//! use social_deception::{
//!     Agent, Clock, Control, CycleReport, Delivery, Event, Handler, Outgoing, Wiring, Writer,
//! };
//!
//! struct Echo;
//!
//! impl Handler<String> for Echo {
//!     fn handle(&mut self, events: &[Event<String>]) -> Vec<Outgoing<String>> {
//!         events
//!             .iter()
//!             .filter_map(|event| match event {
//!                 Event::Message { sender, payload, .. } => {
//!                     Some(Outgoing::to([sender.clone()], payload.clone()))
//!                 }
//!                 _ => None,
//!             })
//!             .collect()
//!     }
//! }
//!
//! let clock = Clock::start();
//! let (to_agent, inbox) = unbounded();
//! let (reports, from_agent) = unbounded();
//! let (records, writer) = Writer::spawn(Vec::new());
//! let wiring = Wiring { id: "echo".into(), clock, inbox, reports, records, think_every: None };
//! let agent = Agent::spawn(wiring, Echo, clock);
//!
//! let hello = Event::message("caller", ["echo"], String::from("hello"));
//! to_agent.send(Delivery::now(clock, Event::Control(Control::Start))).unwrap();
//! to_agent.send(Delivery::now(clock, hello)).unwrap();
//! to_agent.send(Delivery::now(clock, Event::Control(Control::Stop))).unwrap();
//!
//! agent.join().unwrap();
//! let sent: Vec<Event<String>> = from_agent.iter().flat_map(|r: CycleReport<_>| r.sent).collect();
//! assert_eq!(sent, [Event::message("echo", ["caller"], String::from("hello"))]);
//! let trajectory = writer.join().unwrap();
//! assert!(!trajectory.is_empty());
//! ```

use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};
use std::fmt;
use std::panic;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, TryRecvError, never, select};

use crate::clock::{Clock, Timestamp};
use crate::event::{AgentId, Control, Event, Payload};
use crate::timer::TimerSource;
use crate::trajectory::{CycleRecord, EventRecord, LogRecord, Seq};

/// An environment's behaviour for one agent.
///
/// The loop calls [`handle`](Handler::handle) once per pass with everything
/// that came out of the drain, in arrival order, with a `Think` last if the
/// agent's deadline has passed. The handler returns what the agent sends in
/// reply; the loop attaches the agent's own id as the sender, records it, and
/// does the sending. Agent state lives in the implementing type.
pub trait Handler<P> {
    /// Folds one batch of events into the agent's state and says what to
    /// send.
    fn handle(&mut self, events: &[Event<P>]) -> Vec<Outgoing<P>>;
}

/// A message a handler wants sent.
///
/// The sender is the agent itself, and it is the loop that says so; a handler
/// cannot speak as anyone else.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Outgoing<P> {
    /// The agents to send it to.
    pub recipients: BTreeSet<AgentId>,
    /// What to say.
    pub payload: P,
}

impl<P> Outgoing<P> {
    /// A message to the given recipients.
    pub fn to<I, A>(recipients: I, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<AgentId>,
    {
        Self {
            recipients: recipients.into_iter().map(Into::into).collect(),
            payload,
        }
    }
}

/// An event on its way to an agent's inbox.
///
/// Whoever puts an event on the inbox stamps it with the time it did so. That
/// is the event's arrival time in the trajectory, and it has to be set at the
/// sending end because the agent only sees the event when it drains, which
/// may be a whole handling window later.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Delivery<P> {
    /// When the event was put on the inbox.
    pub time: Timestamp,
    /// The event.
    pub event: Event<P>,
}

impl<P> Delivery<P> {
    /// A delivery stamped with the clock's current time.
    pub fn now(clock: Clock, event: Event<P>) -> Self {
        Self {
            time: clock.now(),
            event,
        }
    }
}

/// What one pass of an agent's loop tells the router.
///
/// One report is sent per pass, after the pass's outputs have been recorded
/// and before its cycle record is written. Because the number of deliveries
/// consumed and the messages produced arrive together, whoever counts
/// in-flight deliveries never sees a pass's inputs settled before its outputs
/// exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CycleReport<P> {
    /// The agent whose pass this was.
    pub agent: AgentId,
    /// How many deliveries the pass took off the inbox. A `Think` is not a
    /// delivery.
    pub deliveries: usize,
    /// The messages the pass produced, as events with this agent as sender,
    /// in the order the handler returned them.
    pub sent: Vec<Event<P>>,
}

/// Everything an agent's thread needs besides its handler and timer.
#[derive(Debug)]
pub struct Wiring<P> {
    /// The agent's id: the sender on everything it emits and the `agent` on
    /// every record it writes.
    pub id: AgentId,
    /// The episode clock.
    pub clock: Clock,
    /// The agent's one receiver.
    pub inbox: Receiver<Delivery<P>>,
    /// Where each pass's report goes.
    pub reports: Sender<CycleReport<P>>,
    /// Where the agent's trajectory goes.
    pub records: Sender<LogRecord<P>>,
    /// How often the agent thinks unprompted, or `None` for an agent that
    /// only ever reacts.
    pub think_every: Option<Duration>,
}

/// Why an agent's loop stopped before its inbox closed or it was told to
/// stop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Error {
    /// A record could not be sent: the trajectory writer has gone away.
    WriterClosed,
    /// A report could not be sent: the router has gone away.
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
            deadlines: BinaryHeap::new(),
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
enum Wake<P> {
    /// Something arrived.
    Delivery(Delivery<P>),
    /// The pending deadline passed.
    Deadline,
    /// The inbox is closed and empty.
    Closed,
    /// The wake channel disconnected.
    TimerGone,
}

/// The state of an agent's thread.
struct Loop<P, H, T> {
    wiring: Wiring<P>,
    handler: H,
    timer: T,
    /// The next sequence number to assign.
    next_seq: u64,
    /// Deadlines not yet reached, earliest first. The think interval keeps at
    /// most one here; the heap is where several would go.
    deadlines: BinaryHeap<Reverse<Timestamp>>,
    /// The earliest deadline and the wake channel asked for it, kept across
    /// passes until it fires.
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
            self.arm();
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
            let due = if woken_by_deadline || self.deadline_passed()? {
                self.take_deadline()
            } else {
                None
            };
            let t_start = self.wiring.clock.now();
            let stop = self.pass(t_start, batch, due)?;
            if stop || closed {
                break;
            }
        }
        Ok(self.handler)
    }

    /// Makes sure the pending wake channel is for the earliest deadline.
    fn arm(&mut self) {
        let earliest = self.deadlines.peek().map(|Reverse(deadline)| *deadline);
        match (earliest, &self.pending) {
            (Some(deadline), Some((pending, _))) if *pending == deadline => {}
            (Some(deadline), _) => {
                self.pending = Some((deadline, self.timer.wake_at(deadline)));
            }
            (None, _) => self.pending = None,
        }
    }

    /// Blocks until something arrives or the pending deadline fires.
    fn wait(&self) -> Wake<P> {
        let never = never();
        let wake = self.pending.as_ref().map_or(&never, |(_, wake)| wake);
        select! {
            recv(self.wiring.inbox) -> delivery => delivery.map_or(Wake::Closed, Wake::Delivery),
            recv(wake) -> fired => if fired.is_ok() { Wake::Deadline } else { Wake::TimerGone },
        }
    }

    /// Takes everything waiting in the inbox. Returns whether the inbox
    /// turned out to be closed.
    fn drain(&self, batch: &mut Vec<Delivery<P>>) -> bool {
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

    /// Retires the deadline that just fired and returns it.
    fn take_deadline(&mut self) -> Option<Timestamp> {
        self.pending = None;
        self.deadlines.pop().map(|Reverse(deadline)| deadline)
    }

    /// Schedules a think one interval after `from`, if the agent thinks at
    /// all.
    fn schedule_think(&mut self, from: Timestamp) {
        if let Some(every) = self.wiring.think_every {
            self.deadlines.push(Reverse(from + every));
        }
    }

    /// Handles one batch. `due` is the deadline that fired, if one did, and
    /// becomes a `Think` at the end of the batch. Returns whether the batch
    /// contained a stop.
    fn pass(
        &mut self,
        t_start: Timestamp,
        batch: Vec<Delivery<P>>,
        due: Option<Timestamp>,
    ) -> Result<bool, Error> {
        let deliveries = batch.len();
        let (mut events, mut inputs) = (Vec::with_capacity(deliveries + 1), Vec::new());
        let (mut started, mut stopped) = (false, false);
        for Delivery { time, event } in batch {
            match event {
                Event::Control(Control::Start) => started = true,
                Event::Control(Control::Stop) => stopped = true,
                _ => {}
            }
            inputs.push(self.record(time, event.clone())?);
            events.push(event);
        }
        if let Some(deadline) = due {
            inputs.push(self.record(deadline, Event::Think)?);
            events.push(Event::Think);
            self.schedule_think(t_start);
        }
        if started {
            self.schedule_think(t_start);
        }

        let outgoing = self.handler.handle(&events);

        let (mut sent, mut outputs) = (Vec::with_capacity(outgoing.len()), Vec::new());
        for Outgoing {
            recipients,
            payload,
        } in outgoing
        {
            let event = Event::Message {
                sender: self.wiring.id.clone(),
                recipients,
                payload,
            };
            outputs.push(self.record(self.wiring.clock.now(), event.clone())?);
            sent.push(event);
        }
        let report = CycleReport {
            agent: self.wiring.id.clone(),
            deliveries,
            sent,
        };
        self.wiring
            .reports
            .send(report)
            .map_err(|_| Error::RouterClosed)?;
        let cycle = CycleRecord {
            agent: self.wiring.id.clone(),
            t_start,
            t_stop: self.wiring.clock.now(),
            inputs,
            outputs,
        };
        self.wiring
            .records
            .send(cycle.into())
            .map_err(|_| Error::WriterClosed)?;
        Ok(stopped)
    }

    /// Writes an event record with the next sequence number and returns that
    /// number.
    fn record(&mut self, time: Timestamp, event: Event<P>) -> Result<Seq, Error> {
        let seq = Seq(self.next_seq);
        self.next_seq += 1;
        let record = EventRecord {
            agent: self.wiring.id.clone(),
            seq,
            time,
            event,
        };
        self.wiring
            .records
            .send(record.into())
            .map_err(|_| Error::WriterClosed)?;
        Ok(seq)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossbeam_channel::unbounded;
    use serde::Serialize;
    use serde_json::{Value, json};

    use super::*;
    use crate::timer::{ManualTimer, ManualTimerControl};
    use crate::trajectory::Writer;

    /// How long a test waits on a channel before giving up. A test only ever
    /// waits this long when it has already failed.
    const PATIENCE: Duration = Duration::from_secs(10);

    /// A think interval. Its value never matters: the manual timer decides
    /// when deadlines fire.
    const EVERY: Duration = Duration::from_secs(1);

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    enum TestPayload {
        Step(u64),
    }

    type TestEvent = Event<TestPayload>;

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    fn step(sender: &str, n: u64) -> TestEvent {
        Event::message(sender, ["a"], TestPayload::Step(n))
    }

    fn deliver(time: u64, event: TestEvent) -> Delivery<TestPayload> {
        Delivery {
            time: at(time),
            event,
        }
    }

    fn start() -> TestEvent {
        Event::Control(Control::Start)
    }

    fn stop() -> TestEvent {
        Event::Control(Control::Stop)
    }

    fn recv<T>(receiver: &Receiver<T>) -> T {
        receiver.recv_timeout(PATIENCE).expect("nothing arrived")
    }

    /// Remembers every batch it was given and replies to each message with
    /// the next step, addressed to whoever sent it.
    #[derive(Debug, Default, PartialEq, Eq)]
    struct Recorder {
        batches: Vec<Vec<TestEvent>>,
    }

    impl Handler<TestPayload> for Recorder {
        fn handle(&mut self, events: &[TestEvent]) -> Vec<Outgoing<TestPayload>> {
            self.batches.push(events.to_vec());
            events
                .iter()
                .filter_map(|event| match event {
                    Event::Message {
                        sender,
                        payload: TestPayload::Step(n),
                        ..
                    } => Some(Outgoing::to([sender.clone()], TestPayload::Step(n + 1))),
                    _ => None,
                })
                .collect()
        }
    }

    /// A recorder that, on entering each pass, tells the test it is busy and
    /// then waits to be released. That is how a test makes events arrive
    /// while the agent is provably mid-pass.
    #[derive(Debug)]
    struct Gated {
        inner: Recorder,
        entered: Sender<()>,
        release: Receiver<()>,
    }

    impl Handler<TestPayload> for Gated {
        fn handle(&mut self, events: &[TestEvent]) -> Vec<Outgoing<TestPayload>> {
            self.entered.send(()).unwrap();
            self.release.recv().unwrap();
            self.inner.handle(events)
        }
    }

    /// A panicking handler.
    struct Faulty;

    impl Handler<TestPayload> for Faulty {
        fn handle(&mut self, _: &[TestEvent]) -> Vec<Outgoing<TestPayload>> {
            panic!("handler bug");
        }
    }

    /// An agent and the test's end of every channel it is wired to.
    struct Rig<H> {
        agent: Agent<H>,
        clock: Clock,
        inbox: Sender<Delivery<TestPayload>>,
        reports: Receiver<CycleReport<TestPayload>>,
        records: Receiver<LogRecord<TestPayload>>,
        timer: ManualTimerControl,
    }

    /// The channels of a rig, before the agent is spawned on them.
    struct Wires {
        wiring: Wiring<TestPayload>,
        inbox: Sender<Delivery<TestPayload>>,
        reports: Receiver<CycleReport<TestPayload>>,
        records: Receiver<LogRecord<TestPayload>>,
    }

    fn wires(think_every: Option<Duration>) -> Wires {
        let (inbox, receiver) = unbounded();
        let (reporter, reports) = unbounded();
        let (recorder, records) = unbounded();
        let wiring = Wiring {
            id: AgentId::new("a"),
            clock: Clock::start(),
            inbox: receiver,
            reports: reporter,
            records: recorder,
            think_every,
        };
        Wires {
            wiring,
            inbox,
            reports,
            records,
        }
    }

    fn rig<H: Handler<TestPayload> + Send + 'static>(
        handler: H,
        think_every: Option<Duration>,
    ) -> Rig<H> {
        let wires = wires(think_every);
        let (timer, control) = ManualTimer::new();
        let clock = wires.wiring.clock;
        Rig {
            agent: Agent::spawn(wires.wiring, handler, timer),
            clock,
            inbox: wires.inbox,
            reports: wires.reports,
            records: wires.records,
            timer: control,
        }
    }

    impl<H> Rig<H> {
        fn send(&self, event: TestEvent) {
            self.inbox.send(Delivery::now(self.clock, event)).unwrap();
        }

        fn report(&self) -> CycleReport<TestPayload> {
            recv(&self.reports)
        }

        /// The records of one pass: its event records and then its cycle
        /// record.
        fn cycle(&self) -> (Vec<EventRecord<TestPayload>>, CycleRecord) {
            let mut events = Vec::new();
            loop {
                match recv(&self.records) {
                    LogRecord::Event(record) => events.push(record),
                    LogRecord::Cycle(cycle) => return (events, cycle),
                }
            }
        }

        fn kinds(&self) -> Vec<TestEvent> {
            self.cycle()
                .0
                .into_iter()
                .map(|record| record.event)
                .collect()
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

    #[test]
    fn a_pass_drains_everything_waiting_in_the_inbox() {
        let wires = wires(None);
        for event in [start(), step("b", 6), step("c", 3), step("b", 5)] {
            wires
                .inbox
                .send(Delivery::now(wires.wiring.clock, event))
                .unwrap();
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        drop(wires.inbox);

        let handler = agent.join().unwrap();
        assert_eq!(
            handler.batches,
            [vec![start(), step("b", 6), step("c", 3), step("b", 5)]]
        );
        let reports: Vec<_> = wires.reports.iter().collect();
        assert_eq!(
            reports,
            [CycleReport {
                agent: AgentId::new("a"),
                deliveries: 4,
                sent: vec![
                    Event::message("a", ["b"], TestPayload::Step(7)),
                    Event::message("a", ["c"], TestPayload::Step(4)),
                    Event::message("a", ["b"], TestPayload::Step(6)),
                ],
            }]
        );
    }

    #[test]
    fn events_arriving_mid_pass_wait_for_the_next_pass() {
        let (rig, busy, release) = gated();
        rig.send(start());
        recv(&busy);
        rig.send(step("b", 1));
        rig.send(step("b", 2));
        release.send(()).unwrap();
        recv(&busy);
        release.send(()).unwrap();
        rig.send(stop());
        recv(&busy);
        release.send(()).unwrap();

        let handler = rig.agent.join().unwrap();
        assert_eq!(
            handler.inner.batches,
            [
                vec![start()],
                vec![step("b", 1), step("b", 2)],
                vec![stop()]
            ]
        );
    }

    #[test]
    fn exits_after_the_pass_that_contained_stop() {
        let wires = wires(None);
        for event in [start(), stop(), step("b", 1)] {
            wires
                .inbox
                .send(Delivery::now(wires.wiring.clock, event))
                .unwrap();
        }
        let agent = Agent::spawn(wires.wiring, Recorder::default(), Clock::start());
        // The test still holds the inbox sender, so the join returning at all
        // is the stop path; and an event that arrived before the agent looked
        // is still handled, even after the stop.
        let handler = agent.join().unwrap();
        assert_eq!(handler.batches, [vec![start(), stop(), step("b", 1)]]);
        drop(wires.inbox);
    }

    #[test]
    fn exits_when_the_inbox_closes_without_a_pass() {
        let rig = rig(Recorder::default(), Some(EVERY));
        drop(rig.inbox);
        let handler = rig.agent.join().unwrap();
        assert!(handler.batches.is_empty());
        assert!(rig.reports.try_recv().is_err());
        assert!(rig.records.try_recv().is_err());
    }

    #[test]
    fn think_fires_when_the_deadline_passes_and_not_before() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.send(start());
        assert_eq!(rig.report().deliveries, 1);
        let (_, started) = rig.cycle();
        let first = recv(rig.timer.requests());
        assert_eq!(first, started.t_start + EVERY);

        rig.send(step("b", 1));
        assert_eq!(rig.report().deliveries, 1);
        assert_eq!(
            rig.kinds(),
            [
                step("b", 1),
                Event::message("a", ["b"], TestPayload::Step(2))
            ]
        );

        rig.timer.fire().unwrap();
        assert_eq!(rig.report().deliveries, 0);
        let (events, thought) = rig.cycle();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event, Event::Think);
        assert_eq!(
            events[0].time, first,
            "a think is stamped with its deadline"
        );
        assert_eq!(thought.inputs, [events[0].seq]);

        // The message did not move the deadline: the only request between
        // the first and the one made after thinking is none at all.
        let second = recv(rig.timer.requests());
        assert_eq!(second, thought.t_start + EVERY);
        assert!(second > first);

        rig.send(stop());
        let handler = rig.agent.join().unwrap();
        assert_eq!(
            handler.batches,
            [
                vec![start()],
                vec![step("b", 1)],
                vec![Event::Think],
                vec![stop()]
            ]
        );
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn a_deadline_that_passed_while_busy_is_handled_with_what_arrived() {
        let (rig, busy, release) = gated();
        rig.send(start());
        recv(&busy);
        rig.send(step("b", 1));
        // The fire is queued before the deadline is even armed; the manual
        // timer keeps it for the first wait.
        rig.timer.fire().unwrap();
        release.send(()).unwrap();
        recv(&busy);
        release.send(()).unwrap();
        rig.send(stop());
        recv(&busy);
        release.send(()).unwrap();

        let handler = rig.agent.join().unwrap();
        assert_eq!(
            handler.inner.batches,
            [
                vec![start()],
                vec![step("b", 1), Event::Think],
                vec![stop()]
            ]
        );
    }

    #[test]
    fn an_agent_without_an_interval_never_thinks() {
        let rig = rig(Recorder::default(), None);
        rig.send(start());
        rig.send(step("b", 1));
        rig.send(stop());
        let handler = rig.agent.join().unwrap();
        let seen: Vec<_> = handler.batches.into_iter().flatten().collect();
        assert!(!seen.contains(&Event::Think));
        assert!(rig.timer.requests().try_recv().is_err());
    }

    #[test]
    fn thinking_starts_only_once_the_episode_has() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.send(step("b", 1));
        rig.report();
        rig.send(start());
        rig.report();
        // The first request is made only after the start was handled, so it
        // is the first thing on the channel either way; what the test can
        // check is which pass it was measured from.
        let deadline = recv(rig.timer.requests());
        rig.cycle();
        let (_, started) = rig.cycle();
        assert_eq!(deadline, started.t_start + EVERY);
        rig.send(stop());
        rig.agent.join().unwrap();
    }

    fn lines(bytes: &[u8]) -> Vec<Value> {
        std::str::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    #[test]
    fn a_pass_is_recorded_as_events_then_a_cycle() {
        let mut wires = wires(None);
        let (records, writer) = Writer::spawn(Vec::new());
        wires.wiring.records = records;
        wires.inbox.send(deliver(10, start())).unwrap();
        wires.inbox.send(deliver(20, step("b", 6))).unwrap();
        drop(wires.inbox);
        let clock = wires.wiring.clock;
        Agent::spawn(wires.wiring, Recorder::default(), clock)
            .join()
            .unwrap();
        let lines = lines(&writer.join().unwrap());

        assert_eq!(lines.len(), 4);
        assert_eq!(
            lines[0],
            json!({"type": "event", "agent": "a", "seq": 0, "time": 10,
                   "event": {"kind": "control", "control": "start"}})
        );
        assert_eq!(
            lines[1],
            json!({"type": "event", "agent": "a", "seq": 1, "time": 20,
                   "event": {"kind": "message", "sender": "b", "recipients": ["a"],
                             "payload": {"Step": 6}}})
        );
        let sent = lines[2]["time"].as_u64().unwrap();
        assert_eq!(
            lines[2],
            json!({"type": "event", "agent": "a", "seq": 2, "time": sent,
                   "event": {"kind": "message", "sender": "a", "recipients": ["b"],
                             "payload": {"Step": 7}}})
        );
        let (t_start, t_stop) = (lines[3]["t_start"].as_u64(), lines[3]["t_stop"].as_u64());
        assert_eq!(
            lines[3],
            json!({"type": "cycle", "agent": "a", "t_start": t_start, "t_stop": t_stop,
                   "inputs": [0, 1], "outputs": [2]})
        );
        assert!(20 < t_start.unwrap());
        assert!(t_start.unwrap() <= sent && sent <= t_stop.unwrap());
    }

    #[test]
    fn sequence_numbers_run_on_across_passes() {
        let rig = rig(Recorder::default(), None);
        rig.send(start());
        let (first, cycle) = rig.cycle();
        assert_eq!(first.iter().map(|r| r.seq).collect::<Vec<_>>(), [Seq(0)]);
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(0)], vec![]));
        rig.send(step("b", 1));
        let (second, cycle) = rig.cycle();
        assert_eq!(
            second.iter().map(|r| r.seq).collect::<Vec<_>>(),
            [Seq(1), Seq(2)]
        );
        assert_eq!((cycle.inputs, cycle.outputs), (vec![Seq(1)], vec![Seq(2)]));
        assert!(second.iter().all(|r| r.agent == AgentId::new("a")));
        rig.send(stop());
        rig.agent.join().unwrap();
    }

    #[test]
    fn a_vanished_writer_is_an_error() {
        let rig = rig(Recorder::default(), None);
        rig.send(start());
        drop(rig.records);
        assert_eq!(rig.agent.join(), Err(Error::WriterClosed));
    }

    #[test]
    fn a_vanished_router_is_an_error() {
        let rig = rig(Recorder::default(), None);
        rig.send(start());
        drop(rig.reports);
        assert_eq!(rig.agent.join(), Err(Error::RouterClosed));
    }

    #[test]
    fn a_vanished_timer_is_an_error() {
        let rig = rig(Recorder::default(), Some(EVERY));
        rig.send(start());
        rig.report();
        recv(rig.timer.requests());
        drop(rig.timer);
        assert_eq!(rig.agent.join(), Err(Error::TimerClosed));
    }

    #[test]
    #[should_panic(expected = "handler bug")]
    fn a_handler_panic_reaches_whoever_joins() {
        let rig = rig(Faulty, None);
        rig.send(start());
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
}
