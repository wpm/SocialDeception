//! The log records and the writer that hands them to its sinks.
//!
//! The log is a record of exactly what happened, and nothing more. The agent
//! loop writes its records as it folds and sends them over an in-process
//! channel to a [`Writer`]. Pairing an agent's observations with its actions
//! into a trajectory is what a parser builds *from* the log; the runtime
//! builds nothing (ADR-0017).
//!
//! The writer is the one place in a running episode that sees every record
//! as it happens, so it is where anything that wants the whole stream
//! attaches. It does not write anything itself: it hands each record to a
//! list of [`Sink`]s, each of which is a format bound to a destination.
//! [`JsonLines`] over a file is the log on disk; a domain that knows how to
//! render its own messages can add a sink that shows the game as it plays.
//!
//! # Times come in as instants and go out as offsets
//!
//! An actor reads `Instant::now()` at the moment something happens and puts
//! that [`Instant`] on the record unconverted, because converting is not its
//! job and because an `Instant` is the only time it has: the process's
//! monotonic clock has no epoch. The **writer** converts, with the episode
//! [`Clock`] it holds, and it converts *before* handing a record to any of
//! its sinks, so a live view and the file show identical numbers and no sink
//! ever sees an `Instant`.
//!
//! That is why every record type is generic over its time. A record an actor
//! wrote has `Instant`s on it; an [`Elapsed`] record is one a sink is given,
//! whose times are offsets from the episode's origin and which serialize as
//! whole nanoseconds. Only the second can be serialized at all, which is what
//! keeps a sink from seeing the first: it is the rule "only the writer
//! converts" written as a bound rather than as a comment.
//!
//! # The record types
//!
//! The log's first line is an [`EpisodeRecord`], the header: the episode's
//! wall-clock start, which is the only wall-clock time anywhere in the log.
//! Everything after it is an agent's.
//!
//! An agent runs one cycle per wake-up: it pops the controls at the head of
//! its queue and at most one message, hands that one observation to its
//! handler, and sends the actions that come back. Each cycle produces:
//!
//! - an [`ObservationRecord`] for the observation it popped, if it popped
//!   one, written the instant it is popped and carrying that instant, along
//!   with the `(from, seq)` of the message. At most one per cycle: a cycle
//!   handles one observation (ADR-0008);
//! - a [`ControlRecord`] per control popped, likewise, and with no `seq`: a
//!   control is not a message and nobody numbered it;
//! - an [`ActionRecord`] per action sent, written the instant it is sent and
//!   carrying that instant and the message's own `seq`. Every action a
//!   handler returns is sent, so there is a record per action and no other
//!   kind for one (ADR-0009);
//! - a [`CycleRecord`] closing the cycle: the handling window, what woke it,
//!   and the `(from, seq)` of the observation it was called with, if any.
//!
//! The last is the odd one out. A [`RewardRecord`] is written by the
//! **environment**, and belongs to the agent it names rather than to the
//! agent that wrote it; see [`RewardRecord`] and ADR-0007. It is also the
//! only record carrying something this module does not know the type of,
//! and it carries it already serialized, so that nothing here is generic
//! over a reward.
//!
//! An observation and an action record the same message from the two sides of
//! it, which is what makes the log joinable: an observation in one agent's
//! records matches the action in its sender's with the same `(from, seq)`.
//! **One send to five recipients is one message and one sequence number**, so
//! a single action joins to all of its observations at once, and two messages
//! sent in the same instant cannot tie. Nothing else links them, and nothing
//! else needs to.
//!
//! A sequence number is a **message's**, not a record's, and only a message
//! record has one. Controls, rewards and cycles carry none.
//!
//! # Why a record is written when it is
//!
//! An action is logged the instant it is sent and an observation or a
//! control the instant it is popped, because those are the instants the
//! records refer to. Recording an observation later would mean either
//! recording an arrival that had already passed or inventing one.
//!
//! # Ordering
//!
//! **Line order carries no meaning.** Records reach the writer from as many
//! threads as there are agents, in whatever order the channel delivers them,
//! so the file's order is the channel's and not the episode's. Order is given
//! by `t`, and by `seq` within a sender.
//!
//! What is still true of one agent's records is that a cycle record follows
//! every record of that cycle: the agent is one thread, and its next cycle
//! cannot start until the cycle record has been sent. A cycle names its
//! observation rather than listing record numbers, so the grouping a reader
//! wants comes from the cycle's window and that name.
//!
//! Every sink sees one order, the order the writer received the records off
//! the channel, so what a live view shows is what the file records, in the
//! same order and with the same numbers.
//!
//! If a [`Policy::Required`] sink fails mid-cycle, the writer stops, the
//! agent's loop exits with an error and its records end without a closing
//! cycle record. A [`Policy::Optional`] sink's failure costs that sink
//! alone: it is dropped, and the episode runs on.
//!
//! # On-disk format
//!
//! [`JsonLines`] writes one JSON object per line, wrapped in [`Record`],
//! whose `type` field is `episode`, `observation`, `action`, `control`,
//! `reward` or `cycle`. Every `t` is whole nanoseconds since the episode's
//! origin. Nothing here reads a log back; only `Serialize` is required of a
//! payload or a reward.
//!
//! ```json
//! {"type":"episode","start_unix_ns":1790630400000000000}
//! {"type":"control","agent":"alice","t":12,"control":"start"}
//! {"type":"cycle","agent":"alice","t_start":12,"t_stop":13,"woken":"queue"}
//! {"type":"observation","agent":"alice","t":55,"from":"moderator","seq":0,"message":{"sender":"moderator","recipients":["alice"],"payload":{"Request":{}}}}
//! {"type":"action","agent":"alice","t":90,"seq":0,"message":{"sender":"alice","recipients":["moderator"],"payload":{"Response":{}}}}
//! {"type":"cycle","agent":"alice","t_start":55,"t_stop":90,"woken":"queue","from":"moderator","seq":0}
//! {"type":"reward","agent":"alice","t":500,"value":1}
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crossbeam_channel::{Sender, unbounded};
use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::clock::Clock;
use crate::message::{ActorId, Control, Message, Payload};

/// The time on a record a [`Sink`] is given: how long after the episode's
/// origin it happened.
///
/// It serializes as whole nanoseconds, as a bare integer. A duration too long
/// to fit in 64 bits of them saturates: that is 584 years of episode, so the
/// alternative — refusing to write the record — would be a failure mode
/// nobody will ever reach guarding a number nobody will ever read.
///
/// It is `Copy`, and [`nanos`](Self::nanos) is what a sink that renders a
/// time reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct Elapsed(Duration);

impl Elapsed {
    /// How long after the origin, in whole nanoseconds, as the log writes it.
    #[must_use]
    pub fn nanos(self) -> u64 {
        u64::try_from(self.0.as_nanos()).unwrap_or(u64::MAX)
    }

    /// How long after the origin, exactly.
    #[must_use]
    pub const fn duration(self) -> Duration {
        self.0
    }
}

impl From<Duration> for Elapsed {
    fn from(since_origin: Duration) -> Self {
        Self(since_origin)
    }
}

impl Serialize for Elapsed {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u64(self.nanos())
    }
}

/// What started a cycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Woken {
    /// Something was waiting on the agent's queue.
    Queue,
    /// The agent's deadline had passed when the cycle began.
    ///
    /// It says when the cycle ran, not what it decided from. A deadline
    /// that passes while nothing is waiting runs a cycle that observes
    /// nothing and calls [`Handler::timeout`](crate::Handler::timeout); one
    /// that passes while a message is waiting joins that message's cycle,
    /// which observes it and calls
    /// [`Handler::handle`](crate::Handler::handle) like any other. Either
    /// way the deadline is retired and the next is measured from this
    /// cycle, which is what the mark is for.
    Timeout,
}

/// Which message a record is about: its sender and that sender's sequence
/// number for it.
///
/// This is the whole of the join (ADR-0017). An observation record carries
/// the key of the message it received and an action record carries its own,
/// beside the `agent` that is the sender, so a reader matches one action to
/// every observation of it without reference to any clock.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct Key {
    /// The agent that sent the message.
    pub from: ActorId,
    /// Which of that agent's messages it is.
    pub seq: u64,
}

impl Key {
    /// The key of `message`.
    #[must_use]
    pub fn of<P: Payload>(message: &Message<P>) -> Self {
        Self {
            from: message.sender.clone(),
            seq: message.seq,
        }
    }
}

/// The body of a `message` field: a message without the sequence number,
/// which the record carries at the top level instead.
///
/// It is not a type of its own anywhere else. The number sits beside `agent`
/// and `t` because it is half of what a reader joins on, and repeating it
/// inside the message would be two places to read one number from.
struct Envelope<'a, P: Payload>(&'a Message<P>);

impl<P: Payload> Serialize for Envelope<'_, P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut message = serializer.serialize_struct("Message", 3)?;
        message.serialize_field("sender", &self.0.sender)?;
        message.serialize_field("recipients", &self.0.recipients)?;
        message.serialize_field("payload", &self.0.payload)?;
        message.end()
    }
}

/// The log's header: when the episode started, on the wall clock.
///
/// It is the first line of every log and the **only** wall-clock time
/// anywhere in it. Everything else is a monotonic offset from the episode's
/// origin, which cannot be written absolutely and must not be, since the
/// wall clock can be adjusted mid-episode and can run backwards. The anchor
/// is here so that post-processing can line episodes up against each other,
/// or against a model provider's logs, if it ever needs to.
///
/// The anchor is the **origin's**, taken from the [`Clock`] and not read
/// afresh here. The two are one moment (see [`clock`](crate::clock)), so
/// offset zero is the instant this record names; an anchor read anywhere
/// else would be a moment the log's offsets are not measured from, and
/// lining two episodes up by it would be wrong by however far apart the two
/// readings fell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EpisodeRecord {
    /// The episode's origin as Unix nanoseconds.
    pub start_unix_ns: u64,
}

impl EpisodeRecord {
    /// The header for an episode whose origin is `clock`'s.
    #[must_use]
    pub const fn of(clock: Clock) -> Self {
        Self {
            start_unix_ns: clock.start_unix_ns(),
        }
    }
}

/// A message this agent popped off its queue.
///
/// `Serialize` is written out because the record's wire shape is not its
/// field shape: the time is nanoseconds, the key is spread into `from` and
/// `seq` beside it, and the message is written without the number the record
/// already carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationRecord<P: Payload, T = Instant> {
    /// The agent this record belongs to: the one that received the message.
    pub agent: ActorId,
    /// When this agent popped it: its cycle's `t_start`.
    pub t: T,
    /// The message it is: who sent it and which of theirs it is.
    pub key: Key,
    /// The message.
    pub message: Message<P>,
}

impl<P: Payload> Serialize for ObservationRecord<P, Elapsed> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ObservationRecord", 5)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("t", &self.t)?;
        record.serialize_field("from", &self.key.from)?;
        record.serialize_field("seq", &self.key.seq)?;
        record.serialize_field("message", &Envelope(&self.message))?;
        record.end()
    }
}

/// A message this agent sent.
///
/// `Serialize` is written out for the reason [`ObservationRecord`]'s is,
/// except that the sender is the `agent`, so the key's `from` would repeat it
/// and only the `seq` is written.
///
/// There is no arrival time: an action is logged by its sender, which knows
/// only when it sent it. When each recipient received it is in that
/// recipient's own observation record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionRecord<P: Payload, T = Instant> {
    /// The agent this record belongs to: the one that sent the message.
    ///
    /// For a relay it is the relaying agent, and the message's own sender is
    /// the actor being relayed, so the two differ; the key is the message's.
    pub agent: ActorId,
    /// When the loop sent it.
    pub t: T,
    /// The message it is.
    pub key: Key,
    /// The message as sent.
    pub message: Message<P>,
}

impl<P: Payload> Serialize for ActionRecord<P, Elapsed> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ActionRecord", 4)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("t", &self.t)?;
        record.serialize_field("seq", &self.key.seq)?;
        record.serialize_field("message", &Envelope(&self.message))?;
        record.end()
    }
}

/// A control this agent popped off its queue.
///
/// There is no `seq`: a control is not a message, so nobody numbered it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ControlRecord<T = Instant> {
    /// The agent this record belongs to.
    pub agent: ActorId,
    /// When this agent popped it: its cycle's `t_start`.
    pub t: T,
    /// The control.
    pub control: Control,
}

/// A reward the environment assigned to one agent.
///
/// It is the one record an agent does not write about itself. The
/// environment decides what an agent's behavior was worth, and `agent`
/// names the agent **rewarded**, whose records the reward belongs to, not
/// the environment that wrote it. Training joins a reward to that agent's
/// records by the actor id and the time (ADR-0007).
///
/// There is no `seq`: a reward is not a message, so there is nothing to
/// number it with, and nothing to join it to but the agent and the time.
///
/// The value is the reward **already serialized**. A reward's type is the
/// environment's, and the environment is the only thing in the runtime
/// that has one (ADR-0016): by the time a reward reaches this record it has
/// been assigned, and nothing downstream — not this record, not the
/// [`Writer`], not a [`Sink`] — does anything with it but write it out. So
/// it is serialized where it is assigned and carried as JSON from there,
/// which is what keeps the reward type off every type in this module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RewardRecord<T = Instant> {
    /// The agent rewarded, which is not the agent that wrote it.
    pub agent: ActorId,
    /// When the environment logged it.
    pub t: T,
    /// What the agent's behavior was worth, in the game's own units, as the
    /// environment serialized it.
    pub value: Value,
}

/// One cycle of an agent's loop: pop, hand to the handler, send.
///
/// The observation the cycle was called with is named by its key rather than
/// listed by record number, which is what lets a reader group a cycle's
/// records without the file's order meaning anything: the actions belong to
/// the cycle whose window contains them.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CycleRecord<T = Instant> {
    /// The agent whose cycle this was.
    pub agent: ActorId,
    /// When the agent popped its queue, and so when everything it popped was
    /// received.
    pub t_start: T,
    /// When it finished, including sending its actions.
    pub t_stop: T,
    /// What started the cycle.
    pub woken: Woken,
    /// The observation the handler was called with, if any. A cycle that
    /// popped only controls, or that woke on its deadline, has none.
    #[serde(flatten, skip_serializing_if = "Option::is_none")]
    pub observed: Option<Key>,
}

/// One record of the log: what a [`Sink`] is handed.
///
/// Internally tagged: the `type` field of each line names the record kind.
/// `T` is the record's time; see the [module documentation](self).
/// Only the [`Elapsed`] form can be serialized, and the bound below is what
/// says so: an unconverted record's time is an `Instant`, which means nothing
/// without the origin the writer holds and which serde cannot write.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(
    tag = "type",
    rename_all = "snake_case",
    bound = "ObservationRecord<P, T>: Serialize, ActionRecord<P, T>: Serialize, T: Serialize"
)]
pub enum Record<P: Payload, T = Instant> {
    /// The header: when the episode started.
    Episode(EpisodeRecord),
    /// A message some agent popped.
    Observation(ObservationRecord<P, T>),
    /// A message some agent sent.
    Action(ActionRecord<P, T>),
    /// A control some agent popped.
    Control(ControlRecord<T>),
    /// A reward the environment assigned to some agent.
    Reward(RewardRecord<T>),
    /// A cycle of some agent's loop.
    Cycle(CycleRecord<T>),
}

impl<P: Payload> Record<P, Instant> {
    /// The same record with every instant converted to an offset from
    /// `clock`'s origin, which is the form a [`Sink`] is given.
    #[must_use]
    pub fn elapsed(self, clock: Clock) -> Record<P, Elapsed> {
        let offset = |t| Elapsed::from(clock.offset(t));
        match self {
            Self::Episode(header) => Record::Episode(header),
            Self::Observation(record) => Record::Observation(ObservationRecord {
                agent: record.agent,
                t: offset(record.t),
                key: record.key,
                message: record.message,
            }),
            Self::Action(record) => Record::Action(ActionRecord {
                agent: record.agent,
                t: offset(record.t),
                key: record.key,
                message: record.message,
            }),
            Self::Control(record) => Record::Control(ControlRecord {
                agent: record.agent,
                t: offset(record.t),
                control: record.control,
            }),
            Self::Reward(record) => Record::Reward(RewardRecord {
                agent: record.agent,
                t: offset(record.t),
                value: record.value,
            }),
            Self::Cycle(record) => Record::Cycle(CycleRecord {
                agent: record.agent,
                t_start: offset(record.t_start),
                t_stop: offset(record.t_stop),
                woken: record.woken,
                observed: record.observed,
            }),
        }
    }
}

impl<P: Payload, T> From<EpisodeRecord> for Record<P, T> {
    fn from(record: EpisodeRecord) -> Self {
        Self::Episode(record)
    }
}

impl<P: Payload, T> From<ObservationRecord<P, T>> for Record<P, T> {
    fn from(record: ObservationRecord<P, T>) -> Self {
        Self::Observation(record)
    }
}

impl<P: Payload, T> From<ActionRecord<P, T>> for Record<P, T> {
    fn from(record: ActionRecord<P, T>) -> Self {
        Self::Action(record)
    }
}

impl<P: Payload, T> From<ControlRecord<T>> for Record<P, T> {
    fn from(record: ControlRecord<T>) -> Self {
        Self::Control(record)
    }
}

impl<P: Payload, T> From<RewardRecord<T>> for Record<P, T> {
    fn from(record: RewardRecord<T>) -> Self {
        Self::Reward(record)
    }
}

impl<P: Payload, T> From<CycleRecord<T>> for Record<P, T> {
    fn from(record: CycleRecord<T>) -> Self {
        Self::Cycle(record)
    }
}

/// A consumer of log records: a format bound to a destination.
///
/// The [`Writer`] hands every record it receives to each of its sinks, in
/// the order it was given them, which is the order the channel delivered
/// the records, and with every time already converted to an offset from the
/// episode's origin. [`JsonLines`] is the one the framework provides; a
/// domain that knows how to render its own messages provides its own, which
/// is why this is a trait and not an enumeration of the formats the
/// framework happens to know.
///
/// A sink runs on the writer's thread and owns its destination, so it need
/// not be `Sync`, but it must be `Send` to be moved onto that thread.
pub trait Sink<P: Payload>: Send {
    /// Takes one record, in the order the writer received it.
    ///
    /// # Errors
    ///
    /// Whatever the destination returned. What happens next is the sink's
    /// [`Policy`].
    fn record(&mut self, record: &Record<P, Elapsed>) -> io::Result<()>;

    /// Called once, after the last record: flush, close.
    ///
    /// # Errors
    ///
    /// Whatever the destination returned, treated exactly as an error from
    /// [`record`](Sink::record).
    fn finish(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// What a sink's failure costs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Policy {
    /// An error fails the run: the writer stops, every later send fails at
    /// the agent that attempted it, and [`Writer::join`] returns the error.
    ///
    /// This is what a log file wants: a run whose record of itself is
    /// incomplete is a run that did not happen.
    Required,
    /// An error drops this sink; the others carry on, and [`Writer::join`]
    /// does not report it.
    ///
    /// This is what a terminal wants: a reader that closed the pipe has seen
    /// all it wanted, and the game is no less played for it.
    Optional,
}

/// The log writer: a thread that owns the sinks, converts each record's
/// instants to offsets from the episode's origin, and hands each sink every
/// record it receives.
///
/// Records reach it over an in-process channel whose sender is handed out by
/// [`Writer::spawn`]; every agent gets a clone. The thread runs until every
/// sender has been dropped, then finishes each sink and exits.
/// [`Writer::join`] waits for that and reports how it went.
///
/// Its first act is the [`EpisodeRecord`] header, written before any agent
/// can have sent anything, so it is the first line of the log.
///
/// On a [`Policy::Required`] sink's failure the thread stops and drops its
/// receiver, so every subsequent `send` fails at the agent that attempted
/// it, and the failure is returned from [`Writer::join`]. A
/// [`Policy::Optional`] sink's failure removes that sink and nothing else.
#[derive(Debug)]
pub struct Writer {
    thread: JoinHandle<io::Result<()>>,
}

impl Writer {
    /// Starts a writer thread that converts each record with `clock` and
    /// hands it to every sink, in the order given, and returns the sender
    /// that feeds it.
    ///
    /// `clock` is the episode's, captured before this is called and shared
    /// with every agent, which is what puts a writer's offsets and an
    /// agent's own measurements on one timeline.
    ///
    /// A writer with no sinks at all is allowed: it drains the channel and
    /// discards what it receives, which is what an episode that records
    /// nothing wants.
    #[must_use]
    pub fn spawn<P: Payload>(
        sinks: Vec<(Box<dyn Sink<P>>, Policy)>,
        clock: Clock,
    ) -> (Sender<Record<P>>, Self) {
        let (sender, receiver) = unbounded::<Record<P>>();
        let thread = thread::spawn(move || {
            let mut live = sinks;
            let header = Record::<P, Elapsed>::Episode(EpisodeRecord::of(clock));
            deliver(&mut live, |sink| sink.record(&header))?;
            for record in receiver {
                let record = record.elapsed(clock);
                deliver(&mut live, |sink| sink.record(&record))?;
            }
            deliver(&mut live, |sink| sink.finish())
        });
        (sender, Self { thread })
    }

    /// Waits for the writer to finish.
    ///
    /// The writer finishes when every sender returned by [`Writer::spawn`]
    /// has been dropped, or earlier if a required sink failed. Drop the
    /// senders before calling this, or it never returns.
    ///
    /// # Errors
    ///
    /// The first required sink's error, or an error of kind
    /// [`io::ErrorKind::Other`] if the thread panicked. An optional sink's
    /// error is never reported here: it cost that sink and nothing else.
    pub fn join(self) -> io::Result<()> {
        self.thread
            .join()
            .map_err(|_| io::Error::other("log writer thread panicked"))?
    }

    /// Creates (or truncates) the file at `path` and starts a writer whose
    /// one sink writes JSON Lines to it, as a run whose log is that file
    /// requires.
    ///
    /// # Errors
    ///
    /// Whatever [`File::create`] returns.
    pub fn create<P: Payload>(
        path: impl AsRef<Path>,
        clock: Clock,
    ) -> io::Result<(Sender<Record<P>>, Self)> {
        let sink: Box<dyn Sink<P>> = Box::new(JsonLines::new(File::create(path)?));
        Ok(Self::spawn(vec![(sink, Policy::Required)], clock))
    }
}

/// Runs `act` on every live sink in turn, dropping the optional ones that
/// fail and returning on the first required one that does.
///
/// A sink that has failed is gone: it is neither given later records nor
/// finished, because a destination that refused one write has no reason to
/// accept the next.
fn deliver<P: Payload>(
    sinks: &mut Vec<(Box<dyn Sink<P>>, Policy)>,
    mut act: impl FnMut(&mut dyn Sink<P>) -> io::Result<()>,
) -> io::Result<()> {
    let mut failed = None;
    sinks.retain_mut(|(sink, policy)| match act(sink.as_mut()) {
        Ok(()) => true,
        Err(error) => {
            if *policy == Policy::Required {
                failed = Some(error);
            }
            false
        }
    });
    match failed {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

/// The JSON Lines sink: one JSON object per record, one record per line.
///
/// This is the log format described at the top of this module, and
/// the format of the file a run writes. The destination is buffered
/// internally, so there is no need to wrap it in a [`BufWriter`] first;
/// [`finish`](Sink::finish) flushes that buffer.
#[derive(Debug)]
pub struct JsonLines<W: Write> {
    out: BufWriter<W>,
}

impl<W: Write> JsonLines<W> {
    /// A sink that writes JSON Lines to `out`.
    pub fn new(out: W) -> Self {
        Self {
            out: BufWriter::new(out),
        }
    }
}

impl<P: Payload, W: Write + Send> Sink<P> for JsonLines<W> {
    fn record(&mut self, record: &Record<P, Elapsed>) -> io::Result<()> {
        serde_json::to_writer(&mut self.out, record)?;
        self.out.write_all(b"\n")
    }

    fn finish(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use serde_json::{Value, json};

    use super::*;
    use crate::testing::{Shared, TestPayload, header_anchor, joined, parse_lines, recording};

    /// An instant `nanos` after an origin that every record in one test
    /// shares, so that a converted record's `t` is exactly `nanos`.
    fn at(origin: Instant, nanos: u64) -> Instant {
        origin + Duration::from_nanos(nanos)
    }

    fn key(from: &str, seq: u64) -> Key {
        Key {
            from: ActorId::new(from),
            seq,
        }
    }

    /// A handful of records of every kind: agent `a` pops a start and a
    /// message from `b` in one cycle and replies to it.
    fn sample(origin: Instant) -> Vec<Record<TestPayload>> {
        let a = ActorId::new("a");
        vec![
            ControlRecord {
                agent: a.clone(),
                t: at(origin, 30),
                control: Control::Start,
            }
            .into(),
            ObservationRecord {
                agent: a.clone(),
                t: at(origin, 30),
                key: key("b", 4),
                message: Message::new("b", ["a"], 4, TestPayload::Step(6)),
            }
            .into(),
            ActionRecord {
                agent: a.clone(),
                t: at(origin, 40),
                key: key("a", 0),
                message: Message::new("a", ["b"], 0, TestPayload::Step(3)),
            }
            .into(),
            CycleRecord {
                agent: a,
                t_start: at(origin, 30),
                t_stop: at(origin, 50),
                woken: Woken::Queue,
                observed: Some(key("b", 4)),
            }
            .into(),
        ]
    }

    /// The lines `sample` writes, after the header.
    fn expected_lines() -> Vec<Value> {
        vec![
            json!({"type": "control", "agent": "a", "t": 30, "control": "start"}),
            json!({"type": "observation", "agent": "a", "t": 30, "from": "b", "seq": 4,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "t": 40, "seq": 0,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 3}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "from": "b", "seq": 4}),
        ]
    }

    /// Sends `sample` through a recording writer and returns the lines,
    /// header included.
    fn written(clock: Clock, records: Vec<Record<TestPayload>>) -> Vec<Value> {
        let (sender, writer, bytes) = recording(clock);
        for record in records {
            sender.send(record).unwrap();
        }
        drop(sender);
        parse_lines(&joined(writer, &bytes))
    }

    #[test]
    fn the_first_line_is_the_episode_header_and_the_rest_are_offsets() {
        let clock = Clock::start();
        let lines = written(clock, sample(clock.origin()));
        assert_eq!(
            header_anchor(&lines[0]),
            clock.start_unix_ns(),
            "the header anchors the log to the origin its offsets are measured \
             from, not to whenever the writer thread got around to it"
        );
        assert_eq!(lines[1..], expected_lines());
    }

    #[test]
    fn a_writer_converts_instants_to_offsets_before_any_sink_sees_them() {
        // The one place a time is converted. An actor puts an `Instant` on
        // the record; what a sink is handed is nanoseconds since the
        // origin, so the file and a live view cannot disagree.
        let clock = Clock::start();
        let record: Record<TestPayload> = ControlRecord {
            agent: ActorId::new("a"),
            t: at(clock.origin(), 7),
            control: Control::Stop,
        }
        .into();
        let Record::Control(converted) = record.clone().elapsed(clock) else {
            panic!("a control stays a control");
        };
        assert_eq!(converted.t, Elapsed::from(Duration::from_nanos(7)));
        // An instant before the origin is no time at all rather than a
        // negative one.
        let Record::Control(early) = Record::<TestPayload>::from(ControlRecord {
            agent: ActorId::new("a"),
            t: clock
                .origin()
                .checked_sub(Duration::from_secs(1))
                .expect("the process has been running for a second"),
            control: Control::Stop,
        })
        .elapsed(clock) else {
            panic!("a control stays a control");
        };
        assert_eq!(early.t, Elapsed::default());
    }

    #[test]
    fn a_message_record_carries_its_key_once_and_at_the_top() {
        // The message body is the sender, the recipients and the payload; the
        // sequence number sits beside `agent`, where a reader joins on it,
        // and nowhere else. An observation names the sender there too,
        // because its `agent` is the receiver; an action's `agent` is the
        // sender already.
        let lines = expected_lines();
        assert_eq!(lines[1]["from"], "b");
        assert_eq!(lines[2]["from"], Value::Null);
        for line in &lines[1..3] {
            assert!(line["seq"].is_u64(), "{line}");
            assert!(line["message"]["seq"].is_null(), "{line}");
        }
        // And in the order written, which is what a reader sees on disk.
        let clock = Clock::start();
        let written =
            serde_json::to_string(&sample(clock.origin())[1].clone().elapsed(clock)).unwrap();
        assert!(
            written
                .ends_with(r#""message":{"sender":"b","recipients":["a"],"payload":{"Step":6}}}"#),
            "{written}"
        );
    }

    #[test]
    fn a_reward_names_the_agent_rewarded_and_carries_no_sequence_number() {
        // Three fields and no more: the agent whose records it belongs to,
        // when the environment logged it, and what it is worth. No `seq`,
        // because a reward is not a message.
        // The value is already JSON by the time the record holds it: the
        // environment serialized it where it assigned it.
        let clock = Clock::start();
        let reward: Record<TestPayload> = RewardRecord {
            agent: ActorId::new("alice"),
            t: at(clock.origin(), 500),
            value: json!(1),
        }
        .into();
        let reward = reward.elapsed(clock);
        assert_eq!(
            serde_json::to_value(&reward).unwrap(),
            json!({"type": "reward", "agent": "alice", "t": 500, "value": 1})
        );
        assert_eq!(
            serde_json::to_string(&reward).unwrap(),
            r#"{"type":"reward","agent":"alice","t":500,"value":1}"#
        );
    }

    #[test]
    fn a_reward_record_is_a_value_like_any_other() {
        let reward = RewardRecord {
            agent: ActorId::new("alice"),
            t: Elapsed::from(Duration::from_nanos(500)),
            value: json!(-1),
        };
        assert_eq!(reward, reward.clone());
        assert_ne!(
            reward,
            RewardRecord {
                value: json!(1),
                ..reward.clone()
            }
        );
        assert!(format!("{reward:?}").starts_with("RewardRecord"));
        let line: Record<TestPayload, Elapsed> = reward.into();
        assert!(format!("{line:?}").starts_with("Reward"));
        assert_eq!(line, line.clone());
    }

    #[test]
    fn a_timeout_cycle_records_what_woke_it_and_observes_nothing() {
        let cycle: Record<TestPayload, Elapsed> = CycleRecord {
            agent: ActorId::new("a"),
            t_start: Elapsed::from(Duration::from_nanos(60)),
            t_stop: Elapsed::from(Duration::from_nanos(61)),
            woken: Woken::Timeout,
            observed: None,
        }
        .into();
        assert_eq!(
            serde_json::to_value(&cycle).unwrap(),
            json!({"type": "cycle", "agent": "a", "t_start": 60, "t_stop": 61,
                   "woken": "timeout"})
        );
    }

    #[test]
    fn an_overlong_offset_saturates_rather_than_failing() {
        let cycle: Record<TestPayload, Elapsed> = CycleRecord {
            agent: ActorId::new("a"),
            t_start: Elapsed::from(Duration::MAX),
            t_stop: Elapsed::from(Duration::MAX),
            woken: Woken::Queue,
            observed: None,
        }
        .into();
        assert_eq!(
            serde_json::to_value(&cycle).unwrap()["t_start"],
            json!(u64::MAX)
        );
    }

    #[test]
    fn every_line_is_one_object_and_the_stream_dispatches_on_type() {
        let clock = Clock::start();
        let (sender, writer, bytes) = recording(clock);
        for record in sample(clock.origin()) {
            sender.send(record).unwrap();
        }
        drop(sender);
        let bytes = joined(writer, &bytes);
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text.lines().count(), 5);
        let types: Vec<Value> = parse_lines(&bytes)
            .iter()
            .map(|line| line["type"].clone())
            .collect();
        assert_eq!(
            types,
            ["episode", "control", "observation", "action", "cycle"]
        );
        for line in text.lines() {
            assert!(!line.contains('\n'));
            assert!(line.starts_with('{') && line.ends_with('}'));
        }
    }

    #[test]
    fn writer_thread_survives_records_from_several_senders() {
        let clock = Clock::start();
        let (sender, writer, bytes) = recording(clock);
        let handles: Vec<_> = (0..4u64)
            .map(|i| {
                let sender = sender.clone();
                let origin = clock.origin();
                thread::spawn(move || {
                    let agent = ActorId::new(format!("agent-{i}"));
                    for seq in 0..10 {
                        let record: Record<TestPayload> = ActionRecord {
                            agent: agent.clone(),
                            t: at(origin, seq),
                            key: Key {
                                from: agent.clone(),
                                seq,
                            },
                            message: Message::new(
                                agent.clone(),
                                ["watcher"],
                                seq,
                                TestPayload::Step(seq),
                            ),
                        }
                        .into();
                        sender.send(record).unwrap();
                    }
                })
            })
            .collect();
        drop(sender);
        for handle in handles {
            handle.join().unwrap();
        }
        let lines = parse_lines(&joined(writer, &bytes));
        header_anchor(&lines[0]);
        assert_eq!(lines.len(), 41);
        // Each agent's records come out in its own order, whatever the
        // interleaving between agents.
        for i in 0..4 {
            let agent = format!("agent-{i}");
            let seqs: Vec<u64> = lines
                .iter()
                .filter(|line| line["agent"] == agent)
                .map(|line| line["seq"].as_u64().unwrap())
                .collect();
            assert_eq!(seqs, (0..10).collect::<Vec<_>>());
        }
    }

    #[test]
    fn writer_produces_a_file_on_disk() {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let path = std::env::temp_dir().join(format!(
            "social-deception-{}-{}.jsonl",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let clock = Clock::start();
        let (sender, writer) = Writer::create(&path, clock).unwrap();
        for record in sample(clock.origin()) {
            sender.send(record).unwrap();
        }
        drop(sender);
        writer.join().unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        let lines = parse_lines(&bytes);
        header_anchor(&lines[0]);
        assert_eq!(lines[1..], expected_lines());
    }

    /// A sink that refuses everything.
    #[derive(Debug)]
    struct BrokenSink;

    impl Write for BrokenSink {
        fn write(&mut self, _: &[u8]) -> io::Result<usize> {
            Err(io::Error::new(io::ErrorKind::StorageFull, "no room"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::new(io::ErrorKind::StorageFull, "no room"))
        }
    }

    /// A sink that records every record it is given, and can be told to
    /// refuse the rest from a given one onwards.
    #[derive(Debug, Clone)]
    struct Spy {
        seen: Arc<Mutex<Vec<String>>>,
        fails_from: Option<usize>,
    }

    impl Spy {
        /// A sink that accepts everything.
        fn new() -> Self {
            Self {
                seen: Arc::new(Mutex::new(Vec::new())),
                fails_from: None,
            }
        }

        /// A sink that accepts the first `n` records and refuses the rest.
        fn failing_after(n: usize) -> Self {
            Self {
                fails_from: Some(n),
                ..Self::new()
            }
        }

        /// The records it took, each as the JSON it would have written.
        fn seen(&self) -> Vec<String> {
            self.seen.lock().unwrap().clone()
        }
    }

    impl Sink<TestPayload> for Spy {
        fn record(&mut self, record: &Record<TestPayload, Elapsed>) -> io::Result<()> {
            let mut seen = self.seen.lock().unwrap();
            if self.fails_from.is_some_and(|n| seen.len() >= n) {
                return Err(io::Error::new(io::ErrorKind::BrokenPipe, "no reader"));
            }
            // A panic, not a `?`: a `Record` cannot fail to serialize, so
            // a failure here is a bug in the fixture. Converting it to an
            // `io::Error` would hand the writer the one signal this spy
            // exists to control, and a fixture bug would arrive disguised
            // as the sink failure these tests are built to tell apart.
            seen.push(serde_json::to_string(record).unwrap());
            Ok(())
        }
    }

    /// Sends `sample` to `sender` and joins `writer`.
    fn play(clock: Clock, sender: Sender<Record<TestPayload>>, writer: Writer) -> io::Result<()> {
        for record in sample(clock.origin()) {
            let _ = sender.send(record);
        }
        drop(sender);
        writer.join()
    }

    /// Sends `sample` through a writer over `sinks`, each under its policy.
    fn play_into(sinks: [(Spy, Policy); 2]) -> io::Result<()> {
        let sinks: Vec<(Box<dyn Sink<TestPayload>>, Policy)> = sinks
            .into_iter()
            .map(|(sink, policy)| (Box::new(sink) as Box<dyn Sink<TestPayload>>, policy))
            .collect();
        let clock = Clock::start();
        let (sender, writer) = Writer::spawn(sinks, clock);
        play(clock, sender, writer)
    }

    #[test]
    fn a_required_sink_s_failure_is_loud_at_both_ends() {
        let clock = Clock::start();
        let broken: Box<dyn Sink<TestPayload>> = Box::new(JsonLines::new(BrokenSink));
        let (sender, writer) = Writer::spawn(vec![(broken, Policy::Required)], clock);
        // The header and a record small enough sit in the buffer, so
        // nothing has failed yet.
        let record: Record<TestPayload> = CycleRecord {
            agent: ActorId::new("a"),
            t_start: clock.origin(),
            t_stop: clock.origin(),
            woken: Woken::Queue,
            observed: None,
        }
        .into();
        // A record larger than the buffer is written while the loop is still
        // running; the thread stops and drops its receiver at that point.
        let big: Record<TestPayload> = ActionRecord {
            agent: ActorId::new("a"),
            t: clock.origin(),
            key: key("a", 0),
            message: Message::new(
                "a",
                (0..20_000)
                    .map(|i| format!("agent-{i}"))
                    .collect::<BTreeSet<_>>(),
                0,
                TestPayload::Step(0),
            ),
        }
        .into();
        sender.send(record).unwrap();
        sender.send(big).unwrap();
        // join waits for the thread to have failed before the sender is
        // checked.
        let error = writer.join().unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        let late: Record<TestPayload> = ControlRecord {
            agent: ActorId::new("a"),
            t: clock.origin(),
            control: Control::Stop,
        }
        .into();
        assert!(
            sender.send(late).is_err(),
            "sends after a write failure must fail"
        );
    }

    #[test]
    fn a_required_sink_that_fails_mid_run_stops_the_writer() {
        // The same failure, from a sink that took the header and then
        // refused: the run ends where the sink did.
        let survivor = Spy::new();
        let error = play_into([
            (Spy::failing_after(3), Policy::Required),
            (survivor.clone(), Policy::Optional),
        ])
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        // The writer stopped where the required sink did, so the sink beside
        // it saw no more than the record that failed.
        assert_eq!(survivor.seen().len(), 4);
    }

    /// A destination whose `flush` refuses, and which counts the flushes it
    /// was asked for. Writes succeed, so nothing fails before `finish`.
    #[derive(Debug, Clone, Default)]
    struct UnflushableSink(Arc<Mutex<usize>>);

    impl Write for UnflushableSink {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            *self.0.lock().unwrap() += 1;
            Err(io::Error::new(io::ErrorKind::StorageFull, "no room"))
        }
    }

    /// The first of `sample`, converted: what a sink is handed.
    fn one(clock: Clock) -> Record<TestPayload, Elapsed> {
        sample(clock.origin()).remove(0).elapsed(clock)
    }

    #[test]
    fn json_lines_holds_a_record_in_its_buffer_until_it_is_finished() {
        // Called directly, without a writer: `JsonLines` buffers, so a
        // record small enough to fit is nowhere yet, and `finish` is what
        // puts it on the destination. This is why `finish` is not the
        // trait's default no-op.
        let clock = Clock::start();
        let bytes = Shared::new();
        let mut sink = JsonLines::new(bytes.clone());
        Sink::<TestPayload>::record(&mut sink, &one(clock)).unwrap();
        assert!(
            bytes.bytes().is_empty(),
            "a buffered record must not be on the destination yet"
        );

        Sink::<TestPayload>::finish(&mut sink).unwrap();
        assert_eq!(parse_lines(&bytes.bytes()), expected_lines()[..1]);
    }

    #[test]
    fn a_flush_that_fails_is_reported_by_finish() {
        // `BufWriter`'s own `Drop` flushes and discards the error, so a
        // log could be truncated in silence. `finish` is the call
        // that gets to report it, and it must.
        let clock = Clock::start();
        let destination = UnflushableSink::default();
        let mut sink = JsonLines::new(destination.clone());
        Sink::<TestPayload>::record(&mut sink, &one(clock)).unwrap();
        let error = Sink::<TestPayload>::finish(&mut sink).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::StorageFull);
        assert_eq!(*destination.0.lock().unwrap(), 1, "finish flushed once");
    }

    #[test]
    fn every_sink_sees_every_record_in_the_same_order() {
        let first = Spy::new();
        let second = Spy::new();
        play_into([
            (first.clone(), Policy::Required),
            (second.clone(), Policy::Optional),
        ])
        .unwrap();

        assert_eq!(first.seen().len(), expected_lines().len() + 1);
        assert_eq!(first.seen(), second.seen());
        // The same records, and the same ones the file would have held.
        let taken: Vec<Value> = first
            .seen()
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        header_anchor(&taken[0]);
        assert_eq!(taken[1..], expected_lines());
    }

    #[test]
    fn a_failing_optional_sink_is_dropped_and_nothing_else_notices() {
        let dropped = Spy::failing_after(2);
        let survivor = Spy::new();
        play_into([
            (dropped.clone(), Policy::Optional),
            (survivor.clone(), Policy::Required),
        ])
        .unwrap();
        // It took two and was gone; the run and the other sink went on.
        assert_eq!(dropped.seen().len(), 2);
        assert_eq!(survivor.seen().len(), expected_lines().len() + 1);
    }

    #[test]
    fn a_writer_with_no_sinks_drains_and_joins_cleanly() {
        let clock = Clock::start();
        let (sender, writer) = Writer::spawn::<TestPayload>(Vec::new(), clock);
        play(clock, sender, writer).unwrap();
    }

    #[test]
    fn a_key_names_the_message_it_belongs_to() {
        let message = Message::new("alice", ["bob"], 7, TestPayload::Step(1));
        assert_eq!(Key::of(&message), key("alice", 7));
        // Ordered, so that a reader can put a sender's messages in order
        // whatever order the file gave them in.
        let keys = BTreeSet::from([key("alice", 8), key("alice", 7), key("bob", 0)]);
        assert_eq!(
            keys.into_iter().collect::<Vec<_>>(),
            [key("alice", 7), key("alice", 8), key("bob", 0)]
        );
    }
}
