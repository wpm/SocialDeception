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
//! [`JsonLines`] over a file is the log on disk; a domain that knows
//! how to render its own messages can add a sink that shows the game as it
//! plays.
//!
//! # Five record types
//!
//! An agent runs one cycle per wake-up: it pops the controls at the head of
//! its queue and at most one message, hands that one observation to its
//! handler, and sends the actions that come back. Each cycle produces:
//!
//! - an [`ObservationRecord`] for the observation it popped, if it popped
//!   one, written the instant it is popped, carrying both the instant its
//!   sender created it and the instant this agent received it. At most one
//!   per cycle: a cycle handles one observation (ADR-0008);
//! - a [`ControlRecord`] per control popped, likewise;
//! - an [`ActionRecord`] per action sent, written the instant it is sent,
//!   carrying the `created` stamp the loop has just given it. Every action
//!   a handler returns is sent, so there is a record per action and no
//!   other kind for one (ADR-0009);
//! - a [`CycleRecord`] closing the cycle: the handling window, what woke it,
//!   and the sequence numbers of everything popped and everything sent.
//!
//! The fifth is the odd one out. A [`RewardRecord`] is written by the
//! **environment**, and belongs to the agent it names rather than to the
//! agent that wrote it; see [`RewardRecord`] and ADR-0007. It is also the
//! only record carrying something this module does not know the type of,
//! and it carries it already serialized, so that nothing here is generic
//! over a reward.
//!
//! An observation and an action record the same message from the two sides of
//! it, which is what makes the log joinable: an observation in one agent's
//! records matches the action in its sender's whose `created` and
//! payload it carries. Nothing else links them, and nothing else needs to.
//!
//! Sequence numbers are per agent and cover every non-cycle record, inputs
//! and outputs alike. A reward has none, because it is not the agent loop's
//! to number.
//!
//! # Why a record is written when it is
//!
//! An action is logged the instant it is sent and an observation or a
//! control the instant it is popped, because those are the instants the
//! stamps refer to. Recording an observation later would mean either
//! recording a receipt time that had already passed or inventing one.
//!
//! # Ordering
//!
//! Within one agent's records, each cycle is a contiguous run: one record
//! per input in pop order, then one per output in the order the handler
//! returned them, then that cycle's own record. Nothing else from that agent
//! appears between them: the agent is one thread, and its next cycle cannot
//! start until the cycle record has been sent.
//!
//! So a cycle record always follows every record it references, and,
//! filtered to one agent, the records between two cycle records belong to
//! the cycle that ends the run. A cycle's `inputs` followed by its `outputs`
//! is exactly the sequence numbers since that agent's previous cycle; the
//! lists are a consistency check on the grouping, not the only way to
//! recover it.
//!
//! A cycle woken by the timeout with nothing waiting has no inputs at all,
//! so the smallest cycle is a cycle record alone. Every cycle woken by the
//! queue has at least one input, since it ran only because something was
//! waiting, and no cycle has more than one observation among them.
//!
//! These guarantees are per agent. Every agent sends to the same writer, so
//! records from different agents interleave arbitrarily, and a recipient's
//! cycle can precede the sender's action record that caused it.
//!
//! Every sink sees that one order. A record reaches each sink in the order
//! the writer received it off the channel, so the per-agent guarantees above
//! hold for every sink alike, and so does the arbitrary interleaving between
//! agents: what a live view shows is what the file records, in the same
//! order.
//!
//! A reward record is outside all of it. It belongs to the agent it names
//! and was written by another, so it is in no cycle of that agent's, sits
//! between two of its cycles wherever the writer happened to take it, and
//! carries no sequence number to place it. Its `created` is what places it.
//!
//! If a [`Policy::Required`] sink fails mid-cycle, the writer stops, the
//! agent's loop exits with an error and its records end without a closing
//! cycle record. A [`Policy::Optional`] sink's failure costs that sink
//! alone: it is dropped, and the episode runs on.
//!
//! # On-disk format
//!
//! [`JsonLines`] writes one JSON object per line, wrapped in [`Record`],
//! whose `type` field is `observation`, `action`, `control`, `reward` or
//! `cycle`.
//! Nothing here reads a log back; only `Serialize` is required of a payload
//! or a reward.
//!
//! ```json
//! {"type":"control","agent":"alice","seq":0,"created":10,"received":12,"control":"start"}
//! {"type":"cycle","agent":"alice","t_start":12,"t_stop":13,"woken":"queue","inputs":[0],"outputs":[]}
//! {"type":"observation","agent":"alice","seq":1,"created":40,"received":55,"message":{"sender":"moderator","recipients":["alice"],"payload":{"Request":{}}}}
//! {"type":"action","agent":"alice","seq":2,"created":90,"message":{"sender":"alice","recipients":["moderator"],"payload":{"Response":{}}}}
//! {"type":"cycle","agent":"alice","t_start":55,"t_stop":90,"woken":"queue","inputs":[1],"outputs":[2]}
//! {"type":"reward","agent":"alice","created":500,"value":1}
//! ```

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Sender, unbounded};
use serde::{Serialize, Serializer};
use serde_json::Value;

use crate::clock::{Created, Timestamp};
use crate::message::{ActorId, Control, Message, Payload};

/// An agent's sequence number for one of its own records.
///
/// Sequence numbers are assigned by the agent loop and are meaningful only
/// together with the actor id: two agents both have a sequence number 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

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

/// The body of a `message` field: a message without its creation time, which
/// the record carries at the top level instead.
///
/// It is not a type of its own anywhere else. The creation time sits beside
/// `agent` and `seq` because it is a property of the record's subject and a
/// reader joins on it, and repeating it inside the message would be two
/// places to read the same instant from.
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

/// A message this agent popped off its queue.
///
/// `Serialize` is written out because the record's wire shape is not its
/// field shape: the message is written without the `created` this record
/// already carries at the top level.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObservationRecord<P: Payload> {
    /// The agent whose records this one belongs to.
    pub agent: ActorId,
    /// The agent's sequence number for it.
    pub seq: Seq,
    /// When its sender sent it.
    pub created: Timestamp,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
    /// The message.
    pub message: Message<P>,
}

impl<P: Payload> Serialize for ObservationRecord<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ObservationRecord", 5)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("seq", &self.seq)?;
        record.serialize_field("created", &self.created)?;
        record.serialize_field("received", &self.received)?;
        record.serialize_field("message", &Envelope(&self.message))?;
        record.end()
    }
}

/// A message this agent sent.
///
/// `Serialize` is written out for the reason [`ObservationRecord`]'s is.
///
/// There is no `received`: an action is logged by its sender, which knows
/// only when it sent it. When each recipient received it is in that
/// recipient's own observation record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionRecord<P: Payload> {
    /// The agent whose records this one belongs to.
    pub agent: ActorId,
    /// The agent's sequence number for it.
    pub seq: Seq,
    /// When the loop sent it, which is the `created` on the wire.
    pub created: Timestamp,
    /// The message as sent.
    pub message: Message<P>,
}

impl<P: Payload> Serialize for ActionRecord<P> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ActionRecord", 4)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("seq", &self.seq)?;
        record.serialize_field("created", &self.created)?;
        record.serialize_field("message", &Envelope(&self.message))?;
        record.end()
    }
}

/// A control this agent popped off its queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ControlRecord {
    /// The agent whose records this one belongs to.
    pub agent: ActorId,
    /// The agent's sequence number for it.
    pub seq: Seq,
    /// When the episode sent it.
    pub created: Timestamp,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
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
/// There is no `seq`. Sequence numbers are the agent loop's to assign, and
/// this record was not written by that loop, so numbering it would either
/// invent a number nobody issued or perturb the numbering of the records
/// the agent did write. There is no `received` either: a reward is logged,
/// never sent, so nobody ever receives it, and it implements
/// [`Created`] alone.
///
/// The value is the reward **already serialized**. A reward's type is the
/// environment's, and the environment is the only thing in the runtime
/// that has one (ADR-0016): by the time a reward reaches this record it has
/// been assigned, and nothing downstream — not this record, not the
/// [`Writer`], not a [`Sink`] — does anything with it but write it out. So
/// it is serialized where it is assigned and carried as JSON from there,
/// which is what keeps the reward type off every type in this module.
///
/// `Debug`, `Clone` and equality are derived: no field mentions a game's
/// types any more.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RewardRecord {
    /// The agent rewarded, whose records this one belongs to.
    pub agent: ActorId,
    /// When the environment logged it.
    pub created: Timestamp,
    /// What the agent's behavior was worth, in the game's own units, as the
    /// environment serialized it.
    pub value: Value,
}

impl Created for RewardRecord {
    fn created(&self) -> Timestamp {
        self.created
    }
}

/// One cycle of an agent's loop: pop, hand to the handler, send.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CycleRecord {
    /// The agent whose cycle this was.
    pub agent: ActorId,
    /// When the agent popped its queue, and so when everything it popped was
    /// received.
    pub t_start: Timestamp,
    /// When it finished, including sending its actions.
    pub t_stop: Timestamp,
    /// What started the cycle.
    pub woken: Woken,
    /// The sequence numbers of everything popped, observations and controls
    /// alike, in the order the agent saw them.
    pub inputs: Vec<Seq>,
    /// The sequence numbers of the actions sent, in the order the handler
    /// returned them.
    pub outputs: Vec<Seq>,
}

/// One record of the log: what a [`Sink`] is handed.
///
/// Internally tagged: the `type` field of each line names the record kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case", bound = "")]
pub enum Record<P: Payload> {
    /// A message some agent popped.
    Observation(ObservationRecord<P>),
    /// A message some agent sent.
    Action(ActionRecord<P>),
    /// A control some agent popped.
    Control(ControlRecord),
    /// A reward the environment assigned to some agent.
    Reward(RewardRecord),
    /// A cycle of some agent's loop.
    Cycle(CycleRecord),
}

impl<P: Payload> From<ObservationRecord<P>> for Record<P> {
    fn from(record: ObservationRecord<P>) -> Self {
        Self::Observation(record)
    }
}

impl<P: Payload> From<ActionRecord<P>> for Record<P> {
    fn from(record: ActionRecord<P>) -> Self {
        Self::Action(record)
    }
}

impl<P: Payload> From<ControlRecord> for Record<P> {
    fn from(record: ControlRecord) -> Self {
        Self::Control(record)
    }
}

impl<P: Payload> From<RewardRecord> for Record<P> {
    fn from(record: RewardRecord) -> Self {
        Self::Reward(record)
    }
}

impl<P: Payload> From<CycleRecord> for Record<P> {
    fn from(record: CycleRecord) -> Self {
        Self::Cycle(record)
    }
}

/// A consumer of log records: a format bound to a destination.
///
/// The [`Writer`] hands every record it receives to each of its sinks, in
/// the order it was given them, which is the order the channel delivered
/// the records. [`JsonLines`] is the one the framework provides; a domain
/// that knows how to render its own messages provides its own, which is why
/// this is a trait and not an enumeration of the formats the framework
/// happens to know.
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
    fn record(&mut self, record: &Record<P>) -> io::Result<()>;

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

/// The log writer: a thread that owns the sinks and hands each one
/// every record it receives.
///
/// Records reach it over an in-process channel whose sender is handed out by
/// [`Writer::spawn`]; every agent gets a clone. The thread runs until every
/// sender has been dropped, then finishes each sink and exits.
/// [`Writer::join`] waits for that and reports how it went.
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
    /// Starts a writer thread that hands each record to every sink, in the
    /// order given, and returns the sender that feeds it.
    ///
    /// A writer with no sinks at all is allowed: it drains the channel and
    /// discards what it receives, which is what an episode that records
    /// nothing wants.
    #[must_use]
    pub fn spawn<P: Payload>(sinks: Vec<(Box<dyn Sink<P>>, Policy)>) -> (Sender<Record<P>>, Self) {
        let (sender, receiver) = unbounded::<Record<P>>();
        let thread = thread::spawn(move || {
            let mut live = sinks;
            for record in receiver {
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
    pub fn create<P: Payload>(path: impl AsRef<Path>) -> io::Result<(Sender<Record<P>>, Self)> {
        let sink: Box<dyn Sink<P>> = Box::new(JsonLines::new(File::create(path)?));
        Ok(Self::spawn(vec![(sink, Policy::Required)]))
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
    fn record(&mut self, record: &Record<P>) -> io::Result<()> {
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
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::*;
    use crate::testing::{Shared, TestPayload, joined, parse_lines, recording};

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// A handful of records of every kind: agent `a` pops a start and a
    /// message in one cycle and replies to `b`.
    fn sample() -> Vec<Record<TestPayload>> {
        let a = ActorId::new("a");
        vec![
            ControlRecord {
                agent: a.clone(),
                seq: Seq(0),
                created: at(10),
                received: at(30),
                control: Control::Start,
            }
            .into(),
            ObservationRecord {
                agent: a.clone(),
                seq: Seq(1),
                created: at(20),
                received: at(30),
                message: Message::new("b", ["a"], at(20), TestPayload::Step(6)),
            }
            .into(),
            ActionRecord {
                agent: a.clone(),
                seq: Seq(2),
                created: at(40),
                message: Message::new("a", ["b"], at(40), TestPayload::Step(3)),
            }
            .into(),
            CycleRecord {
                agent: a,
                t_start: at(30),
                t_stop: at(50),
                woken: Woken::Queue,
                inputs: vec![Seq(0), Seq(1)],
                outputs: vec![Seq(2)],
            }
            .into(),
        ]
    }

    fn expected_lines() -> Vec<Value> {
        vec![
            json!({"type": "control", "agent": "a", "seq": 0, "created": 10, "received": 30,
                   "control": "start"}),
            json!({"type": "observation", "agent": "a", "seq": 1, "created": 20, "received": 30,
                   "message": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "seq": 2, "created": 40,
                   "message": {"sender": "a", "recipients": ["b"], "payload": {"Step": 3}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "inputs": [0, 1], "outputs": [2]}),
        ]
    }

    #[test]
    fn records_round_trip_through_the_writer_as_jsonl() {
        let (sender, writer, bytes) = recording();
        for record in sample() {
            sender.send(record).unwrap();
        }
        drop(sender);
        assert_eq!(parse_lines(&joined(writer, &bytes)), expected_lines());
    }

    #[test]
    fn a_message_record_carries_its_creation_time_once_and_at_the_top() {
        // The message body is the sender, the recipients and the payload; the
        // instant it was created sits beside `seq`, where a reader joins on
        // it, and nowhere else.
        let lines = expected_lines();
        for line in &lines[1..3] {
            assert!(line["created"].is_u64(), "{line}");
            assert!(line["message"]["created"].is_null(), "{line}");
        }
        // And in the order written, which is what a reader sees on disk.
        let written = serde_json::to_string(&sample()[1]).unwrap();
        assert!(
            written
                .ends_with(r#""message":{"sender":"b","recipients":["a"],"payload":{"Step":6}}}"#),
            "{written}"
        );
    }

    #[test]
    fn a_reward_names_the_agent_rewarded_and_carries_no_sequence_number() {
        // Three fields and no more: the agent whose records it belongs to,
        // when the environment logged it, and what it is worth. No
        // `seq`, because the agent loop did not write it, and no
        // `received`, because nobody received it.
        // The value is already JSON by the time the record holds it: the
        // environment serialized it where it assigned it, so the line reads
        // exactly as it did when the record was generic over a reward type.
        let reward: Record<TestPayload> = RewardRecord {
            agent: ActorId::new("alice"),
            created: at(500),
            value: json!(1),
        }
        .into();
        assert_eq!(
            serde_json::to_value(&reward).unwrap(),
            json!({"type": "reward", "agent": "alice", "created": 500, "value": 1})
        );
        assert_eq!(
            serde_json::to_string(&reward).unwrap(),
            r#"{"type":"reward","agent":"alice","created":500,"value":1}"#
        );
    }

    #[test]
    fn a_reward_is_created_and_never_received() {
        // `Created` and not `Received`: a reward is logged, never sent,
        // so there is no instant at which anybody got it.
        let reward = RewardRecord {
            agent: ActorId::new("alice"),
            created: at(500),
            value: json!(-1),
        };
        assert_eq!(reward.created(), at(500));
        assert_eq!(reward, reward.clone());
        assert_ne!(
            reward,
            RewardRecord {
                value: json!(1),
                ..reward.clone()
            }
        );
        assert!(format!("{reward:?}").starts_with("RewardRecord"));
        let line: Record<TestPayload> = reward.into();
        assert!(format!("{line:?}").starts_with("Reward"));
        assert_eq!(line, line.clone());
    }

    #[test]
    fn a_timeout_cycle_records_what_woke_it_and_no_inputs() {
        let cycle: Record<TestPayload> = CycleRecord {
            agent: ActorId::new("a"),
            t_start: at(60),
            t_stop: at(61),
            woken: Woken::Timeout,
            inputs: vec![],
            outputs: vec![],
        }
        .into();
        assert_eq!(
            serde_json::to_value(&cycle).unwrap(),
            json!({"type": "cycle", "agent": "a", "t_start": 60, "t_stop": 61,
                   "woken": "timeout", "inputs": [], "outputs": []})
        );
    }

    #[test]
    fn every_line_is_one_object_and_the_stream_dispatches_on_type() {
        let (sender, writer, bytes) = recording();
        for record in sample() {
            sender.send(record).unwrap();
        }
        drop(sender);
        let bytes = joined(writer, &bytes);
        let text = std::str::from_utf8(&bytes).unwrap();
        assert_eq!(text.lines().count(), 4);
        let types: Vec<Value> = parse_lines(&bytes)
            .iter()
            .map(|line| line["type"].clone())
            .collect();
        assert_eq!(types, ["control", "observation", "action", "cycle"]);
        for line in text.lines() {
            assert!(!line.contains('\n'));
            assert!(line.starts_with('{') && line.ends_with('}'));
        }
    }

    #[test]
    fn writer_thread_survives_records_from_several_senders() {
        let (sender, writer, bytes) = recording();
        let handles: Vec<_> = (0..4u64)
            .map(|i| {
                let sender = sender.clone();
                thread::spawn(move || {
                    let agent = ActorId::new(format!("agent-{i}"));
                    for seq in 0..10 {
                        let record: Record<TestPayload> = ControlRecord {
                            agent: agent.clone(),
                            seq: Seq(seq),
                            created: at(seq),
                            received: at(seq + 1),
                            control: Control::Stop,
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
        assert_eq!(lines.len(), 40);
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
        let (sender, writer) = Writer::create(&path).unwrap();
        for record in sample() {
            sender.send(record).unwrap();
        }
        drop(sender);
        writer.join().unwrap();
        let bytes = fs::read(&path).unwrap();
        fs::remove_file(&path).unwrap();
        assert_eq!(parse_lines(&bytes), expected_lines());
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
        fn record(&mut self, record: &Record<TestPayload>) -> io::Result<()> {
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
    fn play(sender: Sender<Record<TestPayload>>, writer: Writer) -> io::Result<()> {
        for record in sample() {
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
        let (sender, writer) = Writer::spawn(sinks);
        play(sender, writer)
    }

    #[test]
    fn a_required_sink_s_failure_is_loud_at_both_ends() {
        let broken: Box<dyn Sink<TestPayload>> = Box::new(JsonLines::new(BrokenSink));
        let (sender, writer) = Writer::spawn(vec![(broken, Policy::Required)]);
        // A record small enough to sit in the buffer.
        let record: Record<TestPayload> = CycleRecord {
            agent: ActorId::new("a"),
            t_start: at(0),
            t_stop: at(1),
            woken: Woken::Queue,
            inputs: vec![Seq(0)],
            outputs: vec![],
        }
        .into();
        // A record larger than the buffer is written while the loop is still
        // running; the thread stops and drops its receiver at that point.
        let big: Record<TestPayload> = ActionRecord {
            agent: ActorId::new("a"),
            seq: Seq(0),
            created: at(0),
            message: Message::new(
                "a",
                (0..20_000)
                    .map(|i| format!("agent-{i}"))
                    .collect::<BTreeSet<_>>(),
                at(0),
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
            seq: Seq(1),
            created: at(2),
            received: at(3),
            control: Control::Stop,
        }
        .into();
        assert!(
            sender.send(late).is_err(),
            "sends after a write failure must fail"
        );
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

    #[test]
    fn json_lines_holds_a_record_in_its_buffer_until_it_is_finished() {
        // Called directly, without a writer: `JsonLines` buffers, so a
        // record small enough to fit is nowhere yet, and `finish` is what
        // puts it on the destination. This is why `finish` is not the
        // trait's default no-op.
        let bytes = Shared::new();
        let mut sink = JsonLines::new(bytes.clone());
        let record = &sample()[0];
        Sink::<TestPayload>::record(&mut sink, record).unwrap();
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
        let destination = UnflushableSink::default();
        let mut sink = JsonLines::new(destination.clone());
        Sink::<TestPayload>::record(&mut sink, &sample()[0]).unwrap();
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

        let expected: Vec<String> = expected_lines()
            .iter()
            .map(|line| serde_json::to_string(line).unwrap())
            .collect();
        assert_eq!(first.seen().len(), expected.len());
        assert_eq!(first.seen(), second.seen());
        // The same records, and the same ones the file would have held.
        let taken: Vec<Value> = first
            .seen()
            .iter()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(taken, expected_lines());
    }

    #[test]
    fn a_failing_required_sink_takes_the_run_with_it() {
        let survivor = Spy::new();
        let error = play_into([
            (Spy::failing_after(2), Policy::Required),
            (survivor.clone(), Policy::Optional),
        ])
        .unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
        // The writer stopped where the required sink did, so the sink beside
        // it saw no more than the record that failed.
        assert_eq!(survivor.seen().len(), 3);
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
        assert_eq!(survivor.seen().len(), sample().len());
    }

    #[test]
    fn a_writer_with_no_sinks_drains_and_joins_cleanly() {
        let (sender, writer) = Writer::spawn::<TestPayload>(Vec::new());
        play(sender, writer).unwrap();
    }
}
