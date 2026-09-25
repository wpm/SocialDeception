//! Trajectory records and the writer that puts them on disk.
//!
//! Each agent's own record sequence is its trajectory. The agent loop
//! records it as it folds and sends the records over an in-process channel
//! to a [`Writer`].
//!
//! # Five record types
//!
//! An agent runs one cycle per wake-up: it pops the controls at the head of
//! its queue and at most one event, hands that one observation to its
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
//! agent that wrote it; see [`RewardRecord`] and ADR-0007.
//!
//! An observation and an action record the same event from the two sides of
//! it, which is what makes the trajectory joinable: an observation in one
//! agent's trajectory matches the action in its sender's whose `created` and
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
//! A reward record is outside all of it. It belongs to the agent it names
//! and was written by another, so it is in no cycle of that agent's, sits
//! between two of its cycles wherever the writer happened to take it, and
//! carries no sequence number to place it. Its `created` is what places it.
//!
//! If the writer fails mid-cycle, the agent's loop exits with an error and
//! its records end without a closing cycle record.
//!
//! # On-disk format
//!
//! One JSON object per line, wrapped in [`LogRecord`], whose `type` field is
//! `observation`, `action`, `control`, `reward` or `cycle`.
//! Nothing here reads a log back; only `Serialize` is required of a payload
//! or a reward.
//!
//! ```json
//! {"type":"control","agent":"alice","seq":0,"created":10,"received":12,"control":"start"}
//! {"type":"cycle","agent":"alice","t_start":12,"t_stop":13,"woken":"queue","inputs":[0],"outputs":[]}
//! {"type":"observation","agent":"alice","seq":1,"created":40,"received":55,"event":{"sender":"moderator","recipients":["alice"],"payload":{"Request":{}}}}
//! {"type":"action","agent":"alice","seq":2,"created":90,"event":{"sender":"alice","recipients":["moderator"],"payload":{"Response":{}}}}
//! {"type":"cycle","agent":"alice","t_start":55,"t_stop":90,"woken":"queue","inputs":[1],"outputs":[2]}
//! {"type":"reward","agent":"alice","created":500,"value":1}
//! ```

use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Sender, unbounded};
use serde::{Serialize, Serializer};

use crate::clock::{Created, Timestamp};
use crate::event::{AgentId, Control, Domain, Event};

/// An agent's sequence number for one of its own records.
///
/// Sequence numbers are assigned by the agent loop and are meaningful only
/// together with the agent id: two agents both have a sequence number 0.
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
    /// that passes while an event is waiting joins that event's cycle,
    /// which observes it and calls
    /// [`Handler::handle`](crate::Handler::handle) like any other. Either
    /// way the deadline is retired and the next is measured from this
    /// cycle, which is what the mark is for.
    Timeout,
}

/// The body of an `event` field: an event without its creation time, which
/// the record carries at the top level instead.
///
/// It is not a type of its own anywhere else. The creation time sits beside
/// `agent` and `seq` because it is a property of the record's subject and a
/// reader joins on it, and repeating it inside the event would be two places
/// to read the same instant from.
struct Envelope<'a, D: Domain>(&'a Event<D>);

impl<D: Domain> Serialize for Envelope<'_, D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut event = serializer.serialize_struct("Event", 3)?;
        event.serialize_field("sender", &self.0.sender)?;
        event.serialize_field("recipients", &self.0.recipients)?;
        event.serialize_field("payload", &self.0.payload)?;
        event.end()
    }
}

/// An event this agent popped off its queue.
///
/// `Debug`, `Clone` and equality are written out rather than derived, for
/// the reason [`Event`]'s are: a derive would ask them of `D`.
pub struct ObservationRecord<D: Domain> {
    /// The agent whose trajectory this record belongs to.
    pub agent: AgentId,
    /// The agent's sequence number for it.
    pub seq: Seq,
    /// When its sender sent it.
    pub created: Timestamp,
    /// When this agent popped it: its cycle's `t_start`.
    pub received: Timestamp,
    /// The event.
    pub event: Event<D>,
}

impl<D: Domain> Serialize for ObservationRecord<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ObservationRecord", 5)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("seq", &self.seq)?;
        record.serialize_field("created", &self.created)?;
        record.serialize_field("received", &self.received)?;
        record.serialize_field("event", &Envelope(&self.event))?;
        record.end()
    }
}

/// An event this agent sent.
///
/// `Debug`, `Clone` and equality are written out for the same reason
/// [`ObservationRecord`]'s are.
///
/// There is no `received`: an action is logged by its sender, which knows
/// only when it sent it. When each recipient received it is in that
/// recipient's own observation record.
pub struct ActionRecord<D: Domain> {
    /// The agent whose trajectory this record belongs to.
    pub agent: AgentId,
    /// The agent's sequence number for it.
    pub seq: Seq,
    /// When the loop sent it, which is the `created` on the wire.
    pub created: Timestamp,
    /// The event as sent.
    pub event: Event<D>,
}

impl<D: Domain> Serialize for ActionRecord<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("ActionRecord", 4)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("seq", &self.seq)?;
        record.serialize_field("created", &self.created)?;
        record.serialize_field("event", &Envelope(&self.event))?;
        record.end()
    }
}

/// A control this agent popped off its queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ControlRecord {
    /// The agent whose trajectory this record belongs to.
    pub agent: AgentId,
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
/// names the agent **rewarded**, whose trajectory the record belongs to,
/// not the environment that wrote it. Training joins a reward to that
/// agent's trajectory by the agent id and the time (ADR-0007).
///
/// There is no `seq`. Sequence numbers are the agent loop's to assign, and
/// this record was not written by that loop, so numbering it would either
/// invent a number nobody issued or perturb the numbering of the records
/// the agent did write. There is no `received` either: a reward is logged,
/// never sent, so nobody ever receives it, and it implements
/// [`Created`] alone.
///
/// `Debug`, `Clone` and equality are written out rather than derived, for
/// the reason [`ObservationRecord`]'s are: a derive would ask them of `D`,
/// the marker type, when what has to have them is `D::Reward`.
pub struct RewardRecord<D: Domain> {
    /// The agent rewarded, whose trajectory this record belongs to.
    pub agent: AgentId,
    /// When the environment logged it.
    pub created: Timestamp,
    /// What the agent's behavior was worth, in the game's own units.
    pub value: D::Reward,
}

impl<D: Domain> Serialize for RewardRecord<D> {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut record = serializer.serialize_struct("RewardRecord", 3)?;
        record.serialize_field("agent", &self.agent)?;
        record.serialize_field("created", &self.created)?;
        record.serialize_field("value", &self.value)?;
        record.end()
    }
}

impl<D: Domain> Created for RewardRecord<D> {
    fn created(&self) -> Timestamp {
        self.created
    }
}

/// One cycle of an agent's loop: pop, hand to the handler, send.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CycleRecord {
    /// The agent whose cycle this was.
    pub agent: AgentId,
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

/// One line of a trajectory file.
///
/// Internally tagged: the `type` field of each line names the record kind.
/// `Debug`, `Clone` and equality are written out for the same reason
/// [`ObservationRecord`]'s are.
#[derive(Serialize)]
#[serde(tag = "type", rename_all = "snake_case", bound = "")]
pub enum LogRecord<D: Domain> {
    /// An event some agent popped.
    Observation(ObservationRecord<D>),
    /// An event some agent sent.
    Action(ActionRecord<D>),
    /// A control some agent popped.
    Control(ControlRecord),
    /// A reward the environment assigned to some agent.
    Reward(RewardRecord<D>),
    /// A cycle of some agent's loop.
    Cycle(CycleRecord),
}

impl<D: Domain> fmt::Debug for ObservationRecord<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ObservationRecord")
            .field("agent", &self.agent)
            .field("seq", &self.seq)
            .field("created", &self.created)
            .field("received", &self.received)
            .field("event", &self.event)
            .finish()
    }
}

impl<D: Domain> Clone for ObservationRecord<D> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            seq: self.seq,
            created: self.created,
            received: self.received,
            event: self.event.clone(),
        }
    }
}

impl<D: Domain> PartialEq for ObservationRecord<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.agent == other.agent
            && self.seq == other.seq
            && self.created == other.created
            && self.received == other.received
            && self.event == other.event
    }
}

impl<D: Domain> Eq for ObservationRecord<D> where D::Payload: Eq {}

impl<D: Domain> fmt::Debug for ActionRecord<D>
where
    D::Payload: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ActionRecord")
            .field("agent", &self.agent)
            .field("seq", &self.seq)
            .field("created", &self.created)
            .field("event", &self.event)
            .finish()
    }
}

impl<D: Domain> Clone for ActionRecord<D> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            seq: self.seq,
            created: self.created,
            event: self.event.clone(),
        }
    }
}

impl<D: Domain> PartialEq for ActionRecord<D>
where
    D::Payload: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.agent == other.agent
            && self.seq == other.seq
            && self.created == other.created
            && self.event == other.event
    }
}

impl<D: Domain> Eq for ActionRecord<D> where D::Payload: Eq {}

impl<D: Domain> fmt::Debug for RewardRecord<D>
where
    D::Reward: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RewardRecord")
            .field("agent", &self.agent)
            .field("created", &self.created)
            .field("value", &self.value)
            .finish()
    }
}

impl<D: Domain> Clone for RewardRecord<D> {
    fn clone(&self) -> Self {
        Self {
            agent: self.agent.clone(),
            created: self.created,
            value: self.value,
        }
    }
}

impl<D: Domain> PartialEq for RewardRecord<D>
where
    D::Reward: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.agent == other.agent && self.created == other.created && self.value == other.value
    }
}

impl<D: Domain> Eq for RewardRecord<D> where D::Reward: Eq {}

impl<D: Domain> fmt::Debug for LogRecord<D>
where
    D::Payload: fmt::Debug,
    D::Reward: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Observation(record) => f.debug_tuple("Observation").field(record).finish(),
            Self::Action(record) => f.debug_tuple("Action").field(record).finish(),
            Self::Control(record) => f.debug_tuple("Control").field(record).finish(),
            Self::Reward(record) => f.debug_tuple("Reward").field(record).finish(),
            Self::Cycle(record) => f.debug_tuple("Cycle").field(record).finish(),
        }
    }
}

impl<D: Domain> Clone for LogRecord<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Observation(record) => Self::Observation(record.clone()),
            Self::Action(record) => Self::Action(record.clone()),
            Self::Control(record) => Self::Control(record.clone()),
            Self::Reward(record) => Self::Reward(record.clone()),
            Self::Cycle(record) => Self::Cycle(record.clone()),
        }
    }
}

impl<D: Domain> PartialEq for LogRecord<D>
where
    D::Payload: PartialEq,
    D::Reward: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Observation(a), Self::Observation(b)) => a == b,
            (Self::Action(a), Self::Action(b)) => a == b,
            (Self::Control(a), Self::Control(b)) => a == b,
            (Self::Reward(a), Self::Reward(b)) => a == b,
            (Self::Cycle(a), Self::Cycle(b)) => a == b,
            _ => false,
        }
    }
}

impl<D: Domain> Eq for LogRecord<D>
where
    D::Payload: Eq,
    D::Reward: Eq,
{
}

impl<D: Domain> From<ObservationRecord<D>> for LogRecord<D> {
    fn from(record: ObservationRecord<D>) -> Self {
        Self::Observation(record)
    }
}

impl<D: Domain> From<ActionRecord<D>> for LogRecord<D> {
    fn from(record: ActionRecord<D>) -> Self {
        Self::Action(record)
    }
}

impl<D: Domain> From<ControlRecord> for LogRecord<D> {
    fn from(record: ControlRecord) -> Self {
        Self::Control(record)
    }
}

impl<D: Domain> From<RewardRecord<D>> for LogRecord<D> {
    fn from(record: RewardRecord<D>) -> Self {
        Self::Reward(record)
    }
}

impl<D: Domain> From<CycleRecord> for LogRecord<D> {
    fn from(record: CycleRecord) -> Self {
        Self::Cycle(record)
    }
}

/// The trajectory writer: a thread that owns the output and turns the records
/// it receives into lines of JSON.
///
/// Records reach it over an in-process channel whose sender is handed out by
/// [`Writer::spawn`]; every agent gets a clone. The thread runs until every
/// sender has been dropped, then flushes and exits. [`Writer::join`] waits for
/// that and reports how it went.
///
/// On a write failure the thread stops and drops its receiver, so every
/// subsequent `send` fails at the agent that attempted it, and the failure is
/// returned from [`Writer::join`].
#[derive(Debug)]
pub struct Writer<W> {
    thread: JoinHandle<io::Result<W>>,
}

impl<W: Write + Send + 'static> Writer<W> {
    /// Starts a writer thread that writes records to `sink`, one JSON object
    /// per line, and returns the sender that feeds it.
    ///
    /// The sink is buffered internally; there is no need to wrap it in a
    /// [`BufWriter`] first.
    pub fn spawn<D: Domain>(sink: W) -> (Sender<LogRecord<D>>, Self) {
        let (sender, receiver) = unbounded::<LogRecord<D>>();
        let thread = thread::spawn(move || {
            let mut out = BufWriter::new(sink);
            for record in receiver {
                serde_json::to_writer(&mut out, &record)?;
                out.write_all(b"\n")?;
            }
            out.into_inner().map_err(io::IntoInnerError::into_error)
        });
        (sender, Self { thread })
    }

    /// Waits for the writer to finish and gives back its sink.
    ///
    /// The writer finishes when every sender returned by [`Writer::spawn`]
    /// has been dropped, or earlier if a write failed. Drop the senders before
    /// calling this, or it never returns.
    ///
    /// # Errors
    ///
    /// The first write or flush error the thread met, or an error of kind
    /// [`io::ErrorKind::Other`] if the thread panicked.
    pub fn join(self) -> io::Result<W> {
        self.thread
            .join()
            .map_err(|_| io::Error::other("trajectory writer thread panicked"))?
    }
}

impl Writer<File> {
    /// Creates (or truncates) the file at `path` and starts a writer on it.
    ///
    /// # Errors
    ///
    /// Whatever [`File::create`] returns.
    pub fn create<D: Domain>(path: impl AsRef<Path>) -> io::Result<(Sender<LogRecord<D>>, Self)> {
        Ok(Self::spawn(File::create(path)?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use serde_json::{Value, json};

    use super::*;
    use crate::testing::{TestDomain, TestPayload, parse_lines};

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// A handful of records of every kind: agent `a` pops a start and a
    /// message in one cycle and replies to `b`.
    fn sample() -> Vec<LogRecord<TestDomain>> {
        let a = AgentId::new("a");
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
                event: Event::new("b", ["a"], at(20), TestPayload::Step(6)),
            }
            .into(),
            ActionRecord {
                agent: a.clone(),
                seq: Seq(2),
                created: at(40),
                event: Event::new("a", ["b"], at(40), TestPayload::Step(3)),
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
                   "event": {"sender": "b", "recipients": ["a"], "payload": {"Step": 6}}}),
            json!({"type": "action", "agent": "a", "seq": 2, "created": 40,
                   "event": {"sender": "a", "recipients": ["b"], "payload": {"Step": 3}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50, "woken": "queue",
                   "inputs": [0, 1], "outputs": [2]}),
        ]
    }

    #[test]
    fn records_round_trip_through_the_writer_as_jsonl() {
        let (sender, writer) = Writer::spawn(Vec::new());
        for record in sample() {
            sender.send(record).unwrap();
        }
        drop(sender);
        let bytes = writer.join().unwrap();
        assert_eq!(parse_lines(&bytes), expected_lines());
    }

    #[test]
    fn an_event_record_carries_its_creation_time_once_and_at_the_top() {
        // The event body is the sender, the recipients and the payload; the
        // instant it was created sits beside `seq`, where a reader joins on
        // it, and nowhere else.
        let lines = expected_lines();
        for line in &lines[1..3] {
            assert!(line["created"].is_u64(), "{line}");
            assert!(line["event"]["created"].is_null(), "{line}");
        }
        // And in the order written, which is what a reader sees on disk.
        let written = serde_json::to_string(&sample()[1]).unwrap();
        assert!(
            written.ends_with(r#""event":{"sender":"b","recipients":["a"],"payload":{"Step":6}}}"#),
            "{written}"
        );
    }

    #[test]
    fn a_reward_names_the_agent_rewarded_and_carries_no_sequence_number() {
        // Three fields and no more: the agent whose trajectory it belongs
        // to, when the environment logged it, and what it is worth. No
        // `seq`, because the agent loop did not write it, and no
        // `received`, because nobody received it.
        let reward: LogRecord<TestDomain> = RewardRecord {
            agent: AgentId::new("alice"),
            created: at(500),
            value: 1,
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
        // `Created` and not `Timestamped`: a reward is logged, never sent,
        // so there is no instant at which anybody got it.
        let reward: RewardRecord<TestDomain> = RewardRecord {
            agent: AgentId::new("alice"),
            created: at(500),
            value: -1,
        };
        assert_eq!(reward.created(), at(500));
        assert_eq!(reward, reward.clone());
        assert_ne!(
            reward,
            RewardRecord {
                value: 1,
                ..reward.clone()
            }
        );
        assert!(format!("{reward:?}").starts_with("RewardRecord"));
        let line: LogRecord<TestDomain> = reward.into();
        assert!(format!("{line:?}").starts_with("Reward"));
        assert_eq!(line, line.clone());
    }

    #[test]
    fn a_timeout_cycle_records_what_woke_it_and_no_inputs() {
        let cycle: LogRecord<TestDomain> = CycleRecord {
            agent: AgentId::new("a"),
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
        let (sender, writer) = Writer::spawn(Vec::new());
        for record in sample() {
            sender.send(record).unwrap();
        }
        drop(sender);
        let bytes = writer.join().unwrap();
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
        let (sender, writer) = Writer::spawn(Vec::new());
        let handles: Vec<_> = (0..4u64)
            .map(|i| {
                let sender = sender.clone();
                thread::spawn(move || {
                    let agent = AgentId::new(format!("agent-{i}"));
                    for seq in 0..10 {
                        let record: LogRecord<TestDomain> = ControlRecord {
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
        let bytes = writer.join().unwrap();
        let lines = parse_lines(&bytes);
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

    #[test]
    fn a_write_failure_is_loud_at_both_ends() {
        let (sender, writer) = Writer::spawn(BrokenSink);
        // A record small enough to sit in the buffer.
        let record: LogRecord<TestDomain> = CycleRecord {
            agent: AgentId::new("a"),
            t_start: at(0),
            t_stop: at(1),
            woken: Woken::Queue,
            inputs: vec![Seq(0)],
            outputs: vec![],
        }
        .into();
        // A record larger than the buffer is written while the loop is still
        // running; the thread stops and drops its receiver at that point.
        let big: LogRecord<TestDomain> = ActionRecord {
            agent: AgentId::new("a"),
            seq: Seq(0),
            created: at(0),
            event: Event::new(
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
        let late: LogRecord<TestDomain> = ControlRecord {
            agent: AgentId::new("a"),
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
}
