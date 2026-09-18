//! Trajectory records and the writer that puts them on disk.
//!
//! Each agent's own event sequence is its trajectory. The agent loop records
//! it as it folds and sends the records over an in-process channel to a
//! [`Writer`].
//!
//! # Two record types
//!
//! An agent handles a batch of events per pass, draining its whole inbox at
//! the start. Each pass produces:
//!
//! - an [`EventRecord`] per event: the agent it belongs to, the agent's
//!   sequence number for it, its time, and the event itself;
//! - a [`CycleRecord`] per pass: the handling window, the sequence numbers of
//!   the events that were in the drain, and the sequence numbers of whatever
//!   the fold emitted.
//!
//! Sequence numbers are per agent and cover everything that agent recorded,
//! inputs and outputs alike. An emitted message has an event record on the
//! sender's side and a separate one on each recipient's side when it arrives
//! there.
//!
//! # On-disk format
//!
//! One JSON object per line. Both record types are wrapped in [`LogRecord`],
//! whose `type` field is `"event"` or `"cycle"`. Nothing here reads a log
//! back; only `Serialize` is required of a payload.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::thread::{self, JoinHandle};

use crossbeam_channel::{Sender, unbounded};
use serde::Serialize;

use crate::clock::Timestamp;
use crate::event::{AgentId, Event, Payload};

/// An agent's sequence number for one of its own records.
///
/// Sequence numbers are assigned by the agent loop and are meaningful only
/// together with the agent id: two agents both have a sequence number 0.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Seq(pub u64);

/// One event in one agent's trajectory.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct EventRecord<P> {
    /// The agent whose trajectory this record belongs to.
    pub agent: AgentId,
    /// The agent's sequence number for this event.
    pub seq: Seq,
    /// When the event arrived on the agent's channel, or, for an event the
    /// agent itself emitted, when the loop handed it to the router.
    pub time: Timestamp,
    /// The event.
    pub event: Event<P>,
}

/// One pass of an agent's loop: drain the inbox, fold, send.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CycleRecord {
    /// The agent whose pass this was.
    pub agent: AgentId,
    /// When the agent began handling the batch.
    pub t_start: Timestamp,
    /// When the agent finished handling it, including sending its outputs.
    pub t_stop: Timestamp,
    /// The sequence numbers of the events that were in the drain, in the
    /// order the agent saw them.
    pub inputs: Vec<Seq>,
    /// The sequence numbers of the events the fold emitted, in the order they
    /// were sent.
    pub outputs: Vec<Seq>,
}

/// One line of a trajectory file.
///
/// Internally tagged: the `type` field of each line names the record kind.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LogRecord<P> {
    /// An event in some agent's trajectory.
    Event(EventRecord<P>),
    /// A pass of some agent's loop.
    Cycle(CycleRecord),
}

impl<P> From<EventRecord<P>> for LogRecord<P> {
    fn from(record: EventRecord<P>) -> Self {
        Self::Event(record)
    }
}

impl<P> From<CycleRecord> for LogRecord<P> {
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
    pub fn spawn<P: Payload>(sink: W) -> (Sender<LogRecord<P>>, Self) {
        let (sender, receiver) = unbounded::<LogRecord<P>>();
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
    pub fn create<P: Payload>(path: impl AsRef<Path>) -> io::Result<(Sender<LogRecord<P>>, Self)> {
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
    use crate::event::Control;

    #[derive(Debug, Clone, PartialEq, Eq, Serialize)]
    enum TestPayload {
        Step(u64),
    }

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// A handful of records of both kinds: agent `a` receives a start and a
    /// message in one pass and replies to `b`.
    fn sample() -> Vec<LogRecord<TestPayload>> {
        let a = AgentId::new("a");
        vec![
            EventRecord {
                agent: a.clone(),
                seq: Seq(0),
                time: at(10),
                event: Event::Control(Control::Start),
            }
            .into(),
            EventRecord {
                agent: a.clone(),
                seq: Seq(1),
                time: at(20),
                event: Event::message("b", ["a"], TestPayload::Step(6)),
            }
            .into(),
            EventRecord {
                agent: a.clone(),
                seq: Seq(2),
                time: at(40),
                event: Event::message("a", ["b"], TestPayload::Step(3)),
            }
            .into(),
            CycleRecord {
                agent: a,
                t_start: at(30),
                t_stop: at(50),
                inputs: vec![Seq(0), Seq(1)],
                outputs: vec![Seq(2)],
            }
            .into(),
        ]
    }

    fn expected_lines() -> Vec<Value> {
        vec![
            json!({"type": "event", "agent": "a", "seq": 0, "time": 10,
                   "event": {"kind": "control", "control": "start"}}),
            json!({"type": "event", "agent": "a", "seq": 1, "time": 20,
                   "event": {"kind": "message", "sender": "b", "recipients": ["a"],
                             "payload": {"Step": 6}}}),
            json!({"type": "event", "agent": "a", "seq": 2, "time": 40,
                   "event": {"kind": "message", "sender": "a", "recipients": ["b"],
                             "payload": {"Step": 3}}}),
            json!({"type": "cycle", "agent": "a", "t_start": 30, "t_stop": 50,
                   "inputs": [0, 1], "outputs": [2]}),
        ]
    }

    fn parse_lines(bytes: &[u8]) -> Vec<Value> {
        let text = std::str::from_utf8(bytes).unwrap();
        assert!(text.ends_with('\n'), "file must end with a newline");
        text.lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
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
        assert_eq!(types, ["event", "event", "event", "cycle"]);
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
                        let record = EventRecord {
                            agent: agent.clone(),
                            seq: Seq(seq),
                            time: at(seq),
                            event: Event::<TestPayload>::Think,
                        };
                        sender.send(record.into()).unwrap();
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
        let record: LogRecord<TestPayload> = CycleRecord {
            agent: AgentId::new("a"),
            t_start: at(0),
            t_stop: at(1),
            inputs: vec![Seq(0)],
            outputs: vec![],
        }
        .into();
        // A record larger than the buffer is written while the loop is still
        // running; the thread stops and drops its receiver at that point.
        let big: LogRecord<TestPayload> = EventRecord {
            agent: AgentId::new("a"),
            seq: Seq(0),
            time: at(0),
            event: Event::message(
                "a",
                (0..20_000)
                    .map(|i| format!("agent-{i}"))
                    .collect::<BTreeSet<_>>(),
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
        let late: LogRecord<TestPayload> = LogRecord::Event(EventRecord {
            agent: AgentId::new("a"),
            seq: Seq(1),
            time: at(2),
            event: Event::Think,
        });
        assert!(
            sender.send(late).is_err(),
            "sends after a write failure must fail"
        );
    }
}
