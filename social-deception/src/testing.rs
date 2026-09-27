//! Helpers shared by the crate's unit tests.

#[path = "../tests/support/temp.rs"]
mod temp;

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use serde::Serialize;
use serde_json::Value;

use crate::agent::Observation;
use crate::clock::Timestamp;
use crate::event::{AgentId, Domain, Event};
use crate::trajectory::{JsonLines, LogRecord, Policy, Sink, Writer};
use crate::werewolf::{
    Assignment, Faction, Knowledge, Message, Narration, Phase, Request, RequestId, RequestKind,
    Role, Round, WerewolfDomain,
};

pub(crate) use temp::TempDir;

/// The agent whose point of view a werewolf unit test takes.
pub(crate) const ME: &str = "me";

/// What a runtime unit test's agents say to each other: a counter, which is
/// enough to tell one message from the next.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) enum TestPayload {
    /// A step, carrying its number.
    Step(u64),
}

/// The domain the runtime's own unit tests are written against, standing in
/// for a game the runtime knows nothing about.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TestDomain;

impl Domain for TestDomain {
    type Payload = TestPayload;
    type Reward = i32;
}

/// A destination a test can read back after the writer has taken ownership
/// of it.
///
/// A sink owns its destination and the writer owns its sinks, so the bytes
/// a sink wrote are not handed back at the end. A test that wants them
/// keeps this, which is a [`Write`] whose bytes are shared, and reads
/// [`Shared::bytes`] once the writer has been joined.
#[derive(Debug, Clone, Default)]
pub(crate) struct Shared(Arc<Mutex<Vec<u8>>>);

impl Shared {
    /// An empty buffer.
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Everything written to it so far.
    ///
    /// # Panics
    ///
    /// If a writer thread panicked while holding the lock.
    pub(crate) fn bytes(&self) -> Vec<u8> {
        self.0.lock().unwrap().clone()
    }
}

impl Write for Shared {
    /// # Panics
    ///
    /// If a writer thread panicked while holding the lock.
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

/// A writer whose one required [`JsonLines`] sink writes to a buffer the
/// test keeps, which is what a test that reads its trajectory back wants.
pub(crate) fn recording<D: Domain>() -> (Sender<LogRecord<D>>, Writer, Shared) {
    let bytes = Shared::new();
    let sink: Box<dyn Sink<D>> = Box::new(JsonLines::new(bytes.clone()));
    let (sender, writer) = Writer::spawn(vec![(sink, Policy::Required)]);
    (sender, writer, bytes)
}

/// Joins `writer` and gives back everything its [`JsonLines`] sink wrote
/// to `bytes`.
///
/// # Panics
///
/// If the writer failed or its thread panicked.
pub(crate) fn joined(writer: Writer, bytes: &Shared) -> Vec<u8> {
    writer.join().unwrap();
    bytes.bytes()
}

/// Parses a trajectory file into one JSON value per line.
///
/// # Panics
///
/// If the bytes are not UTF-8, the text does not end with a newline, or any
/// line is not a JSON value.
pub(crate) fn parse_lines(bytes: &[u8]) -> Vec<Value> {
    let text = std::str::from_utf8(bytes).unwrap();
    assert!(text.ends_with('\n'), "file must end with a newline");
    text.lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect()
}

/// Serializes a value to a JSON value, for asserting on its shape.
///
/// # Panics
///
/// If the value cannot be serialized.
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap()
}

/// An agent id, for tests that name agents by string literal.
pub(crate) fn id(name: &str) -> AgentId {
    AgentId::new(name)
}

/// A set of agent ids, for tests that name agents by string literal.
pub(crate) fn ids<const N: usize>(names: [&str; N]) -> BTreeSet<AgentId> {
    names.map(AgentId::new).into()
}

/// The agent a point targets, for tests that name agents by string
/// literal. The same thing as [`id`], named for the place it is used: it
/// reads as "the target" where a point's target is what is meant.
pub(crate) fn target(name: &str) -> AgentId {
    id(name)
}

/// Timing fast enough that a test does not wait on a real clock, for the
/// unit tests that drive a game with explicit instants and never let a
/// wall-clock deadline fire at all.
pub(crate) fn fast() -> crate::werewolf::config::Timing {
    use std::time::Duration;

    use crate::werewolf::config::{DayTiming, NightTiming, Timing};

    // A night's limit has to outlast the slowest player's one point, or a
    // point misses its session and the game differs from run to run
    // (ADR-0011); a day's is what costs real time, since a random day
    // rarely reaches a majority and so usually runs it out. These unit
    // tests drive a game with explicit instants and never let a
    // wall-clock deadline fire, so neither number decides anything here —
    // they match the integration suite's so that the two agree.
    let night = NightTiming {
        quiet: Duration::from_millis(10),
        limit: Duration::from_millis(400),
    };
    Timing {
        day_cap: None,
        pack: night,
        seer: night,
        doctor: night,
        day: DayTiming {
            limit: Duration::from_millis(80),
        },
    }
}

/// A request of `kind`, for tests where its id and round do not matter.
pub(crate) fn request(kind: RequestKind) -> Request {
    Request {
        id: RequestId(1),
        round: Round(1),
        kind,
    }
}

/// An event from `sender` to [`ME`], created at a time no test reads.
///
/// Every werewolf unit test is a fold over what arrives, and the fold is a
/// pure function of the payloads; the instant each event was created plays
/// no part in it, so one stand-in time serves them all.
pub(crate) fn from(sender: &str, payload: Message) -> Event<WerewolfDomain> {
    Event::new(sender, [ME], Timestamp::default(), payload)
}

/// An event as [`ME`] observes it, received at a time no test reads. Like
/// the creation time in [`from`], it plays no part in any fold.
pub(crate) fn observed(event: Event<WerewolfDomain>) -> Observation<WerewolfDomain> {
    Observation {
        event,
        received: Timestamp::default(),
    }
}

/// A narration from the moderator to [`ME`].
pub(crate) fn narrated(narration: Narration) -> Event<WerewolfDomain> {
    from("moderator", Message::Narration(narration))
}

/// The moderator announcing a phase to [`ME`].
pub(crate) fn phase_began(
    round: u32,
    phase: Phase,
    living: BTreeSet<AgentId>,
) -> Event<WerewolfDomain> {
    narrated(Narration::PhaseBegan {
        round: Round(round),
        phase,
        living,
    })
}

/// The knowledge of [`ME`] playing `role`, with `others` and itself living
/// and nothing else known. The others need not be sorted.
pub(crate) fn knowing<const N: usize>(role: Role, others: [&str; N]) -> Knowledge {
    let mut knowledge = Knowledge::new(id(ME), role);
    knowledge.living = ids(others);
    knowledge.living.insert(id(ME));
    knowledge
}

/// The knowledge of [`ME`] as a werewolf among `others`, whose pack is
/// `pack` and itself.
pub(crate) fn werewolf_knowing<const N: usize, const P: usize>(
    others: [&str; N],
    pack: [&str; P],
) -> Knowledge {
    let mut knowledge = knowing(Role::Werewolf, others);
    knowledge.pack = ids(pack);
    knowledge.pack.insert(id(ME));
    knowledge
}

/// The knowledge of [`ME`] as the seer among `others`, having found each
/// of `investigated` to be a villager.
pub(crate) fn seer_knowing<const N: usize, const I: usize>(
    others: [&str; N],
    investigated: [&str; I],
) -> Knowledge {
    let mut knowledge = knowing(Role::Seer, others);
    knowledge.investigations = ids(investigated)
        .into_iter()
        .map(|who| (who, Faction::Village))
        .collect();
    knowledge
}

/// Five players and one werewolf: alice and erin are villagers, bob is the
/// werewolf, carol the seer and dave the doctor.
pub(crate) fn village() -> Assignment {
    Assignment::new([
        ("alice", Role::Villager),
        ("bob", Role::Werewolf),
        ("carol", Role::Seer),
        ("dave", Role::Doctor),
        ("erin", Role::Villager),
    ])
}

/// Seven players and two werewolves, bob and frank; carol is the seer and
/// dave the doctor.
pub(crate) fn town() -> Assignment {
    Assignment::new([
        ("alice", Role::Villager),
        ("bob", Role::Werewolf),
        ("carol", Role::Seer),
        ("dave", Role::Doctor),
        ("erin", Role::Villager),
        ("frank", Role::Werewolf),
        ("grace", Role::Villager),
    ])
}
