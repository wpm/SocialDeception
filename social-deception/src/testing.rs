//! Helpers shared by the crate's unit tests.

#[path = "../tests/support/temp.rs"]
mod temp;

use std::collections::BTreeSet;
use std::io::{self, Write};
use std::sync::{Arc, Mutex};

use crossbeam_channel::Sender;
use serde::Serialize;
use serde_json::Value;

use std::sync::LazyLock;
use std::time::{Duration, Instant};

use crate::agent::Observation;
use crate::clock::Clock;
use crate::log::{JsonLines, Policy, Record, Sink, Writer};
use crate::message::{ActorId, Message, Payload};
use crate::werewolf::{self, Assignment, Faction, Knowledge, Narration, Phase, Role, Round};

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
/// test keeps, which is what a test that reads its log back wants.
pub(crate) fn recording<P: Payload>(clock: Clock) -> (Sender<Record<P>>, Writer, Shared) {
    let bytes = Shared::new();
    let sink: Box<dyn Sink<P>> = Box::new(JsonLines::new(bytes.clone()));
    let (sender, writer) = Writer::spawn(vec![(sink, Policy::Required)], clock);
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

/// Parses a log file into one JSON value per line.
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

/// Asserts that `line` is the `episode` header a log opens with, and returns
/// the wall-clock anchor it carries.
///
/// The anchor is the one wall-clock time in a log and so differs on every
/// run; a caller that wants to check *which* moment it is compares it with
/// the [`Clock`] the log was written from.
///
/// # Panics
///
/// If the line is not a header, or carries no anchor.
pub(crate) fn header_anchor(line: &Value) -> u64 {
    assert_eq!(
        line["type"], "episode",
        "the first line is the header: {line}"
    );
    assert!(
        line["agent"].is_null(),
        "the header is nobody's record: {line}"
    );
    line["start_unix_ns"]
        .as_u64()
        .unwrap_or_else(|| panic!("the header anchors the episode to the wall clock: {line}"))
}

/// A fixed base for the instants a test names, so that `at(0)`, `at(1)` and
/// so on are one comparable series.
///
/// An `Instant` has no epoch to build one from, so a test that wants
/// arbitrary times offsets one reading taken once for the whole run. It is
/// far enough in the future that a deadline named from it never fires on a
/// real clock; a test that wants one to fire uses a
/// [`ManualTimer`](crate::ManualTimer).
pub(crate) static BASE: LazyLock<Instant> =
    LazyLock::new(|| Instant::now() + Duration::from_secs(3600));

/// `nanos` after [`BASE`].
pub(crate) fn at_nanos(nanos: u64) -> Instant {
    *BASE + Duration::from_nanos(nanos)
}

/// `millis` after [`BASE`].
pub(crate) fn at_millis(millis: u64) -> Instant {
    *BASE + Duration::from_millis(millis)
}

/// Serializes a value to a JSON value, for asserting on its shape.
///
/// # Panics
///
/// If the value cannot be serialized.
pub(crate) fn json<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).unwrap()
}

/// An actor id, for tests that name agents by string literal.
pub(crate) fn id(name: &str) -> ActorId {
    ActorId::new(name)
}

/// A set of actor ids, for tests that name agents by string literal.
pub(crate) fn ids<const N: usize>(names: [&str; N]) -> BTreeSet<ActorId> {
    names.map(ActorId::new).into()
}

/// The agent a selection targets, for tests that name agents by string
/// literal. The same thing as [`id`], named for the place it is used: it
/// reads as "the target" where a selection's target is what is meant.
pub(crate) fn target(name: &str) -> ActorId {
    id(name)
}

/// Timing fast enough that a test does not wait on a real clock, for the
/// unit tests that drive a game with explicit instants and never let a
/// wall-clock deadline fire at all.
pub(crate) fn fast() -> crate::werewolf::config::Timing {
    use std::time::Duration;

    use crate::werewolf::config::{DayTiming, NightTiming, Timing};

    // A night's limit has to outlast the slowest player's one selection, or a
    // selection misses its session and the game differs from run to run
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

/// A message from `sender` to [`ME`], numbered as no test reads.
///
/// Every werewolf unit test is a fold over what arrives, and the fold is a
/// pure function of the payloads; which of its sender's messages each one is
/// plays no part in it, so one stand-in number serves them all.
pub(crate) fn from(sender: &str, payload: werewolf::Message) -> Message<werewolf::Message> {
    Message::new(sender, [ME], 0, payload)
}

/// A message as [`ME`] observes it, arriving at a time no test reads. Like
/// the sequence number in [`from`], it plays no part in any fold.
pub(crate) fn observed(message: Message<werewolf::Message>) -> Observation<werewolf::Message> {
    Observation {
        message,
        at: Instant::now(),
    }
}

/// A narration from the moderator to [`ME`].
pub(crate) fn narrated(narration: Narration) -> Message<werewolf::Message> {
    from("moderator", werewolf::Message::Narration(narration))
}

/// The moderator announcing a phase to [`ME`].
pub(crate) fn phase_began(
    round: u32,
    phase: Phase,
    living: BTreeSet<ActorId>,
) -> Message<werewolf::Message> {
    narrated(Narration::PhaseBegan {
        round: Round::new(round),
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
