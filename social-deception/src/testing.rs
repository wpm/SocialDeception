//! Helpers shared by the crate's unit tests.

#[path = "../tests/support/temp.rs"]
mod temp;

use std::collections::BTreeSet;

use serde::Serialize;
use serde_json::Value;

use crate::agent::Observation;
use crate::clock::Timestamp;
use crate::event::{AgentId, Domain, Event};
use crate::werewolf::{
    Faction, Knowledge, Message, Move, Narration, Phase, Request, RequestId, RequestKind, Role,
    Round, WerewolfDomain,
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

/// A move targeting the named agent, for tests that name agents by
/// string literal.
pub(crate) fn target(name: &str) -> Move {
    Move::Target(id(name))
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
