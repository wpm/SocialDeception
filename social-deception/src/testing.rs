//! Helpers shared by the crate's unit tests.

use std::collections::BTreeSet;
use std::fs;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;

use crate::event::{AgentId, Event};
use crate::werewolf::{
    Action, Faction, Knowledge, Message, Narration, Phase, Request, RequestId, RequestKind, Role,
    Round,
};

/// The agent whose point of view a werewolf unit test takes.
pub(crate) const ME: &str = "me";

/// A path of one test's own under the temp dir, with the given extension,
/// unique across tests and processes. The file, if the test made one, is
/// removed when this is dropped, so that a failing test leaves nothing
/// behind. Derefs to the [`Path`].
pub(crate) struct TempPath(PathBuf);

impl TempPath {
    pub(crate) fn new(extension: &str) -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "social-deception-{}-{}.{extension}",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )))
    }
}

impl Deref for TempPath {
    type Target = Path;

    fn deref(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
    }
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

/// An action targeting the named agent, for tests that name agents by
/// string literal.
pub(crate) fn target(name: &str) -> Action {
    Action::Target(id(name))
}

/// A request of `kind`, for tests where its id and round do not matter.
pub(crate) fn request(kind: RequestKind) -> Request {
    Request {
        id: RequestId(1),
        round: Round(1),
        kind,
    }
}

/// A narration from the moderator to [`ME`], as it arrives on the receiver.
pub(crate) fn narrated(narration: Narration) -> Event<Message> {
    Event::message("moderator", [ME], Message::Narration(narration))
}

/// The moderator announcing a phase to [`ME`].
pub(crate) fn phase_began(round: u32, phase: Phase, living: BTreeSet<AgentId>) -> Event<Message> {
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
