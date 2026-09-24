//! Reading a trajectory back as the game it records, and rendering that game
//! for a person to read.
//!
//! A [`Transcript`] is the logical game: who held which role, what everyone
//! did in each phase, who was eliminated and how, and how it ended. It has no
//! timestamps and no record interleaving, because those are the two things
//! about a trajectory file that are not reproducible: the agents are threads,
//! so the same game written twice differs in when each record was stamped
//! and in how different agents' records happen to be ordered in the file. A
//! transcript is the file with everything non-reproducible projected out, so
//! that equality of two transcripts is exactly the claim that the same seed
//! produced the same game.
//!
//! It contains no seed and no configuration either. Those are provenance,
//! not game: the same logical game is the same transcript however it was
//! produced, and a determinism comparison must not be able to pass or fail
//! on anything but the game itself. The seed lives in the effective
//! configuration written beside the trajectory (see
//! [`config::effective_path`](super::config::effective_path)), and whoever
//! prints a transcript prints the seed from there.
//!
//! # Only the moderator's records
//!
//! [`Transcript::read`] reads the moderator's records and nobody else's. The
//! moderator is the authoritative view: its `action` records are every
//! narration and every request it sent, and its `observation` records are
//! every response it received, each naming the responding player as its
//! sender. Which record type a line is *is* the direction, so the reader
//! needs no direction of its own. Reassembling the game from the players'
//! records would mean recovering hidden information from partial views,
//! which is the thing the design prevents. Within one agent's records the
//! runtime guarantees that sequence numbers are contiguous and increasing in
//! file order, so the moderator's records in file order are the game in
//! order, whatever the other agents' records do around them.
//!
//! # No game logic
//!
//! Nothing here decides anything. The reader records what the moderator
//! said, in the order it said it, and nothing more: it does not tally votes,
//! check a win condition or infer a save. If reconstructing the game ever
//! needed a rule, the moderator would not be recording enough, and the fix
//! would belong with the moderator.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::message::{
    Cause, Message, Move, Narration, Outcome, Phase, Request, RequestId, RequestKind, Response,
    Round,
};
use super::role::{Faction, Role};
use crate::event::AgentId;

/// A game of Werewolf as its moderator recorded it: everything that is
/// reproducible about a run, and nothing that is not.
///
/// See the [module documentation](self) for what is left out and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Each player's role, as the moderator dealt it.
    pub assignment: BTreeMap<AgentId, Role>,
    /// Every round, in order.
    pub rounds: Vec<RoundRecord>,
    /// How the game ended.
    pub outcome: Outcome,
}

/// One round: a night, then a day unless the game ended at night.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoundRecord {
    /// Which round this is.
    pub round: Round,
    /// The night.
    pub night: PhaseRecord,
    /// The day, or `None` if the game ended at night.
    pub day: Option<PhaseRecord>,
}

/// What happened in one phase.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseRecord {
    /// Everyone in the game when the phase began.
    pub living: BTreeSet<AgentId>,
    /// Every move made this phase, by the agent that made it, paired
    /// with the kind of request it answered. The move alone does not say
    /// whether a target was devoured, protected, investigated or nominated.
    pub moves: BTreeMap<AgentId, (RequestKind, Move)>,
    /// The seer's finding: the seer, whom it investigated, and what it
    /// learned. `None` when there is no living seer or it abstained.
    pub investigation: Option<(AgentId, AgentId, Faction)>,
    /// Who was eliminated, the role their death revealed, and how. `None`
    /// on a night when nobody died.
    pub eliminated: Option<(AgentId, Role, Cause)>,
}

impl PhaseRecord {
    /// A phase in which nothing has happened yet.
    fn begun(living: BTreeSet<AgentId>) -> Self {
        Self {
            living,
            moves: BTreeMap::new(),
            investigation: None,
            eliminated: None,
        }
    }
}

/// Why a trajectory could not be read as a game.
///
/// Every variant names the line it was found on, counting from one, so the
/// message points at the record and not just at the problem.
#[derive(Debug)]
pub enum TranscriptError {
    /// A line is not JSON.
    NotJson {
        /// The line.
        line: usize,
        /// What the parser objected to.
        source: serde_json::Error,
    },
    /// A line is JSON, but not an object.
    NotAnObject {
        /// The line.
        line: usize,
    },
    /// A record's `type` is not one the runtime writes.
    UnknownRecordType {
        /// The line.
        line: usize,
        /// The `type` found, or `None` if the record has none.
        found: Option<String>,
    },
    /// A record's envelope lacks something every record of its kind has.
    Malformed {
        /// The line.
        line: usize,
        /// What is missing or wrong.
        what: String,
    },
    /// A message's payload is not a Werewolf [`Message`].
    Payload {
        /// The line.
        line: usize,
        /// What the deserializer objected to.
        source: serde_json::Error,
    },
    /// The moderator's sequence numbers are not contiguous.
    SeqGap {
        /// The line.
        line: usize,
        /// The sequence number that should have come next.
        expected: u64,
        /// The one found.
        found: u64,
    },
    /// A response answers a request the moderator did not send, has already
    /// had answered, or sent to somebody else.
    UnknownRequest {
        /// The line.
        line: usize,
        /// The responding agent.
        from: AgentId,
        /// The request it named.
        request: RequestId,
    },
    /// A record belongs to a phase that has not begun: a response, an
    /// investigation or an elimination before the first night, or a day
    /// whose night has not begun or that has begun already.
    NoPhase {
        /// The line.
        line: usize,
    },
    /// A message the moderator never records: a narration or a request
    /// observed by it, or a response it took as an action.
    Misdirected {
        /// The line.
        line: usize,
    },
    /// The moderator never announced an outcome, so the game is truncated.
    NoOutcome,
}

impl fmt::Display for TranscriptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotJson { line, source } => write!(f, "line {line} is not JSON: {source}"),
            Self::NotAnObject { line } => write!(f, "line {line} is not a JSON object"),
            Self::UnknownRecordType {
                line,
                found: Some(found),
            } => write!(f, "line {line} has unknown record type {found:?}"),
            Self::UnknownRecordType { line, found: None } => {
                write!(f, "line {line} has no record type")
            }
            Self::Malformed { line, what } => write!(f, "line {line}: {what}"),
            Self::Payload { line, source } => {
                write!(f, "line {line} does not carry a Werewolf message: {source}")
            }
            Self::SeqGap {
                line,
                expected,
                found,
            } => write!(
                f,
                "line {line} has sequence number {found} where {expected} was expected"
            ),
            Self::UnknownRequest {
                line,
                from,
                request: RequestId(id),
            } => write!(
                f,
                "line {line}: {from} answered request {id}, which was not asked of it"
            ),
            Self::NoPhase { line } => {
                write!(f, "line {line} belongs to a phase, but none has begun")
            }
            Self::Misdirected { line } => {
                write!(f, "line {line} is a message the moderator never records")
            }
            Self::NoOutcome => f.write_str("the moderator never announced an outcome"),
        }
    }
}

impl Error for TranscriptError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::NotJson { source, .. } | Self::Payload { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Parses the text of a trajectory file into one JSON value per line, the
/// form [`Transcript::read`] takes.
///
/// # Errors
///
/// [`TranscriptError::NotJson`] on the first line that is not JSON.
pub fn lines(text: &str) -> Result<Vec<Value>, TranscriptError> {
    text.lines()
        .enumerate()
        .map(|(index, line)| {
            serde_json::from_str(line).map_err(|source| TranscriptError::NotJson {
                line: index + 1,
                source,
            })
        })
        .collect()
}

/// Which way a message crossed the moderator's boundary, which is which of
/// the two record types it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Direction {
    /// An `action` record: something the moderator sent.
    Sent,
    /// An `observation` record: something the moderator received.
    Received,
}

/// One record of the moderator's, still as JSON.
///
/// `direction` is `None` for a control, which carries a sequence number and
/// so must be counted, but says nothing about the game and is not decoded.
struct Line<'a> {
    record: &'a Map<String, Value>,
    direction: Option<Direction>,
}

/// One message record of the moderator's, with its envelope decoded.
struct Record {
    line: usize,
    direction: Direction,
    sender: AgentId,
    recipients: BTreeSet<AgentId>,
    message: Message,
}

impl Transcript {
    /// Reconstructs the game from the moderator's records in a trajectory.
    ///
    /// `lines` is the whole file, one value per line, as [`lines`] returns
    /// it. Records of every other agent are ignored, but every line is
    /// checked to be a record.
    ///
    /// # Errors
    ///
    /// A [`TranscriptError`] naming the first line that keeps the file from
    /// being a game, or [`TranscriptError::NoOutcome`] if the moderator's
    /// records end without one.
    pub fn read(lines: &[Value], moderator: &AgentId) -> Result<Self, TranscriptError> {
        let mut reader = Reader::default();
        let mut next_seq = 0;
        for (index, value) in lines.iter().enumerate() {
            let line = index + 1;
            let Some(Line { record, direction }) = moderator_record(value, line, moderator)? else {
                continue;
            };
            // The moderator's controls are counted but not read: they carry
            // a sequence number, so skipping them without counting would
            // look like a gap, and they say nothing about the game.
            let seq = integer(record, "seq", line)?;
            if seq != next_seq {
                return Err(TranscriptError::SeqGap {
                    line,
                    expected: next_seq,
                    found: seq,
                });
            }
            next_seq += 1;
            if let Some(direction) = direction {
                reader.fold(message(record, direction, line)?)?;
            }
        }
        let outcome = reader.outcome.ok_or(TranscriptError::NoOutcome)?;
        Ok(Self {
            assignment: reader.assignment,
            rounds: reader.rounds,
            outcome,
        })
    }
}

/// One of the moderator's records, and which way its event went; `None` for
/// a cycle record or another agent's, and an error for something that is not
/// a record at all.
///
/// The direction is `None` for the moderator's own controls. They say
/// nothing about the game, but they carry sequence numbers, so the caller
/// must count them or the numbers look full of gaps.
fn moderator_record<'a>(
    value: &'a Value,
    line: usize,
    moderator: &AgentId,
) -> Result<Option<Line<'a>>, TranscriptError> {
    let record = value
        .as_object()
        .ok_or(TranscriptError::NotAnObject { line })?;
    let direction = match record.get("type").and_then(Value::as_str) {
        Some("action") => Some(Direction::Sent),
        Some("observation") => Some(Direction::Received),
        Some("control") => None,
        Some("cycle") => return Ok(None),
        found => {
            return Err(TranscriptError::UnknownRecordType {
                line,
                found: found.map(str::to_owned),
            });
        }
    };
    let agent = string(record, "agent", line)?;
    Ok((agent == moderator.as_str()).then_some(Line { record, direction }))
}

/// Decodes an event record's envelope and payload.
fn message(
    record: &Map<String, Value>,
    direction: Direction,
    line: usize,
) -> Result<Record, TranscriptError> {
    let event = record
        .get("event")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed(line, "no event"))?;
    let sender: AgentId = decode(event, "sender", line)?;
    let recipients: BTreeSet<AgentId> = decode(event, "recipients", line)?;
    let payload = event
        .get("payload")
        .ok_or_else(|| malformed(line, "no payload"))?;
    let message = Message::deserialize(payload)
        .map_err(|source| TranscriptError::Payload { line, source })?;
    Ok(Record {
        line,
        direction,
        sender,
        recipients,
        message,
    })
}

fn malformed(line: usize, what: &str) -> TranscriptError {
    TranscriptError::Malformed {
        line,
        what: what.to_owned(),
    }
}

fn string<'a>(
    object: &'a Map<String, Value>,
    key: &str,
    line: usize,
) -> Result<&'a str, TranscriptError> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| malformed(line, &format!("no {key}")))
}

fn integer(object: &Map<String, Value>, key: &str, line: usize) -> Result<u64, TranscriptError> {
    object
        .get(key)
        .and_then(Value::as_u64)
        .ok_or_else(|| malformed(line, &format!("no {key}")))
}

fn decode<T: serde::de::DeserializeOwned>(
    object: &Map<String, Value>,
    key: &str,
    line: usize,
) -> Result<T, TranscriptError> {
    let value = object
        .get(key)
        .ok_or_else(|| malformed(line, &format!("no {key}")))?;
    T::deserialize(value).map_err(|error| malformed(line, &format!("{key}: {error}")))
}

/// The one agent a message addressed to exactly one agent was addressed to,
/// or [`TranscriptError::Malformed`] saying `what` was expected if it was
/// addressed to none or to several.
fn only(
    line: usize,
    recipients: BTreeSet<AgentId>,
    what: &str,
) -> Result<AgentId, TranscriptError> {
    let mut recipients = recipients.into_iter();
    match (recipients.next(), recipients.next()) {
        (Some(who), None) => Ok(who),
        _ => Err(malformed(line, what)),
    }
}

/// The game so far, as a fold over the moderator's records.
#[derive(Default)]
struct Reader {
    assignment: BTreeMap<AgentId, Role>,
    rounds: Vec<RoundRecord>,
    /// The requests sent and not yet answered: who was asked, and what.
    outstanding: BTreeMap<RequestId, (AgentId, RequestKind)>,
    outcome: Option<Outcome>,
}

impl Reader {
    fn fold(&mut self, record: Record) -> Result<(), TranscriptError> {
        let Record {
            line,
            direction,
            sender,
            recipients,
            message,
        } = record;
        match (direction, message) {
            (Direction::Sent, Message::Narration(narration)) => {
                self.narrated(line, recipients, narration)
            }
            (Direction::Sent, Message::Request(request)) => self.asked(line, recipients, &request),
            (Direction::Received, Message::Response(response)) => {
                self.answered(line, sender, response)
            }
            _ => Err(TranscriptError::Misdirected { line }),
        }
    }

    fn narrated(
        &mut self,
        line: usize,
        recipients: BTreeSet<AgentId>,
        narration: Narration,
    ) -> Result<(), TranscriptError> {
        match narration {
            Narration::Assigned { role, .. } => {
                for who in recipients {
                    self.assignment.insert(who, role);
                }
            }
            Narration::PhaseBegan {
                round,
                phase: Phase::Night,
                living,
            } => self.rounds.push(RoundRecord {
                round,
                night: PhaseRecord::begun(living),
                day: None,
            }),
            Narration::PhaseBegan {
                round,
                phase: Phase::Day,
                living,
            } => match self.rounds.last_mut() {
                Some(latest) if latest.round == round && latest.day.is_none() => {
                    latest.day = Some(PhaseRecord::begun(living));
                }
                _ => return Err(TranscriptError::NoPhase { line }),
            },
            Narration::Investigated { target, faction } => {
                let seer = only(line, recipients, "a finding is addressed to one seer")?;
                self.current(line)?.investigation = Some((seer, target, faction));
            }
            Narration::Eliminated {
                who, role, cause, ..
            } => self.current(line)?.eliminated = Some((who, role, cause)),
            // The tally repeats the responses already recorded, and a night
            // without a death is one without an elimination.
            Narration::Tally { .. } | Narration::NoDeath { .. } => {}
            Narration::Outcome(outcome) => self.outcome = Some(outcome),
        }
        Ok(())
    }

    fn asked(
        &mut self,
        line: usize,
        recipients: BTreeSet<AgentId>,
        request: &Request,
    ) -> Result<(), TranscriptError> {
        let who = only(line, recipients, "a request is addressed to one player")?;
        self.outstanding.insert(request.id, (who, request.kind));
        Ok(())
    }

    fn answered(
        &mut self,
        line: usize,
        from: AgentId,
        response: Response,
    ) -> Result<(), TranscriptError> {
        // A mismatch ends the read, so removing before checking loses nothing.
        let kind = match self.outstanding.remove(&response.request) {
            Some((asked, kind)) if asked == from => kind,
            _ => {
                return Err(TranscriptError::UnknownRequest {
                    line,
                    from,
                    request: response.request,
                });
            }
        };
        self.current(line)?
            .moves
            .insert(from, (kind, response.chosen));
        Ok(())
    }

    /// The phase in progress: the latest round's day if it has begun, else
    /// its night.
    fn current(&mut self, line: usize) -> Result<&mut PhaseRecord, TranscriptError> {
        let round = self
            .rounds
            .last_mut()
            .ok_or(TranscriptError::NoPhase { line })?;
        Ok(round.day.as_mut().unwrap_or(&mut round.night))
    }
}

/// How many cells a roster or a tally lays out per line.
const COLUMNS: usize = 4;

impl fmt::Display for Transcript {
    /// The game at a glance: the roster with its roles, then each phase with
    /// its moves and its elimination, then the outcome. Only what the
    /// moderator recorded, in the order it recorded it. Ends with a newline.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let width = self
            .assignment
            .keys()
            .map(|who| who.as_str().len())
            .max()
            .unwrap_or(0);
        let role_width = self
            .assignment
            .values()
            .map(|role| role.to_string().len())
            .max()
            .unwrap_or(0);
        let roster = self.assignment.iter().map(|(who, role)| {
            format!(
                "{:<width$}  {:<role_width$}",
                who.as_str(),
                role.to_string()
            )
        });
        columns(f, roster)?;
        for round in &self.rounds {
            let Round(number) = round.round;
            phase(f, &format!("Night {number}"), &round.night, width)?;
            if let Some(day) = &round.day {
                phase(f, &format!("Day {number}"), day, width)?;
            }
        }
        writeln!(f)?;
        let Round(rounds) = self.outcome.rounds;
        match self.outcome.winner {
            Faction::Village => write!(f, "Village wins")?,
            Faction::Werewolves => write!(f, "Werewolves win")?,
        }
        let plural = if rounds == 1 { "" } else { "s" };
        write!(f, " after {rounds} round{plural}.  Survivors: ")?;
        let survivors: Vec<&str> = self.outcome.living.iter().map(AgentId::as_str).collect();
        if survivors.is_empty() {
            writeln!(f, "none")
        } else {
            writeln!(f, "{}", survivors.join(", "))
        }
    }
}

/// Writes a phase: its header with the living count, one line per move,
/// and who died.
fn phase(
    f: &mut fmt::Formatter<'_>,
    name: &str,
    record: &PhaseRecord,
    width: usize,
) -> fmt::Result {
    writeln!(f)?;
    writeln!(f, "{name}  ({} living)", record.living.len())?;
    let (votes, deeds): (Vec<_>, Vec<_>) = record
        .moves
        .iter()
        .partition(|(_, (kind, _))| *kind == RequestKind::Nominate);
    // Nominations are a ballot, laid out in columns; night moves differ
    // by kind and get a line each.
    columns(
        f,
        votes.iter().map(|(who, (_, chosen))| {
            format!(
                "{:<width$} -> {:<width$}",
                who.as_str(),
                chosen.target().map_or("no one", AgentId::as_str)
            )
        }),
    )?;
    for (who, (kind, chosen)) in deeds {
        let verb = match kind {
            RequestKind::Devour => "devours",
            RequestKind::Investigate => "investigates",
            RequestKind::Protect => "protects",
            RequestKind::Nominate => "nominates",
        };
        let whom = chosen.target().map_or("no one", AgentId::as_str);
        write!(f, "  {:<width$} {verb} {whom}", who.as_str())?;
        match &record.investigation {
            Some((seer, _, faction)) if seer == who => writeln!(f, "  ->  {faction}")?,
            _ => writeln!(f)?,
        }
    }
    match &record.eliminated {
        Some((who, role, Cause::Devoured)) => writeln!(f, "  {who} is devoured   ({role})"),
        Some((who, role, Cause::Lynched)) => writeln!(f, "  {who} is lynched   ({role})"),
        None => writeln!(f, "  no one died"),
    }
}

/// Writes cells [`COLUMNS`] to a line, indented, without trailing blanks.
fn columns(f: &mut fmt::Formatter<'_>, cells: impl Iterator<Item = String>) -> fmt::Result {
    let cells: Vec<String> = cells.collect();
    for row in cells.chunks(COLUMNS) {
        writeln!(f, "  {}", row.join("   ").trim_end())?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::testing::{id, ids};
    use crate::werewolf::assignment::Assignment;
    use crate::werewolf::game::{Directive, Game};
    use crate::werewolf::role::Role::{Doctor, Seer, Villager, Werewolf};

    /// The fixture: a seven-player game played to a werewolf win in two
    /// rounds, with its effective config and its expected rendering beside
    /// it. Between them the two rounds cover a saved night, a night the
    /// doctor's own rule bars it from repeating a protection on, the
    /// tie-break on both a split pack and a split ballot, a game that ends
    /// by parity rather than by the pack being wiped out, and the
    /// elimination of a doctor and a seer.
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.jsonl"
    ));
    const RENDERED: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.txt"
    ));

    const MODERATOR: &str = "moderator";

    fn moderator() -> AgentId {
        id(MODERATOR)
    }

    fn fixture() -> Vec<Value> {
        lines(FIXTURE).unwrap()
    }

    fn read(lines: &[Value]) -> Result<Transcript, TranscriptError> {
        Transcript::read(lines, &moderator())
    }

    fn target(who: &str) -> Move {
        Move::Target(id(who))
    }

    fn moves<const N: usize>(
        moves: [(&str, RequestKind, Move); N],
    ) -> BTreeMap<AgentId, (RequestKind, Move)> {
        moves
            .into_iter()
            .map(|(who, kind, chosen)| (id(who), (kind, chosen)))
            .collect()
    }

    fn nominations<const N: usize>(
        votes: [(&str, &str); N],
    ) -> BTreeMap<AgentId, (RequestKind, Move)> {
        votes
            .into_iter()
            .map(|(who, whom)| (id(who), (RequestKind::Nominate, target(whom))))
            .collect()
    }

    fn phase<const N: usize>(
        living: [&str; N],
        moves: BTreeMap<AgentId, (RequestKind, Move)>,
        investigation: Option<(&str, &str, Faction)>,
        eliminated: Option<(&str, Role, Cause)>,
    ) -> PhaseRecord {
        PhaseRecord {
            living: ids(living),
            moves,
            investigation: investigation.map(|(seer, whom, faction)| (id(seer), id(whom), faction)),
            eliminated: eliminated.map(|(who, role, cause)| (id(who), role, cause)),
        }
    }

    /// The game the fixture records, written out by hand from the
    /// moderator's records.
    fn expected() -> Transcript {
        use Cause::{Devoured, Lynched};
        use RequestKind::{Devour, Investigate, Protect};
        let everyone = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];
        Transcript {
            assignment: [
                ("alice", Villager),
                ("bob", Villager),
                ("carol", Doctor),
                ("dave", Werewolf),
                ("erin", Werewolf),
                ("frank", Villager),
                ("grace", Seer),
            ]
            .into_iter()
            .map(|(who, role)| (id(who), role))
            .collect(),
            rounds: vec![
                RoundRecord {
                    round: Round(1),
                    // The pack splits, the tie-break picks alice, and the
                    // doctor has protected her: a saved night.
                    night: phase(
                        everyone,
                        moves([
                            ("carol", Protect, target("alice")),
                            ("dave", Devour, target("alice")),
                            ("erin", Devour, target("bob")),
                            ("grace", Investigate, target("alice")),
                        ]),
                        Some(("grace", "alice", Faction::Village)),
                        None,
                    ),
                    day: Some(phase(
                        everyone,
                        nominations([
                            ("alice", "frank"),
                            ("bob", "carol"),
                            ("carol", "bob"),
                            ("dave", "alice"),
                            ("erin", "alice"),
                            ("frank", "grace"),
                            ("grace", "erin"),
                        ]),
                        None,
                        Some(("alice", Villager, Lynched)),
                    )),
                },
                RoundRecord {
                    round: Round(2),
                    // The pack agrees on carol, whom the doctor did not
                    // protect, and the seer finds a werewolf too late.
                    night: phase(
                        ["bob", "carol", "dave", "erin", "frank", "grace"],
                        moves([
                            ("carol", Protect, target("erin")),
                            ("dave", Devour, target("carol")),
                            ("erin", Devour, target("carol")),
                            ("grace", Investigate, target("dave")),
                        ]),
                        Some(("grace", "dave", Faction::Werewolves)),
                        Some(("carol", Doctor, Devoured)),
                    ),
                    // Grace and frank tie, and the tie-break lynches grace:
                    // two werewolves among four living is parity.
                    day: Some(phase(
                        ["bob", "dave", "erin", "frank", "grace"],
                        nominations([
                            ("bob", "grace"),
                            ("dave", "bob"),
                            ("erin", "frank"),
                            ("frank", "grace"),
                            ("grace", "frank"),
                        ]),
                        None,
                        Some(("grace", Seer, Lynched)),
                    )),
                },
            ],
            outcome: Outcome {
                winner: Faction::Werewolves,
                rounds: Round(2),
                living: ids(["bob", "dave", "erin", "frank"]),
            },
        }
    }

    #[test]
    fn the_fixture_reads_as_the_expected_game() {
        let transcript = read(&fixture()).unwrap();
        let expected = expected();
        assert_eq!(transcript.assignment, expected.assignment);
        assert_eq!(transcript.rounds.len(), expected.rounds.len());
        for (actual, expected) in transcript.rounds.iter().zip(&expected.rounds) {
            assert_eq!(actual.round, expected.round);
            assert_eq!(actual.night, expected.night, "night {:?}", actual.round);
            assert_eq!(actual.day, expected.day, "day {:?}", actual.round);
        }
        assert_eq!(transcript.outcome, expected.outcome);
    }

    #[test]
    fn the_fixture_renders_to_the_golden_text() {
        // Compared exactly, so that a change to the rendering is a
        // deliberate act that updates the golden file.
        assert_eq!(read(&fixture()).unwrap().to_string(), RENDERED);
    }

    #[test]
    fn every_action_carries_the_kind_of_the_request_it_answered() {
        let transcript = read(&fixture()).unwrap();
        let roles = &transcript.assignment;
        for round in &transcript.rounds {
            for (who, (kind, _)) in &round.night.moves {
                let expected = match roles[who] {
                    Werewolf => RequestKind::Devour,
                    Seer => RequestKind::Investigate,
                    Doctor => RequestKind::Protect,
                    Villager => panic!("{who} acted at night"),
                };
                assert_eq!(*kind, expected, "{who} in round {:?}", round.round);
            }
            for (who, (kind, _)) in &round.day.as_ref().unwrap().moves {
                assert_eq!(
                    *kind,
                    RequestKind::Nominate,
                    "{who} in round {:?}",
                    round.round
                );
            }
        }
    }

    /// The fixture with every timestamp rewritten and the agents' records
    /// interleaved differently: the same game as a different file.
    fn rewritten() -> Vec<Value> {
        let mut lines = fixture();
        for (offset, line) in lines.iter_mut().enumerate() {
            for key in ["created", "received", "t_start", "t_stop"] {
                if let Some(time) = line.get_mut(key) {
                    *time = json!(1_000_000 + offset);
                }
            }
        }
        // A stable sort by agent keeps each agent's records in order, which
        // is the one ordering the runtime guarantees, and otherwise changes
        // the interleaving completely.
        lines.sort_by_key(|line| line["agent"].as_str().unwrap().to_owned());
        assert_ne!(lines, fixture());
        lines
    }

    #[test]
    fn a_transcript_has_no_timestamps_and_no_interleaving() {
        assert_eq!(read(&rewritten()).unwrap(), read(&fixture()).unwrap());
    }

    #[test]
    fn only_the_moderators_records_are_read() {
        // Nobody by this name recorded anything, so there is no game.
        let error = Transcript::read(&fixture(), &id("narrator")).unwrap_err();
        assert!(matches!(error, TranscriptError::NoOutcome), "{error:?}");
        // The players' records, on their own, are not the game either.
        let players_only: Vec<Value> = fixture()
            .into_iter()
            .filter(|line| line["agent"] != MODERATOR)
            .collect();
        let error = read(&players_only).unwrap_err();
        assert!(matches!(error, TranscriptError::NoOutcome), "{error:?}");
    }

    /// The index of the first of the moderator's records that carries an
    /// event whose payload satisfies `matching`.
    fn moderator_record(lines: &[Value], matching: impl Fn(&Value) -> bool) -> usize {
        lines
            .iter()
            .position(|line| {
                line["agent"] == MODERATOR
                    && (line["type"] == "action" || line["type"] == "observation")
                    && matching(&line["event"]["payload"])
            })
            .expect("the fixture has such a record")
    }

    fn is_response(payload: &Value) -> bool {
        !payload["Response"].is_null()
    }

    #[test]
    fn a_line_that_is_not_json_is_an_error() {
        let text = FIXTURE.replacen('{', "", 1);
        let error = lines(&text).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NotJson { line: 1, .. }),
            "{error:?}"
        );
        assert!(error.source().is_some());
        assert!(
            error.to_string().starts_with("line 1 is not JSON"),
            "{error}"
        );
    }

    #[test]
    fn a_line_that_is_not_an_object_is_an_error() {
        let mut lines = fixture();
        lines[4] = json!([1, 2, 3]);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NotAnObject { line: 5 }),
            "{error:?}"
        );
        assert_eq!(error.to_string(), "line 5 is not a JSON object");
    }

    #[test]
    fn an_unknown_record_type_is_an_error_whoever_wrote_it() {
        let mut lines = fixture();
        // A player's record: every line is checked to be a record, even
        // the ones that are not read.
        let index = lines
            .iter()
            .position(|line| line["agent"] == "alice")
            .unwrap();
        lines[index]["type"] = json!("note");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::UnknownRecordType { line, found: Some(found) }
                if *line == index + 1 && found == "note"),
            "{error:?}"
        );
        assert!(error.to_string().contains("\"note\""), "{error}");

        let mut lines = fixture();
        lines[index].as_object_mut().unwrap().remove("type");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(
                &error,
                TranscriptError::UnknownRecordType { found: None, .. }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains("no record type"), "{error}");
    }

    #[test]
    fn a_record_missing_part_of_its_envelope_is_an_error() {
        let index = moderator_record(&fixture(), is_response);
        for key in ["agent", "seq", "event"] {
            let mut lines = fixture();
            lines[index].as_object_mut().unwrap().remove(key);
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what == &format!("no {key}")),
                "{key}: {error:?}"
            );
        }
        for key in ["sender", "recipients", "payload"] {
            let mut lines = fixture();
            lines[index]["event"].as_object_mut().unwrap().remove(key);
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what.starts_with(&format!("no {key}"))),
                "{key}: {error:?}"
            );
        }
    }

    #[test]
    fn a_control_record_says_nothing_about_the_game_and_is_skipped() {
        // The moderator's own start and stop are in the file and count
        // toward its sequence numbers, and the reader steps over them
        // without taking them for messages.
        let lines = fixture();
        let controls = lines
            .iter()
            .filter(|line| line["agent"] == MODERATOR && line["type"] == "control")
            .count();
        assert_eq!(controls, 2, "the moderator was started and stopped");
        read(&lines).unwrap();
    }

    #[test]
    fn a_payload_that_will_not_deserialize_is_an_error() {
        let mut lines = fixture();
        let index = moderator_record(&lines, is_response);
        lines[index]["event"]["payload"] = json!({"Response": {"request": "seven"}});
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Payload { line, .. } if *line == index + 1),
            "{error:?}"
        );
        assert!(error.source().is_some());
        assert!(error.to_string().contains("Werewolf message"), "{error}");
    }

    #[test]
    fn a_gap_in_the_moderators_sequence_numbers_is_an_error() {
        let mut lines = fixture();
        let index = moderator_record(&lines, is_response);
        lines[index]["seq"] = json!(lines[index]["seq"].as_u64().unwrap() + 1);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::SeqGap { line, expected, found }
                if *line == index + 1 && *found == *expected + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("sequence number"), "{error}");
    }

    #[test]
    fn a_response_to_an_unknown_request_is_an_error() {
        let mut lines = fixture();
        let index = moderator_record(&lines, is_response);
        lines[index]["event"]["payload"]["Response"]["request"] = json!(99);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::UnknownRequest { line, request: RequestId(99), .. }
                if *line == index + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("request 99"), "{error}");
    }

    #[test]
    fn a_response_from_someone_the_request_was_not_asked_of_is_an_error() {
        let mut lines = fixture();
        let index = moderator_record(&lines, is_response);
        lines[index]["event"]["sender"] = json!("frank");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::UnknownRequest { from, .. } if from.as_str() == "frank"),
            "{error:?}"
        );
    }

    #[test]
    fn a_game_without_an_outcome_is_an_error() {
        let index = moderator_record(&fixture(), |payload| {
            !payload["Narration"]["Outcome"].is_null()
        });
        let mut lines = fixture();
        lines.truncate(index);
        let error = read(&lines).unwrap_err();
        assert!(matches!(error, TranscriptError::NoOutcome), "{error:?}");
        assert_eq!(
            error.to_string(),
            "the moderator never announced an outcome"
        );
    }

    #[test]
    fn an_empty_agent_id_is_an_error_wherever_it_appears() {
        // `AgentId` refuses to deserialize from the empty string, so an
        // empty id in the envelope is a malformed envelope and one in the
        // payload is a payload that is not a Werewolf message, each naming
        // the line.
        let index = moderator_record(&fixture(), is_response);
        for (key, empty) in [("sender", json!("")), ("recipients", json!([""]))] {
            let mut lines = fixture();
            lines[index]["event"][key] = empty;
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what.starts_with(key)),
                "{key}: {error:?}"
            );
            assert!(error.to_string().contains("non-empty agent id"), "{error}");
        }
        let mut lines = fixture();
        lines[index]["event"]["payload"]["Response"]["chosen"]["Target"] = json!("");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Payload { line, .. } if *line == index + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("non-empty agent id"), "{error}");
        // A narration's ids are checked too.
        let mut lines = fixture();
        let index = moderator_record(&lines, |payload| {
            !payload["Narration"]["Eliminated"].is_null()
        });
        lines[index]["event"]["payload"]["Narration"]["Eliminated"]["who"] = json!("");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Payload { line, .. } if *line == index + 1),
            "{error:?}"
        );
    }

    fn is_request(payload: &Value) -> bool {
        !payload["Request"].is_null()
    }

    #[test]
    fn a_request_addressed_to_other_than_one_player_is_an_error() {
        // Naming the request's line, not the line of whichever response
        // later fails to match it.
        for to in [json!([]), json!(["alice", "bob"])] {
            let mut lines = fixture();
            let index = moderator_record(&lines, is_request);
            lines[index]["event"]["recipients"] = to.clone();
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what == "a request is addressed to one player"),
                "{to}: {error:?}"
            );
        }
    }

    #[test]
    fn a_finding_addressed_to_other_than_one_seer_is_an_error() {
        for to in [json!([]), json!(["alice", "grace"])] {
            let mut lines = fixture();
            let index = moderator_record(&lines, |payload| {
                !payload["Narration"]["Investigated"].is_null()
            });
            lines[index]["event"]["recipients"] = to.clone();
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what == "a finding is addressed to one seer"),
                "{to}: {error:?}"
            );
        }
    }

    #[test]
    fn a_response_before_any_phase_is_an_error() {
        let mut lines = fixture();
        // The first night's announcement becomes something harmless, so the
        // first response answers a request in no phase.
        let index = moderator_record(&lines, |payload| {
            !payload["Narration"]["PhaseBegan"].is_null()
        });
        lines[index]["event"]["payload"] = json!({"Narration": {"NoDeath": {"round": 1}}});
        let error = read(&lines).unwrap_err();
        let first_response = moderator_record(&lines, is_response);
        assert!(
            matches!(error, TranscriptError::NoPhase { line } if line == first_response + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("none has begun"), "{error}");
    }

    #[test]
    fn a_day_whose_night_has_not_begun_is_an_error() {
        let is_phase = |payload: &Value| !payload["Narration"]["PhaseBegan"].is_null();
        // The first night announced as a day.
        let mut lines = fixture();
        let index = moderator_record(&lines, is_phase);
        lines[index]["event"]["payload"]["Narration"]["PhaseBegan"]["phase"] = json!("Day");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NoPhase { line } if line == index + 1),
            "{error:?}"
        );
        // The first day announced as belonging to a round yet to come.
        let mut lines = fixture();
        let day = index + 1 + moderator_record(&lines[index + 1..], is_phase);
        lines[day]["event"]["payload"]["Narration"]["PhaseBegan"]["round"] = json!(2);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NoPhase { line } if line == day + 1),
            "{error:?}"
        );
    }

    #[test]
    fn a_message_the_moderator_never_records_is_an_error() {
        // A response the moderator took as an action of its own.
        let mut lines = fixture();
        let index = moderator_record(&lines, is_response);
        lines[index]["type"] = json!("action");
        lines[index].as_object_mut().unwrap().remove("received");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::Misdirected { line } if line == index + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("never records"), "{error}");

        // A narration the moderator observed rather than sent.
        let mut lines = fixture();
        let index = moderator_record(&lines, |payload| !payload["Narration"].is_null());
        lines[index]["type"] = json!("observation");
        lines[index]["received"] = lines[index]["created"].clone();
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::Misdirected { .. }),
            "{error:?}"
        );
    }

    /// Writes the moderator's records of a game played through [`Game`]
    /// with scripted responses: a trajectory of exactly what the moderator
    /// would record, with nobody else's records and made-up stamps.
    struct Scribe {
        lines: Vec<Value>,
        players: BTreeSet<AgentId>,
    }

    impl Scribe {
        fn new(assignment: &Assignment) -> Self {
            Self {
                lines: Vec::new(),
                players: assignment.players().map(|(who, _)| who.clone()).collect(),
            }
        }

        /// One record of the moderator's, of the given type, with stamps
        /// invented from the sequence number: every record needs a
        /// `created`, and an `observation` a `received` as well.
        fn record(
            &mut self,
            kind: &str,
            sender: &str,
            recipients: &BTreeSet<AgentId>,
            payload: &Message,
        ) {
            let seq = self.lines.len();
            let mut line = json!({
                "type": kind,
                "agent": MODERATOR,
                "seq": seq,
                "created": seq * 10,
                "event": {
                    "sender": sender,
                    "recipients": recipients,
                    "payload": payload,
                },
            });
            if kind == "observation" {
                line["received"] = json!(seq * 10 + 1);
            }
            self.lines.push(line);
        }

        fn directives(&mut self, directives: Vec<Directive>) {
            for directive in directives {
                let (to, payload) = match directive {
                    Directive::Narrate { to, narration } => (to, Message::Narration(narration)),
                    Directive::Ask { to, request } => ([to].into(), Message::Request(request)),
                    Directive::Broadcast(narration) => {
                        (self.players.clone(), Message::Narration(narration))
                    }
                };
                self.record("action", MODERATOR, &to, &payload);
            }
        }

        fn response(&mut self, from: &str, response: Response) {
            self.record(
                "observation",
                from,
                &ids([MODERATOR]),
                &Message::Response(response),
            );
        }
    }

    /// Plays `script`, one phase's answers per entry, through a game over
    /// `assignment`, and returns the moderator's records.
    fn scripted(assignment: Assignment, script: &[Vec<(&str, Move)>]) -> Vec<Value> {
        let mut scribe = Scribe::new(&assignment);
        let mut game = Game::new(assignment, 1);
        let mut latest = game.begin();
        scribe.directives(latest.clone());
        for answers in script {
            let asked: BTreeMap<AgentId, RequestId> = latest
                .iter()
                .filter_map(|directive| match directive {
                    Directive::Ask { to, request } => Some((to.clone(), request.id)),
                    _ => None,
                })
                .collect();
            for (who, chosen) in answers {
                let response = Response {
                    request: asked[&id(who)],
                    chosen: chosen.clone(),
                };
                scribe.response(who, response.clone());
                latest = game.record(&id(who), &response);
                scribe.directives(latest.clone());
            }
        }
        scribe.lines
    }

    fn answers<const N: usize>(answers: [(&'static str, &str); N]) -> Vec<(&'static str, Move)> {
        answers
            .into_iter()
            .map(|(who, whom)| {
                (
                    who,
                    if whom == "-" {
                        Move::Abstain
                    } else {
                        target(whom)
                    },
                )
            })
            .collect()
    }

    /// Five players and one werewolf: bob, with carol the seer and dave the
    /// doctor.
    fn village() -> Assignment {
        Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Seer),
            ("dave", Doctor),
            ("erin", Villager),
        ])
    }

    #[test]
    fn a_game_that_ends_at_night_has_no_final_day() {
        // Carol is devoured, erin is lynched, and alice is devoured while
        // the doctor abstains: the werewolves win at parity on night two.
        let lines = scripted(
            village(),
            &[
                answers([("bob", "carol"), ("carol", "bob"), ("dave", "alice")]),
                answers([
                    ("alice", "erin"),
                    ("bob", "erin"),
                    ("dave", "erin"),
                    ("erin", "alice"),
                ]),
                answers([("bob", "alice"), ("dave", "-")]),
            ],
        );
        let transcript = read(&lines).unwrap();
        assert_eq!(transcript.rounds.len(), 2);
        let last = &transcript.rounds[1];
        assert_eq!(last.day, None);
        assert_eq!(
            last.night.moves,
            moves([
                ("bob", RequestKind::Devour, target("alice")),
                ("dave", RequestKind::Protect, Move::Abstain),
            ])
        );
        assert_eq!(
            last.night.eliminated,
            Some((id("alice"), Villager, Cause::Devoured))
        );
        assert_eq!(transcript.outcome.winner, Faction::Werewolves);
        let rendered = transcript.to_string();
        assert!(rendered.contains("Night 2  (3 living)\n"), "{rendered}");
        assert!(!rendered.contains("Day 2"), "{rendered}");
        assert!(rendered.contains("  dave  protects no one\n"), "{rendered}");
        assert!(
            rendered.ends_with("Werewolves win after 2 rounds.  Survivors: bob, dave\n"),
            "{rendered}"
        );
    }

    #[test]
    fn a_game_with_no_seer_has_no_investigations() {
        let assignment = Assignment::new([
            ("alice", Werewolf),
            ("bob", Villager),
            ("carol", Villager),
            ("dave", Villager),
            ("erin", Villager),
        ]);
        // Bob is devoured, carol is lynched, dave is devoured: parity.
        let lines = scripted(
            assignment,
            &[
                answers([("alice", "bob")]),
                answers([
                    ("alice", "carol"),
                    ("carol", "dave"),
                    ("dave", "carol"),
                    ("erin", "carol"),
                ]),
                answers([("alice", "dave")]),
            ],
        );
        let transcript = read(&lines).unwrap();
        assert_eq!(transcript.rounds.len(), 2);
        for round in &transcript.rounds {
            assert_eq!(round.night.investigation, None, "{:?}", round.round);
            assert_eq!(round.night.moves.len(), 1);
            if let Some(day) = &round.day {
                assert_eq!(day.investigation, None);
            }
        }
        assert!(!transcript.to_string().contains("investigates"));
    }

    #[test]
    fn errors_display_the_line_they_were_found_on() {
        let error = TranscriptError::UnknownRequest {
            line: 12,
            from: id("bob"),
            request: RequestId(7),
        };
        assert_eq!(
            error.to_string(),
            "line 12: bob answered request 7, which was not asked of it"
        );
        assert!(error.source().is_none());
    }
}
