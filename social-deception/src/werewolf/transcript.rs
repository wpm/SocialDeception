//! Reading the log back as the game it records, and rendering that game
//! for a person to read.
//!
//! A [`Transcript`] is the logical game: who held which role, what everyone
//! did in each phase, who was eliminated and how, and how it ended. It has no
//! timestamps and no record interleaving, because those are the two things
//! about a log file that are not reproducible: the agents are threads,
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
//! configuration written beside the log (see
//! [`config::effective_path`](super::config::effective_path)), and whoever
//! prints a transcript prints the seed from there.
//!
//! # Only the moderator's records, and every agent's rewards
//!
//! [`Transcript::read`] reads the moderator's records and nobody else's,
//! with one exception: a **reward** record is read whoever it belongs to.
//! It has to be. A reward belongs to the agent rewarded and is written by
//! the environment, so it appears under a player's name and never under
//! the moderator's, and reading only the moderator's records would find
//! none of them. It is no exception to the principle, though: a reward is
//! the environment's own statement about a player, not a partial view
//! recovered from one, and it is identified by its `type` rather than by
//! whose records it sits in.
//!
//! The moderator is the authoritative view: its `action` records are every
//! narration it sent and every selection it passed on, and its `observation`
//! records are every selection it received, each naming the player that made
//! it as its sender. A relay names the player in its envelope instead, since
//! the moderator is what sent it. Which record type a line is *is* the
//! direction, so the reader needs no direction of its own. Reassembling the
//! game from the players' records would mean recovering hidden information
//! from partial views, which is the thing the design prevents.
//!
//! # File order, and the one thing checked about it
//!
//! Line order in a log carries no meaning: records arrive at the writer from
//! as many threads as there are agents (ADR-0017). What holds for one agent
//! is that its own records reach the writer in the order it wrote them, and
//! since the moderator is one thread, its records in file order are the game
//! in the order it played it. That is what this reader folds over.
//!
//! What it checks is the one thing the log can prove about that order: the
//! moderator's own sequence numbers are contiguous from zero in the order it
//! decided them, so a gap means the file is missing one of its messages.
//! Every action of the moderator's is counted, relays included: a relay is
//! the moderator's own message and takes a number of the moderator's
//! (ADR-0017), and the player it passes on is named by the envelope inside
//! it.
//!
//! A **reminder is counted too**, and it is the one message of the
//! moderator's that appears in the log as an *observation* rather than as an
//! action. A reminder is numbered where it is set, like everything else the
//! moderator sends, but it is not logged there: it is logged when it arrives,
//! as the observation it becomes, because that is the one instant about it
//! that means anything (ADR-0016). So the moderator's numbers run over its
//! narrations, its relays and its reminders together, and a reader that
//! counted only the actions would see a gap wherever a reminder took a
//! number.
//!
//! An observation of a *player's* selection carries the number of the player
//! that sent it, and a control or a reward carries none at all, so neither is
//! counted. What tells the two kinds of observation apart is the sender: a
//! reminder is always self-directed, so its sender is the moderator.
//!
//! # No game logic
//!
//! Nothing here decides anything. The reader records what the moderator
//! said and passed on, in the order it did so, and nothing more: it does not
//! count votes, check a win condition or infer a save. If reconstructing the game ever
//! needed a rule, the moderator would not be recording enough, and the fix
//! would belong with the moderator.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;

use serde::Deserialize;
use serde_json::{Map, Value};

use super::message::{Cause, Message, Narration, Outcome, Phase, Round, Select, SessionKind};
use super::role::{Faction, Role};
use crate::message::ActorId;

/// A game of Werewolf as its moderator recorded it: everything that is
/// reproducible about a run, and nothing that is not.
///
/// See the [module documentation](self) for what is left out and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Transcript {
    /// Each player's role, as the moderator dealt it.
    pub assignment: BTreeMap<ActorId, Role>,
    /// Every round, in order.
    pub rounds: Vec<RoundRecord>,
    /// How the game ended.
    pub outcome: Outcome,
    /// What each player's game was worth: +1 for a player on the winning
    /// side and −1 for every other, living or dead, as the moderator
    /// logged it when the game ended (ADR-0007).
    ///
    /// It is read from the `reward` records, which belong to the players
    /// and not to the moderator; see the [module documentation](self). It
    /// is part of the logical game and so part of what two runs of one
    /// seed must agree on, which is why it is here and not left to a
    /// reader of the raw file.
    pub rewards: BTreeMap<ActorId, i32>,
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
    pub living: BTreeSet<ActorId>,
    /// Every selection made this phase, by the player that made it, paired
    /// with the kind of session it was made in. The target alone does not
    /// say whether it was devoured, protected, investigated or nominated.
    pub moves: BTreeMap<ActorId, (SessionKind, ActorId)>,
    /// What each seer learned: whom it investigated and the faction that
    /// came back, by seer. Empty on a phase where no seer investigated,
    /// and holding one entry per seer that did, since a game may deal
    /// more than one.
    pub investigations: BTreeMap<ActorId, (ActorId, Faction)>,
    /// Who was eliminated, the role their death revealed, and how. `None`
    /// on a night when nobody died and on a day that ran out of time.
    pub eliminated: Option<(ActorId, Role, Cause)>,
    /// Whether this day closed at its limit with no majority, so that a
    /// day with nobody lynched is told from a day still being read
    /// (ADR-0011). Always false for a night.
    pub no_lynch: bool,
    /// The player whose selection completed the majority that ended the day:
    /// the *hammer*. `None` for a night, and for a day that ran out.
    pub hammer: Option<ActorId>,
}

impl PhaseRecord {
    /// A phase in which nothing has happened yet.
    fn begun(living: BTreeSet<ActorId>) -> Self {
        Self {
            living,
            moves: BTreeMap::new(),
            investigations: BTreeMap::new(),
            eliminated: None,
            no_lynch: false,
            hammer: None,
        }
    }
}

/// Why the log could not be read as a game.
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
    /// The moderator's own messages are not contiguously numbered, so the
    /// file is missing one of them.
    SeqGap {
        /// The line.
        line: usize,
        /// The sequence number that should have come next.
        expected: u64,
        /// The one found.
        found: u64,
    },
    /// A selection was made in a session its sender is not a member of: the
    /// phase does not ask that of its role, or the selection names a round
    /// that is not the one under way.
    NotAMember {
        /// The line.
        line: usize,
        /// The selecting agent.
        from: ActorId,
        /// The session it selected in.
        kind: SessionKind,
    },
    /// A record belongs to a phase that has not begun: a selection, an
    /// investigation or an elimination before the first night, or a day
    /// whose night has not begun or that has begun already.
    NoPhase {
        /// The line.
        line: usize,
    },
    /// A message the moderator never records: a narration it observed
    /// rather than sent, or a selection it took as an action rather than
    /// received.
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
            Self::NotAMember { line, from, kind } => write!(
                f,
                "line {line}: {from} selected in a {kind:?} session, which it is not a member of"
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

/// Parses the text of a log file into one JSON value per line, the
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
    /// An `undelivered` or `unsent` record: a message that went nowhere,
    /// because the actor it belonged to had been stopped (ADR-0016).
    ///
    /// The reader folds nothing from one — nothing happened in the game — but
    /// it is the moderator's message all the same, and it took a number, so
    /// it is counted. A moderator that is stopped with a reminder still on
    /// its timer leaves exactly one of these, and a reader that did not count
    /// it would see a gap.
    Lost,
}

/// One record the reader takes an interest in.
///
/// A [`Message`](Line::Message) line is one of the moderator's, going one way
/// or the other. A [`Reward`](Line::Reward) line is anybody's: it belongs to
/// the agent rewarded and is recognized by its `type`.
///
/// A control of the moderator's is neither. It says nothing about the game
/// and carries no number to keep count of, so the reader skips it like any
/// other agent's record.
enum Line<'a> {
    /// One of the moderator's message records.
    Message {
        /// The record.
        record: &'a Map<String, Value>,
        /// Which way its message went.
        direction: Direction,
    },
    /// A reward: the agent paid and what it was paid.
    Reward {
        /// The agent rewarded.
        agent: ActorId,
        /// What its game was worth.
        value: i32,
    },
}

/// One message record of the moderator's, with its envelope decoded.
struct Record {
    line: usize,
    direction: Direction,
    sender: ActorId,
    /// Which of `sender`'s messages it is: the second half of the key a
    /// reader joins an observation to its action by (ADR-0017).
    seq: u64,
    recipients: BTreeSet<ActorId>,
    message: Message,
}

/// What one phase settled, as ADR-0011 guarantees it to be reproducible.
///
/// A projection of a [`PhaseRecord`] onto its outcome alone: who died and
/// what each seer found. The selections that led there, and their order, are
/// not part of it, because with timed sessions they are not reproducible.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdict {
    /// Who was eliminated and how, or `None` for a night nobody died in
    /// and a day nobody was lynched in.
    pub eliminated: Option<(ActorId, Role, Cause)>,
    /// What each seer learned this phase.
    pub investigations: BTreeMap<ActorId, (ActorId, Faction)>,
}

/// Everything a run of one seed must reproduce (ADR-0011).
///
/// A game played with timed sessions is reproducible in its *outcomes* and
/// not in its traffic: every death, every finding, the winner and the rewards
/// are the same on every run of a seed, while the order of selections, and
/// which late selections arrive before a session closes, are not. This is the
/// part the determinism tests compare.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Verdicts {
    /// Each player's role, as the moderator dealt it.
    pub assignment: BTreeMap<ActorId, Role>,
    /// Each phase in order, night before day.
    pub phases: Vec<Verdict>,
    /// How the game ended.
    pub outcome: Outcome,
    /// What each player's game was worth.
    pub rewards: BTreeMap<ActorId, i32>,
}

impl Transcript {
    /// The game projected onto what ADR-0011 guarantees to be
    /// reproducible: the deal, each phase's elimination and findings, the
    /// outcome and the rewards.
    ///
    /// Two runs of one seed agree on this and need not agree on anything
    /// else, so it is what the determinism tests compare. Comparing whole
    /// transcripts would compare the order selections arrived in, which is a
    /// fact about thread scheduling rather than about the game.
    #[must_use]
    pub fn verdicts(&self) -> Verdicts {
        let verdict = |record: &PhaseRecord| Verdict {
            eliminated: record.eliminated.clone(),
            investigations: record.investigations.clone(),
        };
        Verdicts {
            assignment: self.assignment.clone(),
            phases: self
                .rounds
                .iter()
                .flat_map(|round| {
                    [Some(&round.night), round.day.as_ref()]
                        .into_iter()
                        .flatten()
                        .map(verdict)
                })
                .collect(),
            outcome: self.outcome.clone(),
            rewards: self.rewards.clone(),
        }
    }

    /// Reconstructs the game from the moderator's records in the log.
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
    pub fn read(lines: &[Value], moderator: &ActorId) -> Result<Self, TranscriptError> {
        let mut reader = Reader::default();
        // Every number the moderator's own records carry, and the line each
        // was found on. It is a **set** and not a running count, because
        // nothing about a log's line order is promised (ADR-0017) and a
        // reminder in particular is logged when it arrives, which is after
        // actions the moderator decided later. What is promised is that the
        // numbers are dense over everything the moderator's handler yielded,
        // which is checked once, at the end.
        let mut mine: BTreeMap<u64, usize> = BTreeMap::new();
        let mut folding = Vec::new();
        let mut rewards = BTreeMap::new();
        for (index, value) in lines.iter().enumerate() {
            let line = index + 1;
            match read_line(value, line, moderator)? {
                None => {}
                // A reward belongs to the agent it names, whoever wrote it.
                Some(Line::Reward { agent, value }) => {
                    rewards.insert(agent, value);
                }
                Some(Line::Message { record, direction }) => {
                    let record = message(record, direction, line)?;
                    // A relay is one of the moderator's own messages and is
                    // counted like a narration, and so is a reminder, which
                    // is the moderator's own message to itself (ADR-0016). An
                    // observation of a *player's* selection carries the
                    // sender's number, which is nothing to count here; the
                    // sender is what tells the two apart.
                    if record.sender == *moderator {
                        mine.insert(record.seq, line);
                    }
                    folding.push(record);
                }
            }
        }
        for record in folding {
            reader.fold(record)?;
        }
        // The numbers last, after the game itself has been read. A file
        // missing one of the moderator's messages is missing whatever that
        // message said, so the fold usually fails first, on something
        // downstream of the hole; where the fold survives the loss — a
        // narration nothing else depends on — this is what catches it.
        //
        // After the outcome, because a truncated file is missing its last
        // numbers as well, and "this is not a whole game" is the more useful
        // thing to say about one than "number 56 is missing".
        let outcome = reader.outcome.ok_or(TranscriptError::NoOutcome)?;
        check_the_numbers(&mine)?;
        Ok(Self {
            assignment: reader.assignment,
            rounds: reader.rounds,
            outcome,
            rewards,
        })
    }
}

/// The moderator's numbers are dense from zero.
///
/// `mine` is every number one of the moderator's records carried, against the
/// line it was found on. The check is a set's and not a running count's: line
/// order in a log carries no meaning (ADR-0017), and a reminder in particular
/// reaches the writer when it arrives rather than when the number was taken,
/// so the numbers are not in file order.
///
/// # Errors
///
/// [`TranscriptError::SeqGap`] naming the line of the first number that is
/// not where it should be, which is the line a reader can actually look at.
fn check_the_numbers(mine: &BTreeMap<u64, usize>) -> Result<(), TranscriptError> {
    for (expected, (found, line)) in mine.iter().enumerate() {
        let expected = expected as u64;
        if *found != expected {
            return Err(TranscriptError::SeqGap {
                line: *line,
                expected,
                found: *found,
            });
        }
    }
    Ok(())
}

/// The record on one line, if the reader has any use for it; `None` for a
/// cycle record or another agent's, and an error for something that is not
/// a record at all.
///
/// Every line is checked to be a record of a kind the runtime writes, even
/// the ones nothing is read from, so that a file with something else in it
/// is not silently read as a game.
///
/// A reward is the one kind read whoever wrote it.
fn read_line<'a>(
    value: &'a Value,
    line: usize,
    moderator: &ActorId,
) -> Result<Option<Line<'a>>, TranscriptError> {
    let record = value
        .as_object()
        .ok_or(TranscriptError::NotAnObject { line })?;
    let direction = match record.get("type").and_then(Value::as_str) {
        Some("action") => Direction::Sent,
        Some("observation") => Direction::Received,
        // A message that went nowhere because its actor had been stopped
        // (ADR-0016). Nothing happened in the game, but the moderator's
        // number was taken, so the reader reads it to count it and folds
        // nothing.
        Some("undelivered" | "unsent") => Direction::Lost,
        Some("reward") => {
            let agent: ActorId = decode(record, "agent", line)?;
            let value = record
                .get("value")
                .and_then(Value::as_i64)
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| malformed(line, "no value"))?;
            return Ok(Some(Line::Reward { agent, value }));
        }
        // Nothing the reader has any use for. A control is out-of-domain
        // and says nothing about the game; a cycle is the loop's own
        // bookkeeping; the header anchors the episode to the wall clock.
        // None of them carries a sequence number to keep count of either.
        Some("control" | "cycle" | "episode") => return Ok(None),
        found => {
            return Err(TranscriptError::UnknownRecordType {
                line,
                found: found.map(str::to_owned),
            });
        }
    };
    let agent = string(record, "agent", line)?;
    Ok((agent == moderator.as_str()).then_some(Line::Message { record, direction }))
}

/// Decodes a message record's wire shape, its key and its payload.
///
/// The sender is the message's own, which is now always the agent that wrote
/// the record: nothing an actor sends claims another (ADR-0017). Whose
/// selection a relay passes on is inside the payload, in the
/// [`Envelope`](crate::Envelope) the `Relayed` variant carries, and the fold
/// reads it from there. The sequence number is beside the message at the top
/// level, where the record carries it once.
fn message(
    record: &Map<String, Value>,
    direction: Direction,
    line: usize,
) -> Result<Record, TranscriptError> {
    let seq = integer(record, "seq", line)?;
    let envelope = record
        .get("message")
        .and_then(Value::as_object)
        .ok_or_else(|| malformed(line, "no message"))?;
    let sender: ActorId = decode(envelope, "sender", line)?;
    let recipients: BTreeSet<ActorId> = decode(envelope, "recipients", line)?;
    let payload = envelope
        .get("payload")
        .ok_or_else(|| malformed(line, "no payload"))?;
    let message = Message::deserialize(payload)
        .map_err(|source| TranscriptError::Payload { line, source })?;
    Ok(Record {
        line,
        direction,
        sender,
        seq,
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
    recipients: BTreeSet<ActorId>,
    what: &str,
) -> Result<ActorId, TranscriptError> {
    let mut recipients = recipients.into_iter();
    match (recipients.next(), recipients.next()) {
        (Some(who), None) => Ok(who),
        _ => Err(malformed(line, what)),
    }
}

/// The game so far, as a fold over the moderator's records.
#[derive(Default)]
struct Reader {
    assignment: BTreeMap<ActorId, Role>,
    rounds: Vec<RoundRecord>,
    /// The player that made the latest nomination the moderator passed
    /// on, and which is therefore still a candidate for the hammer.
    ///
    /// The moderator relays each nomination it accepts as it accepts it, and
    /// the selection that completes a majority is relayed before the lynching
    /// it causes is narrated. So the player named in the envelope of the last
    /// relay before an `Eliminated { cause: Lynched }` is the one whose
    /// selection ended the day, and no narration has to say so (ADR-0015).
    /// Cleared when a phase begins, so that a lynching can never take its
    /// hammer from the day before.
    latest_nomination: Option<ActorId>,
    /// How the game ended, once the moderator has announced it. `None`
    /// until then, and still `None` at the end of a truncated transcript,
    /// which is [`TranscriptError::NoOutcome`].
    outcome: Option<Outcome>,
}

impl Reader {
    fn fold(&mut self, record: Record) -> Result<(), TranscriptError> {
        let Record {
            line,
            direction,
            sender,
            seq: _,
            recipients,
            message,
        } = record;
        // A message that went nowhere is not something that happened in the
        // game. It has already been counted, which is all a reader wants of
        // it.
        if direction == Direction::Lost {
            return Ok(());
        }
        match (direction, message) {
            (Direction::Sent, Message::Narration(narration)) => {
                self.narrated(line, recipients, narration)
            }
            (Direction::Received, Message::Select(selection)) => {
                self.answered(line, sender, selection)
            }
            // A reminder is the moderator's own note to itself, which says
            // nothing about the game: what it prompted the moderator to do is
            // whatever the game said when the reminder arrived, and that is
            // already in the moderator's actions. It is read only to be
            // counted, which `read` has done by the time the fold sees it.
            (Direction::Received, Message::Reminder(_)) => Ok(()),
            // A relay is the moderator passing a selection on to the players
            // who should see it (ADR-0018). The envelope names the player who
            // made it and which of that player's messages it was, which is
            // how this record joins back to the player's own action; here it
            // is the player that matters. It is the same selection this
            // reader already folded when the moderator received it, so
            // folding it again would count one vote as two. What it does say,
            // and the received selection does not, is that the moderator
            // accepted it: that is where the hammer comes from (ADR-0015).
            (Direction::Sent, Message::Relayed(envelope)) => {
                if envelope.payload.kind == SessionKind::Nominate {
                    self.latest_nomination = Some(envelope.from);
                }
                Ok(())
            }
            _ => Err(TranscriptError::Misdirected { line }),
        }
    }

    fn narrated(
        &mut self,
        line: usize,
        recipients: BTreeSet<ActorId>,
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
            } => {
                self.latest_nomination = None;
                self.rounds.push(RoundRecord {
                    round,
                    night: PhaseRecord::begun(living),
                    day: None,
                });
            }
            Narration::PhaseBegan {
                round,
                phase: Phase::Day,
                living,
            } => {
                self.latest_nomination = None;
                match self.rounds.last_mut() {
                    Some(latest) if latest.round == round && latest.day.is_none() => {
                        latest.day = Some(PhaseRecord::begun(living));
                    }
                    _ => return Err(TranscriptError::NoPhase { line }),
                }
            }
            Narration::Investigated { target, faction } => {
                let seer = only(line, recipients, "a finding is addressed to one seer")?;
                self.current(line)?
                    .investigations
                    .insert(seer, (target, faction));
            }
            Narration::Eliminated {
                who, role, cause, ..
            } => {
                // The nomination that lynched somebody is the last one
                // the moderator passed on before saying so (ADR-0015).
                let hammer = (cause == Cause::Lynched)
                    .then(|| self.latest_nomination.clone())
                    .flatten();
                let phase = self.current(line)?;
                phase.eliminated = Some((who, role, cause));
                phase.hammer = hammer;
            }
            Narration::NoLynch { .. } => self.current(line)?.no_lynch = true,
            // A night without a death is one without an elimination.
            Narration::NoDeath { .. } => {}
            Narration::Outcome(outcome) => self.outcome = Some(outcome),
        }
        Ok(())
    }

    fn answered(
        &mut self,
        line: usize,
        from: ActorId,
        selection: Select,
    ) -> Result<(), TranscriptError> {
        // The selection says which session it was made in, so there is
        // nothing to look up (ADR-0014). What has to be checked is that
        // its sender's role is asked that in that phase: a reader learns
        // the roles from the deal, which precedes every selection.
        let role = self
            .assignment
            .get(&from)
            .copied()
            .ok_or(TranscriptError::NotAMember {
                line,
                from: from.clone(),
                kind: selection.kind,
            })?;
        if role.asked_in(selection.kind.phase()) != Some(selection.kind) {
            return Err(TranscriptError::NotAMember {
                line,
                from,
                kind: selection.kind,
            });
        }
        self.current(line)?
            .moves
            .insert(from, (selection.kind, selection.target));
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

/// How many cells a roster or a phase's moves lay out per line.
const COLUMNS: usize = 4;

impl fmt::Display for Transcript {
    /// The game at a glance: the roster with its roles, then each phase with
    /// its moves and its elimination, then the outcome, then what each
    /// player's game was worth. Only what the log records, in the
    /// order it records it. Ends with a newline.
    ///
    /// The rewards come last because they are the game's verdict on the
    /// players, which only the outcome above them explains. A game read
    /// from a log with no reward records renders without the
    /// section rather than with an empty one.
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
            let number = round.round.number();
            phase(f, &format!("Night {number}"), &round.night, width)?;
            if let Some(day) = &round.day {
                phase(f, &format!("Day {number}"), day, width)?;
            }
        }
        writeln!(f)?;
        let rounds = self.outcome.rounds.number();
        match self.outcome.winner {
            Some(Faction::Village) => write!(f, "Village wins")?,
            Some(Faction::Werewolves) => write!(f, "Werewolves win")?,
            // A game that reached the day cap without a winner.
            None => write!(f, "Stalemate")?,
        }
        let plural = if rounds == 1 { "" } else { "s" };
        write!(f, " after {rounds} round{plural}.  Survivors: ")?;
        let survivors: Vec<&str> = self.outcome.living.iter().map(ActorId::as_str).collect();
        if survivors.is_empty() {
            writeln!(f, "none")?;
        } else {
            writeln!(f, "{}", survivors.join(", "))?;
        }
        if self.rewards.is_empty() {
            return Ok(());
        }
        writeln!(f)?;
        writeln!(f, "Rewards")?;
        columns(
            f,
            self.rewards
                .iter()
                .map(|(who, value)| format!("{:<width$} {value:>+2}", who.as_str())),
        )
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
        .partition(|(_, (kind, _))| *kind == SessionKind::Nominate);
    // Nominations are a ballot, laid out in columns; night moves differ
    // by kind and get a line each.
    columns(
        f,
        votes.iter().map(|(who, (_, chosen))| {
            format!("{:<width$} -> {:<width$}", who.as_str(), chosen.as_str())
        }),
    )?;
    for (who, (kind, chosen)) in deeds {
        let verb = match kind {
            SessionKind::Devour => "devours",
            SessionKind::Investigate => "investigates",
            SessionKind::Protect => "protects",
            SessionKind::Nominate => "nominates",
        };
        let whom = chosen.as_str();
        write!(f, "  {:<width$} {verb} {whom}", who.as_str())?;
        match record.investigations.get(who) {
            Some((_, faction)) => writeln!(f, "  ->  {faction}")?,
            None => writeln!(f)?,
        }
    }
    match (&record.eliminated, record.no_lynch) {
        (Some((who, role, Cause::Devoured)), _) => writeln!(f, "  {who} is devoured   ({role})"),
        (Some((who, role, Cause::Lynched)), _) => match &record.hammer {
            Some(hammer) => writeln!(f, "  {who} is lynched   ({role}; hammer: {hammer})"),
            None => writeln!(f, "  {who} is lynched   ({role})"),
        },
        // A day that ran out of time is not a night that nobody died in.
        (None, true) => writeln!(f, "  no one was lynched"),
        (None, false) => writeln!(f, "  no one died"),
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
    use std::time::{Duration, Instant};

    use crate::testing::{fast, id, ids, village};
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

    fn moderator() -> ActorId {
        id(MODERATOR)
    }

    fn fixture() -> Vec<Value> {
        lines(FIXTURE).unwrap()
    }

    fn read(lines: &[Value]) -> Result<Transcript, TranscriptError> {
        Transcript::read(lines, &moderator())
    }

    fn target(who: &str) -> ActorId {
        id(who)
    }

    fn moves<const N: usize>(
        moves: [(&str, SessionKind, ActorId); N],
    ) -> BTreeMap<ActorId, (SessionKind, ActorId)> {
        moves
            .into_iter()
            .map(|(who, kind, chosen)| (id(who), (kind, chosen)))
            .collect()
    }

    fn nominations<const N: usize>(
        votes: [(&str, &str); N],
    ) -> BTreeMap<ActorId, (SessionKind, ActorId)> {
        votes
            .into_iter()
            .map(|(who, whom)| (id(who), (SessionKind::Nominate, target(whom))))
            .collect()
    }

    fn phase<const N: usize>(
        living: [&str; N],
        moves: BTreeMap<ActorId, (SessionKind, ActorId)>,
        investigation: Option<(&str, &str, Faction)>,
        eliminated: Option<(&str, Role, Cause)>,
    ) -> PhaseRecord {
        PhaseRecord {
            no_lynch: false,
            hammer: None,
            living: ids(living),
            moves,
            investigations: investigation
                .map(|(seer, whom, faction)| (id(seer), (id(whom), faction)))
                .into_iter()
                .collect(),
            eliminated: eliminated.map(|(who, role, cause)| (id(who), role, cause)),
        }
    }

    /// The game the fixture records, written out by hand from the
    /// moderator's records.
    fn expected() -> Transcript {
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
                expected_round_one(),
                expected_round_two(),
                expected_round_three(),
                expected_round_four(),
            ],
            outcome: Outcome {
                winner: Some(Faction::Werewolves),
                rounds: Round::new(4),
                living: ids(["bob", "dave", "erin", "grace"]),
            },
            // The werewolves dave and erin won; everybody else, living or
            // dead, lost with the village.
            rewards: [
                ("alice", -1),
                ("bob", -1),
                ("carol", -1),
                ("dave", 1),
                ("erin", 1),
                ("frank", -1),
                ("grace", -1),
            ]
            .into_iter()
            .map(|(who, value)| (id(who), value))
            .collect(),
        }
    }

    /// Everyone, for the rounds before anybody has died.
    const EVERYONE: [&str; 7] = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];

    /// The pack splits, the tie-break picks alice, and the doctor has
    /// protected her: a saved night. Then seven players scatter over five
    /// targets, so nobody reaches the four a majority of the living needs
    /// and the day runs out (ADR-0011).
    fn expected_round_one() -> RoundRecord {
        use SessionKind::{Devour, Investigate, Protect};
        RoundRecord {
            round: Round::new(1),
            night: phase(
                EVERYONE,
                moves([
                    ("carol", Protect, target("alice")),
                    ("dave", Devour, target("alice")),
                    ("erin", Devour, target("bob")),
                    ("grace", Investigate, target("alice")),
                ]),
                Some(("grace", "alice", Faction::Village)),
                None,
            ),
            day: Some(no_lynch(phase(
                EVERYONE,
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
                None,
            ))),
        }
    }

    /// The pack agrees on carol, whom the doctor did not protect, and the
    /// seer finds a werewolf too late to say so.
    fn expected_round_two() -> RoundRecord {
        use Cause::Devoured;
        use SessionKind::{Devour, Investigate, Protect};
        RoundRecord {
            round: Round::new(2),
            night: phase(
                EVERYONE,
                moves([
                    ("carol", Protect, target("erin")),
                    ("dave", Devour, target("carol")),
                    ("erin", Devour, target("bob")),
                    ("grace", Investigate, target("dave")),
                ]),
                Some(("grace", "dave", Faction::Werewolves)),
                Some(("carol", Doctor, Devoured)),
            ),
            day: Some(no_lynch(phase(
                ["alice", "bob", "dave", "erin", "frank", "grace"],
                nominations([
                    ("alice", "grace"),
                    ("bob", "grace"),
                    ("dave", "alice"),
                    ("erin", "bob"),
                    ("frank", "grace"),
                    ("grace", "erin"),
                ]),
                None,
                None,
            ))),
        }
    }

    /// No doctor lives, so no protection session opens and nothing stands
    /// between the pack and the player it agrees on.
    fn expected_round_three() -> RoundRecord {
        use Cause::Devoured;
        use SessionKind::{Devour, Investigate};
        RoundRecord {
            round: Round::new(3),
            night: phase(
                ["alice", "bob", "dave", "erin", "frank", "grace"],
                moves([
                    ("dave", Devour, target("frank")),
                    ("erin", Devour, target("frank")),
                    ("grace", Investigate, target("frank")),
                ]),
                Some(("grace", "frank", Faction::Village)),
                Some(("frank", Villager, Devoured)),
            ),
            day: Some(no_lynch(phase(
                ["alice", "bob", "dave", "erin", "grace"],
                nominations([
                    ("alice", "erin"),
                    ("bob", "dave"),
                    ("dave", "alice"),
                    ("erin", "alice"),
                    ("grace", "dave"),
                ]),
                None,
                None,
            ))),
        }
    }

    /// The pack splits again, the tie-break picks alice, and two
    /// werewolves among four living is parity: the game ends at night, so
    /// the round has no day.
    fn expected_round_four() -> RoundRecord {
        use Cause::Devoured;
        use SessionKind::{Devour, Investigate};
        RoundRecord {
            round: Round::new(4),
            night: phase(
                ["alice", "bob", "dave", "erin", "grace"],
                moves([
                    ("dave", Devour, target("alice")),
                    ("erin", Devour, target("grace")),
                    ("grace", Investigate, target("bob")),
                ]),
                Some(("grace", "bob", Faction::Village)),
                Some(("alice", Villager, Devoured)),
            ),
            day: None,
        }
    }

    /// A day that reached its limit without a majority.
    fn no_lynch(mut record: PhaseRecord) -> PhaseRecord {
        record.no_lynch = true;
        record
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
        assert_eq!(transcript.rewards, expected.rewards);
    }

    #[test]
    fn every_player_has_a_reward_and_it_agrees_with_its_faction() {
        // Read from the `reward` records, which belong to the players and
        // are written by the moderator, so a reader that looked only at
        // the moderator's records would find none of them.
        let transcript = read(&fixture()).unwrap();
        assert_eq!(
            transcript.rewards.keys().collect::<BTreeSet<_>>(),
            transcript.assignment.keys().collect::<BTreeSet<_>>(),
            "every player is paid, and nobody else"
        );
        for (who, value) in &transcript.rewards {
            let expected =
                if transcript.outcome.winner == Some(transcript.assignment[who].faction()) {
                    1
                } else {
                    -1
                };
            assert_eq!(*value, expected, "{who}");
        }
        assert!(
            !transcript.rewards.contains_key(&moderator()),
            "the moderator plays no game and is paid nothing"
        );
    }

    #[test]
    fn a_reward_is_read_whoever_it_belongs_to_and_is_never_counted() {
        // Two claims at once. Every reward in the fixture belongs to a
        // player, not to the moderator, and they are all read: that is
        // what distinguishes a reward from every other record here, which
        // is read only if the moderator wrote it.
        let lines = fixture();
        let rewards: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .collect();
        assert!(!rewards.is_empty());
        assert!(
            rewards.iter().all(|line| line["agent"] != MODERATOR),
            "the fixture's rewards belong to the players"
        );
        assert_eq!(read(&lines).unwrap().rewards.len(), rewards.len());

        // And a reward carries no sequence number, so it is outside the
        // contiguity check: one relabeled to the moderator is read as a
        // reward for the moderator and does not make its numbering look
        // full of gaps.
        let mut moved = fixture();
        let index = moved
            .iter()
            .position(|line| line["type"] == "reward")
            .unwrap();
        moved[index]["agent"] = json!(MODERATOR);
        let transcript = read(&moved).unwrap();
        assert_eq!(
            transcript.rewards.get(&moderator()),
            Some(&-1),
            "the reward is read, and belongs to whoever it names"
        );
        assert_eq!(
            transcript.rounds,
            read(&fixture()).unwrap().rounds,
            "and the game itself is untouched"
        );
    }

    #[test]
    fn a_reward_without_a_value_or_an_agent_is_an_error() {
        let index = fixture()
            .iter()
            .position(|line| line["type"] == "reward")
            .unwrap();
        for key in ["agent", "value"] {
            let mut lines = fixture();
            lines[index].as_object_mut().unwrap().remove(key);
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what.starts_with(&format!("no {key}"))),
                "{key}: {error:?}"
            );
        }
        // A value too large for the reward type is malformed, not silently
        // truncated.
        let mut lines = fixture();
        lines[index]["value"] = json!(i64::from(i32::MAX) + 1);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Malformed { what, .. } if what == "no value"),
            "{error:?}"
        );
    }

    #[test]
    fn a_game_whose_log_records_no_reward_reads_and_renders_without_them() {
        // Nothing in the reader requires a reward: a truncated or
        // hand-written log that has none is still the game it
        // records, and renders with no rewards section rather than an
        // empty one.
        let lines: Vec<Value> = fixture()
            .into_iter()
            .filter(|line| line["type"] != "reward")
            .collect();
        let transcript = read(&lines).unwrap();
        assert!(transcript.rewards.is_empty());
        let rendered = transcript.to_string();
        assert!(!rendered.contains("Rewards"), "{rendered}");
        assert!(
            rendered.ends_with("Survivors: bob, dave, erin, grace\n"),
            "{rendered}"
        );
        // And is otherwise the same game.
        assert_eq!(transcript.rounds, read(&fixture()).unwrap().rounds);
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
                    Werewolf => SessionKind::Devour,
                    Seer => SessionKind::Investigate,
                    Doctor => SessionKind::Protect,
                    Villager => panic!("{who} acted at night"),
                };
                assert_eq!(*kind, expected, "{who} in round {:?}", round.round);
            }
            // The last round ends at night, so it has no day.
            for (who, (kind, _)) in round.day.iter().flat_map(|day| &day.moves) {
                assert_eq!(
                    *kind,
                    SessionKind::Nominate,
                    "{who} in round {:?}",
                    round.round
                );
            }
        }
    }

    /// The fixture with every time rewritten and the agents' records
    /// interleaved differently: the same game as a different file.
    fn rewritten() -> Vec<Value> {
        let mut lines = fixture();
        for (offset, line) in lines.iter_mut().enumerate() {
            for key in ["t", "t_start", "t_stop", "start_unix_ns"] {
                if let Some(time) = line.get_mut(key) {
                    *time = json!(1_000_000 + offset);
                }
            }
        }
        // A stable sort by agent keeps each agent's records in order, which
        // is the one ordering the runtime guarantees, and otherwise changes
        // the interleaving completely. The header has no agent, so it sorts
        // to the front, where it belongs.
        lines.sort_by_key(|line| line["agent"].as_str().unwrap_or("").to_owned());
        assert_ne!(lines, fixture());
        lines
    }

    #[test]
    fn every_observation_joins_the_action_it_came_from_by_from_and_seq() {
        // What replaced the join by sender and creation time (ADR-0017). The
        // moderator's `observation` records are the selections it took in,
        // and each names the player that sent it and which of that player's
        // messages it was; the player's own `action` record carries the same
        // pair, and nothing else links the two.
        //
        // Every action is its own agent's message, relays included, so the
        // join is total on the sending side, with one exception on the
        // receiving side: a **reminder** joins no action, because it is the
        // actor's own message to itself and is logged where it arrives rather
        // than where it was set (ADR-0016). It is recognized by its sender
        // being its own recipient, which is the one shape no other message
        // has.
        let lines = fixture();
        let sent: BTreeSet<(String, u64)> = lines
            .iter()
            .filter(|line| line["type"] == "action")
            .map(|line| {
                let agent = line["agent"].as_str().expect("a record names its agent");
                assert_eq!(
                    line["message"]["sender"], *agent,
                    "an action is a message of its agent's: {line}"
                );
                (
                    agent.to_owned(),
                    line["seq"].as_u64().expect("a message record is numbered"),
                )
            })
            .collect();
        let observations: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "observation")
            .collect();
        assert!(!observations.is_empty(), "the fixture has observations");
        let mut reminders = 0;
        for line in &observations {
            if line["message"]["recipients"] == json!([line["message"]["sender"]]) {
                reminders += 1;
                assert!(
                    !line["message"]["payload"]["Reminder"].is_null(),
                    "the only message an actor sends itself is a reminder: {line}"
                );
                continue;
            }
            let key = (
                line["from"]
                    .as_str()
                    .expect("an observation names its sender")
                    .to_owned(),
                line["seq"].as_u64().expect("a message record is numbered"),
            );
            assert!(
                sent.contains(&key),
                "an observation joins the action it came from by (from, seq): {line}"
            );
        }
        assert!(
            reminders > 0,
            "the fixture has the moderator's reminders, or this exemption is untested"
        );
    }

    #[test]
    fn a_relay_is_the_moderators_action_and_reaches_the_player_through_its_envelope() {
        // Scenario 10, inverted (ADR-0017). A forwarded selection used to be
        // recorded under the player's own name and number, so the relay was
        // invisible and a listener's observation joined straight to the
        // player's action. Now the relay is the **moderator's** action, under
        // a number of the moderator's, and the envelope inside it names the
        // player and the player's number. Following one relayed selection
        // therefore takes both keys, and this walks both.
        let lines = fixture();
        let moderator = json!(MODERATOR);
        // One pass over the file, so the walk below is lookups rather than
        // scans: every action by the key an observation of it names, the
        // relays among them, the keys of the selections the moderator took
        // in, and each observation of a message of the moderator's by its
        // number.
        let mut sent: BTreeMap<(&str, u64), &Value> = BTreeMap::new();
        let mut relays: Vec<&Value> = Vec::new();
        let mut heard: BTreeSet<(&str, u64)> = BTreeSet::new();
        let mut listened: BTreeMap<u64, Vec<&Value>> = BTreeMap::new();
        for line in &lines {
            let seq = || line["seq"].as_u64().expect("a message record is numbered");
            match line["type"].as_str() {
                Some("action") => {
                    let agent = line["agent"].as_str().expect("a record names its agent");
                    sent.insert((agent, seq()), line);
                    if !line["message"]["payload"]["Relayed"].is_null() {
                        relays.push(line);
                    }
                }
                Some("observation") => {
                    let from = line["from"]
                        .as_str()
                        .expect("an observation names its sender");
                    if !line["message"]["payload"]["Select"].is_null() {
                        heard.insert((from, seq()));
                    }
                    if line["from"] == moderator {
                        listened.entry(seq()).or_default().push(line);
                    }
                }
                _ => {}
            }
        }
        assert!(
            !relays.is_empty(),
            "the fixture exercises relaying, or this proves nothing"
        );
        let mut followed = 0;
        for relay in &relays {
            assert_eq!(
                relay["agent"], moderator,
                "a relay is the moderator's action: {relay}"
            );
            assert_eq!(
                relay["message"]["sender"], moderator,
                "with the moderator as its sender: {relay}"
            );
            let envelope = &relay["message"]["payload"]["Relayed"]["envelope"];
            let origin = (
                envelope["from"]
                    .as_str()
                    .expect("an envelope names a player"),
                envelope["seq"].as_u64().expect("and one of its messages"),
            );
            // The envelope names the player and the player's number, so the
            // player's own action record is there to be found, and it is a
            // selection the moderator really took in.
            let original = sent
                .get(&origin)
                .unwrap_or_else(|| panic!("a relay's envelope names a real action: {relay}"));
            assert_eq!(
                original["message"]["payload"]["Select"], envelope["payload"],
                "and the relay carries what the player selected: {relay} against {original}"
            );
            assert!(
                heard.contains(&origin),
                "and the moderator heard it from the player: {relay}"
            );

            // And the walk starts from any listener too: a listener's
            // observation names `(moderator, seq)`, which is how `listened`
            // keyed it and which found this relay, and the envelope it
            // carries is this one — so the envelope's key finds the player's
            // action from the listener's side as well.
            for observation in listened
                .get(&relay["seq"].as_u64().unwrap())
                .into_iter()
                .flatten()
            {
                assert_eq!(
                    &observation["message"]["payload"]["Relayed"]["envelope"], envelope,
                    "a listener sees the envelope the moderator sent: {observation}"
                );
                followed += 1;
            }
        }
        assert!(
            followed > 0,
            "some listener observed a relay, or the walk was never taken"
        );
    }

    #[test]
    fn a_transcript_has_no_times_and_no_interleaving() {
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

    /// The index of the first of the moderator's records that carries a
    /// message whose payload satisfies `matching`.
    fn moderator_record(lines: &[Value], matching: impl Fn(&Value) -> bool) -> usize {
        lines
            .iter()
            .position(|line| {
                line["agent"] == MODERATOR
                    && (line["type"] == "action" || line["type"] == "observation")
                    && matching(&line["message"]["payload"])
            })
            .expect("the fixture has such a record")
    }

    fn is_selection(payload: &Value) -> bool {
        !payload["Select"].is_null()
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
        let index = moderator_record(&fixture(), is_selection);
        for key in ["agent", "seq", "message"] {
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
            lines[index]["message"].as_object_mut().unwrap().remove(key);
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
        // The moderator's own start and stop are in the file, carry no
        // sequence number of their own, and the reader steps over them
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
        let index = moderator_record(&lines, is_selection);
        lines[index]["message"]["payload"] = json!({"Response": {"request": "seven"}});
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
        // The moderator's own messages are numbered densely from zero, so a
        // gap means the file is missing one. Only its own: a narration it
        // sent, or a reminder that came back to it, and not an observation of
        // somebody's selection, whose number is that player's.
        //
        // What is checked is a **set** of numbers rather than a running
        // count, since a log's line order carries no meaning (ADR-0017): a
        // reminder reaches the writer when it arrives and not when its number
        // was taken. So the error names the line where the first number out
        // of place was found, which is the line after the hole.
        //
        // The forgery drops one of the moderator's `NoLynch` narrations,
        // which is a record nothing else in the fold depends on: the game
        // still reads as a game, and the hole its number leaves is the only
        // thing wrong with the file.
        let mut lines = fixture();
        let index = moderator_record(&lines, |payload| !payload["Narration"]["NoLynch"].is_null());
        let missing = lines[index]["seq"].as_u64().expect("a numbered record");
        lines.remove(index);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::SeqGap { expected, found, .. }
                if *expected == missing && *found == missing + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("sequence number"), "{error}");
    }

    #[test]
    fn a_selection_in_a_session_the_senders_role_is_not_in_is_an_error() {
        // There is no id to invent an unknown value for any more
        // (ADR-0014): a selection says which session it was made in, and
        // what the reader checks is that the sender's role puts it in
        // that session. So the forgery is a kind the sender's own role
        // is never asked — which one that is depends on the role the
        // fixture dealt, so it is derived rather than written in.
        let mut lines = fixture();
        let index = moderator_record(&lines, is_selection);
        let kind: SessionKind =
            serde_json::from_value(lines[index]["message"]["payload"]["Select"]["kind"].clone())
                .expect("a selection names its session");
        let forged = [
            SessionKind::Devour,
            SessionKind::Investigate,
            SessionKind::Protect,
        ]
        .into_iter()
        .find(|other| *other != kind)
        .expect("a night has three kinds and a role is asked one");
        lines[index]["message"]["payload"]["Select"]["kind"] = json!(forged);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::NotAMember { line, kind, .. }
                if *line == index + 1 && *kind == forged),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(&format!("{forged:?} session")),
            "{error}"
        );
    }

    #[test]
    fn a_selection_from_someone_the_phase_asks_nothing_of_is_an_error() {
        // frank is a villager, and a villager is a member of no night
        // session at all.
        let mut lines = fixture();
        let index = moderator_record(&lines, is_selection);
        lines[index]["message"]["sender"] = json!("frank");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::NotAMember { from, .. } if from.as_str() == "frank"),
            "{error:?}"
        );
    }

    #[test]
    fn a_selection_from_someone_not_in_the_deal_is_an_error() {
        // A reader learns the roles from the deal, which precedes every
        // selection. A sender the deal never named has no role to check.
        let mut lines = fixture();
        let index = moderator_record(&lines, is_selection);
        lines[index]["message"]["sender"] = json!("zara");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::NotAMember { from, .. } if from.as_str() == "zara"),
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
    fn an_empty_actor_id_is_an_error_wherever_it_appears() {
        // `ActorId` refuses to deserialize from the empty string, so an
        // empty id in the envelope is a malformed envelope and one in the
        // payload is a payload that is not a Werewolf message, each naming
        // the line.
        let index = moderator_record(&fixture(), is_selection);
        for (key, empty) in [("sender", json!("")), ("recipients", json!([""]))] {
            let mut lines = fixture();
            lines[index]["message"][key] = empty;
            let error = read(&lines).unwrap_err();
            assert!(
                matches!(&error, TranscriptError::Malformed { line, what }
                    if *line == index + 1 && what.starts_with(key)),
                "{key}: {error:?}"
            );
            assert!(error.to_string().contains("non-empty actor id"), "{error}");
        }
        let mut lines = fixture();
        lines[index]["message"]["payload"]["Select"]["target"] = json!("");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Payload { line, .. } if *line == index + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("non-empty actor id"), "{error}");
        // A narration's ids are checked too.
        let mut lines = fixture();
        let index = moderator_record(&lines, |payload| {
            !payload["Narration"]["Eliminated"].is_null()
        });
        lines[index]["message"]["payload"]["Narration"]["Eliminated"]["who"] = json!("");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(&error, TranscriptError::Payload { line, .. } if *line == index + 1),
            "{error:?}"
        );
    }

    #[test]
    fn a_transcript_holds_no_request_at_all() {
        // Nothing is asked of anybody (ADR-0014), so no record in a
        // transcript is a request, and the "addressed to one player"
        // check a request once needed has nothing left to guard. The
        // check itself lives on for a finding, which is still addressed
        // to one seer.
        let lines = fixture();
        assert!(
            !lines
                .iter()
                .any(|line| !line["message"]["payload"]["Request"].is_null()),
            "a transcript still holds a request"
        );
    }

    #[test]
    fn a_finding_addressed_to_other_than_one_seer_is_an_error() {
        for to in [json!([]), json!(["alice", "grace"])] {
            let mut lines = fixture();
            let index = moderator_record(&lines, |payload| {
                !payload["Narration"]["Investigated"].is_null()
            });
            lines[index]["message"]["recipients"] = to.clone();
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
        // first selection is made in no phase at all.
        let index = moderator_record(&lines, |payload| {
            !payload["Narration"]["PhaseBegan"].is_null()
        });
        lines[index]["message"]["payload"] = json!({"Narration": {"NoDeath": {"round": 1}}});
        let error = read(&lines).unwrap_err();
        let first_response = moderator_record(&lines, is_selection);
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
        lines[index]["message"]["payload"]["Narration"]["PhaseBegan"]["phase"] = json!("Day");
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NoPhase { line } if line == index + 1),
            "{error:?}"
        );
        // The first day announced as belonging to a round yet to come.
        let mut lines = fixture();
        let day = index + 1 + moderator_record(&lines[index + 1..], is_phase);
        lines[day]["message"]["payload"]["Narration"]["PhaseBegan"]["round"] = json!(2);
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::NoPhase { line } if line == day + 1),
            "{error:?}"
        );
    }

    #[test]
    fn a_message_the_moderator_never_records_is_an_error() {
        // A narration the moderator observed rather than sent. It only
        // ever sends one, so a narration among its observations is a
        // log that does not describe this game.
        let mut lines = fixture();
        let index = moderator_record(&lines, |payload| !payload["Narration"].is_null());
        lines[index]["type"] = json!("observation");
        // An observation names the sender at the top level as well.
        lines[index]["from"] = lines[index]["message"]["sender"].clone();
        let error = read(&lines).unwrap_err();
        assert!(
            matches!(error, TranscriptError::Misdirected { line } if line == index + 1),
            "{error:?}"
        );
        assert!(error.to_string().contains("never records"), "{error}");
    }

    #[test]
    fn a_selection_the_moderator_sent_is_one_it_relayed() {
        // The moderator both receives a selection and sends it on to the
        // players who should see it (ADR-0018), so a relay among its actions
        // is no error: the fixture contains both, and it reads. What a relay
        // looks like, and what it joins back to, is the next test's claim
        // rather than this one's.
        read(&fixture()).expect("a log with relayed selections reads");
    }

    /// Writes the moderator's records of a game played through [`Game`]
    /// with scripted responses: a log of exactly what the moderator
    /// would record, with nobody else's records and made-up stamps.
    struct Scribe {
        lines: Vec<Value>,
        /// The next sequence number for each sender, so that every message
        /// carries its own sender's, as the runtime numbers them.
        next: BTreeMap<String, u64>,
    }

    impl Scribe {
        fn new() -> Self {
            Self {
                lines: Vec::new(),
                next: BTreeMap::new(),
            }
        }

        /// The next number in `sender`'s sequence.
        fn next_seq(&mut self, sender: &str) -> u64 {
            let seq = self.next.entry(sender.to_owned()).or_default();
            let taken = *seq;
            *seq += 1;
            taken
        }

        /// One record of the moderator's, of the given type, carrying
        /// `sender`'s sequence number and a time invented from the line
        /// count.
        fn record(
            &mut self,
            kind: &str,
            sender: &str,
            seq: u64,
            recipients: &BTreeSet<ActorId>,
            payload: &Message,
        ) {
            let t = self.lines.len() * 10;
            let mut line = json!({
                "type": kind,
                "agent": MODERATOR,
                "t": t,
                "seq": seq,
                "message": {
                    "sender": sender,
                    "recipients": recipients,
                    "payload": payload,
                },
            });
            // An observation's `agent` is the receiver, so it names the
            // sender beside the number.
            if kind == "observation" {
                line["from"] = json!(sender);
            }
            self.lines.push(line);
        }

        fn directives(&mut self, directives: Vec<Directive>) {
            for directive in directives {
                let (to, payload) = match directive {
                    Directive::Narrate { to, narration } => (to, Message::Narration(narration)),
                    // A relay is the moderator's own message, numbered among
                    // the moderator's; the envelope inside it is what names
                    // the player and joins it back to the player's action.
                    Directive::Forward { envelope, to } => (to, Message::Relayed(envelope)),
                    // A stop is a control, and `Transcript::read` skips
                    // control records: they say nothing about the game.
                    Directive::Stop { .. } => continue,
                };
                let seq = self.next_seq(MODERATOR);
                self.record("action", MODERATOR, seq, &to, &payload);
            }
        }

        /// Records a selection arriving from `from`, and returns the number
        /// it was sent under, which the relay's envelope will carry.
        fn select(&mut self, from: &str, selection: Select) -> u64 {
            let seq = self.next_seq(from);
            self.record(
                "observation",
                from,
                seq,
                &ids([MODERATOR]),
                &Message::Select(selection),
            );
            seq
        }
    }

    /// Plays `script`, one phase's selections per entry, through a game over
    /// `assignment`, and returns the moderator's records.
    fn scripted(assignment: Assignment, script: &[Vec<(&str, ActorId)>]) -> Vec<Value> {
        /// Far enough apart that one phase's clocks never reach the next.
        const STEP: u64 = 10_000;

        let mut scribe = Scribe::new();
        let roles = assignment.clone();
        let mut game = Game::new(assignment, 1, fast());
        let origin = Instant::now();
        scribe.directives(game.begin(origin));
        for (index, answers) in script.iter().enumerate() {
            let now = origin + Duration::from_millis((index as u64 + 1) * STEP);
            // Taken before any selection is recorded: a day ends on the selection
            // that makes a majority, so selecting alone may finish it.
            let phase = game.phase_now();
            let (phase_kind, round) = phase;
            for (who, chosen) in answers {
                // Which session a selection belongs to is the selector's own
                // business (ADR-0014): the script stands in for the
                // player, so it reads it off the role the same way the
                // player would, rather than off a request.
                let kind = roles
                    .role(&id(who))
                    .and_then(|role| role.asked_in(phase_kind))
                    .expect("the phase asks something of this role");
                // The audience the player would name, so the fixture
                // exercises the moderator's forwarding the way a real game
                // does: a nomination is public, a devour is the pack's, and
                // the seer's and doctor's business is nobody else's.
                let seen_by = match kind {
                    SessionKind::Nominate => game
                        .living()
                        .iter()
                        .filter(|other| *other != &id(who))
                        .cloned()
                        .collect(),
                    SessionKind::Devour => roles
                        .players()
                        .filter(|(other, role)| {
                            role.faction() == Faction::Werewolves && *other != &id(who)
                        })
                        .map(|(other, _)| other.clone())
                        .filter(|other| game.living().contains(other))
                        .collect(),
                    SessionKind::Investigate | SessionKind::Protect => BTreeSet::new(),
                };
                let selection = Select {
                    round,
                    kind,
                    target: chosen.clone(),
                    seen_by,
                };
                let seq = scribe.select(who, selection.clone());
                scribe.directives(game.select(&id(who), &selection, seq, now));
            }
            // Close this phase and no more. Expiring at the earliest
            // deadline open, and stopping as soon as the phase moves,
            // keeps one pass from cascading through every later phase.
            while game.phase_now() == phase && game.outcome().is_none() {
                let Some(deadline) = game.next_deadline() else {
                    break;
                };
                let expired = game.expire(deadline);
                if expired.is_empty() {
                    break;
                }
                scribe.directives(expired);
            }
        }
        scribe.lines
    }

    fn answers<const N: usize>(answers: [(&'static str, &str); N]) -> Vec<(&'static str, ActorId)> {
        answers
            .into_iter()
            .map(|(who, whom)| (who, target(whom)))
            .collect()
    }

    /// A game where carol is devoured, erin is lynched on dave's selection,
    /// and alice is devoured while the doctor abstains: the werewolves win
    /// at parity on night two, so there is no second day.
    ///
    /// On the lynching: three of the four living select erin, so dave's is
    /// the hammer and the day closes before erin is ever asked.
    fn werewolves_win_at_parity() -> Vec<Value> {
        scripted(
            village(),
            &[
                answers([("bob", "carol"), ("carol", "bob"), ("dave", "alice")]),
                answers([("alice", "erin"), ("bob", "erin"), ("dave", "erin")]),
                answers([("bob", "alice"), ("dave", "bob")]),
            ],
        )
    }

    #[test]
    fn a_game_that_ends_at_night_has_no_final_day() {
        let lines = werewolves_win_at_parity();
        let transcript = read(&lines).unwrap();
        assert_eq!(transcript.rounds.len(), 2);
        let last = &transcript.rounds[1];
        assert_eq!(last.day, None);
        assert_eq!(
            last.night.moves,
            moves([
                ("bob", SessionKind::Devour, target("alice")),
                ("dave", SessionKind::Protect, target("bob")),
            ])
        );
        assert_eq!(
            last.night.eliminated,
            Some((id("alice"), Villager, Cause::Devoured))
        );
        assert_eq!(transcript.outcome.winner, Some(Faction::Werewolves));
        let rendered = transcript.to_string();
        assert!(rendered.contains("Night 2  (3 living)\n"), "{rendered}");
        assert!(!rendered.contains("Day 2"), "{rendered}");
        assert!(rendered.contains("  dave  protects bob\n"), "{rendered}");
        assert!(
            rendered.ends_with("Werewolves win after 2 rounds.  Survivors: bob, dave\n"),
            "{rendered}"
        );
    }

    #[test]
    fn the_hammer_is_the_last_nomination_passed_on_before_the_lynching() {
        // Nothing narrates the hammer any more (ADR-0015). It is read
        // from the moderator's own actions: a nomination it passed on is
        // one it accepted, and the last such before it announced the
        // lynching is the selection that completed the majority.
        let lines = werewolves_win_at_parity();
        let day = read(&lines).unwrap().rounds[0]
            .day
            .clone()
            .expect("the game had a day");
        assert_eq!(
            day.eliminated,
            Some((id("erin"), Villager, Cause::Lynched)),
            "erin is lynched"
        );
        assert_eq!(day.hammer, Some(id("dave")), "dave's selection made it");

        // And nothing said so: the day's narrations are the lynching and
        // what follows it, with no summary of the session that decided.
        let narrated: Vec<String> = lines
            .iter()
            .filter(|line| line["type"] == "action" && line["message"]["sender"] == "moderator")
            .filter_map(|line| line["message"]["payload"]["Narration"].as_object())
            .flat_map(|narration| narration.keys().cloned())
            .collect();
        assert!(
            narrated
                .iter()
                .all(|name| name != "Tally" && !name.contains("Hammer")),
            "the hammer is derived, never narrated: {narrated:?}"
        );
    }

    #[test]
    fn a_hammer_never_carries_over_from_an_earlier_day() {
        // A day that runs out of time has no hammer, and must not
        // inherit one from a day that had it. Day 1 is lynched on dave's
        // selection; day 2 is left to the clock.
        let lines = scripted(
            Assignment::new([
                ("alice", Werewolf),
                ("bob", Villager),
                ("carol", Villager),
                ("dave", Villager),
                ("erin", Villager),
                ("frank", Villager),
                ("grace", Villager),
            ]),
            &[
                answers([("alice", "bob")]),
                // Four of six living at frank: the fourth is the hammer.
                answers([
                    ("alice", "frank"),
                    ("carol", "frank"),
                    ("dave", "frank"),
                    ("erin", "frank"),
                ]),
                answers([("alice", "carol")]),
                // Two of four living: never a majority, so the day runs
                // out and nobody is lynched.
                answers([("alice", "dave"), ("dave", "alice")]),
                // The pack takes dave, leaving alice with erin and
                // grace; then a majority of three lynches alice and the
                // village wins.
                answers([("alice", "dave")]),
                answers([("alice", "erin"), ("erin", "alice"), ("grace", "alice")]),
            ],
        );
        let rounds = read(&lines).unwrap().rounds;
        let first = rounds[0].day.clone().expect("day 1");
        assert_eq!(first.hammer, Some(id("erin")), "day 1 ended on a selection");
        let second = rounds[1].day.clone().expect("day 2");
        assert!(second.no_lynch, "day 2 ran out of time");
        assert_eq!(second.hammer, None, "a day that ran out has no hammer");
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
            assert!(round.night.investigations.is_empty(), "{:?}", round.round);
            assert_eq!(round.night.moves.len(), 1);
            if let Some(day) = &round.day {
                assert!(day.investigations.is_empty());
            }
        }
        assert!(!transcript.to_string().contains("investigates"));
    }

    #[test]
    fn errors_display_the_line_they_were_found_on() {
        let error = TranscriptError::NotAMember {
            line: 12,
            from: id("bob"),
            kind: SessionKind::Protect,
        };
        assert_eq!(
            error.to_string(),
            "line 12: bob selected in a Protect session, which it is not a member of"
        );
        assert!(error.source().is_none());
    }
}
