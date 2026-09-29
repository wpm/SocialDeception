//! Watching a game as it plays: the [`Sink`] that renders Werewolf's
//! records as legible text, a line at a time.
//!
//! The log on disk is the record a reader joins and a training
//! pipeline reads; it is not something to watch. This module is the other
//! view of the same stream: one line per thing that happens, printed the
//! instant it happens, so a game — especially the timed, talking games the
//! later milestones build — can be followed as it is played.
//!
//! # Only actions are shown
//!
//! [`Reading::line`] renders an [`ActionRecord`] and nothing else. Every
//! message in an episode is some agent's action, so rendering actions shows
//! each message exactly once, from the side of whoever sent it. An
//! observation would show it a second time, once per recipient; a cycle, a
//! control or a reward is the runtime's bookkeeping rather than something that
//! happened in the game.
//!
//! That is a deliberate limit. What a live line shows is what its sender
//! said, not what each player knew, so the live view is not the transcript.
//! [`Transcript`](super::Transcript), which `werewolf replay` renders, is
//! the reproducible reading of a game; this is a window onto one being
//! played.
//!
//! # The live view is post-processing of the log
//!
//! The log records bare facts and no interpretation (ADR-0017). The live view
//! is not part of the log: it is one of the writer's [`Sink`]s, downstream of
//! every record, writing to the terminal and never to the log. It is
//! post-processing run incrementally over the log as it grows, and it keeps
//! the discipline of any other reader of the log.
//!
//! **It uses only facts from records it has already received.** A [`Reading`]
//! is built from nothing and learns the game from the stream: who the players
//! are, from the ids the records name, and who is still living, from the
//! moderator's own [`PhaseBegan`] and [`Eliminated`] narrations. It holds no
//! [`Game`](super::Game) and no moderator, so it can know nothing the log
//! does not say. That is why it takes no roster: a roster is setup input the
//! log does not carry.
//!
//! **The same computation over the finished file gives the same result.** One
//! actor's records reach the writer in the order it sent them, so every
//! moderator record that changed who is living arrives before any later
//! moderator message addressed to the living. Replaying a finished log
//! through a fresh reading labels every line exactly as the live one did.
//!
//! **It stays live.** Every line is written as soon as its record arrives.
//! Nothing is buffered or deferred to the end of the episode, which is what
//! makes the two renderings the same rather than a second pass over
//! something held back.
//!
//! # Everybody
//!
//! Naming seven players in the recipients column of every announcement says
//! less than one word does. So a message the moderator addressed to all the
//! living reads `everybody`, and so does a relay, whose recipients are all
//! the living but the player its [`Envelope`](crate::Envelope) names: it is
//! understood that a speaker does not address itself. Any other recipient set
//! is listed.
//!
//! This compares a moderator message only with earlier moderator records, so
//! the interleaving of agents below does not reach it. Before the first
//! [`PhaseBegan`] nobody is known to be living and nothing reads `everybody`,
//! which is right for the role assignments that come first: each goes to one
//! player.
//!
//! [`PhaseBegan`]: Narration::PhaseBegan
//! [`Eliminated`]: Narration::Eliminated
//!
//! # The order is the writer's
//!
//! Lines appear in the order the writer received the records, which
//! interleaves agents arbitrarily (see [`log`](crate::log)). A
//! player's selection can appear before the phase announcement that prompted
//! it is rendered. That is expected of a live view of concurrent agents, and
//! is the same interleaving the log file records.

use std::collections::BTreeSet;
use std::fmt;
use std::io::{self, Write};

use super::message::{Cause, Look, Message, Narration, Phase, Select};
use crate::log::{ActionRecord, Elapsed, Record, Sink};
use crate::message::ActorId;

/// How wide the time column is, so that the columns line up for any game
/// shorter than ten minutes and simply grow for a longer one.
const TIME_WIDTH: usize = 8;

/// What the recipients column says in place of every living player.
const EVERYBODY: &str = "everybody";

/// A reading of a game's records: each one as a line, and what the records so
/// far have said.
///
/// This is the whole of the rendering, and [`Text`] is it bound to a writer.
/// It is a type of its own because the rendering is a fold over records and
/// has nothing to do with where the lines go: the same reading over a
/// finished log produces the lines the live one did, which is the property
/// the module documentation states and a test pins.
///
/// A fresh reading knows nothing. It learns the players from the ids its
/// records name and the living from the moderator's narrations.
#[derive(Debug, Default)]
/// [`Text`] is the only thing that folds records into lines in anger, so this
/// is not among the names [`werewolf`](super) re-exports: it is here for the
/// module documentation to name and for a reader who wants the fold without a
/// writer attached.
pub struct Reading {
    /// Everyone still in the game, as the moderator's records last said.
    /// Empty until the first [`PhaseBegan`](Narration::PhaseBegan), and so
    /// matching nothing.
    living: BTreeSet<ActorId>,
    /// How wide the sender column is: the longest id any record has named.
    ///
    /// It grows as ids arrive rather than being given, so the column settles
    /// at the first [`PhaseBegan`](Narration::PhaseBegan), which names
    /// everybody, and never narrows. That is the cost of taking no roster,
    /// and it is paid in the assignments that come before that line.
    senders: usize,
}

impl Reading {
    /// A reading that has seen nothing.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Renders one record as one line, or nothing, and folds what the record
    /// says into what the reading knows.
    ///
    /// The line has four columns: the time since the episode's origin, the
    /// sender, the recipients, and the payload. It carries no trailing
    /// newline; [`Text`] adds one.
    ///
    /// Only an action produces a line. Everything else is [`None`]; see the
    /// module documentation for why.
    pub fn line(&mut self, record: &Record<Message, Elapsed>) -> Option<String> {
        let Record::Action(action) = record else {
            return None;
        };
        // The record is folded in before it is rendered, so that a record
        // announcing a change in who is living is read against the state it
        // announces. An `Eliminated` goes to the living without the player it
        // names (ADR-0012), which is the new living set, so it reads
        // `everybody` — as it should, since that is exactly who heard it.
        self.learn(action);
        Some(self.rendered(action))
    }

    /// Folds one action record into the living set and the column width.
    fn learn(&mut self, action: &ActionRecord<Message, Elapsed>) {
        for who in [&action.agent]
            .into_iter()
            .chain(&action.message.recipients)
        {
            self.senders = self.senders.max(who.as_str().chars().count());
        }
        // Only the moderator narrates, so only a narration can say who is
        // living, and the two that do are the two the module documentation
        // names.
        match &action.message.payload {
            Message::Narration(Narration::PhaseBegan { living, .. }) => {
                self.living.clone_from(living);
            }
            Message::Narration(Narration::Eliminated { who, .. }) => {
                self.living.remove(who);
            }
            Message::Narration(_)
            | Message::Select(_)
            | Message::Relayed(_)
            | Message::Reminder(_) => {}
        }
    }

    /// One action record as its four columns.
    fn rendered(&self, action: &ActionRecord<Message, Elapsed>) -> String {
        // The sender goes in as `&str`, not as the `ActorId` it is: `ActorId`'s
        // `Display` writes straight through and so ignores the width, which is
        // the whole point of the column.
        format!(
            "{:>TIME_WIDTH$} {:<senders$}  \u{2192} {}  {}",
            Clock(action.t),
            action.agent.as_str(),
            self.addressed(&action.message),
            action.message.payload,
            senders = self.senders,
        )
    }

    /// The recipients column: `everybody` when the message went to all the
    /// living, and the list of them otherwise.
    fn addressed(&self, message: &crate::Message<Message>) -> String {
        if self.all_the_living(message) {
            return EVERYBODY.to_owned();
        }
        listed(&message.recipients)
    }

    /// Whether `message` went to exactly the living, or — being a relay — to
    /// exactly the living but the player its envelope names.
    ///
    /// Never true while nobody is known to be living. A reading that has seen
    /// no phase begin has no set to compare against, and comparing against
    /// the empty one would call a message addressed to nobody a message to
    /// everybody.
    fn all_the_living(&self, message: &crate::Message<Message>) -> bool {
        if self.living.is_empty() {
            return false;
        }
        if message.recipients == self.living {
            return true;
        }
        let Message::Relayed(envelope) = &message.payload else {
            return false;
        };
        // The living without the speaker, which is who a relay goes to: the
        // speaker already knows what it said (ADR-0018).
        let mut listeners = self.living.clone();
        listeners.remove(&envelope.from);
        message.recipients == listeners
    }
}

/// A time since the episode's origin, rendered as `m:ss.mmm`.
///
/// Minutes are not padded, so a game runs from `0:00.000` and a long one
/// widens rather than wrapping. Sub-millisecond precision is dropped: a
/// reader watching a game wants to see the shape of the timing, and the
/// log keeps the nanoseconds for anyone who wants them.
struct Clock(Elapsed);

impl fmt::Display for Clock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.0.nanos() / 1_000_000;
        let (minutes, rest) = (millis / 60_000, millis % 60_000);
        write!(f, "{minutes}:{:02}.{:03}", rest / 1000, rest % 1000)
    }
}

impl fmt::Display for Message {
    /// The message compactly, as the payload column of a live line.
    ///
    /// Every message the game can carry is rendered here, so that a feature
    /// that adds one extends this in one place and every live view gets it.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Narration(narration) => narration.fmt(f),
            Self::Select(selection) => selection.fmt(f),
            // The sender column is the actor that sent the message, on every
            // line, which for a relay is the moderator. So the player whose
            // selection it is goes here, where a reader sees both that a
            // relay happened and whose move it passes on (ADR-0018) — the
            // same two facts the log record carries, and neither hidden.
            Self::Relayed(envelope) => {
                write!(f, "Relayed({}: {})", envelope.from, envelope.payload)
            }
            // The moderator's own note to itself, which says which phase's
            // clocks it was watching. It reaches a line only through the
            // observation record the reminder becomes when it fires, and
            // `line` renders actions alone, so nothing in a live view shows
            // one today; rendering it anyway is what keeps every payload of
            // this game legible in one place.
            Self::Reminder(Look::Session { round, phase }) => {
                write!(f, "Reminder({phase} {})", round.number())
            }
            Self::Reminder(Look::Farewell) => f.write_str("Reminder(Farewell)"),
        }
    }
}

impl fmt::Display for Narration {
    /// The narration's name, and what it says between parentheses.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Assigned { role, pack } => {
                write!(f, "Assigned({role}")?;
                if !pack.is_empty() {
                    write!(f, "; pack: {}", listed(pack))?;
                }
                f.write_str(")")
            }
            Self::PhaseBegan {
                round,
                phase,
                living,
            } => write!(
                f,
                "PhaseBegan({phase} {}; {} living)",
                round.number(),
                living.len()
            ),
            Self::Investigated { target, faction } => {
                write!(f, "Investigated({target}: {faction})")
            }
            Self::Eliminated {
                who, role, cause, ..
            } => write!(f, "Eliminated({who}, {role}, {cause})"),
            Self::NoDeath { round } => write!(f, "NoDeath(Night {})", round.number()),
            Self::NoLynch { round } => write!(f, "NoLynch(Day {})", round.number()),
            Self::Outcome(outcome) => {
                let ended = match outcome.winner {
                    Some(winner) => format!("{winner} win"),
                    None => "stalemate".to_owned(),
                };
                write!(
                    f,
                    "Outcome({ended} after {} rounds; survivors: {})",
                    outcome.rounds.number(),
                    listed(&outcome.living),
                )
            }
        }
    }
}

impl fmt::Display for Select {
    /// The session it selects in and whom it selects.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Select({:?}: {})", self.kind, self.target)
    }
}

impl fmt::Display for Cause {
    /// How the death reads in a sentence: lower case, because it is the
    /// manner of a death and not a proper name.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Devoured => "devoured",
            Self::Lynched => "lynched",
        })
    }
}

impl fmt::Display for Phase {
    /// The phase's name as written, the same spelling it serializes as.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Night => "Night",
            Self::Day => "Day",
        })
    }
}

/// A comma-separated list of actor ids, in the order given.
///
/// Not named `ids`: the test helper `testing::ids` builds a set of them,
/// and two functions of that name in one file would be a puzzle.
fn listed<'a>(who: impl IntoIterator<Item = &'a ActorId>) -> String {
    who.into_iter()
        .map(ActorId::as_str)
        .collect::<Vec<_>>()
        .join(", ")
}

/// The live text sink: every action as a line, flushed as it is written.
///
/// It flushes each line rather than buffering, because the point of it is
/// that a watcher sees the game at the pace it is played. That costs a
/// write syscall per line, which is nothing beside the pace of a game with
/// players who think.
///
/// A run binds one of these to stdout, as [`Policy::Optional`]: a reader
/// that closes the pipe drops the sink, and the game finishes with its
/// log complete.
///
/// [`Policy::Optional`]: crate::log::Policy::Optional
#[derive(Debug)]
pub struct Text<W: Write> {
    out: W,
    reading: Reading,
}

impl<W: Write> Text<W> {
    /// A sink writing a fresh [`Reading`]'s lines to `out`.
    ///
    /// It takes no roster: the view learns the game from the records, which
    /// is what keeps the live view post-processing of the log and nothing
    /// more (see the module documentation).
    pub fn new(out: W) -> Self {
        Self {
            out,
            reading: Reading::new(),
        }
    }
}

impl<W: Write + Send> Sink<Message> for Text<W> {
    fn record(&mut self, record: &Record<Message, Elapsed>) -> io::Result<()> {
        let Some(line) = self.reading.line(record) else {
            return Ok(());
        };
        writeln!(self.out, "{line}")?;
        self.out.flush()
    }

    fn finish(&mut self) -> io::Result<()> {
        self.out.flush()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::time::Duration;

    use super::*;
    use crate::log::{ControlRecord, CycleRecord, Key, ObservationRecord, RewardRecord};
    use crate::message::{Control, Envelope};
    use crate::testing::{Shared, id, ids};
    use crate::werewolf::message::{Outcome, Round, SessionKind};
    use crate::werewolf::role::{Faction, Role};

    /// `nanos` into the episode, as the log writes a time.
    fn at(nanos: u64) -> Elapsed {
        Elapsed::from(Duration::from_nanos(nanos))
    }

    /// An action of `sender` to `recipients` carrying `payload`, sent
    /// `nanos` into the episode.
    fn action<const N: usize>(
        sender: &str,
        recipients: [&str; N],
        nanos: u64,
        payload: Message,
    ) -> Record<Message, Elapsed> {
        let message = crate::Message::new(sender, recipients, 0, payload);
        ActionRecord {
            agent: id(sender),
            t: at(nanos),
            key: Key::of(&message),
            message,
        }
        .into()
    }

    /// The one line `record` renders to through a reading of nothing else.
    fn alone(record: &Record<Message, Elapsed>) -> String {
        Reading::new().line(record).unwrap()
    }

    /// The recipients and payload columns of one rendered line: everything
    /// after the arrow, split at the two spaces between them.
    ///
    /// The one place the line's column grammar is written down, so that every
    /// test that reads a column reads it the same way.
    fn columns(line: &str) -> (&str, &str) {
        line.split_once("\u{2192} ")
            .unwrap()
            .1
            .split_once("  ")
            .unwrap()
    }

    /// The payload column of the one line `record` renders to.
    fn payload(record: &Record<Message, Elapsed>) -> String {
        columns(&alone(record)).1.to_owned()
    }

    /// The payload column of an action carrying `message`.
    fn shown(message: Message) -> String {
        payload(&action("moderator", ["alice"], 0, message))
    }

    #[test]
    fn an_assignment_names_the_role_and_the_pack_when_there_is_one() {
        assert_eq!(
            shown(Message::Narration(Narration::Assigned {
                role: Role::Werewolf,
                pack: ids(["dave", "erin"]),
            })),
            "Assigned(Werewolf; pack: dave, erin)"
        );
        // A villager has no pack, and no empty parenthesis where one would
        // have been.
        assert_eq!(
            shown(Message::Narration(Narration::Assigned {
                role: Role::Villager,
                pack: BTreeSet::new(),
            })),
            "Assigned(Villager)"
        );
    }

    #[test]
    fn a_phase_names_its_round_and_how_many_are_left() {
        assert_eq!(
            shown(Message::Narration(Narration::PhaseBegan {
                round: Round::new(1),
                phase: Phase::Night,
                living: ids(["alice", "bob", "carol", "dave", "erin", "frank", "grace"]),
            })),
            "PhaseBegan(Night 1; 7 living)"
        );
        assert_eq!(
            shown(Message::Narration(Narration::PhaseBegan {
                round: Round::new(2),
                phase: Phase::Day,
                living: ids(["alice"]),
            })),
            "PhaseBegan(Day 2; 1 living)"
        );
    }

    #[test]
    fn an_investigation_names_the_target_and_what_it_learned() {
        assert_eq!(
            shown(Message::Narration(Narration::Investigated {
                target: id("grace"),
                faction: Faction::Werewolves,
            })),
            "Investigated(grace: Werewolves)"
        );
    }

    #[test]
    fn an_elimination_names_who_what_they_were_and_how() {
        assert_eq!(
            shown(Message::Narration(Narration::Eliminated {
                who: id("alice"),
                role: Role::Villager,
                round: Round::new(1),
                cause: Cause::Lynched,
            })),
            "Eliminated(alice, Villager, lynched)"
        );
        assert_eq!(
            shown(Message::Narration(Narration::Eliminated {
                who: id("bob"),
                role: Role::Seer,
                round: Round::new(2),
                cause: Cause::Devoured,
            })),
            "Eliminated(bob, Seer, devoured)"
        );
    }

    #[test]
    fn a_quiet_night_names_its_round() {
        assert_eq!(
            shown(Message::Narration(Narration::NoDeath {
                round: Round::new(2)
            })),
            "NoDeath(Night 2)"
        );
    }

    #[test]
    fn a_day_that_ran_out_names_its_round() {
        // A day that reached its limit without a majority is its own
        // narration, told from a night nobody died in by the phase it
        // names (ADR-0011).
        assert_eq!(
            shown(Message::Narration(Narration::NoLynch {
                round: Round::new(3)
            })),
            "NoLynch(Day 3)"
        );
    }

    #[test]
    fn an_outcome_names_the_winner_the_rounds_and_the_survivors() {
        assert_eq!(
            shown(Message::Narration(Narration::Outcome(Outcome {
                winner: Some(Faction::Werewolves),
                rounds: Round::new(2),
                living: ids(["bob", "dave"]),
            }))),
            "Outcome(Werewolves win after 2 rounds; survivors: bob, dave)"
        );
    }

    #[test]
    fn a_selection_names_its_session_and_the_target() {
        // Nothing renders a request any more: there is none to render.
        // A selection says for itself what it is for (ADR-0014), so the line
        // reads without a request to look the id up in.
        assert_eq!(
            shown(Message::Select(Select {
                round: Round::new(1),
                kind: SessionKind::Nominate,
                target: id("frank"),
                seen_by: BTreeSet::new(),
            })),
            "Select(Nominate: frank)"
        );
    }

    #[test]
    fn a_relayed_selection_names_the_player_whose_selection_it_is() {
        // The moderator sent it, so the moderator is in the sender column,
        // as on every line. What the payload column adds is whose move it
        // passes on, which is the envelope's (ADR-0018): both facts the log
        // record carries, and neither hidden.
        let selection = Select {
            round: Round::new(2),
            kind: SessionKind::Nominate,
            target: id("frank"),
            seen_by: ids(["carol", "frank"]),
        };
        let record = action(
            "moderator",
            ["carol", "frank"],
            0,
            Message::Relayed(Envelope::new("bob", 3, selection)),
        );
        assert_eq!(
            alone(&record),
            "0:00.000 moderator  \u{2192} carol, frank  Relayed(bob: Select(Nominate: frank))"
        );
    }

    /// The seven players a game of `examples/werewolf.toml` seats, in the
    /// order a set holds them.
    const SEATS: [&str; 7] = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];

    /// A `PhaseBegan` to `living`, which is how a reading learns who is in
    /// the game at all.
    fn phase_began<const N: usize>(
        round: u32,
        phase: Phase,
        living: [&str; N],
    ) -> Record<Message, Elapsed> {
        action(
            "moderator",
            living,
            0,
            Message::Narration(Narration::PhaseBegan {
                round: Round::new(round),
                phase,
                living: ids(living),
            }),
        )
    }

    /// The recipients column of `record`, read by a reading that saw
    /// `before` first and nothing else.
    fn recipients_after<const N: usize>(
        before: [&Record<Message, Elapsed>; N],
        record: &Record<Message, Elapsed>,
    ) -> String {
        let mut reading = Reading::new();
        for earlier in before {
            reading.line(earlier);
        }
        columns(&reading.line(record).unwrap()).0.to_owned()
    }

    /// A narration to `to`, which the moderator addressed itself.
    fn narrated<const N: usize>(to: [&str; N], narration: Narration) -> Record<Message, Elapsed> {
        action("moderator", to, 0, Message::Narration(narration))
    }

    /// An announcement to `to` whose payload is immaterial: what is under
    /// test is the recipients column, and that the message is not a relay.
    fn announced<const N: usize>(to: [&str; N]) -> Record<Message, Elapsed> {
        narrated(
            to,
            Narration::NoLynch {
                round: Round::new(1),
            },
        )
    }

    /// The moderator relaying `who`'s nomination of `target` to `to`.
    fn relayed<const N: usize>(who: &str, target: &str, to: [&str; N]) -> Record<Message, Elapsed> {
        action(
            "moderator",
            to,
            0,
            Message::Relayed(Envelope::new(
                who,
                3,
                Select {
                    round: Round::new(1),
                    kind: SessionKind::Nominate,
                    target: id(target),
                    seen_by: ids(to),
                },
            )),
        )
    }

    #[test]
    fn a_narration_to_all_the_living_is_everybody_and_one_to_some_of_them_is_a_list() {
        let night = phase_began(1, Phase::Night, SEATS);
        // The announcement that taught the reading who is living is itself
        // addressed to all of them, so it reads `everybody` too.
        assert_eq!(recipients_after([], &night), EVERYBODY);
        assert_eq!(recipients_after([&night], &announced(SEATS)), EVERYBODY);
        // The pack is not everybody, however many of them there are, and
        // neither is one player.
        assert_eq!(
            recipients_after([&night], &announced(["bob", "dave"])),
            "bob, dave"
        );
        assert_eq!(recipients_after([&night], &announced(["alice"])), "alice");
        // Nor is anybody at all before a phase has begun, which is what makes
        // the role assignments that come first name the one player each went
        // to.
        assert_eq!(recipients_after([], &announced(SEATS)), listed(&ids(SEATS)));
    }

    #[test]
    fn a_message_addressed_to_nobody_is_not_addressed_to_everybody() {
        // Nobody is known to be living before the first phase begins, and a
        // message with no recipients is not a message to all of them. The
        // moderator sends no such message; the reading is not given one to
        // say so.
        let nobody: [&str; 0] = [];
        assert_eq!(recipients_after([], &announced(nobody)), "");
    }

    #[test]
    fn a_relay_to_all_the_living_but_the_player_its_envelope_names_is_everybody() {
        let day = phase_began(1, Phase::Day, SEATS);
        // A speaker does not address itself, so the living without it is
        // everybody a relay could have gone to (ADR-0018).
        let listeners = ["alice", "carol", "dave", "erin", "frank", "grace"];
        assert_eq!(
            recipients_after([&day], &relayed("bob", "frank", listeners)),
            EVERYBODY
        );
        // The same recipients under a payload that is not a relay are the
        // living minus a player and nothing more, and are listed.
        assert_eq!(
            recipients_after([&day], &announced(listeners)),
            "alice, carol, dave, erin, frank, grace"
        );
        // A relay that left out somebody other than its speaker is not
        // everybody either.
        assert_eq!(
            recipients_after(
                [&day],
                &relayed("bob", "frank", ["alice", "carol", "dave", "erin", "frank"])
            ),
            "alice, carol, dave, erin, frank"
        );
    }

    #[test]
    fn an_elimination_narrows_who_everybody_is() {
        let night = phase_began(1, Phase::Night, SEATS);
        let living = ["alice", "bob", "carol", "dave", "erin", "frank"];
        let eliminated = narrated(
            living,
            Narration::Eliminated {
                who: id("grace"),
                role: Role::Villager,
                round: Round::new(1),
                cause: Cause::Devoured,
            },
        );
        // The elimination goes to the living without the victim (ADR-0012),
        // which is the new living set, and so reads `everybody` on its own
        // line: that is exactly who heard it.
        assert_eq!(recipients_after([&night], &eliminated), EVERYBODY);
        assert_eq!(
            recipients_after([&night, &eliminated], &announced(living)),
            EVERYBODY
        );
        // The set that was everybody a moment ago is not any more: it names
        // a player who is out of the game.
        assert_eq!(
            recipients_after([&night, &eliminated], &announced(SEATS)),
            "alice, bob, carol, dave, erin, frank, grace"
        );
    }

    #[test]
    fn a_players_selection_is_addressed_to_the_moderator() {
        let night = phase_began(1, Phase::Night, SEATS);
        let devour = action(
            "bob",
            ["moderator"],
            0,
            Message::Select(Select {
                round: Round::new(1),
                kind: SessionKind::Devour,
                target: id("grace"),
                seen_by: BTreeSet::new(),
            }),
        );
        // The moderator is not one of the living and a player addresses it
        // alone, so nothing about `everybody` touches this line.
        assert_eq!(recipients_after([&night], &devour), "moderator");
    }

    #[test]
    fn replaying_a_finished_log_renders_the_lines_the_live_view_did() {
        // A game's worth of records in the order one writer received them,
        // players interleaved with the moderator as they always are.
        let records = vec![
            narrated(
                ["bob"],
                Narration::Assigned {
                    role: Role::Werewolf,
                    pack: ids(["bob"]),
                },
            ),
            narrated(
                ["alice"],
                Narration::Assigned {
                    role: Role::Seer,
                    pack: BTreeSet::new(),
                },
            ),
            phase_began(1, Phase::Night, SEATS),
            action(
                "bob",
                ["moderator"],
                1,
                Message::Select(Select {
                    round: Round::new(1),
                    kind: SessionKind::Devour,
                    target: id("grace"),
                    seen_by: BTreeSet::new(),
                }),
            ),
            narrated(
                ["alice", "bob", "carol", "dave", "erin", "frank"],
                Narration::Eliminated {
                    who: id("grace"),
                    role: Role::Villager,
                    round: Round::new(1),
                    cause: Cause::Devoured,
                },
            ),
            phase_began(
                1,
                Phase::Day,
                ["alice", "bob", "carol", "dave", "erin", "frank"],
            ),
            relayed("bob", "frank", ["alice", "carol", "dave", "erin", "frank"]),
            narrated(
                ["alice", "bob", "carol", "dave", "erin"],
                Narration::Eliminated {
                    who: id("frank"),
                    role: Role::Doctor,
                    round: Round::new(1),
                    cause: Cause::Lynched,
                },
            ),
            narrated(
                ["alice", "bob", "carol", "dave", "erin"],
                Narration::Outcome(Outcome {
                    winner: Some(Faction::Werewolves),
                    rounds: Round::new(1),
                    living: ids(["alice", "bob", "carol", "dave", "erin"]),
                }),
            ),
        ];

        // Live: a sink fed each record as it arrives, flushing every line.
        let shown = Shared::new();
        let mut live = Text::new(shown.clone());
        for record in &records {
            live.record(record).unwrap();
        }
        live.finish().unwrap();

        // After the fact: a fresh reading over the finished file.
        let replayed = {
            let mut reading = Reading::new();
            let lines: Vec<String> = records
                .iter()
                .filter_map(|record| reading.line(record))
                .collect();
            lines.join("\n") + "\n"
        };

        assert_eq!(written(&shown), replayed);
        // And what they agree on is the reading the issue asks for: the
        // announcements to the living say `everybody`, the relay says it too,
        // and the assignments and the selection name who they went to.
        let addressed: Vec<&str> = replayed.lines().map(|line| columns(line).0).collect();
        assert_eq!(
            addressed,
            [
                "bob",
                "alice",
                EVERYBODY,
                "moderator",
                EVERYBODY,
                EVERYBODY,
                EVERYBODY,
                EVERYBODY,
                EVERYBODY,
            ]
        );
    }

    #[test]
    fn the_time_column_is_minutes_seconds_and_milliseconds() {
        for (nanos, expected) in [
            (0, "0:00.000"),
            (1_250_000_000, "0:01.250"),
            (723_004_000_000, "12:03.004"),
        ] {
            let record = action(
                "a",
                ["b"],
                nanos,
                Message::Narration(Narration::NoDeath {
                    round: Round::new(1),
                }),
            );
            let rendered = alone(&record);
            assert!(
                rendered.starts_with(&format!("{expected:>8} ")),
                "{nanos}: {rendered}"
            );
        }
    }

    #[test]
    fn the_columns_are_the_time_the_sender_the_recipients_and_the_payload() {
        let record = action(
            "moderator",
            ["bob", "alice"],
            1_250_000_000,
            Message::Narration(Narration::NoDeath {
                round: Round::new(3),
            }),
        );
        // The recipients are sorted and comma-separated, whatever order
        // they were given in.
        assert_eq!(
            alone(&record),
            "0:01.250 moderator  \u{2192} alice, bob  NoDeath(Night 3)"
        );
    }

    #[test]
    fn the_sender_column_is_as_wide_as_the_widest_id_seen_so_far() {
        let devour = action(
            "bob",
            ["moderator"],
            0,
            Message::Select(Select {
                round: Round::new(1),
                kind: SessionKind::Devour,
                target: id("alice"),
                seen_by: BTreeSet::new(),
            }),
        );
        // The reading has no roster, so `moderator` in this one record is
        // already the widest id it has seen, and the column is that wide.
        assert_eq!(
            alone(&devour),
            "0:00.000 bob        \u{2192} moderator  Select(Devour: alice)"
        );
        // Before any long id has arrived the column is as narrow as what it
        // has seen, and an id wider than it simply overflows it rather than
        // being cut: a name is worth more than an aligned column.
        let mut reading = Reading::new();
        let short = action(
            "a",
            ["b"],
            0,
            Message::Narration(Narration::NoDeath {
                round: Round::new(1),
            }),
        );
        assert_eq!(
            reading.line(&short).unwrap(),
            "0:00.000 a  \u{2192} b  NoDeath(Night 1)"
        );
        assert_eq!(
            reading.line(&devour).unwrap(),
            "0:00.000 bob        \u{2192} moderator  Select(Devour: alice)"
        );
    }

    #[test]
    fn nothing_but_an_action_renders() {
        let message = crate::Message::new(
            "moderator",
            ["alice"],
            0,
            Message::Narration(Narration::NoDeath {
                round: Round::new(1),
            }),
        );
        let records: Vec<Record<Message, Elapsed>> = vec![
            ObservationRecord {
                agent: id("alice"),
                t: at(1),
                key: Key::of(&message),
                message,
            }
            .into(),
            ControlRecord {
                agent: id("alice"),
                t: at(1),
                control: Control::Start,
            }
            .into(),
            CycleRecord {
                agent: id("alice"),
                t_start: at(0),
                t_stop: at(1),
                observed: None,
            }
            .into(),
            RewardRecord {
                agent: id("alice"),
                t: at(2),
                value: serde_json::json!(1),
            }
            .into(),
            crate::EpisodeRecord::of(crate::Clock::start()).into(),
        ];
        let mut reading = Reading::new();
        for record in &records {
            assert_eq!(reading.line(record), None, "{record:?}");
        }
    }

    /// What a sink wrote to `shown`, as text.
    fn written(shown: &Shared) -> String {
        String::from_utf8(shown.bytes()).unwrap()
    }

    #[test]
    fn the_sink_writes_a_line_per_action_and_nothing_for_the_rest() {
        let shown = Shared::new();
        let mut text = Text::new(shown.clone());
        let narration = action(
            "moderator",
            ["alice"],
            0,
            Message::Narration(Narration::NoDeath {
                round: Round::new(1),
            }),
        );
        let cycle: Record<Message, Elapsed> = CycleRecord {
            agent: id("alice"),
            t_start: at(0),
            t_stop: at(1),
            observed: None,
        }
        .into();
        text.record(&narration).unwrap();
        text.record(&cycle).unwrap();
        text.record(&narration).unwrap();
        text.finish().unwrap();
        assert_eq!(
            written(&shown),
            "0:00.000 moderator  \u{2192} alice  NoDeath(Night 1)\n\
             0:00.000 moderator  \u{2192} alice  NoDeath(Night 1)\n"
        );
    }

    #[test]
    fn a_sink_renders_the_first_record_it_is_given() {
        // A sink is built from nothing, so the first record is one it knows
        // nothing about, and it renders anyway.
        let shown = Shared::new();
        let mut text = Text::new(shown.clone());
        text.record(&action(
            "a",
            ["b"],
            0,
            Message::Narration(Narration::NoDeath {
                round: Round::new(1),
            }),
        ))
        .unwrap();
        assert_eq!(
            written(&shown),
            "0:00.000 a  \u{2192} b  NoDeath(Night 1)\n"
        );
    }
}
