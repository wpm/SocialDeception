//! Watching a game as it plays: the [`Sink`] that renders Werewolf's
//! records as legible text, a line at a time.
//!
//! The trajectory on disk is the record a reader joins and a training
//! pipeline reads; it is not something to watch. This module is the other
//! view of the same stream: one line per thing that happens, printed the
//! instant it happens, so a game — especially the timed, talking games the
//! later milestones build — can be followed as it is played.
//!
//! # Only actions are shown
//!
//! [`line()`] renders an [`ActionRecord`] and nothing else. Every event in an
//! episode is some agent's action, so rendering actions shows each event
//! exactly once, from the side of whoever sent it. An observation would
//! show it a second time, once per recipient; a cycle, a control or a
//! reward is the runtime's bookkeeping rather than something that happened
//! in the game.
//!
//! That is a deliberate limit. What a live line shows is what its sender
//! said, not what each player knew, so the live view is not the transcript.
//! [`Transcript`](super::Transcript), which `werewolf replay` renders, is
//! the reproducible reading of a game; this is a window onto one being
//! played.
//!
//! # The order is the writer's
//!
//! Lines appear in the order the writer received the records, which
//! interleaves agents arbitrarily (see [`trajectory`](crate::trajectory)).
//! A player's response can appear before the request that prompted it is
//! rendered. That is expected of a live view of concurrent agents, and is
//! the same interleaving the trajectory file records.

use std::fmt;
use std::io::{self, Write};

use super::WerewolfDomain;
use super::message::{Cause, Message, Move, Narration, Phase, Request, Response};
use crate::event::AgentId;
use crate::trajectory::{ActionRecord, LogRecord, Sink};

/// How wide the time column is, so that the columns line up for any game
/// shorter than ten minutes and simply grow for a longer one.
const TIME_WIDTH: usize = 8;

/// Renders one record as one line, or nothing.
///
/// `senders` is how wide the sender column is: the width of the longest
/// agent id in the game, so that the columns line up. [`Text`] takes it
/// from the roster it is built with.
///
/// The line has four columns: the time since the episode's clock started,
/// the sender, the recipients, and the payload. It carries no trailing
/// newline; [`Text`] adds one.
///
/// Only an action produces a line. Everything else is [`None`]; see the
/// module documentation for why.
#[must_use]
pub fn line(record: &LogRecord<WerewolfDomain>, senders: usize) -> Option<String> {
    let LogRecord::Action(action) = record else {
        return None;
    };
    Some(rendered(action, senders))
}

/// One action record as its four columns.
fn rendered(action: &ActionRecord<WerewolfDomain>, senders: usize) -> String {
    // The sender goes in as `&str`, not as the `AgentId` it is: `AgentId`'s
    // `Display` writes straight through and so ignores the width, which is
    // the whole point of the column.
    format!(
        "{:>TIME_WIDTH$} {:<senders$}  \u{2192} {}  {}",
        Elapsed(action.created.nanos()),
        action.agent.as_str(),
        Ids(action.event.recipients.iter()),
        action.event.payload,
    )
}

/// A duration since the episode's clock started, in nanoseconds, rendered
/// as `m:ss.mmm`.
///
/// Minutes are not padded, so a game runs from `0:00.000` and a long one
/// widens rather than wrapping. Sub-millisecond precision is dropped: a
/// reader watching a game wants to see the shape of the timing, and the
/// trajectory keeps the nanoseconds for anyone who wants them.
struct Elapsed(u64);

impl fmt::Display for Elapsed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.0 / 1_000_000;
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
            Self::Request(request) => request.fmt(f),
            Self::Response(response) => response.fmt(f),
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
                    write!(f, "; pack: {}", Ids(pack.iter()))?;
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
                round.0,
                living.len()
            ),
            Self::Investigated { target, faction } => {
                write!(f, "Investigated({target}: {faction})")
            }
            Self::Tally {
                round,
                phase,
                votes,
            } => {
                write!(f, "Tally({phase} {}: ", round.0)?;
                for (i, (who, chosen)) in votes.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    write!(f, "{who}\u{2192}{chosen}")?;
                }
                f.write_str(")")
            }
            Self::Eliminated {
                who, role, cause, ..
            } => write!(f, "Eliminated({who}, {role}, {cause})"),
            Self::NoDeath { round } => write!(f, "NoDeath(Night {})", round.0),
            Self::Outcome(outcome) => write!(
                f,
                "Outcome({} win after {} rounds; survivors: {})",
                outcome.winner,
                outcome.rounds.0,
                Ids(outcome.living.iter()),
            ),
        }
    }
}

impl fmt::Display for Request {
    /// The request's id and what it asks.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Request(#{}: {:?})", self.id.0, self.kind)
    }
}

impl fmt::Display for Response {
    /// The id it answers and the move it carries.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Response(#{}: {})", self.request.0, self.chosen)
    }
}

impl fmt::Display for Move {
    /// The player targeted, or `abstain`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Target(who) => who.fmt(f),
            Self::Abstain => f.write_str("abstain"),
        }
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

/// A comma-separated list of agent ids, in the order given.
struct Ids<I>(I);

impl<'a, I: Iterator<Item = &'a AgentId> + Clone> fmt::Display for Ids<I> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, who) in self.0.clone().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            who.fmt(f)?;
        }
        Ok(())
    }
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
/// trajectory complete.
///
/// [`Policy::Optional`]: crate::trajectory::Policy::Optional
#[derive(Debug)]
pub struct Text<W: Write> {
    out: W,
    senders: usize,
}

impl<W: Write> Text<W> {
    /// A sink writing to `out`, with the sender column as wide as the
    /// longest id in `roster`.
    ///
    /// The width comes from the roster rather than from the ids seen so far
    /// because a column that widened partway down would not be a column.
    pub fn new<'a>(out: W, roster: impl IntoIterator<Item = &'a AgentId>) -> Self {
        let senders = roster
            .into_iter()
            .map(|who| who.as_str().chars().count())
            .max()
            .unwrap_or(0);
        Self { out, senders }
    }
}

impl<W: Write + Send> Sink<WerewolfDomain> for Text<W> {
    fn record(&mut self, record: &LogRecord<WerewolfDomain>) -> io::Result<()> {
        let Some(line) = line(record, self.senders) else {
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
    use std::collections::{BTreeMap, BTreeSet};
    use std::time::Duration;

    use super::*;
    use crate::clock::Timestamp;
    use crate::event::{Control, Event};
    use crate::testing::{Shared, id, ids};
    use crate::trajectory::{
        ControlRecord, CycleRecord, ObservationRecord, RewardRecord, Seq, Woken,
    };
    use crate::werewolf::message::{Outcome, RequestId, RequestKind, Round};
    use crate::werewolf::role::{Faction, Role};

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

    /// An action of `sender` to `recipients` carrying `payload`, created
    /// `nanos` into the episode.
    fn action<const N: usize>(
        sender: &str,
        recipients: [&str; N],
        nanos: u64,
        payload: Message,
    ) -> LogRecord<WerewolfDomain> {
        ActionRecord {
            agent: id(sender),
            seq: Seq(0),
            created: at(nanos),
            event: Event::new(sender, recipients, at(nanos), payload),
        }
        .into()
    }

    /// The payload column of the one line `record` renders to.
    fn payload(record: &LogRecord<WerewolfDomain>) -> String {
        let rendered = line(record, 0).unwrap();
        rendered
            .split_once("\u{2192} ")
            .unwrap()
            .1
            .split_once("  ")
            .unwrap()
            .1
            .to_owned()
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
                round: Round(1),
                phase: Phase::Night,
                living: ids(["alice", "bob", "carol", "dave", "erin", "frank", "grace"]),
            })),
            "PhaseBegan(Night 1; 7 living)"
        );
        assert_eq!(
            shown(Message::Narration(Narration::PhaseBegan {
                round: Round(2),
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
    fn a_tally_lists_every_vote_in_order() {
        assert_eq!(
            shown(Message::Narration(Narration::Tally {
                round: Round(1),
                phase: Phase::Day,
                votes: BTreeMap::from([
                    (id("alice"), Move::Target(id("frank"))),
                    (id("bob"), Move::Target(id("carol"))),
                    (id("dave"), Move::Abstain),
                ]),
            })),
            "Tally(Day 1: alice\u{2192}frank, bob\u{2192}carol, dave\u{2192}abstain)"
        );
    }

    #[test]
    fn an_elimination_names_who_what_they_were_and_how() {
        assert_eq!(
            shown(Message::Narration(Narration::Eliminated {
                who: id("alice"),
                role: Role::Villager,
                round: Round(1),
                cause: Cause::Lynched,
            })),
            "Eliminated(alice, Villager, lynched)"
        );
        assert_eq!(
            shown(Message::Narration(Narration::Eliminated {
                who: id("bob"),
                role: Role::Seer,
                round: Round(2),
                cause: Cause::Devoured,
            })),
            "Eliminated(bob, Seer, devoured)"
        );
    }

    #[test]
    fn a_quiet_night_names_its_round() {
        assert_eq!(
            shown(Message::Narration(Narration::NoDeath { round: Round(2) })),
            "NoDeath(Night 2)"
        );
    }

    #[test]
    fn an_outcome_names_the_winner_the_rounds_and_the_survivors() {
        assert_eq!(
            shown(Message::Narration(Narration::Outcome(Outcome {
                winner: Faction::Werewolves,
                rounds: Round(2),
                living: ids(["bob", "dave"]),
            }))),
            "Outcome(Werewolves win after 2 rounds; survivors: bob, dave)"
        );
    }

    #[test]
    fn a_request_names_its_id_and_what_it_asks() {
        assert_eq!(
            shown(Message::Request(Request {
                id: RequestId(3),
                round: Round(1),
                kind: RequestKind::Nominate,
            })),
            "Request(#3: Nominate)"
        );
    }

    #[test]
    fn a_response_names_the_request_and_the_move() {
        assert_eq!(
            shown(Message::Response(Response {
                request: RequestId(3),
                chosen: Move::Target(id("frank")),
            })),
            "Response(#3: frank)"
        );
        assert_eq!(
            shown(Message::Response(Response {
                request: RequestId(3),
                chosen: Move::Abstain,
            })),
            "Response(#3: abstain)"
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
                Message::Narration(Narration::NoDeath { round: Round(1) }),
            );
            let rendered = line(&record, 0).unwrap();
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
            Message::Narration(Narration::NoDeath { round: Round(3) }),
        );
        // The recipients are sorted and comma-separated, whatever order
        // they were given in.
        assert_eq!(
            line(&record, 9).unwrap(),
            "0:01.250 moderator  \u{2192} alice, bob  NoDeath(Night 3)"
        );
    }

    #[test]
    fn the_sender_column_is_as_wide_as_the_widest_id() {
        let record = action(
            "bob",
            ["moderator"],
            0,
            Message::Response(Response {
                request: RequestId(1),
                chosen: Move::Abstain,
            }),
        );
        assert_eq!(
            line(&record, 9).unwrap(),
            "0:00.000 bob        \u{2192} moderator  Response(#1: abstain)"
        );
        // An id wider than the column simply overflows it rather than being
        // cut: a name is worth more than an aligned column.
        assert_eq!(
            line(&record, 2).unwrap(),
            "0:00.000 bob  \u{2192} moderator  Response(#1: abstain)"
        );
    }

    #[test]
    fn nothing_but_an_action_renders() {
        let event = Event::new(
            "moderator",
            ["alice"],
            at(0),
            Message::Narration(Narration::NoDeath { round: Round(1) }),
        );
        let records: Vec<LogRecord<WerewolfDomain>> = vec![
            ObservationRecord {
                agent: id("alice"),
                seq: Seq(0),
                created: at(0),
                received: at(1),
                event,
            }
            .into(),
            ControlRecord {
                agent: id("alice"),
                seq: Seq(1),
                created: at(0),
                received: at(1),
                control: Control::Start,
            }
            .into(),
            CycleRecord {
                agent: id("alice"),
                t_start: at(0),
                t_stop: at(1),
                woken: Woken::Queue,
                inputs: vec![Seq(0)],
                outputs: vec![],
            }
            .into(),
            RewardRecord {
                agent: id("alice"),
                created: at(2),
                value: 1,
            }
            .into(),
        ];
        for record in &records {
            assert_eq!(line(record, 8), None, "{record:?}");
        }
    }

    /// What a sink wrote to `shown`, as text.
    fn written(shown: &Shared) -> String {
        String::from_utf8(shown.bytes()).unwrap()
    }

    #[test]
    fn the_sink_writes_a_line_per_action_and_nothing_for_the_rest() {
        let shown = Shared::new();
        let mut text = Text::new(shown.clone(), &ids(["alice", "moderator"]));
        let narration = action(
            "moderator",
            ["alice"],
            0,
            Message::Narration(Narration::NoDeath { round: Round(1) }),
        );
        let cycle: LogRecord<WerewolfDomain> = CycleRecord {
            agent: id("alice"),
            t_start: at(0),
            t_stop: at(1),
            woken: Woken::Queue,
            inputs: vec![],
            outputs: vec![],
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
    fn a_sink_over_an_empty_roster_still_renders() {
        let shown = Shared::new();
        let mut text = Text::new(shown.clone(), &BTreeSet::new());
        text.record(&action(
            "a",
            ["b"],
            0,
            Message::Narration(Narration::NoDeath { round: Round(1) }),
        ))
        .unwrap();
        assert_eq!(
            written(&shown),
            "0:00.000 a  \u{2192} b  NoDeath(Night 1)\n"
        );
    }
}
