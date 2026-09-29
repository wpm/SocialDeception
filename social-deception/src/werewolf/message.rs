//! Everything said in a Werewolf episode: the [`Message`] payload and the
//! vocabulary of rounds, phases, sessions and targets it is built from.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::num::NonZero;

use super::role::{Faction, Role};
use crate::message::{ActorId, Envelope};

/// A round of the game, counted from 1. Each round is a night then a day.
///
/// Serializes as a bare integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Round(pub NonZero<u32>);

impl Round {
    /// The round a game starts in.
    pub const FIRST: Self = Self(NonZero::<u32>::MIN);

    /// The round numbered `number`.
    ///
    /// # Panics
    ///
    /// If `number` is zero: rounds are counted from 1, and there is no
    /// round before the first.
    #[must_use]
    pub const fn new(number: u32) -> Self {
        match NonZero::new(number) {
            Some(number) => Self(number),
            None => panic!("a round is counted from 1"),
        }
    }

    /// Which round this is, counted from 1.
    #[must_use]
    pub const fn number(self) -> u32 {
        self.0.get()
    }

    /// The round after this one.
    #[must_use]
    pub const fn next(self) -> Self {
        Self::new(self.number() + 1)
    }

    /// The round before this one, or `None` for the first.
    #[must_use]
    pub const fn previous(self) -> Option<Self> {
        match NonZero::new(self.number() - 1) {
            Some(number) => Some(Self(number)),
            None => None,
        }
    }
}

/// Half of a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// The werewolves devour, the seer investigates, the doctor protects.
    Night,
    /// Everyone living nominates, and the plurality is lynched.
    Day,
}

/// Everything said in a Werewolf episode.
///
/// Every agent in an episode shares this one payload type, so it covers
/// moderator-to-player and player-to-moderator traffic alike. Serializes as
/// an object with one field, named after the variant.
///
/// This is the payload, not the envelope that carries it: a
/// [`crate::Message`] is what travels between actors, and its `payload`
/// field is one of these.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Message {
    /// Moderator to chosen players: something they now observe.
    Narration(Narration),
    /// Player to the moderator and to nobody else: a target selected. A
    /// player may send more than one in the same session; its latest is its
    /// vote (ADR-0011).
    Select(Select),
    /// Moderator to itself: a session's clock may have run out, so look.
    ///
    /// This is the moderator's private payload, the one nobody else ever
    /// sends and the one nobody else ever receives: a
    /// [`Reminder`](crate::actor::Reminder) is always self-directed
    /// (ADR-0016), and the moderator is the only actor in this game that
    /// keeps a clock. It replaces the old runtime's `deadline`/`timeout`
    /// pair: instead of telling the loop when to wake it, the moderator
    /// reminds itself, and when the reminder arrives it asks the game what
    /// that instant closed (ADR-0018).
    ///
    /// **A reminder for a limit that has since moved is ignored, and nothing
    /// is cancelled.** Reminders accumulate and each fires once, so a night
    /// session whose quiet period restarted has a reminder outstanding for
    /// the limit it no longer has; when that one arrives the game finds
    /// nothing expired and says nothing. What the payload carries is
    /// therefore not authority but provenance: which phase of which round
    /// the moderator set it in, so a reader of the log can tell one
    /// reminder from another.
    Reminder(Look),
    /// Moderator to the players who should see it: a selection a player made,
    /// in the envelope that names who made it.
    ///
    /// A player addresses the moderator alone, so this is the only way a
    /// selection reaches anybody else (ADR-0018). It is the moderator's own
    /// message, with a sequence number of the moderator's, and the envelope
    /// is what says whose selection it passes on: a listener reads the player
    /// from there rather than from the message's sender, and a reader of the
    /// log joins the relay back to the player's own action record on the
    /// envelope's `(from, seq)`.
    Relayed(Envelope<Select>),
}

/// Why the moderator reminded itself to look.
///
/// Two kinds, and the difference matters, because reminders **accumulate** and
/// a game ends with several outstanding (ADR-0016):
///
/// - a [`Session`](Look::Session) reminder asks the moderator to check its
///   clocks. Which phase it names is provenance and nothing else: what a
///   reminder closes is decided by asking
///   [`Game::expire`](super::Game::expire) with the instant the reminder
///   *arrived*, so a reminder set for a deadline that has since moved arrives,
///   closes nothing, and is forgotten;
/// - a [`Farewell`](Look::Farewell) reminder is the one the moderator sets
///   after announcing the outcome, and the one whose arrival **stops
///   everybody**. It has to be told apart from a session reminder rather than
///   recognized by arriving after the outcome, because the stale session
///   reminders still on the timer when the game ends arrive after the outcome
///   too — and one of them arriving first would cut the farewell's wait short,
///   which is the wait the survivors' last narration depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Look {
    /// Check whether a session's clock has run out.
    Session {
        /// The round whose clocks the moderator was watching when it set it.
        round: Round,
        /// Which half of that round.
        phase: Phase,
    },
    /// The game is over and its last narrations have had time to arrive: stop
    /// everybody.
    Farewell,
}

/// A true statement from the moderator to the players it is addressed to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Narration {
    /// To one player at the start: its role and, for a werewolf, the pack.
    Assigned {
        /// The recipient's role.
        role: Role,
        /// The werewolves. Non-empty only when the recipient is one of them:
        /// no message naming the pack is ever addressed to anyone else.
        pack: BTreeSet<ActorId>,
    },
    /// To the living: a phase has begun.
    PhaseBegan {
        /// The round the phase belongs to.
        round: Round,
        /// Which half of the round.
        phase: Phase,
        /// Everyone still in the game.
        living: BTreeSet<ActorId>,
    },
    /// To the seer alone: what it learned tonight.
    Investigated {
        /// The player it looked at.
        target: ActorId,
        /// The side that player is on.
        faction: Faction,
    },
    /// To the living, and to the eliminated player itself: someone is out
    /// of the game, and their role is revealed.
    Eliminated {
        /// The eliminated player.
        who: ActorId,
        /// The role they held.
        role: Role,
        /// The round it happened in.
        round: Round,
        /// How it happened.
        cause: Cause,
    },
    /// To the living: the night ended with nobody dead. A save is never
    /// announced as one, and neither is a pack that selected nowhere.
    NoDeath {
        /// The round whose night it was.
        round: Round,
    },
    /// To the living: the day ran out of time without a majority, so
    /// nobody was lynched (ADR-0011).
    NoLynch {
        /// The round whose day it was.
        round: Round,
    },
    /// To everyone, living and dead: the game is over.
    Outcome(Outcome),
}

/// How a player left the game.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Cause {
    /// By the werewolves, at night.
    Devoured,
    /// By the day's vote.
    Lynched,
}

/// How a game ended. The one narration addressed to every player.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    /// The winning side, or `None` for a **stalemate**: a game that
    /// reached the day cap with nobody having won (ADR-0011).
    ///
    /// A day may end without a lynch, so a doctor who keeps saving and a
    /// village that keeps running out the clock could otherwise go on
    /// forever. A game has a winner unless it reaches the cap.
    pub winner: Option<Faction>,
    /// The round the game ended in.
    pub rounds: Round,
    /// Everyone still in the game at the end.
    pub living: BTreeSet<ActorId>,
}

/// What one of a phase's sessions is for.
///
/// A phase opens a session per kind it calls for, and a player works out
/// from its own role which of them it is a member of (ADR-0014). Nobody is
/// told: a player that has observed the phase begin already knows.
///
/// Ordered so that a night's sessions can be kept in a map: the order is
/// the declaration order below and carries no meaning of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SessionKind {
    /// Name a player to lynch. Asked of every living player by day.
    Nominate,
    /// Name a player to eat. Asked of every living werewolf at night.
    Devour,
    /// Name a player whose faction to learn. Asked of the living seer at
    /// night.
    Investigate,
    /// Name a player to shield from the werewolves. Asked of the living
    /// doctor at night.
    Protect,
}

impl SessionKind {
    /// The phase a session of this kind belongs to: nomination by day,
    /// everything else at night.
    #[must_use]
    pub const fn phase(self) -> Phase {
        match self {
            Self::Nominate => Phase::Day,
            Self::Devour | Self::Investigate | Self::Protect => Phase::Night,
        }
    }
}

/// A player selecting a target.
///
/// A member of an open session may select whenever it likes and as often as
/// it likes; its most recent selection is its vote (ADR-0011). Selecting
/// nowhere is how a member abstains, and it is the absence of a selection
/// rather than a message, which is why there is no move that means "nobody".
///
/// A selection is addressed to the moderator and to nobody else. The other
/// players who should see it are named in `seen_by`, and the moderator
/// relays it to them, as a [`Message::Relayed`] of its own, if the session it
/// names is still open. A player never sends another player anything
/// directly, so there is no path by which a selection can outlive its session
/// in somebody else's queue.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Select {
    /// The round the session belongs to.
    pub round: Round,
    /// Which of the phase's sessions this is: what the selection is for.
    pub kind: SessionKind,
    /// The player selected.
    pub target: ActorId,
    /// The other players who should see this selection, for the moderator to
    /// relay it to.
    ///
    /// Empty for a selection that is nobody else's business: the seer's
    /// investigation and the doctor's protection are between that player
    /// and the moderator. A `Devour` names the rest of the pack and a
    /// `Nominate` the rest of the living, and in neither case does it name
    /// the sender or the moderator.
    #[serde(default, skip_serializing_if = "BTreeSet::is_empty")]
    pub seen_by: BTreeSet<ActorId>,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::message::Payload;
    use crate::testing::json;

    /// Every narration variant, each with the JSON shape it serializes to.
    fn every_narration() -> Vec<(Narration, Value)> {
        vec![
            (
                Narration::Assigned {
                    role: Role::Werewolf,
                    pack: ["wanda", "wolfgang"].map(ActorId::new).into(),
                },
                json!({"Assigned": {"role": "Werewolf", "pack": ["wanda", "wolfgang"]}}),
            ),
            (
                Narration::PhaseBegan {
                    round: Round::new(1),
                    phase: Phase::Night,
                    living: ["alice", "bob"].map(ActorId::new).into(),
                },
                json!({"PhaseBegan": {"round": 1, "phase": "Night", "living": ["alice", "bob"]}}),
            ),
            (
                Narration::Investigated {
                    target: ActorId::new("bob"),
                    faction: Faction::Village,
                },
                json!({"Investigated": {"target": "bob", "faction": "Village"}}),
            ),
            (
                Narration::Eliminated {
                    who: ActorId::new("bob"),
                    role: Role::Seer,
                    round: Round::new(2),
                    cause: Cause::Lynched,
                },
                json!({"Eliminated": {"who": "bob", "role": "Seer", "round": 2, "cause": "Lynched"}}),
            ),
            (
                Narration::NoDeath {
                    round: Round::new(3),
                },
                json!({"NoDeath": {"round": 3}}),
            ),
            (
                Narration::Outcome(Outcome {
                    winner: Some(Faction::Werewolves),
                    rounds: Round::new(3),
                    living: ["wanda"].map(ActorId::new).into(),
                }),
                json!({"Outcome": {"winner": "Werewolves", "rounds": 3, "living": ["wanda"]}}),
            ),
        ]
    }

    /// One message of every kind, including every narration, each with the
    /// JSON shape it serializes to. Between them they hold an actor id in
    /// every position the type has one, so a round trip over this table is
    /// the round trip a log reader depends on.
    fn every_message() -> Vec<(Message, Value)> {
        let mut messages: Vec<_> = every_narration()
            .into_iter()
            .map(|(narration, shape)| (Message::Narration(narration), json!({"Narration": shape})))
            .collect();
        messages.push((
            Message::Select(Select {
                round: Round::new(1),
                kind: SessionKind::Devour,
                target: ActorId::new("alice"),
                seen_by: BTreeSet::new(),
            }),
            json!({"Select": {"round": 1, "kind": "Devour", "target": "alice"}}),
        ));
        messages
    }

    #[test]
    fn message_is_a_payload() {
        fn assert_payload<P: Payload>() {}
        assert_payload::<Message>();
    }

    #[test]
    fn request_kinds_have_a_phase() {
        assert_eq!(SessionKind::Nominate.phase(), Phase::Day);
        for kind in [
            SessionKind::Devour,
            SessionKind::Investigate,
            SessionKind::Protect,
        ] {
            assert_eq!(kind.phase(), Phase::Night, "{kind:?}");
        }
    }

    #[test]
    fn every_message_serializes_to_its_shape() {
        for (message, expected) in every_message() {
            assert_eq!(json(&message), expected, "{message:?}");
        }
    }

    #[test]
    fn every_message_round_trips() {
        for (message, shape) in every_message() {
            let back: Message = serde_json::from_value(shape).unwrap();
            assert_eq!(back, message);
        }
    }

    #[test]
    fn a_round_is_a_bare_integer() {
        assert_eq!(json(&Round::new(4)), json!(4));
        let round: Round = serde_json::from_value(json!(4)).unwrap();
        assert_eq!(round, Round::new(4));
    }

    #[test]
    fn a_game_starts_at_round_one() {
        assert_eq!(Round::FIRST, Round::new(1));
        assert_eq!(Round::FIRST.number(), 1);
    }

    #[test]
    fn a_round_is_followed_by_the_next_one() {
        assert_eq!(Round::FIRST.next(), Round::new(2));
        assert_eq!(Round::new(7).next(), Round::new(8));
    }

    #[test]
    fn a_round_knows_the_one_before_it_unless_it_is_the_first() {
        assert_eq!(Round::new(2).previous(), Some(Round::FIRST));
        assert_eq!(Round::FIRST.previous(), None);
    }

    #[test]
    #[should_panic(expected = "a round is counted from 1")]
    fn there_is_no_round_zero() {
        let _ = Round::new(0);
    }

    #[test]
    fn a_selection_names_its_session_and_a_bare_actor_id() {
        // A target is an agent and nothing else: no move wraps it, and
        // nothing means "nobody" (ADR-0011). The session is the round and
        // the kind, which both sides derive rather than correlate by an
        // id the moderator mints (ADR-0014).
        assert_eq!(
            json(&Select {
                round: Round::new(3),
                kind: SessionKind::Nominate,
                target: ActorId::new("alice"),
                seen_by: BTreeSet::new(),
            }),
            json!({"round": 3, "kind": "Nominate", "target": "alice"})
        );
    }

    #[test]
    fn actor_id_serializes_as_a_bare_string_and_deserializes_from_one() {
        let alice = ActorId::new("alice");
        assert_eq!(json(&alice), json!("alice"));
        let back: ActorId = serde_json::from_value(json!("alice")).unwrap();
        assert_eq!(back, alice);
    }
}
