//! Everything said in a Werewolf episode: the [`Message`] payload and the
//! vocabulary of rounds, phases, requests and moves it is built from.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::role::{Faction, Role};
use crate::event::AgentId;

/// A round of the game, counted from 1. Each round is a night then a day.
///
/// Serializes as a bare integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Round(pub u32);

/// Half of a round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Phase {
    /// The werewolves devour, the seer investigates, the doctor protects.
    Night,
    /// Everyone living nominates, and the plurality is lynched.
    Day,
}

/// The identity of one [`Request`], echoed by every [`Point`] answering it.
///
/// The id makes "is this point an answer to something asked?" an exact
/// check rather than an inferred one, and it is what tells a point meant for
/// a session that has closed from one meant for the session now open.
/// Serializes as a bare integer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RequestId(pub u64);

/// Everything said in a Werewolf episode.
///
/// Every agent in an episode shares this one payload type, so it covers
/// moderator-to-player and player-to-moderator traffic alike. Serializes as
/// an object with one field, named after the variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Message {
    /// Moderator to chosen players: something they now observe.
    Narration(Narration),
    /// Moderator to one player: act now.
    Request(Request),
    /// Player to the moderator and to whoever else may see it: a target
    /// pointed at. A player may send more than one for the same request;
    /// its latest is its vote (ADR-0011).
    Point(Point),
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
        pack: BTreeSet<AgentId>,
    },
    /// To the living: a phase has begun.
    PhaseBegan {
        /// The round the phase belongs to.
        round: Round,
        /// Which half of the round.
        phase: Phase,
        /// Everyone still in the game.
        living: BTreeSet<AgentId>,
    },
    /// To the seer alone: what it learned tonight.
    Investigated {
        /// The player it looked at.
        target: AgentId,
        /// The side that player is on.
        faction: Faction,
    },
    /// How a session's members pointed when it closed, addressed to the
    /// session's observers: the day's to the living, a night session's to
    /// its own members and the moderator.
    ///
    /// A tally marks the close of the session it belongs to, which is how
    /// its members and the transcript know a point arriving later is late.
    Tally {
        /// The round the tally belongs to.
        round: Round,
        /// Which half of the round.
        phase: Phase,
        /// Which session closed, since a night has three.
        kind: RequestKind,
        /// Each member's latest target, in canonical order. A member that
        /// never pointed is absent.
        votes: BTreeMap<AgentId, AgentId>,
    },
    /// To the living, and to the eliminated player itself: someone is out
    /// of the game, and their role is revealed.
    Eliminated {
        /// The eliminated player.
        who: AgentId,
        /// The role they held.
        role: Role,
        /// The round it happened in.
        round: Round,
        /// How it happened.
        cause: Cause,
    },
    /// To the living: the night ended with nobody dead. A save is never
    /// announced as one.
    NoDeath {
        /// The round whose night it was.
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
    /// The winning side. Every game has one: a day always eliminates
    /// someone, so no game can run out of rounds undecided.
    pub winner: Faction,
    /// The round the game ended in.
    pub rounds: Round,
    /// Everyone still in the game at the end.
    pub living: BTreeSet<AgentId>,
}

/// The moderator telling one player it is a member of an open session.
///
/// Under ADR-0011 a request is not a question expecting one answer. It says
/// "you may point at any time until this session closes", and a member may
/// point as often as it likes until then; the session's clock, not the
/// answer, is what ends it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Request {
    /// The id the response must echo.
    pub id: RequestId,
    /// The round the request belongs to.
    pub round: Round,
    /// What is being asked.
    pub kind: RequestKind,
}

/// What a [`Request`] asks a player to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RequestKind {
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

impl RequestKind {
    /// The phase this kind of request is asked in: nomination by day,
    /// everything else at night.
    #[must_use]
    pub const fn phase(self) -> Phase {
        match self {
            Self::Nominate => Phase::Day,
            Self::Devour | Self::Investigate | Self::Protect => Phase::Night,
        }
    }
}

/// A player pointing at a target.
///
/// A member of an open session may point whenever it likes and as often as
/// it likes; its most recent point is its vote (ADR-0011). Pointing nowhere
/// is how a member abstains, and it is the absence of a point rather than a
/// message, which is why there is no move that means "nobody".
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Point {
    /// The id of the request this points for.
    pub request: RequestId,
    /// The player pointed at.
    pub target: AgentId,
}

#[cfg(test)]
mod tests {
    use serde_json::{Value, json};

    use super::*;
    use crate::event::Payload;
    use crate::testing::json;

    /// Every narration variant, each with the JSON shape it serializes to.
    fn every_narration() -> Vec<(Narration, Value)> {
        vec![
            (
                Narration::Assigned {
                    role: Role::Werewolf,
                    pack: ["wanda", "wolfgang"].map(AgentId::new).into(),
                },
                json!({"Assigned": {"role": "Werewolf", "pack": ["wanda", "wolfgang"]}}),
            ),
            (
                Narration::PhaseBegan {
                    round: Round(1),
                    phase: Phase::Night,
                    living: ["alice", "bob"].map(AgentId::new).into(),
                },
                json!({"PhaseBegan": {"round": 1, "phase": "Night", "living": ["alice", "bob"]}}),
            ),
            (
                Narration::Investigated {
                    target: AgentId::new("bob"),
                    faction: Faction::Village,
                },
                json!({"Investigated": {"target": "bob", "faction": "Village"}}),
            ),
            (
                Narration::Tally {
                    round: Round(2),
                    phase: Phase::Day,
                    kind: RequestKind::Nominate,
                    votes: BTreeMap::from([
                        (AgentId::new("alice"), AgentId::new("bob")),
                        (AgentId::new("bob"), AgentId::new("carol")),
                    ]),
                },
                json!({"Tally": {
                    "round": 2,
                    "phase": "Day",
                    "kind": "Nominate",
                    "votes": {"alice": "bob", "bob": "carol"},
                }}),
            ),
            (
                Narration::Eliminated {
                    who: AgentId::new("bob"),
                    role: Role::Seer,
                    round: Round(2),
                    cause: Cause::Lynched,
                },
                json!({"Eliminated": {"who": "bob", "role": "Seer", "round": 2, "cause": "Lynched"}}),
            ),
            (
                Narration::NoDeath { round: Round(3) },
                json!({"NoDeath": {"round": 3}}),
            ),
            (
                Narration::Outcome(Outcome {
                    winner: Faction::Werewolves,
                    rounds: Round(3),
                    living: ["wanda"].map(AgentId::new).into(),
                }),
                json!({"Outcome": {"winner": "Werewolves", "rounds": 3, "living": ["wanda"]}}),
            ),
        ]
    }

    /// One message of every kind, including every narration, each with the
    /// JSON shape it serializes to. Between them they hold an `AgentId` in
    /// every position the type has one, so a round trip over this table is
    /// the round trip a trajectory reader depends on.
    fn every_message() -> Vec<(Message, Value)> {
        let mut messages: Vec<_> = every_narration()
            .into_iter()
            .map(|(narration, shape)| (Message::Narration(narration), json!({"Narration": shape})))
            .collect();
        messages.push((
            Message::Request(Request {
                id: RequestId(7),
                round: Round(1),
                kind: RequestKind::Devour,
            }),
            json!({"Request": {"id": 7, "round": 1, "kind": "Devour"}}),
        ));
        messages.push((
            Message::Point(Point {
                request: RequestId(7),
                target: AgentId::new("alice"),
            }),
            json!({"Point": {"request": 7, "target": "alice"}}),
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
        assert_eq!(RequestKind::Nominate.phase(), Phase::Day);
        for kind in [
            RequestKind::Devour,
            RequestKind::Investigate,
            RequestKind::Protect,
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
    fn rounds_and_request_ids_are_bare_integers() {
        assert_eq!(json(&Round(4)), json!(4));
        assert_eq!(json(&RequestId(12)), json!(12));
        let round: Round = serde_json::from_value(json!(4)).unwrap();
        assert_eq!(round, Round(4));
        let id: RequestId = serde_json::from_value(json!(12)).unwrap();
        assert_eq!(id, RequestId(12));
    }

    #[test]
    fn a_point_carries_a_bare_agent_id() {
        // A target is an agent and nothing else: there is no longer a move
        // wrapping it, and nothing that means "nobody". Pointing nowhere is
        // the absence of a point (ADR-0011).
        assert_eq!(
            json(&Point {
                request: RequestId(3),
                target: AgentId::new("alice"),
            }),
            json!({"request": 3, "target": "alice"})
        );
    }

    #[test]
    fn a_tally_serializes_its_voters_in_sorted_order() {
        let tally = Narration::Tally {
            round: Round(1),
            phase: Phase::Night,
            kind: RequestKind::Devour,
            votes: ["carol", "alice", "bob"]
                .into_iter()
                .map(|who| (AgentId::new(who), AgentId::new("dave")))
                .collect(),
        };
        // A `serde_json::Value` object sorts its own keys, so the order has
        // to be checked on the text.
        assert_eq!(
            serde_json::to_string(&tally).unwrap(),
            r#"{"Tally":{"round":1,"phase":"Night","kind":"Devour","votes":{"alice":"dave","bob":"dave","carol":"dave"}}}"#
        );
    }

    #[test]
    fn agent_id_serializes_as_a_bare_string_and_deserializes_from_one() {
        let alice = AgentId::new("alice");
        assert_eq!(json(&alice), json!("alice"));
        let back: AgentId = serde_json::from_value(json!("alice")).unwrap();
        assert_eq!(back, alice);
    }
}
