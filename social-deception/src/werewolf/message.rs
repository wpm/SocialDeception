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

/// The identity of one [`Request`], echoed by the [`Response`] to it.
///
/// The id makes "is this response an answer to something asked?" an exact
/// check rather than an inferred one, and lets late and duplicate responses
/// be recognized once a policy can be slow. Serializes as a bare integer.
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
    /// Player to moderator: the move chosen.
    Response(Response),
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
    /// How a phase's requests were answered: the day's tally to the living,
    /// the night's to the living werewolves alone.
    Tally {
        /// The round the tally belongs to.
        round: Round,
        /// Which half of the round.
        phase: Phase,
        /// Each responding player's move, in canonical order.
        votes: BTreeMap<AgentId, Move>,
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

/// A decision point: the moderator asking one player to act.
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

    /// Whether [`Move::Abstain`] is in the action space for this kind:
    /// true for `Protect` and `Investigate` only.
    ///
    /// `Nominate` and `Devour` always have at least one valid target when
    /// they are asked, because the game would already be over otherwise.
    /// Protection can genuinely run out of targets, since the doctor may
    /// protect neither itself nor the player it protected last night, so
    /// abstaining has to exist.
    #[must_use]
    pub const fn may_abstain(self) -> bool {
        match self {
            Self::Investigate | Self::Protect => true,
            Self::Nominate | Self::Devour => false,
        }
    }
}

/// A player's reply to a request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Response {
    /// The id of the request being answered.
    pub request: RequestId,
    /// The move chosen.
    pub chosen: Move,
}

/// The move a player's action carries: one of these, drawn from the action
/// space, is what a policy returns and what a [`Response`] reports.
///
/// The derived ordering puts every `Target` before `Abstain`, targets in
/// agent-id order, which is the order the action space lists them in.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Move {
    /// Act on this player. Serializes as `{"Target": "<agent id>"}`.
    Target(AgentId),
    /// Decline to act. In the action space only where
    /// [`RequestKind::may_abstain`] says so.
    Abstain,
}

impl Move {
    /// The player this move targets, if it is not an abstention.
    #[must_use]
    pub const fn target(&self) -> Option<&AgentId> {
        match self {
            Self::Target(who) => Some(who),
            Self::Abstain => None,
        }
    }
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
                    votes: BTreeMap::from([
                        (AgentId::new("alice"), Move::Target(AgentId::new("bob"))),
                        (AgentId::new("bob"), Move::Abstain),
                    ]),
                },
                json!({"Tally": {
                    "round": 2,
                    "phase": "Day",
                    "votes": {"alice": {"Target": "bob"}, "bob": "Abstain"},
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
            Message::Response(Response {
                request: RequestId(7),
                chosen: Move::Target(AgentId::new("alice")),
            }),
            json!({"Response": {"request": 7, "chosen": {"Target": "alice"}}}),
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
    fn only_protect_and_investigate_may_abstain() {
        assert!(RequestKind::Protect.may_abstain());
        assert!(RequestKind::Investigate.may_abstain());
        assert!(!RequestKind::Nominate.may_abstain());
        assert!(!RequestKind::Devour.may_abstain());
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
    fn a_target_carries_a_bare_agent_id() {
        assert_eq!(
            json(&Move::Target(AgentId::new("alice"))),
            json!({"Target": "alice"})
        );
        assert_eq!(json(&Move::Abstain), json!("Abstain"));
    }

    #[test]
    fn only_a_target_names_a_player() {
        assert_eq!(
            Move::Target(AgentId::new("alice")).target(),
            Some(&AgentId::new("alice"))
        );
        assert_eq!(Move::Abstain.target(), None);
    }

    #[test]
    fn targets_sort_by_agent_and_precede_abstain() {
        let mut moves = vec![
            Move::Abstain,
            Move::Target(AgentId::new("bob")),
            Move::Target(AgentId::new("alice")),
        ];
        moves.sort();
        assert_eq!(
            moves,
            [
                Move::Target(AgentId::new("alice")),
                Move::Target(AgentId::new("bob")),
                Move::Abstain,
            ]
        );
    }

    #[test]
    fn a_tally_serializes_its_voters_in_sorted_order() {
        let tally = Narration::Tally {
            round: Round(1),
            phase: Phase::Night,
            votes: ["carol", "alice", "bob"]
                .into_iter()
                .map(|who| (AgentId::new(who), Move::Target(AgentId::new("dave"))))
                .collect(),
        };
        // A `serde_json::Value` object sorts its own keys, so the order has
        // to be checked on the text.
        assert_eq!(
            serde_json::to_string(&tally).unwrap(),
            r#"{"Tally":{"round":1,"phase":"Night","votes":{"alice":{"Target":"dave"},"bob":{"Target":"dave"},"carol":{"Target":"dave"}}}}"#
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
