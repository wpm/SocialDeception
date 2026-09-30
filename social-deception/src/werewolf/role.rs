//! The roles a player can hold, the sides of the game they belong to, and
//! what each role is asked to do: which sessions it is a member of, and the
//! action space it permits in each.
//!
//! A role is an enum value and its rules are its methods. There is no type
//! per role, because a type per role would carry no state the role does not:
//! the only thing four such types ever held was a [`Knowledge`] built for
//! the role, and a `Knowledge` names its own role. What each role is asked
//! is [`Role::asked_in`], and the one rule that distinguishes a role's
//! action space — the doctor's — follows from the kind of session it is
//! asked in, which follows from the role.
//!
//! # The action space, and its order
//!
//! Every action space in the game is computed by one function,
//! [`action_space`]: a target for each living player other than the player
//! itself, in sorted actor order, less the one the doctor protected last
//! night. The two universal rules fall out of that, since the target must
//! be living and no action may target the player taking it; neither is
//! strategy, and nothing can do them. The order is canonical and
//! load-bearing: an index into the vector is a stable action label, the
//! same on every run and in every episode with the same living set, which
//! is why the action space is a `Vec<ActorId>` and not a set.
//!
//! **It may be empty.** The one rule beyond the two universal ones is the
//! doctor's: it may not protect the same player on two consecutive nights,
//! so a `Protect` also excludes [`Knowledge::last_protected`]. That is a
//! rule of the game, not advice, and it is what can leave a player with
//! nowhere to go: the doctor may protect neither itself nor last night's
//! patient, which in a small enough game leaves nobody. A member with
//! nothing it may select is not asked at all — that is what used to be an
//! abstention (ADR-0011) — and no other kind can empty a space, since
//! `Nominate` and `Devour` open only while a valid target lives.
//!
//! [`Role::action_space`] is what a player computes from its own knowledge,
//! and the [`Game`](super::Game) checks every selection against the same
//! function from what it knows, so the two cannot disagree about what the
//! rules permit: a player of any kind, a language-model agent or a test
//! stub, is held to exactly the space a role computes. Asking a role for an
//! action space of a kind [`Role::asked_in`] never asks it is a bug in the
//! caller: it panics naming the role and the kind.
//!
//! Nothing here narrows an action space for strategic reasons. A werewolf's
//! `Devour` includes its living packmates, and a seer's `Investigate`
//! includes players it has already seen; whether to pick them is the
//! strategy's judgment (ADR-0005), and the tests in this module assert that
//! it stays that way.

use std::collections::BTreeSet;
use std::fmt;

use serde::{Deserialize, Serialize};

use super::knowledge::Knowledge;
use super::message::{Phase, SessionKind};
use crate::message::ActorId;

/// A player's role, dealt at the start of an episode and revealed at death.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Has no power beyond the day vote.
    Villager,
    /// Devours at night, together with the rest of the pack.
    Werewolf,
    /// Learns one player's [`Faction`] each night.
    Seer,
    /// Protects one player each night, never itself and never the same
    /// player two nights running.
    Doctor,
}

/// Which side of the game a player is on: both what a seer learns about
/// someone and what a winning side is.
///
/// The two are one type because they are one partition of the players. A
/// role that wins alone, or one that investigates falsely, would split them;
/// neither exists in this version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Faction {
    /// Everyone who is not a werewolf.
    Village,
    /// The pack.
    Werewolves,
}

impl fmt::Display for Role {
    /// The role's name as written, the same spelling it serializes as.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Villager => "Villager",
            Self::Werewolf => "Werewolf",
            Self::Seer => "Seer",
            Self::Doctor => "Doctor",
        })
    }
}

impl fmt::Display for Faction {
    /// The faction's name as written, the same spelling it serializes as.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Village => "Village",
            Self::Werewolves => "Werewolves",
        })
    }
}

impl Role {
    /// The side this role plays for. Only a werewolf is on the werewolves'.
    #[must_use]
    pub const fn faction(self) -> Faction {
        match self {
            Self::Werewolf => Faction::Werewolves,
            Self::Villager | Self::Seer | Self::Doctor => Faction::Village,
        }
    }

    /// The kind of session a living player of this role is a member of in
    /// `phase`, if any: everyone nominates by day; at night a werewolf
    /// devours, the seer investigates, the doctor protects, and a villager
    /// sleeps.
    ///
    /// This is what a player calls on itself to decide whether to act.
    /// Nobody tells it to: on hearing that a phase has begun it asks its
    /// own role what that phase wants of it, and selects if the answer is
    /// something. That is ADR-0014 — a message to an agent is a fact it
    /// conditions on, and an instruction telling a player what its own role
    /// already says is not one.
    ///
    /// It is also the one statement of who is asked what. The moderator
    /// opens a phase's sessions from it and checks an arriving selection
    /// against it, and a role computes its action space from it, so the two
    /// cannot disagree about who is a member of what.
    #[must_use]
    pub const fn asked_in(self, phase: Phase) -> Option<SessionKind> {
        match (phase, self) {
            (Phase::Day, _) => Some(SessionKind::Nominate),
            (Phase::Night, Self::Werewolf) => Some(SessionKind::Devour),
            (Phase::Night, Self::Seer) => Some(SessionKind::Investigate),
            (Phase::Night, Self::Doctor) => Some(SessionKind::Protect),
            (Phase::Night, Self::Villager) => None,
        }
    }

    /// The action space this role permits a player with `knowledge` in a
    /// session of `kind`, from what it knows: who is living, and, for a
    /// doctor, whom it protected last night.
    ///
    /// The targets arrive in canonical order, so an index into the vector is
    /// a stable action label. It may be empty — a doctor can be left with
    /// nobody it may protect — and a player whose action space is empty is
    /// not a member of the session at all, so a player that does select
    /// always had somewhere to select.
    ///
    /// This is [`action_space`] read from a player's own knowledge; the
    /// [`Game`](super::Game) reads the same function from its own state.
    ///
    /// # Panics
    ///
    /// If the kind is one this role is never asked, by [`Role::asked_in`]: a
    /// bug in the caller, not a runtime condition.
    #[must_use]
    pub fn action_space(self, knowledge: &Knowledge, kind: SessionKind) -> Vec<ActorId> {
        assert_eq!(
            self.asked_in(kind.phase()),
            Some(kind),
            "a {self} is never asked to {kind:?}"
        );
        action_space(
            &knowledge.me,
            &knowledge.living,
            kind,
            knowledge.last_protected.as_ref(),
        )
    }
}

/// The action space the rules permit `me` in a session of `kind` while
/// `living` are alive: a target for each living player other than `me`, in
/// sorted actor order, less `last_protected` if the session is a `Protect`.
///
/// This is the whole of the rules about what a player may do, and the one
/// place they are written: [`Role::action_space`] reads it from a player's
/// knowledge, and the game checks every selection against it from its own
/// state. `last_protected` is whom the doctor protected the night before, or
/// `None` for anyone else, for a doctor that selected nowhere, and for a
/// doctor on the first night.
///
/// The result may be empty, and a player whose action space is empty is not
/// asked at all.
#[must_use]
pub fn action_space(
    me: &ActorId,
    living: &BTreeSet<ActorId>,
    kind: SessionKind,
    last_protected: Option<&ActorId>,
) -> Vec<ActorId> {
    let excluded = |who: &&ActorId| {
        *who != me && !(kind == SessionKind::Protect && Some(*who) == last_protected)
    };
    living.iter().filter(excluded).cloned().collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::testing::{ME, id, ids, json, knowing, seer_knowing, target, werewolf_knowing};

    /// The knowledge of a doctor among `others` that has protected nobody.
    fn doctor<const N: usize>(others: [&str; N]) -> Knowledge {
        knowing(Role::Doctor, others)
    }

    /// Selects `target` for a `Protect`, the way a player would.
    fn protected(doctor: &mut Knowledge, target: &ActorId) {
        doctor.acted(SessionKind::Protect, target);
    }

    #[test]
    fn only_a_werewolf_is_on_the_werewolves_side() {
        assert_eq!(Role::Werewolf.faction(), Faction::Werewolves);
        for role in [Role::Villager, Role::Seer, Role::Doctor] {
            assert_eq!(role.faction(), Faction::Village, "{role:?}");
        }
    }

    #[test]
    fn everyone_nominates_by_day_and_each_power_acts_at_night() {
        for role in [Role::Villager, Role::Werewolf, Role::Seer, Role::Doctor] {
            assert_eq!(
                role.asked_in(Phase::Day),
                Some(SessionKind::Nominate),
                "{role:?}"
            );
        }
        assert_eq!(Role::Villager.asked_in(Phase::Night), None);
        assert_eq!(
            Role::Werewolf.asked_in(Phase::Night),
            Some(SessionKind::Devour)
        );
        assert_eq!(
            Role::Seer.asked_in(Phase::Night),
            Some(SessionKind::Investigate)
        );
        assert_eq!(
            Role::Doctor.asked_in(Phase::Night),
            Some(SessionKind::Protect)
        );
    }

    #[test]
    fn a_role_displays_as_its_name() {
        for role in [Role::Villager, Role::Werewolf, Role::Seer, Role::Doctor] {
            assert_eq!(json(&role), json!(role.to_string()), "{role:?}");
        }
    }

    #[test]
    fn a_faction_displays_as_its_name() {
        for faction in [Faction::Village, Faction::Werewolves] {
            assert_eq!(json(&faction), json!(faction.to_string()), "{faction:?}");
        }
    }

    #[test]
    fn roles_and_factions_serialize_as_their_names() {
        assert_eq!(json(&Role::Seer), json!("Seer"));
        assert_eq!(json(&Faction::Werewolves), json!("Werewolves"));
        let role: Role = serde_json::from_value(json!("Doctor")).unwrap();
        assert_eq!(role, Role::Doctor);
    }

    #[test]
    fn the_action_space_is_every_living_other_in_order() {
        // Given out of order, so that the order is the rules' doing.
        let others = ["carol", "alice", "bob"];
        let targets = || ["alice", "bob", "carol"].map(target).to_vec();

        let villager = knowing(Role::Villager, others);
        assert_eq!(
            Role::Villager.action_space(&villager, SessionKind::Nominate),
            targets()
        );
        let werewolf = werewolf_knowing(others, []);
        for kind in [SessionKind::Nominate, SessionKind::Devour] {
            assert_eq!(Role::Werewolf.action_space(&werewolf, kind), targets());
        }
        let seer = seer_knowing(others, []);
        for kind in [SessionKind::Nominate, SessionKind::Investigate] {
            assert_eq!(Role::Seer.action_space(&seer, kind), targets());
        }
        let doctor = doctor(others);
        for kind in [SessionKind::Nominate, SessionKind::Protect] {
            assert_eq!(Role::Doctor.action_space(&doctor, kind), targets());
        }
    }

    #[test]
    fn the_action_space_never_contains_the_dead() {
        let mut villager = knowing(Role::Villager, ["alice", "bob", "carol"]);
        villager.living.remove(&id("bob"));
        assert_eq!(
            Role::Villager.action_space(&villager, SessionKind::Nominate),
            ["alice", "carol"].map(target)
        );
    }

    #[test]
    fn the_action_space_is_the_same_across_calls_and_across_players() {
        // An index into the action space is a stable action label: the
        // same living set gives the same vector, whoever computes it.
        let others = ["dave", "bob", "alice", "carol"];
        let first = seer_knowing(others, []);
        let second = seer_knowing(others, []);
        for kind in [SessionKind::Nominate, SessionKind::Investigate] {
            let space = Role::Seer.action_space(&first, kind);
            assert_eq!(Role::Seer.action_space(&first, kind), space, "{kind:?}");
            assert_eq!(Role::Seer.action_space(&second, kind), space, "{kind:?}");
        }
    }

    #[test]
    fn a_werewolf_may_devour_its_packmates() {
        let werewolf = werewolf_knowing(["alice", "bob", "carol"], ["bob"]);
        for kind in [SessionKind::Devour, SessionKind::Nominate] {
            assert_eq!(
                Role::Werewolf.action_space(&werewolf, kind),
                ["alice", "bob", "carol"].map(target),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_seer_may_investigate_someone_it_has_already_seen() {
        let seer = seer_knowing(["alice", "bob"], ["alice"]);
        assert_eq!(
            Role::Seer.action_space(&seer, SessionKind::Investigate),
            [target("alice"), target("bob")]
        );
    }

    #[test]
    fn a_doctor_may_not_protect_the_same_player_two_nights_running() {
        let mut doctor = doctor(["alice", "bob"]);
        let protect = SessionKind::Protect;
        assert_eq!(
            Role::Doctor.action_space(&doctor, protect),
            [target("alice"), target("bob")]
        );

        protected(&mut doctor, &target("alice"));
        assert_eq!(Role::Doctor.action_space(&doctor, protect), [target("bob")]);

        // The night after, alice is available again.
        protected(&mut doctor, &target("bob"));
        assert_eq!(
            Role::Doctor.action_space(&doctor, protect),
            [target("alice")]
        );
    }

    #[test]
    fn a_doctor_may_be_left_with_nobody_it_can_protect() {
        // The action space is empty rather than holding an abstention, and
        // a player with an empty one is not asked at all (ADR-0011).
        let mut doctor = doctor(["alice"]);
        protected(&mut doctor, &target("alice"));
        assert!(
            Role::Doctor
                .action_space(&doctor, SessionKind::Protect)
                .is_empty()
        );
    }

    #[test]
    fn the_rules_are_one_function_whoever_computes_them() {
        // What a player computes from its knowledge is what the game
        // computes from its state, given the same facts.
        let mut doctor = doctor(["alice", "bob", "carol"]);
        protected(&mut doctor, &target("bob"));
        let living = ids(["alice", "bob", "carol", ME]);
        for kind in [SessionKind::Protect, SessionKind::Nominate] {
            assert_eq!(
                Role::Doctor.action_space(&doctor, kind),
                action_space(&id(ME), &living, kind, Some(&id("bob"))),
                "{kind:?}"
            );
        }
        // Somebody else's last protection is nobody else's constraint.
        assert_eq!(
            action_space(&id(ME), &living, SessionKind::Devour, Some(&id("bob"))),
            ["alice", "bob", "carol"].map(target)
        );
    }

    #[test]
    fn last_nights_protection_does_not_constrain_the_day() {
        let mut doctor = doctor(["alice", "bob"]);
        protected(&mut doctor, &target("alice"));
        assert_eq!(
            Role::Doctor.action_space(&doctor, SessionKind::Nominate),
            [target("alice"), target("bob")]
        );
    }

    #[test]
    #[should_panic(expected = "a Villager is never asked to Devour")]
    fn a_villager_is_never_asked_to_devour() {
        let villager = knowing(Role::Villager, ["alice"]);
        let _ = Role::Villager.action_space(&villager, SessionKind::Devour);
    }

    #[test]
    #[should_panic(expected = "a Werewolf is never asked to Investigate")]
    fn a_werewolf_is_never_asked_to_investigate() {
        let werewolf = werewolf_knowing(["alice"], []);
        let _ = Role::Werewolf.action_space(&werewolf, SessionKind::Investigate);
    }

    #[test]
    #[should_panic(expected = "a Seer is never asked to Protect")]
    fn a_seer_is_never_asked_to_protect() {
        let seer = seer_knowing(["alice"], []);
        let _ = Role::Seer.action_space(&seer, SessionKind::Protect);
    }

    #[test]
    #[should_panic(expected = "a Doctor is never asked to Devour")]
    fn a_doctor_is_never_asked_to_devour() {
        let _ = Role::Doctor.action_space(&doctor(["alice"]), SessionKind::Devour);
    }
}
