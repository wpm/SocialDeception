//! One type per role, each enforcing its own rules: which requests it
//! answers, and the action space it permits for each.
//!
//! Each type is a [`Player`] whose state is a [`Knowledge`] constructed for
//! its role, so that the `Assigned` narration the moderator later sends is
//! checked against the type that was built, and a miswired roster fails at
//! the start of the episode rather than producing a plausible game. A
//! werewolf takes nothing extra at construction: its pack arrives in that
//! same narration and lands in its knowledge.
//!
//! # The action space, and its order
//!
//! Every action space in the game is computed by one function,
//! [`action_space`]: a target for each living player other than the agent
//! itself, in sorted agent order, less the one the doctor protected last
//! night, then [`Action::Abstain`] exactly where [`RequestKind::may_abstain`]
//! permits it. The two universal rules fall out of that, since the target
//! must be living and no action may target the agent taking it; neither is
//! strategy, and nothing can do them. The order is canonical and
//! load-bearing: an index into the vector is a stable action label, the
//! same on every run and in every episode with the same living set, which
//! is why the action space is a `Vec<Action>` and not a set.
//!
//! The roles here compute their action spaces from what they know, and the
//! [`Game`](super::Game) checks every response against the same function
//! from what it knows, so the two cannot disagree about what the rules
//! permit: a player that is not one of these types, a language-model agent
//! or a test stub, is held to exactly the space these types compute.
//!
//! Which requests a role answers is [`Role::asked_in`]'s to say, and a
//! request of any other kind is a bug in the moderator: the role panics
//! naming itself and the kind. The one role-specific rule in the game is
//! the doctor's: it may not protect the same player on two consecutive
//! nights, so its `Protect` action space also excludes
//! [`Knowledge::last_protected`]. That is a rule of the variant, not
//! advice, and it is why `Abstain` has to be in that action space: with few
//! players living, the targets minus last night's can be empty.
//!
//! The action space is never empty when a request is legitimately issued:
//! `Nominate` and `Devour` are only asked while at least one valid target
//! lives, since the game would be over otherwise, and `Protect` and
//! `Investigate` always have `Abstain`.
//!
//! Nothing here narrows an action space for strategic reasons. A werewolf's
//! `Devour` includes its living packmates, and a seer's `Investigate`
//! includes players it has already seen; whether to pick them is the
//! policy's judgement (ADR-0005), and the tests in this module assert that
//! it stays that way.

use std::collections::BTreeSet;

use super::knowledge::Knowledge;
use super::message::{Action, Request, RequestKind};
use super::player::Player;
use super::role::Role;
use crate::event::AgentId;

/// The action space the rules permit `me` for a request of `kind` while
/// `living` are alive: a target for each living player other than `me`, in
/// sorted agent order, less `last_protected` if the request is a `Protect`,
/// then [`Action::Abstain`] if and only if the kind
/// [may be abstained from](RequestKind::may_abstain).
///
/// This is the whole of the rules about what a player may do, and the one
/// place they are written: the roles below compute their action spaces with
/// it from their knowledge, and the game checks every response against it
/// from its own state. `last_protected` is whom the doctor protected the
/// night before, or `None` for anyone else, for a doctor that abstained,
/// and for a doctor on the first night.
#[must_use]
pub fn action_space(
    me: &AgentId,
    living: &BTreeSet<AgentId>,
    kind: RequestKind,
    last_protected: Option<&AgentId>,
) -> Vec<Action> {
    let excluded = |who: &&AgentId| {
        *who != me && !(kind == RequestKind::Protect && Some(*who) == last_protected)
    };
    let mut space: Vec<Action> = living
        .iter()
        .filter(excluded)
        .cloned()
        .map(Action::Target)
        .collect();
    if kind.may_abstain() {
        space.push(Action::Abstain);
    }
    space
}

/// The action space [`action_space`] permits a player with `knowledge` for
/// `request`, from what it knows: who is living, and, for a doctor, whom it
/// protected last night.
///
/// # Panics
///
/// If the request is of a kind the player's role is never asked, by
/// [`Role::asked_in`]: a bug in the moderator, not a runtime condition.
#[must_use]
pub fn base_action_space(knowledge: &Knowledge, request: &Request) -> Vec<Action> {
    let kind = request.kind;
    assert!(
        knowledge.role.asked_in(kind.phase()) == Some(kind),
        "a {} is never asked to {kind:?}",
        knowledge.role
    );
    action_space(
        &knowledge.me,
        &knowledge.living,
        kind,
        knowledge.last_protected.as_ref(),
    )
}

/// A player with no power beyond the day vote.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Villager {
    knowledge: Knowledge,
}

impl Villager {
    /// A villager named `me` that has observed nothing yet.
    #[must_use]
    pub fn new(me: AgentId) -> Self {
        Self {
            knowledge: Knowledge::new(me, Role::Villager),
        }
    }
}

impl Player for Villager {
    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn knowledge_mut(&mut self) -> &mut Knowledge {
        &mut self.knowledge
    }

    fn action_space(&self, request: &Request) -> Vec<Action> {
        base_action_space(&self.knowledge, request)
    }
}

/// A player that devours at night, together with the rest of its pack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Werewolf {
    knowledge: Knowledge,
}

impl Werewolf {
    /// A werewolf named `me` that has observed nothing yet. Its pack arrives
    /// in the moderator's `Assigned` narration.
    #[must_use]
    pub fn new(me: AgentId) -> Self {
        Self {
            knowledge: Knowledge::new(me, Role::Werewolf),
        }
    }
}

impl Player for Werewolf {
    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn knowledge_mut(&mut self) -> &mut Knowledge {
        &mut self.knowledge
    }

    /// Eating a packmate is in the action space; see the
    /// [module documentation](self).
    fn action_space(&self, request: &Request) -> Vec<Action> {
        base_action_space(&self.knowledge, request)
    }
}

/// A player that learns one player's faction each night.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Seer {
    knowledge: Knowledge,
}

impl Seer {
    /// A seer named `me` that has observed nothing yet.
    #[must_use]
    pub fn new(me: AgentId) -> Self {
        Self {
            knowledge: Knowledge::new(me, Role::Seer),
        }
    }
}

impl Player for Seer {
    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn knowledge_mut(&mut self) -> &mut Knowledge {
        &mut self.knowledge
    }

    /// Re-investigating someone is in the action space: permitted but
    /// pointless, and "pointless" is the policy's judgement to make.
    fn action_space(&self, request: &Request) -> Vec<Action> {
        base_action_space(&self.knowledge, request)
    }
}

/// A player that protects one player each night, never itself and never
/// the same player two nights running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doctor {
    knowledge: Knowledge,
}

impl Doctor {
    /// A doctor named `me` that has observed nothing yet and protected
    /// nobody.
    #[must_use]
    pub fn new(me: AgentId) -> Self {
        Self {
            knowledge: Knowledge::new(me, Role::Doctor),
        }
    }
}

impl Player for Doctor {
    fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    fn knowledge_mut(&mut self) -> &mut Knowledge {
        &mut self.knowledge
    }

    /// For `Protect`, everyone living but itself and whoever it protected
    /// last night, which its knowledge remembers. `Abstain` is always there,
    /// so the space is never empty.
    fn action_space(&self, request: &Request) -> Vec<Action> {
        base_action_space(&self.knowledge, request)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{ME, id, ids, knowing, request, seer_knowing, target, werewolf_knowing};

    /// `player` with `knowledge` in place of what it was constructed with.
    fn with<R: Player>(mut player: R, knowledge: Knowledge) -> R {
        *player.knowledge_mut() = knowledge;
        player
    }

    fn villager<const N: usize>(others: [&str; N]) -> Villager {
        with(Villager::new(id(ME)), knowing(Role::Villager, others))
    }

    fn werewolf<const N: usize, const P: usize>(others: [&str; N], pack: [&str; P]) -> Werewolf {
        with(Werewolf::new(id(ME)), werewolf_knowing(others, pack))
    }

    fn seer<const N: usize, const I: usize>(others: [&str; N], investigated: [&str; I]) -> Seer {
        with(Seer::new(id(ME)), seer_knowing(others, investigated))
    }

    fn doctor<const N: usize>(others: [&str; N]) -> Doctor {
        with(Doctor::new(id(ME)), knowing(Role::Doctor, others))
    }

    /// Answers a `Protect` with `action`, the way a seat would.
    fn protected(doctor: &mut Doctor, action: &Action) {
        doctor
            .knowledge_mut()
            .acted(&request(RequestKind::Protect), action);
    }

    #[test]
    fn a_role_is_constructed_knowing_who_it_is() {
        assert_eq!(Villager::new(id(ME)).knowledge().role, Role::Villager);
        assert_eq!(Werewolf::new(id(ME)).knowledge().role, Role::Werewolf);
        assert_eq!(Seer::new(id(ME)).knowledge().role, Role::Seer);
        assert_eq!(Doctor::new(id(ME)).knowledge().role, Role::Doctor);
        assert_eq!(Doctor::new(id(ME)).knowledge().me, id(ME));
    }

    #[test]
    fn the_action_space_is_every_living_other_in_order_then_abstain_where_permitted() {
        // Given out of order, so that the order is the rules' doing.
        let others = ["carol", "alice", "bob"];
        let targets = || ["alice", "bob", "carol"].map(target).to_vec();
        let with_abstain = || {
            let mut space = targets();
            space.push(Action::Abstain);
            space
        };

        assert_eq!(
            villager(others).action_space(&request(RequestKind::Nominate)),
            targets()
        );
        let werewolf = werewolf(others, []);
        assert_eq!(
            werewolf.action_space(&request(RequestKind::Nominate)),
            targets()
        );
        assert_eq!(
            werewolf.action_space(&request(RequestKind::Devour)),
            targets()
        );
        let seer = seer(others, []);
        assert_eq!(
            seer.action_space(&request(RequestKind::Nominate)),
            targets()
        );
        assert_eq!(
            seer.action_space(&request(RequestKind::Investigate)),
            with_abstain()
        );
        let doctor = doctor(others);
        assert_eq!(
            doctor.action_space(&request(RequestKind::Nominate)),
            targets()
        );
        assert_eq!(
            doctor.action_space(&request(RequestKind::Protect)),
            with_abstain()
        );
    }

    #[test]
    fn the_action_space_never_contains_the_dead() {
        let mut villager = villager(["alice", "bob", "carol"]);
        villager.knowledge_mut().living.remove(&id("bob"));
        assert_eq!(
            villager.action_space(&request(RequestKind::Nominate)),
            ["alice", "carol"].map(target)
        );
    }

    #[test]
    fn the_action_space_is_the_same_across_calls_and_across_players() {
        // An index into the action space is a stable action label: the
        // same living set gives the same vector, whoever computes it.
        let others = ["dave", "bob", "alice", "carol"];
        let first = seer(others, []);
        let second = seer(others, []);
        for kind in [RequestKind::Nominate, RequestKind::Investigate] {
            let space = first.action_space(&request(kind));
            assert_eq!(first.action_space(&request(kind)), space, "{kind:?}");
            assert_eq!(second.action_space(&request(kind)), space, "{kind:?}");
        }
    }

    #[test]
    fn a_werewolf_may_devour_its_packmates() {
        let werewolf = werewolf(["alice", "bob", "carol"], ["bob"]);
        for kind in [RequestKind::Devour, RequestKind::Nominate] {
            assert_eq!(
                werewolf.action_space(&request(kind)),
                ["alice", "bob", "carol"].map(target),
                "{kind:?}"
            );
        }
    }

    #[test]
    fn a_seer_may_investigate_someone_it_has_already_seen() {
        let seer = seer(["alice", "bob"], ["alice"]);
        assert_eq!(
            seer.action_space(&request(RequestKind::Investigate)),
            [target("alice"), target("bob"), Action::Abstain]
        );
    }

    #[test]
    fn a_doctor_may_not_protect_the_same_player_two_nights_running() {
        let mut doctor = doctor(["alice", "bob"]);
        let protect = request(RequestKind::Protect);
        assert_eq!(
            doctor.action_space(&protect),
            [target("alice"), target("bob"), Action::Abstain]
        );

        protected(&mut doctor, &target("alice"));
        assert_eq!(
            doctor.action_space(&protect),
            [target("bob"), Action::Abstain]
        );

        // The night after, alice is available again.
        protected(&mut doctor, &target("bob"));
        assert_eq!(
            doctor.action_space(&protect),
            [target("alice"), Action::Abstain]
        );

        // After an abstain there was no protection to repeat.
        protected(&mut doctor, &Action::Abstain);
        assert_eq!(
            doctor.action_space(&protect),
            [target("alice"), target("bob"), Action::Abstain]
        );
    }

    #[test]
    fn a_doctor_with_no_permitted_target_may_still_abstain() {
        let mut doctor = doctor(["alice"]);
        protected(&mut doctor, &target("alice"));
        assert_eq!(
            doctor.action_space(&request(RequestKind::Protect)),
            [Action::Abstain]
        );
    }

    #[test]
    fn the_rules_are_one_function_whoever_computes_them() {
        // What a doctor computes from its knowledge is what the game
        // computes from its state, given the same facts.
        let mut doctor = doctor(["alice", "bob", "carol"]);
        protected(&mut doctor, &target("bob"));
        let living = ids(["alice", "bob", "carol", ME]);
        for kind in [RequestKind::Protect, RequestKind::Nominate] {
            assert_eq!(
                doctor.action_space(&request(kind)),
                action_space(&id(ME), &living, kind, Some(&id("bob"))),
                "{kind:?}"
            );
        }
        // Somebody else's last protection is nobody else's constraint.
        assert_eq!(
            action_space(&id(ME), &living, RequestKind::Devour, Some(&id("bob"))),
            ["alice", "bob", "carol"].map(target)
        );
    }

    #[test]
    fn last_nights_protection_does_not_constrain_the_day() {
        let mut doctor = doctor(["alice", "bob"]);
        protected(&mut doctor, &target("alice"));
        assert_eq!(
            doctor.action_space(&request(RequestKind::Nominate)),
            [target("alice"), target("bob")]
        );
    }

    #[test]
    #[should_panic(expected = "a Villager is never asked to Devour")]
    fn a_villager_is_never_asked_to_devour() {
        villager(["alice"]).action_space(&request(RequestKind::Devour));
    }

    #[test]
    #[should_panic(expected = "a Werewolf is never asked to Investigate")]
    fn a_werewolf_is_never_asked_to_investigate() {
        werewolf(["alice"], []).action_space(&request(RequestKind::Investigate));
    }

    #[test]
    #[should_panic(expected = "a Seer is never asked to Protect")]
    fn a_seer_is_never_asked_to_protect() {
        seer(["alice"], []).action_space(&request(RequestKind::Protect));
    }

    #[test]
    #[should_panic(expected = "a Doctor is never asked to Devour")]
    fn a_doctor_is_never_asked_to_devour() {
        doctor(["alice"]).action_space(&request(RequestKind::Devour));
    }
}
