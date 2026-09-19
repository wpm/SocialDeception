//! One type per role, each enforcing its own rules: which requests it
//! answers, and the action space it permits for each.
//!
//! Each type is a [`Player`]: its state is a [`Knowledge`] constructed for
//! its role, so that the `Assigned` narration the moderator later sends is
//! checked against the type that was built, and a miswired roster fails at
//! the start of the episode rather than producing a plausible game. A
//! werewolf takes nothing extra at construction: its pack arrives in that
//! same narration and lands in its knowledge.
//!
//! Every action space is the [`base_action_space`], and the roles differ
//! only in which requests they may be asked:
//!
//! | Role | Answers |
//! |---|---|
//! | [`Villager`] | `Nominate` |
//! | [`Werewolf`] | `Nominate`, `Devour` |
//! | [`Seer`] | `Nominate`, `Investigate` |
//! | [`Doctor`] | `Nominate`, `Protect` |
//!
//! A request of any other kind is a bug in the moderator, and the role
//! panics naming itself and the kind. The one role-specific rule in the
//! game is the doctor's: it may not protect the same player on two
//! consecutive nights, so its `Protect` action space also excludes whoever
//! it protected last night. That is a rule of the variant, not advice, and
//! it is why `Abstain` has to be in that action space: with few players
//! living, the base set minus last night's target can be empty.
//!
//! Nothing here narrows an action space for strategic reasons. A werewolf's
//! `Devour` includes its living packmates, and a seer's `Investigate`
//! includes players it has already seen; whether to pick them is the
//! policy's judgement, and the tests in this module assert that it stays
//! that way.

use super::knowledge::Knowledge;
use super::message::{Action, Request, RequestKind};
use super::player::{Player, base_action_space};
use super::role::Role;
use crate::event::AgentId;

/// The action space for a request of a kind `role` is never asked: a bug in
/// the moderator.
fn never_asked(role: Role, kind: RequestKind) -> ! {
    panic!("a {role} is never asked to {kind:?}")
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
        match request.kind {
            RequestKind::Nominate => base_action_space(&self.knowledge, request.kind),
            kind => never_asked(Role::Villager, kind),
        }
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
        match request.kind {
            RequestKind::Nominate | RequestKind::Devour => {
                base_action_space(&self.knowledge, request.kind)
            }
            kind => never_asked(Role::Werewolf, kind),
        }
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
        match request.kind {
            RequestKind::Nominate | RequestKind::Investigate => {
                base_action_space(&self.knowledge, request.kind)
            }
            kind => never_asked(Role::Seer, kind),
        }
    }
}

/// A player that protects one player each night, never itself and never
/// the same player two nights running.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doctor {
    knowledge: Knowledge,
    /// Whom the doctor protected last night, if anyone: the one player its
    /// `Protect` action space excludes tonight.
    last_protected: Option<AgentId>,
}

impl Doctor {
    /// A doctor named `me` that has observed nothing yet and protected
    /// nobody.
    #[must_use]
    pub fn new(me: AgentId) -> Self {
        Self {
            knowledge: Knowledge::new(me, Role::Doctor),
            last_protected: None,
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

    /// For `Protect`, the base action space minus whoever it protected last
    /// night. `Abstain` is always there, so the space is never empty.
    fn action_space(&self, request: &Request) -> Vec<Action> {
        let space = base_action_space(&self.knowledge, request.kind);
        match request.kind {
            RequestKind::Nominate => space,
            RequestKind::Protect => {
                let repeat = self.last_protected.clone().map(Action::Target);
                space
                    .into_iter()
                    .filter(|action| Some(action) != repeat.as_ref())
                    .collect()
            }
            kind => never_asked(Role::Doctor, kind),
        }
    }

    /// Remembers the target of a `Protect`, and forgets it on an abstain,
    /// since there was no protection to repeat.
    fn chose(&mut self, request: &Request, action: &Action) {
        if request.kind == RequestKind::Protect {
            self.last_protected = action.target().cloned();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{id, ids, target};
    use crate::werewolf::message::{RequestId, Round};
    use crate::werewolf::role::Faction;

    const ME: &str = "me";

    fn request(kind: RequestKind) -> Request {
        Request {
            id: RequestId(1),
            round: Round(1),
            kind,
        }
    }

    /// `player`, with `others` and itself living. Sorted on purpose out of
    /// order, so that the order of the action space is the rules' doing.
    fn among<R: Player, const N: usize>(mut player: R, others: [&str; N]) -> R {
        let knowledge = player.knowledge_mut();
        knowledge.living = ids(others);
        knowledge.living.insert(id(ME));
        player
    }

    fn villager<const N: usize>(others: [&str; N]) -> Villager {
        among(Villager::new(id(ME)), others)
    }

    fn werewolf<const N: usize, const P: usize>(others: [&str; N], pack: [&str; P]) -> Werewolf {
        let mut werewolf = among(Werewolf::new(id(ME)), others);
        werewolf.knowledge.pack = ids(pack);
        werewolf.knowledge.pack.insert(id(ME));
        werewolf
    }

    fn seer<const N: usize, const S: usize>(others: [&str; N], investigated: [&str; S]) -> Seer {
        let mut seer = among(Seer::new(id(ME)), others);
        seer.knowledge.investigations = ids(investigated)
            .into_iter()
            .map(|who| (who, Faction::Village))
            .collect();
        seer
    }

    fn doctor<const N: usize>(others: [&str; N]) -> Doctor {
        among(Doctor::new(id(ME)), others)
    }

    /// Answers a `Protect` with `action`, the way a seat would.
    fn protected(doctor: &mut Doctor, action: &Action) {
        doctor.chose(&request(RequestKind::Protect), action);
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
    fn last_nights_protection_does_not_constrain_the_day() {
        let mut doctor = doctor(["alice", "bob"]);
        protected(&mut doctor, &target("alice"));
        assert_eq!(
            doctor.action_space(&request(RequestKind::Nominate)),
            [target("alice"), target("bob")]
        );
    }

    #[test]
    fn only_a_protect_is_remembered() {
        let mut doctor = doctor(["alice", "bob"]);
        doctor.chose(&request(RequestKind::Nominate), &target("alice"));
        assert_eq!(doctor.last_protected, None);
        assert_eq!(
            doctor.action_space(&request(RequestKind::Protect)),
            [target("alice"), target("bob"), Action::Abstain]
        );
    }

    #[test]
    fn choosing_is_nothing_to_remember_for_the_other_roles() {
        let mut villager = villager(["alice"]);
        let mut werewolf = werewolf(["alice"], []);
        let mut seer = seer(["alice"], []);
        let before = (villager.clone(), werewolf.clone(), seer.clone());
        villager.chose(&request(RequestKind::Nominate), &target("alice"));
        werewolf.chose(&request(RequestKind::Devour), &target("alice"));
        seer.chose(&request(RequestKind::Investigate), &target("alice"));
        assert_eq!((villager, werewolf, seer), before);
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
