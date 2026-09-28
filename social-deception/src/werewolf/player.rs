//! A player as an agent: a role's rules, the policy that decides for it, and
//! the one handler body every role shares.
//!
//! A player's turn is a fold, the same as any agent's: every observation
//! is folded into its [`Knowledge`], and a phase beginning is what makes it
//! act.
//!
//! **Nobody asks it to.** On observing that a phase has begun, a living
//! player asks its own role what that phase wants of it
//! ([`Role::asked_in`](super::Role::asked_in)) and points if the answer is
//! something. That is ADR-0014: an event to an agent is a fact it
//! conditions on, and an instruction telling a player what its own role
//! already says is not one. In the live game nobody waits to be asked — the
//! moderator says night has fallen and the werewolves point, because they
//! are werewolves and it is night.
//!
//! A seat has no opening move: it says nothing until the first phase
//! begins, so it needs no [`start`](Handler::start). The two halves of
//! deciding are kept apart, and ADR-0005 says why: a [`Player`] is a role,
//! and computes the action space the rules permit it and nothing else; a
//! [`Policy`] is the strategy, and picks one target from that space or
//! none. The roles are in [`roles`](super::roles), the baseline policy in
//! [`policy`](super::policy).
//!
//! [`Seat`] joins the two. It is the [`Handler`] the episode runs, the same
//! for every role, and it is where a target outside the action space is
//! caught: a policy that returns one has a bug, and the game cannot
//! continue from it.
//!
//! # Who a point is addressed to
//!
//! A point is not a sealed ballot. Under ADR-0011 pointing is how a pack
//! agrees on a victim without speaking, and how a village's argument turns
//! into a vote, so a point goes to everyone the rules let see it:
//!
//! | Kind | Recipients |
//! |---|---|
//! | `Devour` | the moderator and every other living member of the pack |
//! | `Investigate`, `Protect` | the moderator |
//! | `Nominate` | the moderator and every other living player |
//!
//! A seat addresses each point itself, from its own [`Knowledge`]: the pack
//! it was told at the deal, and the living it learns from each
//! `PhaseBegan`. It never broadcasts, because a broadcast reaches every
//! agent in the roster and the night's secrets are exactly what must not
//! travel that far.

use super::WerewolfDomain;
use super::knowledge::Knowledge;
use super::message::{Message, Narration, Point, RequestKind, Round};
use super::policy::{Policy, View};
use crate::agent::{self, Handler, Observation};
use crate::event::AgentId;

/// What a role contributes to a player: its state, and the moves the
/// rules permit it.
///
/// Implemented once per role by the types in [`roles`](super::roles). A
/// role computes its action space and nothing else; see the
/// [module documentation](self).
pub trait Player {
    /// The state: the fold of every observation this player has received.
    fn knowledge(&self) -> &Knowledge;

    /// The state, to fold the next observation into.
    fn knowledge_mut(&mut self) -> &mut Knowledge;

    /// The action space for a session of this kind: the targets the rules
    /// permit, in sorted agent order, so that an index into it is a stable
    /// action label.
    ///
    /// It may be empty — a doctor can be left with nobody it may protect —
    /// and a player whose action space is empty is not a member of the
    /// session at all, so a player that does point always had somewhere to
    /// point.
    ///
    /// # Panics
    ///
    /// If the kind is one this role is never asked, which is a bug in the
    /// caller rather than a runtime condition.
    fn action_space(&self, kind: RequestKind) -> Vec<AgentId>;
}

/// A player as an agent in the episode: a role, the policy that decides for
/// it, and the moderator it addresses its points to.
///
/// One type for every role, because the handler body is the same for all of
/// them: fold each event into the role's state and, when a phase begins a
/// session this role is a member of, point where the policy chooses from
/// the role's action space.
#[derive(Debug, Clone)]
pub struct Seat<R, P> {
    player: R,
    policy: P,
    moderator: AgentId,
}

impl<R: Player, P: Policy> Seat<R, P> {
    /// A seat for `player`, deciding with `policy`, pointing to
    /// `moderator`.
    #[must_use]
    pub const fn new(player: R, policy: P, moderator: AgentId) -> Self {
        Self {
            player,
            policy,
            moderator,
        }
    }

    /// The point this player makes in the session of `kind` now open, if
    /// the policy names a target: its choice from the role's action
    /// space, checked against it and folded into the role's state.
    ///
    /// `None` is a policy declining to point for now, which is how a
    /// member abstains, and it is also what an empty action space leaves
    /// it: a doctor with nobody it may protect points nowhere. Nothing is
    /// sent and nothing is folded either way — a point never made is not
    /// a vote.
    fn point(&mut self, round: Round, kind: RequestKind) -> Option<Point> {
        let action_space = self.player.action_space(kind);
        let chosen = self.policy.choose(View {
            knowledge: self.player.knowledge(),
            kind,
            action_space: &action_space,
        })?;
        assert!(
            action_space.contains(&chosen),
            "{}'s policy chose {chosen:?}, which is outside the action space {action_space:?}",
            self.player.knowledge().me
        );
        self.player.knowledge_mut().acted(kind, &chosen);
        Some(Point {
            round,
            kind,
            target: chosen,
        })
    }

    /// Who sees a point of this kind, besides the moderator: the pack for a
    /// `Devour`, every other living player for a `Nominate`, and nobody for
    /// the seer's and the doctor's own business.
    ///
    /// Always from the seat's own knowledge, and always without itself: a
    /// player does not observe its own actions, and the router forbids an
    /// agent addressing one to itself.
    fn audience(&self, kind: RequestKind) -> Vec<AgentId> {
        let knowledge = self.player.knowledge();
        let me = &knowledge.me;
        let seen_by = match kind {
            RequestKind::Devour => &knowledge.pack,
            RequestKind::Nominate => &knowledge.living,
            RequestKind::Investigate | RequestKind::Protect => return vec![self.moderator.clone()],
        };
        seen_by
            .iter()
            .filter(|who| *who != me && knowledge.living.contains(*who))
            .cloned()
            .chain([self.moderator.clone()])
            .collect()
    }
}

impl<R: Player, P: Policy> Handler<WerewolfDomain> for Seat<R, P> {
    /// Folds the observation into the role's state and, if it was a phase
    /// beginning a session this role is a member of, points from the state
    /// every earlier observation produced, this one included. Every other
    /// observation produces nothing but the fold.
    ///
    /// The fold comes first for exactly that reason: the phase's own
    /// narration says who is still living, and the point has to be decided
    /// from the living set that includes it.
    ///
    /// # Panics
    ///
    /// If the policy chooses a target outside the action space. It cannot
    /// panic for the other reason [`Player::action_space`] gives, since the
    /// kind it passes came from this role's own
    /// [`Role::asked_in`](super::Role::asked_in).
    fn handle(
        &mut self,
        observation: &Observation<WerewolfDomain>,
    ) -> Vec<agent::Action<WerewolfDomain>> {
        self.player.knowledge_mut().observe(observation);
        match &observation.event.payload {
            // A phase beginning is what makes a player act, and it acts
            // on its own role rather than on anybody's instruction
            // (ADR-0014). A player the phase asks nothing of, and one
            // the rules leave nowhere to point, both say nothing.
            Message::Narration(Narration::PhaseBegan { round, phase, .. }) => {
                let round = *round;
                let Some(kind) = self.player.knowledge().role.asked_in(*phase) else {
                    return Vec::new();
                };
                self.point(round, kind)
                    .map(|point| agent::Action::to(self.audience(kind), Message::Point(point)))
                    .into_iter()
                    .collect()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::clock::Timestamp;

    use super::*;
    use crate::agent::Recipients;
    use crate::event::Event;
    use crate::testing::{ME, id, ids, narrated, observed, phase_began, target};
    use crate::werewolf::message::{Cause, Narration, Phase, RequestKind, Round};
    use crate::werewolf::role::Role;
    use crate::werewolf::roles::{Doctor, Villager};

    use crate::agent::Action;

    const MODERATOR: &str = "moderator";

    /// Points at the first target in the action space.
    struct First;

    impl Policy for First {
        fn choose(&mut self, view: View<'_>) -> Option<AgentId> {
            view.action_space.first().cloned()
        }
    }

    /// Points at the last target in the action space.
    struct Last;

    impl Policy for Last {
        fn choose(&mut self, view: View<'_>) -> Option<AgentId> {
            view.action_space.last().cloned()
        }
    }

    /// Points at a player who is not in the game at all.
    struct Outside;

    impl Policy for Outside {
        fn choose(&mut self, _: View<'_>) -> Option<AgentId> {
            Some(target("nobody"))
        }
    }

    /// Points nowhere, which is how a member abstains.
    struct Nowhere;

    impl Policy for Nowhere {
        fn choose(&mut self, _: View<'_>) -> Option<AgentId> {
            None
        }
    }

    fn eliminated(who: &str, round: u32) -> Event<WerewolfDomain> {
        narrated(Narration::Eliminated {
            who: id(who),
            role: Role::Villager,
            round: Round(round),
            cause: Cause::Devoured,
        })
    }

    /// The action `Seat` takes when a phase begins: a point naming its own
    /// session, addressed to `seen_by` and always to the moderator.
    ///
    /// There is no request to echo (ADR-0014), so a point is identified by
    /// the round and the kind it was made in, which is what the reader of a
    /// trajectory reads off it directly.
    fn pointing<const N: usize>(
        round: u32,
        kind: RequestKind,
        target: AgentId,
        seen_by: [&str; N],
    ) -> Action<WerewolfDomain> {
        let mut to: Vec<AgentId> = seen_by.iter().map(|who| AgentId::new(*who)).collect();
        to.push(AgentId::new(MODERATOR));
        Action::to(
            to,
            Message::Point(Point {
                round: Round(round),
                kind,
                target,
            }),
        )
    }

    /// A point of a kind only the moderator sees: the seer's and the
    /// doctor's own business.
    fn privately(round: u32, kind: RequestKind, target: AgentId) -> Action<WerewolfDomain> {
        pointing(round, kind, target, [])
    }

    /// What a seat does with a run of events, each in a cycle of its own,
    /// as the loop hands them over (ADR-0008): every action they produced,
    /// in order.
    fn handling<R: Player, P: Policy, const N: usize>(
        seat: &mut Seat<R, P>,
        events: [Event<WerewolfDomain>; N],
    ) -> Vec<Action<WerewolfDomain>> {
        events
            .into_iter()
            .flat_map(|event| seat.handle(&observed(event)))
            .collect()
    }

    fn villager<P: Policy>(policy: P) -> Seat<Villager, P> {
        Seat::new(Villager::new(id(ME)), policy, id(MODERATOR))
    }

    fn doctor<P: Policy>(policy: P) -> Seat<Doctor, P> {
        Seat::new(Doctor::new(id(ME)), policy, id(MODERATOR))
    }

    #[test]
    fn a_policy_that_points_nowhere_sends_nothing() {
        // Pointing nowhere is how a member abstains (ADR-0011). Nothing is
        // sent, and nothing is folded: a point never made is not a vote,
        // so the doctor has no protection to be kept from repeating.
        let mut seat = doctor(Nowhere);
        let silent = handling(
            &mut seat,
            [phase_began(1, Phase::Night, ids(["alice", "bob", ME]))],
        );
        assert!(silent.is_empty(), "{silent:?}");
        assert_eq!(seat.player.knowledge().last_protected, None);
    }

    #[test]
    fn each_phase_is_acted_on_from_the_state_every_earlier_observation_left() {
        // What makes a player point is the phase beginning, not anybody
        // asking it to (ADR-0014). The point it makes is a function of
        // everything folded in before that phase began, so the second day
        // is pointed in from a village bob has left.
        let mut seat = villager(Last);
        let cycle = [
            narrated(Narration::Assigned {
                role: Role::Villager,
                pack: BTreeSet::new(),
            }),
            phase_began(1, Phase::Day, ids(["alice", "bob", ME])),
            eliminated("bob", 1),
            phase_began(2, Phase::Day, ids(["alice", ME])),
        ];
        assert_eq!(
            handling(&mut seat, cycle),
            [
                pointing(1, RequestKind::Nominate, target("bob"), ["alice", "bob"]),
                pointing(2, RequestKind::Nominate, target("alice"), ["alice"]),
            ]
        );
    }

    #[test]
    fn an_observation_that_is_not_a_phase_beginning_produces_nothing_and_still_updates_the_state() {
        let mut seat = villager(Last);
        let silent = handling(
            &mut seat,
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", ME])),
                eliminated("bob", 1),
            ],
        );
        // A villager has nothing to do at night, and an elimination asks
        // nothing of anybody; both are folded in silently.
        assert!(silent.is_empty(), "{silent:?}");

        // The elimination folded in the silent cycle shapes the next point.
        assert_eq!(
            handling(&mut seat, [phase_began(2, Phase::Day, ids(["alice", ME]))]),
            [pointing(
                2,
                RequestKind::Nominate,
                target("alice"),
                ["alice"]
            )]
        );
    }

    #[test]
    fn a_seat_opens_with_nothing() {
        // A player says nothing until the first phase begins, so the
        // default start hook is the right one for every role.
        assert!(villager(First).start(Timestamp::default()).is_empty());
        assert!(doctor(First).start(Timestamp::default()).is_empty());
    }

    #[test]
    fn a_timeout_produces_nothing() {
        // What a timeout cycle looks like from inside a handler: the loop
        // calls `timeout`, not `handle`, and a player has nothing to say on
        // a deadline.
        let mut seat = villager(Last);
        assert!(seat.timeout(Timestamp::default()).is_empty());
    }

    #[test]
    fn every_action_is_a_response_addressed_to_the_moderator_alone() {
        let mut seat = doctor(First);
        let actions = handling(
            &mut seat,
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", "carol", ME])),
                phase_began(1, Phase::Day, ids(["alice", "bob", "carol", ME])),
            ],
        );
        assert_eq!(actions.len(), 2);
        for action in &actions {
            assert!(matches!(action.payload, Message::Point(_)), "{action:?}");
        }
        // The protect is the moderator's alone; the nomination is public.
        assert_eq!(actions[0].recipients, Recipients::To(ids([MODERATOR])));
        assert_eq!(
            actions[1].recipients,
            Recipients::To(ids([MODERATOR, "alice", "bob", "carol"]))
        );
    }

    #[test]
    fn what_was_chosen_is_folded_into_the_state() {
        // The doctor may not protect the same player two nights running,
        // and it is the seat that folds each protection into its knowledge.
        let mut seat = doctor(First);
        let night = |round| [phase_began(round, Phase::Night, ids(["alice", "bob", ME]))];
        assert_eq!(
            handling(&mut seat, night(1)),
            [privately(1, RequestKind::Protect, target("alice"))]
        );
        assert_eq!(
            handling(&mut seat, night(2)),
            [privately(2, RequestKind::Protect, target("bob"))]
        );
        assert_eq!(
            handling(&mut seat, night(3)),
            [privately(3, RequestKind::Protect, target("alice"))]
        );
    }

    #[test]
    #[should_panic(
        expected = "me's policy chose AgentId(\"nobody\"), which is outside the action space"
    )]
    fn an_action_outside_the_action_space_panics() {
        let mut seat = villager(Outside);
        handling(&mut seat, [phase_began(1, Phase::Day, ids(["alice", ME]))]);
    }

    #[test]
    fn a_phase_the_role_is_asked_nothing_in_passes_in_silence() {
        // Under ADR-0014 a role that is asked nothing in a phase cannot be
        // handed a request of the wrong kind, because nothing is handed to
        // it at all: it reads its own role and says nothing. A villager at
        // night never reaches its action space, so the panic that guarded
        // against a miscast request has no path through a seat left. The
        // assertion itself is still exercised, by calling `action_space`
        // directly; `roles` has those tests.
        let mut seat = villager(First);
        let silent = handling(
            &mut seat,
            [phase_began(1, Phase::Night, ids(["alice", ME]))],
        );
        assert!(silent.is_empty(), "{silent:?}");
    }
}
