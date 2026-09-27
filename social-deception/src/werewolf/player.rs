//! A player as an agent: a role's rules, the policy that decides for it, and
//! the one handler body every role shares.
//!
//! A player's turn is a fold, the same as any agent's: every observation is
//! folded into its [`Knowledge`], and every [`Request`] among them is
//! answered with a [`Point`]. A seat has no opening move: it says nothing
//! until it is asked, so it needs no [`start`](Handler::start). The two
//! halves of answering are kept apart, and ADR-0005 says why: a [`Player`]
//! is a role, and computes the action space the rules permit it and nothing
//! else; a [`Policy`] is the strategy, and picks one target from that space
//! or none. The roles are in [`roles`](super::roles), the baseline policy in
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
use super::message::{Message, Point, Request, RequestKind};
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

    /// The action space for this request: the targets the rules permit, in
    /// sorted agent order, so that an index into it is a stable action
    /// label.
    ///
    /// It may be empty — a doctor can be left with nobody it may protect —
    /// and a player whose action space is empty is not asked at all, so a
    /// request that arrives always has somewhere to point.
    ///
    /// # Panics
    ///
    /// If the request is of a kind this role is never asked, which is a bug
    /// in the moderator rather than a runtime condition.
    fn action_space(&self, request: &Request) -> Vec<AgentId>;
}

/// A player as an agent in the episode: a role, the policy that decides for
/// it, and the moderator it answers to.
///
/// One type for every role, because the handler body is the same for all of
/// them: fold each event into the role's state and answer each request with
/// the policy's choice from the role's action space.
#[derive(Debug, Clone)]
pub struct Seat<R, P> {
    player: R,
    policy: P,
    moderator: AgentId,
}

impl<R: Player, P: Policy> Seat<R, P> {
    /// A seat for `player`, deciding with `policy`, answering to
    /// `moderator`.
    #[must_use]
    pub const fn new(player: R, policy: P, moderator: AgentId) -> Self {
        Self {
            player,
            policy,
            moderator,
        }
    }

    /// The point to make for one request, if the policy names a target:
    /// its choice from the role's action space, checked against it and
    /// folded into the role's state.
    ///
    /// `None` is a policy declining to point for now, which is how a member
    /// abstains. Nothing is sent, and nothing is folded: a point never made
    /// is not a vote.
    fn answer(&mut self, request: &Request) -> Option<Point> {
        let action_space = self.player.action_space(request);
        let chosen = self.policy.choose(View {
            knowledge: self.player.knowledge(),
            request,
            action_space: &action_space,
        })?;
        assert!(
            action_space.contains(&chosen),
            "{}'s policy chose {chosen:?}, which is outside the action space {action_space:?}",
            self.player.knowledge().me
        );
        self.player.knowledge_mut().acted(request, &chosen);
        Some(Point {
            request: request.id,
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
    /// Folds the observation into the role's state and answers it if it was
    /// a request, so that the answer is given from the state every earlier
    /// observation produced, this one included. An observation that is not a
    /// request produces nothing but the fold.
    ///
    /// # Panics
    ///
    /// If the policy chooses a target outside the action space, or if the
    /// request is of a kind this role is never asked; see
    /// [`Player::action_space`].
    fn handle(
        &mut self,
        observation: &Observation<WerewolfDomain>,
    ) -> Vec<agent::Action<WerewolfDomain>> {
        self.player.knowledge_mut().observe(observation);
        match &observation.event.payload {
            Message::Request(request) => self
                .answer(request)
                .map(|point| agent::Action::to(self.audience(request.kind), Message::Point(point)))
                .into_iter()
                .collect(),
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
    use crate::testing::{ME, from, id, ids, narrated, observed, phase_began, target};
    use crate::werewolf::message::{Cause, Narration, Phase, RequestId, RequestKind, Round};
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

    fn request(id: u64, round: u32, kind: RequestKind) -> Event<WerewolfDomain> {
        from(
            MODERATOR,
            Message::Request(Request {
                id: RequestId(id),
                round: Round(round),
                kind,
            }),
        )
    }

    /// The action `Seat` takes in reply to a request: a point, addressed to
    /// `seen_by` and always to the moderator.
    fn pointing<const N: usize>(
        id: u64,
        target: AgentId,
        seen_by: [&str; N],
    ) -> Action<WerewolfDomain> {
        let mut to: Vec<AgentId> = seen_by.iter().map(|who| AgentId::new(*who)).collect();
        to.push(AgentId::new(MODERATOR));
        Action::to(
            to,
            Message::Point(Point {
                request: RequestId(id),
                target,
            }),
        )
    }

    /// A point of a kind only the moderator sees: the seer's and the
    /// doctor's own business.
    fn privately(id: u64, target: AgentId) -> Action<WerewolfDomain> {
        pointing(id, target, [])
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
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", ME])),
                request(1, 1, RequestKind::Protect),
            ],
        );
        assert!(silent.is_empty(), "{silent:?}");
        assert_eq!(seat.player.knowledge().last_protected, None);
    }

    #[test]
    fn each_request_is_answered_from_the_state_every_earlier_observation_left() {
        let mut seat = villager(Last);
        let cycle = [
            narrated(Narration::Assigned {
                role: Role::Villager,
                pack: BTreeSet::new(),
            }),
            phase_began(1, Phase::Day, ids(["alice", "bob", ME])),
            request(1, 1, RequestKind::Nominate),
            eliminated("bob", 1),
            request(2, 1, RequestKind::Nominate),
        ];
        assert_eq!(
            handling(&mut seat, cycle),
            [
                pointing(1, target("bob"), ["alice", "bob"]),
                pointing(2, target("alice"), ["alice"]),
            ]
        );
    }

    #[test]
    fn an_observation_that_is_not_a_request_produces_nothing_and_still_updates_the_state() {
        let mut seat = villager(Last);
        let silent = handling(
            &mut seat,
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", ME])),
                eliminated("bob", 1),
            ],
        );
        assert!(silent.is_empty(), "{silent:?}");

        // The elimination folded in the silent cycle shapes the next answer.
        assert_eq!(
            handling(&mut seat, [request(1, 1, RequestKind::Nominate)]),
            [pointing(1, target("alice"), ["alice"])]
        );
    }

    #[test]
    fn a_seat_opens_with_nothing() {
        // A player says nothing until the moderator asks it something, so
        // the default start hook is the right one for every role.
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
                request(1, 1, RequestKind::Protect),
                phase_began(1, Phase::Day, ids(["alice", "bob", "carol", ME])),
                request(2, 1, RequestKind::Nominate),
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
        let night = |round| {
            [
                phase_began(round, Phase::Night, ids(["alice", "bob", ME])),
                request(u64::from(round), round, RequestKind::Protect),
            ]
        };
        assert_eq!(
            handling(&mut seat, night(1)),
            [privately(1, target("alice"))]
        );
        assert_eq!(handling(&mut seat, night(2)), [privately(2, target("bob"))]);
        assert_eq!(
            handling(&mut seat, night(3)),
            [privately(3, target("alice"))]
        );
    }

    #[test]
    #[should_panic(
        expected = "me's policy chose AgentId(\"nobody\"), which is outside the action space"
    )]
    fn an_action_outside_the_action_space_panics() {
        let mut seat = villager(Outside);
        handling(
            &mut seat,
            [
                phase_began(1, Phase::Day, ids(["alice", ME])),
                request(1, 1, RequestKind::Nominate),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "a Villager is never asked to Devour")]
    fn a_request_of_a_kind_the_role_is_never_asked_panics() {
        let mut seat = villager(First);
        handling(
            &mut seat,
            [
                phase_began(1, Phase::Night, ids(["alice", ME])),
                request(1, 1, RequestKind::Devour),
            ],
        );
    }
}
