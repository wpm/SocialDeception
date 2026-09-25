//! A player as an agent: a role's rules, the policy that decides for it, and
//! the one handler body every role shares.
//!
//! A player's turn is a fold, the same as any agent's: every observation is
//! folded into its [`Knowledge`], and every [`Request`] among them is
//! answered with one [`Response`] to the moderator. A seat has no opening
//! move: it says nothing until it is asked, so it needs no
//! [`start`](Handler::start). The two halves of
//! answering are kept apart, and ADR-0005 says why: a [`Player`] is a role,
//! and computes the action space the rules permit it and nothing else; a
//! [`Policy`] is the strategy, and picks one move from that space. The
//! roles are in [`roles`](super::roles), the baseline policy in
//! [`policy`](super::policy).
//!
//! [`Seat`] joins the two. It is the [`Handler`] the episode runs, the same
//! for every role, and it is where an action outside the action space is
//! caught: a policy that returns one has a bug, and the game cannot
//! continue from it.
//!
//! # Players only ever address the moderator
//!
//! Every action a seat takes is addressed to the moderator alone. A player
//! is a peer of every other player, so a broadcast would reach them all,
//! and that must never happen: a response is a sealed ballot, and the others
//! learn of it only from the tally the moderator narrates.

use super::WerewolfDomain;
use super::knowledge::Knowledge;
use super::message::{Message, Move, Request, Response};
use super::policy::{Policy, View};
use crate::agent::{self, Handler, Observation};
use crate::cancel::Cancel;
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

    /// The action space for this request, in canonical order: targets in
    /// sorted agent order, [`Move::Abstain`] last where permitted, so
    /// that an index into it is a stable action label. Never empty for a
    /// request the rules legitimately issue.
    ///
    /// # Panics
    ///
    /// If the request is of a kind this role is never asked, which is a bug
    /// in the moderator rather than a runtime condition.
    fn action_space(&self, request: &Request) -> Vec<Move>;
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

    /// The response to one request: the policy's choice from the role's
    /// action space, checked against it and folded into the role's state.
    ///
    /// `cancel` is the cycle's, passed straight through: the seat has no
    /// opinion about preemption, and every request in a cycle is decided
    /// under the same one.
    fn answer(&mut self, request: &Request, cancel: &Cancel) -> Response {
        let action_space = self.player.action_space(request);
        let chosen = self.policy.choose(
            View {
                knowledge: self.player.knowledge(),
                request,
                action_space: &action_space,
            },
            cancel,
        );
        assert!(
            action_space.contains(&chosen),
            "{}'s policy chose {chosen:?}, which is outside the action space {action_space:?}",
            self.player.knowledge().me
        );
        self.player.knowledge_mut().acted(request, &chosen);
        Response {
            request: request.id,
            chosen,
        }
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
    /// If the policy chooses an action outside the action space, or if the
    /// request is of a kind this role is never asked; see
    /// [`Player::action_space`].
    fn handle(
        &mut self,
        observation: &Observation<WerewolfDomain>,
        cancel: &Cancel,
    ) -> Vec<agent::Action<WerewolfDomain>> {
        self.player.knowledge_mut().observe(observation);
        match &observation.event.payload {
            Message::Request(request) => {
                let response = Message::Response(self.answer(request, cancel));
                vec![agent::Action::to([self.moderator.clone()], response)]
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::agent::Recipients;
    use crate::event::Event;
    use crate::testing::{ME, from, id, ids, narrated, observed, phase_began, target};
    use crate::werewolf::message::{Cause, Narration, Phase, RequestId, RequestKind, Round};
    use crate::werewolf::role::Role;
    use crate::werewolf::roles::{Doctor, Villager};

    use crate::agent::Action;

    const MODERATOR: &str = "moderator";

    /// Picks the first action in the action space.
    struct First;

    impl Policy for First {
        fn choose(&mut self, view: View<'_>, _: &Cancel) -> Move {
            view.action_space[0].clone()
        }
    }

    /// Picks the last action in the action space.
    struct Last;

    impl Policy for Last {
        fn choose(&mut self, view: View<'_>, _: &Cancel) -> Move {
            view.action_space.last().unwrap().clone()
        }
    }

    /// Picks a player who is not in the game at all.
    struct Outside;

    impl Policy for Outside {
        fn choose(&mut self, _: View<'_>, _: &Cancel) -> Move {
            target("nobody")
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

    /// The action `Seat` takes in reply to a request: a response to the
    /// moderator.
    fn response(id: u64, chosen: Move) -> Action<WerewolfDomain> {
        Action::to(
            [MODERATOR],
            Message::Response(Response {
                request: RequestId(id),
                chosen,
            }),
        )
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
            .flat_map(|event| seat.handle(&observed(event), &Cancel::cancelled()))
            .collect()
    }

    fn villager<P: Policy>(policy: P) -> Seat<Villager, P> {
        Seat::new(Villager::new(id(ME)), policy, id(MODERATOR))
    }

    fn doctor<P: Policy>(policy: P) -> Seat<Doctor, P> {
        Seat::new(Doctor::new(id(ME)), policy, id(MODERATOR))
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
            [response(1, target("bob")), response(2, target("alice"))]
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
            [response(1, target("alice"))]
        );
    }

    #[test]
    fn a_seat_opens_with_nothing() {
        // A player says nothing until the moderator asks it something, so
        // the default start hook is the right one for every role.
        assert!(villager(First).start().is_empty());
        assert!(doctor(First).start().is_empty());
    }

    #[test]
    fn a_timeout_produces_nothing() {
        // What a timeout cycle looks like from inside a handler: the loop
        // calls `timeout`, not `handle`, and a player has nothing to say on
        // a deadline.
        let mut seat = villager(Last);
        assert!(seat.timeout(&Cancel::cancelled()).is_empty());
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
            assert_eq!(action.recipients, Recipients::To(ids([MODERATOR])));
            assert!(matches!(action.payload, Message::Response(_)), "{action:?}");
        }
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
            [response(1, target("alice"))]
        );
        assert_eq!(handling(&mut seat, night(2)), [response(2, target("bob"))]);
        assert_eq!(
            handling(&mut seat, night(3)),
            [response(3, target("alice"))]
        );
    }

    #[test]
    #[should_panic(
        expected = "me's policy chose Target(AgentId(\"nobody\")), which is outside the action space"
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
