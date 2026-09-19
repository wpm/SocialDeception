//! A player as an agent: the rules of its role, the policy that decides for
//! it, and the one handler body every role shares.
//!
//! A player's turn is a fold, the same as any agent's: every event that
//! arrives is folded into its [`Knowledge`], and every [`Request`] among them
//! is answered with one [`Response`] to the moderator. The two halves of that
//! are kept apart on purpose, and the reasons are in ADR-0005.
//!
//! - A **role** is the rules. [`Player`] is what a role contributes: its
//!   state, and the action space the rules permit it for a request. It
//!   computes that space and nothing else: no strategic filtering, no
//!   preference ordering, no advice. A werewolf eating its own packmate is in
//!   the action space; whether to do it is the policy's business.
//! - A **policy** is the strategy. [`Policy`] is handed the state and the
//!   action space and picks one action. Every strategic judgement lives
//!   there, so that the action space is the same for every policy, which
//!   is what lets trajectories from one policy be training data for
//!   another.
//!
//! [`Seat`] joins the two: a role, the policy deciding for it, and the
//! moderator it answers to. It is the [`Handler`] the episode runs, once for
//! every role, and it is where an action outside the action space is caught.
//!
//! # The action space, and its order
//!
//! Every action space starts from the same base, [`base_action_space`]: a
//! target for each living player other than the agent itself, in sorted
//! agent order, then [`Action::Abstain`] exactly where
//! [`RequestKind::may_abstain`] permits it. The two universal rules fall out
//! of that base: the target must be living, and no action may target the
//! agent taking it. Neither is strategy; nothing can do them.
//!
//! The order is canonical and load-bearing. An index into the vector is a
//! stable action label, the same on every run and in every episode with the
//! same living set, which is what a learned policy needs and what a
//! constrained decode over a model's output needs. That is why the action
//! space is a `Vec<Action>` and not a set.
//!
//! The action space is never empty when a request is legitimately issued:
//! `Nominate` and `Devour` are only asked while at least one valid target
//! lives, since the game would be over otherwise, and `Protect` and
//! `Investigate` always have `Abstain`.
//!
//! # Players only ever address the moderator
//!
//! Every message a seat sends is addressed to the moderator alone. A player
//! is a peer of every other player, so a broadcast would reach them all,
//! and that must never happen: a response is a sealed ballot, and the others
//! learn of it only from the tally the moderator narrates.

use super::knowledge::Knowledge;
use super::message::{Action, Message, Request, RequestKind, Response};
use super::policy::{Policy, View};
use crate::agent::{Handler, Outgoing};
use crate::event::{AgentId, Event};

/// What a role contributes to a player: its state, and the actions the
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
    /// sorted agent order, [`Action::Abstain`] last where permitted. Never
    /// empty for a request the rules legitimately issue.
    ///
    /// # Panics
    ///
    /// If the request is of a kind this role is never asked, which is a bug
    /// in the moderator rather than a runtime condition.
    fn action_space(&self, request: &Request) -> Vec<Action>;

    /// Called after an action is chosen, so that a role can remember it.
    /// Most roles have nothing to remember.
    fn chose(&mut self, _request: &Request, _action: &Action) {}
}

/// The action space every role starts from for a request of `kind`: a
/// target for each living player other than the agent itself, in sorted
/// agent order, then [`Action::Abstain`] if and only if the kind
/// [may be abstained from](RequestKind::may_abstain).
#[must_use]
pub fn base_action_space(knowledge: &Knowledge, kind: RequestKind) -> Vec<Action> {
    let mut space: Vec<Action> = knowledge
        .living_others()
        .into_iter()
        .map(Action::Target)
        .collect();
    if kind.may_abstain() {
        space.push(Action::Abstain);
    }
    space
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
    /// action space, checked against it and remembered by the role.
    fn answer(&mut self, request: &Request) -> Response {
        let action_space = self.player.action_space(request);
        let action = self.policy.choose(&View {
            knowledge: self.player.knowledge(),
            request,
            action_space: &action_space,
        });
        assert!(
            action_space.contains(&action),
            "{}'s policy chose {action:?}, which is outside the action space {action_space:?}",
            self.player.knowledge().me
        );
        self.player.chose(request, &action);
        Response {
            request: request.id,
            action,
        }
    }
}

impl<R: Player, P: Policy> Handler<Message> for Seat<R, P> {
    /// Folds each event into the role's state in order and answers each
    /// request among them, so that a request is answered from the state
    /// every event before it produced, including those in the same batch.
    /// A batch with two requests produces two responses, in request order;
    /// a batch with none produces nothing.
    ///
    /// # Panics
    ///
    /// If the policy chooses an action outside the action space, or if the
    /// request is of a kind this role is never asked; see
    /// [`Player::action_space`].
    fn handle(&mut self, events: &[Event<Message>]) -> Vec<Outgoing<Message>> {
        let mut outgoing = Vec::new();
        for event in events {
            self.player.knowledge_mut().observe(event);
            if let Event::Message {
                payload: Message::Request(request),
                ..
            } = event
            {
                let response = Message::Response(self.answer(request));
                outgoing.push(Outgoing::to([self.moderator.clone()], response));
            }
        }
        outgoing
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::agent::Recipients;
    use crate::event::Control;
    use crate::testing::{id, ids, target};
    use crate::werewolf::message::{Cause, Narration, Phase, RequestId, Round};
    use crate::werewolf::role::Role;
    use crate::werewolf::roles::{Doctor, Villager};

    const ME: &str = "me";
    const MODERATOR: &str = "moderator";

    /// Picks the first action in the action space.
    struct First;

    impl Policy for First {
        fn choose(&mut self, view: &View<'_>) -> Action {
            view.action_space[0].clone()
        }
    }

    /// Picks the last action in the action space.
    struct Last;

    impl Policy for Last {
        fn choose(&mut self, view: &View<'_>) -> Action {
            view.action_space.last().unwrap().clone()
        }
    }

    /// Picks a player who is not in the game at all.
    struct Outside;

    impl Policy for Outside {
        fn choose(&mut self, _: &View<'_>) -> Action {
            target("nobody")
        }
    }

    fn narrated(narration: Narration) -> Event<Message> {
        Event::message(MODERATOR, [ME], Message::Narration(narration))
    }

    fn phase_began(round: u32, phase: Phase, living: BTreeSet<AgentId>) -> Event<Message> {
        narrated(Narration::PhaseBegan {
            round: Round(round),
            phase,
            living,
        })
    }

    fn eliminated(who: &str, round: u32) -> Event<Message> {
        narrated(Narration::Eliminated {
            who: id(who),
            role: Role::Villager,
            round: Round(round),
            cause: Cause::Devoured,
        })
    }

    fn request(id: u64, round: u32, kind: RequestKind) -> Event<Message> {
        Event::message(
            MODERATOR,
            [ME],
            Message::Request(Request {
                id: RequestId(id),
                round: Round(round),
                kind,
            }),
        )
    }

    /// The response to the moderator that `Seat` sends.
    fn response(id: u64, action: Action) -> Outgoing<Message> {
        Outgoing::to(
            [MODERATOR],
            Message::Response(Response {
                request: RequestId(id),
                action,
            }),
        )
    }

    fn villager<P: Policy>(policy: P) -> Seat<Villager, P> {
        Seat::new(Villager::new(id(ME)), policy, id(MODERATOR))
    }

    fn doctor<P: Policy>(policy: P) -> Seat<Doctor, P> {
        Seat::new(Doctor::new(id(ME)), policy, id(MODERATOR))
    }

    #[test]
    fn a_batch_with_two_requests_produces_two_responses_in_request_order() {
        let mut seat = villager(Last);
        let batch = [
            Event::Control(Control::Start),
            narrated(Narration::Assigned {
                role: Role::Villager,
                pack: BTreeSet::new(),
            }),
            phase_began(1, Phase::Day, ids(["alice", "bob", ME])),
            request(1, 1, RequestKind::Nominate),
            request(2, 1, RequestKind::Nominate),
        ];
        assert_eq!(
            seat.handle(&batch),
            [response(1, target("bob")), response(2, target("bob"))]
        );
    }

    #[test]
    fn a_request_is_answered_from_the_state_the_events_before_it_produced() {
        let mut seat = villager(Last);
        let batch = [
            phase_began(1, Phase::Day, ids(["alice", "bob", ME])),
            request(1, 1, RequestKind::Nominate),
            eliminated("bob", 1),
            request(2, 1, RequestKind::Nominate),
        ];
        assert_eq!(
            seat.handle(&batch),
            [response(1, target("bob")), response(2, target("alice"))]
        );
    }

    #[test]
    fn a_batch_without_a_request_produces_nothing_and_still_updates_the_state() {
        let mut seat = villager(Last);
        let silent = seat.handle(&[
            Event::Control(Control::Start),
            phase_began(1, Phase::Night, ids(["alice", "bob", ME])),
            eliminated("bob", 1),
            Event::Think,
        ]);
        assert!(silent.is_empty(), "{silent:?}");

        // The elimination folded in the silent batch shapes the next answer.
        assert_eq!(
            seat.handle(&[request(1, 1, RequestKind::Nominate)]),
            [response(1, target("alice"))]
        );
    }

    #[test]
    fn every_message_is_a_response_addressed_to_the_moderator_alone() {
        let mut seat = doctor(First);
        let batch = [
            phase_began(1, Phase::Night, ids(["alice", "bob", "carol", ME])),
            request(1, 1, RequestKind::Protect),
            phase_began(1, Phase::Day, ids(["alice", "bob", "carol", ME])),
            request(2, 1, RequestKind::Nominate),
        ];
        let outgoing = seat.handle(&batch);
        assert_eq!(outgoing.len(), 2);
        for message in &outgoing {
            assert_eq!(message.recipients, Recipients::To(ids([MODERATOR])));
            assert!(
                matches!(message.payload, Message::Response(_)),
                "{message:?}"
            );
        }
    }

    #[test]
    fn a_role_remembers_what_was_chosen_for_it() {
        // The doctor may not protect the same player two nights running,
        // and it is the seat that tells it what it chose.
        let mut seat = doctor(First);
        let night = |round| {
            [
                phase_began(round, Phase::Night, ids(["alice", "bob", ME])),
                request(u64::from(round), round, RequestKind::Protect),
            ]
        };
        assert_eq!(seat.handle(&night(1)), [response(1, target("alice"))]);
        assert_eq!(seat.handle(&night(2)), [response(2, target("bob"))]);
        assert_eq!(seat.handle(&night(3)), [response(3, target("alice"))]);
    }

    #[test]
    #[should_panic(
        expected = "me's policy chose Target(AgentId(\"nobody\")), which is outside the action space"
    )]
    fn an_action_outside_the_action_space_panics() {
        let mut seat = villager(Outside);
        seat.handle(&[
            phase_began(1, Phase::Day, ids(["alice", ME])),
            request(1, 1, RequestKind::Nominate),
        ]);
    }

    #[test]
    #[should_panic(expected = "a Villager is never asked to Devour")]
    fn a_request_of_a_kind_the_role_is_never_asked_panics() {
        let mut seat = villager(First);
        seat.handle(&[
            phase_began(1, Phase::Night, ids(["alice", ME])),
            request(1, 1, RequestKind::Devour),
        ]);
    }
}
