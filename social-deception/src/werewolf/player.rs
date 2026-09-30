//! A player as an agent: what it knows, the strategy that decides for it,
//! and the one handler body every role shares.
//!
//! A player's turn is a fold, the same as any agent's: every observation
//! is folded into its [`Knowledge`], and a phase beginning is what makes it
//! act.
//!
//! **Nobody asks it to.** On observing that a phase has begun, a living
//! player asks its own role what that phase wants of it
//! ([`Role::asked_in`](super::Role::asked_in)) and selects if the answer is
//! something. That is ADR-0014: a message to an agent is a fact it
//! conditions on, and an instruction telling a player what its own role
//! already says is not one. In the live game nobody waits to be asked — the
//! moderator says night has fallen and the werewolves select, because they
//! are werewolves and it is night.
//!
//! **A player has no opening move.** It says nothing until the first phase
//! begins, so its [`start`](Policy::start) returns nothing; what it does
//! there is keep the episode's [`Clock`] in its [`Knowledge`], since that is
//! the one origin every actor and the log share (ADR-0017) and a
//! language-model player stamps its prompt with times measured from it.
//!
//! The two halves of deciding are kept apart, and ADR-0005 says why: a
//! [`Role`] computes the action space the rules permit and nothing else; a
//! [`Strategy`] picks one target from that space, or none. The rules are in
//! [`role`](super::role), the baseline strategy in
//! [`strategy`](super::strategy).
//!
//! [`Player`] joins the two with what it knows. It is the [`Policy`] the
//! episode runs, the same for every role, and it is where a target outside
//! the action space is caught: a strategy that returns one has a bug, and
//! the game cannot continue from it.
//!
//! # Who a selection is addressed to
//!
//! A selection is not a sealed ballot. Under ADR-0011 selecting is how a pack
//! agrees on a victim without speaking, and how a village's argument turns
//! into a vote, so a selection goes to everyone the rules let see it:
//!
//! | Kind | Recipients |
//! |---|---|
//! | `Devour` | the moderator and every other living member of the pack |
//! | `Investigate`, `Protect` | the moderator |
//! | `Nominate` | the moderator and every other living player |
//!
//! A player addresses each selection itself, from its own [`Knowledge`]: the
//! pack it was told at the deal, and the living it learns from each
//! `PhaseBegan`. It never addresses the whole roster, because the night's
//! secrets are exactly what must not travel that far.

use std::collections::BTreeSet;

use super::knowledge::Knowledge;
use super::message::{Message, Narration, Round, Select, SessionKind};
use super::role::Role;
use super::strategy::{Strategy, View};
use crate::clock::Clock;
use crate::message::ActorId;
use crate::{Action, Observation, Policy};

/// A player in the episode: what it knows, the strategy that decides for it,
/// and the moderator it addresses its selections to.
///
/// One type for every role, because the handler body is the same for all of
/// them and the role is data its [`Knowledge`] already holds: fold each
/// message into what it knows and, when a phase begins a session its role is
/// a member of, select where the strategy chooses from the role's action
/// space.
#[derive(Debug, Clone)]
pub struct Player<S> {
    knowledge: Knowledge,
    strategy: S,
    moderator: ActorId,
}

impl<S: Strategy> Player<S> {
    /// A player named `me` playing `role`, deciding with `strategy`,
    /// selecting to `moderator`.
    ///
    /// It has observed nothing yet. A werewolf's pack arrives in the
    /// moderator's `Assigned` narration, which is also checked against the
    /// role given here: a miswired roster fails at the start of the episode
    /// rather than producing a plausible game.
    #[must_use]
    pub fn new(me: ActorId, role: Role, strategy: S, moderator: ActorId) -> Self {
        Self {
            knowledge: Knowledge::new(me, role),
            strategy,
            moderator,
        }
    }

    /// What this player has learned: the fold of every observation it has
    /// received, including the [`Role`] it was dealt.
    #[must_use]
    pub const fn knowledge(&self) -> &Knowledge {
        &self.knowledge
    }

    /// The selection this player makes in the session of `kind` now open, if
    /// the strategy names a target: its choice from the role's action
    /// space, checked against it and folded into what it knows.
    ///
    /// `None` is a strategy declining to select for now, which is how a
    /// member abstains, and it is also what an empty action space leaves
    /// it: a doctor with nobody it may protect selects nowhere. Nothing is
    /// sent and nothing is folded either way — a selection never made is not
    /// a vote.
    fn select(&mut self, round: Round, kind: SessionKind) -> Option<Select> {
        let action_space = self.knowledge.role.action_space(&self.knowledge, kind);
        let chosen = self.strategy.choose(View {
            knowledge: &self.knowledge,
            kind,
            action_space: &action_space,
        })?;
        assert!(
            action_space.contains(&chosen),
            "{}'s strategy chose {chosen:?}, which is outside the action space {action_space:?}",
            self.knowledge.me
        );
        self.knowledge.acted(kind, &chosen);
        Some(Select {
            round,
            kind,
            target: chosen,
            seen_by: self.audience(kind),
        })
    }

    /// Who else should see a selection of this kind: the pack for a `Devour`,
    /// every other living player for a `Nominate`, and nobody for the
    /// seer's and the doctor's own business.
    ///
    /// These are not recipients. A selection is addressed to the moderator
    /// alone; this is who the moderator forwards it to, and only if the
    /// session is still open when it arrives. Naming them is the player's
    /// job because the audience follows from the player's own role and its
    /// own knowledge of who is alive.
    ///
    /// Always without itself: a player does not observe its own actions.
    /// The moderator is not named either, since it receives every selection
    /// directly and has no need to be forwarded one.
    fn audience(&self, kind: SessionKind) -> BTreeSet<ActorId> {
        let knowledge = &self.knowledge;
        let me = &knowledge.me;
        let seen_by = match kind {
            SessionKind::Devour => &knowledge.pack,
            SessionKind::Nominate => &knowledge.living,
            SessionKind::Investigate | SessionKind::Protect => return BTreeSet::new(),
        };
        seen_by
            .iter()
            .filter(|who| *who != me && knowledge.living.contains(*who))
            .cloned()
            .collect()
    }

    /// `payload` addressed to the moderator and to nobody else.
    ///
    /// The one place a player names a recipient, so the rule ADR-0018 states
    /// is written once and asserted once. It is a `debug_assert!` rather than
    /// an `assert!` because a player that addressed somebody else would be a
    /// bug in this file and not a rule the game has to enforce against an
    /// adversary: players cooperate with the moderator.
    fn addressed(&self, payload: Message) -> Action<Message> {
        let action = Action::to([self.moderator.clone()], payload);
        debug_assert!(
            matches!(&action, Action::Send { to, .. } if to == std::slice::from_ref(&self.moderator)),
            "{} addresses every action to the moderator alone",
            self.knowledge.me
        );
        action
    }
}

impl<S: Strategy> Policy<Message> for Player<S> {
    /// Keeps the episode's origin and says nothing.
    ///
    /// A player has no opening move: nothing has happened yet, and it is a
    /// phase beginning that makes a player act (ADR-0014). What the hook is
    /// for here is the [`Clock`], which is the episode's one origin, shared
    /// by every actor and the log writer (ADR-0017): a player that measures
    /// time is then measuring it on the log's timeline.
    fn start(&mut self, clock: Clock) -> impl IntoIterator<Item = Action<Message>> {
        self.knowledge.started(clock);
        []
    }

    /// Folds the observation into what the player knows and, if it was a
    /// phase beginning a session its role is a member of, selects from the
    /// state every earlier observation produced, this one included. Every
    /// other observation produces nothing but the fold.
    ///
    /// The fold comes first for exactly that reason: the phase's own
    /// narration says who is still living, and the selection has to be
    /// decided from the living set that includes it.
    ///
    /// **Every action goes to the moderator and to nobody else** (ADR-0018),
    /// which a `debug_assert!` checks. Players cooperate with the moderator
    /// and are not assumed to cheat, so that catches a mistake here rather
    /// than enforcing a rule on a player that might break it.
    ///
    /// # Panics
    ///
    /// If the strategy chooses a target outside the action space. It cannot
    /// panic for the other reason [`Role::action_space`] gives, since the
    /// kind it passes came from this player's own
    /// [`Role::asked_in`].
    fn policy(
        &mut self,
        observation: Observation<Message>,
    ) -> impl IntoIterator<Item = Action<Message>> {
        self.knowledge.observe(&observation);
        // A phase beginning is what makes a player act, and it acts on its
        // own role rather than on anybody's instruction (ADR-0014). A player
        // the phase asks nothing of, and one the rules leave nowhere to
        // select, both say nothing.
        let Message::Narration(Narration::PhaseBegan { round, phase, .. }) =
            &observation.message.payload
        else {
            return Vec::new();
        };
        let (round, phase) = (*round, *phase);
        let Some(kind) = self.knowledge.role.asked_in(phase) else {
            return Vec::new();
        };
        self.select(round, kind)
            .map(|selection| self.addressed(Message::Select(selection)))
            .into_iter()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;
    use crate::testing::{ME, id, ids, narrated, observed, phase_began, target};
    use crate::werewolf::message::{Cause, Narration, Phase, Round, SessionKind};

    use crate::Action;

    const MODERATOR: &str = "moderator";

    /// The payload of a send, for a test that reads one.
    ///
    /// # Panics
    ///
    /// On a reminder, which no player ever sets.
    fn payload_of(action: &Action<Message>) -> &Message {
        match action {
            Action::Send { payload, .. } => payload,
            Action::Remind(_) => panic!("a player sets no reminders: {action:?}"),
        }
    }

    /// Selects the first target in the action space.
    struct First;

    impl Strategy for First {
        fn choose(&mut self, view: View<'_>) -> Option<ActorId> {
            view.action_space.first().cloned()
        }
    }

    /// Selects the last target in the action space.
    struct Last;

    impl Strategy for Last {
        fn choose(&mut self, view: View<'_>) -> Option<ActorId> {
            view.action_space.last().cloned()
        }
    }

    /// Selects a player who is not in the game at all.
    struct Outside;

    impl Strategy for Outside {
        fn choose(&mut self, _: View<'_>) -> Option<ActorId> {
            Some(target("nobody"))
        }
    }

    /// Selects nowhere, which is how a member abstains.
    struct Nowhere;

    impl Strategy for Nowhere {
        fn choose(&mut self, _: View<'_>) -> Option<ActorId> {
            None
        }
    }

    fn eliminated(who: &str, round: u32) -> crate::Message<Message> {
        narrated(Narration::Eliminated {
            who: id(who),
            role: Role::Villager,
            round: Round::new(round),
            cause: Cause::Devoured,
        })
    }

    /// The action a [`Player`] takes when a phase begins: a selection naming
    /// its own session, addressed to the moderator alone and naming `seen_by`
    /// as the players the moderator should forward it to.
    ///
    /// There is no request to echo (ADR-0014), so a selection is identified
    /// by the round and the kind it was made in, which is what the reader of
    /// the log reads off it directly. And there is no player among the
    /// recipients: a player addresses the moderator and nobody else, which is
    /// what keeps a selection from outliving its session in a peer's queue.
    fn selecting<const N: usize>(
        round: u32,
        kind: SessionKind,
        target: ActorId,
        seen_by: [&str; N],
    ) -> Action<Message> {
        Action::to(
            [ActorId::new(MODERATOR)],
            Message::Select(Select {
                round: Round::new(round),
                kind,
                target,
                seen_by: seen_by.iter().map(|who| ActorId::new(*who)).collect(),
            }),
        )
    }

    /// A selection of a kind only the moderator sees: the seer's and the
    /// doctor's own business.
    fn privately(round: u32, kind: SessionKind, target: ActorId) -> Action<Message> {
        selecting(round, kind, target, [])
    }

    /// What a player does with a run of messages, each in a call of its own,
    /// as the handler thread hands them over (ADR-0008): every action they
    /// produced, in order.
    ///
    /// There are no threads here. A policy is a function from an observation
    /// to actions, so a test of one is a run of calls and nothing else
    /// (ADR-0016).
    fn handling<S: Strategy, const N: usize>(
        player: &mut Player<S>,
        messages: [crate::Message<Message>; N],
    ) -> Vec<Action<Message>> {
        messages
            .into_iter()
            .flat_map(|message| {
                player
                    .policy(observed(message))
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn villager<S: Strategy>(strategy: S) -> Player<S> {
        Player::new(id(ME), Role::Villager, strategy, id(MODERATOR))
    }

    fn doctor<S: Strategy>(strategy: S) -> Player<S> {
        Player::new(id(ME), Role::Doctor, strategy, id(MODERATOR))
    }

    #[test]
    fn a_strategy_that_selects_nowhere_sends_nothing() {
        // Selecting nowhere is how a member abstains (ADR-0011). Nothing is
        // sent, and nothing is folded: a selection never made is not a vote,
        // so the doctor has no protection to be kept from repeating.
        let mut player = doctor(Nowhere);
        let silent = handling(
            &mut player,
            [phase_began(1, Phase::Night, ids(["alice", "bob", ME]))],
        );
        assert!(silent.is_empty(), "{silent:?}");
        assert_eq!(player.knowledge().last_protected, None);
    }

    #[test]
    fn each_phase_is_acted_on_from_the_state_every_earlier_observation_left() {
        // What makes a player select is the phase beginning, not anybody
        // asking it to (ADR-0014). The selection it makes is a function of
        // everything folded in before that phase began, so the second day
        // is selected in from a village bob has left.
        let mut player = villager(Last);
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
            handling(&mut player, cycle),
            [
                selecting(1, SessionKind::Nominate, target("bob"), ["alice", "bob"]),
                selecting(2, SessionKind::Nominate, target("alice"), ["alice"]),
            ]
        );
    }

    #[test]
    fn an_observation_that_is_not_a_phase_beginning_produces_nothing_and_still_updates_the_state() {
        let mut player = villager(Last);
        let silent = handling(
            &mut player,
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", ME])),
                eliminated("bob", 1),
            ],
        );
        // A villager has nothing to do at night, and an elimination asks
        // nothing of anybody; both are folded in silently.
        assert!(silent.is_empty(), "{silent:?}");

        // The elimination folded in the silent cycle shapes the next selection.
        assert_eq!(
            handling(
                &mut player,
                [phase_began(2, Phase::Day, ids(["alice", ME]))]
            ),
            [selecting(
                2,
                SessionKind::Nominate,
                target("alice"),
                ["alice"]
            )]
        );
    }

    #[test]
    fn a_player_opens_with_nothing_and_keeps_the_clock() {
        // A player says nothing until the first phase begins, so its `start`
        // yields nothing whatever its role. What it does there is keep the
        // episode's origin, which is the one clock every actor and the log
        // share (ADR-0017).
        let clock = Clock::start();
        for mut player in [villager(First), doctor(First)] {
            let opened: Vec<_> = player.start(clock).into_iter().collect();
            assert!(opened.is_empty(), "{opened:?}");
            assert_eq!(
                player.knowledge().clock,
                Some(clock),
                "the origin the episode gave it"
            );
        }
    }

    #[test]
    fn a_player_knows_no_clock_until_it_is_started() {
        assert_eq!(villager(First).knowledge().clock, None);
    }

    #[test]
    fn every_action_is_a_response_addressed_to_the_moderator_alone() {
        let mut player = doctor(First);
        let actions = handling(
            &mut player,
            [
                phase_began(1, Phase::Night, ids(["alice", "bob", "carol", ME])),
                phase_began(1, Phase::Day, ids(["alice", "bob", "carol", ME])),
            ],
        );
        assert_eq!(actions.len(), 2);
        // Every action, whatever it is, goes to the moderator and to
        // nobody else. A player never addresses another player, which is
        // what stops a selection outliving its session in a peer's queue
        // (ADR-0018).
        for action in &actions {
            assert!(
                matches!(payload_of(action), Message::Select(_)),
                "{action:?}"
            );
            let Action::Send { to, .. } = action else {
                panic!("a player sends and never reminds itself: {action:?}");
            };
            assert_eq!(*to, vec![id(MODERATOR)], "{action:?}");
        }
        // What differs is the audience the moderator is asked to relay to:
        // the protect is nobody else's business, the nomination is public.
        let seen_by = |action: &Action<Message>| match payload_of(action) {
            Message::Select(selection) => selection.seen_by.clone(),
            other => panic!("a selection, not {other:?}"),
        };
        assert_eq!(seen_by(&actions[0]), BTreeSet::new());
        assert_eq!(seen_by(&actions[1]), ids(["alice", "bob", "carol"]));
    }

    #[test]
    fn what_was_chosen_is_folded_into_the_state() {
        // The doctor may not protect the same player two nights running,
        // and it is the player that folds each protection into its knowledge.
        let mut player = doctor(First);
        let night = |round| [phase_began(round, Phase::Night, ids(["alice", "bob", ME]))];
        assert_eq!(
            handling(&mut player, night(1)),
            [privately(1, SessionKind::Protect, target("alice"))]
        );
        assert_eq!(
            handling(&mut player, night(2)),
            [privately(2, SessionKind::Protect, target("bob"))]
        );
        assert_eq!(
            handling(&mut player, night(3)),
            [privately(3, SessionKind::Protect, target("alice"))]
        );
    }

    #[test]
    #[should_panic(
        expected = "me's strategy chose ActorId(\"nobody\"), which is outside the action space"
    )]
    fn an_action_outside_the_action_space_panics() {
        let mut player = villager(Outside);
        handling(
            &mut player,
            [phase_began(1, Phase::Day, ids(["alice", ME]))],
        );
    }

    #[test]
    fn a_phase_the_role_is_asked_nothing_in_passes_in_silence() {
        // Under ADR-0014 a role that is asked nothing in a phase cannot be
        // handed a request of the wrong kind, because nothing is handed to
        // it at all: it reads its own role and says nothing. A villager at
        // night never reaches its action space, so the panic that guarded
        // against a miscast request has no path through a player left. The
        // assertion itself is still exercised, by calling `action_space`
        // directly; `role` has those tests.
        let mut player = villager(First);
        let silent = handling(
            &mut player,
            [phase_began(1, Phase::Night, ids(["alice", ME]))],
        );
        assert!(silent.is_empty(), "{silent:?}");
    }
}
