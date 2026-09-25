//! The moderator: the game state machine as the episode's environment.
//!
//! [`Moderator`] is the [`Environment`] around a [`Game`]. It is plumbing,
//! and thin by design: every rule of Werewolf lives in [`Game`], and this
//! module only folds the observations that arrive on the moderator's queue
//! into calls on the game and turns the [`Directive`]s that come back into
//! [`Effect`]s. What it adds is the invariants the runtime imposes on how
//! those effects are addressed, and the shape of the episode's shutdown.
//!
//! # What the moderator does
//!
//! Being started begins the game, which is what [`start`](Environment::start)
//! is for: the moderator is the agent that opens play, and it does so before
//! anybody has spoken to it. It is also where the players are started, since
//! an episode's environment is the only thing that may send a [`Control`]
//! (ADR-0007). Thereafter a [`Response`](super::Response) from a player is
//! recorded. The moderator never sees the control that started it, nor the
//! one that stops it; those are the loop's, which is why there is no arm for
//! either here.
//!
//! A [`Narration`](super::Narration) or a [`Request`](super::Request)
//! arriving from a player is a bug, and the moderator panics rather than run
//! a game whose state it cannot vouch for.
//!
//! Once the game is over the moderator emits nothing, whatever arrives. The
//! episode is winding down, and a late response is not the game's problem.
//!
//! # No message of this game is broadcast
//!
//! Every message carries an explicit recipient set, and the choice of
//! recipients is the whole hidden-information mechanism (ADR-0004).
//! ADR-0004 made one exception, broadcasting the final [`Outcome`] to every
//! player, living and dead, because it was a dead player's terminal reward
//! signal. A reward is now logged rather than said (ADR-0007), so the
//! exception is withdrawn: the outcome is narrated to the living like any
//! other narration, and a dead player's trajectory ends at the announcement
//! of its own elimination, then its `Stop`.
//!
//! # The outcome reaches the caller on a channel
//!
//! An episode consumes its roster and drops the handlers it joins, so the
//! moderator's final state is otherwise unrecoverable. The moderator is built
//! with a [`Sender`] and publishes the outcome there as well as announcing it
//! in world. It is an observation channel, not a control channel: the
//! in-world announcement remains the record of truth, and a caller that has
//! dropped the receiver is simply not listening.
//!
//! # How the episode ends, in order
//!
//! The cycle in which the game ends does four things, and the order of
//! them is the whole of the shutdown:
//!
//! 1. **narrate** the [`Outcome`] to the living, as an ordinary
//!    [`Effect::Act`];
//! 2. **publish** it on the [`Sender`], for whoever ran the episode;
//! 3. **pay** every player, living and dead, one
//!    [`Effect::Reward`] each: **+1** if its role's faction won and
//!    **−1** otherwise, as [`Game::rewards`] works out;
//! 4. **stop** every player, living and dead, with one
//!    [`Effect::Control`].
//!
//! The rewards come before the stop because they are what the episode was
//! for. They are logged rather than sent (ADR-0007), so their position
//! among the effects changes nothing a player sees; what it does is put
//! each reward in the trajectory ahead of the `Stop` control that closes
//! the trajectory it belongs to, which is the ordering a reader can then
//! rely on.
//!
//! The episode routes a cycle's events before the controls it asked for, so
//! a living player observes the outcome and *then* stops, rather than
//! stopping with the outcome still on its queue and never seeing it. A dead
//! player is sent no outcome and only the stop. The episode stops the
//! moderator itself once every player's thread has ended.
//!
//! A game that never reaches an outcome, because some player did not answer
//! a request, is nobody's to notice here: nothing is in flight, no player
//! has been stopped, and the episode calls that
//! [`EpisodeError::Stalled`](crate::EpisodeError::Stalled).

use crossbeam_channel::Sender;

use std::collections::BTreeSet;

use super::WerewolfDomain;
use super::game::{Directive, Game};
use super::message::{Message, Outcome};
use crate::agent::{Action, Observation, Recipients};
use crate::environment::{Effect, Environment};
use crate::event::{AgentId, Control};

/// The agent that runs a game of Werewolf: a [`Game`] behind an
/// [`Environment`].
#[derive(Debug)]
pub struct Moderator {
    game: Game,
    outcome: Sender<Outcome>,
}

impl Moderator {
    /// A moderator that runs `game` and, when it ends, sends its outcome
    /// on `outcome`.
    ///
    /// The game must not have begun: the moderator begins it when the
    /// episode starts it.
    #[must_use]
    pub const fn new(game: Game, outcome: Sender<Outcome>) -> Self {
        Self { game, outcome }
    }

    /// Every player in the game, living and dead: whom the moderator starts
    /// and, when the game is over, stops.
    fn players(&self) -> BTreeSet<AgentId> {
        self.game.players().cloned().collect()
    }

    /// What one observation makes the game say.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration or a request.
    fn fold(&mut self, observation: &Observation<WerewolfDomain>) -> Vec<Directive> {
        let sender = &observation.event.sender;
        match &observation.event.payload {
            Message::Response(response) => self.game.record(sender, response),
            Message::Narration(_) => panic!("{sender} sent the moderator a narration"),
            Message::Request(_) => panic!("{sender} sent the moderator a request"),
        }
    }

    /// Everything the directives say, said; then, if the game has just
    /// ended, the outcome published on the channel, every player paid, and
    /// every player stopped. The order is the shutdown sequence; see the
    /// [module documentation](self).
    fn say(&mut self, directives: Vec<Directive>) -> Vec<Effect<WerewolfDomain>> {
        let mut effects: Vec<Effect<WerewolfDomain>> =
            directives.into_iter().map(send).map(Effect::Act).collect();
        if let Some(outcome) = self.game.outcome() {
            // The caller may have dropped the receiver. That is not the
            // game's problem: the in-world announcement is the record.
            let _ = self.outcome.send(outcome.clone());
            // Once, and this is the once: `handle` stops folding at the
            // observation that ends the game, so the outcome is seen here
            // on the cycle it first exists and on no later one. So each
            // player is paid exactly once, and then stopped.
            let rewards = self
                .game
                .rewards()
                .expect("a game with an outcome has rewards");
            effects.extend(
                rewards
                    .into_iter()
                    .map(|(who, value)| Effect::Reward { agent: who, value }),
            );
            effects.push(Effect::control(self.players(), Control::Stop));
        }
        effects
    }
}

/// The action that carries one directive.
fn send(directive: Directive) -> Action<WerewolfDomain> {
    match directive {
        Directive::Narrate { to, narration } => Action {
            recipients: Recipients::To(to),
            payload: Message::Narration(narration),
        },
        Directive::Ask { to, request } => Action::to([to], Message::Request(request)),
    }
}

impl Environment<WerewolfDomain> for Moderator {
    /// Starts every player, then begins the game and says what it wants said
    /// at the start: the roles, the first phase, and the first night's
    /// requests.
    ///
    /// The `Start` comes first among the effects, but the episode routes a
    /// cycle's events before its controls either way, so what a player
    /// actually sees is its `Start` — controls are popped first — and then
    /// the opening narrations.
    fn start(&mut self) -> Vec<Effect<WerewolfDomain>> {
        let opening = self.game.begin();
        let mut effects = vec![Effect::control(self.players(), Control::Start)];
        effects.extend(self.say(opening));
        effects
    }

    /// Folds the observation into the game and says what the game wants
    /// said. The observation that ends the game also sends the outcome on
    /// the channel, pays every player and stops every player; after it,
    /// nothing, whatever arrives.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration or a request, or if a
    /// response is one the game cannot accept; see [`Game::record`].
    fn handle(&mut self, observation: &Observation<WerewolfDomain>) -> Vec<Effect<WerewolfDomain>> {
        // Whether the game is over is the game's to say, and it is asked
        // before every observation, so the one that ends it is the last
        // folded and the only one after which the outcome is seen for the
        // first time. A response that arrives after that — one a player sent
        // before its stop reached it — is not folded, and the moderator says
        // nothing about it.
        if self.game.outcome().is_some() {
            return Vec::new();
        }
        let directives = self.fold(observation);
        self.say(directives)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crossbeam_channel::{Receiver, TryRecvError, unbounded};

    use super::*;
    use crate::agent::Recipients;
    use crate::event::{AgentId, Event};
    use crate::testing::{id, ids, observed};
    use crate::werewolf::assignment::Assignment;
    use crate::werewolf::message::{
        Move, Narration, Phase, Request, RequestId, RequestKind, Response, Round,
    };
    use crate::werewolf::role::Faction;
    use crate::werewolf::role::Role::{self, Doctor, Seer, Villager, Werewolf};

    use crate::agent::Action;

    const MODERATOR: &str = "moderator";
    const SEED: u64 = 20_260_918;

    /// Five players and one werewolf: alice and erin are villagers, bob is
    /// the werewolf, carol the seer and dave the doctor.
    fn village() -> Assignment {
        Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Seer),
            ("dave", Doctor),
            ("erin", Villager),
        ])
    }

    /// Seven players and two werewolves, bob and frank; carol is the seer
    /// and dave the doctor.
    fn town() -> Assignment {
        Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Seer),
            ("dave", Doctor),
            ("erin", Villager),
            ("frank", Werewolf),
            ("grace", Villager),
        ])
    }

    fn moderator(assignment: Assignment) -> (Moderator, Receiver<Outcome>) {
        let (sender, receiver) = unbounded();
        let game = Game::new(assignment, SEED);
        (Moderator::new(game, sender), receiver)
    }

    /// An event from a player to the moderator, as the moderator observes
    /// it. The creation time plays no part in the fold, so one stand-in
    /// serves every test here.
    fn from_player(who: &str, payload: Message) -> Observation<WerewolfDomain> {
        observed(Event::new(
            who,
            [MODERATOR],
            crate::clock::Timestamp::default(),
            payload,
        ))
    }

    fn response(who: &AgentId, request: RequestId, chosen: Move) -> Observation<WerewolfDomain> {
        from_player(
            who.as_str(),
            Message::Response(Response { request, chosen }),
        )
    }

    /// A stub player: what it does with a request, given who it is and who
    /// is living.
    type Policy = fn(&AgentId, &Request, &BTreeSet<AgentId>) -> Move;

    /// Targets the first living player other than itself.
    fn first_other(me: &AgentId, _: &Request, living: &BTreeSet<AgentId>) -> Move {
        Move::Target(living.iter().find(|who| *who != me).unwrap().clone())
    }

    /// Targets the last living player other than itself.
    fn last_other(me: &AgentId, _: &Request, living: &BTreeSet<AgentId>) -> Move {
        Move::Target(living.iter().rev().find(|who| *who != me).unwrap().clone())
    }

    /// Abstains wherever the request permits it, and otherwise targets the
    /// last living player other than itself.
    fn abstainer(me: &AgentId, request: &Request, living: &BTreeSet<AgentId>) -> Move {
        if request.kind.may_abstain() {
            Move::Abstain
        } else {
            last_other(me, request, living)
        }
    }

    /// The agents an action is addressed to, which for a request is exactly
    /// one.
    fn asked(action: &Action<WerewolfDomain>) -> &AgentId {
        let Recipients::To(to) = &action.recipients else {
            panic!("a request was broadcast: {action:?}");
        };
        assert_eq!(to.len(), 1, "a request to several players: {action:?}");
        to.iter().next().unwrap()
    }

    /// The actions among some effects, in order.
    fn actions(effects: &[Effect<WerewolfDomain>]) -> Vec<Action<WerewolfDomain>> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Act(action) => Some(action.clone()),
                Effect::Control { .. } | Effect::Reward { .. } => None,
            })
            .collect()
    }

    /// The controls among some effects, in order.
    fn controls(effects: &[Effect<WerewolfDomain>]) -> Vec<(BTreeSet<AgentId>, Control)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Control { to, control } => Some((to.clone(), *control)),
                Effect::Act(_) | Effect::Reward { .. } => None,
            })
            .collect()
    }

    /// The rewards among some effects, by the agent paid, in the order the
    /// moderator assigned them.
    fn rewards(effects: &[Effect<WerewolfDomain>]) -> Vec<(AgentId, i32)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Reward { agent, value } => Some((agent.clone(), *value)),
                Effect::Act(_) | Effect::Control { .. } => None,
            })
            .collect()
    }

    /// The responses stub players following `policy` give to the requests
    /// among some actions, choosing among `living`.
    fn respond(
        actions: &[Action<WerewolfDomain>],
        policy: Policy,
        living: &BTreeSet<AgentId>,
    ) -> Vec<Observation<WerewolfDomain>> {
        actions
            .iter()
            .filter_map(|action| match &action.payload {
                Message::Request(request) => {
                    let who = asked(action);
                    Some(response(who, request.id, policy(who, request, living)))
                }
                Message::Narration(_) | Message::Response(_) => None,
            })
            .collect()
    }

    /// Plays a whole game, with stub players following `policy`, and returns
    /// every effect the moderator produced in order.
    ///
    /// The game opens with the start hook, as the agent loop opens it, and
    /// each response then arrives in a cycle of its own, as the loop hands
    /// them over one at a time (ADR-0008). The game runs until the
    /// moderator asks nothing more.
    fn play(moderator: &mut Moderator, policy: Policy) -> Vec<Effect<WerewolfDomain>> {
        let opening = moderator.start();
        let mut pending = respond(&actions(&opening), policy, moderator.game.living());
        let mut produced = opening;
        while !pending.is_empty() {
            let effects: Vec<Effect<WerewolfDomain>> = pending
                .iter()
                .flat_map(|observation| moderator.handle(observation))
                .collect();
            pending = respond(&actions(&effects), policy, moderator.game.living());
            produced.extend(effects);
        }
        produced
    }

    /// A game played out: its assignment, how the game says it ended, the
    /// receiver of its outcome, every message the moderator sent, every
    /// control it asked for, every reward it paid, and the whole sequence
    /// of effects it produced.
    struct Played {
        assignment: Assignment,
        outcome: Outcome,
        receiver: Receiver<Outcome>,
        sent: Vec<Action<WerewolfDomain>>,
        commanded: Vec<(BTreeSet<AgentId>, Control)>,
        paid: Vec<(AgentId, i32)>,
        /// Every effect in the order the moderator produced it, which is
        /// what the shutdown's ordering is asserted against.
        effects: Vec<Effect<WerewolfDomain>>,
    }

    /// Every combination of assignment and stub policy, played out.
    fn played_games() -> Vec<Played> {
        let policies: [Policy; 3] = [first_other, last_other, abstainer];
        let mut games = Vec::new();
        for assignment in [village(), town()] {
            for policy in policies {
                let (mut moderator, receiver) = moderator(assignment.clone());
                let effects = play(&mut moderator, policy);
                games.push(Played {
                    assignment: assignment.clone(),
                    outcome: moderator.game.outcome().unwrap().clone(),
                    receiver,
                    sent: actions(&effects),
                    commanded: controls(&effects),
                    paid: rewards(&effects),
                    effects,
                });
            }
        }
        games
    }

    /// The outcome announced among some actions, and whom it was announced
    /// to.
    fn announced_outcome(sent: &[Action<WerewolfDomain>]) -> (&BTreeSet<AgentId>, &Outcome) {
        sent.iter()
            .find_map(|action| match action {
                Action {
                    recipients: Recipients::To(to),
                    payload: Message::Narration(Narration::Outcome(outcome)),
                } => Some((to, outcome)),
                _ => None,
            })
            .expect("no outcome was announced")
    }

    fn narrate<const N: usize>(to: [&str; N], narration: Narration) -> Action<WerewolfDomain> {
        Action::to(to, Message::Narration(narration))
    }

    fn assigned<const N: usize>(to: &str, role: Role, pack: [&str; N]) -> Action<WerewolfDomain> {
        narrate(
            [to],
            Narration::Assigned {
                role,
                pack: ids(pack),
            },
        )
    }

    fn ask(to: &str, id: u64, round: u32, kind: RequestKind) -> Action<WerewolfDomain> {
        Action::to(
            [to],
            Message::Request(Request {
                id: RequestId(id),
                round: Round(round),
                kind,
            }),
        )
    }

    #[test]
    fn starting_the_moderator_starts_the_players_and_begins_the_game() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let (mut moderator, _receiver) = moderator(village());
        let opening = moderator.start();
        assert_eq!(
            controls(&opening),
            [(ids(everyone), Control::Start)],
            "every player is started, and nobody is stopped yet"
        );
        assert_eq!(
            actions(&opening),
            [
                assigned("alice", Villager, []),
                assigned("bob", Werewolf, ["bob"]),
                assigned("carol", Seer, []),
                assigned("dave", Doctor, []),
                assigned("erin", Villager, []),
                narrate(
                    everyone,
                    Narration::PhaseBegan {
                        round: Round(1),
                        phase: Phase::Night,
                        living: ids(everyone),
                    }
                ),
                ask("bob", 1, 1, RequestKind::Devour),
                ask("carol", 2, 1, RequestKind::Investigate),
                ask("dave", 3, 1, RequestKind::Protect),
            ]
        );
    }

    #[test]
    fn a_timeout_produces_nothing() {
        // What a timeout cycle looks like from inside the handler: the loop
        // calls `timeout`, not `handle`, and the moderator has nothing to
        // say on a deadline. Controls never reach it either, so there is
        // nothing a cycle can hold that it must ignore.
        let (mut moderator, _receiver) = moderator(village());
        moderator.start();
        assert_eq!(moderator.timeout(), []);
    }

    #[test]
    fn the_game_ending_pays_every_player_for_its_faction() {
        // Living or dead, +1 for the winning side and −1 for the losing
        // one, once each. There are no stalemates, so nobody is paid zero
        // and nobody goes unpaid.
        for Played {
            assignment,
            outcome,
            paid,
            ..
        } in played_games()
        {
            let expected: Vec<(AgentId, i32)> = assignment
                .players()
                .map(|(who, role)| {
                    let value = if role.faction() == outcome.winner {
                        1
                    } else {
                        -1
                    };
                    (who.clone(), value)
                })
                .collect();
            assert_eq!(paid, expected, "{outcome:?}");
            let dead: Vec<&AgentId> = paid
                .iter()
                .map(|(who, _)| who)
                .filter(|who| !outcome.living.contains(who))
                .collect();
            assert!(!dead.is_empty(), "the dead are paid too: {outcome:?}");
            assert!(
                !paid.iter().any(|(who, _)| *who == id(MODERATOR)),
                "the moderator plays no game and is paid nothing: {paid:?}"
            );
        }
    }

    #[test]
    fn the_rewards_come_after_the_outcome_and_before_the_stop() {
        // The shutdown's order, read off the effects as the loop sees
        // them: the narration is said, then every reward is logged, then
        // everybody is stopped. Nothing follows the stop.
        for Played { effects, .. } in played_games() {
            let kinds: Vec<&str> = effects
                .iter()
                .map(|effect| match effect {
                    Effect::Act(Action {
                        payload: Message::Narration(Narration::Outcome(_)),
                        ..
                    }) => "outcome",
                    Effect::Act(_) => "act",
                    Effect::Reward { .. } => "reward",
                    Effect::Control { control, .. } => match control {
                        Control::Start => "start",
                        Control::Stop => "stop",
                    },
                })
                .collect();
            let outcome = kinds.iter().position(|kind| *kind == "outcome").unwrap();
            let stop = kinds.iter().position(|kind| *kind == "stop").unwrap();
            let rewards: Vec<usize> = kinds
                .iter()
                .enumerate()
                .filter(|(_, kind)| **kind == "reward")
                .map(|(at, _)| at)
                .collect();
            assert!(!rewards.is_empty());
            assert!(
                rewards.iter().all(|at| outcome < *at && *at < stop),
                "every reward falls between the outcome and the stop: {kinds:?}"
            );
            assert_eq!(
                stop,
                kinds.len() - 1,
                "the stop is the last word: {kinds:?}"
            );
        }
    }

    #[test]
    fn the_game_ending_narrates_then_publishes_then_pays_then_stops_everybody() {
        for Played {
            assignment,
            outcome,
            sent,
            commanded,
            ..
        } in played_games()
        {
            let everyone: BTreeSet<AgentId> =
                assignment.players().map(|(who, _)| who.clone()).collect();
            // The outcome is narrated to exactly the living, which is what
            // the outcome itself names.
            let (to, announced) = announced_outcome(&sent);
            assert_eq!(*announced, outcome);
            assert_eq!(*to, outcome.living);
            // And is the last thing said.
            assert_eq!(
                sent.last().map(|action| &action.payload),
                Some(&Message::Narration(Narration::Outcome(outcome.clone())))
            );
            // The players are started once and stopped once, the stop last
            // and to everyone, living and dead.
            assert_eq!(
                commanded,
                [
                    (everyone.clone(), Control::Start),
                    (everyone, Control::Stop)
                ]
            );
        }
    }

    #[test]
    fn a_whole_game_reaches_an_outcome_and_sends_it_on_the_channel() {
        for Played {
            outcome,
            receiver,
            sent,
            ..
        } in played_games()
        {
            let (_, announced) = announced_outcome(&sent);
            assert_eq!(*announced, outcome);
            // The moderator, and with it the sender, is gone: the channel
            // holds the one outcome and nothing else.
            assert_eq!(receiver.try_recv().as_ref(), Ok(announced));
            assert_eq!(receiver.try_recv(), Err(TryRecvError::Disconnected));
        }
    }

    #[test]
    fn the_stub_players_between_them_reach_every_terminal_state() {
        let winners: Vec<Faction> = played_games()
            .iter()
            .map(|played| played.outcome.winner)
            .collect();
        assert!(winners.contains(&Faction::Village), "{winners:?}");
        assert!(winners.contains(&Faction::Werewolves), "{winners:?}");
    }

    #[test]
    fn nothing_the_moderator_says_is_broadcast() {
        // Routing is the whole hidden-information mechanism, and there is
        // no longer an exception for the outcome: every message the
        // moderator sends names its recipients.
        for Played { sent, .. } in played_games() {
            for action in &sent {
                assert!(
                    matches!(action.recipients, Recipients::To(_)),
                    "{action:?} is broadcast"
                );
            }
        }
    }

    #[test]
    fn no_action_has_an_empty_recipient_set() {
        for Played { sent, .. } in played_games() {
            for action in &sent {
                if let Recipients::To(to) = &action.recipients {
                    assert!(!to.is_empty(), "{action:?} is addressed to nobody");
                }
            }
        }
    }

    #[test]
    fn no_action_addresses_a_dead_player() {
        for Played { sent, .. } in played_games() {
            let mut dead = BTreeSet::new();
            for message in &sent {
                let Recipients::To(to) = &message.recipients else {
                    continue;
                };
                let own_death = match &message.payload {
                    Message::Narration(Narration::Eliminated { who, .. }) => Some(who),
                    _ => None,
                };
                for who in to {
                    assert!(
                        !dead.contains(who) || own_death == Some(who),
                        "{who} is dead but is sent {message:?}"
                    );
                }
                if let Some(who) = own_death {
                    dead.insert(who.clone());
                }
            }
            assert!(!dead.is_empty());
        }
    }

    #[test]
    fn the_pack_is_named_only_to_werewolves() {
        for Played {
            assignment, sent, ..
        } in played_games()
        {
            for message in &sent {
                let Message::Narration(Narration::Assigned { pack, .. }) = &message.payload else {
                    continue;
                };
                let Recipients::To(to) = &message.recipients else {
                    panic!("an assignment was broadcast: {message:?}");
                };
                for who in to {
                    if assignment.role(who) == Some(Werewolf) {
                        assert_eq!(pack, assignment.pack(), "{who}");
                    } else {
                        assert!(pack.is_empty(), "{who} is told the pack {pack:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn nothing_is_emitted_after_the_outcome() {
        let (mut moderator, _receiver) = moderator(village());
        let sent = actions(&play(&mut moderator, first_other));
        let (who, request) = sent
            .iter()
            .rev()
            .find_map(|message| match &message.payload {
                Message::Request(request) => Some((asked(message).clone(), request.id)),
                _ => None,
            })
            .unwrap();
        let late = response(&who, request, Move::Target(id("bob")));
        assert_eq!(moderator.handle(&late), []);
    }

    #[test]
    fn a_response_repeated_after_the_game_ended_emits_nothing() {
        // Two moderators play the same game in lockstep. One of them is
        // handed each response a second time, in a cycle of its own, and
        // must say nothing for the repeat: before the game ends the fold is
        // the game's to refuse, and after it ends the outcome guard stops
        // the fold before it starts.
        let (mut reference, _receiver) = moderator(village());
        let (mut doubled, _receiver) = moderator(village());
        let opening = reference.start();
        assert_eq!(doubled.start(), opening);
        let mut pending = respond(&actions(&opening), first_other, reference.game.living());
        while !pending.is_empty() {
            let mut effects = Vec::new();
            for observation in &pending {
                let produced = reference.handle(observation);
                assert_eq!(doubled.handle(observation), produced);
                // The repeat is only safe once the game has ended; before
                // that the game would refuse a response it has already
                // recorded, which is a different claim and its own test.
                if doubled.game.outcome().is_some() {
                    assert_eq!(doubled.handle(observation), []);
                }
                effects.extend(produced);
            }
            pending = respond(&actions(&effects), first_other, reference.game.living());
        }
        assert!(reference.game.outcome().is_some() && doubled.game.outcome().is_some());
    }

    #[test]
    #[should_panic(expected = "erin sent the moderator a narration")]
    fn a_narration_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.start();
        moderator.handle(&from_player(
            "erin",
            Message::Narration(Narration::NoDeath { round: Round(1) }),
        ));
    }

    #[test]
    #[should_panic(expected = "carol sent the moderator a request")]
    fn a_request_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.start();
        moderator.handle(&from_player(
            "carol",
            Message::Request(Request {
                id: RequestId(1),
                round: Round(1),
                kind: RequestKind::Nominate,
            }),
        ));
    }

    #[test]
    fn a_dropped_receiver_is_not_an_error() {
        let (mut moderator, receiver) = moderator(town());
        drop(receiver);
        let sent = actions(&play(&mut moderator, last_other));
        assert_eq!(moderator.game.outcome(), Some(announced_outcome(&sent).1));
    }
}
