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
//! (ADR-0007). Thereafter a [`Select`](super::Select) from a player is
//! recorded. The moderator never sees the control that started it, nor the
//! one that stops it; those are the loop's, which is why there is no arm for
//! either here.
//!
//! A [`Narration`](super::Narration) arriving from a player is a bug, and
//! the moderator panics rather than run a game whose state it cannot vouch
//! for. So is a selection in a session the rules never make that player a
//! member of; the moderator checks that when the selection arrives, rather
//! than handing out permission in advance (ADR-0014).
//!
//! Once the game is over the moderator emits nothing, whatever arrives. The
//! episode is winding down, and a late response is not the game's problem.
//!
//! # Every message of this game names somebody
//!
//! Every message carries an explicit recipient set, and the choice of
//! recipients is the whole hidden-information mechanism (ADR-0004).
//! ADR-0004 made one exception, sending the final [`Outcome`] to every
//! player, living and dead, because it was a dead player's terminal reward
//! signal. A reward is now logged rather than said (ADR-0007), so the
//! exception is withdrawn: the outcome is narrated to the living like any
//! other narration, and a dead player's records end at the announcement
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
//! each reward in the log ahead of the `Stop` control that closes the
//! records it belongs to, which is the ordering a reader can then
//! rely on.
//!
//! The episode routes a cycle's messages before the controls it asked for, so
//! a living player observes the outcome and *then* stops, rather than
//! stopping with the outcome still on its queue and never seeing it. A dead
//! player is sent no outcome and only the stop. The episode stops the
//! moderator itself once every player's thread has ended.
//!
//! A game that never reaches an outcome is nobody's to notice here:
//! nothing is in flight, no player has been stopped, and the episode
//! calls that
//! [`EpisodeError::Stalled`](crate::EpisodeError::Stalled).

use crossbeam_channel::Sender;

use std::collections::BTreeSet;
use std::time::Instant;

use super::game::{Directive, Game};
use super::message::{Message, Outcome};
use crate::agent::{Action, Observation};
use crate::environment::{Effect, Environment};
use crate::message::{ActorId, Control};

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
    fn players(&self) -> BTreeSet<ActorId> {
        self.game.players().cloned().collect()
    }

    /// What one observation makes the game say.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration.
    fn fold(&mut self, observation: &Observation<Message>) -> Vec<Directive> {
        let sender = &observation.message.sender;
        let now = observation.at;
        let mut directives = match &observation.message.payload {
            // The message's own sequence number goes in as well as the
            // arrival: a forwarded selection carries the player's
            // `(sender, seq)` so the relay joins back to the player's
            // action, while `now` is when the moderator got it, which is
            // what the session clocks run on.
            Message::Select(selection) => {
                self.game
                    .select(sender, selection, observation.message.seq, now)
            }
            Message::Narration(_) => panic!("{sender} sent the moderator a narration"),
        };
        // A deadline that passed while this observation waited joins its
        // cycle (ADR-0008), so a session whose time is up closes here
        // rather than waiting for a `timeout` that may never come.
        directives.extend(self.game.expire(now));
        directives
    }

    /// Everything the directives say, said; then, if the game has just
    /// ended, the outcome published on the channel, every player paid, and
    /// every player stopped. The order is the shutdown sequence; see the
    /// [module documentation](self).
    fn say(&mut self, directives: Vec<Directive>) -> Vec<Effect<i32, Message>> {
        let mut effects: Vec<Effect<i32, Message>> = directives.into_iter().map(send).collect();
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
            // Only the living: a dead player was stopped in the cycle
            // its death was announced (ADR-0012), and stopping it again
            // would claim in its records that it was told to stop
            // after it had already stopped.
            effects.push(Effect::control(self.game.living().clone(), Control::Stop));
        }
        effects
    }
}

/// The effect that carries one directive.
///
/// Most are something said; a [`Directive::Stop`] is the game putting a
/// player out of it, which is a control rather than a message (ADR-0012).
fn send(directive: Directive) -> Effect<i32, Message> {
    match directive {
        Directive::Narrate { to, narration } => {
            Effect::Act(Action::to(to, Message::Narration(narration)))
        }
        // A forwarded selection is sent as the player that made it, not as the
        // moderator: what a recipient observes is what it would have
        // observed had the player addressed it directly (ADR-0014).
        Directive::Forward {
            from,
            seq,
            to,
            selection,
        } => Effect::Act(Action::relay(from, seq, to, Message::Select(selection))),
        Directive::Stop { who } => Effect::control([who], Control::Stop),
    }
}

impl Environment<i32, Message> for Moderator {
    /// Starts every player, then begins the game and says what it wants
    /// said at the start: the roles, and that the first night has begun.
    /// The players take it from there (ADR-0014).
    ///
    /// The `Start` comes first among the effects, but the episode routes a
    /// cycle's messages before its controls either way, so what a player
    /// actually sees is its `Start` — controls are popped first — and then
    /// the opening narrations.
    fn start(&mut self, now: Instant) -> Vec<Effect<i32, Message>> {
        let opening = self.game.begin(now);
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
    /// If a player sends the moderator a narration, or if a selection is one
    /// the game cannot accept; see [`Game::select`].
    fn handle(&mut self, observation: &Observation<Message>) -> Vec<Effect<i32, Message>> {
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

    /// Closes every session whose time is up.
    ///
    /// The moderator is the only agent in the tree that keeps clocks. It
    /// wakes on the earliest of them and asks the game what that instant
    /// finished. A session's close is the moderator's own business and is
    /// narrated to nobody (ADR-0015); what its members hear is what the
    /// phase came to, and the last of a night's sessions resolves the
    /// night.
    fn timeout(&mut self, now: Instant) -> Vec<Effect<i32, Message>> {
        if self.game.outcome().is_some() {
            return Vec::new();
        }
        let directives = self.game.expire(now);
        self.say(directives)
    }

    /// The earliest instant a session could close, or `None` once the game
    /// is over (ADR-0010, ADR-0011).
    fn deadline(&self) -> Option<Instant> {
        self.game.next_deadline()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crossbeam_channel::{Receiver, TryRecvError, unbounded};

    use super::*;

    use crate::message::ActorId;
    use crate::testing::{fast, id, ids, observed, town, village};
    use crate::werewolf::assignment::Assignment;
    use crate::werewolf::message::{Narration, Phase, Round, Select, SessionKind};
    use crate::werewolf::role::Faction;
    use crate::werewolf::role::Role::{self, Doctor, Seer, Villager, Werewolf};

    use crate::agent::Action;

    const MODERATOR: &str = "moderator";
    const SEED: u64 = 20_260_918;

    fn moderator(assignment: Assignment) -> (Moderator, Receiver<Outcome>) {
        let (sender, receiver) = unbounded();
        let game = Game::new(assignment, SEED, fast());
        (Moderator::new(game, sender), receiver)
    }

    /// A message from a player to the moderator, as the moderator observes
    /// it. Which of that player's messages it is plays no part in the fold
    /// here, so one stand-in number serves every test.
    fn from_player(who: &str, payload: Message) -> Observation<Message> {
        observed(crate::Message::new(who, [MODERATOR], 0, payload))
    }

    fn response(
        who: &ActorId,
        round: Round,
        kind: SessionKind,
        target: ActorId,
    ) -> Observation<Message> {
        from_player(
            who.as_str(),
            Message::Select(Select {
                round,
                kind,
                target,
                seen_by: BTreeSet::new(),
            }),
        )
    }

    /// A stub player: whom it selects, out of the targets the rules
    /// permit it.
    ///
    /// The space is the game's own, so a stub cannot select outside it; the
    /// doctor's "not last night's patient" in particular is the rules'
    /// business rather than every stub's.
    type Strategy = fn(SessionKind, &[ActorId]) -> ActorId;

    /// Selects the first target the rules permit.
    fn first_other(_: SessionKind, space: &[ActorId]) -> ActorId {
        space.first().unwrap().clone()
    }

    /// Selects the last target the rules permit.
    fn last_other(_: SessionKind, space: &[ActorId]) -> ActorId {
        space.last().unwrap().clone()
    }

    /// Selects the last permitted target by night and the first by day,
    /// so that the pack and the village disagree about whom to blame.
    fn two_minded(kind: SessionKind, space: &[ActorId]) -> ActorId {
        match kind.phase() {
            Phase::Night => last_other(kind, space),
            Phase::Day => first_other(kind, space),
        }
    }

    /// The actions among some effects, in order.
    fn actions(effects: &[Effect<i32, Message>]) -> Vec<Action<Message>> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Act(action) => Some(action.clone()),
                Effect::Control { .. } | Effect::Reward { .. } => None,
            })
            .collect()
    }

    /// The controls among some effects, in order.
    fn controls(effects: &[Effect<i32, Message>]) -> Vec<(BTreeSet<ActorId>, Control)> {
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
    fn rewards(effects: &[Effect<i32, Message>]) -> Vec<(ActorId, i32)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Reward { agent, value } => Some((agent.clone(), *value)),
                Effect::Act(_) | Effect::Control { .. } => None,
            })
            .collect()
    }

    /// The selections stub players following `strategy` make when a phase
    /// begins.
    ///
    /// This is a stub of [`Player`](super::Player) and acts the way one does
    /// (ADR-0014): a phase beginning is what makes a player select, and
    /// each recipient asks its own role what that phase wants of it
    /// rather than waiting to be told. Nothing among the actions is a
    /// request, because the moderator no longer sends any.
    ///
    /// A player the rules leave nowhere to select says nothing, which is
    /// the same thing the game means by leaving it out of the session.
    fn respond(
        actions: &[Action<Message>],
        strategy: Strategy,
        game: &Game,
        roles: &Assignment,
    ) -> Vec<Observation<Message>> {
        actions
            .iter()
            .filter_map(|action| match &action.payload {
                Message::Narration(Narration::PhaseBegan { round, phase, .. }) => {
                    Some((*round, *phase, &action.recipients))
                }
                _ => None,
            })
            .flat_map(|(round, phase, to)| {
                to.iter().filter_map(move |who| {
                    // Its own role, from the deal, is what tells a stub
                    // whether this phase wants anything of it. That the
                    // moderator happens to hold the same assignment is
                    // beside the point: a player consults the role it
                    // was dealt.
                    let kind = roles.role(who)?.asked_in(phase)?;
                    let space = game.action_space_for(who, kind);
                    if space.is_empty() {
                        return None;
                    }
                    Some(response(who, round, kind, strategy(kind, &space)))
                })
            })
            .collect()
    }

    /// Plays a whole game, with stub players following `strategy`, and
    /// returns every effect the moderator produced in order.
    ///
    /// The game opens with the start hook, as the agent loop opens it, and
    /// each response then arrives in a cycle of its own, as the loop hands
    /// them over one at a time (ADR-0008). The game runs until the
    /// moderator asks nothing more.
    fn play(
        moderator: &mut Moderator,
        strategy: Strategy,
        roles: &Assignment,
    ) -> Vec<Effect<i32, Message>> {
        /// Far enough apart that one phase's clocks never reach the next.
        const STEP: u64 = 10_000;

        let mut clock = 0;
        let mut produced = moderator.start(at(clock));
        let mut pending = respond(&actions(&produced), strategy, &moderator.game, roles);
        while moderator.game.outcome().is_none() {
            clock += STEP;
            let mut effects: Vec<Effect<i32, Message>> = pending
                .iter()
                .flat_map(|observation| moderator.handle(observation))
                .collect();
            // Every stub selects once and never changes its mind, so
            // running the clock out is what closes the phase (ADR-0011).
            clock += STEP;
            effects.extend(moderator.timeout(at(clock)));
            pending = respond(&actions(&effects), strategy, &moderator.game, roles);
            produced.extend(effects);
            assert!(
                clock < STEP * 200,
                "a stub game should have ended long before now"
            );
        }
        produced
    }

    /// The instants these tests name, offset from the one fixed base every
    /// test module in the crate shares; see [`testing::BASE`](crate::testing).
    use crate::testing::at_millis as at;

    /// A game played out: its assignment, how the game says it ended, the
    /// receiver of its outcome, every message the moderator sent, every
    /// control it asked for, every reward it paid, and the whole sequence
    /// of effects it produced.
    struct Played {
        assignment: Assignment,
        outcome: Outcome,
        receiver: Receiver<Outcome>,
        sent: Vec<Action<Message>>,
        commanded: Vec<(BTreeSet<ActorId>, Control)>,
        paid: Vec<(ActorId, i32)>,
        /// Every effect in the order the moderator produced it, which is
        /// what the shutdown's ordering is asserted against.
        effects: Vec<Effect<i32, Message>>,
    }

    /// Every combination of assignment and stub strategy, played out.
    fn played_games() -> Vec<Played> {
        let strategies: [Strategy; 3] = [first_other, last_other, two_minded];
        let mut games = Vec::new();
        for assignment in [village(), town()] {
            for strategy in strategies {
                let (mut moderator, receiver) = moderator(assignment.clone());
                let effects = play(&mut moderator, strategy, &assignment);
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
    fn announced_outcome(sent: &[Action<Message>]) -> (&BTreeSet<ActorId>, &Outcome) {
        sent.iter()
            .find_map(|action| match &action.payload {
                Message::Narration(Narration::Outcome(outcome)) => {
                    Some((&action.recipients, outcome))
                }
                _ => None,
            })
            .expect("no outcome was announced")
    }

    fn narrate<const N: usize>(to: [&str; N], narration: Narration) -> Action<Message> {
        Action::to(to, Message::Narration(narration))
    }

    fn assigned<const N: usize>(to: &str, role: Role, pack: [&str; N]) -> Action<Message> {
        narrate(
            [to],
            Narration::Assigned {
                role,
                pack: ids(pack),
            },
        )
    }

    #[test]
    fn starting_the_moderator_starts_the_players_and_begins_the_game() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let (mut moderator, _receiver) = moderator(village());
        let opening = moderator.start(at(0));
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
                        round: Round::new(1),
                        phase: Phase::Night,
                        living: ids(everyone),
                    }
                ),
            ],
            "the deals and the phase, and nothing asked of anybody"
        );
    }

    #[test]
    fn a_timeout_produces_nothing() {
        // What a timeout cycle looks like from inside the handler: the loop
        // calls `timeout`, not `handle`, and the moderator has nothing to
        // say on a deadline. Controls never reach it either, so there is
        // nothing a cycle can hold that it must ignore.
        let (mut moderator, _receiver) = moderator(village());
        moderator.start(at(0));
        assert_eq!(moderator.timeout(at(0)), []);
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
            let expected: Vec<(ActorId, i32)> = assignment
                .players()
                .map(|(who, role)| {
                    let value = if outcome.winner == Some(role.faction()) {
                        1
                    } else {
                        -1
                    };
                    (who.clone(), value)
                })
                .collect();
            assert_eq!(paid, expected, "{outcome:?}");
            let dead: Vec<&ActorId> = paid
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
        // whoever is left is stopped. Nothing follows that stop.
        //
        // It is the *last* stop that ends the episode. Earlier ones are
        // dead players, each stopped in the cycle its death was announced
        // (ADR-0012).
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
            let stop = kinds.iter().rposition(|kind| *kind == "stop").unwrap();
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
            let everyone: BTreeSet<ActorId> =
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
            // The players are started once, together. Each is then
            // stopped exactly once: a dead player in the cycle its death
            // was announced, and whoever is left at the end (ADR-0012).
            let (started, stops) = commanded.split_first().expect("a start");
            assert_eq!(*started, (everyone.clone(), Control::Start));
            let mut stopped: Vec<ActorId> = Vec::new();
            for (to, control) in stops {
                assert_eq!(*control, Control::Stop);
                stopped.extend(to.iter().cloned());
            }
            assert_eq!(
                stopped.iter().cloned().collect::<BTreeSet<_>>(),
                everyone,
                "everybody is stopped"
            );
            assert_eq!(stopped.len(), everyone.len(), "and nobody twice");
            // The last stop takes exactly the survivors.
            assert_eq!(
                stops.last().map(|(to, _)| to),
                Some(&outcome.living),
                "the episode ends by stopping whoever is still living"
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
        let winners: Vec<Option<Faction>> = played_games()
            .iter()
            .map(|played| played.outcome.winner)
            .collect();
        assert!(winners.contains(&Some(Faction::Village)), "{winners:?}");
        assert!(winners.contains(&Some(Faction::Werewolves)), "{winners:?}");
    }

    #[test]
    fn every_action_names_somebody() {
        // Routing is the whole hidden-information mechanism, and there is
        // no longer an exception for the outcome: everything the moderator
        // says is said to somebody in particular. The runtime would carry
        // an action addressed to nobody; this game never sends one, and
        // this is where that is held.
        for Played { sent, .. } in played_games() {
            for action in &sent {
                assert!(
                    !action.recipients.is_empty(),
                    "{action:?} is addressed to nobody"
                );
            }
        }
    }

    #[test]
    fn no_action_addresses_a_dead_player() {
        for Played { sent, .. } in played_games() {
            let mut dead = BTreeSet::new();
            for message in &sent {
                let to = &message.recipients;
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
                for who in &message.recipients {
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
        // A selection that arrives once the game has ended. It is built by
        // hand rather than found among the requests sent, because none
        // are sent (ADR-0014): the outcome guard stops the fold before
        // anything about the selection is looked at, so any selection will do.
        let (mut moderator, _receiver) = moderator(village());
        play(&mut moderator, first_other, &village());
        assert!(moderator.game.outcome().is_some());
        let late = response(
            &id("carol"),
            Round::new(1),
            SessionKind::Nominate,
            id("bob"),
        );
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
        let mut clock = 0;
        let opening = reference.start(at(clock));
        assert_eq!(doubled.start(at(clock)), opening);
        let mut pending = respond(&actions(&opening), first_other, &reference.game, &village());
        while reference.game.outcome().is_none() {
            clock += 10_000;
            let mut effects = Vec::new();
            for observation in &pending {
                let produced = reference.handle(observation);
                assert_eq!(doubled.handle(observation), produced);
                // A selection repeated while its session is still open is
                // simply the same vote again, and says nothing new; once
                // the game has ended the outcome guard stops the fold
                // before it starts. Either way the repeat is silent.
                assert_eq!(doubled.handle(observation), []);
                effects.extend(produced);
            }
            // Both clocks run out together, so the two games stay in step.
            clock += 10_000;
            let expired = reference.timeout(at(clock));
            assert_eq!(doubled.timeout(at(clock)), expired);
            effects.extend(expired);
            pending = respond(&actions(&effects), first_other, &reference.game, &village());
            assert!(clock < 2_000_000, "a stub game should have ended by now");
        }
        assert!(reference.game.outcome().is_some() && doubled.game.outcome().is_some());
    }

    #[test]
    #[should_panic(expected = "erin sent the moderator a narration")]
    fn a_narration_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.start(at(0));
        moderator.handle(&from_player(
            "erin",
            Message::Narration(Narration::NoDeath {
                round: Round::new(1),
            }),
        ));
    }

    #[test]
    fn a_dropped_receiver_is_not_an_error() {
        let (mut moderator, receiver) = moderator(town());
        drop(receiver);
        let sent = actions(&play(&mut moderator, last_other, &town()));
        assert_eq!(moderator.game.outcome(), Some(announced_outcome(&sent).1));
    }
}
