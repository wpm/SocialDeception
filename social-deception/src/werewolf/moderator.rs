//! The moderator: the game state machine as an agent in the roster.
//!
//! [`Moderator`] is the [`Handler`] around a [`Game`]. It is plumbing, and
//! thin by design: every rule of Werewolf lives in [`Game`], and this module
//! only folds the events that arrive on the moderator's inbox into calls on
//! the game and turns the [`Directive`]s that come back into [`Outgoing`]
//! messages. The one thing it adds is the invariants the runtime imposes on
//! how those messages are addressed.
//!
//! # What the moderator does with each event
//!
//! [`Control::Start`] begins the game; a [`Response`](super::Response) from a player is
//! recorded; [`Control::Stop`] and [`Event::Think`] do nothing. An episode's
//! agents have no think interval, so no `Think` ever arrives; the arm exists
//! so that turning timers on later is not a change of behaviour here. A
//! [`Narration`](super::Narration) or a [`Request`](super::Request) arriving from a player is a bug, and the
//! moderator panics rather than run a game whose state it cannot vouch for.
//!
//! Once the game is over the moderator emits nothing, whatever arrives. The
//! episode is winding down, and a late response is not the game's problem.
//!
//! # The moderator never broadcasts, except once
//!
//! [`Outgoing::broadcast`] resolves to every other agent in the roster, so it
//! reaches every player, living and dead. That is exactly right for the
//! [`Outcome`] and exactly wrong for everything else, which is why every
//! other directive carries an explicit recipient set: narration is
//! addressed, not broadcast, and the choice of recipients is the whole
//! hidden-information mechanism (ADR-0004). The outcome is broadcast because
//! it is the reward signal. A dead werewolf whose pack went on to win needs
//! to observe that it won, or its trajectory has no terminal reward. It is
//! the one message a dead player receives after the announcement of its own
//! death.
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
//! # How the episode ends
//!
//! Nothing declares the episode over. The moderator broadcasts the outcome
//! and emits nothing further; the players consume it and reply nothing; the
//! episode goes quiescent and stops. The corollary is the failure mode to
//! remember: an episode that goes quiescent *without* an outcome is a
//! truncated game, in which some player did not respond to a request, and it
//! presents as a short trajectory rather than a hang. Detecting that is the
//! business of whoever runs the episode and holds the receiver.

use crossbeam_channel::Sender;

use super::game::{Directive, Game};
use super::message::{Message, Outcome};
use crate::agent::{Handler, Outgoing, Recipients};
use crate::event::{Control, Event};

/// The agent that runs a game of Werewolf: a [`Game`] behind a
/// [`Handler`].
#[derive(Debug)]
pub struct Moderator {
    game: Game,
    outcome: Sender<Outcome>,
    /// Whether the outcome has been announced and sent on the channel.
    over: bool,
}

impl Moderator {
    /// A moderator that runs `game` and, when it ends, sends its outcome
    /// on `outcome`.
    ///
    /// The game must not have begun: the moderator begins it on
    /// [`Control::Start`].
    #[must_use]
    pub const fn new(game: Game, outcome: Sender<Outcome>) -> Self {
        Self {
            game,
            outcome,
            over: false,
        }
    }

    /// What one event makes the game say.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration or a request.
    fn fold(&mut self, event: &Event<Message>) -> Vec<Directive> {
        match event {
            Event::Control(Control::Start) => self.game.begin(),
            Event::Control(Control::Stop) | Event::Think => Vec::new(),
            Event::Message {
                sender, payload, ..
            } => match payload {
                Message::Response(response) => self.game.record(sender, response),
                Message::Narration(_) => panic!("{sender} sent the moderator a narration"),
                Message::Request(_) => panic!("{sender} sent the moderator a request"),
            },
        }
    }
}

/// The message that carries one directive.
fn send(directive: Directive) -> Outgoing<Message> {
    match directive {
        Directive::Narrate { to, narration } => Outgoing {
            recipients: Recipients::To(to),
            payload: Message::Narration(narration),
        },
        Directive::Ask { to, request } => Outgoing::to([to], Message::Request(request)),
        Directive::Broadcast(narration) => Outgoing::broadcast(Message::Narration(narration)),
    }
}

impl Handler<Message> for Moderator {
    /// Folds each event into the game in order and says what the game wants
    /// said. The event that ends the game also sends the outcome on the
    /// channel; after it, nothing, whatever arrives.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration or a request, or if a
    /// response is one the game cannot accept; see [`Game::record`].
    fn handle(&mut self, events: &[Event<Message>]) -> Vec<Outgoing<Message>> {
        let mut outgoing = Vec::new();
        for event in events {
            if self.over {
                break;
            }
            outgoing.extend(self.fold(event).into_iter().map(send));
            if let Some(outcome) = self.game.outcome() {
                self.over = true;
                // The caller may have dropped the receiver. That is not the
                // game's problem: the in-world announcement is the record.
                let _ = self.outcome.send(outcome.clone());
            }
        }
        outgoing
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crossbeam_channel::{Receiver, TryRecvError, unbounded};

    use super::*;
    use crate::agent::Recipients;
    use crate::event::AgentId;
    use crate::testing::{id, ids};
    use crate::werewolf::assignment::Assignment;
    use crate::werewolf::message::{
        Action, Narration, Phase, Request, RequestId, RequestKind, Response, Round,
    };
    use crate::werewolf::role::Faction;
    use crate::werewolf::role::Role::{self, Doctor, Seer, Villager, Werewolf};

    const MODERATOR: &str = "moderator";
    const SEED: u64 = 20_260_918;
    const MAX_ROUNDS: u32 = 100;

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
        let game = Game::new(assignment, MAX_ROUNDS, SEED);
        (Moderator::new(game, sender), receiver)
    }

    fn start() -> Event<Message> {
        Event::Control(Control::Start)
    }

    /// A message from a player to the moderator.
    fn from_player(who: &str, payload: Message) -> Event<Message> {
        Event::message(who, [MODERATOR], payload)
    }

    fn response(who: &AgentId, request: RequestId, action: Action) -> Event<Message> {
        from_player(
            who.as_str(),
            Message::Response(Response { request, action }),
        )
    }

    /// A stub player: what it does with a request, given who it is and who
    /// is living.
    type Policy = fn(&AgentId, &Request, &BTreeSet<AgentId>) -> Action;

    /// Targets the first living player other than itself.
    fn first_other(me: &AgentId, _: &Request, living: &BTreeSet<AgentId>) -> Action {
        Action::Target(living.iter().find(|who| *who != me).unwrap().clone())
    }

    /// Targets the last living player other than itself.
    fn last_other(me: &AgentId, _: &Request, living: &BTreeSet<AgentId>) -> Action {
        Action::Target(living.iter().rev().find(|who| *who != me).unwrap().clone())
    }

    /// Abstains wherever the request permits it, and otherwise targets the
    /// last living player other than itself.
    fn abstainer(me: &AgentId, request: &Request, living: &BTreeSet<AgentId>) -> Action {
        if request.kind.may_abstain() {
            Action::Abstain
        } else {
            last_other(me, request, living)
        }
    }

    /// The agents an outgoing message is addressed to, which for a request
    /// is exactly one.
    fn asked(outgoing: &Outgoing<Message>) -> &AgentId {
        let Recipients::To(to) = &outgoing.recipients else {
            panic!("a request was broadcast: {outgoing:?}");
        };
        assert_eq!(to.len(), 1, "a request to several players: {outgoing:?}");
        to.iter().next().unwrap()
    }

    /// The responses stub players following `policy` give to the requests
    /// among some outgoing messages, choosing among `living`.
    fn respond(
        outgoing: &[Outgoing<Message>],
        policy: Policy,
        living: &BTreeSet<AgentId>,
    ) -> Vec<Event<Message>> {
        outgoing
            .iter()
            .filter_map(|message| match &message.payload {
                Message::Request(request) => {
                    let who = asked(message);
                    Some(response(who, request.id, policy(who, request, living)))
                }
                Message::Narration(_) | Message::Response(_) => None,
            })
            .collect()
    }

    /// Plays a whole game through `handle`, with stub players following
    /// `policy`, and returns every outgoing message in order.
    ///
    /// Each phase's responses arrive as one batch when `batched`, and one
    /// per batch otherwise. Either way the game runs until the moderator
    /// asks nothing more.
    fn play(moderator: &mut Moderator, policy: Policy, batched: bool) -> Vec<Outgoing<Message>> {
        let mut sent = Vec::new();
        let mut pending = vec![start()];
        while !pending.is_empty() {
            let outgoing = if batched {
                moderator.handle(&pending)
            } else {
                pending
                    .iter()
                    .flat_map(|event| moderator.handle(std::slice::from_ref(event)))
                    .collect()
            };
            pending = respond(&outgoing, policy, moderator.game.living());
            sent.extend(outgoing);
        }
        sent
    }

    /// A game played out: its assignment, how the game says it ended, the
    /// receiver of its outcome, and every message the moderator sent.
    struct Played {
        assignment: Assignment,
        outcome: Outcome,
        receiver: Receiver<Outcome>,
        sent: Vec<Outgoing<Message>>,
    }

    /// Every combination of assignment and stub policy, played out.
    fn played_games() -> Vec<Played> {
        let policies: [Policy; 3] = [first_other, last_other, abstainer];
        let mut games = Vec::new();
        for assignment in [village(), town()] {
            for policy in policies {
                let (mut moderator, receiver) = moderator(assignment.clone());
                let sent = play(&mut moderator, policy, true);
                games.push(Played {
                    assignment: assignment.clone(),
                    outcome: moderator.game.outcome().unwrap().clone(),
                    receiver,
                    sent,
                });
            }
        }
        games
    }

    /// The outcome broadcast among some messages.
    fn broadcast_outcome(sent: &[Outgoing<Message>]) -> &Outcome {
        sent.iter()
            .find_map(|message| match message {
                Outgoing {
                    recipients: Recipients::Broadcast,
                    payload: Message::Narration(Narration::Outcome(outcome)),
                } => Some(outcome),
                _ => None,
            })
            .expect("no outcome was broadcast")
    }

    fn narrate<const N: usize>(to: [&str; N], narration: Narration) -> Outgoing<Message> {
        Outgoing::to(to, Message::Narration(narration))
    }

    fn assigned<const N: usize>(to: &str, role: Role, pack: [&str; N]) -> Outgoing<Message> {
        narrate(
            [to],
            Narration::Assigned {
                role,
                pack: ids(pack),
            },
        )
    }

    fn ask(to: &str, id: u64, round: u32, kind: RequestKind) -> Outgoing<Message> {
        Outgoing::to(
            [to],
            Message::Request(Request {
                id: RequestId(id),
                round: Round(round),
                kind,
            }),
        )
    }

    #[test]
    fn start_begins_the_game_with_addressed_messages() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let (mut moderator, _receiver) = moderator(village());
        assert_eq!(
            moderator.handle(&[start()]),
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
    fn stop_and_think_produce_nothing() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.handle(&[start()]);
        assert_eq!(
            moderator.handle(&[Event::Think, Event::Control(Control::Stop)]),
            []
        );
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
            let announced = broadcast_outcome(&sent);
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
    fn the_outcome_is_broadcast_exactly_once_and_last() {
        for Played { sent, .. } in played_games() {
            let broadcasts: Vec<usize> = sent
                .iter()
                .enumerate()
                .filter(|(_, message)| message.recipients == Recipients::Broadcast)
                .map(|(index, message)| {
                    assert!(
                        matches!(message.payload, Message::Narration(Narration::Outcome(_))),
                        "{message:?}"
                    );
                    index
                })
                .collect();
            assert_eq!(broadcasts, [sent.len() - 1]);
        }
    }

    #[test]
    fn no_message_has_an_empty_recipient_set() {
        for Played { sent, .. } in played_games() {
            for message in &sent {
                if let Recipients::To(to) = &message.recipients {
                    assert!(!to.is_empty(), "{message:?} is addressed to nobody");
                }
            }
        }
    }

    #[test]
    fn no_message_addresses_a_dead_player() {
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
        let sent = play(&mut moderator, first_other, true);
        let (who, request) = sent
            .iter()
            .rev()
            .find_map(|message| match &message.payload {
                Message::Request(request) => Some((asked(message).clone(), request.id)),
                _ => None,
            })
            .unwrap();
        let late = [
            response(&who, request, Action::Target(id("bob"))),
            Event::Control(Control::Stop),
            Event::Think,
            start(),
        ];
        assert_eq!(moderator.handle(&late), []);
    }

    #[test]
    fn a_batch_ending_the_game_emits_nothing_for_the_rest_of_it() {
        // Two moderators play the same game in lockstep. The one whose
        // batch ends the game gets a duplicate of the response that ended
        // it, in the same batch, and must say nothing for it.
        let (mut reference, _receiver) = moderator(village());
        let (mut padded, _receiver) = moderator(village());
        let mut pending = vec![start()];
        while !pending.is_empty() {
            let outgoing = reference.handle(&pending);
            let mut batch = pending.clone();
            if outgoing
                .last()
                .is_some_and(|last| last.recipients == Recipients::Broadcast)
            {
                batch.push(pending.last().unwrap().clone());
            }
            assert_eq!(padded.handle(&batch), outgoing);
            pending = respond(&outgoing, first_other, reference.game.living());
        }
        assert!(reference.over && padded.over);
    }

    #[test]
    #[should_panic(expected = "erin sent the moderator a narration")]
    fn a_narration_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.handle(&[start()]);
        moderator.handle(&[from_player(
            "erin",
            Message::Narration(Narration::NoDeath { round: Round(1) }),
        )]);
    }

    #[test]
    #[should_panic(expected = "carol sent the moderator a request")]
    fn a_request_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        moderator.handle(&[start()]);
        moderator.handle(&[from_player(
            "carol",
            Message::Request(Request {
                id: RequestId(1),
                round: Round(1),
                kind: RequestKind::Nominate,
            }),
        )]);
    }

    #[test]
    fn a_dropped_receiver_is_not_an_error() {
        let (mut moderator, receiver) = moderator(town());
        drop(receiver);
        let sent = play(&mut moderator, last_other, true);
        assert_eq!(moderator.game.outcome(), Some(broadcast_outcome(&sent)));
    }

    #[test]
    fn responses_in_one_batch_or_several_produce_the_same_game() {
        for assignment in [village(), town()] {
            let (mut batched, batched_receiver) = moderator(assignment.clone());
            let (mut separate, separate_receiver) = moderator(assignment);
            assert_eq!(
                play(&mut batched, last_other, true),
                play(&mut separate, last_other, false)
            );
            assert_eq!(batched_receiver.try_recv(), separate_receiver.try_recv());
        }
    }
}
