//! The rules of Werewolf as a pure state machine.
//!
//! [`Game`] takes players' responses in and produces [`Directive`]s out: what
//! to say, to whom, in what order. It touches no channel and spawns no
//! thread, so the whole of the rules is testable by calling functions with a
//! scripted sequence of responses. The moderator that wraps it is plumbing,
//! folding the events it receives into [`Game::record`] and turning the
//! directives into messages.
//!
//! # How a game runs
//!
//! [`Game::begin`] tells each player its role, and a werewolf its pack, then
//! opens the first night. A phase issues every request it calls for at once,
//! and resolves at the moment the last of them is answered. A night asks
//! every living werewolf to devour, the living seer to investigate and the
//! living doctor to protect; a day asks every living player to nominate. No
//! request ever goes to a dead player.
//!
//! A night resolves in a fixed order: the tally of the werewolves' choices,
//! to the pack alone; the victim, a plurality of those choices; the seer's
//! finding, to the seer alone, before any death is announced, so a seer
//! devoured that same night still learns what it learned; then either the
//! death, revealed with its role to the living and to the victim, or, if the
//! doctor protected the victim, that nobody died. A save is never announced
//! as a save. A day resolves as the full tally to the living, then the
//! lynching of the plurality, revealed the same way.
//!
//! The win condition is checked after every elimination, and only then: the
//! village wins when no werewolf lives, and the werewolves win when they are
//! at least as many as everyone else, since from parity onward they cannot
//! lose. The outcome is the one thing said to everyone, living and dead.
//!
//! # Determinism
//!
//! The outcome of a phase does not depend on the order its responses arrive
//! in. Responses accumulate in a map keyed by the responding agent, the phase
//! advances exactly when nothing is outstanding, and resolution reads the map
//! in its canonical order. Ties are broken by a generator seeded from the
//! episode's master seed under its own label, so it is independent of every
//! player's, and it is drawn from only when there is an actual tie: whether
//! a vote was unanimous therefore has no effect on the generator's state, and
//! the same game unfolds the same way on every run. The seed stays inside
//! the game; no directive carries it.
//!
//! # Termination
//!
//! The day always eliminates someone, since a `Nominate` cannot abstain, so
//! the living set strictly shrinks every round even on a night when the
//! doctor saves. A game of n players is therefore over within n rounds
//! however its players answer, and it needs no cap to end: every game has a
//! winner.
//!
//! # Player bugs are panics
//!
//! A response that answers nothing outstanding, answers a request asked of
//! somebody else, or takes an action outside the request's action space is a
//! bug in a player, not a condition the game can continue from: the game
//! cannot vouch for a state built on it. Each panics with a message naming
//! the agent and the request. The action space a response is checked
//! against is [`roles::action_space`], the same function the role types
//! compute theirs with, so the game holds every player to exactly the rules
//! the roles apply to themselves, including the doctor's: for that the game
//! remembers whom each doctor protected last night, as the doctor does.

use std::collections::{BTreeMap, BTreeSet};
use std::mem;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::event::AgentId;
use crate::werewolf::assignment::Assignment;
use crate::werewolf::message::{
    Cause, Move, Narration, Outcome, Phase, Request, RequestId, RequestKind, Response, Round,
};
use crate::werewolf::role::{Faction, Role};
use crate::werewolf::roles;
use crate::werewolf::seed::{TIES, pick, seed_for};

/// Termination rests on the day always eliminating someone, so a
/// `Nominate` must never be allowed to abstain: a living set that could
/// abstain unanimously would leave a game to run forever. The check sits
/// next to the rules it protects, and it is a compile-time one because
/// [`RequestKind::may_abstain`] is `const`.
const NOMINATE_DECIDES: () = assert!(
    !RequestKind::Nominate.may_abstain(),
    "a Nominate that may abstain would let a game run forever"
);

/// What the game wants said, in the order it wants it said.
///
/// Every directive names its recipients. There is no broadcast: the choice
/// of recipients is the whole hidden-information mechanism, and the one
/// exception ADR-0004 made for the [`Outcome`] is withdrawn (ADR-0007),
/// which is why the outcome is a `Narrate` to the living like any other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Directive {
    /// To exactly these agents.
    Narrate {
        /// The recipients, never empty.
        to: BTreeSet<AgentId>,
        /// What they are told.
        narration: Narration,
    },
    /// A request for one agent to act.
    Ask {
        /// The agent asked.
        to: AgentId,
        /// What it is asked.
        request: Request,
    },
}

/// A game of Werewolf, from the deal to the outcome.
#[derive(Debug, Clone)]
pub struct Game {
    assignment: Assignment,
    living: BTreeSet<AgentId>,
    round: Round,
    phase: Phase,
    ties: ChaCha8Rng,
    /// How many requests have been issued; the next one's id is one more.
    issued: u64,
    /// The requests not yet answered: who was asked, and what.
    outstanding: BTreeMap<RequestId, (AgentId, RequestKind)>,
    /// This phase's answers so far, by the agent that gave them: what it
    /// was asked, and what it did.
    answers: BTreeMap<AgentId, (RequestKind, Move)>,
    /// Whom each doctor protected last night, for doctors that protected
    /// someone: the state the doctor's own rule constrains its next
    /// `Protect` with.
    last_protected: BTreeMap<AgentId, AgentId>,
    outcome: Option<Outcome>,
}

impl Game {
    /// A game over the given assignment, breaking ties with a generator
    /// derived from `seed`.
    ///
    /// # Panics
    ///
    /// If the assignment has no werewolf, or as many werewolves as other
    /// players, so that the game would be over before it began.
    #[must_use]
    pub fn new(assignment: Assignment, seed: u64) -> Self {
        let living: BTreeSet<AgentId> = assignment.players().map(|(who, _)| who.clone()).collect();
        let werewolves = assignment.pack().len();
        assert!(werewolves >= 1, "a game needs at least one werewolf");
        assert!(
            living.len() > 2 * werewolves,
            "a game needs more other players than werewolves"
        );
        Self {
            assignment,
            living,
            round: Round(1),
            phase: Phase::Night,
            ties: ChaCha8Rng::seed_from_u64(seed_for(seed, TIES)),
            issued: 0,
            outstanding: BTreeMap::new(),
            answers: BTreeMap::new(),
            last_protected: BTreeMap::new(),
            outcome: None,
        }
    }

    /// The opening directives: each player's role assignment, in agent
    /// order, then the first night.
    ///
    /// The pack is named only in a werewolf's assignment and is empty in
    /// everyone else's. That is where hidden information is enforced: by
    /// what is addressed to whom.
    ///
    /// # Panics
    ///
    /// If called more than once.
    pub fn begin(&mut self) -> Vec<Directive> {
        assert!(self.issued == 0, "the game has already begun");
        let mut directives: Vec<Directive> = self
            .assignment
            .players()
            .map(|(who, role)| {
                let pack = match role {
                    Role::Werewolf => self.assignment.pack().clone(),
                    Role::Villager | Role::Seer | Role::Doctor => BTreeSet::new(),
                };
                Directive::Narrate {
                    to: [who.clone()].into(),
                    narration: Narration::Assigned { role, pack },
                }
            })
            .collect();
        directives.extend(self.begin_phase());
        directives
    }

    /// Records one player's response and returns whatever it caused, which
    /// is nothing at all unless it answered the last request outstanding.
    ///
    /// # Panics
    ///
    /// If the response arrives after the game has ended, names a request
    /// that is not outstanding or was asked of another agent, or takes an
    /// action outside the request's action space as [`roles::action_space`]
    /// computes it: an abstention where the kind permits none, a target
    /// that is the responder itself or not living, or a doctor's protection
    /// of the player it protected the night before. Each is a bug in a
    /// player.
    pub fn record(&mut self, from: &AgentId, response: &Response) -> Vec<Directive> {
        let RequestId(id) = response.request;
        assert!(
            self.outcome.is_none(),
            "{from} answered request {id} after the game ended"
        );
        let Some((to, kind)) = self.outstanding.remove(&response.request) else {
            panic!("{from} answered request {id}, which is not outstanding");
        };
        assert!(
            to == *from,
            "{from} answered request {id}, which was asked of {to}"
        );
        let space = roles::action_space(from, &self.living, kind, self.last_protected.get(from));
        assert!(
            space.contains(&response.chosen),
            "{from} {} request {id}, which is outside its action space for {kind:?}",
            match &response.chosen {
                Move::Abstain => "abstained from".to_owned(),
                Move::Target(target) => format!("targeted {target} in"),
            }
        );
        self.answers.insert(to, (kind, response.chosen.clone()));
        if !self.outstanding.is_empty() {
            return Vec::new();
        }
        let answers = mem::take(&mut self.answers);
        match self.phase {
            Phase::Night => self.resolve_night(answers),
            Phase::Day => self.resolve_day(answers),
        }
    }

    /// How the game ended, once it has.
    #[must_use]
    pub fn outcome(&self) -> Option<&Outcome> {
        self.outcome.as_ref()
    }

    /// Everyone still in the game.
    #[must_use]
    pub fn living(&self) -> &BTreeSet<AgentId> {
        &self.living
    }

    /// Everyone dealt into the game, living and dead, in agent order.
    ///
    /// This is the roster the moderator starts and, when the game is over,
    /// stops. A player leaves the *game* when it is eliminated and the
    /// *episode* when it is stopped, and those are not the same moment.
    pub fn players(&self) -> impl Iterator<Item = &AgentId> {
        self.assignment.players().map(|(who, _)| who)
    }

    /// Announces the current phase to the living and issues its requests,
    /// in agent order.
    fn begin_phase(&mut self) -> Vec<Directive> {
        let mut directives = vec![self.narrate_living(Narration::PhaseBegan {
            round: self.round,
            phase: self.phase,
            living: self.living.clone(),
        })];
        let asked: Vec<(AgentId, RequestKind)> = self
            .living
            .iter()
            .filter_map(|who| Some((who.clone(), self.role(who).asked_in(self.phase)?)))
            .collect();
        for (who, kind) in asked {
            directives.push(self.ask(who, kind));
        }
        directives
    }

    /// Issues one request, drawing its id from the counter.
    fn ask(&mut self, to: AgentId, kind: RequestKind) -> Directive {
        self.issued += 1;
        let id = RequestId(self.issued);
        self.outstanding.insert(id, (to.clone(), kind));
        let request = Request {
            id,
            round: self.round,
            kind,
        };
        Directive::Ask { to, request }
    }

    /// Resolves a night from its answers: the pack's tally to the pack,
    /// each seer's finding to that seer, then the death or the lack of one.
    fn resolve_night(&mut self, answers: BTreeMap<AgentId, (RequestKind, Move)>) -> Vec<Directive> {
        let mut votes = BTreeMap::new();
        let mut protected = BTreeSet::new();
        let mut findings = Vec::new();
        for (who, (kind, action)) in answers {
            match kind {
                RequestKind::Devour => {
                    votes.insert(who, action);
                }
                RequestKind::Protect => {
                    protected.extend(action.target().cloned());
                    match action.target() {
                        Some(target) => self.last_protected.insert(who, target.clone()),
                        None => self.last_protected.remove(&who),
                    };
                }
                RequestKind::Investigate => {
                    if let Some(target) = action.target() {
                        findings.push(Directive::Narrate {
                            to: [who].into(),
                            narration: Narration::Investigated {
                                target: target.clone(),
                                faction: self.role(target).faction(),
                            },
                        });
                    }
                }
                RequestKind::Nominate => unreachable!("nobody nominates at night"),
            }
        }
        let victim = plurality(votes.values(), &mut self.ties)
            .expect("every living werewolf devours someone");
        let mut directives = vec![Directive::Narrate {
            to: votes.keys().cloned().collect(),
            narration: Narration::Tally {
                round: self.round,
                phase: Phase::Night,
                votes,
            },
        }];
        directives.extend(findings);
        if protected.contains(&victim) {
            directives.push(self.narrate_living(Narration::NoDeath { round: self.round }));
            directives.extend(self.next_phase());
        } else {
            directives.push(self.eliminate(&victim, Cause::Devoured));
            directives.extend(self.advance());
        }
        directives
    }

    /// Resolves a day from its nominations: the full tally to the living,
    /// then the lynching.
    fn resolve_day(&mut self, answers: BTreeMap<AgentId, (RequestKind, Move)>) -> Vec<Directive> {
        // The lynching below is what makes a game end, and it is certain
        // only because a `Nominate` cannot abstain.
        let () = NOMINATE_DECIDES;
        let votes: BTreeMap<AgentId, Move> = answers
            .into_iter()
            .map(|(who, chosen)| (who, chosen.1))
            .collect();
        let lynched = plurality(votes.values(), &mut self.ties)
            .expect("every living player nominates someone");
        let mut directives = vec![self.narrate_living(Narration::Tally {
            round: self.round,
            phase: Phase::Day,
            votes,
        })];
        directives.push(self.eliminate(&lynched, Cause::Lynched));
        directives.extend(self.advance());
        directives
    }

    /// Removes a player from the living and announces it, with the role
    /// revealed, to the living and to the player itself.
    fn eliminate(&mut self, who: &AgentId, cause: Cause) -> Directive {
        let to = self.living.clone();
        assert!(self.living.remove(who), "{who} is not living");
        Directive::Narrate {
            to,
            narration: Narration::Eliminated {
                who: who.clone(),
                role: self.role(who),
                round: self.round,
                cause,
            },
        }
    }

    /// After an elimination: the outcome if a side has won, and otherwise
    /// the next phase.
    fn advance(&mut self) -> Vec<Directive> {
        match self.winner() {
            Some(winner) => vec![self.end(winner)],
            None => self.next_phase(),
        }
    }

    /// Begins the phase after this one: the day of the same round, or the
    /// night of the next.
    fn next_phase(&mut self) -> Vec<Directive> {
        match self.phase {
            Phase::Night => self.phase = Phase::Day,
            Phase::Day => {
                self.round = Round(self.round.0 + 1);
                self.phase = Phase::Night;
            }
        }
        self.begin_phase()
    }

    /// The side that has won, if one has: the village once no werewolf
    /// lives, the werewolves once they are at least as many as everyone
    /// else.
    fn winner(&self) -> Option<Faction> {
        let werewolves = self
            .living
            .iter()
            .filter(|who| self.role(who).faction() == Faction::Werewolves)
            .count();
        let others = self.living.len() - werewolves;
        if werewolves == 0 {
            Some(Faction::Village)
        } else if werewolves >= others {
            Some(Faction::Werewolves)
        } else {
            None
        }
    }

    /// Ends the game and announces how, to the living.
    ///
    /// The outcome is narrated like any other narration. It was once
    /// broadcast to everyone, living and dead, because it was a dead
    /// player's terminal reward signal; a reward is now logged rather than
    /// said (ADR-0007), so the exception is withdrawn and no message of
    /// this game goes to a player after the announcement of its own death.
    fn end(&mut self, winner: Faction) -> Directive {
        let outcome = Outcome {
            winner,
            rounds: self.round,
            living: self.living.clone(),
        };
        self.outcome = Some(outcome.clone());
        self.narrate_living(Narration::Outcome(outcome))
    }

    fn narrate_living(&self, narration: Narration) -> Directive {
        Directive::Narrate {
            to: self.living.clone(),
            narration,
        }
    }

    /// The role of a player.
    fn role(&self, who: &AgentId) -> Role {
        self.assignment
            .role(who)
            .unwrap_or_else(|| panic!("{who} is not a player"))
    }
}

/// The most-targeted player among some actions, ignoring abstentions, or
/// `None` if nothing was targeted.
///
/// A tie is broken by [`pick`]ing among the tied players, in agent order,
/// from `ties`, which is touched only when there is a tie.
fn plurality<'a>(
    actions: impl IntoIterator<Item = &'a Move>,
    ties: &mut ChaCha8Rng,
) -> Option<AgentId> {
    let mut counts: BTreeMap<&AgentId, usize> = BTreeMap::new();
    for who in actions.into_iter().filter_map(Move::target) {
        *counts.entry(who).or_default() += 1;
    }
    let most = *counts.values().max()?;
    let leaders: Vec<&AgentId> = counts
        .iter()
        .filter(|(_, count)| **count == most)
        .map(|(who, _)| *who)
        .collect();
    Some((*pick(ties, &leaders)).clone())
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::*;
    use crate::testing::{id, ids, target};
    use crate::werewolf::role::Role::{Doctor, Seer, Villager, Werewolf};

    const SEED: u64 = 20_260_918;

    /// One phase of a script: every request's answer, keyed by the agent
    /// asked, in the order the answers are to be recorded.
    type Answers = Vec<(&'static str, Move)>;

    fn answers(pairs: &[(&'static str, &str)]) -> Answers {
        pairs
            .iter()
            .map(|(who, whom)| {
                let chosen = if *whom == "-" {
                    Move::Abstain
                } else {
                    target(whom)
                };
                (*who, chosen)
            })
            .collect()
    }

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

    /// Seven players and three werewolves, alice, bob and carol, with no
    /// seer and no doctor.
    fn pack_of_three() -> Assignment {
        Assignment::new([
            ("alice", Werewolf),
            ("bob", Werewolf),
            ("carol", Werewolf),
            ("dave", Villager),
            ("erin", Villager),
            ("frank", Villager),
            ("grace", Villager),
        ])
    }

    fn game(assignment: Assignment) -> Game {
        Game::new(assignment, SEED)
    }

    /// The requests among some directives, by the agent asked.
    fn asks(directives: &[Directive]) -> BTreeMap<AgentId, Request> {
        directives
            .iter()
            .filter_map(|directive| match directive {
                Directive::Ask { to, request } => Some((to.clone(), request.clone())),
                Directive::Narrate { .. } => None,
            })
            .collect()
    }

    /// Records one phase's answers in the order given, asserting that
    /// nothing comes back until the last, and returns what the last caused.
    fn answer(game: &mut Game, asked: &[Directive], answers: &Answers) -> Vec<Directive> {
        let asks = asks(asked);
        assert_eq!(
            asks.keys().cloned().collect::<BTreeSet<_>>(),
            answers.iter().map(|(who, _)| id(who)).collect(),
            "a script answers exactly the requests issued"
        );
        let mut caused = Vec::new();
        for (index, (who, chosen)) in answers.iter().enumerate() {
            let who = id(who);
            let response = Response {
                request: asks[&who].id,
                chosen: chosen.clone(),
            };
            caused = game.record(&who, &response);
            if index + 1 < answers.len() {
                assert!(
                    caused.is_empty(),
                    "{who}'s response caused {caused:?} with requests still outstanding"
                );
            }
        }
        caused
    }

    /// Plays a script through a game and returns every directive it
    /// produced, in order, from the opening to the last phase scripted.
    fn play(game: &mut Game, script: &[Answers]) -> Vec<Directive> {
        let mut all = game.begin();
        let mut latest = all.clone();
        for phase in script {
            latest = answer(game, &latest, phase);
            all.extend(latest.iter().cloned());
        }
        all
    }

    fn narrate<const N: usize>(to: [&str; N], narration: Narration) -> Directive {
        Directive::Narrate {
            to: ids(to),
            narration,
        }
    }

    fn assigned<const N: usize>(to: &str, role: Role, pack: [&str; N]) -> Directive {
        narrate(
            [to],
            Narration::Assigned {
                role,
                pack: ids(pack),
            },
        )
    }

    fn phase_began<const N: usize>(round: u32, phase: Phase, living: [&str; N]) -> Directive {
        narrate(
            living,
            Narration::PhaseBegan {
                round: Round(round),
                phase,
                living: ids(living),
            },
        )
    }

    fn ask(to: &str, id: u64, round: u32, kind: RequestKind) -> Directive {
        Directive::Ask {
            to: AgentId::new(to),
            request: Request {
                id: RequestId(id),
                round: Round(round),
                kind,
            },
        }
    }

    fn tally<const N: usize>(
        to: [&str; N],
        round: u32,
        phase: Phase,
        votes: &[(&'static str, &str)],
    ) -> Directive {
        narrate(
            to,
            Narration::Tally {
                round: Round(round),
                phase,
                votes: answers(votes)
                    .into_iter()
                    .map(|(who, action)| (id(who), action))
                    .collect(),
            },
        )
    }

    fn eliminated<const N: usize>(
        to: [&str; N],
        who: &str,
        role: Role,
        round: u32,
        cause: Cause,
    ) -> Directive {
        narrate(
            to,
            Narration::Eliminated {
                who: id(who),
                role,
                round: Round(round),
                cause,
            },
        )
    }

    /// The outcome as it is announced: to the living, who are exactly the
    /// survivors it names.
    fn outcome<const N: usize>(winner: Faction, rounds: u32, living: [&str; N]) -> Directive {
        narrate(
            living,
            Narration::Outcome(Outcome {
                winner,
                rounds: Round(rounds),
                living: ids(living),
            }),
        )
    }

    fn investigated(to: &str, target: &str, faction: Faction) -> Directive {
        narrate(
            [to],
            Narration::Investigated {
                target: id(target),
                faction,
            },
        )
    }

    /// Records one response to the request with the given id.
    fn respond(game: &mut Game, from: &str, request: u64, chosen: Move) -> Vec<Directive> {
        let response = Response {
            request: RequestId(request),
            chosen,
        };
        game.record(&id(from), &response)
    }

    /// In the village, alice is devoured and then bob, the only werewolf,
    /// is lynched: a village win in one round.
    fn village_wins() -> Vec<Answers> {
        vec![
            answers(&[("bob", "alice"), ("carol", "bob"), ("dave", "erin")]),
            answers(&[
                ("bob", "carol"),
                ("carol", "bob"),
                ("dave", "bob"),
                ("erin", "bob"),
            ]),
        ]
    }

    /// In the village, carol the seer is devoured the night she
    /// investigates, erin is lynched, and alice is devoured while the
    /// doctor abstains: a werewolf win at parity in round two.
    fn werewolves_win() -> Vec<Answers> {
        vec![
            answers(&[("bob", "carol"), ("carol", "bob"), ("dave", "alice")]),
            answers(&[
                ("alice", "erin"),
                ("bob", "erin"),
                ("dave", "erin"),
                ("erin", "alice"),
            ]),
            answers(&[("bob", "alice"), ("dave", "-")]),
        ]
    }

    /// In the town, the werewolves split between alice and erin on the
    /// first night, so the generator picks the victim.
    fn split_pack() -> Vec<Answers> {
        vec![answers(&[
            ("bob", "alice"),
            ("carol", "grace"),
            ("dave", "carol"),
            ("frank", "erin"),
        ])]
    }

    /// In the village, the doctor protects the victim and the seer abstains:
    /// nobody dies.
    fn saved() -> Vec<Answers> {
        vec![answers(&[
            ("bob", "erin"),
            ("carol", "-"),
            ("dave", "erin"),
        ])]
    }

    /// Every scripted game, with the assignment it is played over.
    fn scripted_games() -> Vec<(Assignment, Vec<Answers>)> {
        vec![
            (village(), village_wins()),
            (village(), werewolves_win()),
            (town(), split_pack()),
            (village(), saved()),
        ]
    }

    /// Every scripted game played out: its assignment, the game as it ends,
    /// and every directive it produced.
    fn played_games() -> Vec<(Assignment, Game, Vec<Directive>)> {
        scripted_games()
            .into_iter()
            .map(|(assignment, script)| {
                let mut game = Game::new(assignment.clone(), SEED);
                let directives = play(&mut game, &script);
                (assignment, game, directives)
            })
            .collect()
    }

    #[test]
    fn the_village_wins_when_the_last_werewolf_is_lynched() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let survivors = ["bob", "carol", "dave", "erin"];
        let mut game = game(village());
        assert_eq!(
            play(&mut game, &village_wins()),
            [
                assigned("alice", Villager, []),
                assigned("bob", Werewolf, ["bob"]),
                assigned("carol", Seer, []),
                assigned("dave", Doctor, []),
                assigned("erin", Villager, []),
                phase_began(1, Phase::Night, everyone),
                ask("bob", 1, 1, RequestKind::Devour),
                ask("carol", 2, 1, RequestKind::Investigate),
                ask("dave", 3, 1, RequestKind::Protect),
                tally(["bob"], 1, Phase::Night, &[("bob", "alice")]),
                investigated("carol", "bob", Faction::Werewolves),
                eliminated(everyone, "alice", Villager, 1, Cause::Devoured),
                phase_began(1, Phase::Day, survivors),
                ask("bob", 4, 1, RequestKind::Nominate),
                ask("carol", 5, 1, RequestKind::Nominate),
                ask("dave", 6, 1, RequestKind::Nominate),
                ask("erin", 7, 1, RequestKind::Nominate),
                tally(
                    survivors,
                    1,
                    Phase::Day,
                    &[
                        ("bob", "carol"),
                        ("carol", "bob"),
                        ("dave", "bob"),
                        ("erin", "bob")
                    ],
                ),
                eliminated(survivors, "bob", Werewolf, 1, Cause::Lynched),
                outcome(Faction::Village, 1, ["carol", "dave", "erin"]),
            ]
        );
        assert_eq!(
            game.outcome(),
            Some(&Outcome {
                winner: Faction::Village,
                rounds: Round(1),
                living: ids(["carol", "dave", "erin"]),
            })
        );
        assert_eq!(*game.living(), ids(["carol", "dave", "erin"]));
    }

    #[test]
    fn the_werewolves_win_at_parity() {
        let mut game = game(village());
        let directives = play(&mut game, &werewolves_win());
        assert_eq!(
            directives[9..],
            [
                tally(["bob"], 1, Phase::Night, &[("bob", "carol")]),
                // The seer is devoured tonight and still learns what it learned.
                investigated("carol", "bob", Faction::Werewolves),
                eliminated(
                    ["alice", "bob", "carol", "dave", "erin"],
                    "carol",
                    Seer,
                    1,
                    Cause::Devoured,
                ),
                phase_began(1, Phase::Day, ["alice", "bob", "dave", "erin"]),
                ask("alice", 4, 1, RequestKind::Nominate),
                ask("bob", 5, 1, RequestKind::Nominate),
                ask("dave", 6, 1, RequestKind::Nominate),
                ask("erin", 7, 1, RequestKind::Nominate),
                tally(
                    ["alice", "bob", "dave", "erin"],
                    1,
                    Phase::Day,
                    &[
                        ("alice", "erin"),
                        ("bob", "erin"),
                        ("dave", "erin"),
                        ("erin", "alice")
                    ],
                ),
                eliminated(
                    ["alice", "bob", "dave", "erin"],
                    "erin",
                    Villager,
                    1,
                    Cause::Lynched,
                ),
                // No seer lives, so the second night asks nothing of one.
                phase_began(2, Phase::Night, ["alice", "bob", "dave"]),
                ask("bob", 8, 2, RequestKind::Devour),
                ask("dave", 9, 2, RequestKind::Protect),
                tally(["bob"], 2, Phase::Night, &[("bob", "alice")]),
                eliminated(
                    ["alice", "bob", "dave"],
                    "alice",
                    Villager,
                    2,
                    Cause::Devoured
                ),
                outcome(Faction::Werewolves, 2, ["bob", "dave"]),
            ]
        );
        assert_eq!(
            game.outcome().map(|outcome| outcome.winner),
            Some(Faction::Werewolves)
        );
    }

    /// Plays a game to its end, answering every request with a move drawn
    /// from the action space the rules compute, and returns the outcome
    /// and how many were living at the start of each round.
    ///
    /// The answers are arbitrary, so nothing but the rules keeps the game
    /// finite: this is the termination guarantee under adversity.
    fn play_out(assignment: Assignment, seed: u64) -> (Outcome, Vec<usize>) {
        let mut game = Game::new(assignment, seed);
        let mut moves = ChaCha8Rng::seed_from_u64(seed);
        let mut latest = game.begin();
        let mut living = vec![game.living.len()];
        let mut round = game.round;
        while game.outcome().is_none() {
            if game.round != round {
                round = game.round;
                living.push(game.living.len());
            }
            let asked = asks(&latest);
            assert!(!asked.is_empty(), "a running game always asks something");
            let mut caused = Vec::new();
            for (who, request) in asked {
                let space = roles::action_space(
                    &who,
                    &game.living,
                    request.kind,
                    game.last_protected.get(&who),
                );
                let response = Response {
                    request: request.id,
                    chosen: pick(&mut moves, &space).clone(),
                };
                caused = game.record(&who, &response);
            }
            latest = caused;
        }
        (game.outcome().unwrap().clone(), living)
    }

    #[test]
    fn a_game_ends_within_as_many_rounds_as_there_are_players() {
        // A `Nominate` cannot abstain, so every day lynches someone and the
        // living set strictly shrinks each round. A game of n players is
        // therefore over by round n however its players answer, which is
        // what makes a round cap unnecessary.
        for assignment in [village(), town(), pack_of_three()] {
            let players = assignment.players().count();
            for seed in 0..200 {
                let (outcome, living) = play_out(assignment.clone(), seed);
                assert!(
                    outcome.rounds.0 as usize <= players,
                    "{players} players, seed {seed}: {outcome:?}"
                );
                // The bound above is loose; this is the reason it holds.
                assert!(
                    living.windows(2).all(|pair| pair[1] < pair[0]),
                    "the living set shrinks every round, seed {seed}: {living:?}"
                );
            }
        }
    }

    #[test]
    fn nothing_is_returned_until_the_last_outstanding_request_is_answered() {
        let mut game = game(village());
        let asks = asks(&game.begin());
        let request = |who: &str| asks[&id(who)].id.0;
        assert_eq!(
            respond(&mut game, "bob", request("bob"), target("alice")),
            []
        );
        assert_eq!(
            respond(&mut game, "carol", request("carol"), target("bob")),
            []
        );
        let caused = respond(&mut game, "dave", request("dave"), target("erin"));
        assert_eq!(caused.len(), 8, "{caused:?}");
        assert_eq!(
            caused[0],
            tally(["bob"], 1, Phase::Night, &[("bob", "alice")])
        );
    }

    #[test]
    fn responses_in_any_order_produce_the_same_directives() {
        let reference = play(&mut game(village()), &village_wins());
        let reorderings = [
            vec![
                answers(&[("dave", "erin"), ("carol", "bob"), ("bob", "alice")]),
                answers(&[
                    ("erin", "bob"),
                    ("dave", "bob"),
                    ("carol", "bob"),
                    ("bob", "carol"),
                ]),
            ],
            vec![
                answers(&[("carol", "bob"), ("bob", "alice"), ("dave", "erin")]),
                answers(&[
                    ("dave", "bob"),
                    ("erin", "bob"),
                    ("bob", "carol"),
                    ("carol", "bob"),
                ]),
            ],
        ];
        for reordered in &reorderings {
            assert_eq!(play(&mut game(village()), reordered), reference);
        }
    }

    #[test]
    fn the_same_seed_and_responses_produce_the_same_game() {
        for (assignment, script) in scripted_games() {
            let mut first = Game::new(assignment.clone(), SEED);
            let mut second = Game::new(assignment, SEED);
            assert_eq!(play(&mut first, &script), play(&mut second, &script));
            assert_eq!(first.outcome(), second.outcome());
            assert_eq!(first.living(), second.living());
        }
    }

    #[test]
    fn a_three_way_tie_is_broken_by_the_pinned_generator() {
        let everyone = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];
        let split = answers(&[("alice", "dave"), ("bob", "erin"), ("carol", "frank")]);
        let mut game = game(pack_of_three());
        let directives = play(&mut game, &[split]);
        // Golden: dave is the victim the generator picks for this seed.
        assert_eq!(
            directives[11..],
            [
                tally(
                    ["alice", "bob", "carol"],
                    1,
                    Phase::Night,
                    &[("alice", "dave"), ("bob", "erin"), ("carol", "frank")],
                ),
                eliminated(everyone, "dave", Villager, 1, Cause::Devoured),
                outcome(
                    Faction::Werewolves,
                    1,
                    ["alice", "bob", "carol", "erin", "frank", "grace"],
                ),
            ]
        );
    }

    #[test]
    fn the_tie_break_generator_is_seeded_under_its_own_label() {
        // The golden victim above is what a fresh generator under the
        // moderator's label draws, so the label is pinned and not merely
        // the value.
        let mut ties = ChaCha8Rng::seed_from_u64(seed_for(SEED, "moderator:ties"));
        let split = [target("dave"), target("erin"), target("frank")];
        assert_eq!(plurality(&split, &mut ties), Some(id("dave")));
    }

    #[test]
    fn a_tie_is_broken_the_same_way_on_repeated_runs() {
        let runs: Vec<Vec<Directive>> = (0..3)
            .map(|_| play(&mut game(town()), &split_pack()))
            .collect();
        assert_eq!(runs[0], runs[1]);
        assert_eq!(runs[1], runs[2]);
    }

    #[test]
    fn the_generator_is_drawn_from_only_on_a_tie() {
        let fresh = || ChaCha8Rng::seed_from_u64(1);
        let mut untouched = fresh();
        let unanimous = [target("alice"), target("alice"), target("bob")];
        assert_eq!(plurality(&unanimous, &mut untouched), Some(id("alice")));
        assert_eq!(untouched.next_u64(), fresh().next_u64());

        let mut drawn = fresh();
        let tied = [target("alice"), target("bob")];
        let chosen = plurality(&tied, &mut drawn).unwrap();
        assert!(chosen == id("alice") || chosen == id("bob"));
        assert_ne!(drawn.next_u64(), fresh().next_u64());
    }

    #[test]
    fn a_plurality_ignores_abstentions() {
        let mut ties = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(plurality(&[Move::Abstain, Move::Abstain], &mut ties), None);
        assert_eq!(plurality(&[], &mut ties), None);
        assert_eq!(
            plurality(&[Move::Abstain, target("bob"), Move::Abstain], &mut ties),
            Some(id("bob"))
        );
    }

    #[test]
    fn a_save_is_announced_as_no_death_and_nothing_more() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let mut game = game(village());
        let directives = play(&mut game, &saved());
        assert_eq!(
            directives[9..],
            [
                tally(["bob"], 1, Phase::Night, &[("bob", "erin")]),
                narrate(everyone, Narration::NoDeath { round: Round(1) }),
                phase_began(1, Phase::Day, everyone),
                ask("alice", 4, 1, RequestKind::Nominate),
                ask("bob", 5, 1, RequestKind::Nominate),
                ask("carol", 6, 1, RequestKind::Nominate),
                ask("dave", 7, 1, RequestKind::Nominate),
                ask("erin", 8, 1, RequestKind::Nominate),
            ]
        );
        assert_eq!(*game.living(), ids(everyone));
        // The doctor hears nothing the living do not, beyond its own requests.
        for directive in &directives {
            if let Directive::Narrate { to, narration } = directive {
                if to.contains(&id("dave")) && to.len() < everyone.len() {
                    assert!(
                        matches!(narration, Narration::Assigned { .. }),
                        "the doctor alone was told {narration:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn an_investigation_reports_the_targets_faction_for_every_role() {
        // Two seers, so that a seer can be investigated too.
        let assignment = Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Seer),
            ("dave", Doctor),
            ("erin", Seer),
        ]);
        let expected = [
            ("alice", Faction::Village),
            ("bob", Faction::Werewolves),
            ("dave", Faction::Village),
            ("erin", Faction::Village),
        ];
        for (whom, faction) in expected {
            let mut game = Game::new(assignment.clone(), SEED);
            let night = answers(&[
                ("bob", "alice"),
                ("carol", whom),
                ("dave", "bob"),
                ("erin", "-"),
            ]);
            let directives = play(&mut game, &[night]);
            let finding = investigated("carol", whom, faction);
            assert!(directives.contains(&finding), "{whom}: {directives:?}");
        }
    }

    #[test]
    fn no_request_is_issued_for_a_missing_seer_or_doctor() {
        let mut game = game(pack_of_three());
        let asked = asks(&game.begin());
        assert_eq!(
            asked.keys().cloned().collect::<BTreeSet<_>>(),
            ids(["alice", "bob", "carol"])
        );
        assert!(
            asked
                .values()
                .all(|request| request.kind == RequestKind::Devour)
        );
    }

    #[test]
    fn a_dead_doctor_is_asked_nothing() {
        let mut game = game(village());
        let directives = play(
            &mut game,
            &[
                answers(&[("bob", "dave"), ("carol", "alice"), ("dave", "alice")]),
                answers(&[
                    ("alice", "erin"),
                    ("bob", "erin"),
                    ("carol", "erin"),
                    ("erin", "alice"),
                ]),
            ],
        );
        let second_night = directives.len() - 3;
        assert_eq!(
            directives[second_night..],
            [
                phase_began(2, Phase::Night, ["alice", "bob", "carol"]),
                ask("bob", 8, 2, RequestKind::Devour),
                ask("carol", 9, 2, RequestKind::Investigate),
            ]
        );
    }

    #[test]
    fn no_directive_addresses_a_dead_player() {
        for (_, _, directives) in played_games() {
            let mut dead = BTreeSet::new();
            for directive in &directives {
                match directive {
                    Directive::Narrate { to, narration } => {
                        let own_death = match narration {
                            Narration::Eliminated { who, .. } => Some(who),
                            _ => None,
                        };
                        for who in to {
                            assert!(
                                !dead.contains(who) || own_death == Some(who),
                                "{who} is dead but is told {narration:?}"
                            );
                        }
                        if let Some(who) = own_death {
                            dead.insert(who.clone());
                        }
                    }
                    Directive::Ask { to, request } => {
                        assert!(!dead.contains(to), "{to} is dead but is asked {request:?}");
                    }
                }
            }
        }
    }

    #[test]
    fn the_pack_is_named_only_to_werewolves() {
        for (assignment, _, directives) in played_games() {
            for directive in &directives {
                let Directive::Narrate { to, narration } = directive else {
                    continue;
                };
                match narration {
                    Narration::Assigned { pack, .. } => {
                        for who in to {
                            if assignment.role(who) == Some(Werewolf) {
                                assert_eq!(pack, assignment.pack(), "{who}");
                            } else {
                                assert!(pack.is_empty(), "{who} is told the pack {pack:?}");
                            }
                        }
                    }
                    Narration::Tally {
                        phase: Phase::Night,
                        ..
                    } => assert!(to.is_subset(assignment.pack()), "{to:?} hear a night tally"),
                    Narration::Investigated { .. } => {
                        for who in to {
                            assert_eq!(assignment.role(who), Some(Seer), "{who} is told a finding");
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn the_outcome_is_announced_to_the_living_exactly_once_and_last() {
        for (_, game, directives) in played_games() {
            let announcements: Vec<usize> = directives
                .iter()
                .enumerate()
                .filter(|(_, directive)| {
                    matches!(
                        directive,
                        Directive::Narrate {
                            narration: Narration::Outcome(_),
                            ..
                        }
                    )
                })
                .map(|(index, _)| index)
                .collect();
            match game.outcome() {
                Some(outcome) => {
                    assert_eq!(announcements, [directives.len() - 1]);
                    assert_eq!(
                        directives.last(),
                        Some(&Directive::Narrate {
                            to: outcome.living.clone(),
                            narration: Narration::Outcome(outcome.clone()),
                        }),
                        "the outcome goes to exactly the survivors it names"
                    );
                }
                None => assert!(announcements.is_empty(), "{announcements:?}"),
            }
        }
    }

    #[test]
    fn no_narration_has_an_empty_recipient_set() {
        for (_, _, directives) in played_games() {
            for directive in &directives {
                if let Directive::Narrate { to, narration } = directive {
                    assert!(!to.is_empty(), "{narration:?} is addressed to nobody");
                }
            }
        }
    }

    #[test]
    fn the_living_set_strictly_shrinks_every_round() {
        let mut game = game(village());
        let mut latest = game.begin();
        let mut sizes = vec![game.living().len()];
        for (index, phase) in werewolves_win().iter().enumerate() {
            latest = answer(&mut game, &latest, phase);
            if index % 2 == 1 {
                sizes.push(game.living().len());
            }
        }
        assert_eq!(sizes, [5, 3]);
        assert_eq!(game.living().len(), 2);
    }

    #[test]
    fn the_win_condition_at_each_boundary() {
        let mut game = game(town());
        game.living = ids(["bob", "alice", "erin"]);
        assert_eq!(game.winner(), None, "one werewolf and two others continue");
        game.living = ids(["bob", "alice"]);
        assert_eq!(game.winner(), Some(Faction::Werewolves), "parity");
        game.living = ids(["bob", "frank", "alice", "erin"]);
        assert_eq!(
            game.winner(),
            Some(Faction::Werewolves),
            "parity with a pack"
        );
        game.living = ids(["bob", "frank", "alice", "erin", "grace"]);
        assert_eq!(game.winner(), None, "outnumbered werewolves continue");
        game.living = ids(["alice"]);
        assert_eq!(game.winner(), Some(Faction::Village), "no werewolf");
    }

    #[test]
    #[should_panic(expected = "bob answered request 99, which is not outstanding")]
    fn an_unknown_request_id_panics() {
        let mut game = game(village());
        game.begin();
        respond(&mut game, "bob", 99, target("alice"));
    }

    #[test]
    #[should_panic(expected = "carol answered request 1, which was asked of bob")]
    fn a_request_id_asked_of_another_agent_panics() {
        let mut game = game(village());
        game.begin();
        respond(&mut game, "carol", 1, target("alice"));
    }

    #[test]
    #[should_panic(
        expected = "bob abstained from request 1, which is outside its action space for Devour"
    )]
    fn an_abstention_where_none_is_permitted_panics() {
        let mut game = game(village());
        game.begin();
        respond(&mut game, "bob", 1, Move::Abstain);
    }

    #[test]
    #[should_panic(
        expected = "dave targeted dave in request 3, which is outside its action space for Protect"
    )]
    fn a_doctor_protecting_itself_panics() {
        let mut game = game(village());
        game.begin();
        respond(&mut game, "dave", 3, target("dave"));
    }

    /// In the village, alice is devoured while the doctor protects erin,
    /// then carol is lynched, so that the second night asks bob (request
    /// 8) and dave (request 9) again.
    fn doctor_protected_erin() -> Vec<Answers> {
        vec![
            answers(&[("bob", "alice"), ("carol", "bob"), ("dave", "erin")]),
            answers(&[
                ("bob", "carol"),
                ("carol", "bob"),
                ("dave", "carol"),
                ("erin", "carol"),
            ]),
        ]
    }

    #[test]
    #[should_panic(
        expected = "dave targeted erin in request 9, which is outside its action space for Protect"
    )]
    fn a_doctor_protecting_the_same_player_two_nights_running_panics() {
        let mut game = game(village());
        play(&mut game, &doctor_protected_erin());
        respond(&mut game, "dave", 9, target("erin"));
    }

    /// Nine players and two werewolves, bob and frank; carol is the seer
    /// and dave the doctor. Big enough for three nights.
    fn hamlet() -> Assignment {
        Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Seer),
            ("dave", Doctor),
            ("erin", Villager),
            ("frank", Werewolf),
            ("grace", Villager),
            ("heidi", Villager),
            ("ivan", Villager),
        ])
    }

    #[test]
    fn the_doctor_may_return_to_a_player_after_a_night_off() {
        // In the hamlet, dave protects erin on the first night, then on
        // the second either abstains or protects carol, and on the third
        // may protect erin again: the constraint is last night's protection
        // alone. The requests of the third night are 23 to 26, and dave's
        // is 25.
        for second_night in ["-", "carol"] {
            let mut game = game(hamlet());
            play(
                &mut game,
                &[
                    answers(&[
                        ("bob", "grace"),
                        ("carol", "bob"),
                        ("dave", "erin"),
                        ("frank", "grace"),
                    ]),
                    answers(&[
                        ("alice", "ivan"),
                        ("bob", "alice"),
                        ("carol", "alice"),
                        ("dave", "alice"),
                        ("erin", "alice"),
                        ("frank", "alice"),
                        ("heidi", "alice"),
                        ("ivan", "alice"),
                    ]),
                    answers(&[
                        ("bob", "heidi"),
                        ("carol", "frank"),
                        ("dave", second_night),
                        ("frank", "heidi"),
                    ]),
                    answers(&[
                        ("bob", "ivan"),
                        ("carol", "ivan"),
                        ("dave", "ivan"),
                        ("erin", "ivan"),
                        ("frank", "ivan"),
                        ("ivan", "erin"),
                    ]),
                ],
            );
            assert_eq!(
                *game.living(),
                ids(["bob", "carol", "dave", "erin", "frank"])
            );
            assert_eq!(game.outstanding.len(), 4, "the third night is under way");
            let permitted = roles::action_space(
                &id("dave"),
                game.living(),
                RequestKind::Protect,
                game.last_protected.get(&id("dave")),
            );
            assert!(
                permitted.contains(&target("erin")),
                "{second_night}: {permitted:?}"
            );
            respond(&mut game, "dave", 25, target("erin"));
        }
    }

    #[test]
    #[should_panic(
        expected = "bob targeted alice in request 4, which is outside its action space for Nominate"
    )]
    fn targeting_a_dead_player_panics() {
        let mut game = game(village());
        play(
            &mut game,
            &[answers(&[
                ("bob", "alice"),
                ("carol", "bob"),
                ("dave", "erin"),
            ])],
        );
        respond(&mut game, "bob", 4, target("alice"));
    }

    #[test]
    #[should_panic(expected = "carol answered request 5 after the game ended")]
    fn a_response_after_the_outcome_panics() {
        let mut game = game(village());
        play(&mut game, &village_wins());
        assert!(game.outcome().is_some());
        respond(&mut game, "carol", 5, target("dave"));
    }

    #[test]
    #[should_panic(expected = "the game has already begun")]
    fn beginning_twice_panics() {
        let mut game = game(village());
        game.begin();
        game.begin();
    }

    #[test]
    #[should_panic(expected = "a game needs at least one werewolf")]
    fn a_game_without_a_werewolf_panics() {
        game(Assignment::new([
            ("alice", Villager),
            ("bob", Seer),
            ("carol", Doctor),
        ]));
    }

    #[test]
    #[should_panic(expected = "a game needs more other players than werewolves")]
    fn a_game_already_at_parity_panics() {
        game(Assignment::new([
            ("alice", Villager),
            ("bob", Werewolf),
            ("carol", Werewolf),
            ("dave", Villager),
        ]));
    }
}
