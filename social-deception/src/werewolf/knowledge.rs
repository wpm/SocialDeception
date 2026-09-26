//! The state a player carries between cycles: [`Knowledge`], a fold over the
//! observations it has received.
//!
//! In the vocabulary of ADR-0007, `Knowledge` is the *state*: a sufficient
//! statistic of an agent's [`Observation`] history, and what a policy
//! conditions on. It is one type for every role, because every role needs
//! the same public picture (the round and phase, who is living, who is dead
//! and what they turned out to be, the tallies it heard) and differs only in
//! what it holds privately: a werewolf its pack, the seer its
//! investigations. The vocabulary, and the reasons for it, are in ADR-0005
//! and ADR-0007.
//!
//! # A state, not a belief
//!
//! Everything here is true. Every observation a player receives is a
//! statement from the moderator, and the moderator does not lie, so the state
//! is *incomplete* about the hidden world and never *incorrect* about it: a
//! villager does not know who the werewolves are, but nothing it does know is
//! wrong. The name is meant in its ordinary sense, in which knowing something
//! entails its being so.
//!
//! That holds because [`Knowledge::observe`] folds only moderator narration
//! and never infers. A doctor that protected someone and then hears that
//! nobody died may conclude it saved them; the conclusion is the policy's to
//! draw, and this type records only that nobody died. Player-to-player
//! dialogue, which may be false, and a role whose investigations can be wrong
//! would each call for a separate type holding what a player believes. The
//! line between that type and this one is drawn here, so that it can be added
//! without touching this one.
//!
//! The one thing here that the moderator never said is what the agent itself
//! did in secret. What a player has done is still knowledge, and it is true
//! for the same reason: the player was there. Almost all of it reaches the
//! state by narration anyway, since a nomination or a devour comes back in
//! a tally and an investigation comes back as its result. The doctor's
//! protection is the exception, announced to nobody, and the rules need it
//! the next night, so [`Knowledge::acted`] folds it in beside the
//! observations.
//!
//! # Purity
//!
//! A `Knowledge` is a pure function of the observations folded into it. The
//! same stream over a fresh value yields the same state on every run, and
//! nothing else is consulted: no clock, no sender, no recipient list. That is
//! the property a policy depends on, and what a prompt for a language-model
//! policy is rendered from.

use std::collections::{BTreeMap, BTreeSet};

use super::WerewolfDomain;
use super::message::{
    Cause, Message, Move, Narration, Outcome, Phase, Request, RequestKind, Round,
};
use super::role::{Faction, Role};
use crate::agent::Observation;
use crate::event::AgentId;

/// What one player knows: the fold of every observation it has received.
///
/// Every field is exactly what the moderator said, kept current; nothing is
/// derived. See the [module documentation](self) for why that matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Knowledge {
    /// This agent's own id.
    pub me: AgentId,
    /// This agent's role, fixed at construction. The `Assigned` narration
    /// must agree with it.
    pub role: Role,
    /// The current round and phase: `None` until the first phase begins.
    ///
    /// One field rather than two, because the moderator announces the two
    /// together and a state with one but not the other is impossible.
    pub moment: Option<(Round, Phase)>,
    /// Everyone still in the game, as last announced and kept current
    /// between announcements.
    pub living: BTreeSet<AgentId>,
    /// Everyone out of the game, with how and when they left and the role
    /// their death revealed.
    pub dead: BTreeMap<AgentId, Death>,
    /// The living werewolves this agent knows of. Empty unless it is one.
    pub pack: BTreeSet<AgentId>,
    /// What the seer has learned, by target. Empty unless it is the seer.
    pub investigations: BTreeMap<AgentId, Faction>,
    /// Whom the doctor protected last night, if anyone: the one player the
    /// rules keep it from protecting again tonight. `None` unless it is the
    /// doctor and its last protection was not an abstention.
    pub last_protected: Option<AgentId>,
    /// Every tally this agent was told, oldest first.
    pub tallies: Vec<Heard>,
    /// Set once the game is over.
    pub outcome: Option<Outcome>,
}

/// How and when a player left the game, and what they turned out to be.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Death {
    /// The round it happened in.
    pub round: Round,
    /// How it happened.
    pub cause: Cause,
    /// The role the death revealed.
    pub role: Role,
}

/// A tally this agent was told.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Heard {
    /// The round the tally belongs to.
    pub round: Round,
    /// Which half of the round.
    pub phase: Phase,
    /// Each responding player's move.
    pub votes: BTreeMap<AgentId, Move>,
}

impl Knowledge {
    /// The state of a player that has observed nothing yet.
    #[must_use]
    pub fn new(me: AgentId, role: Role) -> Self {
        Self {
            me,
            role,
            moment: None,
            living: BTreeSet::new(),
            dead: BTreeMap::new(),
            pack: BTreeSet::new(),
            investigations: BTreeMap::new(),
            last_protected: None,
            tallies: Vec::new(),
            outcome: None,
        }
    }

    /// Folds one of this agent's own moves into the state: the answer it
    /// gave to `request`.
    ///
    /// Total, like [`observe`](Self::observe), and almost always a no-op,
    /// because the moderator narrates the consequences of nearly every
    /// move back to the agent. The one exception is a `Protect`, which is
    /// announced to nobody: its target is remembered as
    /// [`last_protected`](Self::last_protected), and an abstention clears
    /// it, since there was no protection to repeat.
    pub fn acted(&mut self, request: &Request, chosen: &Move) {
        if request.kind == RequestKind::Protect {
            self.last_protected = chosen.target().cloned();
        }
    }

    /// Folds one observation into the state.
    ///
    /// Total: every observation has a defined effect, and most have none.
    /// Only a narration changes the state. A request tells the agent nothing
    /// the phase's announcement did not, and answering it is the role's job;
    /// a response is what a player sends and never receives. Neither is an
    /// error, so the state stays a total function of whatever arrives.
    ///
    /// Controls do not appear here at all: they are out-of-domain, the
    /// agent loop acts on them, and no handler ever sees one.
    ///
    /// # Panics
    ///
    /// If an [`Narration::Assigned`] names a role other than the one this
    /// value was constructed with. The role the moderator dealt and the
    /// player type that was built disagree, which is a wiring bug, and it
    /// fails at the start of the episode rather than producing a plausible
    /// game.
    pub fn observe(&mut self, observation: &Observation<WerewolfDomain>) {
        if let Message::Narration(narration) = &observation.event.payload {
            self.narrated(narration);
        }
    }

    fn narrated(&mut self, narration: &Narration) {
        match narration {
            Narration::Assigned { role, pack } => {
                assert_eq!(
                    *role, self.role,
                    "{} was constructed as a {:?} but the moderator assigned it {:?}",
                    self.me, self.role, role
                );
                self.pack.clone_from(pack);
            }
            // The authoritative living set; eliminations only keep it right
            // between announcements.
            Narration::PhaseBegan {
                round,
                phase,
                living,
            } => {
                self.moment = Some((*round, *phase));
                self.living.clone_from(living);
            }
            Narration::Investigated { target, faction } => {
                self.investigations.insert(target.clone(), *faction);
            }
            Narration::Tally {
                round,
                phase,
                votes,
            } => self.tallies.push(Heard {
                round: *round,
                phase: *phase,
                votes: votes.clone(),
            }),
            Narration::Eliminated {
                who,
                role,
                round,
                cause,
            } => {
                self.living.remove(who);
                self.pack.remove(who);
                self.dead.insert(
                    who.clone(),
                    Death {
                        round: *round,
                        cause: *cause,
                        role: *role,
                    },
                );
            }
            // Still an observation, and still recorded in the trajectory;
            // whether it means a save is for a policy to infer.
            Narration::NoDeath { .. } => {}
            Narration::Outcome(outcome) => self.outcome = Some(outcome.clone()),
        }
    }

    /// Whether `who` is still in the game.
    #[must_use]
    pub fn is_living(&self, who: &AgentId) -> bool {
        self.living.contains(who)
    }

    /// The living players other than this agent: the base of every action
    /// space. Sorted, so that an index into an action space built from it
    /// is a stable action label.
    #[must_use]
    pub fn living_others(&self) -> BTreeSet<AgentId> {
        self.living
            .iter()
            .filter(|who| **who != self.me)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::Event;
    use crate::testing::{ME, from, id, ids, narrated, observed, phase_began, request, target};
    use crate::werewolf::message::{RequestId, Response};

    fn votes<const N: usize>(votes: [(&str, Move); N]) -> BTreeMap<AgentId, Move> {
        votes
            .into_iter()
            .map(|(who, action)| (id(who), action))
            .collect()
    }

    fn assigned(role: Role, pack: BTreeSet<AgentId>) -> Event<WerewolfDomain> {
        narrated(Narration::Assigned { role, pack })
    }

    fn investigated(target: &str, faction: Faction) -> Event<WerewolfDomain> {
        narrated(Narration::Investigated {
            target: id(target),
            faction,
        })
    }

    fn tally(round: u32, phase: Phase, votes: BTreeMap<AgentId, Move>) -> Event<WerewolfDomain> {
        narrated(Narration::Tally {
            round: Round(round),
            phase,
            votes,
        })
    }

    fn eliminated(who: &str, role: Role, round: u32, cause: Cause) -> Event<WerewolfDomain> {
        narrated(Narration::Eliminated {
            who: id(who),
            role,
            round: Round(round),
            cause,
        })
    }

    fn death(round: u32, cause: Cause, role: Role) -> Death {
        Death {
            round: Round(round),
            cause,
            role,
        }
    }

    /// A fresh state for this agent with every event folded in, in order.
    fn folded<'a>(
        role: Role,
        events: impl IntoIterator<Item = &'a Event<WerewolfDomain>>,
    ) -> Knowledge {
        let mut knowledge = Knowledge::new(id(ME), role);
        for event in events {
            knowledge.observe(&observed(event.clone()));
        }
        knowledge
    }

    /// The one day vote in [`a_seers_game`].
    fn day_votes() -> BTreeMap<AgentId, Move> {
        votes([
            ("bob", target("carol")),
            ("carol", target("bob")),
            (ME, target("wolfgang")),
            ("wolfgang", target("carol")),
        ])
    }

    /// A seer's whole game, from the deal to the werewolves' win.
    fn a_seers_game() -> Vec<Event<WerewolfDomain>> {
        vec![
            assigned(Role::Seer, BTreeSet::new()),
            phase_began(
                1,
                Phase::Night,
                ids(["alice", "bob", "carol", ME, "wolfgang"]),
            ),
            investigated("wolfgang", Faction::Werewolves),
            eliminated("alice", Role::Villager, 1, Cause::Devoured),
            phase_began(1, Phase::Day, ids(["bob", "carol", ME, "wolfgang"])),
            tally(1, Phase::Day, day_votes()),
            eliminated("carol", Role::Doctor, 1, Cause::Lynched),
            phase_began(2, Phase::Night, ids(["bob", ME, "wolfgang"])),
            investigated("bob", Faction::Village),
            eliminated("bob", Role::Villager, 2, Cause::Devoured),
            narrated(Narration::Outcome(Outcome {
                winner: Faction::Werewolves,
                rounds: Round(2),
                living: ids([ME, "wolfgang"]),
            })),
        ]
    }

    #[test]
    fn a_scripted_game_produces_exactly_the_expected_state() {
        let knowledge = folded(Role::Seer, &a_seers_game());
        assert_eq!(
            knowledge,
            Knowledge {
                me: id(ME),
                role: Role::Seer,
                moment: Some((Round(2), Phase::Night)),
                living: ids([ME, "wolfgang"]),
                dead: BTreeMap::from([
                    (id("alice"), death(1, Cause::Devoured, Role::Villager)),
                    (id("carol"), death(1, Cause::Lynched, Role::Doctor)),
                    (id("bob"), death(2, Cause::Devoured, Role::Villager)),
                ]),
                pack: BTreeSet::new(),
                investigations: BTreeMap::from([
                    (id("wolfgang"), Faction::Werewolves),
                    (id("bob"), Faction::Village),
                ]),
                last_protected: None,
                tallies: vec![Heard {
                    round: Round(1),
                    phase: Phase::Day,
                    votes: day_votes(),
                }],
                outcome: Some(Outcome {
                    winner: Faction::Werewolves,
                    rounds: Round(2),
                    living: ids([ME, "wolfgang"]),
                }),
            }
        );
    }

    #[test]
    fn living_follows_announcements_authoritatively_and_eliminations_between_them() {
        let mut knowledge = Knowledge::new(id(ME), Role::Villager);
        assert_eq!(knowledge.moment, None);
        assert!(knowledge.living.is_empty());

        knowledge.observe(&observed(phase_began(
            1,
            Phase::Night,
            ids(["alice", "bob", "carol", ME]),
        )));
        assert_eq!(knowledge.moment, Some((Round(1), Phase::Night)));
        assert_eq!(knowledge.living, ids(["alice", "bob", "carol", ME]));

        knowledge.observe(&observed(eliminated(
            "alice",
            Role::Seer,
            1,
            Cause::Devoured,
        )));
        assert_eq!(knowledge.living, ids(["bob", "carol", ME]));
        assert!(!knowledge.is_living(&id("alice")));
        assert!(knowledge.is_living(&id("bob")));

        // The announcement agrees with the elimination, and changes nothing.
        knowledge.observe(&observed(phase_began(
            1,
            Phase::Day,
            ids(["bob", "carol", ME]),
        )));
        assert_eq!(knowledge.living, ids(["bob", "carol", ME]));

        // The announcement is authoritative even where no elimination
        // preceded it.
        knowledge.observe(&observed(phase_began(2, Phase::Night, ids(["bob", ME]))));
        assert_eq!(knowledge.living, ids(["bob", ME]));
        assert_eq!(knowledge.moment, Some((Round(2), Phase::Night)));
    }

    #[test]
    fn an_eliminated_packmate_leaves_the_pack_and_is_recorded_dead() {
        let mut knowledge = folded(
            Role::Werewolf,
            &[
                assigned(Role::Werewolf, ids([ME, "wanda"])),
                phase_began(1, Phase::Night, ids(["alice", ME, "wanda"])),
            ],
        );
        assert_eq!(knowledge.pack, ids([ME, "wanda"]));

        knowledge.observe(&observed(eliminated(
            "wanda",
            Role::Werewolf,
            1,
            Cause::Lynched,
        )));
        assert_eq!(knowledge.pack, ids([ME]));
        assert_eq!(knowledge.living, ids(["alice", ME]));
        assert_eq!(
            knowledge.dead,
            BTreeMap::from([(id("wanda"), death(1, Cause::Lynched, Role::Werewolf))])
        );
    }

    #[test]
    fn investigations_accumulate_by_target() {
        // The seer may look at the same player twice, and learns the same
        // thing each time; the state records what is known, not how often.
        let knowledge = folded(
            Role::Seer,
            &[
                investigated("alice", Faction::Village),
                investigated("bob", Faction::Werewolves),
                investigated("alice", Faction::Village),
            ],
        );
        assert_eq!(
            knowledge.investigations,
            BTreeMap::from([
                (id("alice"), Faction::Village),
                (id("bob"), Faction::Werewolves),
            ])
        );
    }

    #[test]
    fn observations_with_nothing_to_record_change_nothing() {
        // A control is not among these: the loop acts on controls and a
        // handler, and so this fold, never sees one.
        let knowledge = folded(Role::Seer, &a_seers_game()[..8]);
        let no_ops = [
            from(
                "moderator",
                Message::Request(Request {
                    id: RequestId(3),
                    round: Round(2),
                    kind: RequestKind::Investigate,
                }),
            ),
            from(
                "bob",
                Message::Response(Response {
                    request: RequestId(3),
                    chosen: target("alice"),
                }),
            ),
            narrated(Narration::NoDeath { round: Round(2) }),
        ];
        for event in &no_ops {
            let mut after = knowledge.clone();
            after.observe(&observed(event.clone()));
            assert_eq!(after, knowledge, "{event:?}");
        }
    }

    #[test]
    fn a_protection_is_remembered_until_the_next_one_and_forgotten_on_an_abstain() {
        let mut knowledge = Knowledge::new(id(ME), Role::Doctor);
        let protect = request(RequestKind::Protect);
        assert_eq!(knowledge.last_protected, None);

        knowledge.acted(&protect, &target("alice"));
        assert_eq!(knowledge.last_protected, Some(id("alice")));

        knowledge.acted(&protect, &target("bob"));
        assert_eq!(knowledge.last_protected, Some(id("bob")));

        knowledge.acted(&protect, &Move::Abstain);
        assert_eq!(knowledge.last_protected, None);
    }

    #[test]
    fn only_a_protection_is_remembered() {
        // Every other action comes back to the agent by narration, so the
        // state has nothing to record when it is taken.
        let before = folded(Role::Seer, &a_seers_game()[..8]);
        for kind in [
            RequestKind::Nominate,
            RequestKind::Devour,
            RequestKind::Investigate,
        ] {
            let mut after = before.clone();
            after.acted(&request(kind), &target("alice"));
            assert_eq!(after, before, "{kind:?}");
        }
    }

    #[test]
    fn living_others_is_sorted_and_never_contains_me() {
        let mut knowledge = folded(
            Role::Villager,
            &[phase_began(
                1,
                Phase::Night,
                ids(["carol", ME, "alice", "bob"]),
            )],
        );
        assert_eq!(
            knowledge.living_others().into_iter().collect::<Vec<_>>(),
            [id("alice"), id("bob"), id("carol")]
        );

        knowledge.observe(&observed(phase_began(2, Phase::Night, ids(["carol", ME]))));
        assert_eq!(knowledge.living_others(), ids(["carol"]));

        knowledge.observe(&observed(eliminated(
            "carol",
            Role::Werewolf,
            2,
            Cause::Lynched,
        )));
        assert!(knowledge.is_living(&id(ME)));
        assert!(knowledge.living_others().is_empty());
    }

    #[test]
    #[should_panic(
        expected = "me was constructed as a Villager but the moderator assigned it Seer"
    )]
    fn an_assignment_that_contradicts_the_type_panics() {
        folded(Role::Villager, &[assigned(Role::Seer, BTreeSet::new())]);
    }

    #[test]
    fn the_same_stream_yields_the_same_state() {
        let game = a_seers_game();
        assert_eq!(folded(Role::Seer, &game), folded(Role::Seer, &game));
    }

    #[test]
    fn what_the_agent_did_is_part_of_the_same_pure_fold() {
        let night = |knowledge: &mut Knowledge| {
            knowledge.observe(&observed(phase_began(
                1,
                Phase::Night,
                ids(["alice", "bob", ME]),
            )));
            knowledge.acted(&request(RequestKind::Protect), &target("alice"));
            knowledge.observe(&observed(narrated(Narration::NoDeath { round: Round(1) })));
        };
        let mut first = Knowledge::new(id(ME), Role::Doctor);
        let mut second = Knowledge::new(id(ME), Role::Doctor);
        night(&mut first);
        night(&mut second);
        assert_eq!(first, second);
        assert_eq!(first.last_protected, Some(id("alice")));
    }

    #[test]
    fn independent_observations_fold_in_any_order() {
        let opening = [
            assigned(Role::Seer, BTreeSet::new()),
            phase_began(1, Phase::Night, ids(["alice", "bob", "carol", "dave", ME])),
        ];
        let independent = [
            investigated("alice", Faction::Village),
            investigated("bob", Faction::Werewolves),
            eliminated("carol", Role::Villager, 1, Cause::Devoured),
            eliminated("dave", Role::Doctor, 1, Cause::Lynched),
        ];
        assert_eq!(
            folded(Role::Seer, opening.iter().chain(&independent)),
            folded(Role::Seer, opening.iter().chain(independent.iter().rev()))
        );
    }

    #[test]
    fn tallies_keep_the_order_they_were_heard_in() {
        let first = tally(1, Phase::Day, votes([("alice", target("bob"))]));
        let second = tally(2, Phase::Day, votes([("bob", target("alice"))]));
        let heard = |round, who: &str, whom: &str| Heard {
            round: Round(round),
            phase: Phase::Day,
            votes: votes([(who, target(whom))]),
        };

        assert_eq!(
            folded(Role::Villager, [&first, &second]).tallies,
            [heard(1, "alice", "bob"), heard(2, "bob", "alice")]
        );
        assert_eq!(
            folded(Role::Villager, [&second, &first]).tallies,
            [heard(2, "bob", "alice"), heard(1, "alice", "bob")]
        );
    }
}
