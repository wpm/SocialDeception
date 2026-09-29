//! The state a player carries between cycles: [`Knowledge`], a fold over the
//! observations it has received.
//!
//! In the vocabulary of ADR-0007, `Knowledge` is the *state*: a sufficient
//! statistic of an agent's [`Observation`] history, and what a strategy
//! conditions on. It is one type for every role, because every role needs
//! the same public picture (the round and phase, who is living, who is dead
//! and what they turned out to be, how the phases before this one selected)
//! and differs only in
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
//! nobody died may conclude it saved them; the conclusion is the
//! strategy's to draw, and this type records only that nobody died.
//! Player-to-player dialogue, which may be false, and a role whose
//! investigations can be wrong would each call for a separate type holding
//! what a player believes. The line between that type and this one is
//! drawn here, so that it can be added without touching this one.
//!
//! The one thing here that the moderator never said is what the agent itself
//! did in secret. What a player has done is still knowledge, and it is true
//! for the same reason: the player was there. Its nominations and devours
//! it can watch land, since the moderator forwards each accepted selection to
//! the players who should see it and a member of a session watches itself
//! converge with the rest; an investigation comes back as its result. The
//! doctor's protection is the exception, announced to nobody, and the rules
//! need it the next night, so [`Knowledge::acted`] folds it in beside the
//! observations.
//!
//! Nothing here is a summary the moderator sent. The history of the phases
//! that have finished is built here, out of the selections this agent
//! observed and the selections it made itself, because an environment does
//! not tell an agent what that agent has already seen (ADR-0015).
//!
//! # Purity
//!
//! A `Knowledge` is a pure function of the observations folded into it. The
//! same stream over a fresh value yields the same state on every run, and
//! nothing else is consulted: no clock, no sender, no recipient list. That is
//! the property a strategy depends on, and what a prompt for a language-model
//! strategy is rendered from.

use std::collections::{BTreeMap, BTreeSet};

use super::message::{Cause, Message, Narration, Outcome, Phase, Round, SessionKind};
use super::role::{Faction, Role};
use crate::Observation;
use crate::clock::Clock;
use crate::message::ActorId;

/// What one player knows: the fold of every observation it has received.
///
/// Every field is exactly what the moderator said, kept current; nothing is
/// derived. See the [module documentation](self) for why that matters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Knowledge {
    /// This agent's own id.
    pub me: ActorId,
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
    pub living: BTreeSet<ActorId>,
    /// Everyone out of the game, with how and when they left and the role
    /// their death revealed.
    pub dead: BTreeMap<ActorId, Death>,
    /// The living werewolves this agent knows of. Empty unless it is one.
    pub pack: BTreeSet<ActorId>,
    /// What the seer has learned, by target. Empty unless it is the seer.
    pub investigations: BTreeMap<ActorId, Faction>,
    /// Whom the doctor protected last night, if anyone: the one player the
    /// rules keep it from protecting again tonight. `None` unless it is the
    /// doctor and it selected somewhere last night.
    pub last_protected: Option<ActorId>,
    /// The latest target of each player whose selection this agent has seen
    /// in the current phase, including its own, cleared when a new phase
    /// begins.
    ///
    /// This is how a pack watches itself converge and how the village sees
    /// its vote form (ADR-0011). It holds only what this agent was actually
    /// addressed, plus what it selected itself: a villager never sees a
    /// `Devour`, and nobody but the moderator sees an `Investigate` or a
    /// `Protect`.
    pub selections: BTreeMap<ActorId, ActorId>,
    /// How each finished phase selected, oldest first: the `selections` of
    /// that phase, archived when the next one began.
    ///
    /// Nobody narrates this. It is the agent's own record of what it
    /// watched happen, kept because a phase's selections are cleared when the
    /// next phase begins and a strategy may still want the argument that
    /// went before (ADR-0015).
    pub history: Vec<Phased>,
    /// Set once the game is over.
    pub outcome: Option<Outcome>,
    /// The episode's origin, from the moment this player was started, or
    /// `None` before it was.
    ///
    /// It is the one origin every actor and the log share (ADR-0017), handed
    /// to every `start` hook by the episode itself, so a player that
    /// measures time measures it on the log's timeline. A scripted player
    /// reads nothing from it; it is here because a language-model player
    /// stamps its prompt with how long the game has been going, and the
    /// place to keep what a player knows is what it knows.
    pub clock: Option<Clock>,
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

/// How one finished phase selected, as this agent saw it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Phased {
    /// The round the phase belonged to.
    pub round: Round,
    /// Which half of the round.
    pub phase: Phase,
    /// The latest target of each player whose selection this agent saw,
    /// including its own. A player whose selections it never saw is absent,
    /// and so is one that never selected.
    ///
    /// Another player's entry is a selection the moderator accepted, since
    /// that is the only kind it forwards. This agent's own entry is the
    /// selection it *sent*, which may have lost its race with the session's
    /// clock: whether a last-second selection counted is the moderator's
    /// bookkeeping and no agent is told it (ADR-0015).
    pub selections: BTreeMap<ActorId, ActorId>,
}

impl Knowledge {
    /// The state of a player that has observed nothing yet.
    #[must_use]
    pub fn new(me: ActorId, role: Role) -> Self {
        Self {
            me,
            role,
            moment: None,
            living: BTreeSet::new(),
            dead: BTreeMap::new(),
            pack: BTreeSet::new(),
            investigations: BTreeMap::new(),
            last_protected: None,
            selections: BTreeMap::new(),
            history: Vec::new(),
            outcome: None,
            clock: None,
        }
    }

    /// Records the episode's origin, which the episode hands every actor's
    /// `start` hook (ADR-0017).
    pub fn started(&mut self, clock: Clock) {
        self.clock = Some(clock);
    }

    /// Folds one of this agent's own moves into the state: the target it
    /// selected in a session of `kind`.
    ///
    /// Total, like [`observe`](Self::observe). An agent's own selection is
    /// part of the phase it belongs to, so it joins
    /// [`selections`](Self::selections) beside the selections of the others
    /// it can see; that is the one entry no forward could supply, since a
    /// selection is never forwarded back to the player that made it. A
    /// `Protect` is also remembered on its own as
    /// [`last_protected`](Self::last_protected), because the rules ask for it
    /// by name the next night.
    pub fn acted(&mut self, kind: SessionKind, chosen: &ActorId) {
        self.selections.insert(self.me.clone(), chosen.clone());
        if kind == SessionKind::Protect {
            self.last_protected = Some(chosen.clone());
        }
    }

    /// Folds one observation into the state.
    ///
    /// Total: every observation has a defined effect, and most have none.
    /// A narration changes the state, and so does a relayed selection, which
    /// under ADR-0011 another player may see. Nothing that can really arrive
    /// is an error, so the state stays a total function of whatever does.
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
    pub fn observe(&mut self, observation: &Observation<Message>) {
        match &observation.message.payload {
            Message::Narration(narration) => self.narrated(narration),
            // Who selected is read from the **envelope**, not from the
            // message's sender: the sender is the moderator, which relayed
            // it, and the envelope is what names the player whose selection
            // it is (ADR-0018). It replaces whatever that player selected
            // before. The session is not checked: a selection this agent was
            // addressed at all is one the rules let it see, and the phase's
            // own `PhaseBegan` is what clears the slate.
            Message::Relayed(envelope) => {
                self.selections
                    .insert(envelope.from.clone(), envelope.payload.target.clone());
            }
            // A selection a player addressed to the moderator. No player is
            // ever among its recipients — a selection reaches another player
            // only as a relay — so nothing here should observe one. The
            // check is a `debug_assert!` for the reason ADR-0018 gives for
            // the player's own: players cooperate with the moderator and are
            // not assumed to cheat, so this catches a wiring mistake rather
            // than enforcing a rule. In release the fold stays total, and it
            // leaves the state alone rather than pretending the sender
            // selected in this agent's hearing.
            Message::Select(selection) => debug_assert!(
                false,
                "{} was sent {}'s {:?} selection directly rather than relayed",
                self.me, observation.message.sender, selection.kind
            ),
            // A reminder is always self-directed, and only the moderator ever
            // sets one in this game (ADR-0016, ADR-0018), so a player can
            // reach this arm only by having been sent something nobody
            // sends. It is a `debug_assert!` for the same reason the arm
            // above is, and the fold stays total.
            Message::Reminder(look) => debug_assert!(
                false,
                "{} was sent a {:?} reminder, which only the moderator sets and only for itself",
                self.me, look
            ),
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
                // Last phase's selections are last phase's; a vote does not
                // carry over into the argument that follows it. They are
                // archived rather than dropped, since nothing else
                // records what the agent watched happen (ADR-0015).
                // Which phase they belonged to is the phase this agent
                // was in, so selections seen before any phase began — there
                // are none in a game, but the fold is total — are
                // cleared without being archived to a phase that never
                // was.
                let finished = std::mem::take(&mut self.selections);
                if let Some((round, phase)) = self.moment {
                    self.history.push(Phased {
                        round,
                        phase,
                        selections: finished,
                    });
                }
                self.moment = Some((*round, *phase));
                self.living.clone_from(living);
            }
            Narration::Investigated { target, faction } => {
                self.investigations.insert(target.clone(), *faction);
            }
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
            // Still an observation, and still recorded in the log;
            // whether it means a save is for a strategy to infer.
            Narration::NoDeath { .. } | Narration::NoLynch { .. } => {}
            Narration::Outcome(outcome) => self.outcome = Some(outcome.clone()),
        }
    }

    /// Whether `who` is still in the game.
    #[must_use]
    pub fn is_living(&self, who: &ActorId) -> bool {
        self.living.contains(who)
    }

    /// The living players other than this agent: the base of every action
    /// space. Sorted, so that an index into an action space built from it
    /// is a stable action label.
    #[must_use]
    pub fn living_others(&self) -> BTreeSet<ActorId> {
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
    // Two types are called `Message`: this module's payload, which
    // `super::*` brings in, and the runtime message that carries it. The
    // carrier is named more often than the payload here, so it is the one
    // that gets a short name — and not `Envelope`, which is a type of its
    // own and the thing a relay actually carries.
    use crate::Message as Wire;
    use crate::message::Envelope;
    use crate::testing::{ME, from, id, ids, narrated, observed, phase_began, relayed, target};
    use crate::werewolf::message::Select;

    fn votes<const N: usize>(votes: [(&str, ActorId); N]) -> BTreeMap<ActorId, ActorId> {
        votes
            .into_iter()
            .map(|(who, action)| (id(who), action))
            .collect()
    }

    fn assigned(role: Role, pack: BTreeSet<ActorId>) -> Wire<Message> {
        narrated(Narration::Assigned { role, pack })
    }

    fn investigated(target: &str, faction: Faction) -> Wire<Message> {
        narrated(Narration::Investigated {
            target: id(target),
            faction,
        })
    }

    fn eliminated(who: &str, role: Role, round: u32, cause: Cause) -> Wire<Message> {
        narrated(Narration::Eliminated {
            who: id(who),
            role,
            round: Round::new(round),
            cause,
        })
    }

    fn death(round: u32, cause: Cause, role: Role) -> Death {
        Death {
            round: Round::new(round),
            cause,
            role,
        }
    }

    /// A fresh state for this agent with every message folded in, in order.
    fn folded<'a>(role: Role, messages: impl IntoIterator<Item = &'a Wire<Message>>) -> Knowledge {
        let mut knowledge = Knowledge::new(id(ME), role);
        for message in messages {
            knowledge.observe(&observed(message.clone()));
        }
        knowledge
    }

    /// The day-1 selections of [`a_seers_game`], as this seer saw them: the
    /// other living players' nominations, forwarded to it one by one.
    fn day_selections() -> BTreeMap<ActorId, ActorId> {
        votes([
            ("bob", target("carol")),
            ("carol", target("bob")),
            ("wolfgang", target("carol")),
        ])
    }

    /// One player's nomination, relayed by the moderator in the envelope
    /// that names who made it (ADR-0018).
    fn nominated(who: &str, whom: &str, round: u32) -> Wire<Message> {
        relayed(
            who,
            0,
            Select {
                round: Round::new(round),
                kind: SessionKind::Nominate,
                target: id(whom),
                seen_by: BTreeSet::new(),
            },
        )
    }

    /// A seer's whole game, from the deal to the werewolves' win.
    fn a_seers_game() -> Vec<Wire<Message>> {
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
            nominated("bob", "carol", 1),
            nominated("carol", "bob", 1),
            nominated("wolfgang", "carol", 1),
            eliminated("carol", Role::Doctor, 1, Cause::Lynched),
            phase_began(2, Phase::Night, ids(["bob", ME, "wolfgang"])),
            investigated("bob", Faction::Village),
            eliminated("bob", Role::Villager, 2, Cause::Devoured),
            narrated(Narration::Outcome(Outcome {
                winner: Some(Faction::Werewolves),
                rounds: Round::new(2),
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
                moment: Some((Round::new(2), Phase::Night)),
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
                // Cleared by the phase that followed the day it saw, and
                // archived as that day's history.
                selections: BTreeMap::new(),
                history: vec![
                    Phased {
                        round: Round::new(1),
                        phase: Phase::Night,
                        selections: BTreeMap::new(),
                    },
                    Phased {
                        round: Round::new(1),
                        phase: Phase::Day,
                        selections: day_selections(),
                    },
                ],
                outcome: Some(Outcome {
                    winner: Some(Faction::Werewolves),
                    rounds: Round::new(2),
                    living: ids([ME, "wolfgang"]),
                }),
                // The origin is kept by `start`, and this fold never
                // started anybody: it is the fold that is under test here.
                clock: None,
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
        assert_eq!(knowledge.moment, Some((Round::new(1), Phase::Night)));
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
        assert_eq!(knowledge.moment, Some((Round::new(2), Phase::Night)));
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
        let no_ops = [narrated(Narration::NoDeath {
            round: Round::new(2),
        })];
        for message in &no_ops {
            let mut after = knowledge.clone();
            after.observe(&observed(message.clone()));
            assert_eq!(after, knowledge, "{message:?}");
        }
    }

    #[test]
    fn who_selected_is_read_from_the_envelope_and_not_from_the_sender() {
        // The whole point of the envelope (ADR-0018). Every relay is the
        // moderator's own message, so a fold that read the sender would
        // record the moderator as having voted for everybody in turn.
        let mut knowledge = Knowledge::new(id(ME), Role::Villager);
        let relay = relayed(
            "alice",
            4,
            Select {
                round: Round::FIRST,
                kind: SessionKind::Nominate,
                target: target("bob"),
                seen_by: BTreeSet::new(),
            },
        );
        assert_eq!(
            relay.sender,
            id("moderator"),
            "the moderator is what sent it"
        );
        knowledge.observe(&observed(relay));
        assert_eq!(knowledge.selections, votes([("alice", target("bob"))]));
    }

    #[test]
    #[should_panic(expected = "directly rather than relayed")]
    fn a_selection_sent_to_a_player_directly_is_a_wiring_bug() {
        // A player addresses the moderator alone, so no player is ever among
        // a selection's recipients: one arriving here means somebody wired
        // the recipients wrong (ADR-0018). Caught in debug and, in release,
        // left alone rather than recorded as the sender having selected in
        // this agent's hearing.
        let mut knowledge = Knowledge::new(id(ME), Role::Villager);
        knowledge.observe(&observed(from(
            "carol",
            Message::Select(Select {
                round: Round::FIRST,
                kind: SessionKind::Nominate,
                target: target("bob"),
                seen_by: BTreeSet::new(),
            }),
        )));
    }

    #[test]
    fn a_relay_carries_the_players_own_sequence_number() {
        // The number in the envelope is the *player's*, which is what joins
        // the relay back to the player's own action record (ADR-0017). The
        // fold does not read it, so this is a claim about the message the
        // moderator builds rather than about the state.
        let Message::Relayed(envelope) = relayed(
            "alice",
            9,
            Select {
                round: Round::FIRST,
                kind: SessionKind::Nominate,
                target: target("bob"),
                seen_by: BTreeSet::new(),
            },
        )
        .payload
        else {
            panic!("a relay carries an envelope");
        };
        assert_eq!(
            envelope,
            Envelope::new("alice", 9, envelope.payload.clone())
        );
    }

    #[test]
    fn a_protection_is_remembered_until_the_next_one() {
        let mut knowledge = Knowledge::new(id(ME), Role::Doctor);
        assert_eq!(knowledge.last_protected, None);

        knowledge.acted(SessionKind::Protect, &target("alice"));
        assert_eq!(knowledge.last_protected, Some(id("alice")));

        knowledge.acted(SessionKind::Protect, &target("bob"));
        assert_eq!(knowledge.last_protected, Some(id("bob")));
    }

    #[test]
    fn selections_are_folded_in_and_cleared_at_the_next_phase() {
        // A selection another player may see is one this agent conditions on
        // (ADR-0011): the pack watching itself converge, the village
        // watching its vote form.
        let mut knowledge = Knowledge::new(id(ME), Role::Villager);
        let selecting = |who: &str, target: &str| {
            relayed(
                who,
                0,
                Select {
                    round: Round::new(1),
                    kind: SessionKind::Nominate,
                    target: id(target),
                    seen_by: BTreeSet::new(),
                },
            )
        };

        knowledge.observe(&observed(selecting("alice", "bob")));
        knowledge.observe(&observed(selecting("carol", "bob")));
        assert_eq!(
            knowledge.selections,
            votes([("alice", target("bob")), ("carol", target("bob"))])
        );

        // The latest selection of a player replaces whatever it selected
        // before: a member's most recent selection is its vote.
        knowledge.observe(&observed(selecting("alice", "carol")));
        assert_eq!(
            knowledge.selections,
            votes([("alice", target("carol")), ("carol", target("bob"))])
        );

        // The agent's own selection is one of the phase's too, and it is the
        // one entry no forward could supply.
        knowledge.acted(SessionKind::Nominate, &target("alice"));
        assert_eq!(
            knowledge.selections,
            votes([
                ("alice", target("carol")),
                ("carol", target("bob")),
                (ME, target("alice")),
            ])
        );

        // A new phase is a new argument.
        knowledge.observe(&observed(phase_began(
            2,
            Phase::Night,
            ids(["alice", "carol", ME]),
        )));
        assert!(knowledge.selections.is_empty());
    }

    #[test]
    fn an_action_is_this_phase_s_selection_and_only_a_protection_is_more() {
        // Every action is the agent's own latest selection of the phase, and
        // no forward brings it back: a selection is never forwarded to the
        // player that made it. A `Protect` is the one the rules ask for
        // again by name the next night, so it is also remembered on its
        // own.
        let before = folded(Role::Seer, &a_seers_game()[..8]);
        for kind in [
            SessionKind::Nominate,
            SessionKind::Devour,
            SessionKind::Investigate,
        ] {
            let mut after = before.clone();
            after.acted(kind, &target("alice"));
            assert_eq!(
                after.selections.get(&id(ME)),
                Some(&id("alice")),
                "{kind:?}"
            );
            assert_eq!(after.last_protected, None, "{kind:?}");
        }

        let mut protecting = before.clone();
        protecting.acted(SessionKind::Protect, &target("alice"));
        assert_eq!(protecting.selections.get(&id(ME)), Some(&id("alice")));
        assert_eq!(protecting.last_protected, Some(id("alice")));
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
            knowledge.acted(SessionKind::Protect, &target("alice"));
            knowledge.observe(&observed(narrated(Narration::NoDeath {
                round: Round::new(1),
            })));
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
    fn each_phase_is_archived_in_order_and_the_next_starts_empty() {
        // Nobody narrates the history. Each phase's selections are what this
        // agent watched arrive, archived when the next phase begins
        // (ADR-0015), so the record is a fold over observations and the
        // agent's own moves and nothing else.
        let mut knowledge = Knowledge::new(id(ME), Role::Villager);
        let living = ids(["alice", "bob", ME]);

        knowledge.observe(&observed(phase_began(1, Phase::Day, living.clone())));
        knowledge.observe(&observed(nominated("alice", "bob", 1)));
        knowledge.acted(SessionKind::Nominate, &target("alice"));
        assert_eq!(
            knowledge.selections,
            votes([("alice", target("bob")), (ME, target("alice"))]),
            "the agent's own selection belongs to the phase like anybody's"
        );

        knowledge.observe(&observed(phase_began(2, Phase::Night, living.clone())));
        assert!(
            knowledge.selections.is_empty(),
            "a new phase is a new argument"
        );

        knowledge.observe(&observed(phase_began(2, Phase::Day, living)));
        knowledge.observe(&observed(nominated("bob", "alice", 2)));
        knowledge.observe(&observed(narrated(Narration::Outcome(Outcome {
            winner: Some(Faction::Village),
            rounds: Round::new(2),
            living: ids(["alice", ME]),
        }))));

        assert_eq!(
            knowledge.history,
            [
                Phased {
                    round: Round::new(1),
                    phase: Phase::Day,
                    selections: votes([("alice", target("bob")), (ME, target("alice"))]),
                },
                Phased {
                    round: Round::new(2),
                    phase: Phase::Night,
                    selections: BTreeMap::new(),
                },
            ],
            "the phases that finished, oldest first"
        );
        assert_eq!(
            knowledge.selections,
            votes([("bob", target("alice"))]),
            "the phase under way is still current, not history"
        );
    }
}
