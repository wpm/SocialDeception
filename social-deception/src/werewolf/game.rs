//! The rules of Werewolf as a pure state machine.
//!
//! [`Game`] takes players' selections and the passing of time in, and
//! produces [`Directive`]s out: what to say, to whom, in what order. It
//! touches no channel, spawns no thread and reads no clock — every instant it
//! knows is an argument — so the whole of the rules is testable by calling
//! functions with a scripted sequence of selections and instants. The
//! moderator that wraps it is plumbing: it folds the messages it receives
//! into [`Game::select`], wakes on [`Game::next_deadline`] to call
//! [`Game::expire`], and turns the directives into messages.
//!
//! # A phase is made of sessions
//!
//! [`Game::begin`] tells each player its role, and a werewolf its pack,
//! then opens the first night. A phase opens one session per kind it calls
//! for, and announces that it has begun.
//!
//! **Nobody is asked to act.** A player observes the phase and works out
//! from its own role whether the rules ask anything of it (ADR-0014), so
//! the announcement is the whole of what the moderator says. What the
//! moderator keeps is each session's membership, because that is what
//! decides when a session closes, and it checks a selection against the rules
//! when the selection arrives. A player the rules leave nothing to select is
//! not a member of anything.
//!
//! A member may select whenever it likes while its session is open and may
//! change its mind; its most recent selection is its vote, and selecting
//! nowhere is how it abstains (ADR-0011).
//!
//! **A night is three sessions at once**, each with a clock of its own so
//! that a slow role cannot spend another's time: the pack devours, the seer
//! investigates, the doctor protects. Each closes when every member has
//! selected and no selection has changed for its quiet period — any change
//! restarts it — or at its hard limit, whichever comes first. The seer is
//! told its finding when its own session closes, so a seer devoured that
//! same night still learns what it learned.
//!
//! **A session closes silently.** Its members are told nothing of its close,
//! because there is nothing left to tell them: each selection the moderator
//! accepted was forwarded as it arrived, so a member has already watched the
//! session converge selection by selection, and a summary of where it landed
//! would only repeat what that member observed (ADR-0015). What the moderator
//! needs to remember about a session it keeps — it is the moderator that
//! drops a selection arriving after the close — and a reader of the
//! trajectory reconstructs a session from the selections, which are the
//! primary record. What a member learns is the session's *outcome*, and that
//! is announced anyway: the death, the finding, or the night that passed
//! quietly.
//!
//! The night resolves once every session has closed: the victim is the
//! plurality of the pack's latest selections, and no wolf selecting means
//! nobody is devoured; the doctor's protection then applies, and either the
//! death is revealed with its role, or nobody died. A save is never
//! announced as a save, so the village cannot tell one from a pack that
//! selected nowhere.
//!
//! **A day is one session**, and its selections are public. It closes the
//! moment a majority of the *living* — more than half of them, not of those
//! who have selected — select the same player: that player is lynched, and
//! the selection that made the majority is the *hammer*. If the hard limit
//! passes first, nobody is lynched.
//!
//! A selection for a session that has closed is ignored. It lost a race with
//! the clock, which is a fact about timing rather than a bug.
//!
//! The win condition is checked after every elimination, and only then: the
//! village wins when no werewolf lives, and the werewolves win when they are
//! at least as many as everyone else, since from parity onward they cannot
//! lose. The outcome is the one thing said to everyone, living and dead.
//!
//! # What is reproducible, and what is not
//!
//! For a fixed configuration and seed, every phase's *outcome* is the same on
//! every run: each death, each finding, the winner and the rewards. The order
//! selections arrived in is not, and neither is which late selections landed
//! inside a session before it closed. That is the guarantee ADR-0011 makes
//! and [`Transcript::verdicts`](super::Transcript::verdicts) projects a game
//! onto.
//!
//! It holds because a session keeps its members' latest selections in a map
//! read in canonical order, not in arrival order, and because a random player
//! selects once as its session opens and never changes its mind: a night
//! session closes only after all its members have selected, and by day no
//! later selection could have made a different majority. Ties are broken by a
//! generator seeded from the episode's master seed under its own label,
//! independent of every player's, drawn from only on an actual tie — so
//! whether a vote was unanimous has no effect on its state. The seed stays
//! inside the game; no directive carries it.
//!
//! # Termination
//!
//! A day may end without a lynch, so the living set no longer shrinks every
//! round and the rules of play do not bound a game on their own. What
//! bounds it is the **day cap** in the configuration's `[timing]` table,
//! which defaults to the number of players: a game nobody has won by the
//! end of the cap's day ends as a **stalemate**, an [`Outcome`] with no
//! winner. A game of n players therefore ends within its cap however its
//! players act.
//!
//! A stalemate pays −1 to every player, living and dead, exactly as losing
//! does ([`rewards`](Game::rewards)). Stalling can then never beat losing,
//! and a side that is ahead has every reason to finish (ADR-0011).
//!
//! # Player bugs are panics
//!
//! A selection in a session the rules never make its sender a member of, or
//! one naming a target outside that session's action space, is a bug in a
//! player rather than a condition the game can continue from: the game cannot
//! vouch for a state built on it. Each panics with a message naming the agent
//! and the session. A selection that merely arrived too late — for a session
//! that has closed, or a round that has passed — is not among them. The
//! action space a selection is checked against is [`roles::action_space`],
//! the same function the role types compute theirs with, so the game holds
//! every player to exactly the rules the roles apply to themselves, including
//! the doctor's: for that the game remembers whom each doctor protected last
//! night, as the doctor does.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::clock::Timestamp;
use crate::message::ActorId;
use crate::werewolf::assignment::Assignment;
use crate::werewolf::config::Timing;
use crate::werewolf::message::{Cause, Narration, Outcome, Phase, RequestKind, Round, Select};
use crate::werewolf::role::{Faction, Role};
use crate::werewolf::roles;
use crate::werewolf::seed::{TIES, pick, seed_for};

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
        to: BTreeSet<ActorId>,
        /// What they are told.
        narration: Narration,
    },
    /// Pass a player's selection on to the other players who should see it.
    ///
    /// The moderator is the only agent a player addresses, so this is how
    /// a selection reaches anybody else (ADR-0014). The forwarded message
    /// names the player that selected and the instant it selected, so a
    /// recipient cannot tell the selection came by way of the moderator; see
    /// [`Action::relay`](crate::agent::Action::relay).
    ///
    /// Only a selection the game accepted is forwarded. One whose session has
    /// closed is dropped here, which is the whole reason a selection goes
    /// through the moderator at all: it is the only agent that knows
    /// whether the session is still open.
    Forward {
        /// The player whose selection it is.
        from: ActorId,
        /// When that player made it.
        created: Timestamp,
        /// The other players who should see it, never empty.
        to: BTreeSet<ActorId>,
        /// The selection itself, exactly as it was sent.
        selection: Select,
    },
    /// Stop this agent: it is out of the game (ADR-0012).
    ///
    /// A dead player is stopped in the cycle its death is announced, and
    /// it is not among those told: there is nobody left to tell. Its
    /// reward is logged rather than said, so it needs to hear nothing to
    /// be paid.
    Stop {
        /// The agent out of the game.
        who: ActorId,
    },
}

/// One open selection session: who may select, what they were asked, and the
/// clock it closes on (ADR-0011).
///
/// A member may select whenever it likes and as often as it likes; its most
/// recent selection is its vote. The session closes when every member has
/// selected and no selection has changed for its quiet period, or at its hard
/// limit, whichever comes first. The day has no quiet period and closes on
/// a majority instead, so `quiet` is `None` for it.
#[derive(Debug, Clone)]
struct Session {
    /// What its members were asked.
    kind: RequestKind,
    /// Everyone asked, whether or not they have selected.
    members: BTreeSet<ActorId>,
    /// Each member's latest target. A member that has not selected is
    /// absent, which is how it abstains.
    selections: BTreeMap<ActorId, ActorId>,
    /// The quiet period, for a night session; `None` for the day.
    quiet: Option<Duration>,
    /// The instant the session closes however its members behave.
    limit: Timestamp,
    /// When a member's target last changed, which is what the quiet period
    /// is measured from. The session's opening counts as the first change,
    /// so a session nobody selects in still has somewhere to measure from.
    last_change: Timestamp,
}

impl Session {
    /// Records a selection and says whether it changed anything.
    ///
    /// A repeat of the same target is not a change and does not restart
    /// the quiet period; a change of mind is and does.
    fn select(&mut self, from: &ActorId, target: &ActorId, at: Timestamp) -> bool {
        let changed = self.selections.get(from) != Some(target);
        self.selections.insert(from.clone(), target.clone());
        if changed {
            self.last_change = at;
        }
        changed
    }

    /// Whether every member has selected at least once.
    fn settled(&self) -> bool {
        self.selections.len() == self.members.len()
    }

    /// The earliest instant this session could close: its hard limit, or
    /// the end of its quiet period once every member has selected.
    fn deadline(&self) -> Timestamp {
        match self.quiet {
            Some(quiet) if self.settled() => self.limit.min(self.last_change + quiet),
            _ => self.limit,
        }
    }

    /// Whether the session's time is up at `now`.
    fn expired(&self, now: Timestamp) -> bool {
        now >= self.deadline()
    }
}

/// A game of Werewolf, from the deal to the outcome.
#[derive(Debug, Clone)]
pub struct Game {
    assignment: Assignment,
    living: BTreeSet<ActorId>,
    round: Round,
    phase: Phase,
    ties: ChaCha8Rng,
    /// Whether the game has begun, so that beginning twice is caught.
    begun: bool,
    /// The sessions of the phase now open, in the order they close: at
    /// night the pack, the seer and the doctor, by day just the one.
    sessions: Vec<Session>,
    /// What the night's closed sessions decided, kept until the last of
    /// them closes and the night resolves.
    resolved: Vec<(RequestKind, BTreeMap<ActorId, ActorId>)>,
    /// The clocks every session runs on.
    timing: Timing,
    /// Whom each doctor protected last night, for doctors that protected
    /// someone: the state the doctor's own rule constrains its next
    /// `Protect` with.
    last_protected: BTreeMap<ActorId, ActorId>,
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
    pub fn new(assignment: Assignment, seed: u64, timing: Timing) -> Self {
        let living: BTreeSet<ActorId> = assignment.players().map(|(who, _)| who.clone()).collect();
        let werewolves = assignment.pack().len();
        assert!(werewolves >= 1, "a game needs at least one werewolf");
        assert!(
            living.len() > 2 * werewolves,
            "a game needs more other players than werewolves"
        );
        Self {
            assignment,
            living,
            round: Round::FIRST,
            phase: Phase::Night,
            ties: ChaCha8Rng::seed_from_u64(seed_for(seed, TIES)),
            begun: false,
            sessions: Vec::new(),
            resolved: Vec::new(),
            timing,
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
    pub fn begin(&mut self, now: Timestamp) -> Vec<Directive> {
        assert!(!self.begun, "the game has already begun");
        self.begun = true;
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
        directives.extend(self.begin_phase(now));
        directives
    }

    /// Records one player's selection and returns whatever it caused.
    ///
    /// A member may select as often as it likes while its session is open,
    /// and its latest selection is its vote. Selecting usually causes
    /// nothing: a night session closes on its clock, not on its last member,
    /// and the day closes only on the selection that makes a majority.
    ///
    /// **A selection for a session that has closed is ignored**, not a panic.
    /// It lost a race with the clock, which ADR-0011 makes an ordinary
    /// message rather than a bug: no session of that round and kind is open
    /// any more, and nothing comes of the selection.
    ///
    /// # Panics
    ///
    /// If the rules ask nothing of this player in a session of that kind —
    /// because it is not living, or its role is not asked that at all — or
    /// if the target is outside the action space [`roles::action_space`]
    /// computes for it: the player itself, somebody not living, or, for a
    /// doctor, the player it protected the night before. Each is a bug in
    /// a player rather than a lost race.
    ///
    /// The check is made here, on arrival, rather than by handing out
    /// permission in advance (ADR-0014). It is the same rule either way:
    /// what a role is asked in a phase is [`Role::asked_in`], which the
    /// player consulted to decide to select at all.
    pub fn select(
        &mut self,
        from: &ActorId,
        selection: &Select,
        created: Timestamp,
        at: Timestamp,
    ) -> Vec<Directive> {
        let kind = selection.kind;
        // A selection names the session it was made in, so a selection from a
        // round that has passed is one whose session closed while it was
        // in flight: a lost race, and ignored like any other (ADR-0011).
        // It is checked first, so that a stale selection cannot be taken for
        // a current one just because a session of the same kind is open.
        if selection.round != self.round {
            return Vec::new();
        }
        let Some(index) = self
            .sessions
            .iter()
            .position(|session| session.kind == kind && session.members.contains(from))
        else {
            // No open session of that kind with this member. Either it
            // closed — a lost race again — or the rules never asked this
            // of this player, which is a bug in the player.
            assert!(
                self.assignment.role(from).is_some(),
                "{from} selected in a {kind:?} session but is not in this game"
            );
            assert_eq!(
                self.role(from).asked_in(kind.phase()),
                Some(kind),
                "{from} selected in a {kind:?} session, which a {} is never a member of",
                self.role(from)
            );
            return Vec::new();
        };
        let space = self.action_space_for(from, kind);
        assert!(
            space.contains(&selection.target),
            "{from} selected {} in a {kind:?} session, which is outside its action space",
            selection.target
        );
        self.sessions[index].select(from, &selection.target, at);

        // Accepted, so the players who should see it are told, in the
        // order the moderator accepted the selections rather than in whatever
        // order a queue happened to deliver them (ADR-0014). This is the
        // only way a selection reaches another player.
        let mut directives = Vec::new();
        if !selection.seen_by.is_empty() {
            directives.push(Directive::Forward {
                from: from.clone(),
                created,
                to: selection.seen_by.clone(),
                selection: selection.clone(),
            });
        }

        // The day ends the moment a majority of the living agree, and the
        // selection that made it is the hammer. Everything else waits for a
        // clock.
        //
        // The forward comes first: a player learns of the selection that
        // lynched somebody before it learns of the lynch, which is the
        // order the messages happened in.
        if kind == RequestKind::Nominate && self.majority().is_some() {
            // This selection is the one that completed the majority, so the
            // player who made it is the hammer.
            directives.extend(self.close_day(Some(from), at));
        }
        directives
    }

    /// Closes every session whose time is up at `now`, and resolves what
    /// that finishes.
    ///
    /// The moderator calls this whenever it wakes, and after every selection,
    /// because a deadline that passed while an observation waited joins
    /// that observation's cycle (ADR-0008).
    pub fn expire(&mut self, now: Timestamp) -> Vec<Directive> {
        if self.outcome.is_some() {
            return Vec::new();
        }
        // The day is one session, and closing it is its own path: it ends
        // with nobody lynched rather than resolving a plurality.
        if self.phase == Phase::Day {
            return match self.sessions.first() {
                Some(session) if session.expired(now) => self.close_day(None, now),
                _ => Vec::new(),
            };
        }
        let mut directives = Vec::new();
        while let Some(index) = self
            .sessions
            .iter()
            .position(|session| session.expired(now))
        {
            let session = self.sessions.remove(index);
            directives.extend(self.close_night_session(&session));
        }
        // The night resolves once its last session has closed.
        if self.sessions.is_empty() && !self.resolved.is_empty() {
            directives.extend(self.resolve_night(now));
        }
        directives
    }

    /// The earliest instant at which [`expire`](Self::expire) would close
    /// something, or `None` when no session is open.
    ///
    /// This is what the moderator hands
    /// [`Handler::deadline`](crate::Handler::deadline): the minimum over the
    /// open sessions of each one's hard limit and, for a night session whose
    /// members have all selected, the end of its quiet period.
    #[must_use]
    pub fn next_deadline(&self) -> Option<Timestamp> {
        if self.outcome.is_some() {
            return None;
        }
        self.sessions.iter().map(Session::deadline).min()
    }

    /// Which phase of which round the game is in, for a test that has to
    /// tell one phase's directives from the next.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn phase_now(&self) -> (Phase, Round) {
        (self.phase, self.round)
    }

    /// The player more than half of the living are selecting, if there
    /// is one.
    ///
    /// A majority of the *living*, not of those who have selected: a
    /// village that mostly stays quiet does not lynch on two votes.
    fn majority(&self) -> Option<ActorId> {
        let session = self.sessions.first()?;
        let mut counts: BTreeMap<&ActorId, usize> = BTreeMap::new();
        for target in session.selections.values() {
            *counts.entry(target).or_default() += 1;
        }
        counts
            .into_iter()
            .find(|(_, count)| *count * 2 > self.living.len())
            .map(|(who, _)| who.clone())
    }

    /// How the game ended, once it has.
    #[must_use]
    pub fn outcome(&self) -> Option<&Outcome> {
        self.outcome.as_ref()
    }

    /// Everyone still in the game.
    #[must_use]
    pub fn living(&self) -> &BTreeSet<ActorId> {
        &self.living
    }

    /// The targets the rules permit `who` in a session of `kind`, from the
    /// game's own state.
    ///
    /// The same [`roles::action_space`] a player computes from its
    /// knowledge, so the two cannot disagree; this is the game's side of
    /// it, and what [`select`](Self::select) checks against.
    #[must_use]
    pub fn action_space_for(&self, who: &ActorId, kind: RequestKind) -> Vec<ActorId> {
        roles::action_space(who, &self.living, kind, self.last_protected.get(who))
    }

    /// Everyone dealt into the game, living and dead, in agent order.
    ///
    /// This is the roster the moderator starts and, when the game is over,
    /// stops. A player leaves the *game* when it is eliminated and the
    /// *episode* when it is stopped, and those are not the same moment.
    pub fn players(&self) -> impl Iterator<Item = &ActorId> {
        self.assignment.players().map(|(who, _)| who)
    }

    /// What each player's game was worth, once the game has ended: **+1**
    /// for a player whose role's faction won, **−1** for every other,
    /// living or dead, in agent order. `None` while the game is still on.
    ///
    /// It is a rule of Werewolf and so it lives here, with the rules, and
    /// not in the [`Moderator`](super::Moderator), which is plumbing
    /// (ADR-0005). The rule is as simple as a rule gets, and the reason it
    /// stays simple is that there are no stalemates (ADR-0007): a day
    /// always eliminates somebody, so every game reaches a winner and every
    /// player gets exactly one of the two numbers. A game that could end
    /// undecided would need a third.
    ///
    /// Being dead is not being out of the game. A villager the pack
    /// devoured in the first round wins with its faction, and the record
    /// says so, because what a trajectory is being scored for is the
    /// behavior that led to the result and not the length of the episode.
    #[must_use]
    pub fn rewards(&self) -> Option<BTreeMap<ActorId, i32>> {
        let winner = self.outcome.as_ref()?.winner;
        Some(
            self.assignment
                .players()
                .map(|(who, role)| {
                    // A stalemate pays -1 to everyone, the same as losing,
                    // so that stalling can never beat losing and a side
                    // that is ahead has every reason to finish (ADR-0011).
                    let value = if winner == Some(role.faction()) {
                        1
                    } else {
                        -1
                    };
                    (who.clone(), value)
                })
                .collect(),
        )
    }

    /// Announces the current phase to the living and opens the sessions it
    /// calls for. Nobody is asked to act: a player works that out from its
    /// own role (ADR-0014).
    fn begin_phase(&mut self, now: Timestamp) -> Vec<Directive> {
        let directives = vec![self.narrate_living(Narration::PhaseBegan {
            round: self.round,
            phase: self.phase,
            living: self.living.clone(),
        })];
        // Who belongs to which of the phase's sessions. Nobody is told:
        // a player works this out for itself from its own role, and the
        // moderator keeps it because a session's membership is what
        // closes the session (ADR-0014). A player the rules leave
        // nowhere to select is not a member, so a session with no member
        // does not open.
        let mut members: BTreeMap<RequestKind, BTreeSet<ActorId>> = BTreeMap::new();
        for who in &self.living {
            let Some(kind) = self.role(who).asked_in(self.phase) else {
                continue;
            };
            if !self.action_space_for(who, kind).is_empty() {
                members.entry(kind).or_default().insert(who.clone());
            }
        }
        for (kind, members) in members {
            let clock = self.timing_for(kind);
            self.sessions.push(Session {
                kind,
                members,
                selections: BTreeMap::new(),
                quiet: clock.0,
                limit: now + clock.1,
                last_change: now,
            });
        }
        directives
    }

    /// The quiet period and hard limit configured for a session of `kind`.
    /// The day has no quiet period.
    fn timing_for(&self, kind: RequestKind) -> (Option<Duration>, Duration) {
        match kind {
            RequestKind::Devour => (Some(self.timing.pack.quiet), self.timing.pack.limit),
            RequestKind::Investigate => (Some(self.timing.seer.quiet), self.timing.seer.limit),
            RequestKind::Protect => (Some(self.timing.doctor.quiet), self.timing.doctor.limit),
            RequestKind::Nominate => (None, self.timing.day.limit),
        }
    }

    /// Closes one night session: the seer's finding if it looked at
    /// anybody, and nothing else.
    ///
    /// Nobody is told the session closed. Its members watched it converge
    /// through the forwards, so a summary would repeat what they already
    /// observed, and whether a selection arrived too late is the moderator's
    /// own bookkeeping rather than anything a player acts on (ADR-0015).
    /// The seer is told what it found here rather than at the end of the
    /// night, so that its own session's clock is the only one it waits
    /// on; a seer devoured the same night still learns what it learned.
    fn close_night_session(&mut self, session: &Session) -> Vec<Directive> {
        let mut directives = Vec::new();
        for (who, target) in &session.selections {
            match session.kind {
                RequestKind::Protect => {
                    self.last_protected.insert(who.clone(), target.clone());
                }
                RequestKind::Investigate => directives.push(Directive::Narrate {
                    to: [who.clone()].into(),
                    narration: Narration::Investigated {
                        target: target.clone(),
                        faction: self.role(target).faction(),
                    },
                }),
                RequestKind::Devour | RequestKind::Nominate => {}
            }
        }
        self.resolved
            .push((session.kind, session.selections.clone()));
        directives
    }

    /// Resolves the night once every session has closed: the victim is the
    /// plurality of the pack's latest selections, unless the doctor's
    /// protection reached them first.
    fn resolve_night(&mut self, now: Timestamp) -> Vec<Directive> {
        let mut votes = BTreeMap::new();
        let mut protected = BTreeSet::new();
        for (kind, selections) in std::mem::take(&mut self.resolved) {
            match kind {
                RequestKind::Devour => votes = selections,
                RequestKind::Protect => protected.extend(selections.into_values()),
                RequestKind::Investigate => {}
                RequestKind::Nominate => unreachable!("nobody nominates at night"),
            }
        }
        // No wolf selected, no kill: the pack that cannot agree to act does
        // not act.
        let mut directives = Vec::new();
        match plurality(votes.values(), &mut self.ties) {
            Some(victim) if !protected.contains(&victim) => {
                directives.extend(self.eliminate(&victim, Cause::Devoured));
                directives.extend(self.advance(now));
            }
            _ => {
                directives.push(self.narrate_living(Narration::NoDeath { round: self.round }));
                directives.extend(self.next_phase(now));
            }
        }
        directives
    }

    /// Closes the day: the lynching a majority called for, or `NoLynch`
    /// when the limit passed without one.
    ///
    /// The hammer is the moderator's own: it is what names the player who
    /// dies, and it is never narrated. A reader recovers it from the
    /// trajectory as the last selection forwarded before the lynching, and a
    /// player that saw that selection saw the same thing (ADR-0015).
    fn close_day(&mut self, hammer: Option<&ActorId>, now: Timestamp) -> Vec<Directive> {
        let session = self.sessions.remove(0);
        // The hammer is the selection that made the majority, and the target
        // of that majority is who dies for it.
        let lynched = hammer.and_then(|who| session.selections.get(who).cloned());
        let mut directives = Vec::new();
        if let Some(who) = lynched {
            directives.extend(self.eliminate(&who, Cause::Lynched));
            directives.extend(self.advance(now));
        } else {
            directives.push(self.narrate_living(Narration::NoLynch { round: self.round }));
            directives.extend(self.advance(now));
        }
        directives
    }

    /// Removes a player from the living and announces it, with the role
    /// revealed, to the living and to the player itself.
    fn eliminate(&mut self, who: &ActorId, cause: Cause) -> Vec<Directive> {
        assert!(self.living.remove(who), "{who} is not living");
        // To the living, which no longer includes the victim. A dead
        // player observes nothing, its own death least of all: it is
        // stopped in this same cycle, and an agent that has stopped is
        // not somebody to address (ADR-0012).
        vec![
            Directive::Narrate {
                to: self.living.clone(),
                narration: Narration::Eliminated {
                    who: who.clone(),
                    role: self.role(who),
                    round: self.round,
                    cause,
                },
            },
            Directive::Stop { who: who.clone() },
        ]
    }

    /// After an elimination: the outcome if a side has won, and otherwise
    /// the next phase.
    fn advance(&mut self, now: Timestamp) -> Vec<Directive> {
        if let Some(winner) = self.winner() {
            return vec![self.end(Some(winner))];
        }
        // A game nobody has won by the end of the cap's day is a
        // stalemate: it has run out of days, and going on would let a
        // village that keeps running out the clock play forever
        // (ADR-0011).
        if self.phase == Phase::Day && self.round.number() >= self.day_cap() {
            return vec![self.end(None)];
        }
        self.next_phase(now)
    }

    /// The day after which a game nobody has won is a stalemate: what the
    /// configuration set, or the number of players.
    fn day_cap(&self) -> u32 {
        self.timing.day_cap(self.assignment.players().count())
    }

    /// Begins the phase after this one: the day of the same round, or the
    /// night of the next.
    fn next_phase(&mut self, now: Timestamp) -> Vec<Directive> {
        match self.phase {
            Phase::Night => self.phase = Phase::Day,
            Phase::Day => {
                self.round = self.round.next();
                self.phase = Phase::Night;
            }
        }
        self.begin_phase(now)
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
    fn end(&mut self, winner: Option<Faction>) -> Directive {
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
    fn role(&self, who: &ActorId) -> Role {
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
    targets: impl IntoIterator<Item = &'a ActorId>,
    ties: &mut ChaCha8Rng,
) -> Option<ActorId> {
    let mut counts: BTreeMap<&ActorId, usize> = BTreeMap::new();
    for who in targets {
        *counts.entry(who).or_default() += 1;
    }
    let most = *counts.values().max()?;
    let leaders: Vec<&ActorId> = counts
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
    use crate::testing::{fast, id, ids, target, town, village};
    use crate::werewolf::role::Role::{Doctor, Seer, Villager, Werewolf};

    const SEED: u64 = 20_260_918;

    /// One phase of a script: every member's selection, keyed by the agent
    /// making it, in the order they are to be recorded.
    type Answers = Vec<(&'static str, ActorId)>;

    fn answers(pairs: &[(&'static str, &str)]) -> Answers {
        pairs
            .iter()
            .map(|(who, whom)| (*who, target(whom)))
            .collect()
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
        Game::new(assignment, SEED, fast())
    }

    /// An instant, in milliseconds from the start of the episode. The
    /// scripted games below never let a clock decide anything, so their
    /// instants only have to be ordered.
    fn at(millis: u64) -> Timestamp {
        Timestamp::from(Duration::from_millis(millis))
    }

    /// Long after any session opened in these tests could still be open:
    /// [`fast`]'s longest limit is 50 ms, so this closes everything.
    const LATER: u64 = 10_000;

    /// Who is a member of which of the phase's open sessions.
    ///
    /// Nothing tells a player this any more (ADR-0014): each works it out
    /// from its own role, and the moderator keeps the membership only
    /// because it is what closes a session. A test script stands in for
    /// every player at once, so it reads the same membership off the game
    /// rather than off a directive that no longer exists.
    fn asks(game: &Game) -> BTreeMap<ActorId, RequestKind> {
        game.sessions
            .iter()
            .flat_map(|session| {
                session
                    .members
                    .iter()
                    .map(move |who| (who.clone(), session.kind))
            })
            .collect()
    }

    /// Records one phase's selections in the order given, then lets every
    /// session's clock run out, and returns everything that came of it.
    ///
    /// A day that reaches a majority closes on the selection that made it, so
    /// the expiry that follows finds nothing left to close; a night always
    /// closes on its clocks. Either way the phase is over when this
    /// returns, which is what lets a script name one phase per entry.
    fn answer(game: &mut Game, answers: &Answers, at: Timestamp) -> Vec<Directive> {
        let asks = asks(game);
        assert_eq!(
            asks.keys().cloned().collect::<BTreeSet<_>>(),
            answers.iter().map(|(who, _)| id(who)).collect(),
            "a script selects for exactly the members of the open sessions"
        );
        // The phase these selections belong to, taken before any of them is
        // recorded: a day ends on the selection that makes a majority, so
        // selecting alone may finish it.
        let phase = (game.phase, game.round);
        let round = game.round;
        let mut caused = Vec::new();
        for (who, chosen) in answers {
            let who = id(who);
            let selection = Select {
                round,
                kind: asks[&who],
                target: chosen.clone(),
                seen_by: BTreeSet::new(),
            };
            caused.extend(game.select(&who, &selection, at, at));
        }
        // Close this phase and no more. If selecting already closed it,
        // there is nothing to run: running the clocks anyway would close
        // the *next* phase and the script would lose one. Expiring at
        // exactly the earliest deadline open, rather than at some far
        // instant, is what keeps one pass from cascading through every
        // later phase, whose limits would all be behind it.
        while (game.phase, game.round) == phase && game.outcome().is_none() {
            let Some(deadline) = game.next_deadline() else {
                break;
            };
            caused.extend(game.expire(deadline));
        }
        caused
    }

    /// Plays a script through a game and returns every directive it
    /// produced, in order, from the opening to the last phase scripted.
    ///
    /// Each phase is selected in at its own instant, far enough apart that
    /// one phase's clocks can never reach into the next.
    fn play(game: &mut Game, script: &[Answers]) -> Vec<Directive> {
        let mut all = game.begin(at(0));
        for (index, phase) in script.iter().enumerate() {
            let now = at((index as u64 + 1) * LATER);
            all.extend(answer(game, phase, now));
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
                round: Round::new(round),
                phase,
                living: ids(living),
            },
        )
    }

    /// The stop that follows an elimination: a dead player is out of the
    /// game and its agent with it (ADR-0012).
    fn stopped(who: &str) -> Directive {
        Directive::Stop { who: id(who) }
    }

    /// A death announced to `to` — less the victim, who is never told
    /// (ADR-0012), so a caller may pass the living as they were before it.
    fn eliminated<const N: usize>(
        to: [&str; N],
        who: &str,
        role: Role,
        round: u32,
        cause: Cause,
    ) -> Directive {
        Directive::Narrate {
            to: to
                .into_iter()
                .filter(|other| *other != who)
                .map(id)
                .collect(),
            narration: Narration::Eliminated {
                who: id(who),
                role,
                round: Round::new(round),
                cause,
            },
        }
    }

    /// The outcome as it is announced: to the living, who are exactly the
    /// survivors it names.
    fn outcome<const N: usize>(winner: Faction, rounds: u32, living: [&str; N]) -> Directive {
        narrate(
            living,
            Narration::Outcome(Outcome {
                winner: Some(winner),
                rounds: Round::new(rounds),
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

    /// Records one selection in the session of `kind` of the round now under
    /// way. A selection says for itself which session it belongs to
    /// (ADR-0014), so there is no id to look up.
    fn respond(game: &mut Game, from: &str, kind: RequestKind, target: ActorId) -> Vec<Directive> {
        let round = game.round;
        let selection = Select {
            round,
            kind,
            target,
            seen_by: BTreeSet::new(),
        };
        game.select(&id(from), &selection, at(0), at(0))
    }

    /// The same, for a selection naming a round of the caller's choosing.
    fn respond_in(
        game: &mut Game,
        from: &str,
        round: u32,
        kind: RequestKind,
        target: ActorId,
    ) -> Vec<Directive> {
        let selection = Select {
            round: Round::new(round),
            kind,
            target,
            seen_by: BTreeSet::new(),
        };
        game.select(&id(from), &selection, at(0), at(0))
    }

    /// Selections as a player really would, naming the audience the moderator
    /// should forward to, at the instants a caller chooses.
    ///
    /// `created` is when the player selected and `at` when the moderator
    /// received it: the two instants a forward turns on, and the reason
    /// this takes both.
    fn selections_seen_by<const N: usize>(
        game: &mut Game,
        from: &str,
        round: u32,
        kind: RequestKind,
        target: &str,
        seen_by: [&str; N],
        when: (Timestamp, Timestamp),
    ) -> Vec<Directive> {
        let (created, at) = when;
        let selection = Select {
            round: Round::new(round),
            kind,
            target: id(target),
            seen_by: ids(seen_by),
        };
        game.select(&id(from), &selection, created, at)
    }

    /// The forwards among some directives: who is told of whose selection.
    fn forwards(directives: &[Directive]) -> Vec<(&ActorId, &BTreeSet<ActorId>, &Select)> {
        directives
            .iter()
            .filter_map(|directive| match directive {
                Directive::Forward {
                    from,
                    to,
                    selection,
                    ..
                } => Some((from, to, selection)),
                _ => None,
            })
            .collect()
    }

    /// Whether any of these directives forwards anything.
    fn forwarded_anything(directives: &[Directive]) -> bool {
        !forwards(directives).is_empty()
    }

    #[test]
    fn a_selection_for_a_round_that_has_passed_is_forwarded_to_nobody() {
        // Scenario 1. The session it names closed when its round did, so
        // there is nobody it is still news to.
        let mut game = game(village());
        game.begin(at(0));
        // Let each phase run out its clock rather than scripting selections:
        // the night closes with nobody dead, the day with nobody lynched,
        // and round 2's night is the session now open.
        let mut now = at(0);
        while game.round == Round::new(1) {
            now = game.next_deadline().expect("an open phase has a clock");
            game.expire(now);
        }
        assert_eq!(game.round, Round::new(2), "the game moved on");
        let late = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "carol",
            ["alice"],
            (at(1), now + Duration::from_millis(1)),
        );
        assert!(
            !forwarded_anything(&late),
            "a selection from round 1 is nobody's news in round 2: {late:?}"
        );
    }

    #[test]
    fn a_selection_whose_session_has_closed_is_forwarded_to_nobody() {
        // Scenario 2. The round still stands; the session does not.
        let mut game = game(village());
        game.begin(at(0));
        // Close the night on its own hard limit.
        let deadline = game.next_deadline().expect("the night has a clock");
        game.expire(deadline);
        let late = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "carol",
            ["alice"],
            (at(1), deadline + Duration::from_millis(1)),
        );
        assert!(
            !forwarded_anything(&late),
            "the pack's session is closed, so nothing more is passed on: {late:?}"
        );
    }

    #[test]
    fn a_selection_that_arrives_after_the_game_has_ended_is_forwarded_to_nobody() {
        // Scenario 3: the failure this was all found through. A
        // straggler's nomination arrives after the last elimination ended
        // the game, and must reach nobody — a survivor's last observation
        // is the outcome.
        let mut game = game(village());
        let directives = play(&mut game, &village_wins());
        assert!(
            directives.iter().any(|directive| matches!(
                directive,
                Directive::Narrate {
                    narration: Narration::Outcome(_),
                    ..
                }
            )),
            "the script ends the game"
        );
        let round = game.round.number();
        let straggler = selections_seen_by(
            &mut game,
            "erin",
            round,
            RequestKind::Nominate,
            "carol",
            ["carol", "dave"],
            (at(1), at(10 * LATER)),
        );
        assert!(
            !forwarded_anything(&straggler),
            "the game is over; a selection is news to nobody: {straggler:?}"
        );
    }

    #[test]
    fn a_selection_is_forwarded_to_exactly_the_audience_it_names() {
        // Scenario 6, and the one that gives the negatives their meaning:
        // an accepted selection does reach the players it names, and nobody
        // else.
        let mut game = game(village());
        game.begin(at(0));
        let forwarded = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "carol",
            ["alice", "erin"],
            (at(1), at(2)),
        );
        let seen = forwards(&forwarded);
        let [(from, to, selection)] = seen.as_slice() else {
            panic!("exactly one forward: {forwarded:?}");
        };
        assert_eq!(*from, &id("bob"), "whose selection it is");
        assert_eq!(**to, ids(["alice", "erin"]), "and who is told of it");
        assert_eq!(selection.target, id("carol"));
    }

    #[test]
    fn the_hammers_selection_is_forwarded_before_the_day_closes() {
        // Scenario 4. A player learns of the selection that lynched somebody
        // before it learns of the lynch, because that is the order the two
        // things happened in.
        let mut game = game(village());
        game.begin(at(0));
        // Round 1's night runs out with nobody dead, so all five live into
        // the day and three of them are a majority.
        let night = game.next_deadline().expect("the night has a clock");
        game.expire(night);
        let living: Vec<ActorId> = game.living().iter().cloned().collect();
        assert_eq!(living.len(), 5, "nobody died in the night");
        let target = living[0].clone();
        let mut closing = Vec::new();
        for (index, who) in living.iter().skip(1).take(3).enumerate() {
            let seen_by: Vec<&str> = living
                .iter()
                .filter(|other| *other != who)
                .map(ActorId::as_str)
                .collect();
            let at_instant = night + Duration::from_millis(index as u64 + 1);
            let selection = Select {
                round: game.round,
                kind: RequestKind::Nominate,
                target: target.clone(),
                seen_by: seen_by.iter().map(|who| id(who)).collect(),
            };
            closing = game.select(who, &selection, at_instant, at_instant);
        }
        // The last of the three completed the majority, so its cycle both
        // forwards its selection and closes the day.
        let forwarded = closing
            .iter()
            .position(|directive| matches!(directive, Directive::Forward { .. }))
            .unwrap_or_else(|| panic!("the hammer's selection is forwarded: {closing:?}"));
        let narrated = closing
            .iter()
            .position(|directive| matches!(directive, Directive::Narrate { .. }))
            .unwrap_or_else(|| panic!("the hammer closes the day: {closing:?}"));
        assert!(
            forwarded < narrated,
            "the selection comes before the lynch it caused: {closing:?}"
        );
    }

    #[test]
    fn a_selection_that_arrives_on_the_deadline_is_forwarded_and_one_after_it_is_not() {
        // Scenario 5. `Game::select` is called before `Game::expire` for
        // one observation, because a deadline that passed while that
        // observation waited joins its cycle (ADR-0008). So a selection
        // received *at* the deadline is still in an open session, and one
        // received after the session was expired is not. The boundary is
        // asserted from both sides rather than assumed.
        let mut game = game(village());
        game.begin(at(0));
        let deadline = game.next_deadline().expect("the night has a clock");
        let on_time = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "carol",
            ["alice"],
            (at(1), deadline),
        );
        assert!(
            forwarded_anything(&on_time),
            "a selection received on the deadline is still in an open session: {on_time:?}"
        );
        // Now the clock is allowed to close it, and the next selection is late.
        game.expire(deadline);
        let too_late = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "erin",
            ["alice"],
            (at(2), deadline + Duration::from_millis(1)),
        );
        assert!(
            !forwarded_anything(&too_late),
            "and one arriving after the session closed is not: {too_late:?}"
        );
    }

    #[test]
    fn a_change_of_mind_is_forwarded_in_the_order_the_moderator_accepted_it() {
        // Scenario 8. The divergence peer-to-peer delivery cannot rule
        // out: one player selects twice and every recipient must be told of
        // both, in the order the moderator took them, so that everybody's
        // idea of that player's vote ends the same way.
        let mut game = game(village());
        game.begin(at(0));
        let night = game.next_deadline().expect("the night has a clock");
        game.expire(night);
        let living: Vec<ActorId> = game.living().iter().cloned().collect();
        let voter = living[0].clone();
        let seen_by: Vec<&str> = living
            .iter()
            .filter(|other| **other != voter)
            .map(ActorId::as_str)
            .collect();
        let mut targets = Vec::new();
        for (index, target) in [&living[1], &living[2]].into_iter().enumerate() {
            let at_instant = night + Duration::from_millis(index as u64 + 1);
            let selection = Select {
                round: game.round,
                kind: RequestKind::Nominate,
                target: target.clone(),
                seen_by: seen_by.iter().map(|who| id(who)).collect(),
            };
            let directives = game.select(&voter, &selection, at_instant, at_instant);
            let seen = forwards(&directives);
            let [(from, to, forwarded)] = seen.as_slice() else {
                panic!("each selection is forwarded once: {directives:?}");
            };
            assert_eq!(*from, &voter);
            let expected: BTreeSet<ActorId> = seen_by.iter().map(|who| id(who)).collect();
            assert_eq!(**to, expected);
            targets.push(forwarded.target.clone());
        }
        assert_eq!(
            targets,
            vec![living[1].clone(), living[2].clone()],
            "both selections are forwarded, in the order they were accepted"
        );
        // And the vote the game counts is the later of the two.
        let counted = game.sessions[0].selections.get(&voter);
        assert_eq!(
            counted,
            Some(&living[2]),
            "the most recent selection is the vote"
        );
    }

    #[test]
    fn a_selection_the_game_rejects_is_forwarded_to_nobody_and_changes_no_vote() {
        // Scenario 9's other half: what a player could compute from what
        // it observed agrees with what the game counts, because the game
        // forwards exactly the selections it accepted and no others.
        let mut game = game(village());
        game.begin(at(0));
        let accepted = selections_seen_by(
            &mut game,
            "bob",
            1,
            RequestKind::Devour,
            "carol",
            ["alice"],
            (at(1), at(2)),
        );
        assert!(forwarded_anything(&accepted));
        let counted = game
            .sessions
            .iter()
            .find(|session| session.kind == RequestKind::Devour);
        let counted = counted.expect("the pack's session is open");
        assert_eq!(
            counted.selections.get(&id("bob")),
            Some(&id("carol")),
            "the accepted selection is the vote"
        );
        // A selection for a round that is not the open one changes nothing and
        // is told to nobody, so no observer can believe otherwise. Rounds
        // are counted from 1, so the round that is not this one is the
        // next; the game rejects it by the same inequality either way.
        let stale = selections_seen_by(
            &mut game,
            "bob",
            2,
            RequestKind::Devour,
            "erin",
            ["alice"],
            (at(3), at(4)),
        );
        assert!(!forwarded_anything(&stale), "{stale:?}");
        let counted = game
            .sessions
            .iter()
            .find(|session| session.kind == RequestKind::Devour)
            .expect("still open");
        assert_eq!(
            counted.selections.get(&id("bob")),
            Some(&id("carol")),
            "and the vote it would have changed is untouched"
        );
    }

    #[test]
    fn a_selection_nobody_else_sees_is_forwarded_to_nobody() {
        // Scenario 7. The seer's and the doctor's business is their own,
        // and they name no audience, so nothing is forwarded.
        let mut game = game(village());
        game.begin(at(0));
        for (who, kind) in [
            ("carol", RequestKind::Investigate),
            ("dave", RequestKind::Protect),
        ] {
            let directives =
                selections_seen_by(&mut game, who, 1, kind, "alice", [], (at(1), at(2)));
            assert!(
                !forwarded_anything(&directives),
                "{who}'s {kind:?} is nobody else's business: {directives:?}"
            );
        }
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
            answers(&[("bob", "alice"), ("dave", "bob")]),
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

    /// In the village, the doctor protects the victim: nobody dies.
    fn saved() -> Vec<Answers> {
        vec![answers(&[
            ("bob", "erin"),
            ("carol", "bob"),
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
                let mut game = Game::new(assignment.clone(), SEED, fast());
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
                investigated("carol", "bob", Faction::Werewolves),
                eliminated(everyone, "alice", Villager, 1, Cause::Devoured),
                stopped("alice"),
                phase_began(1, Phase::Day, survivors),
                // Nothing stands between the phase and the lynching it
                // came to. The selections that made the majority went to the
                // living as they arrived, so a close has nothing left to
                // tell them (ADR-0015).
                eliminated(survivors, "bob", Werewolf, 1, Cause::Lynched),
                stopped("bob"),
                outcome(Faction::Village, 1, ["carol", "dave", "erin"]),
            ]
        );
        assert_eq!(
            game.outcome(),
            Some(&Outcome {
                winner: Some(Faction::Village),
                rounds: Round::new(1),
                living: ids(["carol", "dave", "erin"]),
            })
        );
        assert_eq!(*game.living(), ids(["carol", "dave", "erin"]));
    }

    #[test]
    fn the_werewolves_win_at_parity() {
        let mut game = game(village());
        let directives = play(&mut game, &werewolves_win());
        // Five deals and the first PhaseBegan, and then the night: no
        // request stands between the phase and what the selecting caused
        // (ADR-0014).
        assert_eq!(
            directives[6..],
            [
                // The seer is devoured tonight and still learns what it
                // learned, because its own session closed before the
                // night resolved.
                investigated("carol", "bob", Faction::Werewolves),
                eliminated(
                    ["alice", "bob", "carol", "dave", "erin"],
                    "carol",
                    Seer,
                    1,
                    Cause::Devoured,
                ),
                stopped("carol"),
                phase_began(1, Phase::Day, ["alice", "bob", "dave", "erin"]),
                // dave's selection is the third of four living, a majority,
                // so it is the hammer and erin never selects at all. The
                // hammer is not narrated: a reader takes it from the
                // last selection the moderator passed on (ADR-0015).
                eliminated(
                    ["alice", "bob", "dave", "erin"],
                    "erin",
                    Villager,
                    1,
                    Cause::Lynched,
                ),
                stopped("erin"),
                // No seer lives, so the second night asks nothing of one.
                phase_began(2, Phase::Night, ["alice", "bob", "dave"]),
                eliminated(
                    ["alice", "bob", "dave"],
                    "alice",
                    Villager,
                    2,
                    Cause::Devoured
                ),
                stopped("alice"),
                outcome(Faction::Werewolves, 2, ["bob", "dave"]),
            ]
        );
        assert_eq!(
            game.outcome().and_then(|outcome| outcome.winner),
            Some(Faction::Werewolves)
        );
    }

    /// Plays a game to its end, selecting in every open session with a
    /// target drawn
    /// from the action space the rules compute, and returns the outcome
    /// and how many were living at the start of each round.
    ///
    /// The selections are arbitrary, so nothing but the rules keeps the game
    /// finite: this is the termination guarantee under adversity.
    fn play_out(assignment: Assignment, seed: u64) -> (Outcome, Vec<usize>) {
        let (game, _, living) = played(assignment, seed);
        (game.outcome().unwrap().clone(), living)
    }

    /// The same, giving back the game itself, whatever else is wanted of
    /// it once it has ended.
    fn played(assignment: Assignment, seed: u64) -> (Game, Outcome, Vec<usize>) {
        played_with(assignment, seed, fast())
    }

    /// The same, under timing the caller chooses.
    fn played_with(
        assignment: Assignment,
        seed: u64,
        timing: Timing,
    ) -> (Game, Outcome, Vec<usize>) {
        let mut game = Game::new(assignment, seed, timing);
        let mut moves = ChaCha8Rng::seed_from_u64(seed);
        let mut clock = 0;
        game.begin(at(clock));
        let mut living = vec![game.living.len()];
        let mut round = game.round;
        while game.outcome().is_none() {
            if game.round != round {
                round = game.round;
                living.push(game.living.len());
            }
            let asked = asks(&game);
            assert!(
                !asked.is_empty(),
                "a running game always has a session open"
            );
            clock += LATER;
            let now = at(clock);
            let selection_round = game.round;
            for (who, kind) in asked {
                let space =
                    roles::action_space(&who, &game.living, kind, game.last_protected.get(&who));
                let selection = Select {
                    round: selection_round,
                    kind,
                    target: pick(&mut moves, &space).clone(),
                    seen_by: BTreeSet::new(),
                };
                game.select(&who, &selection, now, now);
            }
            // Every player selects once and never changes its mind, so
            // running the clock out is what closes the phase.
            game.expire(at(clock + LATER));
        }
        let outcome = game.outcome().unwrap().clone();
        (game, outcome, living)
    }

    #[test]
    fn every_player_is_paid_for_its_faction_living_or_dead() {
        // The reward is the faction's, not the survivor's: a player the
        // pack devoured in round one still wins with its side. Nothing is
        // ever zero, because there are no stalemates.
        for assignment in [village(), town(), pack_of_three()] {
            for seed in 0..20 {
                let (game, outcome, _) = played(assignment.clone(), seed);
                let rewards = game.rewards().expect("the game has ended");
                assert_eq!(
                    rewards.keys().collect::<BTreeSet<_>>(),
                    assignment.players().map(|(who, _)| who).collect(),
                    "every player is paid, and nobody else: seed {seed}"
                );
                for (who, value) in &rewards {
                    let role = assignment.role(who).unwrap();
                    let expected = if outcome.winner == Some(role.faction()) {
                        1
                    } else {
                        -1
                    };
                    assert_eq!(
                        *value, expected,
                        "{who} held {role} and {:?} won: seed {seed}",
                        outcome.winner
                    );
                }
                // The dead are paid too, which is the whole point of
                // paying on the faction rather than on survival.
                let dead: Vec<&ActorId> = rewards
                    .keys()
                    .filter(|who| !outcome.living.contains(who))
                    .collect();
                assert!(!dead.is_empty(), "somebody died: seed {seed}");
            }
        }
    }

    #[test]
    fn a_game_still_running_has_paid_nobody() {
        let mut game = Game::new(village(), SEED, fast());
        assert_eq!(game.rewards(), None, "a game not yet begun pays nobody");
        game.begin(at(0));
        assert_eq!(game.rewards(), None, "nor does one under way");
    }

    /// A game whose day cap is `cap`, for the stalemate tests.
    fn capped(assignment: Assignment, cap: u32) -> Game {
        let timing = Timing {
            day_cap: Some(cap),
            ..fast()
        };
        Game::new(assignment, SEED, timing)
    }

    #[test]
    fn a_random_game_stalemates_only_when_the_cap_is_tight() {
        // Where the cap actually bites. At its default of one day per
        // player a seven-player random game always resolves first, so the
        // rule costs a uniform baseline nothing; tighten the cap and
        // stalemates appear, which is the guard working. Both halves
        // matter: a cap that never fired would be untested, and one that
        // fired at the default would be shaping the baseline's win rates.
        let rate = |cap: u32| {
            let timing = Timing {
                day_cap: Some(cap),
                ..fast()
            };
            (0..40)
                .filter(|seed| {
                    let (_, outcome, _) = played_with(town(), *seed, timing);
                    outcome.winner.is_none()
                })
                .count()
        };
        assert_eq!(rate(7), 0, "the default cap never fires for seven players");
        assert!(rate(3) > 0, "a tight cap does fire");
    }

    #[test]
    fn a_game_that_reaches_its_day_cap_is_a_stalemate() {
        // Nobody dies: the pack splits every night and the doctor is not
        // needed, and every day scatters so no majority forms. On the
        // cap's day the game ends with no winner (ADR-0011).
        let mut game = capped(town(), 2);
        let scattered = || {
            answers(&[
                ("alice", "bob"),
                ("bob", "carol"),
                ("carol", "dave"),
                ("dave", "erin"),
                ("erin", "frank"),
                ("frank", "grace"),
                ("grace", "alice"),
            ])
        };
        // bob and frank are the pack, and they agree, so there is no
        // tie-break and the doctor knows exactly whom to cover. A
        // different victim each night, since the doctor may not protect
        // the same player twice running.
        let night = |victim: &'static str| {
            answers(&[
                ("bob", victim),
                ("carol", "erin"),
                ("dave", victim),
                ("frank", victim),
            ])
        };
        let directives = play(
            &mut game,
            &[night("alice"), scattered(), night("grace"), scattered()],
        );

        let outcome = game.outcome().expect("the cap ended the game");
        assert_eq!(outcome.winner, None, "a stalemate has no winner");
        assert_eq!(outcome.rounds, Round::new(2), "it ended on the cap's day");
        assert_eq!(outcome.living.len(), 7, "nobody died");
        assert!(
            directives.contains(&narrate(
                ["alice", "bob", "carol", "dave", "erin", "frank", "grace"],
                Narration::Outcome(outcome.clone()),
            )),
            "the outcome is narrated to the living like a won game's"
        );
    }

    #[test]
    fn a_stalemate_pays_every_player_the_same_as_losing() {
        // -1 to everyone, living and dead, so that stalling can never
        // beat losing and a side that is ahead has a reason to finish.
        let mut game = capped(town(), 1);
        play(
            &mut game,
            &[
                answers(&[
                    ("bob", "alice"),
                    ("carol", "erin"),
                    ("dave", "alice"),
                    ("frank", "grace"),
                ]),
                answers(&[
                    ("alice", "bob"),
                    ("bob", "carol"),
                    ("carol", "dave"),
                    ("dave", "erin"),
                    ("erin", "frank"),
                    ("frank", "grace"),
                    ("grace", "alice"),
                ]),
            ],
        );
        assert_eq!(game.outcome().unwrap().winner, None);
        for (who, value) in game.rewards().expect("the game ended") {
            assert_eq!(value, -1, "{who} was not paid as a loser");
        }
    }

    #[test]
    fn a_game_won_on_the_caps_day_is_a_win_and_not_a_stalemate() {
        // The cap ends a game nobody has won. A game won *on* that day
        // was won, and the win is checked first.
        let mut game = capped(village(), 1);
        play(&mut game, &village_wins());
        let outcome = game.outcome().expect("the game ended");
        assert_eq!(outcome.winner, Some(Faction::Village));
        assert_eq!(outcome.rounds, Round::new(1));
        for (who, value) in game.rewards().unwrap() {
            let expected = if game.role(&who).faction() == Faction::Village {
                1
            } else {
                -1
            };
            assert_eq!(value, expected, "{who}");
        }
    }

    #[test]
    fn a_game_of_players_that_all_select_ends() {
        // What termination rests on has changed. Under ADR-0004 a
        // `Nominate` could not abstain, so every day lynched somebody and
        // the living set strictly shrank: a game of n players was over by
        // round n whatever anybody did. Under ADR-0011 a day can end with
        // nobody lynched, so the living set may hold steady for a round,
        // and what bounds a game is the day cap — which is #69's, not
        // here yet.
        //
        // What still holds, and is what this checks, is that a game whose
        // players all select does end, and that the living set never
        // grows.
        for assignment in [village(), town(), pack_of_three()] {
            let players = assignment.players().count();
            for seed in 0..200 {
                let (outcome, living) = play_out(assignment.clone(), seed);
                assert!(
                    outcome.rounds.number() as usize <= players * 2,
                    "{players} players, seed {seed}: {outcome:?}"
                );
                assert!(
                    living.windows(2).all(|pair| pair[1] <= pair[0]),
                    "the living set never grows, seed {seed}: {living:?}"
                );
            }
        }
    }

    #[test]
    fn a_night_is_resolved_by_its_clocks_and_never_by_a_selection() {
        // Under ADR-0004 the last answer resolved the phase. Under
        // ADR-0011 a night session closes a quiet period after its
        // members have settled, so selecting says nothing on its own — not
        // even the selection that leaves nothing outstanding.
        let mut game = game(village());
        game.begin(at(0));
        let asks = asks(&game);
        for (who, whom) in [("bob", "alice"), ("carol", "bob"), ("dave", "erin")] {
            assert_eq!(
                respond(&mut game, who, asks[&id(who)], target(whom)),
                [],
                "{who}'s selection resolved something"
            );
        }

        // Every member has selected, so each session now closes a quiet
        // period after its last change rather than at its hard limit.
        let deadline = game.next_deadline().expect("the sessions are open");
        assert_eq!(deadline, at(0) + fast().pack.quiet);
        let caused = game.expire(deadline);
        // Closing a session says nothing to its members (ADR-0015).
        // What the close does produce is the seer's finding, then the
        // death.
        assert_eq!(caused[0], investigated("carol", "bob", Faction::Werewolves));
    }

    #[test]
    fn selections_in_any_order_settle_the_same_game() {
        // Under ADR-0011 the order of selections is not reproducible and the
        // outcome is: a day closes on whoever completes the majority, so
        // reordering changes which player is the hammer and which later
        // selections land inside the session at all. What it cannot change is
        // who dies, what the seer found, or who wins.
        let reference = deaths(&play(&mut game(village()), &village_wins()));
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
            let mut game = game(village());
            let directives = play(&mut game, reordered);
            assert_eq!(deaths(&directives), reference, "{reordered:?}");
            assert_eq!(game.outcome().unwrap().winner, Some(Faction::Village));
        }
    }

    /// What a run settled, as ADR-0011 guarantees it: every elimination
    /// and every finding, in order, and nothing about how they were
    /// reached.
    fn deaths(directives: &[Directive]) -> Vec<Narration> {
        directives
            .iter()
            .filter_map(|directive| match directive {
                Directive::Narrate { narration, .. } => match narration {
                    Narration::Eliminated { .. }
                    | Narration::Investigated { .. }
                    | Narration::NoDeath { .. }
                    | Narration::NoLynch { .. }
                    | Narration::Outcome(_) => Some(narration.clone()),
                    _ => None,
                },
                // A forward is one player's selection passed on, not
                // something the game settled.
                Directive::Forward { .. } | Directive::Stop { .. } => None,
            })
            .collect()
    }

    #[test]
    fn the_same_seed_and_responses_produce_the_same_game() {
        for (assignment, script) in scripted_games() {
            let mut first = Game::new(assignment.clone(), SEED, fast());
            let mut second = Game::new(assignment, SEED, fast());
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
        // Seven deals and the first PhaseBegan come before it.
        assert_eq!(
            directives[8..],
            [
                eliminated(everyone, "dave", Villager, 1, Cause::Devoured),
                stopped("dave"),
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
    fn a_plurality_of_nothing_is_nobody() {
        // A member that never selected is absent from its session's
        // selections rather than present with an abstention, so a session
        // nobody selected in is the only way a plurality comes back empty
        // (ADR-0011).
        let mut ties = ChaCha8Rng::seed_from_u64(1);
        assert_eq!(plurality(&[], &mut ties), None);
        assert_eq!(plurality(&[target("bob")], &mut ties), Some(id("bob")));
    }

    #[test]
    fn a_save_is_announced_as_no_death_and_nothing_more() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let mut game = game(village());
        let directives = play(&mut game, &saved());
        assert_eq!(
            directives[6..],
            [
                // The seer looked at bob and found the pack.
                investigated("carol", "bob", Faction::Werewolves),
                narrate(
                    everyone,
                    Narration::NoDeath {
                        round: Round::new(1)
                    }
                ),
                phase_began(1, Phase::Day, everyone),
            ]
        );
        assert_eq!(*game.living(), ids(everyone));
        // A save is never announced as one. The only thing the doctor
        // hears that the village does not is its own deal: its session
        // closes silently (ADR-0015), and that it protected the victim
        // is told to nobody, so the village cannot tell a save from a
        // pack that selected nowhere.
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
            let mut game = Game::new(assignment.clone(), SEED, fast());
            let night = answers(&[
                ("bob", "alice"),
                ("carol", whom),
                ("dave", "bob"),
                // The second seer is asked too, and selects somewhere it is
                // allowed; only carol's finding is what this test reads.
                ("erin", "alice"),
            ]);
            let directives = play(&mut game, &[night]);
            let finding = investigated("carol", whom, faction);
            assert!(directives.contains(&finding), "{whom}: {directives:?}");
        }
    }

    #[test]
    fn no_session_opens_for_a_missing_seer_or_doctor() {
        // Nobody is asked anything any more (ADR-0014), so what there is
        // to check is the membership itself: with no seer and no doctor
        // in the deal, the night opens the pack's session and no other.
        let mut game = game(pack_of_three());
        game.begin(at(0));
        let asked = asks(&game);
        assert_eq!(
            asked.keys().cloned().collect::<BTreeSet<_>>(),
            ids(["alice", "bob", "carol"])
        );
        assert!(asked.values().all(|kind| *kind == RequestKind::Devour));
        assert_eq!(game.sessions.len(), 1, "one session, the pack's");
    }

    #[test]
    fn a_dead_doctor_is_no_longer_a_member_of_the_nights_sessions() {
        // Nothing is issued to anybody now (ADR-0014), so what "the dead
        // doctor is asked nothing" means is that it is not a member of
        // the second night's sessions — and, dave being the only doctor,
        // that no Protect session opens at all. Membership is what closes
        // a session, so a dead member would hang the night.
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
        assert!(!game.living().contains(&id("dave")), "dave is dead");
        assert_eq!(game.phase, Phase::Night);
        assert_eq!(game.round, Round::new(2));
        // Two sessions where the first night had three, and dave is in
        // neither of them.
        assert_eq!(
            asks(&game),
            [
                (id("bob"), RequestKind::Devour),
                (id("carol"), RequestKind::Investigate),
            ]
            .into_iter()
            .collect::<BTreeMap<_, _>>()
        );
        // The second night opens on the phase and nothing else: there is
        // no longer anything between the announcement and the selecting.
        let second_night = directives
            .iter()
            .rposition(|directive| {
                matches!(
                    directive,
                    Directive::Narrate {
                        narration: Narration::PhaseBegan {
                            phase: Phase::Night,
                            ..
                        },
                        ..
                    }
                )
            })
            .expect("the game reached a second night");
        assert_eq!(
            directives[second_night..],
            [phase_began(2, Phase::Night, ["alice", "bob", "carol"])]
        );
    }

    #[test]
    fn no_directive_addresses_a_dead_player() {
        for (_, _, directives) in played_games() {
            let mut dead = BTreeSet::new();
            for directive in &directives {
                match directive {
                    Directive::Narrate { to, narration } => {
                        // No exception for the victim's own death. A dead
                        // player observes nothing at all, its own death
                        // least of all: it is stopped in the cycle the
                        // death is announced and is not among those told
                        // (ADR-0012).
                        for who in to {
                            assert!(
                                !dead.contains(who),
                                "{who} is dead but is told {narration:?}"
                            );
                        }
                        if let Narration::Eliminated { who, .. } = narration {
                            assert!(!to.contains(who), "{who} is told of its own death");
                        }
                    }
                    // A forwarded selection is an observation like any
                    // other, so the same rule holds: the dead are told
                    // nothing, a selection included.
                    Directive::Forward { from, to, .. } => {
                        for who in to {
                            assert!(
                                !dead.contains(who),
                                "{who} is dead but is forwarded a selection from {from}"
                            );
                        }
                        assert!(!to.contains(from), "{from} is forwarded its own selection");
                    }
                    // The death and the stop are one cycle's work, so a
                    // player counts as dead from the stop onward.
                    Directive::Stop { who } => {
                        dead.insert(who.clone());
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
        game.begin(at(0));
        let mut sizes = vec![game.living().len()];
        for (index, phase) in werewolves_win().iter().enumerate() {
            answer(&mut game, phase, at((index as u64 + 1) * LATER));
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
    #[should_panic(
        expected = "carol selected in a Devour session, which a Seer is never a member of"
    )]
    fn selecting_in_a_session_the_role_is_never_a_member_of_panics() {
        // There is no longer an id to get wrong, so the bug that was
        // "a request nobody was asked" and the bug that was "a request
        // asked of somebody else" are now one bug: selecting in a session
        // the rules never made you a member of (ADR-0014). A seer at
        // night belongs to the Investigate session and no other, and the
        // game can say so from the role alone.
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "carol", RequestKind::Devour, target("alice"));
    }

    #[test]
    #[should_panic(expected = "zara selected in a Devour session but is not in this game")]
    fn selecting_from_outside_the_game_panics() {
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "zara", RequestKind::Devour, target("alice"));
    }

    #[test]
    fn a_selection_for_a_closed_session_is_ignored() {
        // It lost a race with the clock, which ADR-0011 makes an ordinary
        // message rather than a bug. bob is a werewolf, so the Devour
        // session was genuinely its own: what is wrong with the selection is
        // only that it is late.
        let mut game = game(village());
        game.begin(at(0));
        // The pack's session closes at its limit with nobody having
        // selected, so bob has no session left to select in.
        let closed = game.expire(at(LATER));
        assert!(!closed.is_empty(), "the night's sessions closed");
        assert!(
            respond_in(&mut game, "bob", 1, RequestKind::Devour, target("alice")).is_empty(),
            "a selection for a closed session causes nothing"
        );
    }

    #[test]
    fn a_selection_naming_a_round_that_has_passed_is_ignored() {
        // A selection carries its own round now, so a selection in flight while
        // its phase ended names a round the game has left behind. That is
        // the same lost race, and is ignored rather than mistaken for a
        // selection in the session of the same kind now open (ADR-0014).
        let mut game = game(village());
        play(
            &mut game,
            &[
                answers(&[("bob", "alice"), ("carol", "bob"), ("dave", "erin")]),
                answers(&[
                    ("bob", "carol"),
                    ("carol", "dave"),
                    ("dave", "carol"),
                    ("erin", "bob"),
                ]),
            ],
        );
        assert_eq!(game.round, Round::new(2), "the second night is under way");
        assert!(
            respond_in(&mut game, "bob", 1, RequestKind::Devour, target("erin")).is_empty(),
            "a selection from round one causes nothing in round two"
        );
    }

    #[test]
    #[should_panic(
        expected = "dave selected dave in a Protect session, which is outside its action space"
    )]
    fn a_doctor_protecting_itself_panics() {
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "dave", RequestKind::Protect, target("dave"));
    }

    /// In the village, alice is devoured while the doctor protects erin,
    /// then carol is lynched, so that the second night opens a pack
    /// session for bob and a Protect session for dave again.
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
        expected = "dave selected erin in a Protect session, which is outside its action space"
    )]
    fn a_doctor_protecting_the_same_player_two_nights_running_panics() {
        let mut game = game(village());
        play(&mut game, &doctor_protected_erin());
        respond(&mut game, "dave", RequestKind::Protect, target("erin"));
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
        // In the hamlet, dave protects erin on the first night, then
        // carol on the second, and on the third may protect erin again:
        // the constraint is last night's protection alone. There is no
        // request to find any more (ADR-0014) — dave's own role says it
        // is the Protect session it belongs to, so the selection names that
        // and the game checks the doctor's rule on arrival.
        for second_night in ["carol", "frank"] {
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
            assert_eq!(game.round, Round::new(3), "the third night is under way");
            assert_eq!(asks(&game).len(), 4, "four members select tonight");
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
            assert_eq!(
                asks(&game).get(&id("dave")),
                Some(&RequestKind::Protect),
                "the doctor is a member of the third night's Protect session"
            );
            respond(&mut game, "dave", RequestKind::Protect, target("erin"));
        }
    }

    #[test]
    #[should_panic(
        expected = "bob selected alice in a Nominate session, which is outside its action space"
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
        respond(&mut game, "bob", RequestKind::Nominate, target("alice"));
    }

    #[test]
    fn a_selection_after_the_outcome_is_ignored() {
        // A player that selected before its stop reached it has lost the
        // same race as a late selection, and the game says nothing about it.
        let mut game = game(village());
        play(&mut game, &village_wins());
        assert!(game.outcome().is_some());
        assert!(respond(&mut game, "carol", RequestKind::Nominate, target("dave")).is_empty());
    }

    #[test]
    #[should_panic(expected = "the game has already begun")]
    fn beginning_twice_panics() {
        let mut game = game(village());
        game.begin(at(0));
        game.begin(at(0));
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
