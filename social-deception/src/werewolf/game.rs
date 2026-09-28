//! The rules of Werewolf as a pure state machine.
//!
//! [`Game`] takes players' points and the passing of time in, and produces
//! [`Directive`]s out: what to say, to whom, in what order. It touches no
//! channel, spawns no thread and reads no clock — every instant it knows is
//! an argument — so the whole of the rules is testable by calling functions
//! with a scripted sequence of points and instants. The moderator that
//! wraps it is plumbing: it folds the events it receives into
//! [`Game::point`], wakes on [`Game::next_deadline`] to call
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
//! decides when a session closes, and it checks a point against the rules
//! when the point arrives. A player the rules leave nothing to point at is
//! not a member of anything.
//!
//! A member may point whenever it likes while its session is open and may
//! change its mind; its most recent point is its vote, and pointing
//! nowhere is how it abstains (ADR-0011).
//!
//! **A night is three sessions at once**, each with a clock of its own so
//! that a slow role cannot spend another's time: the pack devours, the seer
//! investigates, the doctor protects. Each closes when every member has
//! pointed and no point has changed for its quiet period — any change
//! restarts it — or at its hard limit, whichever comes first. The seer is
//! told its finding when its own session closes, so a seer devoured that
//! same night still learns what it learned.
//!
//! A session of **more than one member** closes with a tally of its
//! members' latest points, to those members. That is what lets a pack see
//! where it has converged. A session of one is sent none: it would tell a
//! lone seer, doctor or wolf the one thing it already knows, having just
//! said it. Everything the moderator sends a player is something that
//! player observes for its own sake; what the moderator needs to remember
//! it keeps, and a reader of the trajectory reconstructs a session from
//! the points the moderator observed, which are the primary record.
//!
//! The night resolves once every session has closed: the victim is the
//! plurality of the pack's latest points, and no wolf pointing means
//! nobody is devoured; the doctor's protection then applies, and either the
//! death is revealed with its role, or nobody died. A save is never
//! announced as a save, so the village cannot tell one from a pack that
//! pointed nowhere.
//!
//! **A day is one session**, and its points are public. It closes the
//! moment a majority of the *living* — more than half of them, not of those
//! who have pointed — point at the same player: that player is lynched, and
//! the point that made the majority is the *hammer*. If the hard limit
//! passes first, nobody is lynched.
//!
//! A point for a session that has closed is ignored. It lost a race with
//! the clock, which is a fact about timing rather than a bug.
//!
//! The win condition is checked after every elimination, and only then: the
//! village wins when no werewolf lives, and the werewolves win when they are
//! at least as many as everyone else, since from parity onward they cannot
//! lose. The outcome is the one thing said to everyone, living and dead.
//!
//! # What is reproducible, and what is not
//!
//! For a fixed configuration and seed, every phase's *outcome* is the same
//! on every run: each death, each finding, the winner and the rewards. The
//! order points arrived in is not, and neither is which late points landed
//! inside a session before it closed. That is the guarantee ADR-0011 makes
//! and [`Transcript::verdicts`](super::Transcript::verdicts) projects a
//! game onto.
//!
//! It holds because a session keeps its members' latest points in a map
//! read in canonical order, not in arrival order, and because a random
//! player points once as its session opens and never changes its mind: a
//! night session closes only after all its members have pointed, and by day
//! no later point could have made a different majority. Ties are broken by
//! a generator seeded from the episode's master seed under its own label,
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
//! A point in a session the rules never make its sender a member of, or
//! one naming a target outside that session's action space, is a bug in a
//! player rather than a condition the game can continue from: the game
//! cannot vouch for a state built on it. Each panics with a message naming
//! the agent and the session. A point that merely arrived too late — for a
//! session that has closed, or a round that has passed — is not among
//! them. The action space a point is checked
//! against is [`roles::action_space`], the same function the role types
//! compute theirs with, so the game holds every player to exactly the rules
//! the roles apply to themselves, including the doctor's: for that the game
//! remembers whom each doctor protected last night, as the doctor does.

use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::clock::Timestamp;
use crate::event::AgentId;
use crate::werewolf::assignment::Assignment;
use crate::werewolf::config::Timing;
use crate::werewolf::message::{Cause, Narration, Outcome, Phase, Point, RequestKind, Round};
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
        to: BTreeSet<AgentId>,
        /// What they are told.
        narration: Narration,
    },
    /// Stop this agent: it is out of the game (ADR-0012).
    ///
    /// A dead player is stopped in the cycle its death is announced, and
    /// it is not among those told: there is nobody left to tell. Its
    /// reward is logged rather than said, so it needs to hear nothing to
    /// be paid.
    Stop {
        /// The agent out of the game.
        who: AgentId,
    },
}

/// One open pointing session: who may point, what they were asked, and the
/// clock it closes on (ADR-0011).
///
/// A member may point whenever it likes and as often as it likes; its most
/// recent point is its vote. The session closes when every member has
/// pointed and no point has changed for its quiet period, or at its hard
/// limit, whichever comes first. The day has no quiet period and closes on
/// a majority instead, so `quiet` is `None` for it.
#[derive(Debug, Clone)]
struct Session {
    /// What its members were asked.
    kind: RequestKind,
    /// Everyone asked, whether or not they have pointed.
    members: BTreeSet<AgentId>,
    /// Each member's latest target. A member that has not pointed is
    /// absent, which is how it abstains.
    points: BTreeMap<AgentId, AgentId>,
    /// The quiet period, for a night session; `None` for the day.
    quiet: Option<Duration>,
    /// The instant the session closes however its members behave.
    limit: Timestamp,
    /// When a member's target last changed, which is what the quiet period
    /// is measured from. The session's opening counts as the first change,
    /// so a session nobody points in still has somewhere to measure from.
    last_change: Timestamp,
}

impl Session {
    /// Records a point and says whether it changed anything.
    ///
    /// A repeat of the same target is not a change and does not restart
    /// the quiet period; a change of mind is and does.
    fn point(&mut self, from: &AgentId, target: &AgentId, at: Timestamp) -> bool {
        let changed = self.points.get(from) != Some(target);
        self.points.insert(from.clone(), target.clone());
        if changed {
            self.last_change = at;
        }
        changed
    }

    /// Whether every member has pointed at least once.
    fn settled(&self) -> bool {
        self.points.len() == self.members.len()
    }

    /// The earliest instant this session could close: its hard limit, or
    /// the end of its quiet period once every member has pointed.
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
    living: BTreeSet<AgentId>,
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
    resolved: Vec<(RequestKind, BTreeMap<AgentId, AgentId>)>,
    /// The clocks every session runs on.
    timing: Timing,
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
    pub fn new(assignment: Assignment, seed: u64, timing: Timing) -> Self {
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

    /// Records one player's point and returns whatever it caused.
    ///
    /// A member may point as often as it likes while its session is open,
    /// and its latest point is its vote. Pointing usually causes nothing:
    /// a night session closes on its clock, not on its last member, and
    /// the day closes only on the point that makes a majority.
    ///
    /// **A point for a session that has closed is ignored**, not a panic.
    /// It lost a race with the clock, which ADR-0011 makes an ordinary
    /// event rather than a bug: no session of that round and kind is open
    /// any more, and nothing comes of the point.
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
    /// player consulted to decide to point at all.
    pub fn point(&mut self, from: &AgentId, point: &Point, at: Timestamp) -> Vec<Directive> {
        let kind = point.kind;
        // A point names the session it was made in, so a point from a
        // round that has passed is one whose session closed while it was
        // in flight: a lost race, and ignored like any other (ADR-0011).
        // It is checked first, so that a stale point cannot be taken for
        // a current one just because a session of the same kind is open.
        if point.round != self.round {
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
                "{from} pointed in a {kind:?} session but is not in this game"
            );
            assert_eq!(
                self.role(from).asked_in(kind.phase()),
                Some(kind),
                "{from} pointed in a {kind:?} session, which a {} is never a member of",
                self.role(from)
            );
            return Vec::new();
        };
        let space = self.action_space_for(from, kind);
        assert!(
            space.contains(&point.target),
            "{from} pointed at {} in a {kind:?} session, which is outside its action space",
            point.target
        );
        self.sessions[index].point(from, &point.target, at);

        // The day ends the moment a majority of the living agree, and the
        // point that made it is the hammer. Everything else waits for a
        // clock.
        if kind == RequestKind::Nominate && self.majority().is_some() {
            // This point is the one that completed the majority, so the
            // player who made it is the hammer.
            return self.close_day(Some(from.clone()), at);
        }
        Vec::new()
    }

    /// Closes every session whose time is up at `now`, and resolves what
    /// that finishes.
    ///
    /// The moderator calls this whenever it wakes, and after every point,
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
    /// This is what the moderator hands [`Handler::deadline`](crate::Handler::deadline):
    /// the minimum over the open sessions of each one's hard limit and,
    /// for a night session whose members have all pointed, the end of its
    /// quiet period.
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

    /// The player more than half of the living are pointing at, if there
    /// is one.
    ///
    /// A majority of the *living*, not of those who have pointed: a
    /// village that mostly stays quiet does not lynch on two votes.
    fn majority(&self) -> Option<AgentId> {
        let session = self.sessions.first()?;
        let mut counts: BTreeMap<&AgentId, usize> = BTreeMap::new();
        for target in session.points.values() {
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
    pub fn living(&self) -> &BTreeSet<AgentId> {
        &self.living
    }

    /// The targets the rules permit `who` in a session of `kind`, from the
    /// game's own state.
    ///
    /// The same [`roles::action_space`] a player computes from its
    /// knowledge, so the two cannot disagree; this is the game's side of
    /// it, and what [`point`](Self::point) checks against.
    #[must_use]
    pub fn action_space_for(&self, who: &AgentId, kind: RequestKind) -> Vec<AgentId> {
        roles::action_space(who, &self.living, kind, self.last_protected.get(who))
    }

    /// Everyone dealt into the game, living and dead, in agent order.
    ///
    /// This is the roster the moderator starts and, when the game is over,
    /// stops. A player leaves the *game* when it is eliminated and the
    /// *episode* when it is stopped, and those are not the same moment.
    pub fn players(&self) -> impl Iterator<Item = &AgentId> {
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
    pub fn rewards(&self) -> Option<BTreeMap<AgentId, i32>> {
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
        // nowhere to point is not a member, so a session with no member
        // does not open.
        let mut members: BTreeMap<RequestKind, BTreeSet<AgentId>> = BTreeMap::new();
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
                points: BTreeMap::new(),
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

    /// Closes one night session: its tally to its observers, and the
    /// seer's finding if it looked at anybody.
    ///
    /// The tally marks the close, which is how the session's members and
    /// the transcript know that a point arriving later is late. The seer
    /// is told what it found here rather than at the end of the night, so
    /// that its own session's clock is the only one it waits on; a seer
    /// devoured the same night still learns what it learned.
    fn close_night_session(&mut self, session: &Session) -> Vec<Directive> {
        // To its members, and to nobody else; the moderator is not a
        // recipient of its own narrations.
        //
        // A session of one is not told its own tally. The point of a
        // tally is that a member sees where the *others* landed, which
        // is what lets a pack converge; told to a lone seer or doctor it
        // repeats the one thing that player already knows, having just
        // said it. The seer's close is marked by the finding below,
        // which says something it did not know; the doctor's is marked
        // by nothing, which is of a piece with a protection being
        // announced to nobody.
        let mut directives = Vec::new();
        if session.members.len() > 1 {
            directives.push(Directive::Narrate {
                to: session.members.clone(),
                narration: Narration::Tally {
                    round: self.round,
                    phase: Phase::Night,
                    kind: session.kind,
                    votes: session.points.clone(),
                    hammer: None,
                },
            });
        }
        for (who, target) in &session.points {
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
        self.resolved.push((session.kind, session.points.clone()));
        directives
    }

    /// Resolves the night once every session has closed: the victim is the
    /// plurality of the pack's latest points, unless the doctor's
    /// protection reached them first.
    fn resolve_night(&mut self, now: Timestamp) -> Vec<Directive> {
        let mut votes = BTreeMap::new();
        let mut protected = BTreeSet::new();
        for (kind, points) in std::mem::take(&mut self.resolved) {
            match kind {
                RequestKind::Devour => votes = points,
                RequestKind::Protect => protected.extend(points.into_values()),
                RequestKind::Investigate => {}
                RequestKind::Nominate => unreachable!("nobody nominates at night"),
            }
        }
        // No wolf pointed, no kill: the pack that cannot agree to act does
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

    /// Closes the day: the tally to the living, then the lynching that a
    /// majority called for, or `NoLynch` when the limit passed without one.
    fn close_day(&mut self, hammer: Option<AgentId>, now: Timestamp) -> Vec<Directive> {
        let session = self.sessions.remove(0);
        // The hammer is the point that made the majority, and the target
        // of that majority is who dies for it.
        let lynched = hammer
            .as_ref()
            .and_then(|who| session.points.get(who).cloned());
        let mut directives = vec![self.narrate_living(Narration::Tally {
            round: self.round,
            phase: Phase::Day,
            kind: RequestKind::Nominate,
            votes: session.points,
            hammer,
        })];
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
    fn eliminate(&mut self, who: &AgentId, cause: Cause) -> Vec<Directive> {
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
        if self.phase == Phase::Day && self.round.0 >= self.day_cap() {
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
                self.round = Round(self.round.0 + 1);
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
    targets: impl IntoIterator<Item = &'a AgentId>,
    ties: &mut ChaCha8Rng,
) -> Option<AgentId> {
    let mut counts: BTreeMap<&AgentId, usize> = BTreeMap::new();
    for who in targets {
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
    use crate::testing::{fast, id, ids, target, town, village};
    use crate::werewolf::role::Role::{Doctor, Seer, Villager, Werewolf};

    const SEED: u64 = 20_260_918;

    /// One phase of a script: every member's point, keyed by the agent
    /// making it, in the order they are to be recorded.
    type Answers = Vec<(&'static str, AgentId)>;

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
    fn asks(game: &Game) -> BTreeMap<AgentId, RequestKind> {
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

    /// Records one phase's points in the order given, then lets every
    /// session's clock run out, and returns everything that came of it.
    ///
    /// A day that reaches a majority closes on the point that made it, so
    /// the expiry that follows finds nothing left to close; a night always
    /// closes on its clocks. Either way the phase is over when this
    /// returns, which is what lets a script name one phase per entry.
    fn answer(game: &mut Game, answers: &Answers, at: Timestamp) -> Vec<Directive> {
        let asks = asks(game);
        assert_eq!(
            asks.keys().cloned().collect::<BTreeSet<_>>(),
            answers.iter().map(|(who, _)| id(who)).collect(),
            "a script points for exactly the members of the open sessions"
        );
        // The phase these points belong to, taken before any of them is
        // recorded: a day ends on the point that makes a majority, so
        // pointing alone may finish it.
        let phase = (game.phase, game.round);
        let round = game.round;
        let mut caused = Vec::new();
        for (who, chosen) in answers {
            let who = id(who);
            let point = Point {
                round,
                kind: asks[&who],
                target: chosen.clone(),
            };
            caused.extend(game.point(&who, &point, at));
        }
        // Close this phase and no more. If pointing already closed it,
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
    /// Each phase is pointed in at its own instant, far enough apart that
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
                round: Round(round),
                phase,
                living: ids(living),
            },
        )
    }

    /// The closing tally of one session, to the members it goes to.
    fn tally<const N: usize>(
        to: [&str; N],
        round: u32,
        kind: RequestKind,
        votes: &[(&'static str, &str)],
    ) -> Directive {
        narrate(
            to,
            Narration::Tally {
                round: Round(round),
                phase: kind.phase(),
                kind,
                hammer: None,
                votes: answers(votes)
                    .into_iter()
                    .map(|(who, action)| (id(who), action))
                    .collect(),
            },
        )
    }

    /// A day's closing tally, naming the hammer that ended it.
    fn day_tally<const N: usize>(
        to: [&str; N],
        round: u32,
        hammer: Option<&str>,
        votes: &[(&'static str, &str)],
    ) -> Directive {
        narrate(
            to,
            Narration::Tally {
                round: Round(round),
                phase: Phase::Day,
                kind: RequestKind::Nominate,
                hammer: hammer.map(id),
                votes: answers(votes)
                    .into_iter()
                    .map(|(who, action)| (id(who), action))
                    .collect(),
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
                round: Round(round),
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

    /// Records one point in the session of `kind` of the round now under
    /// way. A point says for itself which session it belongs to
    /// (ADR-0014), so there is no id to look up.
    fn respond(game: &mut Game, from: &str, kind: RequestKind, target: AgentId) -> Vec<Directive> {
        let round = game.round;
        let point = Point {
            round,
            kind,
            target,
        };
        game.point(&id(from), &point, at(0))
    }

    /// The same, for a point naming a round of the caller's choosing.
    fn respond_in(
        game: &mut Game,
        from: &str,
        round: u32,
        kind: RequestKind,
        target: AgentId,
    ) -> Vec<Directive> {
        let point = Point {
            round: Round(round),
            kind,
            target,
        };
        game.point(&id(from), &point, at(0))
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
                day_tally(
                    survivors,
                    1,
                    Some("erin"),
                    &[
                        ("bob", "carol"),
                        ("carol", "bob"),
                        ("dave", "bob"),
                        ("erin", "bob")
                    ],
                ),
                eliminated(survivors, "bob", Werewolf, 1, Cause::Lynched),
                stopped("bob"),
                outcome(Faction::Village, 1, ["carol", "dave", "erin"]),
            ]
        );
        assert_eq!(
            game.outcome(),
            Some(&Outcome {
                winner: Some(Faction::Village),
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
        // Five deals and the first PhaseBegan, and then the night: no
        // request stands between the phase and what the pointing caused
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
                // dave's point is the third of four living, a majority,
                // so it is the hammer and erin never points at all.
                day_tally(
                    ["alice", "bob", "dave", "erin"],
                    1,
                    Some("dave"),
                    &[("alice", "erin"), ("bob", "erin"), ("dave", "erin")],
                ),
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

    /// Plays a game to its end, pointing in every open session with a
    /// target drawn
    /// from the action space the rules compute, and returns the outcome
    /// and how many were living at the start of each round.
    ///
    /// The points are arbitrary, so nothing but the rules keeps the game
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
            let point_round = game.round;
            for (who, kind) in asked {
                let space =
                    roles::action_space(&who, &game.living, kind, game.last_protected.get(&who));
                let point = Point {
                    round: point_round,
                    kind,
                    target: pick(&mut moves, &space).clone(),
                };
                game.point(&who, &point, now);
            }
            // Every player points once and never changes its mind, so
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
                let dead: Vec<&AgentId> = rewards
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
        assert_eq!(outcome.rounds, Round(2), "it ended on the cap's day");
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
        assert_eq!(outcome.rounds, Round(1));
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
    fn a_game_of_players_that_all_point_ends() {
        // What termination rests on has changed. Under ADR-0004 a
        // `Nominate` could not abstain, so every day lynched somebody and
        // the living set strictly shrank: a game of n players was over by
        // round n whatever anybody did. Under ADR-0011 a day can end with
        // nobody lynched, so the living set may hold steady for a round,
        // and what bounds a game is the day cap — which is #69's, not
        // here yet.
        //
        // What still holds, and is what this checks, is that a game whose
        // players all point does end, and that the living set never
        // grows.
        for assignment in [village(), town(), pack_of_three()] {
            let players = assignment.players().count();
            for seed in 0..200 {
                let (outcome, living) = play_out(assignment.clone(), seed);
                assert!(
                    outcome.rounds.0 as usize <= players * 2,
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
    fn a_night_is_resolved_by_its_clocks_and_never_by_a_point() {
        // Under ADR-0004 the last answer resolved the phase. Under
        // ADR-0011 a night session closes a quiet period after its
        // members have settled, so pointing says nothing on its own — not
        // even the point that leaves nothing outstanding.
        let mut game = game(village());
        game.begin(at(0));
        let asks = asks(&game);
        for (who, whom) in [("bob", "alice"), ("carol", "bob"), ("dave", "erin")] {
            assert_eq!(
                respond(&mut game, who, asks[&id(who)], target(whom)),
                [],
                "{who}'s point resolved something"
            );
        }

        // Every member has pointed, so each session now closes a quiet
        // period after its last change rather than at its hard limit.
        let deadline = game.next_deadline().expect("the sessions are open");
        assert_eq!(deadline, at(0) + fast().pack.quiet);
        let caused = game.expire(deadline);
        // The village's pack is one wolf, so there is no tally to send:
        // a session of one is not told what it alone said. What the
        // close does produce is the seer's finding, then the death.
        assert_eq!(caused[0], investigated("carol", "bob", Faction::Werewolves));
        assert!(
            !caused.iter().any(|directive| matches!(
                directive,
                Directive::Narrate {
                    narration: Narration::Tally { .. },
                    ..
                }
            )),
            "no session of this night had two members: {caused:?}"
        );
    }

    #[test]
    fn points_in_any_order_settle_the_same_game() {
        // Under ADR-0011 the order of points is not reproducible and the
        // outcome is: a day closes on whoever completes the majority, so
        // reordering changes which player is the hammer and which later
        // points land inside the session at all. What it cannot change is
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
                Directive::Stop { .. } => None,
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
                tally(
                    ["alice", "bob", "carol"],
                    1,
                    RequestKind::Devour,
                    &[("alice", "dave"), ("bob", "erin"), ("carol", "frank")],
                ),
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
        // A member that never pointed is absent from the tally rather than
        // present with an abstention, so an empty tally is the only way a
        // plurality comes back empty (ADR-0011).
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
                narrate(everyone, Narration::NoDeath { round: Round(1) }),
                phase_began(1, Phase::Day, everyone),
            ]
        );
        assert_eq!(*game.living(), ids(everyone));
        // A save is never announced as one. What the doctor alone hears is
        // its deal and the tally that closes its own session; that it
        // protected the victim is told to nobody, so the village cannot
        // tell a save from a pack that pointed nowhere.
        for directive in &directives {
            if let Directive::Narrate { to, narration } = directive {
                if to.contains(&id("dave")) && to.len() < everyone.len() {
                    assert!(
                        matches!(
                            narration,
                            Narration::Assigned { .. }
                                | Narration::Tally {
                                    kind: RequestKind::Protect,
                                    ..
                                }
                        ),
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
                // The second seer is asked too, and points somewhere it is
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
        assert_eq!(game.round, Round(2));
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
        // no longer anything between the announcement and the pointing.
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
                    // A night session's tally closes that session and goes
                    // to its own members, so who may hear one depends on
                    // which session it is (ADR-0011).
                    Narration::Tally {
                        phase: Phase::Night,
                        kind,
                        ..
                    } => {
                        let role = match kind {
                            RequestKind::Devour => Werewolf,
                            RequestKind::Investigate => Seer,
                            RequestKind::Protect => Doctor,
                            RequestKind::Nominate => {
                                unreachable!("nobody nominates at night")
                            }
                        };
                        for who in to {
                            assert_eq!(
                                assignment.role(who),
                                Some(role),
                                "{who} hears a {kind:?} tally"
                            );
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
        expected = "carol pointed in a Devour session, which a Seer is never a member of"
    )]
    fn pointing_in_a_session_the_role_is_never_a_member_of_panics() {
        // There is no longer an id to get wrong, so the bug that was
        // "a request nobody was asked" and the bug that was "a request
        // asked of somebody else" are now one bug: pointing in a session
        // the rules never made you a member of (ADR-0014). A seer at
        // night belongs to the Investigate session and no other, and the
        // game can say so from the role alone.
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "carol", RequestKind::Devour, target("alice"));
    }

    #[test]
    #[should_panic(expected = "zara pointed in a Devour session but is not in this game")]
    fn pointing_from_outside_the_game_panics() {
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "zara", RequestKind::Devour, target("alice"));
    }

    #[test]
    fn a_point_for_a_closed_session_is_ignored() {
        // It lost a race with the clock, which ADR-0011 makes an ordinary
        // event rather than a bug. bob is a werewolf, so the Devour
        // session was genuinely its own: what is wrong with the point is
        // only that it is late.
        let mut game = game(village());
        game.begin(at(0));
        // The pack's session closes at its limit with nobody having
        // pointed, so bob has no session left to point in.
        let closed = game.expire(at(LATER));
        assert!(!closed.is_empty(), "the night's sessions closed");
        assert!(
            respond_in(&mut game, "bob", 1, RequestKind::Devour, target("alice")).is_empty(),
            "a point for a closed session causes nothing"
        );
    }

    #[test]
    fn a_point_naming_a_round_that_has_passed_is_ignored() {
        // A point carries its own round now, so a point in flight while
        // its phase ended names a round the game has left behind. That is
        // the same lost race, and is ignored rather than mistaken for a
        // point in the session of the same kind now open (ADR-0014).
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
        assert_eq!(game.round, Round(2), "the second night is under way");
        assert!(
            respond_in(&mut game, "bob", 1, RequestKind::Devour, target("erin")).is_empty(),
            "a point from round one causes nothing in round two"
        );
    }

    #[test]
    #[should_panic(
        expected = "dave pointed at dave in a Protect session, which is outside its action space"
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
        expected = "dave pointed at erin in a Protect session, which is outside its action space"
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
        // is the Protect session it belongs to, so the point names that
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
            assert_eq!(game.round, Round(3), "the third night is under way");
            assert_eq!(asks(&game).len(), 4, "four members point tonight");
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
        expected = "bob pointed at alice in a Nominate session, which is outside its action space"
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
    fn a_point_after_the_outcome_is_ignored() {
        // A player that pointed before its stop reached it has lost the
        // same race as a late point, and the game says nothing about it.
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
