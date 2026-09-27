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
//! then opens the first night. A phase opens one session per kind of
//! request it calls for, and every member of a session is asked at once. A
//! member may point whenever it likes while its session is open and may
//! change its mind; its most recent point is its vote, and pointing nowhere
//! is how it abstains (ADR-0011). A player the rules leave nothing to point
//! at is not asked at all, and no request ever goes to a dead player.
//!
//! **A night is three sessions at once**, each with a clock of its own so
//! that a slow role cannot spend another's time: the pack devours, the seer
//! investigates, the doctor protects. Each closes when every member has
//! pointed and no point has changed for its quiet period — any change
//! restarts it — or at its hard limit, whichever comes first. A session's
//! closing tally goes to its own members and marks the close. The seer is
//! told its finding when its own session closes, so a seer devoured that
//! same night still learns what it learned.
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
//! A day may now end without a lynch, so the living set no longer shrinks
//! every round and nothing here bounds a game on its own. What bounds it is
//! the day cap in the configuration's `[timing]` table, after which a game
//! nobody has won ends as a stalemate.
//!
//! # Player bugs are panics
//!
//! A point for a request the game never asked, or asked of somebody else,
//! or naming a target outside the request's action space, is a bug in a
//! player rather than a condition the game can continue from: the game
//! cannot vouch for a state built on it. Each panics with a message naming
//! the agent and the request. A point that merely arrived too late is not
//! among them. The action space a response is checked
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
use crate::werewolf::message::{
    Cause, Narration, Outcome, Phase, Point, Request, RequestId, RequestKind, Round,
};
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
    /// A request for one agent to act.
    Ask {
        /// The agent asked.
        to: AgentId,
        /// What it is asked.
        request: Request,
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
    /// How many requests have been issued; the next one's id is one more.
    issued: u64,
    /// The requests of the sessions now open: who was asked, and what. A
    /// request leaves this when its session closes, which is what makes a
    /// point arriving afterwards late rather than a bug.
    outstanding: BTreeMap<RequestId, (AgentId, RequestKind)>,
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
            issued: 0,
            outstanding: BTreeMap::new(),
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
        assert_eq!(self.issued, 0, "the game has already begun");
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
    /// event rather than a bug: the request is no longer outstanding, and
    /// nothing comes of the point.
    ///
    /// # Panics
    ///
    /// If the point names a request that was never asked or was asked of
    /// another agent, or a target outside the request's action space as
    /// [`roles::action_space`] computes it: the player itself, somebody
    /// not living, or, for a doctor, the player it protected the night
    /// before. Each is a bug in a player rather than a lost race.
    pub fn point(&mut self, from: &AgentId, point: &Point, at: Timestamp) -> Vec<Directive> {
        let RequestId(id) = point.request;
        let Some((to, kind)) = self.outstanding.get(&point.request) else {
            // Either the session closed or the game ended. Every request
            // this game ever issued is its own, so an id it has never
            // issued is the one thing left that is a bug.
            assert!(
                point.request.0 <= self.issued,
                "{from} pointed for request {id}, which was never asked"
            );
            return Vec::new();
        };
        assert_eq!(
            to, from,
            "{from} pointed for request {id}, which was asked of {to}"
        );
        let kind = *kind;
        let space = self.action_space_for(from, kind);
        assert!(
            space.contains(&point.target),
            "{from} pointed at {} in request {id}, which is outside its action space for {kind:?}",
            point.target
        );
        let session = self
            .sessions
            .iter_mut()
            .find(|session| session.kind == kind)
            .expect("an outstanding request belongs to an open session");
        session.point(from, &point.target, at);

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

    /// The targets the rules permit `who` for a request of `kind`, from the
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
                    let value = if role.faction() == winner { 1 } else { -1 };
                    (who.clone(), value)
                })
                .collect(),
        )
    }

    /// Announces the current phase to the living and issues its requests,
    /// in agent order.
    fn begin_phase(&mut self, now: Timestamp) -> Vec<Directive> {
        let mut directives = vec![self.narrate_living(Narration::PhaseBegan {
            round: self.round,
            phase: self.phase,
            living: self.living.clone(),
        })];
        // Who is asked what, and which session each belongs to. A player
        // with an empty action space is not asked at all, so a session
        // with no member to ask does not open.
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
            for who in &members {
                directives.push(self.ask(who.clone(), kind));
            }
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

    /// Closes one night session: its tally to its observers, and the
    /// seer's finding if it looked at anybody.
    ///
    /// The tally marks the close, which is how the session's members and
    /// the transcript know that a point arriving later is late. The seer
    /// is told what it found here rather than at the end of the night, so
    /// that its own session's clock is the only one it waits on; a seer
    /// devoured the same night still learns what it learned.
    fn close_night_session(&mut self, session: &Session) -> Vec<Directive> {
        self.outstanding
            .retain(|_, (who, kind)| !(*kind == session.kind && session.members.contains(who)));
        // To its members and, for the pack, to nobody else; the moderator
        // is not a recipient of its own narrations.
        let mut directives = vec![Directive::Narrate {
            to: session.members.clone(),
            narration: Narration::Tally {
                round: self.round,
                phase: Phase::Night,
                kind: session.kind,
                votes: session.points.clone(),
                hammer: None,
            },
        }];
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
                directives.push(self.eliminate(&victim, Cause::Devoured));
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
        self.outstanding.clear();
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
            directives.push(self.eliminate(&who, Cause::Lynched));
            directives.extend(self.advance(now));
        } else {
            directives.push(self.narrate_living(Narration::NoLynch { round: self.round }));
            directives.extend(self.next_phase(now));
        }
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
    fn advance(&mut self, now: Timestamp) -> Vec<Directive> {
        match self.winner() {
            Some(winner) => vec![self.end(winner)],
            None => self.next_phase(now),
        }
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

    /// One phase of a script: every request's answer, keyed by the agent
    /// asked, in the order the answers are to be recorded.
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
    /// Records one phase's points in the order given, then lets every
    /// session's clock run out, and returns everything that came of it.
    ///
    /// A day that reaches a majority closes on the point that made it, so
    /// the expiry that follows finds nothing left to close; a night always
    /// closes on its clocks. Either way the phase is over when this
    /// returns, which is what lets a script name one phase per entry.
    fn answer(
        game: &mut Game,
        asked: &[Directive],
        answers: &Answers,
        at: Timestamp,
    ) -> Vec<Directive> {
        let asks = asks(asked);
        assert_eq!(
            asks.keys().cloned().collect::<BTreeSet<_>>(),
            answers.iter().map(|(who, _)| id(who)).collect(),
            "a script answers exactly the requests issued"
        );
        // The phase these answers belong to, taken before any of them is
        // recorded: a day ends on the point that makes a majority, so
        // pointing alone may finish it.
        let phase = (game.phase, game.round);
        let mut caused = Vec::new();
        for (who, chosen) in answers {
            let who = id(who);
            let point = Point {
                request: asks[&who].id,
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
        let mut latest = all.clone();
        for (index, phase) in script.iter().enumerate() {
            let now = at((index as u64 + 1) * LATER);
            latest = answer(game, &latest, phase, now);
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

    /// Records one point for the request with the given id.
    fn respond(game: &mut Game, from: &str, request: u64, target: AgentId) -> Vec<Directive> {
        let point = Point {
            request: RequestId(request),
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
                ask("bob", 1, 1, RequestKind::Devour),
                ask("carol", 2, 1, RequestKind::Investigate),
                ask("dave", 3, 1, RequestKind::Protect),
                tally(["bob"], 1, RequestKind::Devour, &[("bob", "alice")]),
                tally(["carol"], 1, RequestKind::Investigate, &[("carol", "bob")]),
                investigated("carol", "bob", Faction::Werewolves),
                tally(["dave"], 1, RequestKind::Protect, &[("dave", "erin")]),
                eliminated(everyone, "alice", Villager, 1, Cause::Devoured),
                phase_began(1, Phase::Day, survivors),
                ask("bob", 4, 1, RequestKind::Nominate),
                ask("carol", 5, 1, RequestKind::Nominate),
                ask("dave", 6, 1, RequestKind::Nominate),
                ask("erin", 7, 1, RequestKind::Nominate),
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
                tally(["bob"], 1, RequestKind::Devour, &[("bob", "carol")]),
                tally(["carol"], 1, RequestKind::Investigate, &[("carol", "bob")]),
                // The seer is devoured tonight and still learns what it
                // learned, because its own session closed before the
                // night resolved.
                investigated("carol", "bob", Faction::Werewolves),
                tally(["dave"], 1, RequestKind::Protect, &[("dave", "alice")]),
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
                // No seer lives, so the second night asks nothing of one.
                phase_began(2, Phase::Night, ["alice", "bob", "dave"]),
                ask("bob", 8, 2, RequestKind::Devour),
                ask("dave", 9, 2, RequestKind::Protect),
                tally(["bob"], 2, RequestKind::Devour, &[("bob", "alice")]),
                tally(["dave"], 2, RequestKind::Protect, &[("dave", "bob")]),
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
        let (game, _, living) = played(assignment, seed);
        (game.outcome().unwrap().clone(), living)
    }

    /// The same, giving back the game itself, whatever else is wanted of
    /// it once it has ended.
    fn played(assignment: Assignment, seed: u64) -> (Game, Outcome, Vec<usize>) {
        let mut game = Game::new(assignment, seed, fast());
        let mut moves = ChaCha8Rng::seed_from_u64(seed);
        let mut clock = 0;
        let mut latest = game.begin(at(clock));
        let mut living = vec![game.living.len()];
        let mut round = game.round;
        while game.outcome().is_none() {
            if game.round != round {
                round = game.round;
                living.push(game.living.len());
            }
            let asked = asks(&latest);
            assert!(!asked.is_empty(), "a running game always asks something");
            clock += LATER;
            let now = at(clock);
            let mut caused = Vec::new();
            for (who, request) in asked {
                let space = roles::action_space(
                    &who,
                    &game.living,
                    request.kind,
                    game.last_protected.get(&who),
                );
                let point = Point {
                    request: request.id,
                    target: pick(&mut moves, &space).clone(),
                };
                caused.extend(game.point(&who, &point, now));
            }
            // Every player points once and never changes its mind, so
            // running the clock out is what closes the phase.
            caused.extend(game.expire(at(clock + LATER)));
            latest = caused;
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
                    let expected = if role.faction() == outcome.winner {
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
        let asks = asks(&game.begin(at(0)));
        let request = |who: &str| asks[&id(who)].id.0;
        for (who, whom) in [("bob", "alice"), ("carol", "bob"), ("dave", "erin")] {
            assert_eq!(
                respond(&mut game, who, request(who), target(whom)),
                [],
                "{who}'s point resolved something"
            );
        }

        // Every member has pointed, so each session now closes a quiet
        // period after its last change rather than at its hard limit.
        let deadline = game.next_deadline().expect("the sessions are open");
        assert_eq!(deadline, at(0) + fast().pack.quiet);
        let caused = game.expire(deadline);
        assert_eq!(
            caused[0],
            tally(["bob"], 1, RequestKind::Devour, &[("bob", "alice")])
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
            assert_eq!(game.outcome().unwrap().winner, Faction::Village);
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
                Directive::Ask { .. } => None,
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
        assert_eq!(
            directives[11..],
            [
                tally(
                    ["alice", "bob", "carol"],
                    1,
                    RequestKind::Devour,
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
            directives[9..],
            [
                tally(["bob"], 1, RequestKind::Devour, &[("bob", "erin")]),
                tally(["carol"], 1, RequestKind::Investigate, &[("carol", "bob")]),
                // The seer looked at bob and found the pack.
                investigated("carol", "bob", Faction::Werewolves),
                tally(["dave"], 1, RequestKind::Protect, &[("dave", "erin")]),
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
    fn no_request_is_issued_for_a_missing_seer_or_doctor() {
        let mut game = game(pack_of_three());
        let asked = asks(&game.begin(at(0)));
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
        // The second night opens with the phase and the requests it asks,
        // and dave, the dead doctor, is asked nothing: two requests where
        // the first night had three.
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
            directives[second_night..second_night + 3],
            [
                phase_began(2, Phase::Night, ["alice", "bob", "carol"]),
                ask("bob", 8, 2, RequestKind::Devour),
                ask("carol", 9, 2, RequestKind::Investigate),
            ]
        );
        // Nothing else is asked of that night, so dave was not asked.
        assert!(
            !directives[second_night..].iter().any(|directive| matches!(
                directive,
                Directive::Ask { to, request }
                    if *to == id("dave") && request.round == Round(2)
            )),
            "the dead doctor was asked something"
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
        let mut latest = game.begin(at(0));
        let mut sizes = vec![game.living().len()];
        for (index, phase) in werewolves_win().iter().enumerate() {
            latest = answer(&mut game, &latest, phase, at((index as u64 + 1) * LATER));
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
    #[should_panic(expected = "bob pointed for request 99, which was never asked")]
    fn an_id_the_game_never_issued_panics() {
        // A request the game never asked is still a bug in a player: it
        // cannot be a session that closed, because no session ever had it.
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "bob", 99, target("alice"));
    }

    #[test]
    fn a_point_for_a_closed_session_is_ignored() {
        // It lost a race with the clock, which ADR-0011 makes an ordinary
        // event rather than a bug.
        let mut game = game(village());
        game.begin(at(0));
        // The pack's session closes at its limit with nobody having
        // pointed, so bob's request is no longer outstanding.
        let closed = game.expire(at(LATER));
        assert!(!closed.is_empty(), "the night's sessions closed");
        assert!(
            respond(&mut game, "bob", 1, target("alice")).is_empty(),
            "a point for a closed session causes nothing"
        );
    }

    #[test]
    #[should_panic(expected = "carol pointed for request 1, which was asked of bob")]
    fn a_request_id_asked_of_another_agent_panics() {
        let mut game = game(village());
        game.begin(at(0));
        respond(&mut game, "carol", 1, target("alice"));
    }

    #[test]
    #[should_panic(
        expected = "dave pointed at dave in request 3, which is outside its action space for Protect"
    )]
    fn a_doctor_protecting_itself_panics() {
        let mut game = game(village());
        game.begin(at(0));
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
        expected = "dave pointed at erin in request 9, which is outside its action space for Protect"
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
        // In the hamlet, dave protects erin on the first night, then
        // carol on the second, and on the third may protect erin again:
        // the constraint is last night's protection alone. The requests of
        // the third night are 23 to 26, and dave's is 25.
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
            // The request the third night asked dave, whatever number it
            // fell on: the ids move when the sessions do, and this test is
            // about the doctor's constraint rather than about counting.
            let asked = game
                .outstanding
                .iter()
                .find(|(_, (who, kind))| *who == id("dave") && *kind == RequestKind::Protect)
                .map(|(request, _)| request.0)
                .expect("the third night asked dave to protect");
            respond(&mut game, "dave", asked, target("erin"));
        }
    }

    #[test]
    #[should_panic(
        expected = "bob pointed at alice in request 4, which is outside its action space for Nominate"
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
    fn a_point_after_the_outcome_is_ignored() {
        // A player that pointed before its stop reached it has lost the
        // same race as a late point, and the game says nothing about it.
        let mut game = game(village());
        play(&mut game, &village_wins());
        assert!(game.outcome().is_some());
        assert!(respond(&mut game, "carol", 5, target("dave")).is_empty());
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
