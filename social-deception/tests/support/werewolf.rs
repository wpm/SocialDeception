//! Werewolf's own invariants: what every trajectory of a game of Werewolf
//! satisfies beyond what [`super::check`] asserts of any trajectory.
//!
//! The checks are written against the parsed lines, the way the runtime's
//! own checks are, and read the game the way the transcript reader does:
//! from the moderator's records, which are every narration and request it
//! sent and every response it received, in the order it recorded them. The
//! players' records are consulted only for what the moderator cannot vouch
//! for, which is what actually arrived at each of them.
//!
//! They fall into four groups:
//!
//! - **protocol**: every request is answered exactly once, by the agent it
//!   was asked of, with an action inside the action space the rules allowed
//!   it; the dead are neither asked nor heard from; players only ever
//!   address the moderator; and the moderator's last word is the outcome. A
//!   trajectory whose last narration is not an outcome is a truncated game,
//!   and it presents as a short file rather than as a hang, so the last of
//!   these is the only thing that catches it;
//! - **hidden information**: the realized observations stay inside each
//!   role's observation space. No message naming the pack ever reached a
//!   non-werewolf, no night tally either, and no finding reached anyone but
//!   the seer. Routing is the whole of the hidden-information mechanism, so
//!   these are what the design exists to guarantee;
//! - **broadcast discipline**: exactly one message reached a dead player,
//!   the outcome, which went to everyone because it is the reward signal;
//!   every other message is addressed to the living. On the first night the
//!   living are the whole roster, so a trajectory cannot tell a narration to
//!   the living from a broadcast by its recipient set alone; what it can
//!   tell is that nothing but the outcome ever reaches the dead;
//! - **game shape**: the phases alternate from the first night, each
//!   eliminates at most one player and each day exactly one, a night with no
//!   death says so and only then, the living set strictly shrinks every
//!   round, the winner is what the parity rule says of the final living set,
//!   and the roles dealt are the ones configured.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::Value;
use social_deception::AgentId;
use social_deception::werewolf::{
    Action, Cause, Config, Faction, Message, Narration, Outcome, Phase, Request, RequestId,
    RequestKind, Response, Role, Round,
};

/// Something the moderator said.
struct Said<'a> {
    seq: u64,
    to: BTreeSet<AgentId>,
    message: Message,
    line: &'a Value,
}

/// A response the moderator heard.
struct Heard<'a> {
    seq: u64,
    from: AgentId,
    response: Response,
    line: &'a Value,
}

/// A game as its moderator recorded it, with the facts every check needs
/// read out once: who holds which role, when each player was eliminated,
/// and how it ended.
struct Play<'a> {
    config: &'a Config,
    /// Everything the moderator said, in sequence order.
    said: Vec<Said<'a>>,
    /// Every response the moderator heard, in sequence order.
    heard: Vec<Heard<'a>>,
    /// Each player's role, from the `Assigned` narration it was sent.
    roles: BTreeMap<AgentId, Role>,
    /// The sequence number of the narration that eliminated each player.
    eliminated: BTreeMap<AgentId, u64>,
    /// How the game ended: the moderator's last word.
    outcome: Outcome,
}

/// Asserts everything a trajectory of a game played from `config` must
/// satisfy; see the [module documentation](self).
///
/// Run [`super::check`] first: these checks assume the moderator's records
/// are in sequence order and that every message names its recipients.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value], config: &Config) {
    let play = Play::read(lines, config);
    play.check_requests_and_responses();
    play.check_action_spaces();
    play.check_hidden_information();
    play.check_broadcasts();
    play.check_phases();
    play.check_outcome();
    play.check_players(lines);
}

/// The payload of a message record, as a [`Message`].
fn message(line: &Value) -> Message {
    serde_json::from_value(line["event"]["payload"].clone())
        .unwrap_or_else(|error| panic!("a payload is a werewolf message ({error}): {line}"))
}

/// The recipients of a message record.
fn recipients(line: &Value) -> BTreeSet<AgentId> {
    serde_json::from_value(line["event"]["recipients"].clone())
        .expect("a message lists its recipients")
}

/// The message records of `lines`, in file order.
fn messages(lines: &[Value]) -> impl Iterator<Item = &Value> {
    lines
        .iter()
        .filter(|line| line["type"] == "event" && line["event"]["kind"] == "message")
}

/// The message records of `agent` that crossed its boundary in the
/// direction `stamp` names, `"sent"` or `"arrived"`, in file order.
fn records_of<'a>(
    lines: &'a [Value],
    agent: &'a AgentId,
    stamp: &'a str,
) -> impl Iterator<Item = &'a Value> {
    messages(lines)
        .filter(move |line| super::agent(line) == agent.as_str() && !line[stamp].is_null())
}

/// The one recipient of a message addressed to a single player.
fn only<'a>(to: &'a BTreeSet<AgentId>, line: &Value) -> &'a AgentId {
    assert_eq!(to.len(), 1, "addressed to one player: {line}");
    to.first().unwrap()
}

/// The outcome a message carries, if it is one.
fn outcome(message: &Message) -> Option<&Outcome> {
    match message {
        Message::Narration(Narration::Outcome(outcome)) => Some(outcome),
        _ => None,
    }
}

impl<'a> Play<'a> {
    /// Reads the moderator's records out of `lines` and the facts the
    /// checks need out of them.
    ///
    /// Panics on anything that is not even the shape of a game: a moderator
    /// saying anything to a non-player or hearing anything but a response
    /// from one, a player assigned twice or eliminated twice, or a game
    /// without an outcome.
    fn read(lines: &'a [Value], config: &'a Config) -> Self {
        let players: BTreeSet<&AgentId> = config.players.iter().collect();
        let said: Vec<Said> = records_of(lines, &config.moderator, "sent")
            .map(|line| {
                let message = message(line);
                assert!(
                    !matches!(message, Message::Response(_)),
                    "the moderator only narrates and asks: {line}"
                );
                let to = recipients(line);
                assert!(
                    to.iter().all(|who| players.contains(who)),
                    "the moderator addresses only players: {line}"
                );
                Said {
                    seq: super::seq(line),
                    to,
                    message,
                    line,
                }
            })
            .collect();
        let heard: Vec<Heard> = records_of(lines, &config.moderator, "arrived")
            .map(|line| {
                let Message::Response(response) = message(line) else {
                    panic!("the moderator hears only responses: {line}");
                };
                let from = AgentId::new(line["event"]["sender"].as_str().unwrap());
                assert!(
                    players.contains(&from),
                    "the moderator hears only from players: {line}"
                );
                Heard {
                    seq: super::seq(line),
                    from,
                    response,
                    line,
                }
            })
            .collect();

        let mut roles = BTreeMap::new();
        let mut eliminated = BTreeMap::new();
        for Said {
            seq,
            to,
            message,
            line,
        } in &said
        {
            match message {
                Message::Narration(Narration::Assigned { role, .. }) => {
                    let who = only(to, line);
                    assert!(
                        roles.insert(who.clone(), *role).is_none(),
                        "a player is assigned once: {line}"
                    );
                }
                Message::Narration(Narration::Eliminated { who, .. }) => {
                    assert!(
                        eliminated.insert(who.clone(), *seq).is_none(),
                        "a player is eliminated once: {line}"
                    );
                }
                _ => {}
            }
        }
        assert_eq!(
            roles.keys().collect::<BTreeSet<_>>(),
            players,
            "every player, and nobody else, is assigned a role"
        );

        let last = said.last().expect("the moderator said something");
        let outcome = outcome(&last.message)
            .unwrap_or_else(|| {
                panic!(
                    "a trajectory without an outcome is a truncated game; the moderator's last \
                     word was {}",
                    last.line
                )
            })
            .clone();

        Self {
            config,
            said,
            heard,
            roles,
            eliminated,
            outcome,
        }
    }

    /// The role dealt to `who`.
    fn role(&self, who: &AgentId) -> Role {
        self.roles[who]
    }

    /// The players `role` was dealt to.
    fn holders(&self, role: Role) -> BTreeSet<AgentId> {
        self.roles
            .iter()
            .filter(|(_, held)| **held == role)
            .map(|(who, _)| who.clone())
            .collect()
    }

    /// Whether `who` had been eliminated before the record numbered `seq`.
    fn dead_at(&self, who: &AgentId, seq: u64) -> bool {
        self.eliminated.get(who).is_some_and(|&at| at < seq)
    }

    /// Everyone not yet eliminated at the record numbered `seq`.
    fn living_at(&self, seq: u64) -> BTreeSet<AgentId> {
        self.config
            .players
            .iter()
            .filter(|who| !self.dead_at(who, seq))
            .cloned()
            .collect()
    }

    /// Every narration the moderator sent.
    fn narrations(&self) -> impl Iterator<Item = (&Said<'a>, &Narration)> {
        self.said.iter().filter_map(|said| match &said.message {
            Message::Narration(narration) => Some((said, narration)),
            _ => None,
        })
    }

    /// Every request the moderator sent, by id, with the record it was
    /// sent in and the player it was asked of.
    fn requests(&self) -> BTreeMap<RequestId, (&Said<'a>, &AgentId, &Request)> {
        let mut requests = BTreeMap::new();
        for said in &self.said {
            if let Message::Request(request) = &said.message {
                let who = only(&said.to, said.line);
                assert!(
                    requests.insert(request.id, (said, who, request)).is_none(),
                    "a request id is issued once: {}",
                    said.line
                );
            }
        }
        requests
    }

    /// Every request has exactly one response, from the player it was asked
    /// of, echoing its id and arriving after it; no request goes to a dead
    /// player and no response comes from one.
    fn check_requests_and_responses(&self) {
        let requests = self.requests();
        let mut answered = BTreeSet::new();
        for heard in &self.heard {
            let line = heard.line;
            let Some((asked, who, _)) = requests.get(&heard.response.request) else {
                panic!("a response answers a request that was asked: {line}");
            };
            assert_eq!(
                *who, &heard.from,
                "a response comes from the player the request was asked of: {line}"
            );
            assert!(
                asked.seq < heard.seq,
                "a response arrives after its request was sent: {line}"
            );
            assert!(
                answered.insert(heard.response.request),
                "a request is answered once: {line}"
            );
            assert!(
                !self.dead_at(&heard.from, heard.seq),
                "no response arrives from a player after its elimination: {line}"
            );
        }
        for (id, (said, who, _)) in &requests {
            assert!(
                !self.dead_at(who, said.seq),
                "no request is asked of a player after its elimination: {}",
                said.line
            );
            assert!(
                answered.contains(id),
                "every request is answered, but this one was not: {}",
                said.line
            );
        }
    }

    /// Every response's action is in the action space its request allowed:
    /// a living player other than the responder, an abstention only where
    /// the request's kind permits one, and for the doctor never the player
    /// it protected the night before. And every request asks a player what
    /// its role is asked in that phase.
    fn check_action_spaces(&self) {
        let requests = self.requests();
        for (said, who, request) in requests.values() {
            assert_eq!(
                self.role(who).asked_in(request.kind.phase()),
                Some(request.kind),
                "a request asks a player what its role is asked: {}",
                said.line
            );
        }
        let mut last_protected: BTreeMap<&AgentId, &Action> = BTreeMap::new();
        for heard in &self.heard {
            let line = heard.line;
            let (asked, _, request) = requests[&heard.response.request];
            let action = &heard.response.action;
            match action {
                Action::Abstain => assert!(
                    request.kind.may_abstain(),
                    "an abstention is in the action space of Protect and Investigate only: {line}"
                ),
                Action::Target(target) => {
                    assert_ne!(
                        target, &heard.from,
                        "no action targets the player taking it: {line}"
                    );
                    assert!(
                        self.living_at(asked.seq).contains(target),
                        "an action targets a living player: {line}"
                    );
                }
            }
            if request.kind == RequestKind::Protect {
                if let Some(last) = last_protected.insert(&heard.from, action) {
                    assert!(
                        last.target().is_none() || last != action,
                        "the doctor never protects the same player two nights running: {line}"
                    );
                }
            }
        }
    }

    /// The realized observations stay inside each role's observation
    /// space: the pack is named only to werewolves, a night's tally goes
    /// only to werewolves, and a finding goes only to the seer.
    fn check_hidden_information(&self) {
        let werewolves = self.holders(Role::Werewolf);
        let seers = self.holders(Role::Seer);
        for (said, narration) in self.narrations() {
            let (to, line) = (&said.to, said.line);
            match narration {
                Narration::Assigned { pack, .. } if !pack.is_empty() => {
                    assert!(
                        to.is_subset(&werewolves),
                        "no message naming the pack is addressed to a non-werewolf: {line}"
                    );
                    assert_eq!(pack, &werewolves, "the pack named is the pack: {line}");
                }
                Narration::Assigned { role, .. } => {
                    assert_ne!(*role, Role::Werewolf, "a werewolf is told its pack: {line}");
                }
                Narration::Tally {
                    phase: Phase::Night,
                    votes,
                    ..
                } => {
                    assert!(
                        to.is_subset(&werewolves),
                        "no night tally is addressed to a non-werewolf: {line}"
                    );
                    assert!(
                        votes.keys().all(|who| werewolves.contains(who)),
                        "a night tally is the werewolves' votes: {line}"
                    );
                }
                Narration::Investigated { .. } => {
                    assert!(
                        to.is_subset(&seers),
                        "no finding is addressed to anyone but the seer: {line}"
                    );
                }
                _ => {}
            }
        }
    }

    /// The outcome goes to every player, living and dead, and is announced
    /// once; everything else the moderator says is addressed to the living.
    fn check_broadcasts(&self) {
        let everyone: BTreeSet<AgentId> = self.config.players.iter().cloned().collect();
        let mut outcomes = 0;
        for said in &self.said {
            let line = said.line;
            if outcome(&said.message).is_some() {
                assert_eq!(
                    said.to, everyone,
                    "the outcome goes to everyone, living and dead: {line}"
                );
                outcomes += 1;
            } else {
                assert!(
                    said.to.is_subset(&self.living_at(said.seq)),
                    "narration is addressed to the living, never to the dead: {line}"
                );
            }
        }
        assert_eq!(outcomes, 1, "the outcome is announced once");
    }

    /// The phases run Night 1, Day 1, Night 2, Day 2, ... from the first
    /// night; each begins with the living as they are and asks only during
    /// itself; a night has one tally and then one death or one `NoDeath`; a
    /// day has one tally and then one death; a death reveals the role and
    /// names the phase's cause; and the living strictly shrink from one
    /// round to the next.
    fn check_phases(&self) {
        let mut phases = Phases {
            play: self,
            current: None,
            counts: PhaseCounts::default(),
            living_last_night: None,
        };
        for said in &self.said {
            match &said.message {
                Message::Narration(Narration::PhaseBegan {
                    round,
                    phase,
                    living,
                }) => phases.began(said, *round, *phase, living),
                Message::Narration(narration) => phases.narrated(said, narration),
                Message::Request(request) => phases.asked(said, request),
                Message::Response(_) => unreachable!("the moderator sends no responses"),
            }
        }
        phases.counts.close(phases.current);
    }

    /// The outcome names the living as they are, and the winner the parity
    /// rule gives for them: the village if no werewolf lives, the
    /// werewolves if they are at least as many as everyone else, and nobody
    /// only at the round cap.
    fn check_outcome(&self) {
        let outcome = &self.outcome;
        assert_eq!(
            outcome.living,
            self.living_at(u64::MAX),
            "the outcome names the survivors: {outcome:?}"
        );
        let werewolves = outcome
            .living
            .iter()
            .filter(|who| self.role(who).faction() == Faction::Werewolves)
            .count();
        let others = outcome.living.len() - werewolves;
        let winner = if werewolves == 0 {
            Some(Faction::Village)
        } else if werewolves >= others {
            Some(Faction::Werewolves)
        } else {
            None
        };
        assert_eq!(
            outcome.winner, winner,
            "the winner is what the parity rule says of the survivors: {outcome:?}"
        );
        assert!(
            outcome.rounds.0 <= self.config.max_rounds,
            "the game ends by the round cap: {outcome:?}"
        );
        if winner.is_none() {
            assert_eq!(
                outcome.rounds.0, self.config.max_rounds,
                "a stalemate happens only at the round cap: {outcome:?}"
            );
        }
    }

    /// What the players' own records show: each sent nothing but responses,
    /// to the moderator alone; each received its role exactly once, the one
    /// the moderator dealt it, and the outcome as the last thing; and a
    /// dead player received nothing between its own death and the outcome.
    /// And the roles dealt are the ones configured.
    fn check_players(&self, lines: &[Value]) {
        for who in &self.config.players {
            for line in records_of(lines, who, "sent") {
                assert!(
                    matches!(message(line), Message::Response(_))
                        && recipients(line) == BTreeSet::from([self.config.moderator.clone()]),
                    "a player sends only responses, to the moderator alone: {line}"
                );
            }
            let arrived: Vec<Message> = records_of(lines, who, "arrived").map(message).collect();
            let assigned: Vec<&Role> = arrived
                .iter()
                .filter_map(|message| match message {
                    Message::Narration(Narration::Assigned { role, .. }) => Some(role),
                    _ => None,
                })
                .collect();
            assert_eq!(
                assigned,
                [&self.role(who)],
                "{who} is assigned its role exactly once"
            );
            let last = arrived
                .last()
                .unwrap_or_else(|| panic!("{who} received nothing"));
            assert_eq!(
                outcome(last),
                Some(&self.outcome),
                "the last thing {who} received is the outcome"
            );
            let death = arrived.iter().position(|message| {
                matches!(message, Message::Narration(Narration::Eliminated { who: dead, .. }) if dead == who)
            });
            match death {
                Some(death) => assert_eq!(
                    arrived.len(),
                    death + 2,
                    "a dead player receives nothing between its death and the outcome: {who}"
                ),
                None => assert!(
                    self.outcome.living.contains(who),
                    "a player never eliminated survives: {who}"
                ),
            }
        }
        let counts = &self.config.roles;
        let villagers = self.config.players.len() - counts.special();
        for (role, count) in [
            (Role::Werewolf, counts.werewolves),
            (Role::Seer, counts.seers),
            (Role::Doctor, counts.doctors),
            (Role::Villager, villagers),
        ] {
            assert_eq!(
                self.holders(role).len(),
                count,
                "the roles dealt are the ones configured: {role}"
            );
        }
    }
}

/// A walk over the moderator's records phase by phase.
struct Phases<'p, 'a> {
    play: &'p Play<'a>,
    /// The phase in progress: its round and phase, and the record that
    /// began it.
    current: Option<(Round, Phase, &'p Value)>,
    /// What has been narrated in it so far.
    counts: PhaseCounts,
    /// The living when the most recent night began.
    living_last_night: Option<BTreeSet<AgentId>>,
}

impl<'p, 'a> Phases<'p, 'a> {
    /// A phase began: it is the one after the last, its living are the
    /// living, and the last phase was complete.
    fn began(
        &mut self,
        said: &'p Said<'a>,
        round: Round,
        phase: Phase,
        living: &BTreeSet<AgentId>,
    ) {
        let line = said.line;
        self.counts.close(self.current);
        self.counts = PhaseCounts::default();
        let expected = match self.current {
            None => (Round(1), Phase::Night),
            Some((round, Phase::Night, _)) => (round, Phase::Day),
            Some((Round(round), Phase::Day, _)) => (Round(round + 1), Phase::Night),
        };
        assert_eq!(
            (round, phase),
            expected,
            "phases alternate from the first night: {line}"
        );
        assert_eq!(
            *living,
            self.play.living_at(said.seq),
            "a phase begins with the living as they are: {line}"
        );
        if phase == Phase::Night {
            if let Some(before) = &self.living_last_night {
                assert!(
                    living.is_subset(before) && living.len() < before.len(),
                    "the living strictly shrink every round: {line}"
                );
            }
            self.living_last_night = Some(living.clone());
        }
        self.current = Some((round, phase, line));
    }

    /// A narration other than a phase beginning: a role before the first
    /// phase, anything else within one.
    fn narrated(&mut self, said: &Said<'a>, narration: &Narration) {
        let line = said.line;
        if let Narration::Assigned { .. } = narration {
            assert!(
                self.current.is_none(),
                "roles are assigned before the first phase: {line}"
            );
            return;
        }
        let Some((round, phase, _)) = self.current else {
            panic!("nothing but a role is narrated before the first phase: {line}");
        };
        self.counts.count(narration, round, phase, line);
        if let Narration::Eliminated {
            who, role, cause, ..
        } = narration
        {
            assert_eq!(
                *role,
                self.play.role(who),
                "a death reveals the role: {line}"
            );
            let expected = match phase {
                Phase::Night => Cause::Devoured,
                Phase::Day => Cause::Lynched,
            };
            assert_eq!(*cause, expected, "the cause is the phase's: {line}");
        }
    }

    /// A request: asked within the phase it belongs to.
    fn asked(&self, said: &Said<'a>, request: &Request) {
        let line = said.line;
        let Some((round, phase, _)) = self.current else {
            panic!("a request is asked within a phase: {line}");
        };
        assert_eq!(
            (request.round, request.kind.phase()),
            (round, phase),
            "a request belongs to the phase it is asked in: {line}"
        );
    }
}

/// What has been narrated so far in one phase.
#[derive(Default)]
struct PhaseCounts {
    tallies: usize,
    eliminated: usize,
    no_death: usize,
}

impl PhaseCounts {
    /// Counts one narration of the phase `(round, phase)`, checking it
    /// belongs to that phase.
    fn count(&mut self, narration: &Narration, round: Round, phase: Phase, line: &Value) {
        match narration {
            Narration::Tally {
                round: r, phase: p, ..
            } => {
                assert_eq!(
                    (*r, *p),
                    (round, phase),
                    "a tally belongs to its phase: {line}"
                );
                self.tallies += 1;
            }
            Narration::Eliminated { round: r, .. } => {
                assert_eq!(*r, round, "a death belongs to its round: {line}");
                self.eliminated += 1;
            }
            Narration::NoDeath { round: r } => {
                assert_eq!(*r, round, "a quiet night belongs to its round: {line}");
                assert_eq!(phase, Phase::Night, "only a night has no death: {line}");
                self.no_death += 1;
            }
            Narration::Investigated { .. } => {
                assert_eq!(
                    phase,
                    Phase::Night,
                    "the seer investigates at night: {line}"
                );
            }
            Narration::Outcome(_) | Narration::Assigned { .. } | Narration::PhaseBegan { .. } => {}
        }
    }

    /// Checks the counts of a phase that has ended, if one had begun.
    fn close(&self, phase: Option<(Round, Phase, &Value)>) {
        let Some((_, phase, line)) = phase else {
            return;
        };
        assert_eq!(self.tallies, 1, "a phase has one tally: {line}");
        match phase {
            Phase::Night => assert_eq!(
                self.eliminated + self.no_death,
                1,
                "a night has one death or one NoDeath, never both or neither: {line}"
            ),
            Phase::Day => {
                assert_eq!(self.eliminated, 1, "a day eliminates exactly one: {line}");
                assert_eq!(self.no_death, 0, "only a night has no death: {line}");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    /// The fixture: a seven-player game played to a village win, with its
    /// effective config beside it.
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.jsonl"
    ));
    const EFFECTIVE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.jsonl.toml"
    ));

    /// Every player in the fixture.
    const EVERYONE: [&str; 7] = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];

    fn fixture() -> Vec<Value> {
        super::super::parse(FIXTURE.as_bytes())
    }

    fn config() -> Config {
        Config::parse(EFFECTIVE).unwrap()
    }

    /// The index of the first message record of `agent` in the direction
    /// `stamp` whose payload satisfies `wanted`.
    fn find(lines: &[Value], agent: &str, stamp: &str, wanted: impl Fn(&Value) -> bool) -> usize {
        lines
            .iter()
            .position(|line| {
                line["type"] == "event"
                    && line["agent"] == agent
                    && !line[stamp].is_null()
                    && wanted(&line["event"]["payload"])
            })
            .expect("the fixture has such a record")
    }

    /// A narration of the given kind.
    fn narration(kind: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| !payload["Narration"][kind].is_null()
    }

    /// The narration that began the given phase.
    fn began(round: u32, phase: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| {
            let began = &payload["Narration"]["PhaseBegan"];
            began["round"] == round && began["phase"] == phase
        }
    }

    /// The request with the given id.
    fn request(id: u64) -> impl Fn(&Value) -> bool {
        move |payload| payload["Request"]["id"] == id
    }

    /// The response to the request with the given id.
    fn response(id: u64) -> impl Fn(&Value) -> bool {
        move |payload| payload["Response"]["request"] == id
    }

    /// The fixture with `edit` applied to the record `find` names.
    fn edited(
        agent: &str,
        stamp: &str,
        wanted: impl Fn(&Value) -> bool,
        edit: impl FnOnce(&mut Value),
    ) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, stamp, wanted);
        edit(&mut lines[index]);
        lines
    }

    /// The fixture with `edit` applied to something the moderator said.
    fn said(wanted: impl Fn(&Value) -> bool, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        edited("moderator", "sent", wanted, edit)
    }

    /// The fixture with `edit` applied to a response the moderator heard.
    fn heard(wanted: impl Fn(&Value) -> bool, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        edited("moderator", "arrived", wanted, edit)
    }

    /// The fixture without the record `find` names.
    fn without(agent: &str, stamp: &str, wanted: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, stamp, wanted);
        lines.remove(index);
        lines
    }

    /// The fixture with the record `find` names repeated, right after
    /// itself.
    fn doubled(agent: &str, stamp: &str, wanted: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, stamp, wanted);
        lines.insert(index, lines[index].clone());
        lines
    }

    fn recipients(line: &mut Value, to: &[&str]) {
        line["event"]["recipients"] = json!(to);
    }

    fn sender(line: &mut Value, from: &str) {
        line["event"]["sender"] = json!(from);
    }

    fn target(line: &mut Value, whom: &str) {
        line["event"]["payload"]["Response"]["action"] = json!({"Target": whom});
    }

    #[test]
    fn the_fixture_passes() {
        check(&fixture(), &config());
    }

    #[test]
    #[should_panic(expected = "truncated game")]
    fn a_game_without_an_outcome_is_caught() {
        check(
            &without("moderator", "sent", narration("Outcome")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "addresses only players")]
    fn a_message_to_a_stranger_is_caught() {
        let lines = said(narration("Assigned"), |line| recipients(line, &["zed"]));
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a response comes from the player the request was asked of")]
    fn a_response_from_the_wrong_player_is_caught() {
        check(&heard(response(5), |line| sender(line, "bob")), &config());
    }

    #[test]
    #[should_panic(expected = "a response answers a request that was asked")]
    fn a_response_to_nothing_is_caught() {
        let lines = heard(response(5), |line| {
            line["event"]["payload"]["Response"]["request"] = json!(99);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request is answered once")]
    fn a_request_answered_twice_is_caught() {
        check(&doubled("moderator", "arrived", response(5)), &config());
    }

    #[test]
    #[should_panic(expected = "every request is answered")]
    fn an_unanswered_request_is_caught() {
        check(&without("moderator", "arrived", response(5)), &config());
    }

    #[test]
    #[should_panic(expected = "no request is asked of a player after its elimination")]
    fn a_request_to_the_dead_is_caught() {
        // alice was lynched on day 1; request 16 is bob's nomination on day 2.
        let mut lines = said(request(16), |line| recipients(line, &["alice"]));
        lines.remove(find(&lines, "moderator", "arrived", response(16)));
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no response arrives from a player after its elimination")]
    fn a_response_from_the_dead_is_caught() {
        let mut lines = said(request(16), |line| recipients(line, &["alice"]));
        let index = find(&lines, "moderator", "arrived", response(16));
        sender(&mut lines[index], "alice");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request asks a player what its role is asked")]
    fn a_request_of_the_wrong_kind_is_caught() {
        // Request 2 asks dave, a werewolf, to devour; bob is a villager.
        let mut lines = said(request(2), |line| recipients(line, &["bob"]));
        let index = find(&lines, "moderator", "arrived", response(2));
        sender(&mut lines[index], "bob");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no action targets the player taking it")]
    fn a_self_target_is_caught() {
        check(&heard(response(5), |line| target(line, "alice")), &config());
    }

    #[test]
    #[should_panic(expected = "an action targets a living player")]
    fn a_dead_target_is_caught() {
        check(
            &heard(response(16), |line| target(line, "alice")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "an abstention is in the action space of Protect and Investigate")]
    fn an_abstention_from_nominating_is_caught() {
        let lines = heard(response(5), |line| {
            line["event"]["payload"]["Response"]["action"] = json!("Abstain");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the doctor never protects the same player two nights running")]
    fn a_repeated_protection_is_caught() {
        // Requests 1 and 12 are carol's protections on nights 1 and 2.
        let mut lines = heard(response(1), |line| target(line, "bob"));
        let index = find(&lines, "moderator", "arrived", response(12));
        target(&mut lines[index], "bob");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no message naming the pack is addressed to a non-werewolf")]
    fn a_pack_named_to_a_villager_is_caught() {
        let lines = said(narration("Assigned"), |line| {
            line["event"]["payload"]["Narration"]["Assigned"]["pack"] = json!(["dave", "erin"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a werewolf is told its pack")]
    fn a_werewolf_kept_from_its_pack_is_caught() {
        let werewolf = |payload: &Value| payload["Narration"]["Assigned"]["role"] == "Werewolf";
        let lines = said(werewolf, |line| {
            line["event"]["payload"]["Narration"]["Assigned"]["pack"] = json!([]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no night tally is addressed to a non-werewolf")]
    fn a_night_tally_sent_to_a_villager_is_caught() {
        let lines = said(narration("Tally"), |line| {
            recipients(line, &["alice", "dave", "erin"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no finding is addressed to anyone but the seer")]
    fn a_finding_sent_to_a_villager_is_caught() {
        let lines = said(narration("Investigated"), |line| {
            recipients(line, &["alice", "grace"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "never to the dead")]
    fn a_narration_to_the_dead_is_caught() {
        let lines = said(began(2, "Night"), |line| recipients(line, &EVERYONE));
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome goes to everyone")]
    fn an_outcome_kept_from_a_dead_player_is_caught() {
        let lines = said(narration("Outcome"), |line| {
            recipients(line, &["bob", "carol", "dave", "erin", "frank", "grace"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome is announced once")]
    fn an_outcome_announced_twice_is_caught() {
        check(
            &doubled("moderator", "sent", narration("Outcome")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "phases alternate from the first night")]
    fn a_phase_out_of_order_is_caught() {
        let lines = said(began(2, "Night"), |line| {
            line["event"]["payload"]["Narration"]["PhaseBegan"]["round"] = json!(3);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a phase begins with the living as they are")]
    fn a_phase_that_miscounts_the_living_is_caught() {
        let lines = said(began(2, "Night"), |line| {
            line["event"]["payload"]["Narration"]["PhaseBegan"]["living"] = json!(EVERYONE);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request belongs to the phase it is asked in")]
    fn a_request_from_another_round_is_caught() {
        let lines = said(request(5), |line| {
            line["event"]["payload"]["Request"]["round"] = json!(2);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night has one death or one NoDeath")]
    fn a_night_that_says_nothing_of_deaths_is_caught() {
        check(
            &without("moderator", "sent", narration("NoDeath")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a day eliminates exactly one")]
    fn a_day_without_a_lynching_is_caught() {
        check(
            &without("moderator", "sent", narration("Eliminated")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a day eliminates exactly one")]
    fn a_day_with_two_lynchings_is_caught() {
        // Day 3 lynches dave; a second death the same day.
        let lynched = |payload: &Value| payload["Narration"]["Eliminated"]["who"] == "dave";
        let mut lines = doubled("moderator", "sent", lynched);
        let index = find(&lines, "moderator", "sent", lynched);
        lines[index]["event"]["payload"]["Narration"]["Eliminated"] =
            json!({"who": "bob", "role": "Villager", "round": 3, "cause": "Lynched"});
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the cause is the phase's")]
    fn a_devouring_by_day_is_caught() {
        let lines = said(narration("Eliminated"), |line| {
            line["event"]["payload"]["Narration"]["Eliminated"]["cause"] = json!("Devoured");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a death reveals the role")]
    fn a_death_revealing_the_wrong_role_is_caught() {
        let lines = said(narration("Eliminated"), |line| {
            line["event"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Seer");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the winner is what the parity rule says")]
    fn a_winner_against_the_parity_rule_is_caught() {
        let lines = said(narration("Outcome"), |line| {
            line["event"]["payload"]["Narration"]["Outcome"]["winner"] = json!("Werewolves");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome names the survivors")]
    fn an_outcome_that_miscounts_the_survivors_is_caught() {
        let lines = said(narration("Outcome"), |line| {
            line["event"]["payload"]["Narration"]["Outcome"]["living"] = json!(["bob"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the roles dealt are the ones configured")]
    fn a_deal_that_disagrees_with_the_configuration_is_caught() {
        let mut config = config();
        config.roles.doctors = 0;
        check(&fixture(), &config);
    }

    #[test]
    #[should_panic(expected = "alice is assigned its role exactly once")]
    fn a_player_assigned_twice_is_caught() {
        check(
            &doubled("alice", "arrived", narration("Assigned")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "the last thing alice received is the outcome")]
    fn a_player_that_never_hears_the_outcome_is_caught() {
        check(
            &without("alice", "arrived", narration("Outcome")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a dead player receives nothing between its death and the outcome")]
    fn a_dead_player_that_hears_more_is_caught() {
        // Bob's copy of the day 2 tally, delivered to the dead alice too.
        let mut lines = fixture();
        let tally = |payload: &Value| {
            payload["Narration"]["Tally"]["round"] == 2
                && payload["Narration"]["Tally"]["phase"] == "Day"
        };
        let mut leaked = lines[find(&lines, "bob", "arrived", tally)].clone();
        leaked["agent"] = json!("alice");
        let outcome = find(&lines, "alice", "arrived", narration("Outcome"));
        lines.insert(outcome, leaked);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a player sends only responses, to the moderator alone")]
    fn a_player_addressing_another_player_is_caught() {
        let lines = edited("alice", "sent", response(5), |line| {
            recipients(line, &["bob", "moderator"]);
        });
        check(&lines, &config());
    }
}
