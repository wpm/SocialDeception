//! Werewolf's own invariants: what every trajectory of a game of Werewolf
//! satisfies beyond what [`super::check`] asserts of any trajectory.
//!
//! The checks are written against the parsed lines, the way the runtime's
//! own checks are, and read the game the way the transcript reader does:
//! from the moderator's records, whose `action` records are every narration
//! and request it sent and whose `observation` records are every response it
//! received, in the order it recorded them. The players' records are
//! consulted only for what the moderator cannot vouch for, which is what
//! actually reached each of them.
//!
//! They fall into four groups:
//!
//! - **protocol**: every request is answered exactly once, by the agent it
//!   was asked of, with an action inside the action space the rules allowed
//!   it; the dead are neither asked nor heard from; players only ever
//!   address the moderator; and the moderator's last word is the outcome. A
//!   trajectory whose last narration is not an outcome is a game the
//!   moderator never ended, which the episode also catches as a stall;
//! - **hidden information**: the realized observations stay inside each
//!   role's observation space. Every message goes to exactly the players
//!   the rules address it to: the pack is named only to werewolves, a
//!   night's tally goes to the werewolves who cast it, a finding goes to the
//!   seer alone, and a narration to the living goes to exactly the living.
//!   Routing is the whole of the hidden-information mechanism, so these are
//!   what the design exists to guarantee;
//! - **the episode's shape**: every trajectory, the moderator's included,
//!   begins with a `Start` control and ends with a `Stop`, because the
//!   moderator is the episode's environment and starting and stopping the
//!   players is its doing. Nothing at all reaches a dead player after the
//!   announcement of its own death: no message of this game is broadcast,
//!   and the outcome, which ADR-0004 once excepted, is narrated to the
//!   living like everything else;
//! - **the rewards**: every player has exactly one, +1 exactly when the
//!   role it was dealt belongs to the winning faction and −1 otherwise,
//!   living or dead, and logged before the `Stop` that ends its
//!   trajectory. The moderator has none: it plays no game, so there is
//!   nothing its behavior could be worth;
//! - **game shape**: the phases alternate from the first night, each
//!   eliminates at most one player and each day exactly one, the player
//!   eliminated is one the phase's tally names most often, a night with no
//!   death is one on which the doctor protected such a player and only
//!   then, the living set strictly shrinks every round, the game ends
//!   within as many rounds as there are players, the winner is what the
//!   parity rule says of the final living set, and the roles dealt are the
//!   ones configured.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;
use social_deception::AgentId;
use social_deception::werewolf::{
    Assignment, Cause, Config, Faction, Message, Move, Narration, Outcome, Phase, Request,
    RequestId, RequestKind, Response, Role, Round,
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
/// read out once: who holds which role, which request is which, when each
/// player was eliminated, and how it ended.
struct Play<'a> {
    config: &'a Config,
    /// Everything the moderator said, in sequence order.
    said: Vec<Said<'a>>,
    /// Every response the moderator heard, in sequence order.
    heard: Vec<Heard<'a>>,
    /// Where in `said` each request was asked.
    requests: BTreeMap<RequestId, usize>,
    /// Each player's role, from the `Assigned` narration it was sent.
    assignment: Assignment,
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
    play.check_phases();
    play.check_recipients();
    play.check_hidden_information();
    play.check_outcome();
    play.check_players(lines);
    // The episode's shape first: a reward is checked against the `Stop`
    // that ends its agent's trajectory, so "everybody was stopped" should
    // fail by its own name rather than as a missing stop to compare with.
    check_episode(lines, config);
    play.check_rewards(lines);
}

/// Every agent's trajectory, the moderator's included, begins with a `Start`
/// control and ends with a `Stop`, and nobody is started or stopped twice.
///
/// It is the moderator that sends both, being the episode's environment, so
/// this is the check that the game's own shutdown happened: a trajectory
/// whose players were stopped by the episode picking up the pieces would
/// look the same here, but one where somebody was never stopped at all
/// would not.
fn check_episode(lines: &[Value], config: &Config) {
    let everybody = config
        .players
        .iter()
        .chain([&config.moderator])
        .cloned()
        .collect::<BTreeSet<AgentId>>();
    for who in &everybody {
        let controls: Vec<&str> = lines
            .iter()
            .filter(|line| line["type"] == "control" && super::agent(line) == who.as_str())
            .map(|line| line["control"].as_str().expect("a control names itself"))
            .collect();
        assert_eq!(
            controls,
            ["start", "stop"],
            "{who}'s trajectory begins with a start and ends with a stop"
        );
    }
    let agents: BTreeSet<AgentId> = lines
        .iter()
        .map(|line| AgentId::new(super::agent(line)))
        .collect();
    assert_eq!(
        agents, everybody,
        "the roster is every player and the moderator, and nobody else"
    );
}

/// The payload of a message record, as a [`Message`].
fn message(line: &Value) -> Message {
    Message::deserialize(&line["event"]["payload"])
        .unwrap_or_else(|error| panic!("a payload is a werewolf message ({error}): {line}"))
}

/// The recipients of a message record.
fn recipients(line: &Value) -> BTreeSet<AgentId> {
    BTreeSet::deserialize(&line["event"]["recipients"]).expect("a message lists its recipients")
}

/// The records of `agent` of the given type, `"action"` for what it sent or
/// `"observation"` for what it received, in file order.
fn records_of<'a>(
    lines: &'a [Value],
    agent: &'a AgentId,
    kind: &'a str,
) -> impl Iterator<Item = &'a Value> {
    lines
        .iter()
        .filter(move |line| line["type"] == kind && super::agent(line) == agent.as_str())
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

/// The players a tally names most often: the ones the elimination is
/// drawn from. Abstentions name nobody.
fn leaders(votes: &BTreeMap<AgentId, Move>) -> BTreeSet<AgentId> {
    let mut counts: BTreeMap<&AgentId, usize> = BTreeMap::new();
    for who in votes.values().filter_map(Move::target) {
        *counts.entry(who).or_default() += 1;
    }
    let most = counts.values().copied().max().unwrap_or(0);
    counts
        .into_iter()
        .filter(|(_, count)| *count == most)
        .map(|(who, _)| who.clone())
        .collect()
}

impl<'a> Play<'a> {
    /// Reads the moderator's records out of `lines` and the facts the
    /// checks need out of them.
    ///
    /// Panics on anything that is not even the shape of a game: a moderator
    /// saying anything to a non-player or hearing anything but a response
    /// from one, a request issued twice, a player assigned twice or
    /// eliminated twice, or a game without an outcome.
    fn read(lines: &'a [Value], config: &'a Config) -> Self {
        let players: BTreeSet<&AgentId> = config.players.iter().collect();
        let said: Vec<Said> = records_of(lines, &config.moderator, "action")
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
        let heard: Vec<Heard> = records_of(lines, &config.moderator, "observation")
            .map(|line| {
                let Message::Response(response) = message(line) else {
                    panic!("the moderator hears only responses: {line}");
                };
                let from = AgentId::deserialize(&line["event"]["sender"]).unwrap();
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

        let mut requests = BTreeMap::new();
        let mut roles = Vec::new();
        let mut eliminated = BTreeMap::new();
        for (index, said) in said.iter().enumerate() {
            let line = said.line;
            match &said.message {
                Message::Request(request) => {
                    only(&said.to, line);
                    assert!(
                        requests.insert(request.id, index).is_none(),
                        "a request id is issued once: {line}"
                    );
                }
                Message::Narration(Narration::Assigned { role, .. }) => {
                    roles.push((only(&said.to, line).clone(), *role));
                }
                Message::Narration(Narration::Eliminated { who, .. }) => {
                    assert!(
                        eliminated.insert(who.clone(), said.seq).is_none(),
                        "a player is eliminated once: {line}"
                    );
                }
                _ => {}
            }
        }
        let assignment = Assignment::new(roles);
        assert_eq!(
            assignment
                .players()
                .map(|(who, _)| who)
                .collect::<BTreeSet<_>>(),
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
            requests,
            assignment,
            eliminated,
            outcome,
        }
    }

    /// The role dealt to `who`.
    fn role(&self, who: &AgentId) -> Role {
        self.assignment
            .role(who)
            .unwrap_or_else(|| panic!("{who} is not a player"))
    }

    /// The players `role` was dealt to.
    fn holders(&self, role: Role) -> BTreeSet<AgentId> {
        self.assignment
            .players()
            .filter(|(_, held)| *held == role)
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

    /// The request with the given id, if one was asked: the record it was
    /// asked in, the player it was asked of, and the request.
    fn asked(&self, id: RequestId) -> Option<(&Said<'a>, &AgentId, &Request)> {
        let said = &self.said[*self.requests.get(&id)?];
        match &said.message {
            Message::Request(request) => Some((said, said.to.first().unwrap(), request)),
            _ => unreachable!("`requests` indexes only requests"),
        }
    }

    /// Every request the moderator asked, in id order.
    fn requests(&self) -> impl Iterator<Item = (&Said<'a>, &AgentId, &Request)> {
        self.requests.keys().map(|&id| self.asked(id).unwrap())
    }

    /// No request goes to a dead player, and every request has exactly one
    /// response, from the player it was asked of, echoing its id, arriving
    /// after it, and before that player's elimination.
    fn check_requests_and_responses(&self) {
        for (said, who, _) in self.requests() {
            assert!(
                !self.dead_at(who, said.seq),
                "no request is asked of a player after its elimination: {}",
                said.line
            );
        }
        let mut answered = BTreeSet::new();
        for heard in &self.heard {
            let line = heard.line;
            let Some((asked, who, _)) = self.asked(heard.response.request) else {
                panic!("a response answers a request that was asked: {line}");
            };
            assert_eq!(
                *who, heard.from,
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
        for (said, _, request) in self.requests() {
            assert!(
                answered.contains(&request.id),
                "every request is answered, but this one was not: {}",
                said.line
            );
        }
    }

    /// Every request asks a player what its role is asked in that phase,
    /// and every response's action is in the action space its request
    /// allowed: a living player other than the responder, an abstention
    /// only where the request's kind permits one, and for the doctor never
    /// the player it protected the night before.
    fn check_action_spaces(&self) {
        for (said, who, request) in self.requests() {
            assert_eq!(
                self.role(who).asked_in(request.kind.phase()),
                Some(request.kind),
                "a request asks a player what its role is asked: {}",
                said.line
            );
        }
        let mut last_protected: BTreeMap<&AgentId, &Move> = BTreeMap::new();
        for heard in &self.heard {
            let line = heard.line;
            let (asked, _, request) = self.asked(heard.response.request).unwrap();
            let chosen = &heard.response.chosen;
            match chosen {
                Move::Abstain => assert!(
                    request.kind.may_abstain(),
                    "an abstention is outside the action space of {:?}: {line}",
                    request.kind
                ),
                Move::Target(target) => {
                    assert_ne!(
                        target, &heard.from,
                        "no move targets the player taking it: {line}"
                    );
                    assert!(
                        self.assignment.role(target).is_some() && !self.dead_at(target, asked.seq),
                        "a move targets a living player: {line}"
                    );
                }
            }
            if request.kind == RequestKind::Protect {
                if let Some(last) = last_protected.insert(&heard.from, chosen) {
                    assert!(
                        last.target().is_none() || last != chosen,
                        "the doctor never protects the same player two nights running: {line}"
                    );
                }
            }
        }
    }

    /// Every message goes to exactly the players the rules address it to.
    /// A request and a role assignment go to one player, which `read`
    /// checked. A finding goes to the seer alone; a night tally to the
    /// werewolves who cast it; and a phase, a day tally, a death, a quiet
    /// night and the outcome to the living.
    ///
    /// The outcome is in the last group, not a group of its own: the
    /// broadcast exception ADR-0004 made for it is withdrawn, so it is a
    /// narration to the living like any other and the only thing left to
    /// say about it is that it happens once.
    fn check_recipients(&self) {
        let seers = self.holders(Role::Seer);
        let mut outcomes = 0;
        for said in &self.said {
            let line = said.line;
            match &said.message {
                Message::Request(_) | Message::Narration(Narration::Assigned { .. }) => {}
                Message::Narration(Narration::Investigated { .. }) => {
                    assert_eq!(said.to, seers, "a finding goes to the seer alone: {line}");
                }
                Message::Narration(Narration::Tally {
                    phase: Phase::Night,
                    votes,
                    ..
                }) => assert_eq!(
                    said.to,
                    votes.keys().cloned().collect(),
                    "a night tally goes to the werewolves who cast it, and nobody else: {line}"
                ),
                Message::Narration(
                    Narration::Tally { .. }
                    | Narration::PhaseBegan { .. }
                    | Narration::Eliminated { .. }
                    | Narration::NoDeath { .. },
                ) => assert_eq!(
                    said.to,
                    self.living_at(said.seq),
                    "a narration to the living goes to exactly the living: {line}"
                ),
                Message::Narration(Narration::Outcome(_)) => {
                    assert_eq!(
                        said.to,
                        self.living_at(said.seq),
                        "the outcome goes to exactly the living: {line}"
                    );
                    outcomes += 1;
                }
                Message::Response(_) => unreachable!("the moderator sends no responses"),
            }
        }
        assert_eq!(outcomes, 1, "the outcome is announced once");
    }

    /// What the addressed messages say stays inside the recipient's
    /// observation space: a werewolf is told the pack and nobody else is
    /// told anything of it, and a night tally is the werewolves' votes.
    fn check_hidden_information(&self) {
        for said in &self.said {
            let line = said.line;
            match &said.message {
                Message::Narration(Narration::Assigned { role, pack }) => {
                    if *role == Role::Werewolf {
                        assert_eq!(
                            pack,
                            self.assignment.pack(),
                            "a werewolf is told its pack: {line}"
                        );
                    } else {
                        assert!(
                            pack.is_empty(),
                            "no message naming the pack is addressed to a non-werewolf: {line}"
                        );
                    }
                }
                Message::Narration(Narration::Tally {
                    phase: Phase::Night,
                    votes,
                    ..
                }) => {
                    assert!(
                        votes.keys().all(|who| self.assignment.pack().contains(who)),
                        "a night tally is the werewolves' votes: {line}"
                    );
                }
                _ => {}
            }
        }
    }

    /// The phases run Night 1, Day 1, Night 2, Day 2, ... from the first
    /// night; each begins with the living as they are and asks only during
    /// itself; a night has one tally and then one death or one `NoDeath`; a
    /// day has one tally and then one death; the player eliminated is one
    /// the tally names most, unprotected at night, and a quiet night is one
    /// on which the doctor protected such a player; a death reveals the
    /// role and names the phase's cause; and the living strictly shrink
    /// from one round to the next.
    fn check_phases(&self) {
        let protected: BTreeMap<Round, &AgentId> = self
            .heard
            .iter()
            .filter_map(|heard| {
                let (_, _, request) = self.asked(heard.response.request)?;
                match (request.kind, heard.response.chosen.target()) {
                    (RequestKind::Protect, Some(target)) => Some((request.round, target)),
                    _ => None,
                }
            })
            .collect();
        let mut phases = Phases {
            play: self,
            protected,
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
    /// werewolves if they are at least as many as everyone else. A game
    /// always reaches one of the two, within as many rounds as there are
    /// players.
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
        assert!(
            werewolves == 0 || werewolves >= others,
            "the game ends only once the parity rule decides it: {outcome:?}"
        );
        let winner = if werewolves == 0 {
            Faction::Village
        } else {
            Faction::Werewolves
        };
        assert_eq!(
            outcome.winner, winner,
            "the winner is what the parity rule says of the survivors: {outcome:?}"
        );
        assert!(
            outcome.rounds.0 as usize <= self.config.players.len(),
            "the game ends within as many rounds as there are players: {outcome:?}"
        );
    }

    /// Every player has exactly one reward, worth +1 if the role it was
    /// dealt belongs to the winning faction and −1 if it does not, logged
    /// at or before the `Stop` that ends its trajectory. The moderator has
    /// none.
    ///
    /// The roles come from the `Assigned` narrations the moderator sent,
    /// which is the same place every other check here gets them, so the
    /// claim is that the reward agrees with the game as it was actually
    /// dealt and not merely with itself. There are no stalemates
    /// (ADR-0007), so every player is paid one of the two and never zero.
    fn check_rewards(&self, lines: &[Value]) {
        let rewards: BTreeMap<AgentId, Vec<&Value>> = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .fold(BTreeMap::new(), |mut paid, line| {
                let who = AgentId::new(super::agent(line));
                paid.entry(who).or_default().push(line);
                paid
            });
        for who in &self.config.players {
            let paid = rewards
                .get(who)
                .unwrap_or_else(|| panic!("{who} has a reward"));
            assert_eq!(paid.len(), 1, "{who} has exactly one reward: {paid:?}");
            let line = paid[0];
            let expected = if self.role(who).faction() == self.outcome.winner {
                1
            } else {
                -1
            };
            assert_eq!(
                line["value"].as_i64(),
                Some(expected),
                "{who} held {} and {:?} won: {line}",
                self.role(who),
                self.outcome.winner
            );
        }
        assert!(
            !rewards.contains_key(&self.config.moderator),
            "the moderator plays no game and is paid nothing: {:?}",
            rewards.get(&self.config.moderator)
        );
        assert_eq!(
            rewards.len(),
            self.config.players.len(),
            "the players, and nobody else, are paid"
        );
    }

    /// What the players' own records show: each took no action but a
    /// response, to the moderator alone; each observed its role exactly
    /// once, the one the moderator dealt it; a survivor's last observation
    /// is the outcome; and a dead player observed nothing at all after the
    /// announcement of its own death. And the roles dealt are the ones
    /// configured.
    fn check_players(&self, lines: &[Value]) {
        let moderator = BTreeSet::from([self.config.moderator.clone()]);
        for who in &self.config.players {
            for line in records_of(lines, who, "action") {
                assert!(
                    matches!(message(line), Message::Response(_)) && recipients(line) == moderator,
                    "a player sends only responses, to the moderator alone: {line}"
                );
            }
            let received: Vec<Message> =
                records_of(lines, who, "observation").map(message).collect();
            let assigned: Vec<&Role> = received
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
            let last = received
                .last()
                .unwrap_or_else(|| panic!("{who} received nothing"));
            let death = received.iter().position(|message| {
                matches!(message, Message::Narration(Narration::Eliminated { who: dead, .. }) if dead == who)
            });
            if let Some(death) = death {
                assert_eq!(
                    received.len(),
                    death + 1,
                    "a dead player observes nothing after its own death: {who}"
                );
                assert!(
                    !self.outcome.living.contains(who),
                    "a player told it was eliminated is not among the survivors: {who}"
                );
            } else {
                assert!(
                    self.outcome.living.contains(who),
                    "a player never eliminated survives: {who}"
                );
                assert_eq!(
                    outcome(last),
                    Some(&self.outcome),
                    "the last thing the survivor {who} received is the outcome"
                );
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
                self.assignment.count(role),
                count,
                "the roles dealt are the ones configured: {role}"
            );
        }
    }
}

/// A walk over the moderator's records phase by phase.
struct Phases<'p, 'a> {
    play: &'p Play<'a>,
    /// Whom the doctor protected each round, from its responses.
    protected: BTreeMap<Round, &'p AgentId>,
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
        let protected = self.protected.get(&round).copied();
        match narration {
            Narration::Eliminated {
                who, role, cause, ..
            } => {
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
                assert!(
                    self.counts.leaders.contains(who),
                    "an elimination is of a player the tally names most: {line}"
                );
                if phase == Phase::Night {
                    assert_ne!(
                        protected,
                        Some(who),
                        "a death at night is of a player the doctor did not protect: {line}"
                    );
                }
            }
            Narration::NoDeath { .. } => {
                assert!(
                    protected.is_some_and(|who| self.counts.leaders.contains(who)),
                    "a night with no death is one on which the doctor protected a player the \
                     pack chose: {line}"
                );
            }
            _ => {}
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
    /// The players the phase's tally named most, once it has been narrated.
    leaders: BTreeSet<AgentId>,
}

impl PhaseCounts {
    /// Counts one narration of the phase `(round, phase)`, checking it
    /// belongs to that phase.
    fn count(&mut self, narration: &Narration, round: Round, phase: Phase, line: &Value) {
        match narration {
            Narration::Tally {
                round: r,
                phase: p,
                votes,
            } => {
                assert_eq!(
                    (*r, *p),
                    (round, phase),
                    "a tally belongs to its phase: {line}"
                );
                self.tallies += 1;
                self.leaders = leaders(votes);
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
    use serde_json::{Map, json};

    use super::*;

    /// The fixture: a seven-player game played to a werewolf win in two
    /// rounds, with its effective config beside it. Its first night is a
    /// saved one, so the fixture's first `Eliminated` is a lynching, and it
    /// ends by parity rather than by the pack being wiped out. Every
    /// request in it is answered with a target: it holds no abstention.
    const FIXTURE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.jsonl"
    ));
    const EFFECTIVE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/werewolf.jsonl.toml"
    ));

    fn fixture() -> Vec<Value> {
        super::super::parse(FIXTURE.as_bytes())
    }

    fn config() -> Config {
        Config::parse(EFFECTIVE).unwrap()
    }

    /// Every player in the fixture, as the recipients of a message.
    fn everyone() -> Vec<String> {
        config()
            .players
            .iter()
            .map(|who| who.as_str().to_owned())
            .collect()
    }

    /// The index of the first record of `agent` of the given type whose
    /// payload satisfies `wanted`.
    fn find(lines: &[Value], agent: &str, kind: &str, wanted: impl Fn(&Value) -> bool) -> usize {
        lines
            .iter()
            .position(|line| {
                line["type"] == kind && line["agent"] == agent && wanted(&line["event"]["payload"])
            })
            .expect("the fixture has such a record")
    }

    /// A narration of the given kind.
    fn narration(kind: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| !payload["Narration"][kind].is_null()
    }

    /// A role assignment to a player of the given role, or of any other
    /// role when `to` is false.
    fn assigned(role: &str, to: bool) -> impl Fn(&Value) -> bool + '_ {
        move |payload| {
            let assigned = &payload["Narration"]["Assigned"];
            !assigned.is_null() && (assigned["role"] == role) == to
        }
    }

    /// The narration that began the given phase.
    fn began(round: u32, phase: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| {
            let began = &payload["Narration"]["PhaseBegan"];
            began["round"] == round && began["phase"] == phase
        }
    }

    /// The tally of the given phase.
    fn tally(round: u32, phase: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| {
            let tally = &payload["Narration"]["Tally"];
            tally["round"] == round && tally["phase"] == phase
        }
    }

    /// An elimination by the day's vote.
    fn lynched(payload: &Value) -> bool {
        payload["Narration"]["Eliminated"]["cause"] == "Lynched"
    }

    /// The elimination of the given player.
    fn eliminated(who: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| payload["Narration"]["Eliminated"]["who"] == who
    }

    /// The request with the given id.
    fn request(id: u64) -> impl Fn(&Value) -> bool {
        move |payload| payload["Request"]["id"] == id
    }

    /// The response to the request with the given id.
    fn response(id: u64) -> impl Fn(&Value) -> bool {
        move |payload| payload["Response"]["request"] == id
    }

    /// The id of the request of `kind` asked of `who` in `round`.
    fn asked(who: &str, round: u32, kind: &str) -> u64 {
        let lines = fixture();
        let index = find(&lines, who, "observation", |payload| {
            payload["Request"]["round"] == round && payload["Request"]["kind"] == kind
        });
        lines[index]["event"]["payload"]["Request"]["id"]
            .as_u64()
            .unwrap()
    }

    /// The fixture with `edit` applied to the record `find` names.
    fn edited(
        agent: &str,
        kind: &str,
        wanted: impl Fn(&Value) -> bool,
        edit: impl FnOnce(&mut Value),
    ) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, kind, wanted);
        edit(&mut lines[index]);
        lines
    }

    /// The fixture with `edit` applied to something the moderator said.
    fn said(wanted: impl Fn(&Value) -> bool, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        edited(config().moderator.as_str(), "action", wanted, edit)
    }

    /// The fixture with `edit` applied to a response the moderator heard.
    fn heard(wanted: impl Fn(&Value) -> bool, edit: impl FnOnce(&mut Value)) -> Vec<Value> {
        edited(config().moderator.as_str(), "observation", wanted, edit)
    }

    /// The fixture without the record `find` names.
    fn without(agent: &str, kind: &str, wanted: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, kind, wanted);
        lines.remove(index);
        lines
    }

    /// The fixture with the record `find` names repeated, right after
    /// itself.
    fn doubled(agent: &str, kind: &str, wanted: impl Fn(&Value) -> bool) -> Vec<Value> {
        let mut lines = fixture();
        let index = find(&lines, agent, kind, wanted);
        lines.insert(index, lines[index].clone());
        lines
    }

    /// Exchanges two players' names everywhere in `lines`, so that a
    /// forgery that kills one in the other's place leaves a trajectory
    /// consistent about who is alive.
    fn swap(lines: &mut [Value], one: &str, other: &str) {
        fn rename(value: &mut Value, one: &str, other: &str) {
            match value {
                Value::String(name) if name == one => *name = other.to_owned(),
                Value::String(name) if name == other => *name = one.to_owned(),
                Value::Array(items) => {
                    for item in items.iter_mut() {
                        rename(item, one, other);
                    }
                    items.sort_by_key(ToString::to_string);
                }
                Value::Object(fields) => {
                    let swapped: Map<String, Value> = std::mem::take(fields)
                        .into_iter()
                        .map(|(key, mut value)| {
                            rename(&mut value, one, other);
                            let key = if key == one {
                                other.to_owned()
                            } else if key == other {
                                one.to_owned()
                            } else {
                                key
                            };
                            (key, value)
                        })
                        .collect();
                    *fields = swapped;
                }
                _ => {}
            }
        }
        for line in lines {
            rename(line, one, other);
        }
    }

    fn recipients(line: &mut Value, to: &[&str]) {
        line["event"]["recipients"] = json!(to);
    }

    fn sender(line: &mut Value, from: &str) {
        line["event"]["sender"] = json!(from);
    }

    fn target(line: &mut Value, whom: &str) {
        line["event"]["payload"]["Response"]["chosen"] = json!({"Target": whom});
    }

    #[test]
    fn the_fixture_passes() {
        check(&fixture(), &config());
    }

    #[test]
    #[should_panic(expected = "truncated game")]
    fn a_game_without_an_outcome_is_caught() {
        check(
            &without("moderator", "action", narration("Outcome")),
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
        let nominate = asked("alice", 1, "Nominate");
        check(
            &heard(response(nominate), |line| sender(line, "bob")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a response answers a request that was asked")]
    fn a_response_to_nothing_is_caught() {
        let lines = heard(response(asked("alice", 1, "Nominate")), |line| {
            line["event"]["payload"]["Response"]["request"] = json!(99);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request is answered once")]
    fn a_request_answered_twice_is_caught() {
        let nominate = asked("alice", 1, "Nominate");
        check(
            &doubled("moderator", "observation", response(nominate)),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "every request is answered")]
    fn an_unanswered_request_is_caught() {
        let nominate = asked("alice", 1, "Nominate");
        check(
            &without("moderator", "observation", response(nominate)),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "no request is asked of a player after its elimination")]
    fn a_request_to_the_dead_is_caught() {
        // alice was lynched on day 1; bob's nomination on day 2 goes to her.
        let nominate = asked("bob", 2, "Nominate");
        check(
            &said(request(nominate), |line| recipients(line, &["alice"])),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "no response arrives from a player after its elimination")]
    fn a_response_from_the_dead_is_caught() {
        // alice's nomination on day 1, dated after her lynching that day.
        let lines = heard(response(asked("alice", 1, "Nominate")), |line| {
            line["seq"] = json!(81);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request asks a player what its role is asked")]
    fn a_request_of_the_wrong_kind_is_caught() {
        // dave, a werewolf, is asked to devour; bob is a villager.
        let devour = asked("dave", 1, "Devour");
        let mut lines = said(request(devour), |line| recipients(line, &["bob"]));
        let index = find(&lines, "moderator", "observation", response(devour));
        sender(&mut lines[index], "bob");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no move targets the player taking it")]
    fn a_self_target_is_caught() {
        let nominate = asked("alice", 1, "Nominate");
        check(
            &heard(response(nominate), |line| target(line, "alice")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a move targets a living player")]
    fn a_dead_target_is_caught() {
        // alice was lynched on day 1; bob nominates her on day 2.
        let nominate = asked("bob", 2, "Nominate");
        check(
            &heard(response(nominate), |line| target(line, "alice")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "an abstention is outside the action space of Nominate")]
    fn an_abstention_from_nominating_is_caught() {
        let lines = heard(response(asked("alice", 1, "Nominate")), |line| {
            line["event"]["payload"]["Response"]["chosen"] = json!("Abstain");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the doctor never protects the same player two nights running")]
    fn a_repeated_protection_is_caught() {
        // carol is the doctor; bob lives through both nights.
        let mut lines = heard(response(asked("carol", 1, "Protect")), |line| {
            target(line, "bob");
        });
        let index = find(
            &lines,
            "moderator",
            "observation",
            response(asked("carol", 2, "Protect")),
        );
        target(&mut lines[index], "bob");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a finding goes to the seer alone")]
    fn a_finding_sent_to_a_villager_is_caught() {
        let lines = said(narration("Investigated"), |line| {
            recipients(line, &["alice", "grace"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night tally goes to the werewolves who cast it")]
    fn a_night_tally_sent_to_a_villager_is_caught() {
        let lines = said(tally(1, "Night"), |line| {
            recipients(line, &["alice", "dave", "erin"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a narration to the living goes to exactly the living")]
    fn a_narration_to_the_dead_is_caught() {
        let everyone = everyone();
        let lines = said(began(2, "Night"), |line| {
            line["event"]["recipients"] = json!(everyone);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a narration to the living goes to exactly the living")]
    fn a_narration_withheld_from_a_living_player_is_caught() {
        let lines = said(narration("NoDeath"), |line| {
            recipients(line, &["alice", "bob", "carol", "dave", "erin", "frank"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome goes to exactly the living")]
    fn an_outcome_sent_to_a_dead_player_is_caught() {
        // The broadcast the design used to make: the outcome to everyone,
        // living and dead. It is what the withdrawal of the exception rules
        // out, so it is what this catches.
        let everyone = everyone();
        let lines = said(narration("Outcome"), |line| {
            line["event"]["recipients"] = json!(everyone);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome is announced once")]
    fn an_outcome_announced_twice_is_caught() {
        check(
            &doubled("moderator", "action", narration("Outcome")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "no message naming the pack is addressed to a non-werewolf")]
    fn a_pack_named_to_a_villager_is_caught() {
        let lines = said(assigned("Werewolf", false), |line| {
            line["event"]["payload"]["Narration"]["Assigned"]["pack"] = json!(["dave", "erin"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a werewolf is told its pack")]
    fn a_werewolf_kept_from_its_pack_is_caught() {
        let lines = said(assigned("Werewolf", true), |line| {
            line["event"]["payload"]["Narration"]["Assigned"]["pack"] = json!([]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night tally is the werewolves' votes")]
    fn a_night_tally_with_a_villager_in_it_is_caught() {
        let lines = said(tally(1, "Night"), |line| {
            line["event"]["payload"]["Narration"]["Tally"]["votes"]["carol"] =
                json!({"Target": "alice"});
            recipients(line, &["carol", "dave", "erin"]);
        });
        check(&lines, &config());
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
        let everyone = everyone();
        let lines = said(began(2, "Night"), |line| {
            line["event"]["payload"]["Narration"]["PhaseBegan"]["living"] = json!(everyone);
            line["event"]["recipients"] = json!(everyone);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a request belongs to the phase it is asked in")]
    fn a_request_from_another_round_is_caught() {
        let lines = said(request(asked("alice", 1, "Nominate")), |line| {
            line["event"]["payload"]["Request"]["round"] = json!(2);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night has one death or one NoDeath")]
    fn a_night_that_says_nothing_of_deaths_is_caught() {
        check(
            &without("moderator", "action", narration("NoDeath")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a day eliminates exactly one")]
    fn a_day_without_a_lynching_is_caught() {
        check(&without("moderator", "action", lynched), &config());
    }

    #[test]
    #[should_panic(expected = "an elimination is of a player the tally names most")]
    fn a_lynching_the_tally_does_not_call_for_is_caught() {
        // On the last day the tally names grace twice and frank twice, and
        // bob once; bob, a villager, is lynched in grace's place. The last
        // day, so that no later request to grace trips the check on the
        // dead first, and the outcome is corrected to match, so that the
        // tally is what the check trips on rather than the survivors.
        let mut lines = fixture();
        let death = find(&lines, "moderator", "action", eliminated("grace"));
        lines[death]["event"]["payload"]["Narration"]["Eliminated"] =
            json!({"who": "bob", "role": "Villager", "round": 2, "cause": "Lynched"});
        let outcome = find(&lines, "moderator", "action", narration("Outcome"));
        lines[outcome]["event"]["payload"]["Narration"]["Outcome"]["living"] =
            json!(["dave", "erin", "frank", "grace"]);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a death at night is of a player the doctor did not protect")]
    fn a_death_of_a_protected_player_is_caught() {
        // On night 2 the pack agrees on carol, carol protects erin, and
        // carol is devoured. The forgery has the pack name grace and carol
        // protect grace, so the night's victim is the protected player.
        // Grace dies in carol's place, so from that death on the two are
        // exchanged everywhere — carol is lynched on day 2 in grace's
        // stead — and each still dies exactly once.
        let mut lines = fixture();
        for who in ["dave", "erin"] {
            let index = find(
                &lines,
                "moderator",
                "observation",
                response(asked(who, 2, "Devour")),
            );
            target(&mut lines[index], "grace");
        }
        let protect = find(
            &lines,
            "moderator",
            "observation",
            response(asked("carol", 2, "Protect")),
        );
        target(&mut lines[protect], "grace");
        let tally = find(&lines, "moderator", "action", tally(2, "Night"));
        lines[tally]["event"]["payload"]["Narration"]["Tally"]["votes"] =
            json!({"dave": {"Target": "grace"}, "erin": {"Target": "grace"}});
        let death = find(&lines, "moderator", "action", eliminated("carol"));
        swap(&mut lines[death..], "carol", "grace");
        // A swap exchanges the names, not the roles each death reveals.
        lines[death]["event"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Seer");
        let lynched = find(&lines[death..], "moderator", "action", eliminated("carol")) + death;
        lines[lynched]["event"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Doctor");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night with no death is one on which the doctor protected")]
    fn a_quiet_night_without_a_save_is_caught() {
        // On night 1 the pack splits between alice and bob, and carol
        // protects alice.
        let lines = heard(response(asked("carol", 1, "Protect")), |line| {
            target(line, "grace");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the cause is the phase's")]
    fn a_devouring_by_day_is_caught() {
        let lines = said(lynched, |line| {
            line["event"]["payload"]["Narration"]["Eliminated"]["cause"] = json!("Devoured");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a death reveals the role")]
    fn a_death_revealing_the_wrong_role_is_caught() {
        let lines = said(lynched, |line| {
            line["event"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Seer");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the winner is what the parity rule says")]
    fn a_winner_against_the_parity_rule_is_caught() {
        // The werewolves won this game, so it is the village that is the
        // claim the survivors do not bear out.
        let lines = said(narration("Outcome"), |line| {
            line["event"]["payload"]["Narration"]["Outcome"]["winner"] = json!("Village");
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
            &doubled("alice", "observation", narration("Assigned")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "the last thing the survivor bob received is the outcome")]
    fn a_survivor_that_never_hears_the_outcome_is_caught() {
        check(
            &without("bob", "observation", narration("Outcome")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a dead player observes nothing after its own death")]
    fn a_dead_player_that_hears_more_is_caught() {
        // bob's copy of the day 2 tally, delivered to the dead alice too,
        // after the narration of alice's own elimination.
        let mut lines = fixture();
        let leaked = lines[find(&lines, "bob", "observation", tally(2, "Day"))].clone();
        let mut leaked = leaked;
        leaked["agent"] = json!("alice");
        let stop = lines
            .iter()
            .position(|line| line["agent"] == "alice" && line["control"] == "stop")
            .expect("alice is stopped");
        lines.insert(stop, leaked);
        check(&lines, &config());
    }

    /// The index of the reward belonging to `who`.
    fn reward_of(lines: &[Value], who: &str) -> usize {
        lines
            .iter()
            .position(|line| line["type"] == "reward" && line["agent"] == who)
            .unwrap_or_else(|| panic!("{who} has a reward"))
    }

    #[test]
    #[should_panic(expected = "has a reward")]
    fn a_player_never_paid_is_caught() {
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        lines.remove(index);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "exactly one reward")]
    fn a_player_paid_twice_is_caught() {
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        let again = lines[index].clone();
        lines.insert(index, again);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "grace held Seer")]
    fn a_reward_that_contradicts_the_faction_that_won_is_caught() {
        // Grace held the seer and the werewolves won, so grace lost. A
        // reward saying otherwise is the reward disagreeing with the game
        // the same file records.
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        lines[index]["value"] = json!(1);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "dave held Werewolf")]
    fn a_winner_paid_as_a_loser_is_caught() {
        let mut lines = fixture();
        let index = reward_of(&lines, "dave");
        lines[index]["value"] = json!(-1);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "logged before the stop that ends its agent's trajectory")]
    fn a_reward_logged_after_its_agents_stop_is_caught() {
        // An agent's trajectory ends at its stop, so a reward stamped
        // after one is scoring an episode that was already over for it.
        // The claim is the shared checker's, since it holds of any
        // environment's rewards and not only Werewolf's.
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        let stop = lines
            .iter()
            .find(|line| line["agent"] == "grace" && line["control"] == "stop")
            .expect("grace is stopped");
        lines[index]["created"] = json!(stop["created"].as_u64().unwrap() + 1);
        super::super::check(&lines);
    }

    #[test]
    #[should_panic(expected = "plays no game and is paid nothing")]
    fn a_reward_for_the_moderator_is_caught() {
        // The environment runs the game rather than playing it, so there
        // is nothing its behavior could be worth.
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        let mut moderators = lines[index].clone();
        moderators["agent"] = json!("moderator");
        lines.insert(index, moderators);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "begins with a start and ends with a stop")]
    fn an_agent_never_stopped_is_caught() {
        // The moderator ends the episode by stopping every player. One left
        // running is a game the moderator did not finish ending.
        let mut lines = fixture();
        let stop = lines
            .iter()
            .position(|line| line["agent"] == "grace" && line["control"] == "stop")
            .expect("grace is stopped");
        lines.remove(stop);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "begins with a start and ends with a stop")]
    fn a_moderator_never_stopped_is_caught() {
        // The environment's own trajectory has the same shape as everyone
        // else's; the episode is what stops it, once the players have gone.
        let mut lines = fixture();
        let stop = lines
            .iter()
            .position(|line| line["agent"] == "moderator" && line["control"] == "stop")
            .expect("the moderator is stopped");
        lines.remove(stop);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a player sends only responses, to the moderator alone")]
    fn a_player_addressing_another_player_is_caught() {
        let lines = edited(
            "alice",
            "action",
            response(asked("alice", 1, "Nominate")),
            |line| {
                recipients(line, &["bob", "moderator"]);
            },
        );
        check(&lines, &config());
    }
}
