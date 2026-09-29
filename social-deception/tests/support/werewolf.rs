//! Werewolf's own invariants: what the log of a game of Werewolf
//! satisfies beyond what [`super::actor::check`] asserts of any log of an
//! [`actor`](social_deception::actor) runtime episode.
//!
//! The checks are written against the parsed lines, the way the runtime's own
//! checks are, and read the game the way the transcript reader does: from the
//! moderator's records, whose `action` records are every narration it sent
//! and whose `observation` records are every selection it received, in the
//! order it recorded them. The players' records are consulted only for what
//! the moderator cannot vouch for, which is what actually reached each of
//! them.
//!
//! Since ADR-0014 the log holds no request at all. Nobody is told to
//! act: a player observes that a phase has begun, asks its own role what
//! that phase wants of it, and selects. So a selection is no longer half of a
//! pair to be joined up — it carries the round and the kind of the session
//! it was made in, and every check below reads what it needs straight off
//! the selection and off the phase narrations that frame it.
//!
//! They fall into four groups:
//!
//! - **protocol**: every selection names a session its sender's role really
//!   is a member of, in the phase under way when it was made, with a
//!   target inside the action space the rules allow it; the dead are never
//!   heard from; players only ever address the moderator; and the
//!   moderator's last word is the outcome. A
//!   log whose last narration is not an outcome is a game the
//!   moderator never ended, which the episode also catches as a stall;
//! - **hidden information**: the realized observations stay inside each
//!   role's observation space. Every message goes to exactly the players
//!   the rules address it to: the pack is named only to werewolves, a
//!   devour is passed on to the living pack alone, a finding goes to the
//!   seer alone, and a narration to the living goes to exactly the living.
//!   Routing is the whole of the hidden-information mechanism, so these are
//!   what the design exists to guarantee;
//! - **the episode's shape**: every agent's records, the moderator's
//!   included, begin with a `Start` control and end with a `Stop`, because
//!   the moderator is the episode's environment and starting and stopping the
//!   players is its doing. Nothing at all reaches a dead player from the
//!   moment of its death: not its own death, which it is never told, not
//!   a peer's selection, nothing. The victim is left out of the `Eliminated`
//!   narration and its agent is stopped in the same cycle (ADR-0012), so
//!   its records simply end where the game ended for it. No message
//!   of this game is broadcast, and the outcome, which ADR-0004 once
//!   excepted, is narrated to the living like everything else;
//! - **the rewards**: every player has exactly one, +1 exactly when the
//!   role it was dealt belongs to the winning faction and −1 otherwise,
//!   living or dead. Death does not affect it: a reward is logged rather
//!   than sent (ADR-0007), so a player stopped mid-game is still paid at
//!   the end, after its own `Stop`. The moderator has none: it plays no
//!   game, so there is nothing its behavior could be worth;
//! - **game shape**: the phases alternate from the first night, each
//!   eliminates at most one player and each day exactly one, the player
//!   eliminated is one the phase's counted selections name most often, a
//!   night with no
//!   death is one on which the doctor protected such a player and only
//!   then, the living set strictly shrinks every round, the game ends
//!   within as many rounds as there are players, the winner is what the
//!   parity rule says of the final living set, and the roles dealt are the
//!   ones configured.

use std::collections::{BTreeMap, BTreeSet};

use serde::Deserialize;
use serde_json::Value;
use social_deception::ActorId;
use social_deception::werewolf::{
    Assignment, Cause, Config, Faction, Message, Narration, Outcome, Phase, Role, Round, Select,
    SessionKind, role,
};

/// Something the moderator said.
struct Said<'a> {
    at: usize,
    to: BTreeSet<ActorId>,
    message: Message,
    line: &'a Value,
}

/// A selection the moderator heard.
struct Heard<'a> {
    at: usize,
    from: ActorId,
    selection: Select,
    line: &'a Value,
}

/// A selection the moderator passed on: one it accepted.
///
/// The moderator forwards only a selection whose session was still open
/// (ADR-0014), so a forward is the log's record that a selection was
/// counted. Since nothing summarizes a session any more (ADR-0015), it is
/// also the only such record: what a session decided is read off these.
struct Forwarded<'a> {
    at: usize,
    from: ActorId,
    to: BTreeSet<ActorId>,
    selection: Select,
    line: &'a Value,
}

/// A game as its moderator recorded it, with the facts every check needs
/// read out once: who holds which role, where each phase began, when each
/// player was eliminated, and how it ended.
struct Play<'a> {
    config: &'a Config,
    /// Everything the moderator said, in sequence order.
    said: Vec<Said<'a>>,
    /// Every selection the moderator heard, in sequence order.
    heard: Vec<Heard<'a>>,
    /// Every selection the moderator passed on, in sequence order: the
    /// selections it accepted.
    forwarded: Vec<Forwarded<'a>>,
    /// Each phase, in the order it began: its round, its phase, and the
    /// line of the narration that announced it.
    ///
    /// This is what a selection is placed in. Nothing is issued to a player
    /// any more (ADR-0014), so the phase narration is the only record of
    /// a session opening, and the phase a selection falls in is the latest
    /// one announced before it.
    phases: Vec<(Round, Phase, usize)>,
    /// Each player's role, from the `Assigned` narration it was sent.
    assignment: Assignment,
    /// The line of the narration that eliminated each player.
    eliminated: BTreeMap<ActorId, usize>,
    /// How the game ended: the moderator's last word.
    outcome: Outcome,
}

/// Asserts everything the log of a game played from `config` must
/// satisfy; see the [module documentation](self).
///
/// Run [`super::actor::check`] first: these checks assume the moderator's
/// records are in sequence order and that every message names its recipients.
///
/// # Panics
///
/// On the first invariant that does not hold, naming the record.
pub fn check(lines: &[Value], config: &Config) {
    let play = Play::read(lines, config);
    // The phases first: a selection is placed by the phase it names
    // (ADR-0014), so a log whose phases are themselves wrong
    // should fail by that name rather than as a selection that cannot be
    // placed in them.
    play.check_phases();
    play.check_selections();
    play.check_action_spaces();
    play.check_recipients();
    play.check_forwards();
    play.check_hidden_information();
    play.check_outcome();
    play.check_players(lines);
    // The episode's shape first: a reward is checked against the `Stop`
    // that ends its agent's records, so "everybody was stopped" should
    // fail by its own name rather than as a missing stop to compare with.
    check_episode(lines, config);
    play.check_rewards(lines);
}

/// Every agent's records, the moderator's included, begin with a `Start`
/// control and end with a `Stop`, and nobody is started or stopped twice.
///
/// It is the moderator that sends both, being the episode's environment, so
/// this is the check that the game's own shutdown happened: a log
/// whose players were stopped by the episode picking up the pieces would
/// look the same here, but one where somebody was never stopped at all
/// would not.
fn check_episode(lines: &[Value], config: &Config) {
    let everybody = config
        .players
        .iter()
        .chain([&config.moderator])
        .cloned()
        .collect::<BTreeSet<ActorId>>();
    for who in &everybody {
        let controls: Vec<&str> = lines
            .iter()
            .filter(|line| line["type"] == "control" && super::agent(line) == who.as_str())
            .map(|line| line["control"].as_str().expect("a control names itself"))
            .collect();
        assert_eq!(
            controls,
            ["start", "stop"],
            "{who}'s records begin with a start and end with a stop"
        );
    }
    // The `episode` header is nobody's record: it is the log's wall-clock
    // anchor and names no agent (ADR-0017).
    let agents: BTreeSet<ActorId> = lines
        .iter()
        .filter(|line| line["type"] != "episode")
        .map(|line| ActorId::new(super::agent(line)))
        .collect();
    assert_eq!(
        agents, everybody,
        "the roster is every player and the moderator, and nobody else"
    );
}

/// The payload of a message record, as a [`Message`].
fn message(line: &Value) -> Message {
    Message::deserialize(&line["message"]["payload"])
        .unwrap_or_else(|error| panic!("a payload is a werewolf message ({error}): {line}"))
}

/// The sender named inside a message record's wire shape, which is always the
/// agent that sent it: nothing an actor sends claims another (ADR-0017).
fn sender_of(line: &Value) -> ActorId {
    ActorId::deserialize(&line["message"]["sender"]).expect("a message names its sender")
}

/// Which line each message the moderator sent went out on, by its key.
///
/// Every message a player observes came from the moderator, so this is the
/// order in which the game said things, against which what a dead player
/// heard is checked.
fn sent_lines(lines: &[Value], moderator: &ActorId) -> BTreeMap<(ActorId, u64), usize> {
    records_of(lines, moderator, "action")
        .map(|(at, line)| ((sender_of(line), super::seq(line)), at))
        .collect()
}

/// The recipients of a message record.
fn recipients(line: &Value) -> BTreeSet<ActorId> {
    BTreeSet::deserialize(&line["message"]["recipients"]).expect("a message lists its recipients")
}

/// The records of `agent` of the given type, `"action"` for what it sent or
/// `"observation"` for what it received, each with its line's position in the
/// file, in file order.
///
/// The position is how every check below orders one agent's records. Line
/// order across agents carries no meaning (ADR-0017), but an agent is one
/// thread, so its own records reach the writer in the order it wrote them,
/// and for the moderator that is the order it played the game in. A sequence
/// number will not do: it belongs to the *message*, so the moderator's
/// observations carry the players' numbers, which are dense over each
/// player's records and not over the moderator's.
fn records_of<'a>(
    lines: &'a [Value],
    agent: &'a ActorId,
    kind: &'a str,
) -> impl Iterator<Item = (usize, &'a Value)> {
    lines
        .iter()
        .enumerate()
        .filter(move |(_, line)| line["type"] == kind && super::agent(line) == agent.as_str())
}

/// The selections the moderator passed on for a player: the
/// [`Message::Relayed`] among its actions (ADR-0018).
///
/// A relay is what says the moderator accepted the selection, so the checks
/// read what a session decided off these (ADR-0015). The player it belongs to
/// is the envelope's sender, not the message's: the message is the
/// moderator's own.
fn forwards_of<'a>(
    lines: &'a [Value],
    config: &'a Config,
    players: &BTreeSet<&ActorId>,
) -> Vec<Forwarded<'a>> {
    records_of(lines, &config.moderator, "action")
        .filter_map(|(at, line)| match message(line) {
            Message::Relayed(envelope) => Some((at, line, envelope)),
            Message::Narration(_) => None,
            Message::Select(_) => {
                panic!("the moderator relays a selection rather than sending one: {line}")
            }
            Message::Reminder(_) => panic!(
                "a reminder is logged where it arrives and not where it is set (ADR-0016): {line}"
            ),
        })
        .map(|(at, line, envelope)| {
            assert_eq!(
                sender_of(line),
                config.moderator,
                "a relay is the moderator's own message: {line}"
            );
            assert!(
                players.contains(&envelope.from),
                "the moderator passes on a selection of a player's: {line}"
            );
            Forwarded {
                at,
                from: envelope.from,
                to: recipients(line),
                selection: envelope.payload,
                line,
            }
        })
        .collect()
}

/// The one recipient of a message addressed to a single player.
fn only<'a>(to: &'a BTreeSet<ActorId>, line: &Value) -> &'a ActorId {
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

/// The players a session's selections name most often: the ones the
/// elimination is drawn from. A member whose selection was never counted is
/// absent and names nobody.
fn leaders(votes: &BTreeMap<ActorId, ActorId>) -> BTreeSet<ActorId> {
    let mut counts: BTreeMap<&ActorId, usize> = BTreeMap::new();
    for who in votes.values() {
        *counts.entry(who).or_default() += 1;
    }
    let most = counts.values().copied().max().unwrap_or(0);
    counts
        .into_iter()
        .filter(|(_, count)| *count == most)
        .map(|(who, _)| who.clone())
        .collect()
}

/// Every selection the moderator heard, in file order.
///
/// The moderator hears two things: selections from players, and the reminders
/// it set for itself, which arrive as ordinary messages from itself
/// (ADR-0016). Its own reminders are its clocks running and say nothing about
/// the game, so they are checked to be its own and then dropped; everything
/// else is a selection or the log is wrong.
fn heard_by<'a>(
    lines: &'a [Value],
    config: &'a Config,
    players: &BTreeSet<&ActorId>,
) -> Vec<Heard<'a>> {
    records_of(lines, &config.moderator, "observation")
        .filter_map(|(at, line)| {
            let from = ActorId::deserialize(&line["message"]["sender"]).unwrap();
            let selection = match message(line) {
                Message::Select(selection) => selection,
                Message::Reminder(_) => {
                    assert_eq!(
                        from, config.moderator,
                        "a reminder is always self-directed: {line}"
                    );
                    assert_eq!(
                        recipients(line),
                        [config.moderator.clone()]
                            .into_iter()
                            .collect::<BTreeSet<ActorId>>(),
                        "a reminder is addressed to the actor that set it and nobody else: {line}"
                    );
                    return None;
                }
                other => panic!(
                    "the moderator hears selections and its own reminders, not {other:?}: {line}"
                ),
            };
            assert!(
                players.contains(&from),
                "the moderator hears only from players: {line}"
            );
            Some(Heard {
                at,
                from,
                selection,
                line,
            })
        })
        .collect()
}

impl<'a> Play<'a> {
    /// Reads the moderator's records out of `lines` and the facts the
    /// checks need out of them.
    ///
    /// Panics on anything that is not even the shape of a game: a moderator
    /// saying anything to a non-player or hearing anything but a selection
    /// from one, a request issued twice, a player assigned twice or
    /// eliminated twice, or a game without an outcome.
    fn read(lines: &'a [Value], config: &'a Config) -> Self {
        let players: BTreeSet<&ActorId> = config.players.iter().collect();
        let forwarded = forwards_of(lines, config, &players);
        let said: Vec<Said> = records_of(lines, &config.moderator, "action")
            // What the moderator says for itself, which is every narration
            // and nothing else: the relays above carry the players'
            // selections, read from the same records, and a bare selection is
            // not the moderator's to send at all — `forwards_of` has already
            // caught one.
            .filter_map(|(at, line)| match message(line) {
                Message::Narration(narration) => Some((at, line, narration)),
                Message::Relayed(_) | Message::Select(_) | Message::Reminder(_) => None,
            })
            .map(|(at, line, narration)| {
                let to = recipients(line);
                assert!(
                    to.iter().all(|who| players.contains(who)),
                    "the moderator addresses only players: {line}"
                );
                Said {
                    at,
                    to,
                    message: Message::Narration(narration),
                    line,
                }
            })
            .collect();
        let heard = heard_by(lines, config, &players);

        let mut phases = Vec::new();
        let mut roles = Vec::new();
        let mut eliminated = BTreeMap::new();
        for said in &said {
            let line = said.line;
            match &said.message {
                Message::Narration(Narration::PhaseBegan { round, phase, .. }) => {
                    phases.push((*round, *phase, said.at));
                }
                Message::Narration(Narration::Assigned { role, .. }) => {
                    roles.push((only(&said.to, line).clone(), *role));
                }
                Message::Narration(Narration::Eliminated { who, .. }) => {
                    assert!(
                        eliminated.insert(who.clone(), said.at).is_none(),
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
                    "a log without an outcome is a truncated game; the moderator's last \
                     word was {}",
                    last.line
                )
            })
            .clone();

        Self {
            config,
            said,
            heard,
            forwarded,
            phases,
            assignment,
            eliminated,
            outcome,
        }
    }

    /// The role dealt to `who`.
    fn role(&self, who: &ActorId) -> Role {
        self.assignment
            .role(who)
            .unwrap_or_else(|| panic!("{who} is not a player"))
    }

    /// The players `role` was dealt to.
    fn holders(&self, role: Role) -> BTreeSet<ActorId> {
        self.assignment
            .players()
            .filter(|(_, held)| *held == role)
            .map(|(who, _)| who.clone())
            .collect()
    }

    /// Whether `who` had been eliminated before the record on line `at`.
    fn dead_at(&self, who: &ActorId, at: usize) -> bool {
        self.eliminated.get(who).is_some_and(|&died| died < at)
    }

    /// Everyone not yet eliminated at the record on line `at`.
    fn living_at(&self, at: usize) -> BTreeSet<ActorId> {
        self.config
            .players
            .iter()
            .filter(|who| !self.dead_at(who, at))
            .cloned()
            .collect()
    }

    /// Whom each doctor protected in `round`, from the selections it made.
    ///
    /// A selection says which round and which session it belongs to, so this
    /// reads straight off the selections with no request to join to. The
    /// doctor's latest selection of the round is its protection, the same
    /// rule the session itself applies (ADR-0011).
    fn protections_in(&self, round: Round) -> BTreeMap<ActorId, ActorId> {
        self.heard
            .iter()
            .filter(|heard| {
                heard.selection.round == round && heard.selection.kind == SessionKind::Protect
            })
            .map(|heard| (heard.from.clone(), heard.selection.target.clone()))
            .collect()
    }

    /// The action space the rules permit `who` in a session of `kind` in
    /// `round`, with `living` alive.
    ///
    /// The only history it depends on is last night's protection, which
    /// is the doctor's own rule; everything else is the living set.
    fn action_space(
        &self,
        who: &ActorId,
        kind: SessionKind,
        round: Round,
        living: &BTreeSet<ActorId>,
    ) -> Vec<ActorId> {
        let last_protected = round
            .previous()
            .filter(|_| kind == SessionKind::Protect)
            .and_then(|before| self.protections_in(before).get(who).cloned());
        role::action_space(who, living, kind, last_protected.as_ref())
    }

    /// The members of the session of `kind` in `round`, with `living`
    /// alive: the living players whose role is asked that kind in that
    /// phase and whom the rules leave somewhere to select.
    ///
    /// Nobody is told this and nothing in the log states it
    /// (ADR-0014), so the check derives it the way the players and the
    /// moderator each derive it: from the roles. The non-empty action
    /// space matters — a doctor the rules leave nobody it may protect is
    /// not a member, and its session does not open — so a check that
    /// went by role alone would hold a player to a session it was never
    /// in.
    fn members_of(
        &self,
        kind: SessionKind,
        round: Round,
        living: &BTreeSet<ActorId>,
    ) -> BTreeSet<ActorId> {
        living
            .iter()
            .filter(|who| self.role(who).asked_in(kind.phase()) == Some(kind))
            .filter(|who| !self.action_space(who, kind, round, living).is_empty())
            .cloned()
            .collect()
    }

    /// The record that began the phase a selection names, if the game ever
    /// had such a phase.
    fn began(&self, round: Round, phase: Phase) -> Option<usize> {
        self.phases
            .iter()
            .find(|(had, was, _)| *had == round && *was == phase)
            .map(|(_, _, at)| *at)
    }

    /// The record that began the phase *after* the given one, which is
    /// the first record after that phase had certainly ended. `None` for
    /// the last phase of the game, which nothing follows.
    fn ended(&self, round: Round, phase: Phase) -> Option<usize> {
        let index = self
            .phases
            .iter()
            .position(|(had, was, _)| *had == round && *was == phase)?;
        self.phases.get(index + 1).map(|(_, _, at)| *at)
    }

    /// Every selection names a session the game really opened, with its
    /// sender a member of it.
    ///
    /// There is no request to answer any more, so what was once "every
    /// request is answered exactly once by the agent it was asked of"
    /// becomes a claim about the selections alone (ADR-0014): each one is a
    /// selection its sender could legitimately have made, in a session that
    /// really existed and that it really belonged to.
    ///
    /// Each selection is placed by the phase it *names*, never by the phase
    /// under way when the moderator recorded hearing it. A selection that
    /// lost a race with its own session's clock — or with its sender's
    /// stop — arrives after that phase has ended, and ADR-0011 makes
    /// that an ordinary message rather than a bug. Naming its own round is
    /// exactly what lets a reader place such a selection correctly instead
    /// of mistaking it for a selection in the phase that has since opened.
    ///
    /// Nothing here claims a member selects. A session closes on its
    /// clock, so a member that never selected is simply one nobody saw
    /// selection, which is how it abstains (ADR-0011).
    fn check_selections(&self) {
        for heard in &self.heard {
            let line = heard.line;
            let who = &heard.from;
            let kind = heard.selection.kind;
            assert_eq!(
                self.role(who).asked_in(kind.phase()),
                Some(kind),
                "a selection names a session its sender's role is a member of: {line}"
            );
            let round = heard.selection.round;
            let began = self
                .began(round, kind.phase())
                .unwrap_or_else(|| panic!("a selection names a phase the game played: {line}"));
            assert!(
                began < heard.at,
                "a selection is heard after the phase it names began: {line}"
            );
            // Membership as it stood when that phase began, which is
            // when the session opened. A selection heard later may have lost
            // a race with the clock or with its sender's own stop, and
            // neither makes it a selection its sender could not have made.
            assert!(
                self.members_of(kind, round, &self.living_at(began))
                    .contains(who),
                "a selection comes from a member of the session it names: {line}"
            );
        }
    }

    /// Every selection's target is in the action space the rules allow its
    /// sender in the session it names: a living player other than the one
    /// selecting, and for the doctor never the player it protected the
    /// night before.
    ///
    /// The action space is recomputed here from the living set and, for
    /// a doctor, from the previous round's protection, because the selection
    /// says which round and which session it belongs to (ADR-0014).
    /// Nothing has to be joined to a request to know what the rules
    /// allowed, and nothing has to be carried across rounds by hand: the
    /// doctor's constraint is last night's protection alone, so it is
    /// read from last night's selections and from nowhere else.
    ///
    /// Like [`check_selections`](Self::check_selections), a selection is
    /// judged against the phase it names rather than the one under way when
    /// the moderator heard it: a selection that lost a race with the clock is
    /// still a selection the rules allowed when it was made.
    fn check_action_spaces(&self) {
        // The doctor's rule first, on its own, because it is the one
        // rule that reaches across a round boundary and the general
        // action-space check below would otherwise swallow it and
        // report it as an anonymous "outside the space".
        for (round, phase, _) in &self.phases {
            let Some(previous) = round.previous().filter(|_| *phase == Phase::Night) else {
                continue;
            };
            let before = self.protections_in(previous);
            for (doctor, chosen) in self.protections_in(*round) {
                if let Some(last) = before.get(&doctor) {
                    assert_ne!(
                        *last,
                        chosen,
                        "the doctor never protects the same player two nights running: \
                         {doctor} protected {chosen} in round {}",
                        round.number()
                    );
                }
            }
        }
        for heard in &self.heard {
            let line = heard.line;
            let chosen = &heard.selection.target;
            assert_ne!(
                chosen, &heard.from,
                "no selection targets the player making it: {line}"
            );
            let round = heard.selection.round;
            let kind = heard.selection.kind;
            let began = self
                .began(round, kind.phase())
                .expect("`check_selections` ran first and found the phase");
            let living = self.living_at(began);
            assert!(
                self.assignment.role(chosen).is_some() && living.contains(chosen),
                "a selection targets a living player: {line}"
            );
            assert!(
                self.action_space(&heard.from, kind, round, &living)
                    .contains(chosen),
                "a selection targets somebody the rules allow it: {line}"
            );
        }
    }

    /// Every message goes to exactly the players the rules address it to.
    /// A role assignment goes to one player, which `read` checked. A
    /// finding goes to the seer alone; and a phase, a death, a quiet
    /// night, a quiet day and the outcome to the living. Where each
    /// forwarded selection goes is checked in
    /// [`check_forwards`](Self::check_forwards).
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
                Message::Narration(Narration::Assigned { .. }) => {}
                Message::Narration(Narration::Investigated { .. }) => {
                    assert_eq!(said.to, seers, "a finding goes to the seer alone: {line}");
                }
                // A death is announced to the living *after* it: the
                // victim is not told, because it is stopped in the same
                // cycle and there is nobody left to tell (ADR-0012).
                Message::Narration(Narration::Eliminated { who, .. }) => {
                    let mut living = self.living_at(said.at);
                    assert!(living.remove(who), "{who} was already dead: {line}");
                    assert_eq!(
                        said.to, living,
                        "a death is announced to the living, less the victim: {line}"
                    );
                }
                Message::Narration(
                    Narration::PhaseBegan { .. }
                    | Narration::NoDeath { .. }
                    | Narration::NoLynch { .. },
                ) => assert_eq!(
                    said.to,
                    self.living_at(said.at),
                    "a narration to the living goes to exactly the living: {line}"
                ),
                Message::Narration(Narration::Outcome(_)) => {
                    assert_eq!(
                        said.to,
                        self.living_at(said.at),
                        "the outcome goes to exactly the living: {line}"
                    );
                    outcomes += 1;
                }
                Message::Select(_) | Message::Relayed(_) | Message::Reminder(_) => {
                    unreachable!("a relay and a reminder are not among the moderator's own words")
                }
            }
        }
        assert_eq!(outcomes, 1, "the outcome is announced once");
    }

    /// Every selection the moderator passed on went to exactly the players
    /// the rules let see it, and to living players only.
    ///
    /// Since ADR-0015 nothing summarizes a session, so a forward is the
    /// only thing the moderator says about one selection of one member, and
    /// where each goes is the
    /// whole of the hidden information a session leaks: a `Devour` to the
    /// rest of the living pack, a `Nominate` to the rest of the living,
    /// and nothing at all for the seer's and the doctor's, which are
    /// between that player and the moderator and so are never forwarded.
    fn check_forwards(&self) {
        for forwarded in &self.forwarded {
            let line = forwarded.line;
            let living = self.living_at(forwarded.at);
            assert!(
                forwarded.to.iter().all(|who| living.contains(who)),
                "a selection is passed on to living players only: {line}"
            );
            assert!(
                !forwarded.to.contains(&forwarded.from),
                "a selection is not passed back to the player that made it: {line}"
            );
            let round = forwarded.selection.round;
            match forwarded.selection.kind {
                // The pack sees where the pack is selecting, and nobody
                // else sees a devour at all. Who the pack is comes from
                // `members_of`, the one place this file works out who a
                // session's members are, so the audience a forward is
                // held to is the session the rules opened.
                SessionKind::Devour => {
                    let mut pack = self.members_of(SessionKind::Devour, round, &living);
                    pack.remove(&forwarded.from);
                    assert_eq!(
                        forwarded.to, pack,
                        "a devour goes to the rest of the living pack: {line}"
                    );
                    assert_eq!(
                        self.role(&forwarded.from),
                        Role::Werewolf,
                        "only a werewolf devours: {line}"
                    );
                }
                // The day's vote is public among the living, whether or
                // not the rules leave each of them somewhere to select.
                SessionKind::Nominate => {
                    let mut others = living;
                    others.remove(&forwarded.from);
                    assert_eq!(
                        forwarded.to, others,
                        "a nomination goes to the rest of the living: {line}"
                    );
                }
                SessionKind::Investigate | SessionKind::Protect => panic!(
                    "a {:?} selection is nobody else's business and is never passed on: {line}",
                    forwarded.selection.kind
                ),
            }
        }
    }

    /// What the addressed messages say stays inside the recipient's
    /// observation space: a werewolf is told the pack and nobody else is
    /// told anything of it.
    fn check_hidden_information(&self) {
        for said in &self.said {
            let line = said.line;
            if let Message::Narration(Narration::Assigned { role, pack }) = &said.message {
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
        }
    }

    /// The phases run Night 1, Day 1, Night 2, Day 2, ... from the first
    /// night; each begins with the living as they are and asks only during
    /// itself; a night ends in one death or one `NoDeath`; a day ends in
    /// one lynching or one `NoLynch`; the player eliminated is one the
    /// phase's counted selections name most, unprotected at night, and a
    /// quiet night is one on which the doctor protected such a player; a
    /// death reveals the role and names the phase's cause; and the living
    /// strictly shrink from one round to the next.
    fn check_phases(&self) {
        // Straight off the selections: each says which round and which
        // session it was made in (ADR-0014), so there is nothing to join
        // it to. A doctor's latest selection of a round is its protection.
        let protected: BTreeMap<Round, &ActorId> = self
            .heard
            .iter()
            .filter(|heard| heard.selection.kind == SessionKind::Protect)
            .map(|heard| (heard.selection.round, &heard.selection.target))
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
                Message::Select(_) | Message::Relayed(_) | Message::Reminder(_) => {
                    unreachable!("`said` is the moderator's narrations alone")
                }
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
            self.living_at(usize::MAX),
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
            outcome.winner,
            Some(winner),
            "the winner is what the parity rule says of the survivors: {outcome:?}"
        );
        assert!(
            outcome.rounds.number() as usize <= self.config.players.len(),
            "the game ends within as many rounds as there are players: {outcome:?}"
        );
    }

    /// Every player has exactly one reward, worth +1 if the role it was
    /// dealt belongs to the winning faction and −1 if it does not. The
    /// moderator has none.
    ///
    /// Living or dead makes no difference. A reward is logged rather than
    /// sent (ADR-0007), so a player stopped in the middle of the game for
    /// dying (ADR-0012) is paid at the end like everybody else, after its
    /// own `Stop`. When a reward may be logged is the shared checker's
    /// business, and the bound it keeps is the episode's last stop.
    ///
    /// The roles come from the `Assigned` narrations the moderator sent,
    /// which is the same place every other check here gets them, so the
    /// claim is that the reward agrees with the game as it was actually
    /// dealt and not merely with itself. There are no stalemates
    /// (ADR-0007), so every player is paid one of the two and never zero.
    fn check_rewards(&self, lines: &[Value]) {
        let rewards: BTreeMap<ActorId, Vec<&Value>> = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .fold(BTreeMap::new(), |mut paid, line| {
                let who = ActorId::new(super::agent(line));
                paid.entry(who).or_default().push(line);
                paid
            });
        for who in &self.config.players {
            let paid = rewards
                .get(who)
                .unwrap_or_else(|| panic!("{who} has a reward"));
            assert_eq!(paid.len(), 1, "{who} has exactly one reward: {paid:?}");
            let line = paid[0];
            let expected = if self.outcome.winner == Some(self.role(who).faction()) {
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
    /// selection, always including the moderator among its recipients and
    /// never anyone the rules keep it from; each observed its role exactly
    /// once, the one the moderator dealt it; a survivor's last observation
    /// is the outcome; and a dead player observed nothing at all from its
    /// own death onward. And the roles dealt are the ones configured.
    ///
    /// Whether a player died is read from the moderator's records and not
    /// from the player's own, because the victim is never told: the
    /// `Eliminated` narration goes to the living after the death, which no
    /// longer includes the victim, and the victim's agent is stopped in the
    /// same cycle (ADR-0012). A dead player's records therefore simply
    /// end where the game ended for it, with no announcement to mark the
    /// spot.
    ///
    /// Who may see a selection is the kind's to say (ADR-0011): a `Devour`
    /// goes to the pack, a `Nominate` to every other living player, and the
    /// seer's and the doctor's to the moderator alone.
    fn check_players(&self, lines: &[Value]) {
        let moderator = &self.config.moderator;
        // Built once, not once per player: it is the same map every time, and
        // every player's records are checked against it.
        let sent_at = sent_lines(lines, moderator);
        for who in &self.config.players {
            for (_, line) in records_of(lines, who, "action") {
                let Message::Select(selection) = message(line) else {
                    panic!("a player sends only selections: {line}");
                };
                // A player addresses the moderator and nobody else. That
                // is the whole of the fix for a selection outliving its
                // session: there is no path to a peer that does not pass
                // the one agent that knows whether the session is open
                // (ADR-0014).
                let to = recipients(line);
                assert_eq!(
                    to,
                    [moderator.clone()]
                        .into_iter()
                        .collect::<BTreeSet<ActorId>>(),
                    "a selection is addressed to the moderator alone: {line}"
                );
                // Who should see it is named in the selection, for the
                // moderator to forward to.
                let others: BTreeSet<&ActorId> = selection.seen_by.iter().collect();
                assert!(
                    !others.contains(&who),
                    "a selection does not name its own author as an observer: {line}"
                );
                assert!(
                    !others.contains(moderator),
                    "the moderator hears every selection directly and is not forwarded one: {line}"
                );
                match selection.kind {
                    // The pack sees its own selecting and nobody else does.
                    SessionKind::Devour => assert!(
                        others
                            .iter()
                            .all(|other| self.role(other) == Role::Werewolf),
                        "a devour selection is seen by the pack alone: {line}"
                    ),
                    // The day's vote is public among the living.
                    SessionKind::Nominate => assert!(
                        others.iter().all(|other| *other != who),
                        "a nomination does not name its own author: {line}"
                    ),
                    // Nobody's business but the moderator's.
                    SessionKind::Investigate | SessionKind::Protect => assert!(
                        others.is_empty(),
                        "a {:?} selection is nobody else's business: {line}",
                        selection.kind
                    ),
                }
            }
            let received: Vec<Message> = records_of(lines, who, "observation")
                .map(|(_, line)| message(line))
                .collect();
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
            self.check_where_it_ended(lines, who, &received, &sent_at);
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

    /// Where the game ended for one player, read from its own records:
    /// a survivor's last observation is the outcome, and a dead player
    /// observed nothing from the moment of its death.
    ///
    /// Whether it died is the moderator's record to say, not this
    /// player's. Since ADR-0012 the victim is never told: the
    /// `Eliminated` narration goes to the living after the death, which
    /// no longer includes the victim, and its agent is stopped in the
    /// same cycle. A dead player's records simply stop, with no
    /// announcement in it to mark the place.
    fn check_where_it_ended(
        &self,
        lines: &[Value],
        who: &ActorId,
        received: &[Message],
        sent_at: &BTreeMap<(ActorId, u64), usize>,
    ) {
        // No player ever observes its own death, whatever else it saw.
        assert!(
            !received.iter().any(|message| {
                matches!(message, Message::Narration(Narration::Eliminated { who: dead, .. }) if dead == who)
            }),
            "{who} was told of its own death"
        );
        let Some(&death) = self.eliminated.get(who) else {
            assert!(
                self.outcome.living.contains(who),
                "a player never eliminated survives: {who}"
            );
            let last = received
                .last()
                .unwrap_or_else(|| panic!("{who} received nothing"));
            assert_eq!(
                outcome(last),
                Some(&self.outcome),
                "the last thing the survivor {who} received is the outcome; it was {last:?}, \
                 and it received {received:?}"
            );
            return;
        };
        // A dead player observes nothing sent after its death: not its own
        // death, not a peer's selection, nothing at all (ADR-0012). It is
        // stopped in the cycle the death is announced, and the router drops
        // whatever is addressed to it after that.
        //
        // "After the death" is read off the moderator's own records rather
        // than off a clock. The moderator sends everything a player observes,
        // directly or as a relay, and it is one thread, so its records are in
        // the order it sent them; a sequence number will not serve, since it
        // is the *message's* and so is the relayed player's on a forward
        // (ADR-0017). What every observation of this player joins to is
        // therefore a line of the moderator's at or before the one that
        // announced the death — and the death's own line is allowed, because
        // the stop and the messages of that cycle are one batch and one of
        // those messages may have been in flight.
        for (_, line) in records_of(lines, who, "observation") {
            let at = *sent_at
                .get(&(sender_of(line), super::seq(line)))
                .unwrap_or_else(|| {
                    panic!("{who} observed something the moderator never sent: {line}")
                });
            assert!(
                at <= death,
                "{who} died at line {death} but observed something the moderator \
                 sent on line {at}: {line}"
            );
        }
        assert!(
            !self.outcome.living.contains(who),
            "a player the moderator eliminated is not among the survivors: {who}"
        );
    }
}

/// A walk over the moderator's records phase by phase.
struct Phases<'p, 'a> {
    play: &'p Play<'a>,
    /// Whom the doctor protected each round, from its responses.
    protected: BTreeMap<Round, &'p ActorId>,
    /// The phase in progress: its round and phase, and the record that
    /// began it.
    current: Option<(Round, Phase, &'p Value)>,
    /// What has been narrated in it so far.
    counts: PhaseCounts,
    /// The living when the most recent night began.
    living_last_night: Option<BTreeSet<ActorId>>,
}

impl<'p, 'a> Phases<'p, 'a> {
    /// The players the session that decides a death named most often:
    /// the pack's at night, the day's by day.
    ///
    /// Read from the selections the moderator *passed on*, which are the
    /// selections it accepted. Nothing summarizes a session any more
    /// (ADR-0015), so a forward is the log's record that a selection
    /// counted: a selection the moderator merely heard may have lost a race
    /// with its session's clock, and counting it would hold the
    /// moderator to a vote it never took.
    ///
    /// A selection with nobody to see it is the exception. A lone werewolf's
    /// `Devour` names no audience, so there is nothing for the moderator
    /// to pass on and no forward is written however the race went; the
    /// log simply does not say whether that selection was accepted.
    /// For those the check falls back to the selections the moderator heard
    /// *while the phase was still running*, which is as much as the
    /// log does say. Bounding it by the phase matters: an
    /// unforwarded selection heard after the phase ended certainly lost its
    /// race, and counting it would credit the session with a vote it
    /// never took.
    fn leaders_of(&self, round: Round, phase: Phase) -> BTreeSet<ActorId> {
        let deciding = match phase {
            Phase::Night => SessionKind::Devour,
            Phase::Day => SessionKind::Nominate,
        };
        let mine = |selection: &Select| selection.round == round && selection.kind == deciding;
        let counted = self
            .play
            .forwarded
            .iter()
            .filter(|forwarded| mine(&forwarded.selection))
            .map(|forwarded| (forwarded.from.clone(), forwarded.selection.target.clone()));
        let ended = self.play.ended(round, phase).unwrap_or(usize::MAX);
        let unseen = self
            .play
            .heard
            .iter()
            .filter(|heard| mine(&heard.selection) && heard.selection.seen_by.is_empty())
            .filter(|heard| heard.at < ended)
            .map(|heard| (heard.from.clone(), heard.selection.target.clone()));
        let votes: BTreeMap<ActorId, ActorId> = counted.chain(unseen).collect();
        leaders(&votes)
    }

    /// A phase began: it is the one after the last, its living are the
    /// living, and the last phase was complete.
    fn began(
        &mut self,
        said: &'p Said<'a>,
        round: Round,
        phase: Phase,
        living: &BTreeSet<ActorId>,
    ) {
        let line = said.line;
        self.counts.close(self.current);
        self.counts = PhaseCounts::default();
        let expected = match self.current {
            None => (Round::FIRST, Phase::Night),
            Some((round, Phase::Night, _)) => (round, Phase::Day),
            Some((round, Phase::Day, _)) => (round.next(), Phase::Night),
        };
        assert_eq!(
            (round, phase),
            expected,
            "phases alternate from the first night: {line}"
        );
        assert_eq!(
            *living,
            self.play.living_at(said.at),
            "a phase begins with the living as they are: {line}"
        );
        if phase == Phase::Night {
            // Never grow, rather than strictly shrink. A round may now
            // pass with nobody dead: the doctor saves the pack's victim
            // and the day runs out without a majority (ADR-0011). What
            // bounds a game is the day cap.
            if let Some(before) = &self.living_last_night {
                assert!(living.is_subset(before), "the living never grow: {line}");
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
                    self.leaders_of(round, phase).contains(who),
                    "an elimination is of a player the deciding session named most: {line}"
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
                // Two ways a night passes with nobody dead (ADR-0011).
                // The doctor protected whoever the pack settled on; or
                // the pack named nobody at all, because no wolf's selection
                // reached its session before the clock closed it. The
                // second is rare with random players, which selection as
                // soon as they are asked, but it is a legal night and
                // not a bug: a pack that cannot agree to act does not
                // act.
                let chosen = self.leaders_of(round, Phase::Night);
                let saved = protected.is_some_and(|who| chosen.contains(who));
                let nobody_chosen = chosen.is_empty();
                assert!(
                    saved || nobody_chosen,
                    "a night with no death is one the doctor saved or one the pack named \
                     nobody in: {line}"
                );
            }
            _ => {}
        }
    }
}

/// What has been narrated so far in one phase.
#[derive(Default)]
struct PhaseCounts {
    eliminated: usize,
    no_death: usize,
}

impl PhaseCounts {
    /// Counts one narration of the phase `(round, phase)`, checking it
    /// belongs to that phase.
    fn count(&mut self, narration: &Narration, round: Round, phase: Phase, line: &Value) {
        match narration {
            Narration::Eliminated { round: r, .. } => {
                assert_eq!(*r, round, "a death belongs to its round: {line}");
                self.eliminated += 1;
            }
            Narration::NoDeath { round: r } => {
                assert_eq!(*r, round, "a quiet night belongs to its round: {line}");
                assert_eq!(phase, Phase::Night, "only a night has no death: {line}");
                self.no_death += 1;
            }
            Narration::NoLynch { round: r } => {
                assert_eq!(*r, round, "a quiet day belongs to its round: {line}");
                assert_eq!(phase, Phase::Day, "only a day has no lynch: {line}");
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
        // How a phase came out, which is the whole of what a phase
        // narrates about its sessions since ADR-0015: the sessions
        // themselves close silently, so there is nothing to count but
        // the outcome. The rule is one for both halves of a round: a
        // night ends in a death or a `NoDeath`, and a day in a lynching
        // or a `NoLynch`, now that a day may close on its limit without
        // a majority.
        let outcome = match phase {
            Phase::Night => "a death or one NoDeath",
            Phase::Day => "a lynching or one NoLynch",
        };
        assert_eq!(
            self.eliminated + self.no_death,
            1,
            "a {phase:?} ends in one {outcome}, never both or neither: {line}"
        );
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;

    /// The fixture: a seven-player game played to a werewolf win in four
    /// rounds, with its effective config beside it. Its first night is a
    /// saved one and no day of it ever reaches a majority, so every death
    /// in it is a devouring — carol on night 2, frank on night 3, alice on
    /// night 4 — and it ends by parity rather than by the pack being wiped
    /// out. Its doctor, carol, is the first to die, so nights 3 and 4 open
    /// no protection session at all. Every request in it is answered with a
    /// target: it holds no abstention.
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
                line["type"] == kind
                    && line["agent"] == agent
                    && wanted(&line["message"]["payload"])
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

    /// A **relayed** selection of the given round and session kind: what the
    /// moderator sends, whose payload is the envelope rather than the
    /// selection itself (ADR-0018).
    fn relay_in(round: u32, kind: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| {
            let selection = &payload["Relayed"]["envelope"]["payload"];
            selection["round"] == round && selection["kind"] == kind
        }
    }

    /// An elimination by the pack's night vote. Every death in the fixture
    /// is one of these: no day of it ever reaches a majority, so there is
    /// no lynching anywhere in the file to corrupt.
    fn devoured(payload: &Value) -> bool {
        payload["Narration"]["Eliminated"]["cause"] == "Devoured"
    }

    /// The elimination of the given player.
    fn eliminated(who: &str) -> impl Fn(&Value) -> bool + '_ {
        move |payload| payload["Narration"]["Eliminated"]["who"] == who
    }

    /// The selection `who` made in the session of `kind` in `round`.
    ///
    /// A selection says for itself which session it belongs to (ADR-0014),
    /// so a record is named by the round, the kind and the sender rather
    /// than by an id looked up in a request that no longer exists. The
    /// sender lives in the envelope rather than the payload, which is
    /// why this matches the whole record.
    fn selection_of<'w>(who: &'w str, round: u32, kind: &'w str) -> impl Fn(&Value) -> bool + 'w {
        move |line| {
            let selection = &line["message"]["payload"]["Select"];
            line["message"]["sender"] == who
                && selection["round"] == round
                && selection["kind"] == kind
        }
    }

    /// The index of the first record of `agent` of the given type that
    /// satisfies `wanted`, which is given the whole record.
    fn find_record(
        lines: &[Value],
        agent: &str,
        kind: &str,
        wanted: impl Fn(&Value) -> bool,
    ) -> usize {
        lines
            .iter()
            .position(|line| line["type"] == kind && line["agent"] == agent && wanted(line))
            .expect("the fixture has such a record")
    }

    /// The fixture with `edit` applied to the selection `selection_of` names,
    /// as the moderator recorded hearing it.
    fn heard_selection(
        wanted: impl Fn(&Value) -> bool,
        edit: impl FnOnce(&mut Value),
    ) -> Vec<Value> {
        let mut lines = fixture();
        let index = find_record(&lines, config().moderator.as_str(), "observation", &wanted);
        edit(&mut lines[index]);
        lines
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
    /// forgery that kills one in the other's place leaves a log
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
        line["message"]["recipients"] = json!(to);
    }

    /// Rewrites the audience a selection names: who the moderator is asked to
    /// forward it to. A leak is a name in here that does not belong.
    fn seen_by(line: &mut Value, to: &[&str]) {
        line["message"]["payload"]["Select"]["seen_by"] = json!(to);
    }

    fn sender(line: &mut Value, from: &str) {
        line["message"]["sender"] = json!(from);
    }

    /// Rewrites who a relay says selected, which is the envelope's sender and
    /// not the message's: the message is the moderator's own.
    fn selected_by(line: &mut Value, from: &str) {
        line["message"]["payload"]["Relayed"]["envelope"]["from"] = json!(from);
    }

    fn target(line: &mut Value, whom: &str) {
        line["message"]["payload"]["Select"]["target"] = json!(whom);
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
    #[should_panic(expected = "a selection names a session its sender's role is a member of")]
    fn a_selection_in_a_session_the_role_is_never_in_is_caught() {
        // bob is a villager, so no phase ever puts him in a `Protect`
        // session. This is what a selection "answering nothing" became under
        // ADR-0014: with no request to fail to match, a forged selection is
        // one nobody of that role could have made.
        //
        // A day selection is chosen so that rewriting it leaves the nights
        // alone: a night's save is read from the protections, and losing
        // one would fail as a quiet night without a save instead.
        let lines = heard_selection(selection_of("bob", 3, "Nominate"), |line| {
            line["message"]["payload"]["Select"]["kind"] = json!("Protect");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a selection names a phase the game played")]
    fn a_selection_naming_a_phase_the_game_never_played_is_caught() {
        // A selection says which round it was made in (ADR-0014), so a selection
        // naming a round the game never reached describes a session that
        // never opened. This is what replaced the check that a selection
        // arrived after the request it answered.
        let lines = heard_selection(selection_of("bob", 3, "Nominate"), |line| {
            line["message"]["payload"]["Select"]["round"] = json!(9);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a selection is heard after the phase it names began")]
    fn a_selection_heard_before_its_own_phase_began_is_caught() {
        // A selection may arrive late — that is a lost race, and ADR-0011
        // makes it ordinary. Arriving early is the impossible direction:
        // nobody selects in a session that has not opened, because what
        // opens it is the phase narration the player selects in answer to.
        let lines = heard_selection(selection_of("bob", 1, "Nominate"), |line| {
            line["message"]["payload"]["Select"]["round"] = json!(3);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a selection comes from a member of the session it names")]
    fn a_selection_from_the_dead_is_caught() {
        // carol was devoured on night 2; bob's nomination on day 3 is
        // recorded as hers. Nothing is asked of anybody any more
        // (ADR-0014), so "the dead are asked nothing" is now "the dead
        // are members of nothing": carol is the doctor, and a doctor
        // really is a member of a `Nominate` session — but only while
        // it lives.
        //
        // This is the one way a selection can come from somebody whose role
        // is right and who is still not a member, which is why the two
        // are one test rather than two.
        let lines = heard_selection(selection_of("bob", 3, "Nominate"), |line| {
            sender(line, "carol");
        });
        check(&lines, &config());
    }

    #[test]
    fn a_session_selected_in_twice_is_a_change_of_mind() {
        // A member may select as often as it likes while its session is
        // open, so two selections in one session is the rules working rather
        // than a bug (ADR-0011).
        let mut lines = fixture();
        let index = find_record(
            &lines,
            "moderator",
            "observation",
            selection_of("alice", 1, "Nominate"),
        );
        lines.insert(index, lines[index].clone());
        check(&lines, &config());
    }

    #[test]
    fn a_member_may_never_select_at_all() {
        // A session closes on its clock, so a member that never selected
        // is simply one nobody saw select (ADR-0011). Nothing claims a
        // member selects, so removing a selection is a game, not a forgery.
        let mut lines = fixture();
        let index = find_record(
            &lines,
            "moderator",
            "observation",
            selection_of("alice", 1, "Nominate"),
        );
        lines.remove(index);
        check(&lines, &config());
    }

    #[test]
    fn a_selection_that_lost_a_race_with_its_own_death_is_not_a_bug() {
        // alice's nomination on day 1, dated after her lynching that day:
        // a selection she sent before her stop reached her. ADR-0011 makes
        // that a lost race rather than a bug, and the game ignores it.
        let lines = heard_selection(selection_of("alice", 1, "Nominate"), |line| {
            line["seq"] = json!(81);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "no selection targets the player making it")]
    fn a_self_target_is_caught() {
        let lines = heard_selection(selection_of("alice", 1, "Nominate"), |line| {
            target(line, "alice");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a selection targets a living player")]
    fn a_dead_target_is_caught() {
        // carol was devoured on night 2; bob nominates her on day 3, a
        // round after she stopped being a player anyone may select.
        let lines = heard_selection(selection_of("bob", 3, "Nominate"), |line| {
            target(line, "carol");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the doctor never protects the same player two nights running")]
    fn a_repeated_protection_is_caught() {
        // carol is the doctor; bob lives through both nights.
        let mut lines = heard_selection(selection_of("carol", 1, "Protect"), |line| {
            target(line, "bob");
        });
        let index = find_record(
            &lines,
            "moderator",
            "observation",
            selection_of("carol", 2, "Protect"),
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
    #[should_panic(expected = "a devour goes to the rest of the living pack")]
    fn a_devour_passed_on_to_a_villager_is_caught() {
        // What a night session leaks is where its selections go, and since
        // ADR-0015 a relay is the only thing the moderator says about one.
        // dave and erin are the pack; alice is a villager, and a devour
        // passed on to her tells her both that somebody is being eaten and,
        // by the envelope, which of the two is a wolf.
        //
        // Who the relay really went to is left alone and alice is added, so
        // that the only thing wrong with the line is the villager among its
        // recipients: naming a fixed pair would risk naming the wolf whose
        // selection it is, which a different check catches first.
        let mut lines = fixture();
        let index = find(&lines, "moderator", "action", relay_in(1, "Devour"));
        let mut to: Vec<String> = lines[index]["message"]["recipients"]
            .as_array()
            .expect("a relay lists its recipients")
            .iter()
            .map(|who| who.as_str().expect("an id is a string").to_owned())
            .collect();
        to.push("alice".to_owned());
        lines[index]["message"]["recipients"] = json!(to);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a narration to the living goes to exactly the living")]
    fn a_narration_to_the_dead_is_caught() {
        // Night 3 is the first phase to begin with somebody dead: carol was
        // devoured on night 2. Sending its opening to the whole roster is
        // sending it to her too.
        let everyone = everyone();
        let lines = said(began(3, "Night"), |line| {
            line["message"]["recipients"] = json!(everyone);
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
            line["message"]["recipients"] = json!(everyone);
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
            line["message"]["payload"]["Narration"]["Assigned"]["pack"] = json!(["dave", "erin"]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a werewolf is told its pack")]
    fn a_werewolf_kept_from_its_pack_is_caught() {
        let lines = said(assigned("Werewolf", true), |line| {
            line["message"]["payload"]["Narration"]["Assigned"]["pack"] = json!([]);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "only a werewolf devours")]
    fn a_devour_passed_on_for_a_villager_is_caught() {
        // A devour the moderator accepted from somebody the rules never
        // put in the pack's session. The recipients are made the rest of
        // the living pack, so that it is who made the selection that trips
        // the check rather than where it went.
        let mut lines = fixture();
        let index = find(&lines, "moderator", "action", relay_in(1, "Devour"));
        selected_by(&mut lines[index], "alice");
        recipients(&mut lines[index], &["dave", "erin"]);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "phases alternate from the first night")]
    fn a_phase_out_of_order_is_caught() {
        let lines = said(began(2, "Night"), |line| {
            line["message"]["payload"]["Narration"]["PhaseBegan"]["round"] = json!(3);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a phase begins with the living as they are")]
    fn a_phase_that_miscounts_the_living_is_caught() {
        // Night 3 opens with six, carol having been devoured on night 2. A
        // phase claiming all seven is one whose own record of the living
        // disagrees with the deaths the same moderator narrated. The
        // recipients are widened to match, so that the miscount is what
        // trips the check rather than the routing.
        let everyone = everyone();
        let lines = said(began(3, "Night"), |line| {
            line["message"]["payload"]["Narration"]["PhaseBegan"]["living"] = json!(everyone);
            line["message"]["recipients"] = json!(everyone);
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a Night ends in one a death or one NoDeath")]
    fn a_night_that_says_nothing_of_deaths_is_caught() {
        check(
            &without("moderator", "action", narration("NoDeath")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "a Day ends in one a lynching or one NoLynch")]
    fn a_day_that_neither_lynches_nor_says_so_is_caught() {
        // A day may end with nobody lynched, but it says so with a
        // `NoLynch`. Every day of this fixture ends that way, so the
        // forgery is the other half of the same rule: drop day 1's
        // `NoLynch` and the day narrates neither a lynching nor its
        // absence, which is a day that did not happen.
        check(
            &without("moderator", "action", narration("NoLynch")),
            &config(),
        );
    }

    #[test]
    #[should_panic(expected = "an elimination is of a player the deciding session named most")]
    fn an_elimination_the_selections_do_not_call_for_is_caught() {
        // No day of this fixture lynches anybody, so the elimination to
        // forge is a devouring. On night 4 the pack splits, dave naming
        // alice and erin naming grace, and alice is the one taken; bob,
        // whom neither named, is devoured in her place. The last night, so
        // that no later phase's record of the living trips first, and the
        // outcome is corrected to match, so that whom the pack selected
        // is what the check trips on rather than the survivors.
        //
        // What it is checked against is the pack's *selections*, the ones
        // the moderator passed on. Nothing summarizes a session
        // (ADR-0015), so there is no summary to forge alongside the
        // death: the selections are the whole record of what the pack
        // decided.
        let mut lines = fixture();
        let death = find(&lines, "moderator", "action", eliminated("alice"));
        lines[death]["message"]["payload"]["Narration"]["Eliminated"] =
            json!({"who": "bob", "role": "Villager", "round": 4, "cause": "Devoured"});
        let outcome = find(&lines, "moderator", "action", narration("Outcome"));
        lines[outcome]["message"]["payload"]["Narration"]["Outcome"]["living"] =
            json!(["alice", "dave", "erin", "grace"]);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a death at night is of a player the doctor did not protect")]
    fn a_death_of_a_protected_player_is_caught() {
        // Night 2 is the last night with a doctor in it — carol, the
        // doctor, is the one devoured there. On it dave names carol, erin
        // names bob, carol protects erin, and carol dies. The forgery has
        // both werewolves name bob and carol protect bob, so the night's
        // victim is the very player the doctor covered.
        //
        // Bob dies in carol's place, so from that death on the two are
        // exchanged everywhere: carol goes on to play out bob's rounds 2
        // through 4 and survives, and bob, never eliminated again, still
        // dies exactly once. Rounds 3 and 4 open no Protect session, so
        // a doctor living through them leaves nothing unaccounted for.
        let mut lines = fixture();
        for who in ["dave", "erin"] {
            let index = find_record(
                &lines,
                "moderator",
                "observation",
                selection_of(who, 2, "Devour"),
            );
            target(&mut lines[index], "bob");
        }
        let protect = find_record(
            &lines,
            "moderator",
            "observation",
            selection_of("carol", 2, "Protect"),
        );
        target(&mut lines[protect], "bob");
        let death = find(&lines, "moderator", "action", eliminated("carol"));
        swap(&mut lines[death..], "carol", "bob");
        // A swap exchanges the names, not the role the death reveals.
        lines[death]["message"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Villager");
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a night with no death is one the doctor saved")]
    fn a_quiet_night_without_a_save_is_caught() {
        // On night 1 the pack splits between alice and bob, and carol
        // protects alice. Moving her protection to grace leaves a night
        // that reports no death although the pack named somebody and
        // nobody shielded them — neither of the two ways a night is
        // quiet (ADR-0011).
        let lines = heard_selection(selection_of("carol", 1, "Protect"), |line| {
            target(line, "grace");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the cause is the phase's")]
    fn a_lynching_by_night_is_caught() {
        // The night's deaths are the pack's and the day's are the village's,
        // and the rule runs both ways. No day of this fixture lynches
        // anybody, so the mismatch to forge is the other one: carol's
        // devouring on night 2, blamed on a vote that no night takes.
        let lines = said(devoured, |line| {
            line["message"]["payload"]["Narration"]["Eliminated"]["cause"] = json!("Lynched");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a death reveals the role")]
    fn a_death_revealing_the_wrong_role_is_caught() {
        // carol, the doctor, is devoured on night 2 — the fixture's first
        // death — and her death is made to announce a seer instead.
        let lines = said(devoured, |line| {
            line["message"]["payload"]["Narration"]["Eliminated"]["role"] = json!("Seer");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the winner is what the parity rule says")]
    fn a_winner_against_the_parity_rule_is_caught() {
        // The werewolves won this game, so it is the village that is the
        // claim the survivors do not bear out.
        let lines = said(narration("Outcome"), |line| {
            line["message"]["payload"]["Narration"]["Outcome"]["winner"] = json!("Village");
        });
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "the outcome names the survivors")]
    fn an_outcome_that_miscounts_the_survivors_is_caught() {
        let lines = said(narration("Outcome"), |line| {
            line["message"]["payload"]["Narration"]["Outcome"]["living"] = json!(["bob"]);
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
    #[should_panic(expected = "but observed something the moderator")]
    fn a_dead_player_that_hears_more_is_caught() {
        // A dead player observes nothing from its death onward — not its
        // own death, not a peer's selection, nothing (ADR-0012). The leak is
        // bob's copy of a narration that came after carol had died,
        // handed to the dead carol as well.
        //
        // The victim is the one the moderator eliminated, and no longer
        // one that was told so: carol is never sent its own death, so the
        // check reads the death off the moderator's records and this test
        // has to put the leak after that moment rather than after an
        // announcement carol never received.
        let mut lines = fixture();
        let mut leaked = lines[find(&lines, "bob", "observation", relay_in(2, "Nominate"))].clone();
        leaked["agent"] = json!("carol");
        let stop = lines
            .iter()
            .position(|line| line["agent"] == "carol" && line["control"] == "stop")
            .expect("carol is stopped");
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
    #[should_panic(expected = "logged before the episode's last stop")]
    fn a_reward_logged_after_the_episode_is_caught() {
        // A reward stamped after the run has finished for everybody is
        // scoring an episode that no longer existed.
        //
        // The bound is the episode's last stop and not the stop of the
        // agent being paid. Since ADR-0012 an agent may be stopped while
        // the others play on — a dead Werewolf player is — and it is
        // still paid at the end, after its own records have closed.
        // That is sound because a reward is logged rather than sent
        // (ADR-0007): nobody has to be there to receive it. So the test
        // pushes the stamp past the *last* stop in the file, which is
        // what no reward may follow.
        let mut lines = fixture();
        let index = reward_of(&lines, "grace");
        let end = lines
            .iter()
            .filter(|line| line["type"] == "control" && line["control"] == "stop")
            .map(|line| super::super::time(line, "t"))
            .max()
            .expect("somebody is stopped");
        lines[index]["t"] = json!(end + 1);
        super::super::actor::check(&lines);
    }

    #[test]
    fn a_reward_after_its_own_agents_stop_is_not_a_bug() {
        // The companion to the above, and the rule ADR-0012 replaced.
        //
        // A reward for an agent stopped mid-episode is logged at the end,
        // after that agent's own stop, and that is correct rather than
        // tolerated: the dead are paid like everybody else. The fixture
        // already contains the case — carol and frank both die — so this
        // asserts the shape is really there and that the checker accepts
        // it, which stops the bound above from being quietly tightened
        // back to the agent's own stop.
        let lines = fixture();
        let stops: BTreeMap<&str, u64> = lines
            .iter()
            .filter(|line| line["type"] == "control" && line["control"] == "stop")
            .map(|line| (super::super::agent(line), super::super::time(line, "t")))
            .collect();
        let late = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .filter(|line| {
                let who = super::super::agent(line);
                stops
                    .get(who)
                    .is_some_and(|&stop| super::super::time(line, "t") > stop)
            })
            .count();
        assert!(
            late > 0,
            "the fixture should hold a player stopped before it was paid"
        );
        super::super::actor::check(&lines);
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
    #[should_panic(expected = "begin with a start and end with a stop")]
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
    #[should_panic(expected = "begin with a start and end with a stop")]
    fn a_moderator_never_stopped_is_caught() {
        // The environment's own records have the same shape as everyone
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
    #[should_panic(expected = "a Investigate selection is nobody else's business")]
    fn a_seers_selection_shown_to_another_player_is_caught() {
        // A selection goes to the moderator alone, so a leak is no longer a
        // recipient: it is the seer naming somebody for the moderator to
        // forward its own business to.
        let mut lines = fixture();
        let index = find_record(
            &lines,
            "grace",
            "action",
            selection_of("grace", 1, "Investigate"),
        );
        seen_by(&mut lines[index], &["bob"]);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a devour selection is seen by the pack alone")]
    fn a_devour_selection_shown_to_a_villager_is_caught() {
        let mut lines = fixture();
        let index = find_record(&lines, "dave", "action", selection_of("dave", 1, "Devour"));
        seen_by(&mut lines[index], &["alice", "erin"]);
        check(&lines, &config());
    }

    #[test]
    #[should_panic(expected = "a selection is addressed to the moderator alone")]
    fn a_selection_that_never_reaches_the_moderator_is_caught() {
        // A selection addressed to a player instead of the moderator is the
        // very thing this design removes: there would be no check on
        // whether its session is still open (ADR-0014).
        let mut lines = fixture();
        let index = find_record(
            &lines,
            "alice",
            "action",
            selection_of("alice", 1, "Nominate"),
        );
        recipients(&mut lines[index], &["bob"]);
        check(&lines, &config());
    }
}
