//! The moderator: the game state machine as the episode's environment.
//!
//! [`Moderator`] is the [`Step`] around a [`Game`]. It is plumbing, and thin
//! by design: every rule of Werewolf lives in [`Game`], and this module only
//! folds the observations that arrive into calls on the game and turns the
//! [`Directive`]s that come back into [`Effect`]s. What it adds is the
//! invariants the runtime imposes on how those effects are addressed, the
//! reminders that make the game's clocks run, and the shape of the episode's
//! shutdown.
//!
//! # What the moderator does
//!
//! Being started begins the game, which is what [`start`](Step::start) is
//! for: the moderator is the actor that opens play, and it does so before
//! anybody has spoken to it. It is also where the players are started, since
//! an episode's environment is the only actor that may send a [`Control`]
//! (ADR-0016). Thereafter a [`Select`](super::Select) from a player is
//! recorded, and a [`Look`] the moderator set for itself is a prompt to
//! check its clocks. The moderator never sees the control that started it,
//! nor the one that stops it; those are the runtime's.
//!
//! A [`Narration`](super::Narration) arriving from a player is a bug, and
//! the moderator panics rather than run a game whose state it cannot vouch
//! for. So is a relay, and so is a reminder from anybody but itself. So is a
//! selection in a session the rules never make that player a member of; the
//! moderator checks that when the selection arrives, rather than handing out
//! permission in advance (ADR-0014).
//!
//! Once the game is over the moderator emits nothing but the one stop that
//! ends the episode, whatever arrives. The episode is winding down, and a
//! late selection is not the game's problem.
//!
//! # The clocks are reminders the moderator sets for itself
//!
//! The old runtime asked a handler for its next deadline and called
//! `timeout` when it passed. Under ADR-0016 there is no such hook: a handler
//! that wants to hear from itself later sets a [`Reminder`], which comes back
//! as an ordinary message from itself, and **reminders accumulate — each
//! fires once and nothing is cancelled**.
//!
//! So the moderator sets one whenever [`Game::next_deadline`] names an
//! instant it has not already reminded itself about: at each session's hard
//! limit when the session opens, and again at the end of the quiet period
//! each time a change of mind restarts it. When a reminder arrives it calls
//! [`Game::expire`] with the instant the reminder *arrived*, and the game
//! closes whatever that instant closed.
//!
//! **A reminder for a limit that has since moved is ignored.** Nothing
//! cancels it: it arrives, the game finds that no session's deadline has
//! passed, and nothing comes of it. That is the whole of the handling, and
//! it is why the [`Look`] payload carries provenance and no authority.
//!
//! One reminder per distinct deadline is what the moderator tracks, in
//! its `reminded` set. Not to cancel anything — it cannot —
//! but so that a burst of selections that leaves the same deadline in place
//! does not set the same reminder once per selection.
//!
//! # Every message of this game names somebody
//!
//! Every message carries an explicit recipient set, and the choice of
//! recipients is the whole hidden-information mechanism (ADR-0004).
//! ADR-0004 made one exception, sending the final [`Outcome`] to every
//! player, living and dead, because it was a dead player's terminal reward
//! signal. A reward is now logged rather than said (ADR-0007), so the
//! exception is withdrawn: the outcome is narrated to the living like any
//! other narration, and a dead player's records end at the announcement
//! of its own elimination, then its `Stop`.
//!
//! A reminder is the one message of this game addressed to nobody else. It is
//! addressed to the moderator itself, which is what a [`Reminder`] is: the
//! one way a message reaches the actor that sent it (ADR-0016).
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
//! # How the episode ends, in two steps
//!
//! A `Stop` **preempts everything** (ADR-0016): it takes effect when the
//! recipient's perception thread sees it, ahead of anything in its inbox. So
//! a moderator that narrated the outcome and stopped everybody in the same
//! call would usually have the narration overtaken by the stop, and the log
//! would record it `undelivered` — the survivors would never hear how the
//! game ended. ADR-0016 says what to do instead: **an environment that wants
//! its last word heard sets a reminder and stops everybody when it arrives.**
//!
//! So the call in which the game ends does four things:
//!
//! 1. **narrate** the [`Outcome`] to the living, as an ordinary
//!    [`Effect::Act`];
//! 2. **publish** it on the [`Sender`], for whoever ran the episode;
//! 3. **pay** every player, living and dead, one [`Effect::Reward`] each:
//!    **+1** if its role's faction won and **−1** otherwise, as
//!    [`Game::rewards`] works out;
//! 4. set a **farewell reminder**, a short interval ahead, and nothing else.
//!
//! and the call the reminder wakes does the fifth:
//!
//! 5. **stop** every actor still running — the living and **the moderator
//!    itself** — with one [`Effect::Command`]. That is how an episode of this
//!    runtime ends: the episode holds no view of what is in flight and stops
//!    nobody of its own accord until its time limit runs out (ADR-0016).
//!
//! **Not the dead.** A dead player was stopped where it died, so its threads
//! have ended and its channels are gone; naming it again would be a send the
//! router refuses, which would fail the moderator's own thread. It would also
//! claim in that player's records that it was told to stop after it had
//! already stopped. So "every actor" means every actor there is still an
//! actor to stop, which is what the episode is waiting on: a dead player
//! reported when it was stopped at its death, and the episode counted it
//! then.
//!
//! The rewards come before the stop because they are what the episode was
//! for. They are logged rather than sent (ADR-0007), so their position among
//! the effects changes nothing a player sees; what it does is put each reward
//! in the log ahead of the `Stop` control that closes the records it belongs
//! to, which is the ordering a reader can then rely on.
//!
//! A **dead** player is stopped where it dies, in the same call its death is
//! announced (ADR-0012), and it is not among those told, so there is nothing
//! of its own for its stop to overtake. It is the one stop that does not
//! wait.
//!
//! A game that never reaches an outcome is nobody's to notice here: the
//! episode runs past its hard time limit and reports
//! [`EpisodeError::Timeout`](crate::EpisodeError).

use crossbeam_channel::Sender;

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use super::game::{Directive, Game};
use super::message::{Look, Message, Outcome};
use crate::clock::Clock;
use crate::message::{ActorId, Control};
use crate::{Action, Effect, Observation, Reminder, Step};

/// The least time the moderator leaves between announcing the outcome and
/// stopping everybody.
///
/// A [`Stop`](Control::Stop) preempts everything in a player's inbox
/// (ADR-0016), so a moderator that narrated the outcome and stopped the
/// survivors in the same call would usually have the narration logged
/// `undelivered`: nobody would have heard how the game ended. ADR-0016's answer
/// is a reminder, and a reminder needs an interval.
///
/// **The interval is a race and there is nothing to make it not one.** Nothing
/// acknowledges an observation, so the moderator cannot know that a player has
/// drained its inbox; all it can do is wait long enough that every player's
/// perception thread has been scheduled at least once. What that takes is a
/// fact about the machine and not about the game, and on a loaded one — several
/// test suites at a time, which is how this repository stresses it — seven
/// threads can take orders of magnitude longer to be scheduled than they do
/// idle. Measured over dozens of concurrent games the longest a survivor took
/// was single-digit milliseconds; this is that with two orders of magnitude of
/// room, because the cost of being wrong is a game whose survivors never heard
/// how it ended and the cost of being generous is this much wall-clock time per
/// game.
///
/// So the moderator waits the longer of this floor and the configuration's own
/// shortest quiet period, which is the interval that configuration already
/// asserts is long enough for a player to be scheduled and answer: a game whose
/// sessions close on a quiet period of *q* is a game in which a player that
/// takes longer than *q* to be heard has already lost its selections. See
/// [`Moderator::farewell`].
const FAREWELL: Duration = Duration::from_millis(250);

/// The actor that runs a game of Werewolf: a [`Game`] behind a [`Step`].
#[derive(Debug)]
pub struct Moderator {
    /// The moderator's own id, so that it can stop itself: an episode ends
    /// when the environment stops every actor, itself included (ADR-0016),
    /// and the only actor that knows the moderator's seat is the moderator.
    me: ActorId,
    game: Game,
    outcome: Sender<Outcome>,
    /// The deadlines the moderator has already reminded itself about.
    ///
    /// Not a cancellation list — nothing is cancelled (ADR-0016) — but the
    /// answer to "have I already asked to be woken then?", so that a burst of
    /// selections leaving one deadline in place sets one reminder and not one
    /// per selection. A deadline that has passed is never set again, since
    /// [`Game::next_deadline`] only ever names one in the future of the
    /// session that owns it.
    reminded: BTreeSet<Instant>,
    /// Whether the farewell reminder has been set, so that it is set once and
    /// the stop it brings is sent once.
    farewell: bool,
    /// Whether everybody has been stopped, so that nothing follows the stop.
    stopped: bool,
}

impl Moderator {
    /// A moderator seated under `me` that runs `game` and, when it ends,
    /// sends its outcome on `outcome`.
    ///
    /// The game must not have begun: the moderator begins it when the
    /// episode starts it.
    #[must_use]
    pub fn new(me: impl Into<ActorId>, game: Game, outcome: Sender<Outcome>) -> Self {
        Self {
            me: me.into(),
            game,
            outcome,
            reminded: BTreeSet::new(),
            farewell: false,
            stopped: false,
        }
    }

    /// Every player in the game, living and dead: whom the moderator starts
    /// and, when the game is over, stops.
    fn players(&self) -> BTreeSet<ActorId> {
        self.game.players().cloned().collect()
    }

    /// What one observation makes the game say.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration or a relay: both are the
    /// moderator's to send, so either is a player claiming to be it. Or if
    /// anybody but the moderator sends it a reminder, which only the
    /// moderator sets and only for itself.
    fn fold(&mut self, observation: &Observation<Message>) -> Vec<Directive> {
        let sender = &observation.message.sender;
        let now = observation.at;
        match &observation.message.payload {
            // The message's own sequence number goes in as well as the
            // arrival: the relay's envelope carries the player's `(sender,
            // seq)` so it joins back to the player's action, while `now` is
            // when the moderator perceived it, which is what the session
            // clocks run on. **Not when `step` runs**: a selection that
            // landed a nanosecond before its session's limit counts even if
            // the handler was busy and only got to it afterwards (ADR-0018).
            Message::Select(selection) => {
                let mut directives =
                    self.game
                        .select(sender, selection, observation.message.seq, now);
                // A deadline that passed while this selection waited on the
                // handler closes here, in the same call, rather than waiting
                // for the reminder that is still on its way.
                directives.extend(self.game.expire(now));
                directives
            }
            // A reminder the moderator set for itself. Whether anything has
            // actually expired is the game's to say, and a reminder for a
            // deadline that has since moved closes nothing (ADR-0018).
            Message::Reminder(_) => {
                self.reminded.remove(&now);
                self.game.expire(now)
            }
            Message::Narration(_) => panic!("{sender} sent the moderator a narration"),
            // Relaying is the moderator's own move, so a player sending one
            // is a player claiming to be the moderator.
            Message::Relayed(_) => panic!("{sender} sent the moderator a relay"),
        }
    }

    /// Everything the directives say, said; then the reminders the game's
    /// clocks now call for; then, if the game has just ended, the outcome
    /// published on the channel, every player paid, and the farewell reminder
    /// set. The order is the shutdown sequence; see the
    /// [module documentation](self).
    ///
    /// **The farewell reminder is yielded lazily**, and that is load-bearing.
    /// The runtime carries out each effect as the iterator yields it
    /// (ADR-0016), so a reminder in a `Vec` would have had its deadline read
    /// before the outcome it is waiting on was even sent, and the interval
    /// would be eaten by the very sends it is meant to cover. Yielded last, it
    /// reads the clock after the outcome has gone out, which is the instant the
    /// wait should start from.
    fn say(&mut self, directives: Vec<Directive>) -> Said {
        let mut effects: Vec<Effect<i32, Message>> = directives.into_iter().map(send).collect();
        let Some(outcome) = self.game.outcome() else {
            effects.extend(self.reminders());
            return Said::of(effects);
        };
        if self.farewell {
            return Said::of(effects);
        }
        self.farewell = true;
        // The caller may have dropped the receiver. That is not the game's
        // problem: the in-world announcement is the record.
        let _ = self.outcome.send(outcome.clone());
        let rewards = self
            .game
            .rewards()
            .expect("a game with an outcome has rewards");
        effects.extend(
            rewards
                .into_iter()
                .map(|(to, reward)| Effect::Reward { to, reward }),
        );
        // Not the stop: a `Stop` preempts everything, so stopping the living in
        // the same call as the outcome they are owed would usually mean they
        // never observed it. The reminder is what buys the ordering
        // (ADR-0016), and it is yielded rather than pushed for the reason
        // above.
        Said {
            effects: effects.into_iter(),
            farewell: Some((self.farewell(), Message::Reminder(Look::Farewell))),
        }
    }

    /// A reminder at the game's next deadline, unless the moderator has
    /// already set one for that instant.
    ///
    /// At most one effect, because [`Game::next_deadline`] names at most one
    /// instant: the earliest anything could close. Every later deadline is
    /// reached by this being called again after the reminder for the earlier
    /// one arrives, which is what makes a list of deadlines unnecessary.
    fn reminders(&mut self) -> Vec<Effect<i32, Message>> {
        let Some(deadline) = self.game.next_deadline() else {
            return Vec::new();
        };
        if !self.reminded.insert(deadline) {
            return Vec::new();
        }
        vec![Effect::Act(Action::Remind(Reminder::new(
            deadline,
            self.session_reminder(),
        )))]
    }

    /// How long to leave between announcing the outcome and stopping
    /// everybody: the longer of [`FAREWELL`] and the shortest quiet period any
    /// of this game's sessions runs on.
    ///
    /// The quiet period is in it because a configuration that sets one is
    /// asserting that it is long enough for a player to be scheduled and
    /// answer; a game whose clocks are generous should be generous here too,
    /// and one whose clocks are tight still gets the floor. See [`FAREWELL`].
    fn farewell(&self) -> Duration {
        FAREWELL.max(self.game.shortest_quiet_period())
    }

    /// A session reminder's payload: which phase of which round the moderator
    /// was watching when it set it, which is provenance and nothing the game
    /// reads back.
    fn session_reminder(&self) -> Message {
        let (phase, round) = self.game.phase_now();
        Message::Reminder(Look::Session { round, phase })
    }

    /// The one command that ends the episode: every actor still running
    /// stopped, **the moderator itself** among them (ADR-0016).
    ///
    /// The dead are left out, for the reason the [module
    /// documentation](self) gives: each was stopped where it died, and its
    /// channels have gone with its threads.
    fn stop_everybody(&mut self) -> Vec<Effect<i32, Message>> {
        if self.stopped {
            return Vec::new();
        }
        self.stopped = true;
        let mut running = self.game.living().clone();
        running.insert(self.me.clone());
        vec![Effect::command(running, Control::Stop)]
    }
}

/// What one call to the moderator yields: the effects it decided, and — on the
/// call that ends the game — a farewell reminder whose deadline is read at the
/// moment the runtime reaches it.
///
/// It exists so that the reminder can be **genuinely lazy**. The handler thread
/// carries out each effect as the iterator yields it (ADR-0016), so a reminder
/// sitting in a `Vec` would have had its deadline read before the outcome it is
/// waiting on was sent, and the interval would be spent on the very sends it
/// exists to cover. Under load those sends take tens of milliseconds, which is
/// a real share of the interval and was enough to lose the outcome.
///
/// Building the reminder in [`IntoIterator::into_iter`] is not late enough
/// either, because `for item in items` calls that before it pulls anything. So
/// this is an iterator of its own whose last `next` is where the clock is read.
///
/// See [`Moderator::say`] and [`FAREWELL`].
struct Said {
    effects: std::vec::IntoIter<Effect<i32, Message>>,
    /// How long to wait and what to bring back, taken when the effects run
    /// out; `None` on every call but the one that ends the game, and taken
    /// once.
    farewell: Option<(Duration, Message)>,
}

impl Said {
    /// What a call that did not end the game yields: these effects and no
    /// reminder.
    fn of(effects: Vec<Effect<i32, Message>>) -> Self {
        Self {
            effects: effects.into_iter(),
            farewell: None,
        }
    }
}

impl Iterator for Said {
    type Item = Effect<i32, Message>;

    /// The effects in order and then, once they have all been carried out, the
    /// farewell reminder, whose deadline is read **here**: this call happens
    /// after the handler thread has sent everything ahead of it, which is the
    /// instant the wait should start from.
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(effect) = self.effects.next() {
            return Some(effect);
        }
        let (wait, payload) = self.farewell.take()?;
        Some(Effect::Act(Action::Remind(Reminder::new(
            Instant::now() + wait,
            payload,
        ))))
    }
}

/// The effect that carries one directive.
///
/// Most are something said; a [`Directive::Stop`] is the game putting a
/// player out of it, which is a control rather than a message (ADR-0012).
fn send(directive: Directive) -> Effect<i32, Message> {
    match directive {
        Directive::Narrate { to, narration } => {
            Effect::Act(Action::to(to, Message::Narration(narration)))
        }
        // A relayed selection is the moderator's own message, numbered among
        // the moderator's, and the envelope inside it names the player whose
        // selection it is (ADR-0018). No message claims a sender other than
        // the actor that sent it.
        Directive::Forward { envelope, to } => {
            Effect::Act(Action::to(to, Message::Relayed(envelope)))
        }
        Directive::Stop { who } => Effect::command([who], Control::Stop),
    }
}

impl Step<i32, Message> for Moderator {
    /// Starts every player, then begins the game and says what it wants
    /// said at the start: the roles, that the first night has begun, and the
    /// first reminders its sessions' clocks call for. The players take it
    /// from there (ADR-0014).
    ///
    /// The `Start` comes first among the effects, and each is carried out as
    /// it is yielded (ADR-0016), so a player is started before anything is
    /// addressed to it. A control is popped ahead of the inbox in any case.
    ///
    /// **The first night begins at the clock's origin**, which is the
    /// episode's: the one instant every actor and the log share (ADR-0017).
    /// The moderator is the only actor whose rules turn on an absolute
    /// instant, and measuring the first night's clocks from the origin puts
    /// the game's whole timeline on the log's, so a session's limit falls
    /// where a reader of the log reads it. The episode starts the moderator
    /// the moment after it captures the origin, so the two differ by the cost
    /// of spawning a thread.
    ///
    /// The clock is not kept. Every later instant the rules turn on is an
    /// arrival or a deadline read off the same monotonic clock, and the
    /// origin is what the log converts against rather than anything the game
    /// reads.
    fn start(&mut self, clock: Clock) -> impl IntoIterator<Item = Effect<i32, Message>> {
        let origin = clock.origin();
        let opening = self.game.begin(origin);
        let mut effects = vec![Effect::command(self.players(), Control::Start)];
        // Collected rather than chained lazily, because a game cannot be over
        // before it has begun: `say` has no farewell reminder to yield here,
        // and so nothing whose deadline it matters when the clock is read for.
        effects.extend(self.say(opening));
        effects
    }

    /// Folds the observation into the game and says what the game wants
    /// said. The observation that ends the game also sends the outcome on
    /// the channel, pays every player and sets the farewell reminder; the
    /// reminder that follows stops everybody, and after that nothing,
    /// whatever arrives.
    ///
    /// # Panics
    ///
    /// If a player sends the moderator a narration, a relay or a reminder, or
    /// if a selection is one the game cannot accept; see [`Game::select`].
    fn step(
        &mut self,
        observation: Observation<Message>,
    ) -> impl IntoIterator<Item = Effect<i32, Message>> {
        // Whether the game is over is the game's to say, and it is asked
        // before every observation, so the one that ends it is the last
        // folded. What comes after it is the farewell reminder, which is the
        // moderator's own and is the one thing still to act on: it brings the
        // stop that ends the episode. A selection that arrives in the
        // meantime is not folded, and the moderator says nothing about it.
        if self.game.outcome().is_some() {
            // Only the **farewell** ends the episode. A stale session reminder
            // arriving after the outcome is one whose wait the game overtook,
            // and stopping on it would cut the farewell's wait short — which is
            // the wait the survivors' last narration depends on.
            let effects = match &observation.message.payload {
                Message::Reminder(Look::Farewell) if observation.message.sender == self.me => {
                    self.stop_everybody()
                }
                _ => Vec::new(),
            };
            return Said::of(effects);
        }
        let directives = self.fold(&observation);
        self.say(directives)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crossbeam_channel::{Receiver, TryRecvError, unbounded};

    use super::*;

    use crate::message::{ActorId, Envelope};
    use crate::testing::{BASE, fast, id, ids, town, village};
    use crate::werewolf::assignment::Assignment;
    use crate::werewolf::config::{DayTiming, NightTiming, Timing};
    use crate::werewolf::message::{Narration, Phase, Round, Select, SessionKind};
    use crate::werewolf::role::Faction;
    use crate::werewolf::role::Role::{self, Doctor, Seer, Villager, Werewolf};

    const MODERATOR: &str = "moderator";
    const SEED: u64 = 20_260_918;

    /// The instants these tests name, offset from the one fixed base every
    /// test module in the crate shares; see [`testing::BASE`](crate::testing).
    use crate::testing::at_millis as at;

    fn moderator(assignment: Assignment) -> (Moderator, Receiver<Outcome>) {
        timed(assignment, fast())
    }

    /// A moderator over `assignment` whose sessions run on `timing`.
    fn timed(assignment: Assignment, timing: Timing) -> (Moderator, Receiver<Outcome>) {
        let (sender, receiver) = unbounded();
        let game = Game::new(assignment, SEED, timing);
        (Moderator::new(MODERATOR, game, sender), receiver)
    }

    /// A message from a player to the moderator, arriving at `at`.
    ///
    /// The arrival is what a session clocks a selection by (ADR-0018), so
    /// every test here names it. Which of that player's messages it is plays
    /// no part in the fold, so one stand-in number serves most of them.
    fn from_player(who: &str, at: Instant, payload: Message) -> Observation<Message> {
        Observation {
            at,
            message: crate::Message::new(who, [MODERATOR], 0, payload),
        }
    }

    fn selects(
        who: &str,
        round: Round,
        kind: SessionKind,
        target: &str,
        at: Instant,
    ) -> Observation<Message> {
        from_player(
            who,
            at,
            Message::Select(Select {
                round,
                kind,
                target: id(target),
                seen_by: BTreeSet::new(),
            }),
        )
    }

    /// A selection whose `seen_by` names `audience`, for a test that reads
    /// whether the game accepted it off whether the moderator relayed it.
    fn seen_by<const N: usize>(
        who: &str,
        round: Round,
        kind: SessionKind,
        target: &str,
        audience: [&str; N],
        at: Instant,
    ) -> Observation<Message> {
        from_player(
            who,
            at,
            Message::Select(Select {
                round,
                kind,
                target: id(target),
                seen_by: ids(audience),
            }),
        )
    }

    /// One of the moderator's own reminders coming back to it, at `at`.
    ///
    /// A reminder is delivered as an ordinary message from the actor that set
    /// it (ADR-0016), so this is what the runtime hands the moderator. The
    /// payload is the one the moderator set, taken from the effect it
    /// returned, because a reminder's payload is provenance and a test that
    /// invented one would be inventing a fact about the log.
    fn reminder(payload: Message, at: Instant) -> Observation<Message> {
        Observation {
            at,
            message: crate::Message::new(MODERATOR, [MODERATOR], 0, payload),
        }
    }

    /// Everything one call to [`Step::step`] produced, as a `Vec`.
    ///
    /// `step` returns `impl IntoIterator`, which is not a collection: it
    /// borrows the handler, so it has to be drained before the handler is
    /// touched again. Every test here drains it at once, which is also what
    /// the handler thread does.
    fn stepped(
        moderator: &mut Moderator,
        observation: Observation<Message>,
    ) -> Vec<Effect<i32, Message>> {
        moderator.step(observation).into_iter().collect()
    }

    /// Everything the moderator's `start` produced, as a `Vec`.
    ///
    /// The clock's origin is [`at(0)`](at), the one fixed base every test
    /// module in the crate shares, because the moderator begins the game at
    /// the origin: a test that names a deadline in seconds from `at(0)` is
    /// naming the instant the moderator will pick. A clock an hour in the
    /// future also means no deadline named here ever fires on a real clock.
    fn started(moderator: &mut Moderator) -> Vec<Effect<i32, Message>> {
        moderator
            .start(Clock::from_origin(at(0)))
            .into_iter()
            .collect()
    }

    /// What some effects said and did, with the **farewell** reminder's
    /// deadline replaced by a fixed instant.
    ///
    /// Everything that turns on the game is left exactly as it was — every
    /// message, every control, every reward, and every *session* reminder's
    /// deadline, which is a session's clock and must match. The one thing that
    /// need not is the farewell reminder's deadline, which is read off the
    /// clock and so differs between two moderators handed the same
    /// observations. See [`FAREWELL`].
    ///
    /// The farewell is the reminder that comes after the outcome, which is why
    /// `ended` is what tells the two apart: the call that announces the outcome
    /// is the one that sets it, and no session reminder is set after that.
    fn said_and_done(effects: &[Effect<i32, Message>]) -> Vec<Effect<i32, Message>> {
        let ended = effects.iter().any(|effect| {
            matches!(
                effect,
                Effect::Act(Action::Send {
                    payload: Message::Narration(Narration::Outcome(_)),
                    ..
                })
            )
        });
        effects
            .iter()
            .map(|effect| match effect {
                Effect::Act(Action::Remind(reminder)) if ended => Effect::Act(Action::Remind(
                    Reminder::new(*BASE, reminder.payload.clone()),
                )),
                other => other.clone(),
            })
            .collect()
    }

    /// A stub player: whom it selects, out of the targets the rules
    /// permit it.
    ///
    /// The space is the game's own, so a stub cannot select outside it; the
    /// doctor's "not last night's patient" in particular is the rules'
    /// business rather than every stub's.
    type Strategy = fn(SessionKind, &[ActorId]) -> ActorId;

    /// Selects the first target the rules permit.
    fn first_other(_: SessionKind, space: &[ActorId]) -> ActorId {
        space.first().unwrap().clone()
    }

    /// Selects the last target the rules permit.
    fn last_other(_: SessionKind, space: &[ActorId]) -> ActorId {
        space.last().unwrap().clone()
    }

    /// Selects the last permitted target by night and the first by day,
    /// so that the pack and the village disagree about whom to blame.
    fn two_minded(kind: SessionKind, space: &[ActorId]) -> ActorId {
        match kind.phase() {
            Phase::Night => last_other(kind, space),
            Phase::Day => first_other(kind, space),
        }
    }

    /// The actions among some effects, in order, each as its recipients and
    /// its payload.
    ///
    /// A reminder is left out: it is the moderator's clock running and not
    /// something it said, and [`reminders_among`] is what reads those.
    fn actions(effects: &[Effect<i32, Message>]) -> Vec<(Vec<ActorId>, Message)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Act(Action::Send { to, payload }) => Some((to.clone(), payload.clone())),
                Effect::Act(Action::Remind(_)) | Effect::Command { .. } | Effect::Reward { .. } => {
                    None
                }
            })
            .collect()
    }

    /// The reminders among some effects, in order: each one's deadline and
    /// the payload it will bring back.
    fn reminders_among(effects: &[Effect<i32, Message>]) -> Vec<(Instant, Message)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Act(Action::Remind(reminder)) => {
                    Some((reminder.deadline, reminder.payload.clone()))
                }
                Effect::Act(Action::Send { .. })
                | Effect::Command { .. }
                | Effect::Reward { .. } => None,
            })
            .collect()
    }

    /// The one reminder among some effects.
    ///
    /// # Panics
    ///
    /// Unless there is exactly one. [`Game::next_deadline`] names at most one
    /// instant, so a call that sets two would be the moderator setting a
    /// reminder it has no deadline for.
    fn one_reminder(effects: &[Effect<i32, Message>]) -> (Instant, Message) {
        let set = reminders_among(effects);
        let [reminder] = set.as_slice() else {
            panic!("exactly one reminder: {set:?}");
        };
        reminder.clone()
    }

    /// The controls among some effects, in order.
    fn controls(effects: &[Effect<i32, Message>]) -> Vec<(BTreeSet<ActorId>, Control)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Command { to, control } => Some((to.iter().cloned().collect(), *control)),
                Effect::Act(_) | Effect::Reward { .. } => None,
            })
            .collect()
    }

    /// The rewards among some effects, by the actor paid, in the order the
    /// moderator assigned them.
    fn rewards(effects: &[Effect<i32, Message>]) -> Vec<(ActorId, i32)> {
        effects
            .iter()
            .filter_map(|effect| match effect {
                Effect::Reward { to, reward } => Some((to.clone(), *reward)),
                Effect::Act(_) | Effect::Command { .. } => None,
            })
            .collect()
    }

    /// The selections stub players following `strategy` make when a phase
    /// begins, each arriving at `at`.
    ///
    /// This is a stub of [`Player`](super::super::Player), and it acts the
    /// way one does (ADR-0014): a phase beginning is what makes a player select, and
    /// each recipient asks its own role what that phase wants of it rather
    /// than waiting to be told.
    ///
    /// A player the rules leave nowhere to select says nothing, which is
    /// the same thing the game means by leaving it out of the session.
    fn respond(
        actions: &[(Vec<ActorId>, Message)],
        strategy: Strategy,
        game: &Game,
        roles: &Assignment,
        at: Instant,
    ) -> Vec<Observation<Message>> {
        actions
            .iter()
            .filter_map(|(to, payload)| match payload {
                Message::Narration(Narration::PhaseBegan { round, phase, .. }) => {
                    Some((*round, *phase, to))
                }
                _ => None,
            })
            .flat_map(|(round, phase, to)| {
                to.iter().filter_map(move |who| {
                    // Its own role, from the deal, is what tells a stub
                    // whether this phase wants anything of it.
                    let kind = roles.role(who)?.asked_in(phase)?;
                    let space = game.action_space_for(who, kind);
                    if space.is_empty() {
                        return None;
                    }
                    Some(selects(
                        who.as_str(),
                        round,
                        kind,
                        strategy(kind, &space).as_str(),
                        at,
                    ))
                })
            })
            .collect()
    }

    /// Plays a whole game, with stub players following `strategy`, and
    /// returns every effect the moderator produced in order.
    ///
    /// The game opens with `start`, as the handler thread opens it, and each
    /// selection then arrives in a call of its own, as the runtime hands them
    /// over one at a time (ADR-0008). **The reminders the moderator sets are
    /// delivered by hand**, at their own deadlines, which is exactly what the
    /// runtime's timer does and what makes a game here run on a clock the
    /// test holds rather than on the wall.
    ///
    /// The instants are a test's own: `at` is far enough in the future that
    /// no deadline named from it ever fires on a real clock, so nothing here
    /// races anything.
    fn play(
        moderator: &mut Moderator,
        strategy: Strategy,
        roles: &Assignment,
    ) -> Vec<Effect<i32, Message>> {
        /// Far enough apart that one selection never lands after the next
        /// phase's clocks have started.
        const STEP: u64 = 1;

        let mut clock = 0;
        let mut produced = started(moderator);
        let mut pending = respond(
            &actions(&produced),
            strategy,
            &moderator.game,
            roles,
            at(clock),
        );
        let mut due: Vec<(Instant, Message)> = reminders_among(&produced);
        while moderator.game.outcome().is_none() {
            clock += STEP;
            let mut effects: Vec<Effect<i32, Message>> = Vec::new();
            for observation in pending {
                effects.extend(stepped(moderator, observation));
            }
            // Then the earliest reminder outstanding, which is what closes a
            // session. Each is delivered at its own deadline, in order, and a
            // stale one is delivered too: a reminder for a limit that has
            // since moved is not cancelled, it simply comes to nothing
            // (ADR-0016).
            due.extend(reminders_among(&effects));
            due.sort_by_key(|(deadline, _)| *deadline);
            if let Some((deadline, payload)) = due.first().cloned() {
                due.remove(0);
                effects.extend(stepped(moderator, reminder(payload, deadline)));
            }
            pending = respond(
                &actions(&effects),
                strategy,
                &moderator.game,
                roles,
                at(clock),
            );
            due.extend(reminders_among(&effects));
            let latest = effects.len();
            produced.extend(effects);
            let _ = latest;
            assert!(
                clock < 10_000,
                "a stub game should have ended long before now"
            );
        }
        // The farewell reminder the ending call set is what brings the stop
        // that ends the episode, so the game is not finished playing until it
        // has been delivered (ADR-0016).
        due.sort_by_key(|(deadline, _)| *deadline);
        for (deadline, payload) in due {
            produced.extend(stepped(moderator, reminder(payload, deadline)));
        }
        produced
    }

    /// A game played out: its assignment, how the game says it ended, the
    /// receiver of its outcome, every message the moderator sent, every
    /// control it asked for, every reward it paid, and the whole sequence
    /// of effects it produced.
    struct Played {
        assignment: Assignment,
        outcome: Outcome,
        receiver: Receiver<Outcome>,
        sent: Vec<(Vec<ActorId>, Message)>,
        commanded: Vec<(BTreeSet<ActorId>, Control)>,
        paid: Vec<(ActorId, i32)>,
        effects: Vec<Effect<i32, Message>>,
    }

    /// One game per stub strategy, over both assignments, played to its
    /// outcome.
    ///
    /// Between them these reach every terminal state a game has, which
    /// `the_stub_players_between_them_reach_every_terminal_state` asserts.
    fn played_games() -> Vec<Played> {
        let mut games = Vec::new();
        for assignment in [village(), town()] {
            for strategy in [
                first_other as Strategy,
                last_other as Strategy,
                two_minded as Strategy,
            ] {
                let (mut moderator, receiver) = moderator(assignment.clone());
                let effects = play(&mut moderator, strategy, &assignment);
                let outcome = moderator
                    .game
                    .outcome()
                    .expect("a played game has an outcome")
                    .clone();
                games.push(Played {
                    assignment: assignment.clone(),
                    outcome,
                    receiver,
                    sent: actions(&effects),
                    commanded: controls(&effects),
                    paid: rewards(&effects),
                    effects,
                });
            }
        }
        games
    }

    /// The recipients of the outcome narration among some messages, and the
    /// outcome it announced.
    fn announced_outcome(sent: &[(Vec<ActorId>, Message)]) -> (BTreeSet<ActorId>, &Outcome) {
        sent.iter()
            .find_map(|(to, payload)| match payload {
                Message::Narration(Narration::Outcome(outcome)) => {
                    Some((to.iter().cloned().collect(), outcome))
                }
                _ => None,
            })
            .expect("no outcome was announced")
    }

    fn narrate<const N: usize>(to: [&str; N], narration: Narration) -> (Vec<ActorId>, Message) {
        (
            to.iter().map(|who| id(who)).collect(),
            Message::Narration(narration),
        )
    }

    fn assigned<const N: usize>(to: &str, role: Role, pack: [&str; N]) -> (Vec<ActorId>, Message) {
        narrate(
            [to],
            Narration::Assigned {
                role,
                pack: ids(pack),
            },
        )
    }

    #[test]
    fn starting_the_moderator_starts_the_players_begins_the_game_and_sets_a_reminder() {
        let everyone = ["alice", "bob", "carol", "dave", "erin"];
        let (mut moderator, _receiver) = moderator(village());
        let opening = started(&mut moderator);
        assert_eq!(
            controls(&opening),
            [(ids(everyone), Control::Start)],
            "every player is started, and nobody is stopped yet"
        );
        assert_eq!(
            actions(&opening),
            [
                assigned("alice", Villager, []),
                assigned("bob", Werewolf, ["bob"]),
                assigned("carol", Seer, []),
                assigned("dave", Doctor, []),
                assigned("erin", Villager, []),
                narrate(
                    everyone,
                    Narration::PhaseBegan {
                        round: Round::new(1),
                        phase: Phase::Night,
                        living: ids(everyone),
                    }
                ),
            ],
            "the deals and the phase, and nothing asked of anybody"
        );
        // And the first night's clocks: one reminder, at the earliest instant
        // anything could close, which for a night nobody has selected in is
        // the earliest of the three hard limits (ADR-0018).
        let (deadline, payload) = one_reminder(&opening);
        assert_eq!(
            Some(deadline),
            moderator.game.next_deadline(),
            "the reminder is set for the game's next deadline"
        );
        assert_eq!(
            payload,
            Message::Reminder(Look::Session {
                round: Round::FIRST,
                phase: Phase::Night,
            }),
            "and says which phase's clocks it was set for"
        );
    }

    /// Timing whose numbers are far enough apart to be read off a test's
    /// assertions: a one-second quiet period, a ten-second night limit and a
    /// five-second day. No deadline named from these ever fires on a real
    /// clock, because [`crate::testing::BASE`] is an hour in the future.
    const SPACED: Timing = Timing {
        day_cap: None,
        pack: SPACED_NIGHT,
        seer: SPACED_NIGHT,
        doctor: SPACED_NIGHT,
        day: DayTiming {
            limit: Duration::from_secs(5),
        },
    };

    const SPACED_NIGHT: NightTiming = NightTiming {
        quiet: Duration::from_secs(1),
        limit: Duration::from_secs(10),
    };

    /// One werewolf and two villagers, the smallest game the rules allow.
    ///
    /// With a lone wolf the pack settles the moment it selects, so the quiet
    /// period is the only clock left, which is what makes a night session
    /// readable. This game ends at parity on its first night, so a test that
    /// needs a day wants [`lone_wolf_among_five`] instead.
    fn lone_wolf() -> Assignment {
        Assignment::new([
            ("alice", Role::Villager),
            ("bob", Role::Werewolf),
            ("carol", Role::Villager),
        ])
    }

    /// One werewolf among five, so that the pack is one member and the
    /// village survives its first night.
    fn lone_wolf_among_five() -> Assignment {
        Assignment::new([
            ("alice", Role::Villager),
            ("bob", Role::Werewolf),
            ("carol", Role::Villager),
            ("dave", Role::Villager),
            ("erin", Role::Villager),
        ])
    }

    /// Two werewolves among five, so that the pack's session has two members
    /// and does not settle on one selection.
    fn pack_of_two() -> Assignment {
        Assignment::new([
            ("alice", Role::Villager),
            ("bob", Role::Werewolf),
            ("carol", Role::Villager),
            ("dave", Role::Villager),
            ("frank", Role::Werewolf),
        ])
    }

    #[test]
    fn a_night_session_closes_a_quiet_period_after_its_last_members_first_selection() {
        // A one-member session settles on its one selection, so the session's
        // deadline moves in from the hard limit to a quiet period after that
        // selection *arrived* — not after `step` ran (ADR-0018).
        //
        // Five players and one wolf, so the pack is one member and the
        // village survives the night: a three-player game would end at parity
        // that same night and there would be no day to see begin.
        let (mut moderator, _receiver) = timed(lone_wolf_among_five(), SPACED);
        let opening = started(&mut moderator);
        let (limit, _) = one_reminder(&opening);
        assert_eq!(limit, at(10_000), "the pack's hard limit, ten seconds out");

        let selected = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(2_000)),
        );
        let (quiet, payload) = one_reminder(&selected);
        assert_eq!(
            quiet,
            at(3_000),
            "the quiet period runs from the selection's arrival, not from now"
        );
        assert_eq!(
            payload,
            Message::Reminder(Look::Session {
                round: Round::FIRST,
                phase: Phase::Night,
            })
        );

        // The quiet reminder closes the night: the devour resolves, alice
        // dies, and the day begins.
        let closed = stepped(&mut moderator, reminder(payload, quiet));
        let said = actions(&closed);
        assert!(
            said.iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::Eliminated { .. })
            )),
            "the night resolved: {said:?}"
        );
        assert!(
            said.iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::PhaseBegan {
                    phase: Phase::Day,
                    ..
                })
            )),
            "and the day began: {said:?}"
        );
    }

    #[test]
    fn a_change_of_mind_restarts_the_quiet_period_and_a_repeat_does_not() {
        let (mut moderator, _receiver) = timed(lone_wolf(), SPACED);
        started(&mut moderator);

        // The first selection settles the session, so its deadline is a quiet
        // period after the selection arrived.
        let first = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(2_000)),
        );
        assert_eq!(one_reminder(&first).0, at(3_000));

        // Selecting alice again is not a change of mind, so nothing moves and
        // no new reminder is set: the moderator has already asked to be woken
        // at that instant.
        let repeat = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(2_500)),
        );
        assert_eq!(
            reminders_among(&repeat),
            [],
            "a repeat of the same target restarts nothing: {repeat:?}"
        );
        assert_eq!(moderator.game.next_deadline(), Some(at(3_000)));

        // Selecting carol is a change of mind, so the quiet period restarts
        // from *that* arrival and the moderator sets a reminder for the new
        // instant.
        let changed = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "carol", at(2_800)),
        );
        assert_eq!(one_reminder(&changed).0, at(3_800));
    }

    #[test]
    fn a_reminder_for_a_limit_that_has_since_moved_does_nothing() {
        // The stale reminder is a real one the moderator set and then had
        // reason to move away from. Nothing cancels it (ADR-0016): it
        // arrives, the game finds no session's deadline has passed, and
        // nothing comes of it.
        let (mut moderator, _receiver) = timed(lone_wolf(), SPACED);
        started(&mut moderator);
        let first = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(2_000)),
        );
        let (stale, payload) = one_reminder(&first);
        assert_eq!(stale, at(3_000));
        // A change of mind moves the quiet period out to 3.8 seconds, so the
        // reminder for 3 seconds is now for a limit the session no longer
        // has.
        stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "carol", at(2_800)),
        );

        let nothing = stepped(&mut moderator, reminder(payload, stale));
        assert_eq!(
            nothing,
            [],
            "a stale reminder closes nothing and says nothing: {nothing:?}"
        );
        assert!(moderator.game.outcome().is_none());
        assert_eq!(
            moderator.game.phase_now(),
            (Phase::Night, Round::FIRST),
            "the night is still open"
        );
    }

    #[test]
    fn a_session_whose_members_never_all_select_closes_at_its_hard_limit() {
        // Two wolves, one of which says nothing. The session never settles,
        // so its deadline never moves in from the hard limit, and the hard
        // limit is what closes it.
        let (mut moderator, _receiver) = timed(pack_of_two(), SPACED);
        let opening = started(&mut moderator);
        let (limit, payload) = one_reminder(&opening);
        assert_eq!(limit, at(10_000));

        let one = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(2_000)),
        );
        assert_eq!(
            reminders_among(&one),
            [],
            "one of two members has selected, so nothing has settled: {one:?}"
        );
        assert_eq!(
            moderator.game.next_deadline(),
            Some(limit),
            "the deadline is still the hard limit"
        );

        let closed = stepped(&mut moderator, reminder(payload, limit));
        let said = actions(&closed);
        assert!(
            said.iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::Eliminated { who, .. }) if *who == id("alice")
            )),
            "the one wolf that selected carried the night: {said:?}"
        );
    }

    #[test]
    fn a_selection_a_nanosecond_before_the_limit_counts_even_when_step_runs_after_it() {
        // The whole point of stamping by arrival. The selection arrived
        // before its session's limit; the handler was busy and only got to it
        // afterwards. A session that timed selections by when `step` ran
        // would drop it, and the game would depend on scheduling (ADR-0016).
        //
        // The moderator's own reminder for the limit is delivered *after* the
        // selection, which is the order the runtime would deliver them in
        // when the selection arrived first: perception hands the handler one
        // observation at a time, in arrival order.
        let (mut moderator, _receiver) = timed(pack_of_two(), SPACED);
        let opening = started(&mut moderator);
        let (limit, payload) = one_reminder(&opening);
        assert_eq!(limit, at(10_000));

        // A nanosecond inside the limit, which is also a nanosecond inside
        // the hard limit of the pack's session: the one clock this game has
        // open.
        let just_in_time = limit
            .checked_sub(Duration::from_nanos(1))
            .expect("the limit is an hour into the future");
        // The selection names the rest of the pack as its audience, so that
        // whether the game accepted it is visible at once: a selection the
        // game refused is relayed to nobody.
        let accepted = stepped(
            &mut moderator,
            seen_by(
                "bob",
                Round::FIRST,
                SessionKind::Devour,
                "alice",
                ["frank"],
                just_in_time,
            ),
        );
        assert!(
            actions(&accepted)
                .iter()
                .any(|(_, payload)| matches!(payload, Message::Relayed(_))),
            "a selection a nanosecond inside the limit is accepted: {accepted:?}"
        );

        // Now the limit's reminder, which closes the session and which `step`
        // runs *after* the limit has passed. The selection still counts,
        // because a session stamps a selection by its arrival (ADR-0018).
        let closed = stepped(&mut moderator, reminder(payload, limit));
        assert!(
            actions(&closed).iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::Eliminated { who, .. }) if *who == id("alice")
            )),
            "and it decided the night: {closed:?}"
        );
    }

    #[test]
    fn the_hammer_ends_the_day_even_when_a_switch_away_is_the_very_next_observation() {
        // The determinism rule, stated as a test (ADR-0016). The moderator
        // applies its rules after *each* observation and never waits for a
        // lull, so the selection that reaches a majority ends the day before
        // the switch away from it is even looked at.
        //
        // Three players, so a majority of the living is two. alice and carol
        // both nominate bob; alice then changes her mind a nanosecond later,
        // and it is too late: the day is over.
        let (mut moderator, _receiver) = timed(lone_wolf(), SPACED);
        started(&mut moderator);
        // Close the night with the pack's lone selection so the day opens.
        let selected = stepped(
            &mut moderator,
            selects("bob", Round::FIRST, SessionKind::Devour, "alice", at(1_000)),
        );
        let (quiet, payload) = one_reminder(&selected);
        stepped(&mut moderator, reminder(payload, quiet));
        // With alice devoured, two players are left and a majority is two,
        // which they cannot reach. A three-player game with a lone wolf ends
        // at parity that night, so this is the game over already.
        assert!(moderator.game.outcome().is_some());

        // So the day is played on a game the pack does not carry: five
        // players, two of them wolves, where a majority of five is three.
        let (mut moderator, _receiver) = timed(pack_of_two(), SPACED);
        let opening = started(&mut moderator);
        let (limit, payload) = one_reminder(&opening);
        // Run the night out with nobody selecting: nobody is devoured, so all
        // five live into the day and a majority is three.
        let day = stepped(&mut moderator, reminder(payload, limit));
        assert!(
            actions(&day).iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::NoDeath { .. })
            )),
            "nobody was devoured: {day:?}"
        );
        let round = Round::FIRST;
        for who in ["alice", "bob"] {
            stepped(
                &mut moderator,
                selects(who, round, SessionKind::Nominate, "dave", at(6_000)),
            );
        }
        // carol's is the third and the hammer.
        let hammer = stepped(
            &mut moderator,
            selects("carol", round, SessionKind::Nominate, "dave", at(6_001)),
        );
        let said = actions(&hammer);
        assert!(
            said.iter().any(|(_, payload)| matches!(
                payload,
                Message::Narration(Narration::Eliminated { who, cause: crate::werewolf::Cause::Lynched, .. })
                    if *who == id("dave")
            )),
            "the hammer lynched dave in the call that carried it: {said:?}"
        );
        // And the switch away, arriving one nanosecond later, is a selection
        // for a session that has closed: ignored, and nothing is relayed.
        let too_late = stepped(
            &mut moderator,
            selects(
                "alice",
                round,
                SessionKind::Nominate,
                "bob",
                at(6_001) + Duration::from_nanos(1),
            ),
        );
        assert!(
            !actions(&too_late)
                .iter()
                .any(|(_, payload)| matches!(payload, Message::Relayed(_))),
            "a switch away after the hammer is relayed to nobody: {too_late:?}"
        );
    }

    #[test]
    fn one_deadline_gets_one_reminder_however_many_selections_leave_it_where_it_is() {
        // The moderator cannot cancel a reminder, so the one thing it can do
        // about setting too many is not set the same one twice. A burst of
        // selections that leaves the deadline alone sets nothing.
        let (mut moderator, _receiver) = timed(pack_of_two(), SPACED);
        let opening = started(&mut moderator);
        assert_eq!(reminders_among(&opening).len(), 1);
        for (who, at) in [("bob", at(1_000)), ("bob", at(1_500))] {
            let produced = stepped(
                &mut moderator,
                selects(who, Round::FIRST, SessionKind::Devour, "alice", at),
            );
            assert_eq!(
                reminders_among(&produced),
                [],
                "the hard limit has not moved, so nothing new is asked for"
            );
        }
    }

    #[test]
    fn the_game_ending_pays_every_player_for_its_faction() {
        // Living or dead, +1 for the winning side and −1 for the losing
        // one, once each. There are no stalemates, so nobody is paid zero
        // and nobody goes unpaid.
        for Played {
            assignment,
            outcome,
            paid,
            ..
        } in played_games()
        {
            let expected: Vec<(ActorId, i32)> = assignment
                .players()
                .map(|(who, role)| {
                    let value = if outcome.winner == Some(role.faction()) {
                        1
                    } else {
                        -1
                    };
                    (who.clone(), value)
                })
                .collect();
            assert_eq!(paid, expected, "{outcome:?}");
            let dead: Vec<&ActorId> = paid
                .iter()
                .map(|(who, _)| who)
                .filter(|who| !outcome.living.contains(who))
                .collect();
            assert!(!dead.is_empty(), "the dead are paid too: {outcome:?}");
            assert!(
                !paid.iter().any(|(who, _)| *who == id(MODERATOR)),
                "the moderator plays no game and is paid nothing: {paid:?}"
            );
        }
    }

    #[test]
    fn the_outcome_is_narrated_then_the_rewards_then_a_reminder_then_the_stop() {
        // The shutdown's order, read off the effects as the handler thread
        // sees them: the narration is said, every reward is logged, a
        // farewell reminder is set — and the stop comes only when that
        // reminder arrives, in a call of its own, because a `Stop` preempts
        // everything and would otherwise overtake the narration (ADR-0016).
        //
        // It is the *last* stop that ends the episode. Earlier ones are dead
        // players, each stopped in the call its death was announced
        // (ADR-0012).
        for Played { effects, .. } in played_games() {
            let kinds: Vec<&str> = effects
                .iter()
                .map(|effect| match effect {
                    Effect::Act(Action::Send {
                        payload: Message::Narration(Narration::Outcome(_)),
                        ..
                    }) => "outcome",
                    Effect::Act(Action::Send { .. }) => "act",
                    Effect::Act(Action::Remind(_)) => "remind",
                    Effect::Reward { .. } => "reward",
                    Effect::Command { control, .. } => match control {
                        Control::Start => "start",
                        Control::Stop => "stop",
                    },
                })
                .collect();
            let outcome = kinds.iter().position(|kind| *kind == "outcome").unwrap();
            let stop = kinds.iter().rposition(|kind| *kind == "stop").unwrap();
            let rewards: Vec<usize> = kinds
                .iter()
                .enumerate()
                .filter(|(_, kind)| **kind == "reward")
                .map(|(at, _)| at)
                .collect();
            assert!(!rewards.is_empty());
            assert!(
                rewards.iter().all(|at| outcome < *at && *at < stop),
                "every reward falls between the outcome and the stop: {kinds:?}"
            );
            // The farewell reminder is the last thing the ending call yields,
            // and the stop is the first thing of the call that follows it.
            let farewell = kinds.iter().rposition(|kind| *kind == "remind").unwrap();
            assert!(
                rewards.iter().all(|at| *at < farewell) && farewell < stop,
                "the farewell reminder comes after the rewards and before the stop: {kinds:?}"
            );
            assert_eq!(
                stop,
                kinds.len() - 1,
                "the stop is the last word: {kinds:?}"
            );
        }
    }

    #[test]
    fn the_episode_ends_with_the_moderator_stopping_everybody_itself_included() {
        for Played {
            assignment,
            outcome,
            commanded,
            ..
        } in played_games()
        {
            let everyone: BTreeSet<ActorId> =
                assignment.players().map(|(who, _)| who.clone()).collect();
            // The players are started once, together.
            let (started, stops) = commanded.split_first().expect("a start");
            assert_eq!(*started, (everyone.clone(), Control::Start));
            for (_, control) in stops {
                assert_eq!(*control, Control::Stop);
            }
            // The last stop is the one that ends the episode, and it takes
            // every actor still running: the survivors and the moderator
            // itself (ADR-0016). The dead are not named again — each was
            // stopped where it died (ADR-0012), which is checked below.
            let mut running = outcome.living.clone();
            running.insert(id(MODERATOR));
            assert_eq!(
                stops.last().map(|(to, _)| to),
                Some(&running),
                "the episode ends when the environment stops every actor still running, itself \
                 included"
            );
            // Each dead player was stopped where it died, before that.
            let dead: BTreeSet<ActorId> = everyone.difference(&outcome.living).cloned().collect();
            let earlier: BTreeSet<ActorId> = stops[..stops.len() - 1]
                .iter()
                .flat_map(|(to, _)| to.iter().cloned())
                .collect();
            assert_eq!(earlier, dead, "a dead player is stopped at its death");
        }
    }

    #[test]
    fn the_outcome_is_narrated_to_the_living_and_is_the_last_word() {
        for Played { outcome, sent, .. } in played_games() {
            let (to, announced) = announced_outcome(&sent);
            assert_eq!(*announced, outcome);
            assert_eq!(to, outcome.living);
            assert_eq!(
                sent.last().map(|(_, payload)| payload),
                Some(&Message::Narration(Narration::Outcome(outcome.clone()))),
                "and nothing is said after it"
            );
        }
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
            let (_, announced) = announced_outcome(&sent);
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
    fn every_message_names_somebody_and_a_reminder_names_only_the_moderator() {
        // Routing is the whole hidden-information mechanism, and there is no
        // longer an exception for the outcome: everything the moderator says
        // is said to somebody in particular. The runtime would carry an
        // action addressed to nobody; this game never sends one.
        //
        // A reminder is the one message of this game that reaches its own
        // sender, which is what a reminder is (ADR-0016).
        for Played { effects, .. } in played_games() {
            for (to, payload) in actions(&effects) {
                assert!(!to.is_empty(), "{payload:?} is addressed to nobody");
                assert!(
                    !to.contains(&id(MODERATOR)),
                    "the moderator does not send itself a message: {payload:?}"
                );
            }
            for (_, payload) in reminders_among(&effects) {
                assert!(
                    matches!(payload, Message::Reminder(_)),
                    "a reminder carries the reminder payload and nothing else: {payload:?}"
                );
            }
        }
    }

    #[test]
    fn a_relayed_selection_is_the_moderators_own_message_naming_the_player() {
        // The moderator no longer impersonates anybody (ADR-0017). A
        // selection it passes on is a message of its own, carrying a
        // `Relayed` whose envelope names the player who made it and which of
        // that player's messages it was. The players never see a bare
        // `Select`, and the moderator's own name never appears in an
        // envelope.
        //
        // The pack's night session is the smallest case: bob and frank are
        // the pack, so bob's devour names frank as the one who should see it
        // and the moderator has somebody to relay it to.
        let assignment = town();
        let (mut moderator, _receiver) = moderator(assignment.clone());
        started(&mut moderator);
        let selection = Select {
            round: Round::FIRST,
            kind: SessionKind::Devour,
            target: id("alice"),
            seen_by: ids(["frank"]),
        };
        let observed = Observation {
            at: at(0),
            message: crate::Message::new("bob", [MODERATOR], 7, Message::Select(selection.clone())),
        };
        let sent = actions(&stepped(&mut moderator, observed));
        let [(to, payload)] = sent.as_slice() else {
            panic!("accepting the devour relays it and nothing else: {sent:?}");
        };
        assert_eq!(
            *payload,
            Message::Relayed(Envelope::new("bob", 7, selection)),
            "the envelope names bob and bob's message 7"
        );
        assert_eq!(
            *to,
            vec![id("frank")],
            "and it goes to exactly the audience the selection named"
        );
    }

    #[test]
    fn no_message_addresses_a_dead_player() {
        for Played { sent, .. } in played_games() {
            let mut dead = BTreeSet::new();
            for (to, payload) in &sent {
                let own_death = match payload {
                    Message::Narration(Narration::Eliminated { who, .. }) => Some(who),
                    _ => None,
                };
                for who in to {
                    assert!(
                        !dead.contains(who) || own_death == Some(who),
                        "{who} is dead but is sent {payload:?}"
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
            for (to, payload) in &sent {
                let Message::Narration(Narration::Assigned { pack, .. }) = payload else {
                    continue;
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
    fn nothing_but_the_farewell_reminder_is_acted_on_after_the_outcome() {
        // A selection that arrives once the game has ended is not folded: the
        // outcome guard stops the fold before anything about the selection is
        // looked at, so any selection will do. What is still acted on is the
        // moderator's own farewell reminder, which brings the stop that ends
        // the episode.
        let (mut moderator, _receiver) = moderator(village());
        let played = play(&mut moderator, first_other, &village());
        assert!(moderator.game.outcome().is_some());
        let late = selects("carol", Round::FIRST, SessionKind::Nominate, "bob", at(0));
        assert_eq!(stepped(&mut moderator, late), []);
        // And the stop has already been sent once, by the reminder `play`
        // delivered, so a second reminder brings nothing.
        assert!(
            controls(&played)
                .iter()
                .any(|(_, control)| *control == Control::Stop)
        );
        let again = stepped(
            &mut moderator,
            reminder(
                Message::Reminder(Look::Session {
                    round: Round::FIRST,
                    phase: Phase::Day,
                }),
                at(0),
            ),
        );
        assert_eq!(again, [], "everybody is stopped once: {again:?}");
    }

    #[test]
    fn a_selection_repeated_after_the_game_ended_says_nothing() {
        // Two moderators play the same game in lockstep. One of them is
        // handed each selection a second time, in a call of its own, and must
        // say nothing for the repeat: before the game ends the fold is the
        // game's to refuse, and after it ends the outcome guard stops the
        // fold before it starts.
        let (mut reference, _receiver) = moderator(village());
        let (mut doubled, _receiver) = moderator(village());
        let opening = started(&mut reference);
        assert_eq!(started(&mut doubled), opening);
        let mut clock = 0;
        let mut pending = respond(
            &actions(&opening),
            first_other,
            &reference.game,
            &village(),
            at(clock),
        );
        let mut due = reminders_among(&opening);
        while reference.game.outcome().is_none() {
            clock += 1;
            let mut effects = Vec::new();
            for observation in pending {
                let produced = stepped(&mut reference, observation.clone());
                assert_eq!(
                    said_and_done(&stepped(&mut doubled, observation.clone())),
                    said_and_done(&produced)
                );
                // A selection repeated while its session is still open is
                // simply the same vote again, and says nothing new; once the
                // game has ended the outcome guard stops the fold before it
                // starts. Either way the repeat is silent.
                assert_eq!(stepped(&mut doubled, observation), []);
                effects.extend(produced);
            }
            // Both moderators are handed the same reminder at the same
            // instant, so the two games stay in step.
            due.extend(reminders_among(&effects));
            due.sort_by_key(|(deadline, _)| *deadline);
            if let Some((deadline, payload)) = due.first().cloned() {
                due.remove(0);
                let woken = reminder(payload, deadline);
                let expired = stepped(&mut reference, woken.clone());
                assert_eq!(
                    said_and_done(&stepped(&mut doubled, woken)),
                    said_and_done(&expired)
                );
                effects.extend(expired);
            }
            pending = respond(
                &actions(&effects),
                first_other,
                &reference.game,
                &village(),
                at(clock),
            );
            due.extend(reminders_among(&effects));
            assert!(clock < 10_000, "a stub game should have ended by now");
        }
        assert!(reference.game.outcome().is_some() && doubled.game.outcome().is_some());
    }

    #[test]
    #[should_panic(expected = "erin sent the moderator a narration")]
    fn a_narration_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        started(&mut moderator);
        let _ = stepped(
            &mut moderator,
            from_player(
                "erin",
                at(0),
                Message::Narration(Narration::NoDeath {
                    round: Round::new(1),
                }),
            ),
        );
    }

    #[test]
    #[should_panic(expected = "erin sent the moderator a relay")]
    fn a_relay_from_a_player_panics() {
        let (mut moderator, _receiver) = moderator(village());
        started(&mut moderator);
        let _ = stepped(
            &mut moderator,
            from_player(
                "erin",
                at(0),
                Message::Relayed(Envelope::new(
                    "bob",
                    0,
                    Select {
                        round: Round::FIRST,
                        kind: SessionKind::Devour,
                        target: id("alice"),
                        seen_by: BTreeSet::new(),
                    },
                )),
            ),
        );
    }

    #[test]
    fn a_dropped_receiver_is_not_an_error() {
        let (mut moderator, receiver) = moderator(town());
        drop(receiver);
        let sent = actions(&play(&mut moderator, last_other, &town()));
        assert_eq!(moderator.game.outcome(), Some(announced_outcome(&sent).1));
    }

    #[test]
    fn a_reminder_names_the_phase_it_was_set_in() {
        // Provenance, and nothing the game reads back. Over a whole game
        // every reminder names the phase the moderator was in when it set it,
        // which is what makes one legible in a log.
        let mut phases: Vec<Phase> = Vec::new();
        for Played { effects, .. } in played_games() {
            for (_, payload) in reminders_among(&effects) {
                match payload {
                    Message::Reminder(Look::Session { phase, .. }) => phases.push(phase),
                    // The farewell names no phase: it is the wait after the
                    // game, not a clock inside one.
                    Message::Reminder(Look::Farewell) => {}
                    other => panic!("a reminder carries a `Look`: {other:?}"),
                }
            }
        }
        assert!(phases.contains(&Phase::Night), "{phases:?}");
        assert!(phases.contains(&Phase::Day), "{phases:?}");
    }
}
