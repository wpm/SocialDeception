//! Assembling an episode of Werewolf from a configuration, and running one
//! to its outcome.
//!
//! This is the whole of the seam between Werewolf and the runtime. Nothing
//! in the runtime knows about Werewolf: [`episode`] *constructs* an
//! [`Episode`], dealing the roles and adding a [`Player`] for everyone and
//! the [`Moderator`] that runs their game, and [`run`] runs one to
//! completion and hands back how it ended. If a change here ever wants to
//! modify `Episode`, `Agent` or the router, something upstream was designed
//! wrong (ADR-0004).
//!
//! # One deal
//!
//! The moderator and the players are built from the same [`Assignment`]
//! value. A second call to [`Assignment::deal`] would give the same answer,
//! but passing one value around makes it impossible for that to stop being
//! true.
//!
//! # A missing outcome is the runtime's error, not this module's
//!
//! The moderator is the episode's environment, so an episode ends when the
//! moderator says it does: it stops every actor, itself included, once it has
//! announced the outcome and its players have heard it (ADR-0016). A
//! moderator that never gets there leaves the episode running until its
//! **hard time limit**, which it reports as
//! [`EpisodeError::Timeout`] naming the
//! actors still going. So there is nothing for this module to detect: [`run`]
//! takes the outcome from the moderator's channel and a clean run always has
//! one.
//!
//! # How the time limit is derived
//!
//! The limit is a **backstop against a runtime or moderator bug**, not a
//! schedule anything is meant to meet, so it is derived generously from the
//! clocks the configuration sets and then multiplied.
//!
//! A game ends within its `day_cap` rounds however its players act
//! (ADR-0011), and a round costs at most one night and one day. A night's
//! three sessions run **at once**, so a night costs the longest of the three
//! hard limits rather than their sum; a day costs its own limit. So the rules
//! bound a game at
//!
//! ```text
//! day_cap × (max(pack.limit, seer.limit, doctor.limit) + day.limit)
//! ```
//!
//! plus the moderator's farewell interval at the end. [`limit`] is
//! [`limit`] is a slack multiple of that, floored, and both of those are for
//! the same
//! thing: nothing above accounts for thread scheduling, and a game whose
//! sessions are measured in tens of milliseconds — which is how this
//! repository's tests play them — can spend a comparable share of its time
//! simply waiting to be scheduled on a loaded machine. A game played on the
//! example's clocks, which are seconds, is bounded by a number so much larger
//! than it needs that the slack costs nothing.
//!
//! What the limit must never be is *tight*. A limit that fired on a slow but
//! correct game would turn a scheduling delay into a failed run, which is
//! worse than the hang it is there to prevent.
//!
//! # The seed never enters the game
//!
//! The master seed is the setup's and the moderator's. With it, the roster
//! and the public algorithm, anyone could recompute the deal and every
//! agent's random stream, which is to say every piece of hidden information
//! in the game. So it lies outside every player's observation space: no
//! [`Message`] has a field that could carry it, and this
//! module never puts it in one. It is recorded beside the log, in the
//! effective configuration the `werewolf` binary writes, never in it.

use std::error;
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::path::PathBuf;
use std::time::Duration;

use crossbeam_channel::{Receiver, unbounded};

use super::assignment::Assignment;
use super::config::Config;
use super::config::Timing;
use super::game::Game;
use super::live::Text;
use super::message::{Message, Outcome};
use super::moderator::Moderator;
use super::player::Player;
use super::role::Role;
use super::strategy::RandomStrategy;
use crate::actor::{Episode, EpisodeError, Policy as Policies};
use crate::log::{JsonLines, Policy, Sinks};
use crate::message::ActorId;

/// Why a run did not end with an outcome.
#[derive(Debug)]
pub enum RunError {
    /// The log could not be created or written.
    Io {
        /// Where it was being written, or `None` if it was going nowhere.
        log: Option<PathBuf>,
        /// What went wrong.
        source: io::Error,
    },
    /// The episode did not run cleanly. A game the moderator never ended
    /// arrives here as
    /// [`EpisodeError::Timeout`].
    Episode(EpisodeError),
    /// The episode ran cleanly and the moderator announced no outcome.
    ///
    /// Nothing known produces this: an episode the moderator did not end is
    /// a timeout, and one it did end it ended by announcing the outcome. It
    /// is here because `run` cannot prove that from the types, and a silent
    /// `unwrap` would be a worse answer than a named error.
    NoOutcome,
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                log: Some(path),
                source,
            } => write!(f, "cannot write {}: {source}", path.display()),
            Self::Io { log: None, source } => write!(f, "cannot write the log: {source}"),
            Self::Episode(error) => error.fmt(f),
            Self::NoOutcome => {
                f.write_str("the episode ended cleanly but the moderator announced no outcome")
            }
        }
    }
}

impl error::Error for RunError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Episode(error) => Some(error),
            Self::NoOutcome => None,
        }
    }
}

impl From<EpisodeError> for RunError {
    fn from(error: EpisodeError) -> Self {
        Self::Episode(error)
    }
}

/// How much of the slack the derived time limit is: the rules' bound on a
/// game, multiplied by this.
///
/// It is there for thread scheduling, which nothing in the rules accounts
/// for; see the [module documentation](self) for why it is generous rather
/// than tight.
const SLACK: u32 = 8;

/// The least a derived time limit ever is, whatever the configuration's
/// clocks.
///
/// A configuration may set clocks in single milliseconds, and a bound of a
/// few hundred milliseconds would be a limit a correct game could lose a race
/// with on a loaded machine. This is the floor beneath which the multiplier
/// stops being the thing that matters.
const FLOOR: Duration = Duration::from_secs(30);

/// The hard time limit an episode played under `timing` with `players`
/// players is given, derived as the [module documentation](self) sets out.
#[must_use]
pub fn limit(timing: &Timing, players: usize) -> Duration {
    // A night's three sessions run at once, so a night costs the longest of
    // their hard limits and not their sum.
    let night = timing
        .pack
        .limit
        .max(timing.seer.limit)
        .max(timing.doctor.limit);
    let round = night + timing.day.limit;
    let rules = round.saturating_mul(timing.day_cap(players));
    // The farewell interval the moderator waits before stopping everybody is
    // inside the slack, which is orders of magnitude larger than it.
    rules.saturating_mul(SLACK).max(FLOOR)
}

/// Builds an episode from a configuration, whose log goes to `sinks`. The
/// receiver yields the outcome once the moderator has announced it.
///
/// The roles are dealt once, from the configuration's seed; every player is
/// seated with the role it was dealt, deciding with a [`RandomStrategy`]
/// seeded for it alone; and the moderator runs a [`Game`] over that same
/// deal.
///
/// **There is no clock argument.** The episode starts the one clock itself,
/// before the writer and before any actor, and hands a copy to every actor's
/// `start` hook, which is what makes the shared origin structural rather than
/// something a caller can get wrong (ADR-0017).
///
/// The episode is given the hard time limit [`limit`] derives from the
/// configuration's clocks.
///
/// # Panics
///
/// If the configuration would not pass [`Config::validate`], which rules
/// out every way the roster could fail to assemble.
#[must_use]
pub fn episode(
    config: &Config,
    sinks: Sinks<Message>,
) -> (Episode<i32, Message>, Receiver<Outcome>) {
    let assignment = Assignment::deal(config);
    let (mut episode, outcomes) = moderate(config, assignment.clone(), sinks);
    for (who, role) in assignment.players() {
        seat(&mut episode, config, who, role);
    }
    (episode, outcomes)
}

/// An episode whose environment is a moderator running a game over
/// `assignment`, and no players yet. The receiver yields the outcome once it
/// has been announced.
fn moderate(
    config: &Config,
    assignment: Assignment,
    sinks: Sinks<Message>,
) -> (Episode<i32, Message>, Receiver<Outcome>) {
    let (outcome, outcomes) = unbounded();
    let players = assignment.players().count();
    let game = Game::new(assignment, config.seed, config.timing);
    let episode = Episode::new(
        sinks,
        config.moderator.clone(),
        Moderator::new(config.moderator.clone(), game, outcome),
    )
    .within(limit(&config.timing, players));
    (episode, outcomes)
}

/// Seats `who` in the roster as its role, with a strategy seeded for it.
fn seat(episode: &mut Episode<i32, Message>, config: &Config, who: &ActorId, role: Role) {
    let strategy = RandomStrategy::for_agent(config.seed, who);
    let moderator = config.moderator.clone();
    add(
        episode,
        who,
        Player::new(who.clone(), role, strategy, moderator),
    );
}

/// Adds an agent to the roster.
///
/// Adding fails only on a duplicate id, and a validated configuration has
/// none: its players are distinct and none of them is the moderator. That
/// is the configuration's check to make, so a failure here is a panic and
/// not an error of its own.
fn add(
    episode: &mut Episode<i32, Message>,
    who: &ActorId,
    handler: impl Policies<Message> + Send + 'static,
) {
    episode
        .add(who.clone(), handler)
        .expect("a validated configuration has no two agents with the same id");
}

/// Runs one episode to completion and returns how it ended.
///
/// Two sinks, either of which may be absent:
///
/// - the log, a **required** [`JsonLines`] over `config.trajectory`
///   when it is set. Required because a run whose record of itself is
///   incomplete is a run that did not happen;
/// - the live text, an **optional** [`Text`] over `live` when one is given.
///   Optional because a watcher who closes the pipe has seen all they
///   wanted, and the game is no less played for it.
///
/// With neither, the writer has no sinks at all and discards what it
/// receives: an episode with no log and nobody watching exercises
/// everything a fully observed one does.
///
/// The summary a caller prints afterwards is not a sink. It is the
/// [`Outcome`] returned here, which is the game's end rather than one more
/// thing that happened in it.
///
/// The configuration is taken as given. Any overrides the command line
/// applies are applied to it before it gets here.
///
/// # Errors
///
/// [`RunError::Io`] if the log cannot be created or written,
/// [`RunError::Episode`] if the episode did not run cleanly — a game the
/// moderator never ended arrives as
/// [`EpisodeError::Timeout`] — and
/// [`RunError::NoOutcome`] if a clean run left no outcome on the channel.
///
/// # Panics
///
/// If the configuration would not pass [`Config::validate`]; see
/// [`episode`].
pub fn run(config: &Config, live: Option<Box<dyn Write + Send>>) -> Result<Outcome, RunError> {
    let log = config.trajectory.as_deref();
    let mut sinks: Sinks<Message> = Vec::new();
    if let Some(path) = log {
        let file = File::create(path).map_err(|source| RunError::Io {
            log: Some(path.to_path_buf()),
            source,
        })?;
        sinks.push((Box::new(JsonLines::new(file)), Policy::Required));
    }
    if let Some(live) = live {
        sinks.push((Box::new(Text::new(live)), Policy::Optional));
    }
    let (episode, outcomes) = episode(config, sinks);
    play(episode, &outcomes)
}

/// Runs an assembled episode and takes the outcome off the moderator's
/// channel.
///
/// **The episode joins its own writer** (ADR-0017), because it is the episode
/// that started it, so a required sink that fails mid-episode surfaces as the
/// actors' `WriterClosed` inside
/// [`EpisodeError::Agents`] rather than as
/// a [`RunError::Io`] naming the path. The path is still named for the
/// failure this function can see first: a log that cannot be created at all.
fn play(episode: Episode<i32, Message>, outcomes: &Receiver<Outcome>) -> Result<Outcome, RunError> {
    episode.run()?;
    // The moderator's sender went with its handler when the episode joined
    // it, so the receiver holds the outcome now or never will.
    outcomes.try_recv().map_err(|_| RunError::NoOutcome)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::actor::{Action, Observation};
    use crate::testing::{TempDir, fast, id, ids, parse_lines};
    use crate::werewolf::config::{DEFAULT_MODERATOR, RoleCounts};
    use crate::werewolf::transcript::{self, Transcript};

    const SEED: u64 = 20_260_918;

    /// A validated configuration for `players`, with the given special
    /// roles and no log.
    fn config<const N: usize>(
        players: [&str; N],
        werewolves: usize,
        seers: usize,
        doctors: usize,
    ) -> Config {
        let config = Config {
            seed: SEED,
            players: players.map(ActorId::new).into(),
            roles: RoleCounts {
                werewolves,
                seers,
                doctors,
            },
            trajectory: None,
            moderator: id(DEFAULT_MODERATOR),
            timing: fast(),
        };
        config.validate().unwrap();
        config
    }

    /// Seven players, two werewolves, a seer and a doctor.
    fn town() -> Config {
        config(
            ["alice", "bob", "carol", "dave", "erin", "frank", "grace"],
            2,
            1,
            1,
        )
    }

    /// Reads the log at `path` back as the game it records.
    fn transcript(path: &Path, config: &Config) -> Transcript {
        let text = fs::read_to_string(path).unwrap();
        Transcript::read(&transcript::lines(&text).unwrap(), &config.moderator).unwrap()
    }

    /// Asserts that `outcome` is a finished game among `config`'s players,
    /// decided within as many rounds as there are players.
    fn check(config: &Config, outcome: &Outcome) {
        let players: BTreeSet<&ActorId> = config.players.iter().collect();
        // No lower bound to check: a `Round` cannot be zero, so that a
        // finished game lasted at least one round is the type's guarantee.
        assert!(
            outcome.rounds.number() as usize <= config.players.len(),
            "{outcome:?}"
        );
        assert!(!outcome.living.is_empty(), "{outcome:?}");
        assert!(
            outcome.living.iter().all(|who| players.contains(who)),
            "{outcome:?}"
        );
    }

    #[test]
    fn the_roster_is_every_player_and_the_moderator() {
        let (episode, _outcomes) = episode(&town(), Vec::new());
        let roster: BTreeSet<ActorId> = episode.ids().cloned().collect();
        assert_eq!(
            roster,
            ids([
                "alice",
                "bob",
                "carol",
                "dave",
                "erin",
                "frank",
                "grace",
                "moderator"
            ])
        );
    }

    #[test]
    fn a_seven_player_game_runs_to_an_outcome_with_a_winner() {
        let config = town();
        let outcome = run(&config, None).unwrap();
        check(&config, &outcome);
    }

    #[test]
    fn the_same_config_gives_the_same_outcome() {
        let config = town();
        assert_eq!(run(&config, None).unwrap(), run(&config, None).unwrap());
    }

    #[test]
    fn a_game_with_no_seer_and_no_doctor_runs_to_an_outcome() {
        // Three players and one werewolf is the smallest game the
        // configuration allows.
        for config in [
            config(["alice", "bob", "carol"], 1, 0, 0),
            config(["alice", "bob", "carol", "dave", "erin"], 1, 0, 0),
        ] {
            let outcome = run(&config, None).unwrap();
            check(&config, &outcome);
        }
    }

    #[test]
    fn the_log_is_written_and_agrees_with_the_outcome() {
        let dir = TempDir::new();
        let log = dir.join("werewolf.jsonl");
        let mut config = town();
        config.trajectory = Some(log.clone());
        let outcome = run(&config, None).unwrap();

        let lines = parse_lines(&fs::read(&log).unwrap());
        assert!(!lines.is_empty());

        // The outcome on the channel and the one the moderator announced in
        // world are the same game's; if they ever diverge, the side channel
        // and the record of truth have parted company.
        assert_eq!(transcript(&log, &config).outcome, outcome);
    }

    #[test]
    fn the_log_is_the_same_game_across_runs() {
        let dir = TempDir::new();
        let first = dir.join("first.jsonl");
        let second = dir.join("second.jsonl");
        let mut config = town();
        config.trajectory = Some(first.clone());
        run(&config, None).unwrap();
        config.trajectory = Some(second.clone());
        run(&config, None).unwrap();
        assert_eq!(transcript(&first, &config), transcript(&second, &config));
    }

    #[test]
    fn a_log_that_cannot_be_created_is_an_io_error_naming_it() {
        let mut config = town();
        let path = Path::new("/no-such-directory/werewolf.jsonl");
        config.trajectory = Some(path.to_path_buf());
        let error = run(&config, None).unwrap_err();
        assert!(
            matches!(&error, RunError::Io { log: Some(t), .. } if t == path),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .starts_with("cannot write /no-such-directory/werewolf.jsonl: ")
        );
        assert!(error::Error::source(&error).is_some());
    }

    /// A player that never selects.
    struct Silent;

    impl Policies<Message> for Silent {
        fn policy(&mut self, _: Observation<Message>) -> impl IntoIterator<Item = Action<Message>> {
            []
        }
    }

    #[test]
    fn a_silent_player_no_longer_stalls_the_run() {
        // The same roster `episode` would build, except that one werewolf
        // never selects. Under ADR-0004 that was a stall: the phase
        // resolved on its last answer and one that never came stopped the
        // game. Under ADR-0011 a session closes on its clock, so the
        // silent player is simply a member that never selected, and the
        // game finishes without it.
        let config = config(["alice", "bob", "carol"], 1, 0, 0);
        let assignment = Assignment::deal(&config);
        let silent = assignment.pack().iter().next().unwrap().clone();
        let (mut episode, outcomes) = moderate(&config, assignment.clone(), Vec::new());
        for (who, role) in assignment.players() {
            if *who == silent {
                add(&mut episode, who, Silent);
            } else {
                seat(&mut episode, &config, who, role);
            }
        }

        // A pack of one that never selects devours nobody, and three
        // random players never put two on one target either, so nobody
        // dies at all and the game runs to its day cap: a stalemate,
        // which pays -1 to everyone (ADR-0011).
        let outcome = play(episode, &outcomes).unwrap();
        assert_eq!(outcome.winner, None);
        assert_eq!(outcome.living.len(), 3, "nobody died");
    }
}
