//! Assembling an episode of Werewolf from a configuration, and running one
//! to its outcome.
//!
//! This is the whole of the seam between Werewolf and the runtime. Nothing
//! in the runtime knows about Werewolf: [`episode`] *constructs* an
//! [`Episode`], dealing the roles and adding a [`Seat`] for every player and
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
//! # A missing outcome is an error
//!
//! [`Episode::run`] returning `Ok(())` means the episode went quiescent,
//! which normally means the moderator announced the outcome and fell silent.
//! It can also mean a player failed to answer a request, in which case the
//! game stopped mid-way and the trajectory is simply short. There is no hang
//! to notice, so this is the only place it can be caught: [`run`] takes the
//! outcome from the moderator's channel, and finding none there is
//! [`RunError::Truncated`].
//!
//! # The seed never enters the game
//!
//! The master seed is the setup's and the moderator's. With it, the roster
//! and the public algorithm, anyone could recompute the deal and every
//! agent's random stream, which is to say every piece of hidden information
//! in the game. So it lies outside every player's observation space: no
//! [`Message`] has a field that could carry it, and this module never puts
//! it in one. It is recorded beside the trajectory, in the effective
//! configuration the `werewolf` binary writes, never in it.

use std::error;
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crossbeam_channel::{Receiver, Sender, unbounded};

use super::assignment::Assignment;
use super::config::Config;
use super::game::Game;
use super::message::{Message, Outcome};
use super::moderator::Moderator;
use super::player::Seat;
use super::policy::RandomPolicy;
use super::role::Role;
use super::roles::{Doctor, Seer, Villager, Werewolf};
use crate::agent::Handler;
use crate::episode::{Episode, EpisodeError};
use crate::event::AgentId;
use crate::trajectory::{LogRecord, Writer};

/// Why a run did not end with an outcome.
#[derive(Debug)]
pub enum RunError {
    /// The trajectory could not be created or written.
    Io {
        /// Where it was being written, or `None` if it was going nowhere.
        trajectory: Option<PathBuf>,
        /// What went wrong.
        source: io::Error,
    },
    /// The episode did not run cleanly.
    Episode(EpisodeError),
    /// The episode ended without the moderator announcing an outcome: some
    /// player failed to answer a request, and the game stopped there.
    Truncated,
}

impl fmt::Display for RunError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io {
                trajectory: Some(path),
                source,
            } => write!(f, "cannot write {}: {source}", path.display()),
            Self::Io {
                trajectory: None,
                source,
            } => write!(f, "cannot write the trajectory: {source}"),
            Self::Episode(error) => error.fmt(f),
            Self::Truncated => f.write_str(
                "the episode ended without an outcome: a player did not answer a request",
            ),
        }
    }
}

impl error::Error for RunError {
    fn source(&self) -> Option<&(dyn error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Episode(error) => Some(error),
            Self::Truncated => None,
        }
    }
}

impl From<EpisodeError> for RunError {
    fn from(error: EpisodeError) -> Self {
        Self::Episode(error)
    }
}

/// Builds an episode from a configuration. The receiver yields the outcome
/// once the moderator has announced it.
///
/// The roles are dealt once, from the configuration's seed; every player is
/// seated as the type its role calls for, deciding with a [`RandomPolicy`]
/// seeded for it alone; and the moderator runs a [`Game`] over that same
/// deal. The trajectory goes to `records`.
///
/// # Panics
///
/// If the configuration would not pass [`Config::validate`], which rules
/// out every way the roster could fail to assemble.
#[must_use]
pub fn episode(
    config: &Config,
    records: Sender<LogRecord<Message>>,
) -> (Episode<Message>, Receiver<Outcome>) {
    let assignment = Assignment::deal(config);
    let mut episode = Episode::new(records);
    for (who, role) in assignment.players() {
        seat(&mut episode, config, who, role);
    }
    let outcomes = moderate(&mut episode, config, assignment);
    (episode, outcomes)
}

/// Seats the moderator in the roster, running a game over `assignment`.
/// The receiver yields the outcome once it has been announced.
fn moderate(
    episode: &mut Episode<Message>,
    config: &Config,
    assignment: Assignment,
) -> Receiver<Outcome> {
    let (outcome, outcomes) = unbounded();
    let game = Game::new(assignment, config.max_rounds, config.seed);
    add(episode, &config.moderator, Moderator::new(game, outcome));
    outcomes
}

/// Seats `who` in the roster as its role, with a policy seeded for it.
fn seat(episode: &mut Episode<Message>, config: &Config, who: &AgentId, role: Role) {
    let policy = RandomPolicy::for_agent(config.seed, who);
    let moderator = config.moderator.clone();
    let me = who.clone();
    match role {
        Role::Villager => add(
            episode,
            who,
            Seat::new(Villager::new(me), policy, moderator),
        ),
        Role::Werewolf => add(
            episode,
            who,
            Seat::new(Werewolf::new(me), policy, moderator),
        ),
        Role::Seer => add(episode, who, Seat::new(Seer::new(me), policy, moderator)),
        Role::Doctor => add(episode, who, Seat::new(Doctor::new(me), policy, moderator)),
    }
}

/// Adds an agent to the roster.
///
/// Adding fails only on a duplicate id, and a validated configuration has
/// none: its players are distinct and none of them is the moderator. That
/// is the configuration's check to make, so a failure here is a panic and
/// not an error of its own.
fn add(
    episode: &mut Episode<Message>,
    who: &AgentId,
    handler: impl Handler<Message> + Send + 'static,
) {
    episode
        .add(who.clone(), handler)
        .expect("a validated configuration has no two agents with the same id");
}

/// Runs one episode to completion and returns how it ended.
///
/// The trajectory is written to `config.trajectory` if it is set. Otherwise
/// it goes nowhere, by the same path: an episode with no trajectory
/// exercises everything one with a trajectory does.
///
/// The configuration is taken as given. Any overrides the command line
/// applies are applied to it before it gets here.
///
/// # Errors
///
/// [`RunError::Io`] if the trajectory cannot be created or written,
/// [`RunError::Episode`] if the episode did not run cleanly, and
/// [`RunError::Truncated`] if it ran to quiescence without the moderator
/// announcing an outcome.
///
/// # Panics
///
/// If the configuration would not pass [`Config::validate`]; see
/// [`episode`].
pub fn run(config: &Config) -> Result<Outcome, RunError> {
    let trajectory = config.trajectory.as_deref();
    let sink: Box<dyn Write + Send> = match trajectory {
        Some(path) => Box::new(File::create(path).map_err(|source| RunError::Io {
            trajectory: Some(path.to_path_buf()),
            source,
        })?),
        None => Box::new(io::sink()),
    };
    let (records, writer) = Writer::spawn(sink);
    let (episode, outcomes) = episode(config, records);
    play(episode, &outcomes, writer, trajectory)
}

/// Runs an assembled episode, joins its writer, which is writing the
/// trajectory to `trajectory` if anywhere, and takes the outcome off the
/// moderator's channel.
fn play<W: Write + Send + 'static>(
    episode: Episode<Message>,
    outcomes: &Receiver<Outcome>,
    writer: Writer<W>,
    trajectory: Option<&Path>,
) -> Result<Outcome, RunError> {
    let ran = episode.run();
    // The episode drops every sender to the writer on its way out, whether
    // or not it ran cleanly, so the writer can be joined now for the whole
    // trajectory. A failed run is the more informative error of the two.
    let written = writer.join();
    ran?;
    written.map_err(|source| RunError::Io {
        trajectory: trajectory.map(Path::to_path_buf),
        source,
    })?;
    // The moderator's sender went with its handler when the episode joined
    // it, so the receiver holds the outcome now or never will.
    outcomes.try_recv().map_err(|_| RunError::Truncated)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::agent::Outgoing;
    use crate::event::Event;
    use crate::testing::{TempPath, id, ids, parse_lines};
    use crate::werewolf::config::{DEFAULT_MAX_ROUNDS, DEFAULT_MODERATOR, RoleCounts};
    use crate::werewolf::message::Round;
    use crate::werewolf::transcript::{self, Transcript};

    const SEED: u64 = 20_260_918;

    /// A validated configuration for `players`, with the given special
    /// roles and no trajectory.
    fn config<const N: usize>(
        players: [&str; N],
        werewolves: usize,
        seers: usize,
        doctors: usize,
    ) -> Config {
        let config = Config {
            seed: SEED,
            players: players.map(AgentId::new).into(),
            roles: RoleCounts {
                werewolves,
                seers,
                doctors,
            },
            trajectory: None,
            max_rounds: DEFAULT_MAX_ROUNDS,
            moderator: id(DEFAULT_MODERATOR),
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

    /// Reads the trajectory at `path` back as the game it records.
    fn transcript(path: &Path, config: &Config) -> Transcript {
        let text = fs::read_to_string(path).unwrap();
        Transcript::read(&transcript::lines(&text).unwrap(), &config.moderator).unwrap()
    }

    /// Asserts that `outcome` is a finished game among `config`'s players.
    fn check(config: &Config, outcome: &Outcome) {
        let players: BTreeSet<&AgentId> = config.players.iter().collect();
        assert!(outcome.rounds >= Round(1), "{outcome:?}");
        assert!(outcome.rounds.0 <= config.max_rounds, "{outcome:?}");
        assert!(!outcome.living.is_empty(), "{outcome:?}");
        assert!(
            outcome.living.iter().all(|who| players.contains(who)),
            "{outcome:?}"
        );
    }

    #[test]
    fn the_roster_is_every_player_and_the_moderator() {
        let (records, _writer) = Writer::spawn(io::sink());
        let (episode, _outcomes) = episode(&town(), records);
        let roster: BTreeSet<AgentId> = episode.ids().cloned().collect();
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
        let outcome = run(&config).unwrap();
        check(&config, &outcome);
        assert!(outcome.winner.is_some(), "{outcome:?}");
    }

    #[test]
    fn the_same_config_gives_the_same_outcome() {
        let config = town();
        assert_eq!(run(&config).unwrap(), run(&config).unwrap());
    }

    #[test]
    fn a_game_with_no_seer_and_no_doctor_runs_to_an_outcome() {
        // Three players and one werewolf is the smallest game the
        // configuration allows.
        for config in [
            config(["alice", "bob", "carol"], 1, 0, 0),
            config(["alice", "bob", "carol", "dave", "erin"], 1, 0, 0),
        ] {
            let outcome = run(&config).unwrap();
            check(&config, &outcome);
            assert!(outcome.winner.is_some(), "{outcome:?}");
        }
    }

    #[test]
    fn the_round_cap_is_a_stalemate() {
        // Seven players and two werewolves cannot finish in one round: the
        // night takes at most one player and the day exactly one, which
        // leaves the pack neither dead nor at parity.
        let mut config = town();
        config.max_rounds = 1;
        let outcome = run(&config).unwrap();
        check(&config, &outcome);
        assert_eq!(outcome.winner, None, "{outcome:?}");
        assert_eq!(outcome.rounds, Round(1));
    }

    #[test]
    fn the_trajectory_is_written_and_agrees_with_the_outcome() {
        let trajectory = TempPath::new("jsonl");
        let mut config = town();
        config.trajectory = Some(trajectory.to_path_buf());
        let outcome = run(&config).unwrap();

        let lines = parse_lines(&fs::read(&*trajectory).unwrap());
        assert!(!lines.is_empty());

        // The outcome on the channel and the one the moderator announced in
        // world are the same game's; if they ever diverge, the side channel
        // and the record of truth have parted company.
        assert_eq!(transcript(&trajectory, &config).outcome, outcome);
    }

    #[test]
    fn the_trajectory_is_the_same_game_across_runs() {
        let first = TempPath::new("jsonl");
        let second = TempPath::new("jsonl");
        let mut config = town();
        config.trajectory = Some(first.to_path_buf());
        run(&config).unwrap();
        config.trajectory = Some(second.to_path_buf());
        run(&config).unwrap();
        assert_eq!(transcript(&first, &config), transcript(&second, &config));
    }

    #[test]
    fn a_trajectory_that_cannot_be_created_is_an_io_error_naming_it() {
        let mut config = town();
        let path = Path::new("/no-such-directory/werewolf.jsonl");
        config.trajectory = Some(path.to_path_buf());
        let error = run(&config).unwrap_err();
        assert!(
            matches!(&error, RunError::Io { trajectory: Some(t), .. } if t == path),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .starts_with("cannot write /no-such-directory/werewolf.jsonl: ")
        );
        assert!(error::Error::source(&error).is_some());
    }

    /// A player that never answers.
    struct Silent;

    impl Handler<Message> for Silent {
        fn handle(&mut self, _: &[Event<Message>]) -> Vec<Outgoing<Message>> {
            Vec::new()
        }
    }

    #[test]
    fn a_silent_player_truncates_the_run() {
        // The same roster `episode` would build, except that one werewolf
        // never answers. The first night's request to it goes unanswered,
        // the episode goes quiescent, and there is no outcome to take.
        let config = config(["alice", "bob", "carol"], 1, 0, 0);
        let assignment = Assignment::deal(&config);
        let silent = assignment.pack().iter().next().unwrap().clone();
        let (records, writer) = Writer::spawn(io::sink());
        let mut episode = Episode::new(records);
        for (who, role) in assignment.players() {
            if *who == silent {
                add(&mut episode, who, Silent);
            } else {
                seat(&mut episode, &config, who, role);
            }
        }
        let outcomes = moderate(&mut episode, &config, assignment);

        let error = play(episode, &outcomes, writer, None).unwrap_err();
        assert!(matches!(error, RunError::Truncated), "{error:?}");
        assert!(error.to_string().contains("without an outcome"));
    }
}
