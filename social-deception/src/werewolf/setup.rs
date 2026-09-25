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
//! # A missing outcome is the runtime's error, not this module's
//!
//! The moderator is the episode's [`Environment`](crate::Environment), so
//! an episode ends when the moderator says it does, which is when it has
//! announced the outcome. A player that fails to answer a request leaves
//! nothing in flight and nobody stopped, which the episode reports as
//! [`EpisodeError::Stalled`] naming the players still running. So there is
//! nothing for this module to detect: [`run`] takes the outcome from the
//! moderator's channel and a clean run always has one.
//!
//! # The seed never enters the game
//!
//! The master seed is the setup's and the moderator's. With it, the roster
//! and the public algorithm, anyone could recompute the deal and every
//! agent's random stream, which is to say every piece of hidden information
//! in the game. So it lies outside every player's observation space: no
//! [`Message`](super::Message) has a field that could carry it, and this
//! module never puts it in one. It is recorded beside the trajectory, in the
//! effective configuration the `werewolf` binary writes, never in it.

use std::error;
use std::fmt;
use std::fs::File;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use crossbeam_channel::{Receiver, Sender, unbounded};

use super::WerewolfDomain;
use super::assignment::Assignment;
use super::config::Config;
use super::game::Game;
use super::message::Outcome;
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
    /// The episode did not run cleanly. A game in which some player did not
    /// answer a request arrives here as
    /// [`EpisodeError::Stalled`].
    Episode(EpisodeError),
    /// The episode ran cleanly and the moderator announced no outcome.
    ///
    /// Nothing known produces this: an episode the moderator did not end is
    /// a stall, and one it did end it ended by announcing the outcome. It is
    /// here because `run` cannot prove that from the types, and a silent
    /// `unwrap` would be a worse answer than a named error.
    NoOutcome,
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
    records: Sender<LogRecord<WerewolfDomain>>,
) -> (Episode<WerewolfDomain>, Receiver<Outcome>) {
    let assignment = Assignment::deal(config);
    let (mut episode, outcomes) = moderate(config, assignment.clone(), records);
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
    records: Sender<LogRecord<WerewolfDomain>>,
) -> (Episode<WerewolfDomain>, Receiver<Outcome>) {
    let (outcome, outcomes) = unbounded();
    let game = Game::new(assignment, config.seed);
    let episode = Episode::new(
        records,
        config.moderator.clone(),
        Moderator::new(game, outcome),
    );
    (episode, outcomes)
}

/// Seats `who` in the roster as its role, with a policy seeded for it.
fn seat(episode: &mut Episode<WerewolfDomain>, config: &Config, who: &AgentId, role: Role) {
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
    episode: &mut Episode<WerewolfDomain>,
    who: &AgentId,
    handler: impl Handler<WerewolfDomain> + Send + 'static,
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
/// [`RunError::Episode`] if the episode did not run cleanly — a player that
/// did not answer a request arrives as
/// [`EpisodeError::Stalled`] — and
/// [`RunError::NoOutcome`] if a clean run left no outcome on the channel.
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
    episode: Episode<WerewolfDomain>,
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
    outcomes.try_recv().map_err(|_| RunError::NoOutcome)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;

    use super::*;
    use crate::agent::{Action, Observation};
    use crate::cancel::Cancel;
    use crate::testing::{TempDir, id, ids, parse_lines};
    use crate::werewolf::config::{DEFAULT_MODERATOR, RoleCounts};
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

    /// Asserts that `outcome` is a finished game among `config`'s players,
    /// decided within as many rounds as there are players.
    fn check(config: &Config, outcome: &Outcome) {
        let players: BTreeSet<&AgentId> = config.players.iter().collect();
        assert!(outcome.rounds >= Round(1), "{outcome:?}");
        assert!(
            outcome.rounds.0 as usize <= config.players.len(),
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
        }
    }

    #[test]
    fn the_trajectory_is_written_and_agrees_with_the_outcome() {
        let dir = TempDir::new();
        let trajectory = dir.join("werewolf.jsonl");
        let mut config = town();
        config.trajectory = Some(trajectory.clone());
        let outcome = run(&config).unwrap();

        let lines = parse_lines(&fs::read(&trajectory).unwrap());
        assert!(!lines.is_empty());

        // The outcome on the channel and the one the moderator announced in
        // world are the same game's; if they ever diverge, the side channel
        // and the record of truth have parted company.
        assert_eq!(transcript(&trajectory, &config).outcome, outcome);
    }

    #[test]
    fn the_trajectory_is_the_same_game_across_runs() {
        let dir = TempDir::new();
        let first = dir.join("first.jsonl");
        let second = dir.join("second.jsonl");
        let mut config = town();
        config.trajectory = Some(first.clone());
        run(&config).unwrap();
        config.trajectory = Some(second.clone());
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

    impl Handler<WerewolfDomain> for Silent {
        fn handle(
            &mut self,
            _: &Observation<WerewolfDomain>,
            _: &Cancel,
        ) -> Vec<Action<WerewolfDomain>> {
            Vec::new()
        }
    }

    #[test]
    fn a_silent_player_stalls_the_run() {
        // The same roster `episode` would build, except that one werewolf
        // never answers. The first night's request to it goes unanswered,
        // nothing is left in flight, and the moderator has stopped nobody,
        // which is a stall naming every player.
        let config = config(["alice", "bob", "carol"], 1, 0, 0);
        let assignment = Assignment::deal(&config);
        let silent = assignment.pack().iter().next().unwrap().clone();
        let (records, writer) = Writer::spawn(io::sink());
        let (mut episode, outcomes) = moderate(&config, assignment.clone(), records);
        for (who, role) in assignment.players() {
            if *who == silent {
                add(&mut episode, who, Silent);
            } else {
                seat(&mut episode, &config, who, role);
            }
        }

        let error = play(episode, &outcomes, writer, None).unwrap_err();
        let RunError::Episode(EpisodeError::Stalled { running }) = &error else {
            panic!("unexpected error: {error:?}");
        };
        assert_eq!(*running, config.players.iter().cloned().collect());
        assert!(error.to_string().contains("stalled"), "{error}");
    }
}
