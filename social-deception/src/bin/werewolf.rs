//! `werewolf`: play and replay episodes of Werewolf.
//!
//! `play` loads a configuration file, applies any overrides from the command
//! line, plays one episode and prints how it ended: the effective seed, the
//! winner, the number of rounds and the survivors. The effective seed is
//! printed on every run because `--seed` means the file is no longer the
//! sole determinant of the run.
//!
//! For the same reason, when a trajectory is written the *effective
//! configuration* is written beside it, at the trajectory's path with
//! `.toml` appended: the configuration the game was played from, after
//! overrides, without its `trajectory` field. The seed is never in the
//! trajectory itself (see [`werewolf::setup`]), and this file is where a
//! run's seed is kept. It is a configuration file in its own right, so
//! reproducing a run is `werewolf play run.jsonl.toml --trajectory
//! rerun.jsonl`, whatever flags produced the original.
//!
//! `replay` reads a trajectory written by an earlier run back as a
//! [`Transcript`] and prints it, under a header naming the seed, which it
//! takes from the effective configuration beside the trajectory, along with
//! the moderator whose records are the game. Without that file the
//! trajectory is still a game, just not a reproducible one: the header says
//! the seed is unknown, and the moderator's id is `--moderator`.
//!
//! Configuration, trajectory and I/O errors go to stderr with a non-zero
//! exit; usage errors are clap's.

use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use social_deception::AgentId;
use social_deception::werewolf::transcript;
use social_deception::werewolf::{self, Config, ConfigError, Outcome, Role, Transcript, config};

/// Run a Werewolf episode.
#[derive(Debug, Parser)]
#[command(name = "werewolf", version, about = "Run a Werewolf episode")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

/// What to do.
#[derive(Debug, Subcommand)]
enum Command {
    /// Play one episode from a configuration file.
    Play {
        /// The TOML configuration to run.
        config: PathBuf,
        /// Override the configuration's seed.
        #[arg(long)]
        seed: Option<u64>,
        /// Override where the trajectory is written.
        #[arg(long)]
        trajectory: Option<PathBuf>,
    },
    /// Render a trajectory written by an earlier run.
    Replay {
        /// The JSON Lines trajectory to read.
        trajectory: PathBuf,
        /// The moderator's agent id, when the trajectory has no effective
        /// config beside it [default: moderator].
        #[arg(long)]
        moderator: Option<String>,
    },
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("werewolf: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), Box<dyn Error>> {
    match cli.command {
        Command::Play {
            config,
            seed,
            trajectory,
        } => {
            let played = play(&config, seed, trajectory)?;
            print!("{played}");
            Ok(())
        }
        Command::Replay {
            trajectory,
            moderator,
        } => {
            let replay = replay(&trajectory, moderator.as_deref())?;
            let effective_path = config::effective_path(&trajectory);
            if let Some(given) = &replay.overridden {
                eprintln!(
                    "werewolf: --moderator {given} disagrees with {}, which names {}; using {}",
                    effective_path.display(),
                    replay.moderator,
                    replay.moderator
                );
            }
            if replay.effective.is_none() {
                eprintln!(
                    "werewolf: no effective config at {}; the seed is unknown and the moderator \
                     is taken to be {}",
                    effective_path.display(),
                    replay.moderator
                );
            }
            print!("{replay}");
            Ok(())
        }
    }
}

/// A game played to its end.
#[derive(Debug)]
struct Played {
    /// The effective configuration: the file's, after overrides. Its
    /// trajectory, if any, is where the trajectory was written, with the
    /// effective configuration beside it.
    config: Config,
    outcome: Outcome,
}

/// Plays one episode from the configuration at `path`, with the overrides
/// applied.
///
/// When the configuration names a trajectory, the effective configuration
/// is written beside it before the game begins, so that even a run that
/// ends badly leaves the pair that reproduces it. Without a trajectory there
/// is nothing to pair it with, and nothing is written.
fn play(
    path: &Path,
    seed: Option<u64>,
    trajectory: Option<PathBuf>,
) -> Result<Played, Box<dyn Error>> {
    let config = load(path, seed, trajectory)?;
    if let Some(trajectory) = &config.trajectory {
        config::write_effective(&config, trajectory)?;
    }
    let outcome = werewolf::run(&config)?;
    Ok(Played { config, outcome })
}

impl fmt::Display for Played {
    /// The effective seed, how the game ended, and what was written.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "seed: {}", self.config.seed)?;
        match self.outcome.winner {
            Some(winner) => writeln!(f, "winner: {winner}")?,
            None => writeln!(f, "winner: nobody, a stalemate at the round cap")?,
        }
        writeln!(f, "rounds: {}", self.outcome.rounds.0)?;
        let survivors: Vec<&str> = self.outcome.living.iter().map(AgentId::as_str).collect();
        writeln!(f, "survivors: {}", survivors.join(", "))?;
        match &self.config.trajectory {
            Some(trajectory) => {
                writeln!(f, "trajectory: {}", trajectory.display())?;
                writeln!(
                    f,
                    "effective config: {}",
                    config::effective_path(trajectory).display()
                )
            }
            None => writeln!(f, "trajectory: none"),
        }
    }
}

/// A trajectory read back as a game, with what the header needs.
#[derive(Debug)]
struct Replay {
    /// The effective config found beside the trajectory, if there was one.
    effective: Option<Config>,
    /// The moderator whose records were read.
    moderator: AgentId,
    /// The `--moderator` that was ignored because the effective config
    /// named someone else.
    overridden: Option<String>,
    transcript: Transcript,
}

/// Reads a trajectory and the effective config beside it, if there is one.
///
/// The effective config is the record of what actually ran, so where it
/// exists it names the moderator, and a `--moderator` that disagrees with
/// it is overridden. Where it does not, the moderator is `--moderator` or
/// the default.
fn replay(trajectory: &Path, moderator: Option<&str>) -> Result<Replay, Box<dyn Error>> {
    let text = fs::read_to_string(trajectory)
        .map_err(|error| format!("cannot read {}: {error}", trajectory.display()))?;
    let effective = match config::load(config::effective_path(trajectory)) {
        Ok(config) => Some(config),
        Err(ConfigError::Read { source, .. }) if source.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let (moderator, overridden) = match &effective {
        Some(effective) => (
            effective.moderator.clone(),
            moderator
                .filter(|given| *given != effective.moderator.as_str())
                .map(str::to_owned),
        ),
        None => (
            AgentId::new(moderator.unwrap_or(config::DEFAULT_MODERATOR)),
            None,
        ),
    };
    let transcript = Transcript::read(&transcript::lines(&text)?, &moderator)?;
    Ok(Replay {
        effective,
        moderator,
        overridden,
        transcript,
    })
}

impl fmt::Display for Replay {
    /// The header, a blank line, then the transcript.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Werewolf \u{2014} seed ")?;
        match &self.effective {
            Some(config) => write!(f, "{}", config.seed)?,
            None => write!(f, "unknown")?,
        }
        let roles = &self.transcript.assignment;
        let count = |role: Role| roles.values().filter(|held| **held == role).count();
        writeln!(
            f,
            ", {} players ({}, {}, {})",
            roles.len(),
            plural(count(Role::Werewolf), "werewolf", "werewolves"),
            plural(count(Role::Seer), "seer", "seers"),
            plural(count(Role::Doctor), "doctor", "doctors"),
        )?;
        writeln!(f)?;
        write!(f, "{}", self.transcript)
    }
}

fn plural(count: usize, one: &str, many: &str) -> String {
    format!("{count} {}", if count == 1 { one } else { many })
}

/// Loads the configuration and applies the command line's overrides, so the
/// run has one source of truth. The result is validated after the overrides,
/// so an override cannot let through what the file could not.
fn load(
    path: &Path,
    seed: Option<u64>,
    trajectory: Option<PathBuf>,
) -> Result<Config, config::ConfigError> {
    let mut config = config::load(path)?;
    if let Some(seed) = seed {
        config.seed = seed;
    }
    if let Some(trajectory) = trajectory {
        config.trajectory = Some(trajectory);
    }
    config.validate()?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use std::process;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use clap::CommandFactory;
    use clap::error::ErrorKind;

    use super::*;

    /// The example configuration at the repository root.
    fn example() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../examples/werewolf.toml"
        ))
    }

    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn play_with_only_its_config() {
        let cli = Cli::try_parse_from(["werewolf", "play", "x.toml"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Play { config, seed: None, trajectory: None }
                if config == Path::new("x.toml")
        ));
    }

    #[test]
    fn play_with_every_flag() {
        let cli = Cli::try_parse_from([
            "werewolf",
            "play",
            "x.toml",
            "--seed",
            "7",
            "--trajectory",
            "out.jsonl",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Command::Play { config, seed: Some(7), trajectory: Some(trajectory) }
                if config == Path::new("x.toml") && trajectory == Path::new("out.jsonl")
        ));
    }

    #[test]
    fn replay_with_only_its_trajectory() {
        let cli = Cli::try_parse_from(["werewolf", "replay", "run.jsonl"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Replay { trajectory, moderator: None }
                if trajectory == Path::new("run.jsonl")
        ));
    }

    #[test]
    fn replay_with_a_moderator() {
        let cli =
            Cli::try_parse_from(["werewolf", "replay", "run.jsonl", "--moderator", "narrator"])
                .unwrap();
        assert!(matches!(
            cli.command,
            Command::Replay { moderator: Some(moderator), .. } if moderator == "narrator"
        ));
    }

    #[test]
    fn a_missing_argument_is_a_usage_error() {
        let error = Cli::try_parse_from(["werewolf", "play"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
        let error = Cli::try_parse_from(["werewolf"]).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error() {
        let error =
            Cli::try_parse_from(["werewolf", "play", "x.toml", "--players", "3"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn a_seed_that_is_not_a_number_is_a_usage_error() {
        let error =
            Cli::try_parse_from(["werewolf", "play", "x.toml", "--seed", "lucky"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn help_and_version_are_wired() {
        let help = Cli::try_parse_from(["werewolf", "--help"]).unwrap_err();
        assert_eq!(help.kind(), ErrorKind::DisplayHelp);
        let text = help.to_string();
        assert!(text.contains("play"), "{text}");
        assert!(text.contains("replay"), "{text}");
        let version = Cli::try_parse_from(["werewolf", "--version"]).unwrap_err();
        assert_eq!(version.kind(), ErrorKind::DisplayVersion);
        assert!(
            version.to_string().contains(env!("CARGO_PKG_VERSION")),
            "{version}"
        );
    }

    #[test]
    fn overrides_replace_the_file_s_values() {
        let from_file = load(&example(), None, None).unwrap();
        assert_eq!(from_file.seed, 20_260_918);
        assert_eq!(from_file.trajectory, Some(PathBuf::from("werewolf.jsonl")));
        let overridden = load(&example(), Some(7), Some(PathBuf::from("out.jsonl"))).unwrap();
        assert_eq!(overridden.seed, 7);
        assert_eq!(overridden.trajectory, Some(PathBuf::from("out.jsonl")));
        assert_eq!(overridden.players, from_file.players);
    }

    /// A directory of one test's own under the temp dir, removed with
    /// everything in it when this is dropped, so that a failing test leaves
    /// nothing behind.
    struct TempDir(PathBuf);

    impl TempDir {
        fn new() -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "social-deception-werewolf-{}-{}",
                process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir_all(&path).unwrap();
            Self(path)
        }

        /// A path in the directory.
        fn path(&self, name: &str) -> PathBuf {
            self.0.join(name)
        }

        /// Writes a file in the directory and returns its path.
        fn file(&self, name: &str, contents: &str) -> PathBuf {
            let path = self.path(name);
            fs::write(&path, contents).unwrap();
            path
        }

        /// Writes a five-player configuration file whose trajectory, if
        /// any, is in the directory too. The path is written as a TOML
        /// literal string, so it needs no escaping whatever the platform.
        fn config(&self, name: &str, trajectory: Option<&str>) -> PathBuf {
            let trajectory = trajectory.map_or_else(String::new, |trajectory| {
                format!("trajectory = '{}'\n", self.path(trajectory).display())
            });
            let text = format!(
                "seed = 3\nplayers = [\"alice\", \"bob\", \"carol\", \"dave\", \"erin\"]\n\
                 {trajectory}[roles]\nwerewolves = 1\nseers = 1\ndoctors = 1\n"
            );
            self.file(name, &text)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn play_plays_a_game_and_prints_how_it_ended() {
        let dir = TempDir::new();
        let trajectory = dir.path("out.jsonl");
        let played = play(&example(), Some(7), Some(trajectory.clone())).unwrap();
        assert_eq!(played.config.seed, 7);
        assert!(played.outcome.winner.is_some(), "{played:?}");
        let text = played.to_string();
        assert!(text.starts_with("seed: 7\nwinner: "), "{text}");
        assert!(text.contains("\nrounds: "), "{text}");
        assert!(text.contains("\nsurvivors: "), "{text}");
        assert!(
            text.ends_with(&format!(
                "trajectory: {}\neffective config: {}\n",
                trajectory.display(),
                config::effective_path(&trajectory).display()
            )),
            "{text}"
        );
    }

    #[test]
    fn play_writes_the_effective_config_beside_the_trajectory() {
        let dir = TempDir::new();
        let trajectory = dir.path("out.jsonl");
        let played = play(&example(), Some(7), Some(trajectory.clone())).unwrap();
        assert!(fs::metadata(&trajectory).unwrap().len() > 0);
        let effective = config::load(config::effective_path(&trajectory)).unwrap();
        assert_eq!(
            effective,
            Config {
                trajectory: None,
                ..played.config
            }
        );
        assert_eq!(effective.seed, 7);
    }

    #[test]
    fn play_without_a_trajectory_writes_nothing() {
        let dir = TempDir::new();
        let config = dir.config("game.toml", None);
        let played = play(&config, None, None).unwrap();
        assert!(played.to_string().ends_with("trajectory: none\n"));
        let mut entries = fs::read_dir(&dir.0).unwrap();
        assert_eq!(
            entries.next().unwrap().unwrap().path(),
            config,
            "only the configuration is in the directory"
        );
        assert!(entries.next().is_none());
    }

    #[test]
    fn the_effective_config_reproduces_the_run() {
        let dir = TempDir::new();
        let first = dir.path("first.jsonl");
        let original = play(&example(), Some(7), Some(first.clone())).unwrap();
        // The whole reproduction recipe: the effective config, and a
        // trajectory of the reproduction's own.
        let second = dir.path("second.jsonl");
        let reproduced = play(&config::effective_path(&first), None, Some(second.clone())).unwrap();
        assert_eq!(reproduced.outcome, original.outcome);
        assert_eq!(reproduced.config.seed, 7);
        assert_eq!(
            fs::read_to_string(config::effective_path(&second)).unwrap(),
            fs::read_to_string(config::effective_path(&first)).unwrap()
        );
    }

    #[test]
    fn the_effective_config_never_overwrites_the_input_config() {
        // The natural naming: the configuration is `game.toml` and its
        // trajectory `game.jsonl`, so replacing the trajectory's extension
        // would land on the configuration. Appending does not.
        let dir = TempDir::new();
        let config = dir.config("game.toml", Some("game.jsonl"));
        let before = fs::read_to_string(&config).unwrap();
        play(&config, Some(4), None).unwrap();
        assert_eq!(fs::read_to_string(&config).unwrap(), before);
        let effective = dir.path("game.jsonl.toml");
        assert_eq!(config::load(&effective).unwrap().seed, 4);
    }

    #[test]
    fn an_effective_config_that_cannot_be_written_is_an_error_naming_it() {
        let dir = TempDir::new();
        let trajectory = dir.path("no-such-directory/out.jsonl");
        let error = play(&example(), None, Some(trajectory)).unwrap_err();
        assert!(
            error
                .to_string()
                .ends_with("no-such-directory/out.jsonl.toml"),
            "{error}"
        );
    }

    #[test]
    fn a_trajectory_that_cannot_be_written_is_an_error_naming_it() {
        // The effective config beside the trajectory is written first, so a
        // trajectory that is a directory is what makes the run itself fail.
        let dir = TempDir::new();
        let trajectory = dir.path("out");
        fs::create_dir(&trajectory).unwrap();
        let error = play(&example(), None, Some(trajectory.clone())).unwrap_err();
        assert!(config::effective_path(&trajectory).exists());
        assert!(
            error
                .to_string()
                .starts_with(&format!("cannot write {}: ", trajectory.display())),
            "{error}"
        );
    }

    #[test]
    fn a_bad_config_is_an_error_not_a_panic() {
        let cli = Cli::try_parse_from(["werewolf", "play", "no-such-file.toml"]).unwrap();
        let error = run(cli).unwrap_err();
        assert!(error.to_string().contains("no-such-file.toml"), "{error}");
    }

    /// The fixture trajectory, with its effective config beside it.
    fn fixture() -> PathBuf {
        PathBuf::from(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/werewolf.jsonl"
        ))
    }

    fn seed(replay: &Replay) -> Option<u64> {
        replay.effective.as_ref().map(|config| config.seed)
    }

    #[test]
    fn replay_with_the_effective_config_uses_its_seed_and_moderator() {
        let replayed = replay(&fixture(), None).unwrap();
        assert_eq!(seed(&replayed), Some(26));
        assert_eq!(replayed.moderator, AgentId::new("moderator"));
        assert_eq!(replayed.overridden, None);
        assert_eq!(replayed.transcript.rounds.len(), 3);
        let golden = fs::read_to_string(fixture().with_extension("txt")).unwrap();
        assert_eq!(
            replayed.to_string(),
            format!(
                "Werewolf \u{2014} seed 26, 7 players (2 werewolves, 1 seer, 1 doctor)\n\n{golden}"
            )
        );
    }

    #[test]
    fn the_effective_config_wins_over_a_disagreeing_moderator_flag() {
        let disagreeing = replay(&fixture(), Some("narrator")).unwrap();
        assert_eq!(seed(&disagreeing), Some(26));
        assert_eq!(disagreeing.moderator, AgentId::new("moderator"));
        assert_eq!(disagreeing.overridden.as_deref(), Some("narrator"));
        assert_eq!(disagreeing.transcript.rounds.len(), 3);
        // The same flag, agreeing, is not overridden.
        let agreeing = replay(&fixture(), Some("moderator")).unwrap();
        assert_eq!(agreeing.overridden, None);
    }

    #[test]
    fn replay_without_the_effective_config_falls_back_to_the_flag() {
        let dir = TempDir::new();
        let alone = dir.file("alone.jsonl", &fs::read_to_string(fixture()).unwrap());
        assert!(!config::effective_path(&alone).exists());

        let replayed = replay(&alone, None).unwrap();
        assert_eq!(seed(&replayed), None);
        assert_eq!(replayed.moderator, AgentId::new("moderator"));
        assert_eq!(replayed.overridden, None);
        assert_eq!(replayed.transcript.rounds.len(), 3);
        assert!(
            replayed
                .to_string()
                .starts_with("Werewolf \u{2014} seed unknown, 7 players")
        );

        // The flag names the moderator, and the wrong one finds no game.
        let error = replay(&alone, Some("narrator")).unwrap_err();
        assert!(
            error.to_string().contains("never announced an outcome"),
            "{error}"
        );
    }

    #[test]
    fn a_missing_trajectory_is_an_error_not_a_panic() {
        let cli = Cli::try_parse_from(["werewolf", "replay", "no-such-run.jsonl"]).unwrap();
        let error = run(cli).unwrap_err();
        assert!(error.to_string().contains("no-such-run.jsonl"), "{error}");
    }

    #[test]
    fn a_corrupt_trajectory_is_an_error_naming_the_line() {
        let dir = TempDir::new();
        let corrupt = dir.file("corrupt.jsonl", "not json\n");
        let error = replay(&corrupt, None).unwrap_err();
        assert!(
            error.to_string().starts_with("line 1 is not JSON"),
            "{error}"
        );
    }

    #[test]
    fn a_broken_effective_config_is_an_error() {
        let dir = TempDir::new();
        let trajectory = dir.file("empty.jsonl", "");
        fs::write(config::effective_path(&trajectory), "seed = 26\n").unwrap();
        let error = replay(&trajectory, None).unwrap_err();
        assert!(
            error.to_string().contains("invalid configuration"),
            "{error}"
        );
    }
}
