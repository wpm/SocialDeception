//! `werewolf`: play and replay episodes of Werewolf.
//!
//! `play` loads a configuration file, applies any overrides from the command
//! line, deals the roles and prints the result: the effective seed, the
//! roster with each player's role, and the counts. The effective seed is
//! printed on every run because `--seed` means the file is no longer the
//! sole determinant of the run. No game is played yet.
//!
//! `replay` reads a trajectory written by an earlier run back as a
//! [`Transcript`] and prints it, under a header naming the seed. The seed is
//! not in the trajectory: it comes from the effective configuration the run
//! wrote beside it, at the trajectory's path with `.toml` appended, which
//! also names the moderator whose records are the game. Without that file
//! the trajectory is still a game, just not a reproducible one: the header
//! says the seed is unknown, and the moderator's id is `--moderator`.
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
use social_deception::werewolf::{Assignment, Config, ConfigError, Role, Transcript, config};

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
            let config = load(&config, seed, trajectory)?;
            let assignment = Assignment::deal(&config);
            let mut text = String::new();
            describe(&mut text, &config, &assignment)?;
            print!("{text}");
            Ok(())
        }
        Command::Replay {
            trajectory,
            moderator,
        } => {
            let replay = replay(&trajectory, moderator.as_deref())?;
            for note in &replay.notes {
                eprintln!("werewolf: {note}");
            }
            print!("{replay}");
            Ok(())
        }
    }
}

/// A trajectory read back as a game, with what the header needs.
#[derive(Debug)]
struct Replay {
    /// The seed the run was played from, if its effective config was found.
    seed: Option<u64>,
    /// What the reader had to decide on the way, for stderr.
    notes: Vec<String>,
    transcript: Transcript,
}

/// Reads a trajectory and the effective config beside it, if there is one.
///
/// The effective config is the record of what actually ran, so where it
/// exists it names the moderator; a `--moderator` that disagrees with it is
/// noted and ignored. Where it does not, the moderator is `--moderator` or
/// the default, and the seed is unknown.
fn replay(trajectory: &Path, moderator: Option<&str>) -> Result<Replay, Box<dyn Error>> {
    let text = fs::read_to_string(trajectory)
        .map_err(|error| format!("cannot read {}: {error}", trajectory.display()))?;
    let effective_path = config::effective_path(trajectory);
    let effective = match config::load(&effective_path) {
        Ok(config) => Some(config),
        Err(ConfigError::Read { source, .. }) if source.kind() == io::ErrorKind::NotFound => None,
        Err(error) => return Err(error.into()),
    };
    let mut notes = Vec::new();
    let moderator = match (&effective, moderator) {
        (Some(effective), Some(given)) if given != effective.moderator.as_str() => {
            notes.push(format!(
                "--moderator {given} disagrees with {}, which names {}; using {}",
                effective_path.display(),
                effective.moderator,
                effective.moderator
            ));
            effective.moderator.clone()
        }
        (Some(effective), _) => effective.moderator.clone(),
        (None, given) => {
            let moderator = AgentId::new(given.unwrap_or(config::DEFAULT_MODERATOR));
            notes.push(format!(
                "no effective config at {}; the seed is unknown and the moderator is taken to be \
                 {moderator}",
                effective_path.display()
            ));
            moderator
        }
    };
    let transcript = Transcript::read(&transcript::lines(&text)?, &moderator)?;
    Ok(Replay {
        seed: effective.map(|config| config.seed),
        notes,
        transcript,
    })
}

impl fmt::Display for Replay {
    /// The header, a blank line, then the transcript.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Werewolf \u{2014} seed ")?;
        match self.seed {
            Some(seed) => write!(f, "{seed}")?,
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

/// Writes the effective configuration and the deal, as printed by `play`.
fn describe(out: &mut impl fmt::Write, config: &Config, assignment: &Assignment) -> fmt::Result {
    writeln!(out, "seed: {}", config.seed)?;
    writeln!(out, "moderator: {}", config.moderator)?;
    match &config.trajectory {
        Some(path) => writeln!(out, "trajectory: {}", path.display())?,
        None => writeln!(out, "trajectory: none")?,
    }
    writeln!(out, "max_rounds: {}", config.max_rounds)?;
    writeln!(out)?;
    let width = assignment
        .players()
        .map(|(who, _)| who.as_str().len())
        .max()
        .unwrap_or(0);
    for (who, role) in assignment.players() {
        writeln!(out, "{:<width$}  {role}", who.as_str())?;
    }
    writeln!(out)?;
    writeln!(
        out,
        "werewolves: {}, seers: {}, doctors: {}, villagers: {}",
        assignment.count(Role::Werewolf),
        assignment.count(Role::Seer),
        assignment.count(Role::Doctor),
        assignment.count(Role::Villager),
    )
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

    #[test]
    fn play_prints_the_effective_seed_the_roster_and_the_counts() {
        let config = load(&example(), Some(7), None).unwrap();
        let assignment = Assignment::deal(&config);
        let mut text = String::new();
        describe(&mut text, &config, &assignment).unwrap();
        assert!(text.starts_with("seed: 7\n"), "{text}");
        assert!(text.contains("trajectory: werewolf.jsonl\n"), "{text}");
        for (who, role) in assignment.players() {
            assert!(
                text.contains(&format!("{:<5}  {role}\n", who.as_str())),
                "{text}"
            );
        }
        assert!(
            text.ends_with("werewolves: 2, seers: 1, doctors: 1, villagers: 3\n"),
            "{text}"
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

    /// A copy of a file in the temp dir, removed when this is dropped.
    struct TempCopy(PathBuf);

    impl TempCopy {
        fn of(source: &Path, name: &str) -> Self {
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "social-deception-werewolf-{}-{}-{name}",
                process::id(),
                COUNTER.fetch_add(1, Ordering::Relaxed)
            ));
            fs::copy(source, &path).unwrap();
            Self(path)
        }
    }

    impl Drop for TempCopy {
        fn drop(&mut self) {
            let _ = fs::remove_file(&self.0);
        }
    }

    #[test]
    fn replay_with_the_effective_config_uses_its_seed_and_moderator() {
        let replayed = replay(&fixture(), None).unwrap();
        assert_eq!(replayed.seed, Some(26));
        assert!(replayed.notes.is_empty(), "{:?}", replayed.notes);
        assert_eq!(replayed.transcript.rounds.len(), 3);
        let text = replayed.to_string();
        let golden = fs::read_to_string(fixture().with_extension("txt")).unwrap();
        assert_eq!(
            text,
            format!(
                "Werewolf \u{2014} seed 26, 7 players (2 werewolves, 1 seer, 1 doctor)\n\n{golden}"
            )
        );
    }

    #[test]
    fn the_effective_config_wins_over_a_disagreeing_moderator_flag() {
        let disagreeing = replay(&fixture(), Some("narrator")).unwrap();
        assert_eq!(disagreeing.seed, Some(26));
        let notes = &disagreeing.notes;
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("--moderator narrator"), "{notes:?}");
        assert!(notes[0].contains("using moderator"), "{notes:?}");
        assert_eq!(disagreeing.transcript.rounds.len(), 3);
        // The same flag, agreeing, is not worth a note.
        let agreeing = replay(&fixture(), Some("moderator")).unwrap();
        assert!(agreeing.notes.is_empty(), "{:?}", agreeing.notes);
    }

    #[test]
    fn replay_without_the_effective_config_falls_back_to_the_flag() {
        let alone = TempCopy::of(&fixture(), "alone.jsonl");
        assert!(!config::effective_path(&alone.0).exists());

        let replayed = replay(&alone.0, None).unwrap();
        assert_eq!(replayed.seed, None);
        let notes = &replayed.notes;
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].contains("seed is unknown"), "{notes:?}");
        assert!(notes[0].contains("taken to be moderator"), "{notes:?}");
        assert_eq!(replayed.transcript.rounds.len(), 3);
        assert!(
            replayed
                .to_string()
                .starts_with("Werewolf \u{2014} seed unknown, 7 players")
        );

        // The wrong moderator finds no game, and says so.
        let error = replay(&alone.0, Some("narrator")).unwrap_err();
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
        let corrupt = TempCopy::of(&fixture(), "corrupt.jsonl");
        let mut text = fs::read_to_string(&corrupt.0).unwrap();
        text.insert_str(0, "not json\n");
        fs::write(&corrupt.0, text).unwrap();
        let error = replay(&corrupt.0, None).unwrap_err();
        assert!(
            error.to_string().starts_with("line 1 is not JSON"),
            "{error}"
        );
    }

    #[test]
    fn a_broken_effective_config_is_an_error() {
        let trajectory = TempCopy::of(&fixture(), "broken.jsonl");
        let effective = config::effective_path(&trajectory.0);
        fs::write(&effective, "seed = 26\n").unwrap();
        let error = replay(&trajectory.0, None).unwrap_err();
        fs::remove_file(&effective).unwrap();
        assert!(
            error.to_string().contains("invalid configuration"),
            "{error}"
        );
    }
}
