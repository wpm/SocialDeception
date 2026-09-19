//! `werewolf`: play and replay episodes of Werewolf.
//!
//! `play` loads a configuration file, applies any overrides from the command
//! line, deals the roles and prints the result: the effective seed, the
//! roster with each player's role, and the counts. The effective seed is
//! printed on every run because `--seed` means the file is no longer the
//! sole determinant of the run. No game is played yet. `replay` is not
//! implemented yet.
//!
//! Configuration and I/O errors go to stderr with a non-zero exit; usage
//! errors are clap's.

use std::error::Error;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand};
use social_deception::werewolf::{Assignment, Config, Role, config};

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
        /// The moderator's agent id in that trajectory.
        #[arg(long, default_value = "moderator")]
        moderator: String,
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
            print!("{}", describe(&config, &assignment));
            Ok(())
        }
        Command::Replay { .. } => Err("replay is not implemented yet".into()),
    }
}

/// Loads the configuration and applies the command line's overrides, so the
/// run has one source of truth.
fn load(
    path: &PathBuf,
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
    Ok(config)
}

/// The effective configuration and the deal, as printed by `play`.
fn describe(config: &Config, assignment: &Assignment) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "seed: {}", config.seed);
    let _ = writeln!(out, "moderator: {}", config.moderator);
    match &config.trajectory {
        Some(path) => {
            let _ = writeln!(out, "trajectory: {}", path.display());
        }
        None => out.push_str("trajectory: none\n"),
    }
    let _ = writeln!(out, "max_rounds: {}", config.max_rounds);
    out.push('\n');
    let width = assignment
        .players()
        .map(|(who, _)| who.as_str().len())
        .max()
        .unwrap_or(0);
    for (who, role) in assignment.players() {
        let _ = writeln!(out, "{:<width$}  {role:?}", who.as_str());
    }
    out.push('\n');
    let _ = writeln!(
        out,
        "werewolves: {}, seers: {}, doctors: {}, villagers: {}",
        assignment.count(Role::Werewolf),
        assignment.count(Role::Seer),
        assignment.count(Role::Doctor),
        assignment.count(Role::Villager),
    );
    out
}

#[cfg(test)]
mod tests {
    use std::path::Path;

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

    fn parse<const N: usize>(args: [&str; N]) -> Result<Cli, clap::Error> {
        Cli::try_parse_from(args)
    }

    #[test]
    fn the_command_line_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn play_with_only_its_config() {
        let cli = parse(["werewolf", "play", "x.toml"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Play { config, seed: None, trajectory: None }
                if config == Path::new("x.toml")
        ));
    }

    #[test]
    fn play_with_every_flag() {
        let cli = parse([
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
        let cli = parse(["werewolf", "replay", "run.jsonl"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Replay { trajectory, moderator }
                if trajectory == Path::new("run.jsonl") && moderator == "moderator"
        ));
    }

    #[test]
    fn replay_with_a_moderator() {
        let cli = parse(["werewolf", "replay", "run.jsonl", "--moderator", "narrator"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Replay { moderator, .. } if moderator == "narrator"
        ));
    }

    #[test]
    fn a_missing_argument_is_a_usage_error() {
        let error = parse(["werewolf", "play"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::MissingRequiredArgument);
        let error = parse(["werewolf"]).unwrap_err();
        assert_eq!(
            error.kind(),
            ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
        );
    }

    #[test]
    fn an_unknown_flag_is_a_usage_error() {
        let error = parse(["werewolf", "play", "x.toml", "--players", "3"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::UnknownArgument);
    }

    #[test]
    fn a_seed_that_is_not_a_number_is_a_usage_error() {
        let error = parse(["werewolf", "play", "x.toml", "--seed", "lucky"]).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::ValueValidation);
    }

    #[test]
    fn help_and_version_are_wired() {
        let help = parse(["werewolf", "--help"]).unwrap_err();
        assert_eq!(help.kind(), ErrorKind::DisplayHelp);
        let text = help.to_string();
        assert!(text.contains("play"), "{text}");
        assert!(text.contains("replay"), "{text}");
        let version = parse(["werewolf", "--version"]).unwrap_err();
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
    fn a_seed_override_changes_the_deal() {
        let from_file = load(&example(), None, None).unwrap();
        let overridden = load(&example(), Some(7), None).unwrap();
        assert_ne!(Assignment::deal(&from_file), Assignment::deal(&overridden));
    }

    #[test]
    fn play_prints_the_effective_seed_the_roster_and_the_counts() {
        let config = load(&example(), Some(7), None).unwrap();
        let assignment = Assignment::deal(&config);
        let text = describe(&config, &assignment);
        assert!(text.starts_with("seed: 7\n"), "{text}");
        assert!(text.contains("trajectory: werewolf.jsonl\n"), "{text}");
        for (who, role) in assignment.players() {
            assert!(
                text.contains(&format!("{:<5}  {role:?}\n", who.as_str())),
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
        let cli = parse(["werewolf", "play", "no-such-file.toml"]).unwrap();
        let error = run(cli).unwrap_err();
        assert!(error.to_string().contains("no-such-file.toml"), "{error}");
    }

    #[test]
    fn replay_is_not_implemented() {
        let cli = parse(["werewolf", "replay", "run.jsonl"]).unwrap();
        let error = run(cli).unwrap_err();
        assert!(error.to_string().contains("not implemented"), "{error}");
    }
}
