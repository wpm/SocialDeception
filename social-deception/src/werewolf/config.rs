//! The configuration of a Werewolf episode, read from a TOML file.
//!
//! ```toml
//! seed = 20260918
//! players = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
//! trajectory = "werewolf.jsonl"   # optional
//! max_rounds = 100                # optional, default 100
//! moderator = "moderator"         # optional, default "moderator"
//!
//! [roles]
//! werewolves = 2
//! seers = 1                       # optional, default 0
//! doctors = 1                     # optional, default 0
//! # villagers are whatever is left over
//! ```
//!
//! [`load`] reads, parses and validates in one step, so a [`Config`] that
//! came from it has passed every check [`ConfigError`] names and a bad file
//! never reaches a thread. A key the schema does not know is a parse error,
//! so a misspelled optional key cannot silently take its default.
//!
//! A configuration also writes back out, as the *effective configuration* of
//! a run: [`Config::effective`] is the TOML that [`load`] reads back to the
//! same value, and [`write_effective`] puts it beside a trajectory, at
//! [`effective_path`], so that a run can be reproduced from its artifacts
//! alone.

use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::event::AgentId;

/// The round cap when the file does not set one.
pub const DEFAULT_MAX_ROUNDS: u32 = 100;

/// The moderator's id when the file does not set one.
pub const DEFAULT_MODERATOR: &str = "moderator";

/// Everything a run of Werewolf is parameterized by.
///
/// A `Config` built by hand can be invalid; [`Config::validate`] is the check
/// that [`load`] and [`Config::parse`] apply.
///
/// It serializes with the same field names and defaults [`load`] reads, so
/// what is written back reads back to an equal `Config`; an unset
/// `trajectory` is left out rather than written as nothing.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The master seed every generator in the episode is derived from.
    pub seed: u64,
    /// The players, in the order written. The order does not affect the
    /// deal, which sorts them first.
    pub players: Vec<AgentId>,
    /// How many of each special role to deal. The rest are villagers.
    pub roles: RoleCounts,
    /// Where to write the trajectory, if anywhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trajectory: Option<PathBuf>,
    /// The round after which an unfinished game is a stalemate.
    #[serde(default = "default_max_rounds")]
    pub max_rounds: u32,
    /// The moderator's agent id. The moderator is an agent in the same
    /// roster as the players, so no player may have this id.
    #[serde(default = "default_moderator")]
    pub moderator: AgentId,
}

/// How many of each special role a game has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleCounts {
    /// The pack. At least one.
    pub werewolves: usize,
    /// Zero or one.
    #[serde(default)]
    pub seers: usize,
    /// Zero or one.
    #[serde(default)]
    pub doctors: usize,
}

impl RoleCounts {
    /// How many players hold a special role: everyone but the villagers.
    #[must_use]
    pub const fn special(self) -> usize {
        self.werewolves + self.seers + self.doctors
    }
}

fn default_max_rounds() -> u32 {
    DEFAULT_MAX_ROUNDS
}

fn default_moderator() -> AgentId {
    AgentId::new(DEFAULT_MODERATOR)
}

/// Why a configuration was rejected.
#[derive(Debug)]
pub enum ConfigError {
    /// The file could not be read.
    Read {
        /// The file.
        path: PathBuf,
        /// What went wrong.
        source: io::Error,
    },
    /// The text is not a configuration.
    Parse(toml::de::Error),
    /// The effective configuration could not be written.
    Write {
        /// The file.
        path: PathBuf,
        /// What went wrong.
        source: io::Error,
    },
    /// `roles.werewolves` is zero: a game with no werewolves is over before
    /// it starts.
    NoWerewolves,
    /// The werewolves are at least half the players, so the werewolves' win
    /// condition already holds at deal time.
    WerewolfParity {
        /// `roles.werewolves`.
        werewolves: usize,
        /// `players.len()`.
        players: usize,
    },
    /// There are more special roles than players to hold them.
    TooManyRoles {
        /// Werewolves, seers and doctors together.
        roles: usize,
        /// `players.len()`.
        players: usize,
    },
    /// `roles.seers` is more than one.
    TooManySeers(usize),
    /// `roles.doctors` is more than one.
    TooManyDoctors(usize),
    /// A player id, or the moderator's, is the empty string.
    EmptyAgentId,
    /// A player id appears more than once.
    DuplicatePlayer(AgentId),
    /// A player has the moderator's id.
    PlayerIsModerator(AgentId),
    /// `max_rounds` is zero.
    NoRounds,
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Read { path, .. } => write!(f, "cannot read {}", path.display()),
            Self::Parse(error) => write!(f, "invalid configuration: {error}"),
            Self::Write { path, .. } => write!(f, "cannot write {}", path.display()),
            Self::NoWerewolves => f.write_str("roles.werewolves must be at least 1"),
            Self::WerewolfParity {
                werewolves,
                players,
            } => write!(
                f,
                "{werewolves} werewolves among {players} players already satisfy the werewolves' \
                 win condition; players must outnumber werewolves more than two to one"
            ),
            Self::TooManyRoles { roles, players } => {
                write!(f, "{roles} special roles but only {players} players")
            }
            Self::TooManySeers(seers) => write!(f, "roles.seers must be 0 or 1, not {seers}"),
            Self::TooManyDoctors(doctors) => {
                write!(f, "roles.doctors must be 0 or 1, not {doctors}")
            }
            Self::EmptyAgentId => f.write_str("agent ids must not be empty"),
            Self::DuplicatePlayer(who) => write!(f, "player {who:?} is listed more than once"),
            Self::PlayerIsModerator(who) => {
                write!(f, "player {who:?} has the moderator's id")
            }
            Self::NoRounds => f.write_str("max_rounds must be at least 1"),
        }
    }
}

impl Error for ConfigError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Read { source, .. } | Self::Write { source, .. } => Some(source),
            Self::Parse(source) => Some(source),
            _ => None,
        }
    }
}

/// Where the effective configuration of a run is written, beside its
/// trajectory: the trajectory's path with `.toml` appended, so
/// `werewolf.jsonl` has `werewolf.jsonl.toml` beside it.
///
/// The effective configuration is the [`Config`] a run was played from after
/// any command-line overrides, without its `trajectory` field. It exists
/// because the seed never appears in a trajectory, and a run has to be
/// reproducible from its artifacts. Appending the extension rather than
/// replacing it means the file can never collide with the configuration the
/// run was started from, however the two are named.
#[must_use]
pub fn effective_path(trajectory: &Path) -> PathBuf {
    let mut path = trajectory.as_os_str().to_owned();
    path.push(".toml");
    PathBuf::from(path)
}

/// Writes the effective configuration of a run played from `config` beside
/// its trajectory, at [`effective_path`], and returns where it was written.
///
/// # Errors
///
/// [`ConfigError::Write`] if the file cannot be written.
pub fn write_effective(config: &Config, trajectory: &Path) -> Result<PathBuf, ConfigError> {
    let path = effective_path(trajectory);
    fs::write(&path, config.effective()).map_err(|source| ConfigError::Write {
        path: path.clone(),
        source,
    })?;
    Ok(path)
}

/// Reads and validates a configuration file.
///
/// # Errors
///
/// If the file cannot be read, is not TOML of the documented shape, or fails
/// any check in [`ConfigError`].
pub fn load(path: impl AsRef<Path>) -> Result<Config, ConfigError> {
    let path = path.as_ref();
    let text = fs::read_to_string(path).map_err(|source| ConfigError::Read {
        path: path.to_path_buf(),
        source,
    })?;
    Config::parse(&text)
}

impl Config {
    /// Parses and validates the text of a configuration file.
    ///
    /// # Errors
    ///
    /// If the text is not TOML of the documented shape, or fails any check in
    /// [`ConfigError`].
    pub fn parse(text: &str) -> Result<Self, ConfigError> {
        let config: Self = toml::from_str(text).map_err(ConfigError::Parse)?;
        config.validate()?;
        Ok(config)
    }

    /// Checks that the configuration describes a game that can be played.
    ///
    /// # Errors
    ///
    /// The first check that fails, in the order [`ConfigError`] lists them.
    pub fn validate(&self) -> Result<(), ConfigError> {
        let RoleCounts {
            werewolves,
            seers,
            doctors,
        } = self.roles;
        let players = self.players.len();
        let special = self.roles.special();
        if werewolves == 0 {
            return Err(ConfigError::NoWerewolves);
        }
        if players <= 2 * werewolves {
            return Err(ConfigError::WerewolfParity {
                werewolves,
                players,
            });
        }
        // Implied by the parity check and the caps on seers and doctors, so
        // it only ever fires alongside one of those; kept, and checked
        // before the caps, as the general rule the caps are cases of.
        if special > players {
            return Err(ConfigError::TooManyRoles {
                roles: special,
                players,
            });
        }
        if seers > 1 {
            return Err(ConfigError::TooManySeers(seers));
        }
        if doctors > 1 {
            return Err(ConfigError::TooManyDoctors(doctors));
        }
        if self.moderator.as_str().is_empty() {
            return Err(ConfigError::EmptyAgentId);
        }
        let mut seen = BTreeSet::new();
        for player in &self.players {
            if player.as_str().is_empty() {
                return Err(ConfigError::EmptyAgentId);
            }
            if !seen.insert(player) {
                return Err(ConfigError::DuplicatePlayer(player.clone()));
            }
            if *player == self.moderator {
                return Err(ConfigError::PlayerIsModerator(player.clone()));
            }
        }
        if self.max_rounds == 0 {
            return Err(ConfigError::NoRounds);
        }
        Ok(())
    }

    /// The effective configuration of a run played from this one: the same
    /// configuration as TOML, without its `trajectory` field.
    ///
    /// It is a valid configuration file in the schema [`load`] reads, and
    /// reads back as this configuration with no trajectory. That is what
    /// makes reproducing a run `werewolf play <trajectory>.toml --trajectory
    /// <elsewhere>`: the trajectory is omitted so that replaying the file
    /// cannot truncate the very trajectory it describes, and a reproduction
    /// names its own output.
    ///
    /// # Panics
    ///
    /// Never for a configuration [`load`] accepted; a configuration is plain
    /// data that TOML can always represent.
    #[must_use]
    pub fn effective(&self) -> String {
        let effective = Self {
            trajectory: None,
            ..self.clone()
        };
        toml::to_string(&effective).expect("a configuration is representable as TOML")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempPath;

    /// The example configuration, as committed at the repository root.
    const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/werewolf.toml");

    /// A configuration with every field present.
    const FULL: &str = r#"
        seed = 20260918
        players = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
        trajectory = "werewolf.jsonl"
        max_rounds = 50
        moderator = "narrator"

        [roles]
        werewolves = 2
        seers = 1
        doctors = 1
    "#;

    /// A configuration with only the required fields.
    const MINIMAL: &str = r#"
        seed = 1
        players = ["alice", "bob", "carol"]

        [roles]
        werewolves = 1
    "#;

    fn ids<const N: usize>(names: [&str; N]) -> Vec<AgentId> {
        names.map(AgentId::new).into()
    }

    /// A valid configuration to break one field of at a time.
    fn valid() -> Config {
        Config::parse(FULL).unwrap()
    }

    #[test]
    fn every_field_parses() {
        assert_eq!(
            valid(),
            Config {
                seed: 20_260_918,
                players: ids(["alice", "bob", "carol", "dave", "erin", "frank", "grace"]),
                roles: RoleCounts {
                    werewolves: 2,
                    seers: 1,
                    doctors: 1,
                },
                trajectory: Some(PathBuf::from("werewolf.jsonl")),
                max_rounds: 50,
                moderator: AgentId::new("narrator"),
            }
        );
    }

    #[test]
    fn required_fields_alone_get_the_defaults() {
        assert_eq!(
            Config::parse(MINIMAL).unwrap(),
            Config {
                seed: 1,
                players: ids(["alice", "bob", "carol"]),
                roles: RoleCounts {
                    werewolves: 1,
                    seers: 0,
                    doctors: 0,
                },
                trajectory: None,
                max_rounds: DEFAULT_MAX_ROUNDS,
                moderator: AgentId::new(DEFAULT_MODERATOR),
            }
        );
    }

    #[test]
    fn the_example_loads() {
        let config = load(EXAMPLE).unwrap();
        assert_eq!(config.players.len(), 7);
        assert_eq!(config.roles.werewolves, 2);
    }

    #[test]
    fn the_effective_config_sits_beside_the_trajectory() {
        assert_eq!(
            effective_path(Path::new("werewolf.jsonl")),
            PathBuf::from("werewolf.jsonl.toml")
        );
        assert_eq!(
            effective_path(Path::new("runs/first.jsonl")),
            PathBuf::from("runs/first.jsonl.toml")
        );
        // The extension is appended, never replaced, so a trajectory named
        // like a configuration cannot have its configuration overwritten.
        assert_eq!(
            effective_path(Path::new("werewolf.toml")),
            PathBuf::from("werewolf.toml.toml")
        );
    }

    #[test]
    fn a_missing_file_is_a_read_error() {
        let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/no-such-file.toml"));
        let error = load(&path).unwrap_err();
        assert!(
            matches!(&error, ConfigError::Read { path: p, .. } if *p == path),
            "{error:?}"
        );
        assert!(error.source().is_some());
        assert!(error.to_string().contains("no-such-file.toml"));
    }

    #[test]
    fn a_missing_required_field_is_a_parse_error() {
        let error = Config::parse("seed = 1\n[roles]\nwerewolves = 1\n").unwrap_err();
        assert!(matches!(error, ConfigError::Parse(_)), "{error:?}");
        assert!(error.to_string().contains("players"), "{error}");
    }

    #[test]
    fn an_unknown_field_is_a_parse_error() {
        let text = format!("{MINIMAL}\nmax_round = 5\n");
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::Parse(_)), "{error:?}");
        assert!(error.to_string().contains("max_round"), "{error}");
    }

    #[test]
    fn no_werewolves() {
        let mut config = valid();
        config.roles.werewolves = 0;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::NoWerewolves), "{error:?}");
        assert!(error.to_string().contains("werewolves"));
    }

    #[test]
    fn werewolves_must_be_outnumbered_more_than_two_to_one() {
        let mut config = valid();
        config.players = ids(["alice", "bob", "carol", "dave"]);
        let error = config.validate().unwrap_err();
        assert!(
            matches!(
                error,
                ConfigError::WerewolfParity {
                    werewolves: 2,
                    players: 4
                }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains("2 werewolves among 4 players"));
        // One more player and the check passes.
        config.players.push(AgentId::new("erin"));
        config.validate().unwrap();
    }

    #[test]
    fn too_many_roles() {
        let mut config = valid();
        config.players = ids(["alice", "bob", "carol"]);
        config.roles.werewolves = 1;
        config.roles.seers = 1;
        config.roles.doctors = 1;
        // Parity holds for one werewolf among three, but every player has a
        // special role and there are none to spare.
        config.validate().unwrap();
        config.players = ids(["alice", "bob", "carol", "dave", "erin"]);
        config.roles.werewolves = 2;
        config.roles.seers = 2;
        config.roles.doctors = 2;
        let error = config.validate().unwrap_err();
        assert!(
            matches!(
                error,
                ConfigError::TooManyRoles {
                    roles: 6,
                    players: 5
                }
            ),
            "{error:?}"
        );
        assert!(error.to_string().contains("6 special roles"));
    }

    #[test]
    fn too_many_seers() {
        let mut config = valid();
        config.roles.seers = 2;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::TooManySeers(2)), "{error:?}");
        assert!(error.to_string().contains("seers"));
    }

    #[test]
    fn too_many_doctors() {
        let mut config = valid();
        config.roles.doctors = 2;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::TooManyDoctors(2)), "{error:?}");
        assert!(error.to_string().contains("doctors"));
    }

    #[test]
    fn empty_player_id() {
        let mut config = valid();
        config.players[3] = AgentId::new("");
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::EmptyAgentId), "{error:?}");
        assert!(error.to_string().contains("empty"));
    }

    #[test]
    fn empty_moderator_id() {
        let mut config = valid();
        config.moderator = AgentId::new("");
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::EmptyAgentId), "{error:?}");
    }

    #[test]
    fn duplicate_player() {
        let mut config = valid();
        config.players[5] = AgentId::new("bob");
        let error = config.validate().unwrap_err();
        assert!(
            matches!(&error, ConfigError::DuplicatePlayer(who) if who.as_str() == "bob"),
            "{error:?}"
        );
        assert!(error.to_string().contains("\"bob\""));
    }

    #[test]
    fn player_is_moderator() {
        let mut config = valid();
        config.players[0] = AgentId::new("narrator");
        let error = config.validate().unwrap_err();
        assert!(
            matches!(&error, ConfigError::PlayerIsModerator(who) if who.as_str() == "narrator"),
            "{error:?}"
        );
        assert!(error.to_string().contains("moderator"));
    }

    #[test]
    fn no_rounds() {
        let mut config = valid();
        config.max_rounds = 0;
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::NoRounds), "{error:?}");
        assert!(error.to_string().contains("max_rounds"));
    }

    #[test]
    fn the_effective_config_reads_back_as_the_config_without_its_trajectory() {
        let config = valid();
        let text = config.effective();
        assert!(!text.contains("trajectory"), "{text}");
        assert_eq!(
            Config::parse(&text).unwrap(),
            Config {
                trajectory: None,
                ..config
            }
        );
    }

    #[test]
    fn the_effective_config_writes_the_defaults_out_in_full() {
        // What was defaulted on the way in is explicit on the way out, so the
        // file says what ran even if a default changes later.
        let text = Config::parse(MINIMAL).unwrap().effective();
        assert!(
            text.contains(&format!("max_rounds = {DEFAULT_MAX_ROUNDS}")),
            "{text}"
        );
        assert!(
            text.contains(&format!("moderator = \"{DEFAULT_MODERATOR}\"")),
            "{text}"
        );
        assert!(text.contains("seers = 0"), "{text}");
        assert_eq!(
            Config::parse(&text).unwrap(),
            Config::parse(MINIMAL).unwrap()
        );
    }

    #[test]
    fn the_effective_config_is_written_beside_the_trajectory_and_loads() {
        let trajectory = TempPath::new("jsonl");
        let written = write_effective(&valid(), &trajectory).unwrap();
        assert_eq!(written, effective_path(&trajectory));
        assert_eq!(
            load(&written).unwrap(),
            Config {
                trajectory: None,
                ..valid()
            }
        );
        fs::remove_file(written).unwrap();
    }

    #[test]
    fn an_unwritable_effective_config_is_a_write_error_naming_it() {
        let trajectory = Path::new("/no-such-directory/werewolf.jsonl");
        let error = write_effective(&valid(), trajectory).unwrap_err();
        assert!(
            matches!(&error, ConfigError::Write { path, .. } if *path == effective_path(trajectory)),
            "{error:?}"
        );
        assert!(error.source().is_some());
        assert_eq!(
            error.to_string(),
            "cannot write /no-such-directory/werewolf.jsonl.toml"
        );
    }

    #[test]
    fn validation_happens_in_parse() {
        let text = FULL.replace("werewolves = 2", "werewolves = 0");
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::NoWerewolves), "{error:?}");
    }
}
