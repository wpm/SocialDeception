//! The configuration of a Werewolf episode, read from a TOML file.
//!
//! ```toml
//! seed = 20260918
//! players = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
//! trajectory = "werewolf.jsonl"   # optional
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
use std::time::Duration;

use serde::{Deserialize, Serialize};

use super::seed;
use crate::event::AgentId;

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
    /// The moderator's agent id. The moderator is an agent in the same
    /// roster as the players, so no player may have this id.
    #[serde(default = "default_moderator")]
    pub moderator: AgentId,
    /// The clocks the game's pointing sessions run on.
    #[serde(default)]
    pub timing: Timing,
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

fn default_moderator() -> AgentId {
    AgentId::new(DEFAULT_MODERATOR)
}

/// The clocks a game's pointing sessions run on (ADR-0011).
///
/// A phase is made of sessions, and a session closes on a clock rather than
/// when the last member has pointed. Each night session has a quiet period
/// and a hard limit of its own, so that a slow role cannot spend another
/// role's time; the day has a hard limit only, since it closes on a
/// majority rather than on quiet.
///
/// Every field has a default, so a configuration with no `[timing]` table
/// is still a configuration. Durations are written as seconds and held as
/// [`Duration`], and `day_cap` is resolved from the number of players when
/// the file does not set it.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Timing {
    /// Days after which a game that has not been won ends as a stalemate.
    /// `None` means the number of players; [`Timing::day_cap`] resolves it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub day_cap: Option<u32>,
    /// The pack's session: the living werewolves choosing a victim.
    pub pack: NightTiming,
    /// The seer's session.
    pub seer: NightTiming,
    /// The doctor's session.
    pub doctor: NightTiming,
    /// The day's one session.
    pub day: DayTiming,
}

/// One night session's clock.
///
/// The session closes when every member has pointed and no point has
/// changed for `quiet`, or at `limit`, whichever comes first. Any change of
/// mind restarts the quiet period; a repeat of the same target does not.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NightTiming {
    /// How long the members must leave their points alone before the
    /// session closes.
    #[serde(with = "seconds")]
    pub quiet: Duration,
    /// How long the session may run whatever its members do.
    #[serde(with = "seconds")]
    pub limit: Duration,
}

/// The day session's clock.
///
/// The day has no quiet period: it closes the moment a majority of the
/// living point at the same player, or at `limit` with nobody lynched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct DayTiming {
    /// How long the day may run without a majority.
    #[serde(with = "seconds")]
    pub limit: Duration,
}

impl Default for NightTiming {
    fn default() -> Self {
        Self {
            quiet: Duration::from_secs(1),
            limit: Duration::from_secs(20),
        }
    }
}

impl Default for DayTiming {
    fn default() -> Self {
        Self {
            limit: Duration::from_secs(60),
        }
    }
}

impl Timing {
    /// The day cap this timing means for a game of `players` players: what
    /// the file set, or the number of players.
    ///
    /// A game of *n* players cannot run past *n* days under ADR-0011's
    /// rules, so the default is the smallest cap that never cuts a game
    /// short on its own.
    #[must_use]
    pub fn day_cap(&self, players: usize) -> u32 {
        self.day_cap
            .unwrap_or_else(|| u32::try_from(players).unwrap_or(u32::MAX))
    }

    /// This timing with its `day_cap` resolved against `players`, as the
    /// effective configuration writes it.
    #[must_use]
    pub fn resolved(&self, players: usize) -> Self {
        Self {
            day_cap: Some(self.day_cap(players)),
            ..*self
        }
    }

    /// Each session's clock, with the name the error messages use. The day
    /// has no quiet period, so its is `None`.
    fn sessions(&self) -> [(&'static str, Duration, Option<Duration>); 4] {
        [
            ("pack", self.pack.limit, Some(self.pack.quiet)),
            ("seer", self.seer.limit, Some(self.seer.quiet)),
            ("doctor", self.doctor.limit, Some(self.doctor.quiet)),
            ("day", self.day.limit, None),
        ]
    }

    /// Checks that every clock describes a session that can be pointed in.
    ///
    /// Parsing has already rejected anything that is not a finite,
    /// non-negative number of seconds; what is left to check is that a
    /// duration is greater than zero, that a quiet period is not longer
    /// than the limit it has to fit inside, and that the day cap leaves at
    /// least one day to play.
    ///
    /// # Errors
    ///
    /// The first check that fails, session by session in the order
    /// [`Timing::sessions`] lists them.
    fn validate(&self) -> Result<(), ConfigError> {
        for (session, limit, quiet) in self.sessions() {
            if limit.is_zero() {
                return Err(ConfigError::TimingNotPositive {
                    field: format!("timing.{session}.limit"),
                });
            }
            if let Some(quiet) = quiet {
                if quiet.is_zero() {
                    return Err(ConfigError::TimingNotPositive {
                        field: format!("timing.{session}.quiet"),
                    });
                }
                if quiet > limit {
                    return Err(ConfigError::QuietExceedsLimit {
                        session,
                        quiet,
                        limit,
                    });
                }
            }
        }
        if self.day_cap == Some(0) {
            return Err(ConfigError::DayCapTooSmall);
        }
        Ok(())
    }
}

/// Durations in a configuration file are seconds as floating-point numbers,
/// and [`Duration`] everywhere else.
///
/// `Duration::from_secs_f64` panics on a negative, infinite or absurdly
/// large value, so the conversion rejects those here rather than letting a
/// file crash a run. What survives is finite and non-negative; whether it
/// is *usable* — greater than zero, and a quiet period no longer than its
/// limit — is [`Config::validate`]'s business, so that the error names the
/// field.
mod seconds {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer, de::Error as _};

    pub(super) fn serialize<S: Serializer>(
        duration: &Duration,
        serializer: S,
    ) -> Result<S::Ok, S::Error> {
        serializer.serialize_f64(duration.as_secs_f64())
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Duration, D::Error> {
        let seconds = f64::deserialize(deserializer)?;
        if !seconds.is_finite() {
            return Err(D::Error::custom(format!("{seconds} is not a duration")));
        }
        Duration::try_from_secs_f64(seconds)
            .map_err(|error| D::Error::custom(format!("{seconds} is not a duration: {error}")))
    }
}

/// Why a configuration was rejected.
///
/// An empty agent id is not among these: `AgentId` does not deserialize
/// from the empty string, so a file naming one fails to parse.
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
    /// `roles.doctors` is more than one.
    TooManyDoctors(usize),
    /// A player id appears more than once.
    DuplicatePlayer(AgentId),
    /// A player has the moderator's id.
    PlayerIsModerator(AgentId),
    /// A player has the name of one of the seed streams that are not a
    /// player's, [`seed::RESERVED`], and would share its generator with it.
    ReservedPlayer(AgentId),
    /// A timing duration is zero. A session with no time cannot be pointed
    /// in.
    TimingNotPositive {
        /// The field, as it is written in the file, such as `timing.pack.quiet`.
        field: String,
    },
    /// A night session's quiet period is longer than its hard limit, so the
    /// session could never close on quiet.
    QuietExceedsLimit {
        /// The session, such as `pack`.
        session: &'static str,
        /// `quiet`.
        quiet: Duration,
        /// `limit`.
        limit: Duration,
    },
    /// `timing.day_cap` is zero: a game must have at least one day to play.
    DayCapTooSmall,
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
            Self::TooManyDoctors(doctors) => {
                write!(f, "roles.doctors must be 0 or 1, not {doctors}")
            }
            Self::DuplicatePlayer(who) => {
                write!(f, "player {:?} is listed more than once", who.as_str())
            }
            Self::PlayerIsModerator(who) => {
                write!(f, "player {:?} has the moderator's id", who.as_str())
            }
            Self::TimingNotPositive { field } => {
                write!(f, "{field} must be greater than zero")
            }
            Self::QuietExceedsLimit {
                session,
                quiet,
                limit,
            } => write!(
                f,
                "timing.{session}.quiet ({} s) must not be greater than timing.{session}.limit \
                 ({} s)",
                quiet.as_secs_f64(),
                limit.as_secs_f64()
            ),
            Self::DayCapTooSmall => f.write_str("timing.day_cap must be at least 1"),
            Self::ReservedPlayer(who) => write!(
                f,
                "player {:?} has a name reserved for the game's own random streams; the \
                 reserved names are {}",
                who.as_str(),
                seed::RESERVED
                    .iter()
                    .map(|label| format!("{label:?}"))
                    .collect::<Vec<_>>()
                    .join(" and ")
            ),
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
            seers: _,
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
        if doctors > 1 {
            return Err(ConfigError::TooManyDoctors(doctors));
        }
        let mut seen = BTreeSet::new();
        for player in &self.players {
            if !seen.insert(player) {
                return Err(ConfigError::DuplicatePlayer(player.clone()));
            }
            if *player == self.moderator {
                return Err(ConfigError::PlayerIsModerator(player.clone()));
            }
            if seed::RESERVED.contains(&player.as_str()) {
                return Err(ConfigError::ReservedPlayer(player.clone()));
            }
        }
        self.timing.validate()?;
        Ok(())
    }

    /// The effective configuration of a run played from this one: the same
    /// configuration as TOML, without its `trajectory` field and with its
    /// `[timing]` table resolved.
    ///
    /// It is a valid configuration file in the schema [`load`] reads, and
    /// reads back as this configuration with no trajectory. That is what
    /// makes reproducing a run `werewolf play <trajectory>.toml --trajectory
    /// <elsewhere>`: the trajectory is omitted so that replaying the file
    /// cannot truncate the very trajectory it describes, and a reproduction
    /// names its own output.
    ///
    /// `day_cap` is written as the number the run actually played to, rather
    /// than left out to be defaulted again. It defaults from the number of
    /// players, so a file whose player list was overridden would otherwise
    /// resolve it differently on the way back in, and the effective
    /// configuration has to describe the run that happened.
    ///
    /// # Panics
    ///
    /// Never for a configuration [`load`] accepted; a configuration is plain
    /// data that TOML can always represent.
    #[must_use]
    pub fn effective(&self) -> String {
        let effective = Self {
            trajectory: None,
            timing: self.timing.resolved(self.players.len()),
            ..self.clone()
        };
        toml::to_string(&effective).expect("a configuration is representable as TOML")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TempDir;

    /// The example configuration, as committed at the repository root.
    const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/werewolf.toml");

    /// A configuration with every field present.
    const FULL: &str = r#"
        seed = 20260918
        players = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"]
        trajectory = "werewolf.jsonl"
        moderator = "narrator"

        [roles]
        werewolves = 2
        seers = 1
        doctors = 1

        [timing]
        day_cap = 5

        [timing.pack]
        quiet = 0.5
        limit = 9.0

        [timing.seer]
        quiet = 0.25
        limit = 8.0

        [timing.doctor]
        quiet = 0.125
        limit = 7.0

        [timing.day]
        limit = 6.0
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
                moderator: AgentId::new("narrator"),
                timing: Timing {
                    day_cap: Some(5),
                    pack: NightTiming {
                        quiet: Duration::from_millis(500),
                        limit: Duration::from_secs(9),
                    },
                    seer: NightTiming {
                        quiet: Duration::from_millis(250),
                        limit: Duration::from_secs(8),
                    },
                    doctor: NightTiming {
                        quiet: Duration::from_millis(125),
                        limit: Duration::from_secs(7),
                    },
                    day: DayTiming {
                        limit: Duration::from_secs(6),
                    },
                },
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
                moderator: AgentId::new(DEFAULT_MODERATOR),
                timing: Timing::default(),
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
    fn a_game_may_deal_more_than_one_seer() {
        // Each seer is asked to investigate and told what it found, and the
        // transcript keeps a finding per seer. Nothing in the rules wants
        // there to be only one, so nothing here says there is.
        let mut config = valid();
        config.roles.seers = 2;
        config.validate().unwrap();
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
    fn empty_player_id_does_not_parse() {
        let text = FULL.replace("\"dave\"", "\"\"");
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::Parse(_)), "{error:?}");
        assert!(error.to_string().contains("non-empty agent id"), "{error}");
    }

    #[test]
    fn empty_moderator_id_does_not_parse() {
        let text = FULL.replace("\"narrator\"", "\"\"");
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::Parse(_)), "{error:?}");
        assert!(error.to_string().contains("non-empty agent id"), "{error}");
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
        assert_eq!(error.to_string(), "player \"bob\" is listed more than once");
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
        assert_eq!(
            error.to_string(),
            "player \"narrator\" has the moderator's id"
        );
    }

    #[test]
    fn a_player_may_not_be_named_for_a_reserved_seed_stream() {
        // Such a player's policy would draw from the deal's generator, or
        // the moderator's, and the streams would not be independent.
        for reserved in seed::RESERVED {
            let mut config = valid();
            config.players[2] = AgentId::new(reserved);
            let error = config.validate().unwrap_err();
            assert!(
                matches!(&error, ConfigError::ReservedPlayer(who) if who.as_str() == reserved),
                "{error:?}"
            );
            let text = error.to_string();
            assert!(text.contains(&format!("{reserved:?}")), "{text}");
            assert!(
                text.contains("\"assignment\" and \"moderator:ties\""),
                "{text}"
            );
        }
        // The moderator has no policy stream, so its name is free.
        let mut config = valid();
        config.moderator = AgentId::new(seed::TIES);
        config.validate().unwrap();
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
                timing: config.timing.resolved(config.players.len()),
                ..config.clone()
            }
        );
    }

    #[test]
    fn the_effective_config_writes_the_defaults_out_in_full() {
        // What was defaulted on the way in is explicit on the way out, so the
        // file says what ran even if a default changes later.
        let text = Config::parse(MINIMAL).unwrap().effective();
        assert!(
            text.contains(&format!("moderator = \"{DEFAULT_MODERATOR}\"")),
            "{text}"
        );
        assert!(text.contains("seers = 0"), "{text}");
        // Including the day cap, which defaults from the number of players
        // and so has to be pinned to the number this run played to.
        assert!(text.contains("day_cap = 3"), "{text}");
        let minimal = Config::parse(MINIMAL).unwrap();
        assert_eq!(
            Config::parse(&text).unwrap(),
            Config {
                timing: minimal.timing.resolved(minimal.players.len()),
                ..minimal.clone()
            }
        );
    }

    #[test]
    fn the_effective_config_is_written_beside_the_trajectory_and_loads() {
        let dir = TempDir::new();
        let trajectory = dir.join("werewolf.jsonl");
        let written = write_effective(&valid(), &trajectory).unwrap();
        assert_eq!(written, effective_path(&trajectory));
        let config = valid();
        assert_eq!(
            load(&written).unwrap(),
            Config {
                trajectory: None,
                timing: config.timing.resolved(config.players.len()),
                ..config.clone()
            }
        );
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
    fn a_config_without_timing_gets_the_default_clocks() {
        let config = Config::parse(MINIMAL).unwrap();
        assert_eq!(config.timing, Timing::default());
        for session in [config.timing.pack, config.timing.seer, config.timing.doctor] {
            assert_eq!(session.quiet, Duration::from_secs(1));
            assert_eq!(session.limit, Duration::from_secs(20));
        }
        assert_eq!(config.timing.day.limit, Duration::from_secs(60));
        // Unset, the day cap is the number of players: the smallest cap that
        // never ends a game the rules would have ended anyway.
        assert_eq!(config.timing.day_cap, None);
        assert_eq!(config.timing.day_cap(config.players.len()), 3);
        assert_eq!(config.timing.day_cap(12), 12);
    }

    #[test]
    fn a_day_cap_that_is_set_is_kept() {
        let config = valid();
        assert_eq!(config.timing.day_cap, Some(5));
        // Set, it is what it says whatever the game's size.
        assert_eq!(config.timing.day_cap(config.players.len()), 5);
        assert_eq!(config.timing.day_cap(99), 5);
    }

    #[test]
    fn a_duration_of_zero_is_rejected_by_the_field_that_holds_it() {
        // Every duration, named one at a time, so that each is checked and
        // each error says which field it was.
        /// A field of a [`Timing`], named as the file writes it, and the
        /// way to set it.
        type Case = (&'static str, fn(&mut Timing));

        let cases: [Case; 7] = [
            ("timing.pack.limit", |t| t.pack.limit = Duration::ZERO),
            ("timing.pack.quiet", |t| t.pack.quiet = Duration::ZERO),
            ("timing.seer.limit", |t| t.seer.limit = Duration::ZERO),
            ("timing.seer.quiet", |t| t.seer.quiet = Duration::ZERO),
            ("timing.doctor.limit", |t| t.doctor.limit = Duration::ZERO),
            ("timing.doctor.quiet", |t| t.doctor.quiet = Duration::ZERO),
            ("timing.day.limit", |t| t.day.limit = Duration::ZERO),
        ];
        for (field, break_it) in cases {
            let mut config = valid();
            break_it(&mut config.timing);
            let error = config.validate().unwrap_err();
            assert!(
                matches!(&error, ConfigError::TimingNotPositive { field: f } if f == field),
                "{field}: {error:?}"
            );
            assert_eq!(
                error.to_string(),
                format!("{field} must be greater than zero")
            );
        }
    }

    #[test]
    fn a_quiet_period_may_not_outlast_its_limit() {
        // A session whose quiet period is longer than its limit could never
        // close on quiet, so the limit would be the only way it ever ended.
        for session in ["pack", "seer", "doctor"] {
            let mut config = valid();
            let clock = match session {
                "pack" => &mut config.timing.pack,
                "seer" => &mut config.timing.seer,
                _ => &mut config.timing.doctor,
            };
            clock.quiet = Duration::from_secs(2);
            clock.limit = Duration::from_secs(1);
            let error = config.validate().unwrap_err();
            assert!(
                matches!(&error, ConfigError::QuietExceedsLimit { session: s, .. } if *s == session),
                "{session}: {error:?}"
            );
            let text = error.to_string();
            assert!(
                text.contains(&format!("timing.{session}.quiet (2 s)")),
                "{text}"
            );
            assert!(
                text.contains(&format!("timing.{session}.limit (1 s)")),
                "{text}"
            );
        }
    }

    #[test]
    fn a_quiet_period_equal_to_its_limit_is_allowed() {
        // Not greater than, so equal is fine: the session closes on quiet at
        // the same instant its limit would have closed it.
        let mut config = valid();
        config.timing.pack.quiet = config.timing.pack.limit;
        config.validate().unwrap();
    }

    #[test]
    fn a_day_cap_of_zero_leaves_no_game_to_play() {
        let mut config = valid();
        config.timing.day_cap = Some(0);
        let error = config.validate().unwrap_err();
        assert!(matches!(error, ConfigError::DayCapTooSmall), "{error:?}");
        assert_eq!(error.to_string(), "timing.day_cap must be at least 1");
        // One is the smallest cap there is.
        config.timing.day_cap = Some(1);
        config.validate().unwrap();
    }

    #[test]
    fn a_duration_that_is_not_a_duration_does_not_parse() {
        // Rejected at parse rather than in `validate`, because there is no
        // `Duration` to hold a negative or an infinity in the first place.
        for bad in ["-1.0", "nan", "inf"] {
            let text = format!(
                "{MINIMAL}
[timing.pack]
quiet = {bad}
limit = 1.0
"
            );
            let error = Config::parse(&text).unwrap_err();
            assert!(matches!(error, ConfigError::Parse(_)), "{bad}: {error:?}");
        }
    }

    #[test]
    fn a_quiet_period_under_the_day_is_an_unknown_field() {
        // The day has no quiet period, and `deny_unknown_fields` is what
        // says so rather than silently taking a default.
        let text = format!(
            "{MINIMAL}
[timing.day]
quiet = 1.0
limit = 2.0
"
        );
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::Parse(_)), "{error:?}");
        assert!(error.to_string().contains("quiet"), "{error}");
    }

    #[test]
    fn the_effective_timing_round_trips() {
        // Written, parsed back, equal: what a run is reproduced from.
        let config = valid();
        let text = config.effective();
        let read = Config::parse(&text).unwrap();
        assert_eq!(read.timing, config.timing.resolved(config.players.len()));
        // And again, to show the resolved form is a fixed point.
        assert_eq!(
            Config::parse(&read.effective()).unwrap().timing,
            read.timing
        );
    }

    #[test]
    fn validation_happens_in_parse() {
        let text = FULL.replace("werewolves = 2", "werewolves = 0");
        let error = Config::parse(&text).unwrap_err();
        assert!(matches!(error, ConfigError::NoWerewolves), "{error:?}");
    }
}
