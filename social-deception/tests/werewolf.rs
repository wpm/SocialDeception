//! Werewolf, end to end.
//!
//! Each test plays real episodes, with the trajectory going to a file, reads
//! the file back, and checks it against two sets of invariants: the ones in
//! [`support`] that every trajectory satisfies whatever the environment,
//! unchanged, and Werewolf's own in [`support::werewolf`]. The first set is
//! run on every trajectory produced here; if it ever needed changing to
//! accommodate Werewolf, Werewolf would be doing something the runtime does
//! not intend.
//!
//! The fixture trajectory under `tests/fixtures`, which the transcript
//! reader and the `werewolf replay` command are tested against, goes
//! through both sets too, so that it cannot rot into something the runtime
//! would never have written.
//!
//! # Determinism, precisely
//!
//! For a fixed configuration and seed, the *logical transcript* (the role
//! assignment, every request, response, tally, elimination and the outcome)
//! is identical on every run. The *wall-clock timestamps* and the
//! *interleaving of different agents' records* in the trajectory are not,
//! and cannot be, because the agents are threads. So the determinism tests
//! compare [`Transcript`]s, which are the trajectory with everything
//! non-reproducible projected out, and never the files.

mod support;

use std::fs;
use std::path::{Path, PathBuf};
use std::process::{self, Command};
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::Value;
use social_deception::AgentId;
use social_deception::werewolf::config::{DEFAULT_MAX_ROUNDS, DEFAULT_MODERATOR};
use social_deception::werewolf::{
    self, Config, Faction, Message, Narration, RoleCounts, Transcript, config,
};

/// A seven-player game played to a village win, with its effective config
/// and its expected rendering beside it.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/werewolf.jsonl");

/// The example configuration at the repository root, the one the README
/// says to play.
const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/werewolf.toml");

/// The `werewolf` binary, as built for these tests.
const WEREWOLF: &str = env!("CARGO_BIN_EXE_werewolf");

/// The seed the fixed-seed tests play from.
const SEED: u64 = 20_260_918;

/// How many seeds a search for a particular kind of game tries. With a
/// uniform policy over seven players the events searched for turn up in a
/// good fraction of games, so this is generous.
const SEEDS: u64 = 40;

/// A trajectory file in the temp dir, removed when this is dropped along
/// with any effective config written beside it, so that a failing test
/// leaves nothing behind.
struct TempFile(PathBuf);

impl TempFile {
    fn new() -> Self {
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        Self(std::env::temp_dir().join(format!(
            "social-deception-werewolf-{}-{}.jsonl",
            process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        )))
    }
}

impl Drop for TempFile {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.0);
        let _ = fs::remove_file(config::effective_path(&self.0));
    }
}

/// A validated configuration for `players` with the given special roles,
/// played from `seed`, writing no trajectory.
fn config(players: &[&str], werewolves: usize, seers: usize, doctors: usize, seed: u64) -> Config {
    let config = Config {
        seed,
        players: players.iter().map(|who| AgentId::new(*who)).collect(),
        roles: RoleCounts {
            werewolves,
            seers,
            doctors,
        },
        trajectory: None,
        max_rounds: DEFAULT_MAX_ROUNDS,
        moderator: AgentId::new(DEFAULT_MODERATOR),
    };
    config.validate().unwrap();
    config
}

/// The headline case: seven players, two werewolves, a seer and a doctor.
fn town(seed: u64) -> Config {
    config(
        &["alice", "bob", "carol", "dave", "erin", "frank", "grace"],
        2,
        1,
        1,
        seed,
    )
}

/// One episode, played to its trajectory and read back.
struct Played {
    /// The trajectory, one value per line.
    lines: Vec<Value>,
    /// The game the trajectory records.
    transcript: Transcript,
}

impl Played {
    /// Every narration the moderator sent, in the order it sent them.
    fn narrations(&self, config: &Config) -> Vec<Narration> {
        self.lines
            .iter()
            .filter(|line| support::agent(line) == config.moderator.as_str())
            .filter(|line| !line["sent"].is_null())
            .filter_map(
                |line| match serde_json::from_value(line["event"]["payload"].clone()) {
                    Ok(Message::Narration(narration)) => Some(narration),
                    _ => None,
                },
            )
            .collect()
    }
}

/// Reads the trajectory at `path` back as the game it records.
fn read(path: &Path, config: &Config) -> (Vec<Value>, Transcript) {
    let lines = support::parse(&fs::read(path).unwrap());
    let transcript = Transcript::read(&lines, &config.moderator).unwrap();
    (lines, transcript)
}

/// Runs one episode of `config` with the trajectory going to a temp file,
/// and returns the trajectory read back from disk with the game it records.
///
/// Before it returns, the trajectory is checked against the invariants in
/// [`support`], which know nothing about Werewolf, and then against
/// Werewolf's own in [`support::werewolf`], so that a bad trajectory fails
/// by the name of the invariant it breaks. The outcome the run reported on
/// its channel is checked against the one the moderator announced in world:
/// the announcement is the record of truth, and the channel must agree.
fn run(config: &Config) -> Played {
    let file = TempFile::new();
    let config = Config {
        trajectory: Some(file.0.clone()),
        ..config.clone()
    };
    let outcome = werewolf::run(&config).unwrap();
    let (lines, transcript) = read(&file.0, &config);
    support::check(&lines);
    support::werewolf::check(&lines, &config);
    assert_eq!(
        transcript.outcome, outcome,
        "the channel agrees with the announcement"
    );
    Played { lines, transcript }
}

/// Runs `werewolf` with `args` and returns what it printed, panicking with
/// its stderr if it failed.
fn werewolf(args: &[&str]) -> String {
    let output = Command::new(WEREWOLF).args(args).output().unwrap();
    assert!(
        output.status.success(),
        "werewolf {args:?} failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn the_fixture_is_a_trajectory_the_runtime_could_have_written() {
    let config = config::load(config::effective_path(Path::new(FIXTURE))).unwrap();
    let lines = support::parse(&fs::read(FIXTURE).unwrap());
    support::check(&lines);
    support::werewolf::check(&lines, &config);
}

#[test]
fn seven_players_with_two_werewolves_a_seer_and_a_doctor() {
    let played = run(&town(SEED));
    assert_eq!(played.transcript.assignment.len(), 7);
    assert!(played.transcript.outcome.winner.is_some());
}

#[test]
fn three_players_with_one_werewolf_is_the_smallest_game() {
    // The werewolf devours one of the other two on the first night, which
    // is parity, unless the game has a doctor to save them; either way no
    // game this small lasts long.
    let played = run(&config(&["alice", "bob", "carol"], 1, 0, 0, SEED));
    assert_eq!(played.transcript.outcome.winner, Some(Faction::Werewolves));
    assert_eq!(played.transcript.rounds.len(), 1);
}

#[test]
fn a_game_without_a_seer_or_a_doctor_or_either() {
    let players = ["alice", "bob", "carol", "dave", "erin", "frank"];
    for (seers, doctors) in [(0, 1), (1, 0), (0, 0)] {
        let config = config(&players, 2, seers, doctors, SEED);
        let played = run(&config);
        let investigations = played
            .transcript
            .rounds
            .iter()
            .filter(|round| round.night.investigation.is_some())
            .count();
        assert!(
            seers == 1 || investigations == 0,
            "with no seer nothing is investigated: {seers} seers, {doctors} doctors"
        );
    }
}

#[test]
fn a_game_that_reaches_the_round_cap_is_a_stalemate() {
    // Nine players and three werewolves cannot finish in one round: the
    // night takes at most one player and the day exactly one, so at least
    // one werewolf survives it and at least four others do, which is neither
    // side's win. So the cap ends the game with nobody winning, whatever the
    // seed.
    let mut config = config(
        &[
            "alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan",
        ],
        3,
        1,
        1,
        SEED,
    );
    config.max_rounds = 1;
    let played = run(&config);
    assert_eq!(played.transcript.outcome.winner, None);
    assert_eq!(played.transcript.rounds.len(), 1);
    assert!(played.transcript.rounds[0].day.is_some());
}

#[test]
fn the_doctor_saves_in_some_game() {
    // A save cannot be arranged by choosing the roles, so search the seeds
    // for a game with a night on which nobody died. Every game searched
    // goes through the full invariant suite on the way.
    let saved = (0..SEEDS).find(|&seed| {
        let config = town(seed);
        run(&config)
            .narrations(&config)
            .iter()
            .any(|narration| matches!(narration, Narration::NoDeath { .. }))
    });
    assert!(
        saved.is_some(),
        "the doctor never saved anyone in {SEEDS} games; the doctor is not working"
    );
}

#[test]
fn each_side_wins_some_game() {
    // Found by searching the seeds rather than by contrivance, so the test
    // exercises whatever the rules actually produce.
    let winners: Vec<Option<Faction>> = (0..SEEDS)
        .map(|seed| run(&town(seed)).transcript.outcome.winner)
        .collect();
    assert!(
        winners.contains(&Some(Faction::Village)),
        "the village never won in {SEEDS} games: {winners:?}"
    );
    assert!(
        winners.contains(&Some(Faction::Werewolves)),
        "the werewolves never won in {SEEDS} games: {winners:?}"
    );
}

#[test]
fn the_same_seed_plays_the_same_game() {
    let config = town(SEED);
    let first = run(&config);
    let second = run(&config);
    assert_eq!(first.transcript, second.transcript);
    // The two trajectory *files* are deliberately not compared. They record
    // wall-clock timestamps, and the records of different agents' threads
    // interleave however the scheduler ran them, so the files of the same
    // game differ from run to run. Tightening this test to compare them
    // would assert something the runtime cannot honour, and is not meant
    // to. The transcript is the claim.
}

#[test]
fn different_seeds_play_different_games() {
    // A `seed_for` that ignored its input would pass every other test here.
    let transcripts: Vec<Transcript> = (1..=4).map(|seed| run(&town(seed)).transcript).collect();
    assert!(
        transcripts
            .iter()
            .any(|transcript| *transcript != transcripts[0]),
        "four seeds played the same game"
    );
}

#[test]
fn a_dozen_runs_play_the_same_game() {
    // A determinism bug that depends on thread scheduling will not show up
    // in two runs.
    let config = town(SEED);
    let first = run(&config).transcript;
    for i in 1..12 {
        assert_eq!(
            run(&config).transcript,
            first,
            "run {i} played a different game"
        );
    }
}

#[test]
fn a_run_is_reproduced_from_its_artifacts() {
    // Play the example with a seed override, so the file on disk is not the
    // whole recipe, then play the effective config the run wrote beside its
    // trajectory. The second game must be the first: a run is reproducible
    // from what it left behind, whatever flags produced it.
    let original = TempFile::new();
    let rerun = TempFile::new();
    let seed = SEED.to_string();
    let played = werewolf(&[
        "play",
        EXAMPLE,
        "--seed",
        &seed,
        "--trajectory",
        original.0.to_str().unwrap(),
    ]);

    let effective_path = config::effective_path(&original.0);
    let effective = config::load(&effective_path).unwrap();
    assert_eq!(
        effective.seed, SEED,
        "the effective config records the override"
    );
    assert_eq!(effective.trajectory, None, "and names no trajectory");

    let (lines, transcript) = read(&original.0, &effective);
    support::check(&lines);
    support::werewolf::check(&lines, &effective);
    let winner = match transcript.outcome.winner {
        Some(winner) => winner.to_string(),
        None => "nobody, a stalemate at the round cap".to_owned(),
    };
    assert!(
        played.starts_with(&format!("seed: {SEED}\nwinner: {winner}\n")),
        "{played}"
    );
    assert!(played.contains(&format!("effective config: {}\n", effective_path.display())));

    let replayed = werewolf(&["replay", original.0.to_str().unwrap()]);
    assert!(replayed.starts_with(&format!("Werewolf \u{2014} seed {SEED}, 7 players")));
    assert!(replayed.ends_with(&transcript.to_string()), "{replayed}");

    werewolf(&[
        "play",
        effective_path.to_str().unwrap(),
        "--trajectory",
        rerun.0.to_str().unwrap(),
    ]);
    let (lines, reproduced) = read(&rerun.0, &effective);
    support::check(&lines);
    support::werewolf::check(&lines, &effective);
    assert_eq!(reproduced, transcript);
}
