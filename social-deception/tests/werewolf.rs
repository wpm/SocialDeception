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
use std::path::Path;
use std::process::{Command, Stdio};

use social_deception::AgentId;
use social_deception::werewolf::config::{DEFAULT_MAX_ROUNDS, DEFAULT_MODERATOR};
use social_deception::werewolf::{self, Config, Faction, RoleCounts, Transcript, config};
use support::TempDir;

/// A seven-player game played to a village win, with its effective config
/// and its expected rendering beside it.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/werewolf.jsonl");

/// The example configuration at the repository root, the one the README's
/// "Playing Werewolf" section says to play.
const EXAMPLE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../examples/werewolf.toml");

/// The `werewolf` binary, as built for these tests.
const WEREWOLF: &str = env!("CARGO_BIN_EXE_werewolf");

/// The seed the fixed-seed tests play from.
const SEED: u64 = 20_260_918;

/// How many seeds a search for particular kinds of game tries at most.
/// With a uniform policy over seven players the events searched for turn
/// up in a good fraction of games, so this is generous.
const SEEDS: u64 = 40;

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

/// Reads the trajectory at `path` back as the game it records, once it has
/// passed both sets of invariants: the ones in [`support`], which know
/// nothing about Werewolf, and then Werewolf's own in
/// [`support::werewolf`], so that a bad trajectory fails by the name of the
/// invariant it breaks.
fn read(path: &Path, config: &Config) -> Transcript {
    let lines = support::parse(&fs::read(path).unwrap());
    support::check(&lines);
    support::werewolf::check(&lines, config);
    Transcript::read(&lines, &config.moderator).unwrap()
}

/// Runs one episode of `config` with the trajectory going to a temp file,
/// and returns the game the trajectory records, checked as [`read`] checks
/// it. The outcome the run reported on its channel is checked against the
/// one the moderator announced in world: the announcement is the record of
/// truth, and the channel must agree.
fn run(config: &Config) -> Transcript {
    let dir = TempDir::new();
    let file = dir.join("werewolf.jsonl");
    let config = Config {
        trajectory: Some(file.clone()),
        ..config.clone()
    };
    let outcome = werewolf::run(&config).unwrap();
    let transcript = read(&file, &config);
    assert_eq!(
        transcript.outcome, outcome,
        "the channel agrees with the announcement"
    );
    transcript
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
    read(Path::new(FIXTURE), &config);
}

#[test]
fn seven_players_with_two_werewolves_a_seer_and_a_doctor() {
    let transcript = run(&town(SEED));
    assert!(transcript.outcome.winner.is_some());
}

#[test]
fn three_players_with_one_werewolf_is_the_smallest_game() {
    // The werewolf devours one of the other two on the first night, and
    // with no doctor to save them that is parity, whatever the seed.
    let transcript = run(&config(&["alice", "bob", "carol"], 1, 0, 0, SEED));
    assert_eq!(transcript.outcome.winner, Some(Faction::Werewolves));
    assert_eq!(transcript.rounds.len(), 1);
}

#[test]
fn a_game_without_a_seer_or_a_doctor_or_either() {
    // Nothing to assert beyond the invariants: with no seer, no request to
    // investigate may be asked, and with no doctor no night may be quiet,
    // and the suite checks both.
    let players = ["alice", "bob", "carol", "dave", "erin", "frank"];
    for (seers, doctors) in [(0, 1), (1, 0), (0, 0)] {
        run(&config(&players, 2, seers, doctors, SEED));
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
    let transcript = run(&config);
    assert_eq!(transcript.outcome.winner, None);
    assert_eq!(transcript.rounds.len(), 1);
    assert!(transcript.rounds[0].day.is_some());
}

#[test]
fn the_seeds_hold_a_save_and_a_win_for_each_side() {
    // None of these can be arranged by choosing the roles, so search the
    // seeds for them rather than contrive them, and stop once all three
    // have turned up. Every game searched goes through the full invariant
    // suite on the way, which is where "the doctor is working" is actually
    // asserted: a quiet night is one on which it protected the pack's
    // choice. Here it need only happen.
    let mut saved = false;
    let mut winners = Vec::new();
    for seed in 0..SEEDS {
        let transcript = run(&town(seed));
        saved |= transcript
            .rounds
            .iter()
            .any(|round| round.night.eliminated.is_none());
        winners.push(transcript.outcome.winner);
        if saved
            && winners.contains(&Some(Faction::Village))
            && winners.contains(&Some(Faction::Werewolves))
        {
            return;
        }
    }
    assert!(
        saved,
        "the doctor never saved anyone in {SEEDS} games; the doctor is not working"
    );
    panic!("one side never won in {SEEDS} games: {winners:?}");
}

#[test]
fn the_same_seed_plays_the_same_game() {
    let config = town(SEED);
    assert_eq!(run(&config), run(&config));
    // The two trajectory *files* are deliberately not compared. They record
    // wall-clock timestamps, and the records of different agents' threads
    // interleave however the scheduler ran them, so the files of the same
    // game differ from run to run. Tightening this test to compare them
    // would assert something the runtime cannot honor, and is not meant
    // to. The transcript is the claim.
}

#[test]
fn different_seeds_play_different_games() {
    // A `seed_for` that ignored its input would pass every other test here.
    let transcripts: Vec<Transcript> = (1..=4).map(|seed| run(&town(seed))).collect();
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
    let first = run(&config);
    for i in 1..12 {
        assert_eq!(run(&config), first, "run {i} played a different game");
    }
}

#[test]
fn a_run_is_reproduced_from_its_artifacts() {
    // Play the example with a seed override, so the file on disk is not the
    // whole recipe, then play the effective config the run wrote beside its
    // trajectory. The second game must be the first: a run is reproducible
    // from what it left behind, whatever flags produced it.
    let dir = TempDir::new();
    let original = dir.join("original.jsonl");
    let effective_path = config::effective_path(&original);
    let rerun = dir.join("rerun.jsonl");
    let seed = SEED.to_string();
    let played = werewolf(&[
        "play",
        EXAMPLE,
        "--seed",
        &seed,
        "--trajectory",
        original.to_str().unwrap(),
    ]);

    let effective = config::load(&effective_path).unwrap();
    assert_eq!(
        effective.seed, SEED,
        "the effective config records the override"
    );
    assert_eq!(effective.trajectory, None, "and names no trajectory");

    let transcript = read(&original, &effective);
    let winner = match transcript.outcome.winner {
        Some(winner) => winner.to_string(),
        None => "nobody, a stalemate at the round cap".to_owned(),
    };
    assert!(
        played.starts_with(&format!("seed: {SEED}\nwinner: {winner}\n")),
        "{played}"
    );
    assert!(played.contains(&format!("effective config: {}\n", effective_path.display())));

    let replayed = werewolf(&["replay", original.to_str().unwrap()]);
    assert!(replayed.starts_with(&format!(
        "Werewolf \u{2014} seed {SEED}, {} players",
        effective.players.len()
    )));
    assert!(replayed.ends_with(&transcript.to_string()), "{replayed}");

    werewolf(&[
        "play",
        effective_path.to_str().unwrap(),
        "--trajectory",
        rerun.to_str().unwrap(),
    ]);
    assert_eq!(read(&rerun, &effective), transcript);
}

#[test]
fn a_reader_that_stops_early_is_not_an_error() {
    // `werewolf replay run.jsonl | head` closes the pipe before the
    // transcript is fully written. That is the reader's business, and the
    // command exits quietly rather than panicking on the broken pipe. The
    // pipe is closed before the child has had time to write, in practice
    // every time; when it has not, the test proves nothing and passes.
    let mut child = Command::new(WEREWOLF)
        .args(["replay", FIXTURE])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stderr), "");
}
