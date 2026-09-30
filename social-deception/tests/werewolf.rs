//! Werewolf, end to end.
//!
//! Each test plays real episodes, with the log going to a file, reads
//! the file back, and checks it against two sets of invariants: the ones in
//! [`support::actor`] that every log of an
//! actor runtime episode satisfies whatever the
//! game, unchanged, and Werewolf's own in [`support::werewolf`]. The first
//! set is run on every log produced here; if it ever needed changing to
//! accommodate Werewolf, Werewolf would be doing something the runtime does
//! not intend.
//!
//! The fixture log under `tests/fixtures`, which the transcript
//! reader and the `werewolf replay` command are tested against, goes
//! through both sets too, so that it cannot rot into something the runtime
//! would never have written.
//!
//! # Determinism, precisely
//!
//! For a fixed configuration and seed, the *logical transcript* (the role
//! assignment, every selection, elimination and the outcome)
//! is identical on every run. The *wall-clock timestamps* and the
//! *interleaving of different agents' records* in the log are not,
//! and cannot be, because the agents are threads. So the determinism tests
//! compare [`Transcript`]s, which are the log with everything
//! non-reproducible projected out, and never the files.

mod support;

use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};

use std::time::Duration;

use social_deception::ActorId;
use social_deception::werewolf::config::DEFAULT_MODERATOR;
use social_deception::werewolf::config::{DayTiming, NightTiming, Timing};
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
///
/// A village win is the scarce one. Under ADR-0011 a day closes on a
/// majority of the living, and seven uniform players seldom put four on
/// one target, so most days run out and the pack wins by attrition; the
/// first village win is at seed 41. This is that with room to spare, and
/// it is why the number is no longer small.
const SEEDS: u64 = 120;

/// How many seeds the search plays at once.
///
/// A game is seventeen threads that spend nearly all of their time
/// blocked — on a session's limit, on a quiet period, on an empty inbox —
/// so seeds overlap almost for free, and the search is bounded by wall
/// clock rather than by cores. What bounds the batch is the other end: the
/// `FAST` clocks hold only while every player's one selection is scheduled
/// inside its session, and enough concurrent games will starve one of them,
/// at which point a game stops being the game its seed names — silently,
/// for the reason [`FAST`] gives, since a starved player is
/// indistinguishable from one that abstained. Widening this batch spends
/// the same margin those limits are set for.
///
/// Eight is chosen for margin rather than for speed. Twenty-four seeds
/// played this way were compared against the same seeds played one at a
/// time, and the verdicts still matched at a batch of twenty-four — some
/// four hundred threads on a twelve-core machine. Three times the headroom
/// is what is left to the slower and smaller machines this also runs on,
/// and to whatever else `cargo test` is running beside it.
const BATCH: usize = 8;

/// A validated configuration for `players` with the given special roles,
/// played from `seed`, writing no log.
/// Timing fast enough that a test does not spend real time waiting on a
/// session's clock, and slow enough that a random player's one selection
/// always lands inside it.
///
/// A game of random players selects once and never changes its mind, so a
/// night session closes a quiet period after its last member's only
/// selection, and a day runs to its limit unless a majority falls out of the
/// deal. The outcome is reproducible only while every one of those selections
/// arrives before its session closes (ADR-0011), which is a claim about
/// thread latency: the limits have to exceed however long the slowest
/// player takes to be scheduled and answer.
///
/// The two clocks are set for different reasons. A **hard limit** has to
/// outlast the slowest player's one selection, or a selection misses its
/// session and the game genuinely differs from run to run; at 50 ms these
/// tests passed alone and failed a few times in ten with several suites at
/// once, because seven agent threads on a loaded machine can outrun a margin
/// that small. A **quiet period** costs real time on every night, since a
/// night closes one quiet period after its members settle, so it stays short.
/// Only the hard limit is exposed this way: a quiet period is measured from
/// the moment every member has selected, so it cannot close a session on
/// somebody who has not been heard from.
///
/// **Nothing catches it when a limit is too short.** A player whose thread
/// was not scheduled in time is simply absent from its session's selections,
/// and absent is how a member abstains (ADR-0011) — the game cannot tell a
/// player that chose nowhere from a player the machine never got to. So the
/// session closes on a smaller field, a plurality falls differently, and the
/// seed goes on to play a *different* game that breaks no invariant and
/// fails no assertion. That is a fact about the computer wearing the costume
/// of a fact about the game, and it is the reason these limits are set with
/// margin rather than trimmed until the tests are fast: the failure they
/// guard against is silent, and would be read as the game's own behavior.
///
/// The day's limit is the expensive one — a random day rarely reaches a
/// majority, so most days run it out — but it is also the one a slow
/// selection matters least for, because a day closes on a majority of the
/// living and a selection that misses cannot have made one. It is kept below
/// the night's for that reason.
const FAST: Timing = Timing {
    day_cap: None,
    pack: FAST_NIGHT,
    seer: FAST_NIGHT,
    doctor: FAST_NIGHT,
    day: DayTiming {
        limit: Duration::from_millis(80),
    },
};

const FAST_NIGHT: NightTiming = NightTiming {
    quiet: Duration::from_millis(10),
    limit: Duration::from_millis(400),
};

fn config(players: &[&str], werewolves: usize, seers: usize, doctors: usize, seed: u64) -> Config {
    let config = Config {
        seed,
        players: players.iter().map(|who| ActorId::new(*who)).collect(),
        roles: RoleCounts {
            werewolves,
            seers,
            doctors,
        },
        trajectory: None,
        moderator: ActorId::new(DEFAULT_MODERATOR),
        timing: FAST,
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

/// Reads the log at `path` back as the game it records, once it has
/// passed both sets of invariants: the ones in [`support`], which know
/// nothing about Werewolf, and then Werewolf's own in
/// [`support::werewolf`], so that a bad log fails by the name of the
/// invariant it breaks.
fn read(path: &Path, config: &Config) -> Transcript {
    let lines = support::parse(&fs::read(path).unwrap());
    support::actor::check(&lines);
    support::werewolf::check(&lines, config);
    Transcript::read(&lines, &config.moderator).unwrap()
}

/// Runs one episode of `config` with the log going to a temp file,
/// and returns the game the log records, checked as [`read`] checks
/// it. The outcome the run reported on its channel is checked against the
/// one the moderator announced in world: the announcement is the record of
/// truth, and the channel must agree.
///
/// Every game played here is a game of random players on the [`FAST`]
/// clocks, so it is also held to
/// [`check_everybody_was_heard`](support::werewolf::check_everybody_was_heard):
/// no night of it closed on a player whose thread was not scheduled in
/// time. That is a claim about this machine rather than about Werewolf,
/// which is why it is asked for here and not inside
/// [`support::werewolf::check`] — the fixture goes through `read` too, and
/// a log is not required to have heard from everybody.
fn run(config: &Config) -> Transcript {
    let dir = TempDir::new();
    let file = dir.join("werewolf.jsonl");
    let config = Config {
        trajectory: Some(file.clone()),
        ..config.clone()
    };
    let outcome = werewolf::run(&config, None).unwrap();
    let lines = support::parse(&fs::read(&file).unwrap());
    support::werewolf::check_everybody_was_heard(&lines, &config);
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
        "werewolf {args:?} failed with {:?}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn the_fixture_is_a_log_the_runtime_could_have_written() {
    let config = config::load(config::effective_path(Path::new(FIXTURE))).unwrap();
    read(Path::new(FIXTURE), &config);
}

#[test]
fn seven_players_with_two_werewolves_a_seer_and_a_doctor() {
    // Nothing to assert beyond the invariants: reaching an `Outcome` at
    // all is reaching a winner, now that a game cannot end without one,
    // and `run` checks the announcement against the channel and puts the
    // log through the full suite.
    run(&town(SEED));
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
fn the_seeds_hold_a_save_and_a_win_for_each_side() {
    // None of these can be arranged by choosing the roles, so search the
    // seeds for them rather than contrive them, and stop once all three
    // have turned up. A stalemate is not among them: at the default cap
    // of one day per player a seven-player random game always resolves
    // first, which 400 seeds confirm. Where the cap does bite is a unit
    // test of its own,
    // `a_random_game_stalemates_only_when_the_cap_is_tight`.
    //
    // Every game searched goes through the full invariant suite on the
    // way, which is where "the doctor is working" is actually asserted: a
    // quiet night is one on which it protected the pack's choice. Here it
    // need only happen.
    //
    // A game here is almost all waiting — on a day's limit, on a night's
    // quiet period — so a seed costs wall clock rather than a core, and
    // the search plays `BATCH` seeds at once. The batch is what keeps the
    // early stop: the search still gives up as soon as a batch has
    // completed the set, having played at most `BATCH - 1` seeds it did
    // not need.
    let mut saved = false;
    let mut winners = Vec::new();
    for batch in (0..SEEDS).step_by(BATCH) {
        let seeds = batch..SEEDS.min(batch + BATCH as u64);
        let transcripts: Vec<Transcript> = std::thread::scope(|scope| {
            let played: Vec<_> = seeds
                .map(|seed| scope.spawn(move || run(&town(seed))))
                .collect();
            played
                .into_iter()
                .map(|game| game.join().expect("a game played to its end"))
                .collect()
        });
        for transcript in transcripts {
            saved |= transcript
                .rounds
                .iter()
                .any(|round| round.night.eliminated.is_none());
            winners.push(transcript.outcome.winner);
        }
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
    assert_eq!(run(&config).verdicts(), run(&config).verdicts());
    // What is compared is the *verdicts*, not the whole transcript, and
    // certainly not the log files. Under ADR-0011 a phase is a
    // timed session: every death, every finding and the winner are the
    // same on every run of a seed, while the order selections arrived in, and
    // which late ones landed before a session closed, are facts about
    // thread scheduling. The files differ for that reason and for their
    // wall-clock stamps. Comparing more than the verdicts would assert
    // something the runtime does not promise.
}

#[test]
fn different_seeds_play_different_games() {
    // A `seed_for` that ignored its input would pass every other test here.
    let verdicts: Vec<_> = (1..=4).map(|seed| run(&town(seed)).verdicts()).collect();
    assert!(
        verdicts.iter().any(|verdict| *verdict != verdicts[0]),
        "four seeds played the same game"
    );
}

#[test]
fn a_dozen_runs_play_the_same_game() {
    // A determinism bug that depends on thread scheduling will not show up
    // in two runs.
    let config = town(SEED);
    let first = run(&config).verdicts();
    for i in 1..12 {
        assert_eq!(
            run(&config).verdicts(),
            first,
            "run {i} played a different game"
        );
    }
}

#[test]
fn a_run_is_reproduced_from_its_artifacts() {
    // Play the example with a seed override, so the file on disk is not the
    // whole recipe, then play the effective config the run wrote beside its
    // log. The second game must be the first: a run is reproducible
    // from what it left behind, whatever flags produced it.
    let dir = TempDir::new();
    let original = dir.join("original.jsonl");
    let effective_path = config::effective_path(&original);
    let rerun = dir.join("rerun.jsonl");
    let seed = SEED.to_string();
    // `--quiet`, because what this test reads is the summary: the
    // narration above it is the subject of its own tests.
    let played = werewolf(&[
        "play",
        EXAMPLE,
        "--seed",
        &seed,
        "--trajectory",
        original.to_str().unwrap(),
        "--quiet",
    ]);

    let effective = config::load(&effective_path).unwrap();
    assert_eq!(
        effective.seed, SEED,
        "the effective config records the override"
    );
    assert_eq!(effective.trajectory, None, "and names no trajectory");

    let transcript = read(&original, &effective);
    let winner = transcript.outcome.winner;
    assert!(
        played.starts_with(&format!(
            "seed: {SEED}\nwinner: {}\n",
            winner.expect("a random game has a winner")
        )),
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
        "--quiet",
    ]);
    assert_eq!(read(&rerun, &effective).verdicts(), transcript.verdicts());
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

/// A five-player configuration file in `dir`, whose log is beside
/// it, for the tests that play a game through the binary.
fn playable(dir: &TempDir) -> (std::path::PathBuf, std::path::PathBuf) {
    let trajectory = dir.join("played.jsonl");
    let config = dir.join("game.toml");
    // Fast clocks, or the game would run on the defaults a
    // language-model game wants: twenty seconds a night session and sixty
    // for the day (ADR-0011).
    fs::write(
        &config,
        format!(
            "seed = 26\nplayers = [\"alice\", \"bob\", \"carol\", \"dave\", \"erin\", \"frank\", \
             \"grace\"]\ntrajectory = '{}'\n[roles]\nwerewolves = 2\nseers = 1\ndoctors = 1\n\
             [timing.pack]\nquiet = 0.01\nlimit = 0.4\n\
             [timing.seer]\nquiet = 0.01\nlimit = 0.4\n\
             [timing.doctor]\nquiet = 0.01\nlimit = 0.4\n\
             [timing.day]\nlimit = 0.08\n",
            trajectory.display()
        ),
    )
    .unwrap();
    (config, trajectory)
}

/// The summary `play` prints after a game: the five or six lines from the
/// seed to what was written.
fn summary(printed: &str) -> String {
    let at = printed.find("seed: ").expect("a summary");
    printed[at..].to_owned()
}

#[test]
fn play_narrates_the_game_above_its_summary() {
    let dir = TempDir::new();
    let (config, _) = playable(&dir);
    let printed = werewolf(&["play", config.to_str().unwrap()]);

    // The game, line by line, from the first phase to the outcome.
    assert!(
        printed
            .lines()
            .any(|line| line.contains("PhaseBegan(Night 1")),
        "{printed}"
    );
    assert!(
        printed.lines().any(|line| line.contains("Outcome(")),
        "{printed}"
    );
    // Then a blank line, then the summary and nothing after it.
    let ended = summary(&printed);
    assert!(printed.ends_with(&ended), "{printed}");
    assert!(
        printed[..printed.len() - ended.len()].ends_with("\n\n"),
        "{printed}"
    );
    assert!(ended.starts_with("seed: 26\nwinner: "), "{ended}");
}

#[test]
fn quiet_prints_the_summary_alone() {
    let dir = TempDir::new();
    let (config, _) = playable(&dir);
    let loud = werewolf(&["play", config.to_str().unwrap()]);
    let quiet = werewolf(&["play", config.to_str().unwrap(), "--quiet"]);

    // Exactly the summary: no narration, and no blank line where the
    // narration would have been.
    assert!(quiet.starts_with("seed: 26\n"), "{quiet}");
    assert_eq!(quiet, summary(&loud));
    assert!(!quiet.contains("PhaseBegan"), "{quiet}");
}

#[test]
fn a_watcher_who_stops_reading_still_leaves_a_whole_log() {
    // `werewolf play … | head` closes the pipe partway through the
    // narration. The text sink is optional, so it is dropped and the game
    // plays on: the log is complete and the run succeeds.
    let dir = TempDir::new();
    let (config, trajectory) = playable(&dir);
    let mut child = Command::new(WEREWOLF)
        .args(["play", config.to_str().unwrap()])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take());
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success(), "{output:?}");
    assert_eq!(String::from_utf8_lossy(&output.stderr), "");

    // The whole game is on disk, and it is the game the fixed seed plays.
    // Four rounds where the old rules took two: a day closes on a
    // majority now, and seven random players seldom put four on one
    // target, so most days run out and the pack wins by attrition
    // (ADR-0011).
    let effective = config::load(config::effective_path(&trajectory)).unwrap();
    let transcript = read(&trajectory, &effective);
    assert_eq!(transcript.outcome.winner, Some(Faction::Werewolves));
    assert_eq!(transcript.rounds.len(), 4);
}
