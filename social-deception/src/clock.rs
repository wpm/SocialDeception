//! The episode clock: one origin, shared by everything in an episode.
//!
//! Times in the runtime are [`Instant`]s. An actor reads `Instant::now()` at
//! the moment something happens and puts that instant on the log record
//! unconverted; the log writer is the one thing that converts, and it
//! converts by measuring from the origin a [`Clock`] holds (ADR-0017).
//!
//! That is all a clock is. There is no timestamp type, because an `Instant`
//! already is one, and a clock never appears on a message: a message carries
//! a sequence number and no time.
//!
//! # Why an origin at all
//!
//! Rust's `Instant` is monotonic and has no epoch, so it can only be written
//! down as an offset from another `Instant`. The one time that can be written
//! absolutely, `SystemTime`, is the wall clock, which can be adjusted in the
//! middle of an episode and can go backwards. So the log measures monotonic
//! offsets within the episode from a single origin, and anchors the whole
//! episode to the wall clock exactly once, in its header record.
//!
//! That anchor is the origin's own, which is why the clock reads both clocks
//! at once. Taking the wall-clock reading anywhere else would anchor the log
//! to a moment that is not offset zero, and the one thing the header is for —
//! lining an episode up against another log — would be wrong by however long
//! the two readings were apart.

use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The episode's origin: the instant it started, and the only thing the log's
/// offsets are measured from.
///
/// It is `Copy`. The [`Writer`](crate::log::Writer) and every actor hold a
/// copy of the same clock, so everything either of them measures from the
/// start of the episode is on one timeline.
///
/// It holds one moment read off both of the process's clocks: the monotonic
/// one, which every offset is measured from, and the wall clock, which the
/// log's header carries so that an episode can be lined up against another
/// log. They are the same moment by construction.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    origin: Instant,
    start_unix_ns: u64,
}

impl Clock {
    /// Starts a clock whose origin is now.
    #[must_use]
    pub fn start() -> Self {
        Self {
            origin: Instant::now(),
            start_unix_ns: unix_nanos(SystemTime::now()),
        }
    }

    /// The episode's origin.
    #[must_use]
    pub const fn origin(self) -> Instant {
        self.origin
    }

    /// The origin on the wall clock, as Unix nanoseconds: the one wall-clock
    /// time the log carries, and the moment offset zero refers to.
    #[must_use]
    pub const fn start_unix_ns(self) -> u64 {
        self.start_unix_ns
    }

    /// How long after the origin `at` is, or no time at all if it is
    /// earlier.
    ///
    /// An instant before the origin is not an error to report: it is a
    /// record stamped in the moment between a clock being started and the
    /// thing it times being wired up, and a log whose time ran backwards
    /// would be worse than one whose first record sits at zero.
    #[must_use]
    pub fn offset(self, at: Instant) -> Duration {
        at.saturating_duration_since(self.origin)
    }
}

/// A system time as whole nanoseconds since the Unix epoch.
///
/// A clock set before 1970, or further than 584 years after it, saturates
/// rather than failing: the anchor is for lining logs up, and a log worth
/// nothing to an archaeologist is better than no log.
fn unix_nanos(at: SystemTime) -> u64 {
    let since_epoch = at.duration_since(UNIX_EPOCH).unwrap_or(Duration::ZERO);
    u64::try_from(since_epoch.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A second before `at`, which every platform this runs on can
    /// represent: a `Clock` started now has a monotonic clock behind it that
    /// has been running since boot.
    fn before(at: Instant) -> Instant {
        at.checked_sub(Duration::from_secs(1))
            .expect("the process has been running for a second")
    }

    #[test]
    fn the_origin_is_where_offsets_are_measured_from() {
        let clock = Clock::start();
        assert_eq!(clock.offset(clock.origin()), Duration::ZERO);
        assert_eq!(
            clock.offset(clock.origin() + Duration::from_nanos(7)),
            Duration::from_nanos(7)
        );
    }

    #[test]
    fn an_instant_before_the_origin_is_no_time_at_all() {
        let clock = Clock::start();
        assert_eq!(clock.offset(before(clock.origin())), Duration::ZERO);
    }

    #[test]
    fn a_copy_of_a_clock_reads_the_same_origin() {
        let clock = Clock::start();
        let copy = clock;
        assert_eq!(copy.origin(), clock.origin());
    }

    #[test]
    fn a_clock_anchors_its_origin_to_the_wall_clock() {
        // Both readings are of one moment, so the anchor is the moment
        // offset zero refers to and not some later one. Checked against the
        // wall clock either side of the reading, which is the most a
        // monotonic origin can be compared with.
        let before = unix_nanos(SystemTime::now());
        let clock = Clock::start();
        let after = unix_nanos(SystemTime::now());
        assert!(
            before <= clock.start_unix_ns() && clock.start_unix_ns() <= after,
            "the anchor is read when the origin is: {} is not in {before}..={after}",
            clock.start_unix_ns()
        );
        // And it is carried, not re-read: a copy reports the same anchor.
        assert_eq!(clock.start_unix_ns(), { clock }.start_unix_ns());
    }

    #[test]
    fn an_impossible_wall_clock_saturates_rather_than_failing() {
        assert_eq!(unix_nanos(UNIX_EPOCH), 0);
        assert_eq!(
            unix_nanos(UNIX_EPOCH - Duration::from_secs(1)),
            0,
            "a clock set before the epoch anchors at zero"
        );
        // 584 years after the epoch is the last moment 64 bits of
        // nanoseconds can name; anything past it saturates there rather than
        // wrapping to an anchor in the past.
        let overflowing = UNIX_EPOCH + Duration::from_secs(600 * 365 * 24 * 60 * 60);
        assert_eq!(unix_nanos(overflowing), u64::MAX);
    }

    #[test]
    fn an_offset_grows_with_real_time() {
        let clock = Clock::start();
        let first = clock.offset(Instant::now());
        let second = clock.offset(Instant::now());
        assert!(first <= second);
    }
}
