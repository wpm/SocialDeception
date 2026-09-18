//! The episode clock.
//!
//! ADR-0001: times are taken from a monotonic clock, and one process means one
//! clock, so timestamps are directly comparable across agents with no skew to
//! correct. Rust's monotonic clock is [`Instant`], which cannot be serialised
//! because it has no defined origin. A [`Clock`] fixes the origin — the moment
//! the episode started — and hands out [`Timestamp`]s measured from it.

use std::time::{Duration, Instant};

use serde::Serialize;

/// A moment in an episode, as whole nanoseconds since the episode's [`Clock`]
/// was started.
///
/// Serialises as a bare integer. Two timestamps are comparable only when they
/// came from the same clock, which in practice means the same episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Nanoseconds since the clock was started.
    #[must_use]
    pub const fn nanos(self) -> u64 {
        self.0
    }
}

impl From<Duration> for Timestamp {
    /// Converts a duration since the clock's origin. A duration too long to fit
    /// in 64 bits of nanoseconds — more than five centuries — saturates.
    fn from(since_start: Duration) -> Self {
        Self(u64::try_from(since_start.as_nanos()).unwrap_or(u64::MAX))
    }
}

/// A monotonic clock with a fixed origin, shared by every agent in an episode.
///
/// It is `Copy` so that each agent thread can carry its own copy; they all
/// read the same underlying instant and so produce comparable timestamps.
#[derive(Debug, Clone, Copy)]
pub struct Clock {
    origin: Instant,
}

impl Clock {
    /// Starts a clock whose origin is now.
    #[must_use]
    pub fn start() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    /// The current time, measured from this clock's origin.
    #[must_use]
    pub fn now(self) -> Timestamp {
        Timestamp::from(self.origin.elapsed())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timestamps_are_monotonic() {
        let clock = Clock::start();
        let first = clock.now();
        let second = clock.now();
        assert!(first <= second);
    }

    #[test]
    fn timestamp_is_nanoseconds() {
        let stamp = Timestamp::from(Duration::new(1, 500));
        assert_eq!(stamp.nanos(), 1_000_000_500);
        assert_eq!(serde_json::to_string(&stamp).unwrap(), "1000000500");
    }

    #[test]
    fn overlong_duration_saturates() {
        assert_eq!(Timestamp::from(Duration::MAX).nanos(), u64::MAX);
    }
}
