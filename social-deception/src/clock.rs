//! The episode clock.
//!
//! A [`Clock`] fixes an origin, the moment it was started, and hands out
//! [`Timestamp`]s: whole nanoseconds since that origin, read from the
//! process's monotonic clock.
//!
//! # What is known about a thing's time
//!
//! Two traits say what a timestamped thing knows about itself, and every
//! type the runtime stamps implements one of them. [`Created`] is the
//! instant something came into being: for an event, the instant its sender
//! sent it. [`Timestamped`] adds the instant it reached whoever holds it,
//! and with it a [`latency`](Timestamped::latency), the delay that holder
//! actually suffered.
//!
//! The split is not decoration. An event on the wire has a creation time and
//! nothing else, because it has not been received by anyone yet; the same
//! event, popped off a queue as an observation, has both. Keeping them apart
//! in the types means a value that cannot say when it was received cannot be
//! asked.

use std::ops::{Add, Sub};
use std::time::{Duration, Instant};

use serde::Serialize;

/// A moment in an episode, as whole nanoseconds since the episode's [`Clock`]
/// was started.
///
/// Serializes as a bare integer. Two timestamps are comparable only when they
/// came from the same clock. The default is the clock's origin, which is
/// what a caller that does not care when something happened wants.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct Timestamp(u64);

impl Timestamp {
    /// Nanoseconds since the clock was started.
    #[must_use]
    pub const fn nanos(self) -> u64 {
        self.0
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    /// The timestamp `rhs` later. Saturates rather than wrapping.
    fn add(self, rhs: Duration) -> Self {
        Self(u64::try_from(u128::from(self.0) + rhs.as_nanos()).unwrap_or(u64::MAX))
    }
}

impl Sub for Timestamp {
    type Output = Duration;

    /// How long `rhs` was before this timestamp. Saturates at zero rather
    /// than wrapping, so subtracting a later timestamp from an earlier one
    /// is no time at all rather than an enormous one.
    fn sub(self, rhs: Self) -> Duration {
        Duration::from_nanos(self.0.saturating_sub(rhs.0))
    }
}

impl From<Duration> for Timestamp {
    /// Converts a duration since the clock's origin. A duration too long to fit
    /// in 64 bits of nanoseconds saturates.
    fn from(since_start: Duration) -> Self {
        Self(u64::try_from(since_start.as_nanos()).unwrap_or(u64::MAX))
    }
}

/// Something that knows when it came into being.
///
/// For everything the runtime stamps, that is the instant the sender sent
/// it. An [`Action`](crate::Action) a handler returns is deliberately not
/// one of these: it has no creation time until the loop sends it, and the
/// handler cannot know that instant.
pub trait Created {
    /// When it was created.
    fn created(&self) -> Timestamp;
}

/// Something that knows both when it was created and when it reached the
/// agent holding it.
///
/// An observation and a popped control are the two: each was created by
/// somebody else and has since arrived here. The gap between the two is the
/// [`latency`](Timestamped::latency).
pub trait Timestamped: Created {
    /// When it was received: the instant the agent popped it off its queue.
    fn received(&self) -> Timestamp;

    /// How stale it was when the agent saw it: routing, plus however long it
    /// waited in the queue. That is the whole delay the agent suffered, and
    /// it is the reason `received` is the pop and not the enqueue, which
    /// ADR-0002 argues is send time plus scheduler jitter in one process.
    fn latency(&self) -> Duration {
        self.received() - self.created()
    }
}

/// A monotonic clock with a fixed origin, shared by every agent in an episode.
///
/// It is `Copy`; every copy reads the same origin and produces comparable
/// timestamps.
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

    /// The process instant a timestamp from this clock refers to, or `None`
    /// if the process clock cannot represent it.
    pub(crate) fn instant_of(self, stamp: Timestamp) -> Option<Instant> {
        self.origin.checked_add(Duration::from_nanos(stamp.nanos()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(nanos: u64) -> Timestamp {
        Timestamp::from(Duration::from_nanos(nanos))
    }

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

    #[test]
    fn adding_a_duration_moves_a_timestamp_later() {
        let stamp = Timestamp::from(Duration::from_nanos(5)) + Duration::from_nanos(7);
        assert_eq!(stamp.nanos(), 12);
        assert_eq!(
            (Timestamp::from(Duration::MAX) + Duration::MAX).nanos(),
            u64::MAX
        );
    }

    #[test]
    fn subtracting_timestamps_gives_the_duration_between_them() {
        let (earlier, later) = (at(5), at(12));
        assert_eq!(later - earlier, Duration::from_nanos(7));
        assert_eq!(earlier - earlier, Duration::ZERO);
        // Backwards is no time at all, never a wrapped enormity.
        assert_eq!(earlier - later, Duration::ZERO);
    }

    /// Something created at 40 and received at 55.
    struct Late;

    impl Created for Late {
        fn created(&self) -> Timestamp {
            at(40)
        }
    }

    impl Timestamped for Late {
        fn received(&self) -> Timestamp {
            at(55)
        }
    }

    #[test]
    fn latency_is_the_gap_between_creation_and_receipt() {
        assert_eq!(Late.latency(), Duration::from_nanos(15));
    }

    #[test]
    fn instant_of_inverts_now() {
        let clock = Clock::start();
        let stamp = clock.now();
        let instant = clock.instant_of(stamp).unwrap();
        assert_eq!(instant, clock.origin + Duration::from_nanos(stamp.nanos()));
    }
}
