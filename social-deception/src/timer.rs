//! Where an agent's wake-ups come from.
//!
//! An agent that has a deadline waits on its queues and on a wake channel at
//! the same time, and the wake channel firing is what makes it run a cycle
//! it was not given anything to observe: one that calls
//! [`Handler::timeout`](crate::Handler::timeout), unless an event was
//! waiting too, in which case the deadline joins that event's cycle and the
//! agent observes it as usual. A [`TimerSource`] is whatever hands out those
//! wake channels. In production it is the episode [`Clock`], whose channels fire
//! when the process's monotonic clock reaches the deadline. In tests it is a
//! [`ManualTimer`], whose channel fires when the test says so, so that timing
//! behavior can be exercised without sleeping.
//!
//! A wake channel is asked for with an absolute deadline, never with an
//! interval. That is what keeps the deadline fixed while events arrive: an
//! agent that is spoken to keeps waiting on the same channel, so its deadline
//! is never pushed back.

use std::time::Instant;

use crossbeam_channel::{Receiver, SendError, Sender, at, never, unbounded};

use crate::clock::{Clock, Timestamp};

/// A source of wake channels.
///
/// The value a wake channel delivers is the instant it fired; the agent loop
/// ignores it. A channel fires at most once and must never disconnect while
/// the agent may still be waiting on it.
pub trait TimerSource {
    /// A channel that delivers one message once `deadline` has passed.
    fn wake_at(&mut self, deadline: Timestamp) -> Receiver<Instant>;
}

impl TimerSource for Clock {
    /// A channel driven by real elapsed time. A deadline already in the past
    /// fires immediately; one too far in the future for the process clock to
    /// represent never fires.
    fn wake_at(&mut self, deadline: Timestamp) -> Receiver<Instant> {
        self.instant_of(deadline).map_or_else(never, at)
    }
}

/// A timer source driven by a test.
///
/// Every deadline an agent asks for is reported on the request channel that
/// [`ManualTimer::new`] returns alongside it, and the agent's wake channel
/// fires when [`ManualTimerControl::fire`] is called. A fire that arrives
/// before the agent is waiting is kept and delivered at the next wait; the
/// deadline value plays no part in when a fire is delivered.
#[derive(Debug)]
pub struct ManualTimer {
    requests: Sender<Timestamp>,
    wake: Receiver<Instant>,
}

/// The test's end of a [`ManualTimer`].
///
/// Dropping it disconnects the agent's wake channel, which the agent reports
/// as an error rather than a wake-up.
#[derive(Debug)]
pub struct ManualTimerControl {
    requests: Receiver<Timestamp>,
    fire: Sender<Instant>,
}

impl ManualTimer {
    /// Creates a manual timer and the control that drives it.
    #[must_use]
    pub fn new() -> (Self, ManualTimerControl) {
        let (requests, requested) = unbounded();
        let (fire, wake) = unbounded();
        (
            Self { requests, wake },
            ManualTimerControl {
                requests: requested,
                fire,
            },
        )
    }
}

impl TimerSource for ManualTimer {
    fn wake_at(&mut self, deadline: Timestamp) -> Receiver<Instant> {
        // A control that has stopped listening for requests is not an error:
        // the test may only care about firing.
        let _ = self.requests.send(deadline);
        self.wake.clone()
    }
}

impl ManualTimerControl {
    /// The deadlines the agent has asked for, in the order it asked.
    #[must_use]
    pub fn requests(&self) -> &Receiver<Timestamp> {
        &self.requests
    }

    /// Fires the agent's wake channel.
    ///
    /// # Errors
    ///
    /// If the timer has been dropped, which happens when the agent's thread
    /// has exited.
    pub fn fire(&self) -> Result<(), SendError<Instant>> {
        self.fire.send(Instant::now())
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossbeam_channel::TryRecvError;

    use super::*;

    #[test]
    fn clock_wakes_no_earlier_than_the_deadline() {
        let mut clock = Clock::start();
        let deadline = clock.now() + Duration::from_millis(2);
        clock.wake_at(deadline).recv().unwrap();
        assert!(clock.now() >= deadline);
    }

    #[test]
    fn clock_wakes_immediately_for_a_deadline_that_has_passed() {
        let mut clock = Clock::start();
        let deadline = clock.now();
        assert!(clock.wake_at(deadline).recv().is_ok());
    }

    #[test]
    fn clock_does_not_wake_early_for_a_far_deadline() {
        let mut clock = Clock::start();
        let wake = clock.wake_at(Timestamp::from(Duration::MAX));
        assert_eq!(wake.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn manual_timer_reports_requests_and_fires_on_command() {
        let (mut timer, control) = ManualTimer::new();
        let deadline = Timestamp::from(Duration::from_secs(1));
        let wake = timer.wake_at(deadline);
        assert_eq!(control.requests().try_recv(), Ok(deadline));
        assert_eq!(wake.try_recv(), Err(TryRecvError::Empty));
        control.fire().unwrap();
        assert!(wake.try_recv().is_ok());
        assert_eq!(wake.try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn manual_timer_keeps_an_early_fire_for_the_next_wait() {
        let (mut timer, control) = ManualTimer::new();
        control.fire().unwrap();
        let wake = timer.wake_at(Timestamp::from(Duration::ZERO));
        assert!(wake.try_recv().is_ok());
    }

    #[test]
    fn dropping_the_control_disconnects_the_wake_channel() {
        let (mut timer, control) = ManualTimer::new();
        let wake = timer.wake_at(Timestamp::from(Duration::ZERO));
        drop(control);
        assert_eq!(wake.try_recv(), Err(TryRecvError::Disconnected));
    }

    #[test]
    fn firing_a_dropped_timer_is_an_error() {
        let (timer, control) = ManualTimer::new();
        drop(timer);
        assert!(control.fire().is_err());
    }
}
