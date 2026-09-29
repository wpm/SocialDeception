//! The reminders an actor is holding, and what tells it one is due.
//!
//! An [`Action::Remind`](super::Action::Remind) is handed to the actor's own
//! perception thread, which holds it in [`Reminders`] until its deadline. At
//! the deadline the payload is delivered back to the actor as an ordinary
//! message from itself and observed like any other.
//!
//! # Two kinds of timer, and no trait
//!
//! The perception thread waits on **one** channel for its reminders, because
//! `crossbeam`'s `select!` waits on a fixed set of channels and an actor may
//! hold any number of reminders. So [`Reminders`] keeps them ordered by
//! deadline and arms one channel for the earliest of them.
//!
//! In production that channel comes from `crossbeam_channel::at`, which fires
//! when the process's monotonic clock reaches the deadline. In tests it is an
//! **ordinary channel the test holds the sender of**, so a test fires the
//! timer when it chooses and timed behavior is deterministic with no sleeping
//! and no trait: [`Timer::held`] hands back the sender, and firing it makes
//! the perception thread deliver whatever is due — which under a held timer is
//! everything, since the test rather than the clock decides when a deadline
//! has passed.
//!
//! That is what replaced the old runtime's `TimerSource` trait and its
//! `ManualTimer`. One enum with two cases is smaller than a trait with two
//! implementations, and it is what lets [`Clock::start`](crate::Clock::start)
//! be private: nothing in a test constructs a clock to serve as a timer any
//! more.

use std::collections::BTreeMap;
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, at, never, unbounded};

/// Where an actor's reminder wake-ups come from.
///
/// A [`real`](Timer::real) timer is driven by the process's monotonic clock; a
/// [`held`](Timer::held) one is driven by whoever holds the sender it gave
/// back, which in a test is the test.
#[derive(Debug)]
pub enum Timer {
    /// Driven by elapsed time: each deadline is armed with
    /// `crossbeam_channel::at`.
    Real,
    /// Driven by a channel somebody else holds the sender of. Every fire on
    /// it makes the perception thread deliver every reminder it is holding,
    /// earliest first, because under a held timer the holder and not the
    /// clock is what says a deadline has passed.
    Held(Receiver<Instant>),
}

impl Timer {
    /// A timer driven by real elapsed time, as a run uses.
    #[must_use]
    pub const fn real() -> Self {
        Self::Real
    }

    /// A timer driven by the sender this returns beside it, as a test uses.
    #[must_use]
    pub fn held() -> (Self, Sender<Instant>) {
        let (fire, fired) = unbounded();
        (Self::Held(fired), fire)
    }

    /// Whether every fire on this timer delivers everything held rather than
    /// only what the clock says is due.
    const fn is_held(&self) -> bool {
        matches!(self, Self::Held(_))
    }

    /// The channel to wait on for a reminder due at `earliest`, or one that
    /// never fires if the actor holds none.
    fn arm(&self, earliest: Option<Instant>) -> Receiver<Instant> {
        match self {
            // A real timer arms the earliest deadline it has been given; a
            // deadline already past fires at once.
            Self::Real => earliest.map_or_else(never, at),
            // A held timer is the one channel throughout, whether or not
            // anything is held: a fire that arrives before any reminder was
            // set finds nothing due and delivers nothing.
            Self::Held(fired) => fired.clone(),
        }
    }
}

/// The reminders one actor is holding, ordered by deadline, and the channel
/// armed for the earliest of them.
///
/// Reminders **accumulate** and each fires once; nothing is cancelled. Two
/// set for the same instant both fire, in the order they were set, which is
/// why the map's values are queues.
#[derive(Debug)]
pub struct Reminders<P> {
    timer: Timer,
    /// What is held, earliest deadline first, and within a deadline in the
    /// order the reminders were set.
    ///
    /// Each entry carries the sequence number the reminder was given when it
    /// was set, because that is the number the message it becomes will carry
    /// (ADR-0016): a reminder is numbered where every other message is, at
    /// the moment its actor decided to send it.
    held: BTreeMap<Instant, Vec<(u64, P)>>,
    /// The channel armed for [`earliest`](Self::earliest), kept so that the
    /// perception thread's `select!` has something to borrow.
    armed: Receiver<Instant>,
}

impl<P> Reminders<P> {
    /// An actor holding nothing, whose wake-ups come from `timer`.
    #[must_use]
    pub fn new(timer: Timer) -> Self {
        let armed = timer.arm(None);
        Self {
            timer,
            held: BTreeMap::new(),
            armed,
        }
    }

    /// The channel the perception thread waits on.
    ///
    /// It is re-armed whenever the earliest deadline changes, so an actor
    /// that sets a nearer reminder waits on the nearer one.
    #[must_use]
    pub const fn armed(&self) -> &Receiver<Instant> {
        &self.armed
    }

    /// The earliest deadline held, if any.
    fn earliest(&self) -> Option<Instant> {
        self.held.keys().next().copied()
    }

    /// Holds `payload` until `deadline`, numbered `seq`.
    pub fn hold(&mut self, deadline: Instant, seq: u64, payload: P) {
        let earliest = self.earliest();
        self.held.entry(deadline).or_default().push((seq, payload));
        // Re-armed only when the earliest moved, so an actor that keeps
        // setting later reminders waits on the channel it already has.
        if earliest != self.earliest() {
            self.armed = self.timer.arm(self.earliest());
        }
    }

    /// Whatever is due at `now`, earliest deadline first, taken out of the
    /// held set: each reminder fires once.
    ///
    /// Under a [`held`](Timer::held) timer everything is due, because the
    /// holder of the timer and not the clock is what says a deadline has
    /// passed.
    ///
    /// # Panics
    ///
    /// Never in practice: the earliest deadline is one of the held ones by
    /// construction, so the removal below cannot miss.
    pub fn due(&mut self, now: Instant) -> Vec<(u64, P)> {
        let mut due = Vec::new();
        while let Some(deadline) = self.earliest() {
            if !self.timer.is_held() && deadline > now {
                break;
            }
            due.extend(
                self.held
                    .remove(&deadline)
                    .expect("the earliest deadline is one of the held ones"),
            );
        }
        self.armed = self.timer.arm(self.earliest());
        due
    }

    /// Everything still held, earliest deadline first, leaving nothing.
    ///
    /// This is what a stopped actor's perception thread logs as undelivered:
    /// a reminder its actor set and will never observe.
    pub fn drain(&mut self) -> Vec<(u64, P)> {
        let held = std::mem::take(&mut self.held);
        self.armed = self.timer.arm(None);
        held.into_values().flatten().collect()
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crossbeam_channel::TryRecvError;

    use super::*;

    /// An instant far enough ahead that no test reaches it.
    fn later() -> Instant {
        Instant::now() + Duration::from_secs(3600)
    }

    #[test]
    fn an_actor_holding_nothing_waits_on_a_channel_that_never_fires() {
        let reminders = Reminders::<u64>::new(Timer::real());
        assert_eq!(reminders.armed().try_recv(), Err(TryRecvError::Empty));
    }

    #[test]
    fn a_real_timer_fires_when_the_deadline_passes() {
        let mut reminders = Reminders::new(Timer::real());
        let deadline = Instant::now() + Duration::from_millis(2);
        reminders.hold(deadline, 0, "soon");
        let fired = reminders.armed().recv().unwrap();
        assert!(fired >= deadline);
        assert_eq!(reminders.due(Instant::now()), [(0, "soon")]);
    }

    #[test]
    fn a_real_timer_does_not_fire_early() {
        let mut reminders = Reminders::new(Timer::real());
        reminders.hold(later(), 0, "far");
        assert_eq!(reminders.armed().try_recv(), Err(TryRecvError::Empty));
        assert!(reminders.due(Instant::now()).is_empty());
    }

    #[test]
    fn a_nearer_reminder_re_arms_the_channel() {
        let mut reminders = Reminders::new(Timer::real());
        reminders.hold(later(), 0, "far");
        reminders.hold(Instant::now() + Duration::from_millis(2), 1, "soon");
        // The nearer one fires, which the far channel would not have.
        reminders.armed().recv().unwrap();
        assert_eq!(reminders.due(Instant::now()), [(1, "soon")]);
    }

    #[test]
    fn reminders_accumulate_and_each_fires_once() {
        let mut reminders = Reminders::new(Timer::real());
        let now = Instant::now();
        reminders.hold(now, 0, "first");
        reminders.hold(now, 1, "second");
        assert_eq!(reminders.due(now), [(0, "first"), (1, "second")]);
        assert!(reminders.due(now).is_empty(), "each fires once");
    }

    #[test]
    fn what_is_due_comes_earliest_first_and_leaves_the_rest() {
        let mut reminders = Reminders::new(Timer::real());
        let now = Instant::now();
        reminders.hold(now + Duration::from_nanos(2), 1, "later");
        reminders.hold(now, 0, "now");
        reminders.hold(later(), 2, "far");
        assert_eq!(
            reminders.due(now + Duration::from_nanos(2)),
            [(0, "now"), (1, "later")]
        );
        assert_eq!(reminders.drain(), [(2, "far")]);
    }

    #[test]
    fn a_held_timer_fires_when_the_test_says_so_and_delivers_everything() {
        let (timer, fire) = Timer::held();
        let mut reminders = Reminders::new(timer);
        // Both deadlines are in the future, so a real timer would deliver
        // neither; a held one delivers what the test asked for.
        reminders.hold(later(), 0, "far");
        reminders.hold(later() + Duration::from_secs(1), 1, "further");
        assert_eq!(reminders.armed().try_recv(), Err(TryRecvError::Empty));
        fire.send(Instant::now()).unwrap();
        reminders.armed().recv().unwrap();
        assert_eq!(reminders.due(Instant::now()), [(0, "far"), (1, "further")]);
    }

    #[test]
    fn a_held_timer_fired_with_nothing_held_delivers_nothing() {
        let (timer, fire) = Timer::held();
        let mut reminders = Reminders::<&str>::new(timer);
        fire.send(Instant::now()).unwrap();
        reminders.armed().recv().unwrap();
        assert!(reminders.due(Instant::now()).is_empty());
    }

    #[test]
    fn draining_leaves_nothing_and_disarms() {
        let mut reminders = Reminders::new(Timer::real());
        reminders.hold(Instant::now(), 0, "due");
        reminders.hold(later(), 1, "far");
        assert_eq!(reminders.drain(), [(0, "due"), (1, "far")]);
        assert!(reminders.drain().is_empty());
        assert_eq!(reminders.armed().try_recv(), Err(TryRecvError::Empty));
    }
}
