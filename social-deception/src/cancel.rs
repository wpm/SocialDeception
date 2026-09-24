//! Cooperative cancellation: the [`Cancel`] a handler is given, and the
//! [`Trip`] whoever sends it a control pulls.
//!
//! Rust cannot kill a thread, so a cycle cannot be preempted; it can only be
//! *asked* to stop, and the asking has to cost nothing when nobody asks.
//! That rules out the obvious shapes. A watcher thread per agent costs a
//! thread per agent to do nothing almost always. Running the handler on a
//! worker thread while the loop waits on the control queue costs the same
//! thread and still cannot stop the work, because an abandoned call to a
//! model provider runs, and bills, to completion whether or not anyone is
//! reading it (ADR-0007). The handler has to cooperate either way, and
//! cooperation alone costs one flag read per check.
//!
//! # A closed channel, not a flag
//!
//! A flag can only be polled, and a policy blocked on a model's answer has
//! nowhere to poll from. So a `Cancel` is a channel that is **closed** when
//! it trips. Closing is what makes it visible to `select!`: a receiver whose
//! every sender has been dropped becomes ready at once and stays ready, so a
//! policy can wait on its own result and [`Cancel::receiver`] together and
//! wake on whichever comes first. The channel carries [`Never`], which has
//! no values, so the only thing that can ever come out of it is the
//! disconnection.
//!
//! [`Cancel::is_cancelled`] is there for a handler that loops rather than
//! blocks, and is a channel emptiness check rather than a separate flag, so
//! there is one fact and not two that could disagree.
//!
//! # Who trips it, and when
//!
//! Nobody watches for a control to arrive: whoever *puts* one on an agent's
//! control queue trips that agent's current cancel in the same step. That is
//! what [`ControlSender`] is — the sending half of the control queue, which
//! is a channel sender and a shared slot holding the cycle's live [`Trip`] —
//! and it is why a control cannot be queued without the cycle in progress
//! hearing about it.
//!
//! The loop puts a fresh [`Cancel`] in the slot at the top of every cycle.
//! A cycle in which no control arrives is never cancelled, and its `Trip` is
//! dropped, unpulled, when the next cycle replaces it.
//!
//! # Example
//!
//! A handler that blocks until it is told to give up:
//!
//! ```
//! use crossbeam_channel::{Receiver, select};
//! use social_deception::Cancel;
//!
//! fn deliberate(answer: &Receiver<String>, cancel: &Cancel) -> Option<String> {
//!     select! {
//!         recv(answer) -> said => said.ok(),
//!         recv(cancel.receiver()) -> _ => None,
//!     }
//! }
//! ```

use std::sync::{Arc, Mutex};

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, unbounded};

use crate::clock::{Clock, Timestamp};
use crate::event::Control;

/// A type with no values.
///
/// It is the message type of the channel behind a [`Cancel`], which says in
/// the type what the channel is for: nothing is ever sent on it, and the
/// only thing a receiver can observe is its closing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Never {}

/// A handler's view of whether its cycle has been cancelled.
///
/// The loop makes one per cycle and hands it to [`Handler::handle`]. It is
/// cheap to hold and cheap to check; see the [module documentation](self)
/// for why it is a closed channel rather than a flag.
///
/// [`Handler::handle`]: crate::Handler::handle
#[derive(Debug, Clone)]
pub struct Cancel {
    tripped: Receiver<Never>,
}

impl Cancel {
    /// A fresh cancel and the trip-wire that fires it.
    ///
    /// Dropping the [`Trip`] without pulling it leaves the cancel untripped
    /// forever, which is what happens to the cancel of a cycle nobody
    /// interrupted.
    #[must_use]
    pub(crate) fn new() -> (Self, Trip) {
        let (sender, tripped) = bounded(0);
        (Self { tripped }, Trip { _sender: sender })
    }

    /// A cancel that is already tripped, for a caller with nothing to
    /// cancel: a handler called outside a loop, or a test.
    #[must_use]
    pub fn cancelled() -> Self {
        let (cancel, trip) = Self::new();
        trip.pull();
        cancel
    }

    /// Whether the cycle has been cancelled.
    ///
    /// For a handler that works in steps and checks between them. A handler
    /// that blocks wants [`receiver`](Self::receiver) instead.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        // Nothing is ever sent on the channel, so `try_recv` can only ever
        // report it empty or report it disconnected, and disconnected is
        // exactly what tripped means. Reading the one channel rather than a
        // flag beside it means there are not two facts to disagree.
        matches!(self.tripped.try_recv(), Err(TryRecvError::Disconnected))
    }

    /// The receiver that becomes ready, by disconnecting, when the cancel
    /// trips.
    ///
    /// This is the arm a blocked handler adds to its own `select!`; see the
    /// [module documentation](self).
    #[must_use]
    pub fn receiver(&self) -> &Receiver<Never> {
        &self.tripped
    }
}

/// The trip-wire of one [`Cancel`]: pulling it, or dropping it, trips the
/// cancel.
///
/// It is the channel's only sender, and the cancel trips when that sender
/// goes away. Pulling it is therefore the same as dropping it, and the
/// method exists so that the code that does it says what it means.
#[derive(Debug)]
pub(crate) struct Trip {
    _sender: Sender<Never>,
}

impl Trip {
    /// Trips the cancel this wire belongs to.
    pub(crate) fn pull(self) {
        drop(self);
    }
}

/// The live trip-wire of an agent's current cycle, shared between the
/// agent's thread and whoever sends it a control.
///
/// The loop puts each cycle's wire in; a sender takes it out and pulls it.
/// Taking rather than borrowing is what makes a second control in the same
/// cycle a no-op rather than a second pull of an already-pulled wire.
type Slot = Arc<Mutex<Option<Trip>>>;

/// The sending half of an agent's control queue: push the control, then trip
/// the cycle.
///
/// The two steps are one operation and are never available separately, which
/// is the whole point of the type: a control on the queue and a cycle that
/// has not heard about it is the state this makes unrepresentable. The order
/// matters too. The control is on the queue before the cancel trips, so a
/// handler that wakes on the cancel and returns finds the control already
/// waiting when the loop looks.
///
/// It is not generic over a [`Domain`](crate::Domain), as the event queue
/// is, because a control carries no domain data: it is the same queue
/// whatever game is being played.
///
/// `Clone` is a clone of the same queue and the same slot, as a channel
/// sender's is.
#[derive(Debug, Clone)]
pub struct ControlSender {
    controls: Sender<Signal>,
    trip: Slot,
}

/// A control on its way to an agent's control queue, stamped by whoever sent
/// it.
///
/// The popped form, with the instant the agent popped it, is
/// [`Instruction`](crate::Instruction).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Signal {
    /// The control.
    pub control: Control,
    /// When the environment sent it.
    pub created: Timestamp,
}

impl ControlSender {
    /// The sending and receiving halves of one agent's control queue, and
    /// the slot the agent's loop arms each cycle through.
    #[must_use]
    pub fn new() -> (Self, Receiver<Signal>, Arm) {
        let (controls, receiver) = unbounded();
        let trip: Slot = Arc::new(Mutex::new(None));
        let arm = Arm {
            trip: Arc::clone(&trip),
        };
        (Self { controls, trip }, receiver, arm)
    }

    /// Puts a control on the queue and trips the receiving agent's current
    /// cycle, in that order.
    ///
    /// # Errors
    ///
    /// The control, if the agent's control queue has been dropped. The
    /// cancel is not tripped in that case: there is no cycle to preempt.
    pub fn send(&self, signal: Signal) -> Result<(), Signal> {
        self.controls.send(signal).map_err(|error| error.0)?;
        self.trip();
        Ok(())
    }

    /// A control stamped with the clock's current time, sent as
    /// [`send`](Self::send) sends one.
    ///
    /// # Errors
    ///
    /// As [`send`](Self::send).
    pub fn control(&self, clock: Clock, control: Control) -> Result<(), Signal> {
        self.send(Signal {
            control,
            created: clock.now(),
        })
    }

    /// Trips the current cycle's cancel, if it has one and it has not
    /// already been tripped.
    fn trip(&self) {
        // A poisoned lock means a thread panicked holding it; the wire is
        // still there to pull, and refusing to pull it would leave an agent
        // blocked forever on a cancel nobody can trip.
        let taken = match self.trip.lock() {
            Ok(mut slot) => slot.take(),
            Err(poisoned) => poisoned.into_inner().take(),
        };
        if let Some(trip) = taken {
            trip.pull();
        }
    }
}

/// The agent loop's end of the trip slot: where each cycle's wire is put.
///
/// It is held by the loop alone. Every [`ControlSender`] for the same agent
/// shares the slot it arms.
#[derive(Debug)]
pub struct Arm {
    trip: Slot,
}

impl Arm {
    /// Makes a cancel for a new cycle and puts its wire in the slot, so that
    /// the next control sent trips this one.
    ///
    /// The previous cycle's wire, if it was never pulled, is dropped here,
    /// which trips a cancel nobody is holding any more.
    #[must_use]
    pub fn arm(&self) -> Cancel {
        let (cancel, trip) = Cancel::new();
        match self.trip.lock() {
            Ok(mut slot) => *slot = Some(trip),
            Err(poisoned) => *poisoned.into_inner() = Some(trip),
        }
        cancel
    }
}

#[cfg(test)]
mod tests {
    use std::thread;
    use std::time::Duration;

    use crossbeam_channel::select;

    use super::*;

    /// How long a test waits before giving up. A test only ever waits this
    /// long when it has already failed.
    const PATIENCE: Duration = Duration::from_secs(10);

    fn signal(control: Control) -> Signal {
        Signal {
            control,
            created: Timestamp::default(),
        }
    }

    #[test]
    fn a_fresh_cancel_is_not_cancelled_and_its_receiver_is_not_ready() {
        let (cancel, trip) = Cancel::new();
        assert!(!cancel.is_cancelled());
        select! {
            recv(cancel.receiver()) -> _ => panic!("an untripped cancel is not ready"),
            default => {}
        }
        drop(trip);
    }

    #[test]
    fn pulling_the_trip_cancels_and_stays_cancelled() {
        let (cancel, trip) = Cancel::new();
        trip.pull();
        assert!(cancel.is_cancelled());
        // Readiness is not consumed by being observed: every later check
        // sees it too, which is what a handler that checks twice needs.
        assert!(cancel.is_cancelled());
        assert!(cancel.receiver().recv().is_err());
    }

    #[test]
    fn a_blocked_handler_wakes_when_the_cancel_trips() {
        let (cancel, trip) = Cancel::new();
        let waiter = thread::spawn(move || {
            let (_never, answer) = unbounded::<()>();
            select! {
                recv(answer) -> _ => "answered",
                recv(cancel.receiver()) -> _ => "cancelled",
            }
        });
        trip.pull();
        assert_eq!(waiter.join().unwrap(), "cancelled");
    }

    #[test]
    fn an_already_cancelled_cancel_is_cancelled_from_the_start() {
        assert!(Cancel::cancelled().is_cancelled());
    }

    #[test]
    fn sending_a_control_queues_it_and_trips_the_armed_cycle() {
        let (sender, controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        assert!(!cancel.is_cancelled());
        sender.send(signal(Control::Stop)).unwrap();
        assert!(cancel.is_cancelled());
        // The control was on the queue before the cancel tripped, so a
        // handler that wakes on the cancel finds it already waiting.
        assert_eq!(controls.try_recv().unwrap(), signal(Control::Stop));
    }

    #[test]
    fn a_second_control_in_one_cycle_queues_without_a_second_pull() {
        let (sender, controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        for control in [Control::Start, Control::Stop] {
            sender.send(signal(control)).unwrap();
        }
        assert!(cancel.is_cancelled());
        assert_eq!(controls.len(), 2);
    }

    #[test]
    fn arming_a_new_cycle_leaves_the_new_cancel_for_the_next_control() {
        let (sender, _controls, arm) = ControlSender::new();
        let first = arm.arm();
        let second = arm.arm();
        // Re-arming drops the first cycle's wire, which trips a cancel
        // nobody is holding any more; the live one is the second.
        assert!(first.is_cancelled());
        assert!(!second.is_cancelled());
        sender.send(signal(Control::Stop)).unwrap();
        assert!(second.is_cancelled());
    }

    #[test]
    fn a_cycle_nobody_interrupts_is_never_cancelled() {
        let (_sender, _controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        for _ in 0..100 {
            assert!(!cancel.is_cancelled());
        }
    }

    #[test]
    fn a_control_sent_before_any_cycle_is_armed_still_queues() {
        let (sender, controls, arm) = ControlSender::new();
        sender.send(signal(Control::Start)).unwrap();
        // Nothing was armed, so nothing was tripped, and the cycle armed
        // afterwards is live rather than born cancelled: it is the cycle
        // that will pop the control.
        let cancel = arm.arm();
        assert!(!cancel.is_cancelled());
        assert_eq!(controls.try_recv().unwrap(), signal(Control::Start));
    }

    #[test]
    fn a_dropped_control_queue_gives_the_control_back_untripped() {
        let (sender, controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        drop(controls);
        assert_eq!(
            sender.send(signal(Control::Stop)),
            Err(signal(Control::Stop))
        );
        assert!(
            !cancel.is_cancelled(),
            "there is no cycle to preempt when the queue is gone"
        );
    }

    #[test]
    fn a_control_stamped_from_the_clock_carries_when_it_was_sent() {
        let clock = Clock::start();
        let (sender, controls, _arm) = ControlSender::new();
        let before = clock.now();
        sender.control(clock, Control::Start).unwrap();
        let after = clock.now();
        let queued = controls.try_recv().unwrap();
        assert_eq!(queued.control, Control::Start);
        assert!(before <= queued.created && queued.created <= after);
    }

    #[test]
    fn a_clone_sends_on_the_same_queue_and_trips_the_same_cycle() {
        let (sender, controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        let clone = sender.clone();
        clone.send(signal(Control::Stop)).unwrap();
        assert!(cancel.is_cancelled());
        assert_eq!(controls.len(), 1);
    }

    #[test]
    fn a_trip_pulled_from_another_thread_is_seen_here() {
        let (sender, _controls, arm) = ControlSender::new();
        let cancel = arm.arm();
        thread::spawn(move || sender.send(signal(Control::Stop)).unwrap());
        assert!(
            cancel.receiver().recv_timeout(PATIENCE).is_err(),
            "the wait ends with the channel disconnected"
        );
        assert!(cancel.is_cancelled());
    }
}
