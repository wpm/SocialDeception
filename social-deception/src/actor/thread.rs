//! The two threads of one actor, and the [`Context`] that is the plumbing
//! between them.
//!
//! ```text
//! control channel ──┐ checked first, with try_recv
//!                   ▼
//! inbox ──────▶ perception thread ──Call──▶ handler thread ──▶ recipients' inboxes
//! reminders ──▶ stamp · log                policy / step      controls · rewards
//!                   │                            │
//!                   └────────── log ◀────────────┘
//! ```
//!
//! # The perception thread does only fast work
//!
//! It receives from the inbox, from its reminder timer and from its control
//! channel; stamps what arrives with `Instant::now()`; logs the observation;
//! and forwards it. Nothing it does can block for long, which is the whole
//! point: a handler blocked in a multi-second model call does not delay the
//! stamp of anything said during the call (ADR-0016).
//!
//! `crossbeam`'s `select!` picks **at random** among ready channels, so the
//! loop checks the control channel with `try_recv` before every `select!`. A
//! waiting control always goes first.
//!
//! # The handler thread does one call at a time
//!
//! It receives one call at a time, calls `start` or the handler, and
//! carries out each item **as the iterator yields it**: a send goes through
//! the [`Router`] to each recipient's inbox and is logged as an action; a
//! reminder goes back to the perception thread's timer; a command goes to its
//! recipients' control channels; a reward is logged. One call in flight per
//! actor.
//!
//! # Stop preempts everything
//!
//! When the perception thread sees a [`Control::Stop`], ahead of anything in
//! its inbox, it:
//!
//! 1. logs everything still in its inbox, and every reminder it holds, as
//!    **undelivered**, forwarding none of it;
//! 2. sets the actor's stopped flag and drops its end of the call channel, so
//!    the handler thread ends once its current call returns;
//! 3. and the handler thread logs anything yielded after the flag is set as
//!    **unsent**, carrying none of it out.
//!
//! A call already in progress is **not** interrupted, so joining an actor can
//! wait for one model call, bounded by the HTTP client's request timeout.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Instant;

use crossbeam_channel::{Receiver, Sender, TryRecvError, bounded, select, unbounded};
use serde::Serialize;
use serde_json::Value;

use super::contract::{Action, Effect, Observation, Policy, Reminder, Step};
use super::router::{RouteError, Router};
use super::timer::{Reminders, Timer};
use crate::clock::Clock;
use crate::log::{
    ActionRecord, ControlRecord, CycleRecord, Key, ObservationRecord, Record, RewardRecord,
    UndeliveredRecord, UnsentRecord,
};
use crate::message::{ActorId, Control, Message, Payload};

/// Why an actor's thread stopped before it was told to.
///
/// Each is a bug somewhere: in the actor's own handler, for a refused send, or
/// in the episode, for a channel that went away while the actor still needed
/// it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActorError {
    /// A record could not be sent: the log writer has gone away.
    WriterClosed,
    /// The router refused something this actor sent. That is a bug in this
    /// actor's handler; see [`RouteError`].
    Refused(RouteError),
    /// A reward this actor assigned would not serialize, so there was no
    /// record to write for the actor named.
    ///
    /// The reward type is the game's, and one the log cannot hold makes every
    /// reward of that type unloggable, so the actor fails rather than
    /// finishing with a log that is silently missing them.
    Unserializable(ActorId),
    /// A reminder could not be handed back to the perception thread, which has
    /// already ended.
    TimerClosed,
}

impl fmt::Display for ActorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WriterClosed => f.write_str("the log writer has gone away"),
            Self::Refused(error) => write!(f, "the router refused a send: {error}"),
            Self::Unserializable(who) => {
                write!(f, "the reward for {who} could not be serialized")
            }
            Self::TimerClosed => f.write_str("the reminder timer has gone away"),
        }
    }
}

impl std::error::Error for ActorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Refused(error) => Some(error),
            Self::WriterClosed | Self::Unserializable(_) | Self::TimerClosed => None,
        }
    }
}

/// What the perception thread hands the handler thread.
///
/// A start and an observation are two shapes of one thing — a handler call —
/// so they travel on one channel and the handler thread makes one call per
/// item it receives, which is what keeps `start` ahead of the first
/// observation without a second channel to race against.
enum Call<P: Payload> {
    /// The actor has been started: call the hook, with the episode's clock.
    Start,
    /// Something arrived: call the handler with it.
    Observed(Observation<P>),
}

/// One of an actor's threads reporting that it has ended: who it was, and
/// whether the actor had been stopped when it did.
///
/// The second half is how an episode tells a run that ended the way it was
/// meant to from one where an actor left unbidden. The episode does not see the
/// environment's commands, so it cannot know who was stopped; the actor's own
/// flag is the one place that knows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ended {
    /// The actor whose thread ended.
    pub who: ActorId,
    /// Whether it had been stopped. `false` is a departure.
    pub stopped: bool,
}

/// Reports that a thread has ended, whatever ended it.
///
/// It is a guard rather than a line at the end of the thread's body because a
/// panic skips the body's end, and an episode waiting on a report that a
/// panicking thread never sent would hang until its time limit. `Drop` runs on
/// the way out of a panic, so this reports either way.
///
/// Nobody listening is fine: the episode has already given up.
struct Reporting {
    id: ActorId,
    stopped: Arc<AtomicBool>,
    done: Sender<Ended>,
}

impl Drop for Reporting {
    fn drop(&mut self) {
        let _ = self.done.send(Ended {
            who: self.id.clone(),
            stopped: self.stopped.load(Ordering::Acquire),
        });
    }
}

/// A reminder on its way from the handler thread back to the perception
/// thread's timer, already numbered.
struct Held<P: Payload> {
    seq: u64,
    reminder: Reminder<P>,
}

/// The per-actor plumbing: its id, the router, where its records go, and the
/// next number in its message sequence.
///
/// **No application code sees one.** It is the handler thread's own state, and
/// the reason a handler returns actions rather than calling `send` on
/// something: a function from an observation to actions is the shape that
/// makes this a reinforcement learning system, and side effects through a
/// handle obscure it (ADR-0016).
pub struct Context<P: Payload> {
    id: ActorId,
    router: Arc<Router<P>>,
    records: Sender<Record<P>>,
    /// The next number in this actor's message sequence.
    ///
    /// One per message, so the numbers are dense over everything this actor
    /// sent, reminders and relays included: a message an actor sends is its
    /// own however it came by what it carries (ADR-0017).
    next_seq: u64,
    /// Whether the actor has been stopped. Set by the perception thread, read
    /// by the handler thread between every yielded item.
    stopped: Arc<AtomicBool>,
}

impl<P: Payload> fmt::Debug for Context<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Context")
            .field("id", &self.id)
            .field("next_seq", &self.next_seq)
            .finish_non_exhaustive()
    }
}

impl<P: Payload> Context<P> {
    /// This actor's id.
    #[must_use]
    pub const fn id(&self) -> &ActorId {
        &self.id
    }

    /// Whether this actor has been stopped, so that what its handler yields
    /// from here on is logged as unsent and carried nowhere.
    fn is_stopped(&self) -> bool {
        self.stopped.load(Ordering::Acquire)
    }

    /// The next number in this actor's sequence.
    fn stamp(&mut self) -> u64 {
        let seq = self.next_seq;
        self.next_seq += 1;
        seq
    }

    fn send_record(&self, record: Record<P>) -> Result<(), ActorError> {
        self.records.send(record).or(Err(ActorError::WriterClosed))
    }
}

/// An actor whose channels exist but whose threads do not.
///
/// An episode builds every inbox and control channel before anybody runs, so
/// that the [`Router`] every handler thread shares is complete before the
/// first message can be sent. This is what it holds in between.
pub struct UnstartedActor<P: Payload> {
    id: ActorId,
    inbox: Receiver<Message<P>>,
    /// The actor's own end of its inbox, which is where its perception thread
    /// puts a reminder that has come due, so that the reminder queues behind
    /// whatever was already waiting.
    remind: Sender<Message<P>>,
    controls: Receiver<Control>,
    control: Sender<Control>,
}

impl<P: Payload> fmt::Debug for UnstartedActor<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UnstartedActor")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<P: Payload> UnstartedActor<P> {
    /// An actor's channels, and the two senders an episode's router is built
    /// from.
    #[must_use]
    pub fn new(id: impl Into<ActorId>) -> (Self, Sender<Message<P>>, Sender<Control>) {
        let id = id.into();
        let (inbox, heard) = unbounded();
        let (control, told) = unbounded();
        (
            Self {
                id,
                inbox: heard,
                remind: inbox.clone(),
                controls: told,
                control: control.clone(),
            },
            inbox,
            control,
        )
    }

    /// This actor's id.
    #[must_use]
    pub const fn id(&self) -> &ActorId {
        &self.id
    }

    /// The sender for this actor's control channel, which is how an episode
    /// stops an actor it has not started yet.
    #[must_use]
    pub const fn control(&self) -> &Sender<Control> {
        &self.control
    }

    /// The receiving end of this actor's control channel, which is what its
    /// perception thread will wait on.
    #[must_use]
    pub const fn controls(&self) -> &Receiver<Control> {
        &self.controls
    }
}

/// A running actor: its id, the sender for its control channel, and the join
/// handles of its two threads.
///
/// [`join`](Actor::join) waits for both and gives back the handler.
pub struct Actor<H> {
    id: ActorId,
    control: Sender<Control>,
    perception: JoinHandle<Result<(), ActorError>>,
    handling: JoinHandle<Result<H, ActorError>>,
}

impl<H> fmt::Debug for Actor<H> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Actor")
            .field("id", &self.id)
            .finish_non_exhaustive()
    }
}

impl<H> Actor<H> {
    /// The actor's id.
    #[must_use]
    pub const fn id(&self) -> &ActorId {
        &self.id
    }

    /// The sender for this actor's control channel, which is how an episode
    /// stops it.
    #[must_use]
    pub const fn control(&self) -> &Sender<Control> {
        &self.control
    }

    /// Waits for both threads and gives back the handler.
    ///
    /// The handler thread ends once its current call returns, so this can wait
    /// for one model call.
    ///
    /// # Errors
    ///
    /// Whichever thread failed, the handler thread's error first: it is the
    /// one that ran the game.
    ///
    /// # Panics
    ///
    /// If either thread panicked, the panic is propagated to the caller.
    pub fn join(self) -> Result<H, ActorError> {
        let decided = self
            .handling
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
        let perceived = self
            .perception
            .join()
            .unwrap_or_else(|payload| std::panic::resume_unwind(payload));
        // The handler's error first: it is the one that ran the game, and a
        // perception thread that failed because the handler had already gone
        // would otherwise mask the reason.
        let handler = decided?;
        perceived?;
        Ok(handler)
    }
}

/// How an actor is told to carry out what its handler yields, which is the one
/// thing an agent and an environment do differently.
///
/// An agent's [`Action`]s and an environment's [`Effect`]s go through the same
/// handler thread; what differs is that only an environment can produce a
/// command or a reward. This trait is that difference, written once so that
/// the thread is written once.
trait Carry<P: Payload> {
    /// Carries out one item, or logs it as unsent if the actor has stopped.
    fn carry(self, context: &mut Context<P>, held: &Sender<Held<P>>) -> Result<(), ActorError>;
}

impl<P: Payload> Carry<P> for Action<P> {
    fn carry(self, context: &mut Context<P>, held: &Sender<Held<P>>) -> Result<(), ActorError> {
        act(self, context, held)
    }
}

impl<W: Serialize, P: Payload> Carry<P> for Effect<W, P> {
    fn carry(self, context: &mut Context<P>, held: &Sender<Held<P>>) -> Result<(), ActorError> {
        match self {
            Self::Act(action) => act(action, context, held),
            Self::Command { to, control } => command(&to, control, context),
            Self::Reward { to, reward } => pay(&to, &reward, context),
        }
    }
}

/// Carries out one action: a send through the router, or a reminder back to
/// the perception thread's timer.
///
/// Every message is numbered here, as it is carried out, whether or not it
/// goes anywhere: an actor that has been stopped still decided, and the number
/// it decided under is what the unsent record carries, so the numbers stay
/// dense over everything the handler yielded.
fn act<P: Payload>(
    action: Action<P>,
    context: &mut Context<P>,
    held: &Sender<Held<P>>,
) -> Result<(), ActorError> {
    let seq = context.stamp();
    match action {
        Action::Send { to, payload } => {
            let message = Message::new(context.id.clone(), to, seq, payload);
            let key = Key::of(&message);
            let t = Instant::now();
            if context.is_stopped() {
                // The actor was stopped while the call that yielded this was
                // running. The decision is logged, so the log says what the
                // handler decided; nothing is carried out.
                return context.send_record(
                    UnsentRecord {
                        agent: context.id.clone(),
                        t,
                        key,
                        message,
                    }
                    .into(),
                );
            }
            context.send_record(
                ActionRecord {
                    agent: context.id.clone(),
                    t,
                    key,
                    message: message.clone(),
                }
                .into(),
            )?;
            context.router.route(&message).map_err(ActorError::Refused)
        }
        Action::Remind(reminder) => {
            if context.is_stopped() {
                // A reminder yielded too late brings nothing back, so it is
                // unsent in exactly the sense a send is: the message it would
                // have become never travels.
                let message = reminder_message(&context.id, seq, reminder.payload);
                let key = Key::of(&message);
                return context.send_record(
                    UnsentRecord {
                        agent: context.id.clone(),
                        t: Instant::now(),
                        key,
                        message,
                    }
                    .into(),
                );
            }
            // A reminder is not logged where it is set. It is logged when it
            // arrives, as the observation it becomes, because that is the one
            // instant about it that means anything to the actor.
            held.send(Held { seq, reminder })
                .map_err(|_| ActorError::TimerClosed)
        }
    }
}

/// Carries out one command: a control to each of its recipients' control
/// channels.
///
/// A control is not a message and is not numbered: nobody numbered it, and
/// there is nothing to join it to but the actor and the time.
fn command<P: Payload>(
    to: &[ActorId],
    control: Control,
    context: &mut Context<P>,
) -> Result<(), ActorError> {
    if context.is_stopped() {
        // A command yielded after the stop carries nowhere, and there is no
        // unsent record for one: an unsent record names a message, and a
        // control is not a message. What the log has is the stop that
        // preempted it.
        return Ok(());
    }
    context
        .router
        .command(&context.id, to, control)
        .map_err(ActorError::Refused)
}

/// Carries out one reward: a record, and nothing sent.
///
/// The reward is serialized here, where it is assigned, and carried as JSON
/// from there, which is what keeps the reward type off every type in
/// [`log`](crate::log).
fn pay<W: Serialize, P: Payload>(
    to: &ActorId,
    reward: &W,
    context: &mut Context<P>,
) -> Result<(), ActorError> {
    if context.is_stopped() {
        return Ok(());
    }
    // Whom an environment may reward is the roster's question, and the roster
    // is settled before any thread is spawned.
    context
        .router
        .rewardable(&context.id, to)
        .map_err(ActorError::Refused)?;
    let value: Value =
        serde_json::to_value(reward).map_err(|_| ActorError::Unserializable(to.clone()))?;
    context.send_record(
        RewardRecord {
            agent: to.clone(),
            t: Instant::now(),
            value,
        }
        .into(),
    )
}

/// The perception thread's state.
struct Perception<P: Payload> {
    id: ActorId,
    inbox: Receiver<Message<P>>,
    /// This actor's own end of its inbox, which is where a due reminder goes so
    /// that it queues behind whatever was already waiting.
    remind: Sender<Message<P>>,
    controls: Receiver<Control>,
    /// Reminders the handler thread has set and not yet had back.
    requests: Receiver<Held<P>>,
    reminders: Reminders<P>,
    calls: Sender<Call<P>>,
    /// The one call taken off the inbox and not yet handed over. **At most one
    /// at a time**, which is what keeps the inbox the queue a `Stop` preempts:
    /// perception stays one message ahead of the handler and no further.
    pending: Option<Pending<P>>,
    records: Sender<Record<P>>,
    stopped: Arc<AtomicBool>,
}

/// A message taken off the inbox and stamped, waiting for the handler thread.
///
/// Its record is not written until it has been handed over, so a message a
/// `Stop` preempted is logged as undelivered rather than as something observed.
/// The stamp is the instant it arrived, which is a fact about the message and
/// not about when the handler got to it.
///
/// One copy of the message is kept here and one is moved into the call, which is
/// the minimum: both the record written on a handoff and the undelivered record
/// written on a `Stop` name the message, and only one of the two can be the copy
/// the handler was given.
struct Pending<P: Payload> {
    at: Instant,
    message: Message<P>,
}

/// What ended one turn of the perception loop.
enum Sensed<P: Payload> {
    /// A control arrived.
    Told(Control),
    /// A message arrived on the inbox.
    Heard(Message<P>),
    /// The pending call reached the handler thread, so its observation record
    /// is due.
    Handed,
    /// The handler thread set a reminder.
    Setting(Held<P>),
    /// The reminder timer fired.
    Due,
    /// Every channel this thread waits on has closed, which can only be the
    /// episode dropping its senders, so there is nothing left to perceive.
    Deaf,
}

/// The message a reminder becomes: **the actor's own, addressed to itself**.
///
/// The one place that rule is written, because it is the one exception to the
/// router's no-loopback rule and three paths need it — a reminder set, a
/// reminder that came due, and a reminder a `Stop` preempted. It never goes
/// through the router: a reminder is the one way a message reaches the actor
/// that sent it (ADR-0016).
///
/// `seq` is the number the reminder was given when it was set, not a new one: a
/// reminder is numbered where every other message is, at the moment its actor
/// decided to send it.
fn reminder_message<P: Payload>(id: &ActorId, seq: u64, payload: P) -> Message<P> {
    Message::new(id.clone(), [id.clone()], seq, payload)
}

/// What a fire on the reminder timer means: what is due, or that nobody is
/// driving the timer any more.
///
/// A held timer whose sender was dropped stops firing; everything else still
/// arrives, so this is not an error.
const fn sensed_timer<P: Payload>(fired: bool) -> Sensed<P> {
    if fired { Sensed::Due } else { Sensed::Deaf }
}

impl<P: Payload> Perception<P> {
    /// Receives, stamps, logs and forwards, until it is stopped or every
    /// channel it waits on has closed.
    ///
    /// **At most one call is pending at a time.** The channel to the handler
    /// thread is a rendezvous, so a message only leaves the inbox when the
    /// handler is ready for it, and what a slow handler leaves behind piles up
    /// in the inbox — which is exactly what a `Stop` preempts. Perception does
    /// not *block* on the handoff, though: a pending call is offered on a send
    /// arm of the same `select!` that watches for controls, so a `Stop` reaches
    /// a thread whose handler has been thinking for a minute at once.
    fn run(mut self) -> Result<(), ActorError> {
        loop {
            match self.sense() {
                Sensed::Told(Control::Start) => {
                    self.record(
                        ControlRecord {
                            agent: self.id.clone(),
                            t: Instant::now(),
                            control: Control::Start,
                        }
                        .into(),
                    )?;
                    // A start goes ahead of any message already in hand, which
                    // it must: the hook runs before the first observation.
                    // `send` here can only wait for the handler thread to reach
                    // its first receive, which is immediate.
                    if self.calls.send(Call::Start).is_err() {
                        return self.deafened();
                    }
                }
                Sensed::Told(Control::Stop) => return self.stop(),
                Sensed::Heard(message) => {
                    self.pending = Some(Pending {
                        at: Instant::now(),
                        message,
                    });
                }
                Sensed::Handed => {
                    let Pending { at, message } = self
                        .pending
                        .take()
                        .expect("only a pending call can have been handed over");
                    self.record(
                        ObservationRecord {
                            agent: self.id.clone(),
                            t: at,
                            key: Key::of(&message),
                            message,
                        }
                        .into(),
                    )?;
                }
                Sensed::Setting(Held { seq, reminder }) => {
                    self.reminders
                        .hold(reminder.deadline, seq, reminder.payload);
                }
                Sensed::Due => {
                    // A reminder arrives as an ordinary message from this actor
                    // to itself, carrying the number it was given when it was
                    // set (ADR-0016). It never went through the router, which
                    // refuses loopback; a reminder is the one way a message
                    // reaches the actor that sent it.
                    //
                    // What is due goes on the inbox rather than straight to the
                    // handler, so that it queues behind whatever was already
                    // waiting and is preempted by a `Stop` like anything else.
                    for (seq, payload) in self.reminders.due(Instant::now()) {
                        let reminder = reminder_message(&self.id, seq, payload);
                        if self.remind.send(reminder).is_err() {
                            return self.deafened();
                        }
                    }
                }
                Sensed::Deaf => return self.deafened(),
            }
        }
    }

    /// Waits for the next thing to perceive, **checking the control channel
    /// first**, offering the pending call to the handler thread while it waits.
    ///
    /// `select!` picks at random among ready channels, so a control that is
    /// waiting would otherwise take its chances against a full inbox. A `Stop`
    /// preempts everything, which it cannot do from a coin toss.
    ///
    /// While a call is pending the inbox is not read, which is what keeps the
    /// inbox the queue a `Stop` preempts: perception stays one message ahead of
    /// the handler and no further. The handoff itself costs nothing while the
    /// handler is busy, because it is a send arm of this same `select!` rather
    /// than a blocking send: a `Stop` reaches an actor whose handler has been
    /// thinking for a minute at once.
    ///
    /// The pending message stays in its field throughout. `select!`'s send arm
    /// takes its value by move, so what is offered is a copy; the field's copy
    /// is what the observation record is written from on
    /// [`Handed`](Sensed::Handed), and what the undelivered record is written
    /// from if a `Stop` arrives first.
    fn sense(&mut self) -> Sensed<P> {
        match self.controls.try_recv() {
            Ok(control) => return Sensed::Told(control),
            // The episode holds every control sender until it has joined, so a
            // disconnected control channel means the episode is gone.
            Err(TryRecvError::Disconnected) => return Sensed::Deaf,
            Err(TryRecvError::Empty) => {}
        }
        let Some(Pending { at, message }) = &self.pending else {
            // Nothing in hand, so nothing is offered and the inbox is read.
            return select! {
                recv(self.controls) -> told => told.map_or(Sensed::Deaf, Sensed::Told),
                recv(self.inbox) -> heard => heard.map_or(Sensed::Deaf, Sensed::Heard),
                recv(self.requests) -> set => set.map_or(Sensed::Deaf, Sensed::Setting),
                recv(self.reminders.armed()) -> fired => sensed_timer(fired.is_ok()),
            };
        };
        let offered = Call::Observed(Observation {
            at: *at,
            message: message.clone(),
        });
        select! {
            recv(self.controls) -> told => told.map_or(Sensed::Deaf, Sensed::Told),
            recv(self.requests) -> set => set.map_or(Sensed::Deaf, Sensed::Setting),
            recv(self.reminders.armed()) -> fired => sensed_timer(fired.is_ok()),
            send(self.calls, offered) -> sent => if sent.is_ok() {
                Sensed::Handed
            } else {
                Sensed::Deaf
            },
        }
    }

    /// Carries out a `Stop`: logs whatever is still in hand or in the inbox,
    /// and every held reminder, as undelivered; sets the stopped flag; and
    /// drops the call channel.
    ///
    /// The flag is set **before** the undelivered records are written, so a
    /// handler call in progress cannot have anything it yields afterwards
    /// carried out. The records come after because the log's order within one
    /// actor is the order it wrote them, and the stop is what explains them.
    fn stop(self) -> Result<(), ActorError> {
        let t = Instant::now();
        self.record(
            ControlRecord {
                agent: self.id.clone(),
                t,
                control: Control::Stop,
            }
            .into(),
        )?;
        self.stopped.store(true, Ordering::Release);
        self.abandon(t)
    }

    /// Logs everything this actor will never observe, and drops the channel
    /// the handler thread waits on.
    ///
    /// `at` is the instant the perception thread gave up, which every
    /// undelivered record carries. The one exception is the message already in
    /// hand, which was stamped when it arrived: that instant is a fact about
    /// the message and not about the giving up.
    fn abandon(self, at: Instant) -> Result<(), ActorError> {
        let Self {
            id,
            inbox,
            requests,
            mut reminders,
            calls,
            pending,
            records,
            ..
        } = self;
        let write = |record: Record<P>| records.send(record).or(Err(ActorError::WriterClosed));
        // Dropped before the records are written so that the handler thread
        // ends the moment its current call returns rather than waiting for this
        // thread to finish writing.
        drop(calls);
        if let Some(Pending { at, message }) = pending {
            write(
                UndeliveredRecord {
                    agent: id.clone(),
                    t: at,
                    key: Key::of(&message),
                    message,
                }
                .into(),
            )?;
        }
        // `try_iter` takes what is there now: the senders are still alive, so
        // iterating the channel would wait for the episode to drop them.
        let waiting: Vec<Message<P>> = inbox.try_iter().collect();
        // A reminder still on its way from the handler thread is one this actor
        // set and will never observe, exactly like one already held, so the two
        // are logged together. Draining the requests first is also what makes
        // the set-then-stop race decidable: a reminder yielded before the stop
        // is accounted for whichever side of the handoff it was on.
        let in_flight: Vec<(u64, P)> = requests
            .try_iter()
            .map(|Held { seq, reminder }| (seq, reminder.payload))
            .collect();
        let held = reminders
            .drain()
            .into_iter()
            .chain(in_flight)
            .map(|(seq, payload)| reminder_message(&id, seq, payload));
        for message in waiting.into_iter().chain(held) {
            write(
                UndeliveredRecord {
                    agent: id.clone(),
                    t: at,
                    key: Key::of(&message),
                    message,
                }
                .into(),
            )?;
        }
        Ok(())
    }

    /// Ends because every channel closed rather than because of a `Stop`.
    ///
    /// An actor whose episode dropped its senders was never told to stop, so
    /// nothing sets its stopped flag; what it had in hand is still undelivered,
    /// and the log says so.
    fn deafened(self) -> Result<(), ActorError> {
        let at = Instant::now();
        self.abandon(at)
    }

    fn record(&self, record: Record<P>) -> Result<(), ActorError> {
        self.records.send(record).or(Err(ActorError::WriterClosed))
    }
}

/// Runs one handler call and closes it with a cycle record.
///
/// Each item is carried out as the iterator yields it, so a lazy iterator's
/// first send reaches its recipient before the iterator's next item is
/// produced. That is why the cycle record is written last and may therefore
/// appear in the log *after* the actions of the call it closes.
fn called<P: Payload, I, T>(
    context: &mut Context<P>,
    held: &Sender<Held<P>>,
    t_start: Instant,
    observed: Option<Key>,
    items: I,
) -> Result<(), ActorError>
where
    I: IntoIterator<Item = T>,
    T: Carry<P>,
{
    for item in items {
        item.carry(context, held)?;
    }
    context.send_record(
        CycleRecord {
            agent: context.id.clone(),
            t_start,
            t_stop: Instant::now(),
            woken: None,
            observed,
        }
        .into(),
    )
}

/// Starts an agent's two threads.
///
/// `clock` is the episode's, handed to the handler's `start` hook so that a
/// handler measuring time from the start of the game is on the log's timeline.
///
/// It is crate-private, and that is what makes the shared origin structural:
/// [`Episode`](super::Episode) is the only thing that can start an actor, and it
/// starts the clock itself, so there is no way to hand one actor a different
/// origin from another or from the log writer (ADR-0017).
///
/// # Panics
///
/// If the operating system refuses to create a thread.
pub(super) fn spawn_agent<P, H>(
    unstarted: UnstartedActor<P>,
    handler: H,
    router: Arc<Router<P>>,
    records: Sender<Record<P>>,
    clock: Clock,
    timer: Timer,
    done: Sender<Ended>,
) -> Actor<H>
where
    P: Payload,
    H: Policy<P> + Send + 'static,
{
    spawn(
        unstarted,
        router,
        records,
        clock,
        timer,
        done,
        handler,
        |handler, context, held, clock| {
            let t_start = Instant::now();
            let items = handler.start(clock);
            called(context, held, t_start, None, items)
        },
        |handler, context, held, observation| {
            let t_start = observation.at;
            let observed = Key::of(&observation.message);
            let items = handler.policy(observation);
            called(context, held, t_start, Some(observed), items)
        },
    )
}

/// Starts an environment's two threads. The difference from [`spawn_agent`] is
/// only which trait's method is called, and it is crate-private for the same
/// reason.
///
/// # Panics
///
/// If the operating system refuses to create a thread.
pub(super) fn spawn_environment<W, P, H>(
    unstarted: UnstartedActor<P>,
    handler: H,
    router: Arc<Router<P>>,
    records: Sender<Record<P>>,
    clock: Clock,
    timer: Timer,
    done: Sender<Ended>,
) -> Actor<H>
where
    W: Serialize + Send + 'static,
    P: Payload,
    H: Step<W, P> + Send + 'static,
{
    spawn(
        unstarted,
        router,
        records,
        clock,
        timer,
        done,
        handler,
        |handler, context, held, clock| {
            let t_start = Instant::now();
            let items = Step::<W, P>::start(handler, clock);
            called(context, held, t_start, None, items)
        },
        |handler, context, held, observation| {
            let t_start = observation.at;
            let observed = Key::of(&observation.message);
            let items = handler.step(observation);
            called(context, held, t_start, Some(observed), items)
        },
    )
}

/// Starts both threads of one actor, given the two closures that make its
/// handler calls.
///
/// The closures are what the two roles differ by, and they are taken rather
/// than dispatched on because [`Policy`] and [`Step`] return
/// `impl IntoIterator` and so cannot be used as `dyn`. Nothing needs them to
/// be: this is generic over the handler, and an episode erases the type into
/// a boxed closure that calls this.
#[allow(clippy::too_many_arguments)]
fn spawn<P, H, S, C>(
    unstarted: UnstartedActor<P>,
    router: Arc<Router<P>>,
    records: Sender<Record<P>>,
    clock: Clock,
    timer: Timer,
    done: Sender<Ended>,
    mut handler: H,
    mut opening: S,
    mut calling: C,
) -> Actor<H>
where
    P: Payload,
    H: Send + 'static,
    S: FnMut(&mut H, &mut Context<P>, &Sender<Held<P>>, Clock) -> Result<(), ActorError>
        + Send
        + 'static,
    C: FnMut(&mut H, &mut Context<P>, &Sender<Held<P>>, Observation<P>) -> Result<(), ActorError>
        + Send
        + 'static,
{
    let UnstartedActor {
        id,
        inbox,
        remind,
        controls,
        control,
    } = unstarted;
    let stopped = Arc::new(AtomicBool::new(false));
    let (calls, called) = bounded(0);
    let (held, requests) = unbounded();

    let perception = Perception {
        id: id.clone(),
        inbox,
        remind,
        controls,
        requests,
        reminders: Reminders::new(timer),
        calls,
        pending: None,
        records: records.clone(),
        stopped: Arc::clone(&stopped),
    };
    let perceiving = {
        let (id, done, flag) = (id.clone(), done.clone(), Arc::clone(&stopped));
        thread::Builder::new()
            .name(format!("{id} perceiving"))
            .spawn(move || {
                // Every thread reports when it ends, which is how an episode
                // tells a run that finished from one that has hung.
                let _reporting = Reporting {
                    id,
                    stopped: flag,
                    done,
                };
                perception.run()
            })
            .expect("failed to spawn a perception thread")
    };

    let handling = {
        let id = id.clone();
        let flag = Arc::clone(&stopped);
        let mut context = Context {
            id: id.clone(),
            router,
            records,
            next_seq: 0,
            stopped,
        };
        thread::Builder::new()
            .name(format!("{id} handling"))
            .spawn(move || {
                let _reporting = Reporting {
                    id,
                    stopped: flag,
                    done,
                };
                for call in called {
                    match call {
                        Call::Start => opening(&mut handler, &mut context, &held, clock)?,
                        Call::Observed(observation) => {
                            calling(&mut handler, &mut context, &held, observation)?;
                        }
                    }
                }
                Ok(handler)
            })
            .expect("failed to spawn a handler thread")
    };

    Actor {
        id,
        control,
        perception: perceiving,
        handling,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::time::Duration;

    use crossbeam_channel::RecvTimeoutError;
    use serde_json::json;

    use super::*;
    use crate::actor::router::Seat;
    use crate::log::Writer;
    use crate::testing::{Shared, TestPayload, joined, parse_lines, recording};

    /// How long a test waits on a channel before giving up. Generous, because
    /// a loaded machine can stall a thread for a while and a test that fails
    /// on load is worse than one that takes a moment.
    const PATIENCE: Duration = Duration::from_secs(10);

    const ENVIRONMENT: &str = "environment";

    /// Everything a test needs to drive one actor: its channels, the records
    /// it writes, and where messages it sends land.
    struct Bench<P: Payload> {
        inbox: Sender<Message<P>>,
        peer: Receiver<Message<P>>,
        told: Receiver<Control>,
        log: Shared,
        writer: Writer,
        records: Sender<Record<P>>,
    }

    /// A router over one actor under test and a peer it may talk to.
    fn wired<P: Payload>(
        who: &str,
    ) -> (UnstartedActor<P>, Arc<Router<P>>, Bench<P>, Sender<Ended>) {
        let (unstarted, inbox, control) = UnstartedActor::new(who);
        let (peer_inbox, peer) = unbounded();
        let (peer_control, told) = unbounded();
        let seats = BTreeMap::from([
            (
                ActorId::new(who),
                Seat {
                    inbox: inbox.clone(),
                    control: control.clone(),
                },
            ),
            (
                ActorId::new("peer"),
                Seat {
                    inbox: peer_inbox,
                    control: peer_control,
                },
            ),
        ]);
        // The actor under test is the environment wherever a test needs one
        // to command, and its own name otherwise; a router names one.
        let router = Arc::new(Router::new(seats, ActorId::new(who)));
        let (records, writer, log) = recording(Clock::start());
        // The reports these actors send go nowhere: what an episode does with
        // them is `episode`'s business, and these tests drive one actor.
        let (report, _done) = unbounded::<Ended>();
        (
            unstarted,
            router,
            Bench {
                inbox,
                peer,
                told,
                log,
                writer,
                records: records.clone(),
            },
            report,
        )
    }

    impl<P: Payload> Bench<P> {
        /// The log, parsed, once every sender has been dropped.
        fn lines(self) -> Vec<Value> {
            drop(self.records);
            parse_lines(&joined(self.writer, &self.log))
        }

        /// Waits for a message the actor sent to its peer.
        fn heard(&self) -> Message<P> {
            self.peer.recv_timeout(PATIENCE).expect("the peer heard it")
        }
    }

    /// A payload as a step, for the tests that only care about the number.
    fn step(n: u64) -> TestPayload {
        TestPayload::Step(n)
    }

    /// An agent that answers each step by sending the next one to its peer.
    struct Passing;

    impl Policy<TestPayload> for Passing {
        fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
            [Action::to(["peer"], step(0))]
        }

        fn policy(
            &mut self,
            observation: Observation<TestPayload>,
        ) -> impl IntoIterator<Item = Action<TestPayload>> {
            let n = observation.message.payload.step();
            [Action::to(["peer"], step(n + 1))]
        }
    }

    #[test]
    fn a_start_runs_the_hook_and_an_observation_runs_the_handler() {
        let (unstarted, router, bench, report) = wired("a");
        let actor = spawn_agent(
            unstarted,
            Passing,
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(bench.heard().payload, step(0));
        bench
            .inbox
            .send(Message::new("peer", ["a"], 0, step(4)))
            .unwrap();
        assert_eq!(bench.heard().payload, step(5));
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        let lines = bench.lines();
        // One cycle per call: the start, and the one observation.
        assert_eq!(
            lines.iter().filter(|line| line["type"] == "cycle").count(),
            2
        );
    }

    #[test]
    fn a_cycle_of_the_actor_runtime_says_nothing_about_what_woke_it() {
        // There is nothing to say: a handler thread runs a cycle when an
        // observation reaches it, and a reminder arrives as one of those.
        let (unstarted, router, bench, report) = wired("a");
        let actor = spawn_agent(
            unstarted,
            Passing,
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        bench.heard();
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        for cycle in bench.lines().iter().filter(|line| line["type"] == "cycle") {
            assert!(cycle["woken"].is_null(), "{cycle}");
        }
    }

    #[test]
    fn a_start_hook_is_given_the_episodes_clock() {
        /// An agent that reports the origin it was started with.
        struct Reporting(Sender<Clock>);

        impl Policy<TestPayload> for Reporting {
            fn start(&mut self, clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                self.0.send(clock).unwrap();
                []
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                []
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let (told, clocks) = unbounded();
        let clock = Clock::start();
        let actor = spawn_agent(
            unstarted,
            Reporting(told),
            router,
            bench.records.clone(),
            clock,
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(
            clocks.recv_timeout(PATIENCE).unwrap().origin(),
            clock.origin(),
            "the hook is given the episode's origin, not one of its own"
        );
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        bench.lines();
    }

    #[test]
    fn a_send_that_names_its_sender_fails_the_actor() {
        /// An agent that addresses itself, which the router refuses: a
        /// reminder is the one way an actor reaches itself.
        struct Selfish;

        impl Policy<TestPayload> for Selfish {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                [Action::to(["a"], step(0))]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                []
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let actor = spawn_agent(
            unstarted,
            Selfish,
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(
            actor.join().err(),
            Some(ActorError::Refused(RouteError::Loopback(ActorId::new("a")))),
            "a send that names its sender is a bug in the sending actor"
        );
        bench.lines();
    }

    #[test]
    fn an_empty_recipient_set_logs_an_action_and_delivers_nothing() {
        /// An agent that announces to nobody and then speaks to its peer.
        ///
        /// The second send is what the test waits on: actions are carried out
        /// in the order they are yielded, so hearing it means the first was
        /// already carried out.
        struct Announcing;

        impl Policy<TestPayload> for Announcing {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                [
                    Action::to(Vec::<ActorId>::new(), step(9)),
                    Action::to(["peer"], step(1)),
                ]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                []
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let actor = spawn_agent(
            unstarted,
            Announcing,
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(
            bench.heard().payload,
            step(1),
            "the peer heard the second send and so the first has happened"
        );
        assert_eq!(
            bench.peer.try_recv(),
            Err(TryRecvError::Empty),
            "and nothing else, because the first was addressed to nobody"
        );
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        let lines = bench.lines();
        let actions: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "action")
            .collect();
        assert_eq!(actions.len(), 2, "an action to nobody is still logged");
        assert_eq!(
            actions[0]["message"]["recipients"],
            json!([]),
            "{:?}",
            actions[0]
        );
    }

    #[test]
    fn a_reminder_comes_back_as_a_message_from_the_actor_itself() {
        /// An agent that reminds itself once when it starts and passes on
        /// whatever comes back.
        struct Reminding(Instant);

        impl Policy<TestPayload> for Reminding {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                [Action::remind(self.0, step(7))]
            }

            fn policy(
                &mut self,
                observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                [Action::to(["peer"], observation.message.payload)]
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        // The deadline is an hour away, so only the test's own fire can
        // deliver it: what is being tested is the delivery, not the clock.
        let (timer, fire) = Timer::held();
        let actor = spawn_agent(
            unstarted,
            Reminding(Instant::now() + Duration::from_secs(3600)),
            router,
            bench.records.clone(),
            Clock::start(),
            timer,
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(
            bench.peer.recv_timeout(Duration::from_millis(50)),
            Err(RecvTimeoutError::Timeout),
            "a reminder that has not fired has not arrived"
        );
        fire.send(Instant::now()).unwrap();
        assert_eq!(bench.heard().payload, step(7));
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();

        let lines = bench.lines();
        let observed: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "observation")
            .collect();
        assert_eq!(observed.len(), 1);
        assert_eq!(
            observed[0]["from"],
            json!("a"),
            "a reminder arrives from the actor itself: {}",
            observed[0]
        );
        assert_eq!(
            observed[0]["seq"],
            json!(0),
            "and carries the number it was given when it was set"
        );
    }

    #[test]
    fn stop_preempts_a_full_inbox_and_what_a_handler_yields_after_it() {
        /// An agent whose call blocks until the test lets it finish, then
        /// yields one send. What it yields lands after the stop.
        struct Slow {
            released: Receiver<()>,
            entered: Sender<()>,
        }

        impl Policy<TestPayload> for Slow {
            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                self.entered.send(()).unwrap();
                self.released.recv_timeout(PATIENCE).unwrap();
                [Action::to(["peer"], step(1))]
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let (release, released) = unbounded();
        let (entered, running) = unbounded();
        // A reminder far off, so it is still held when the stop arrives.
        let (timer, _fire) = Timer::held();
        let actor = spawn_agent(
            unstarted,
            Slow { released, entered },
            router,
            bench.records.clone(),
            Clock::start(),
            timer,
            report,
        );
        // One message to occupy the handler, and three behind it that the
        // stop will preempt.
        for seq in 0..4 {
            bench
                .inbox
                .send(Message::new("peer", ["a"], seq, step(seq)))
                .unwrap();
        }
        running.recv_timeout(PATIENCE).expect("the call started");
        actor.control().send(Control::Stop).unwrap();
        // The stop has to have taken effect before the call returns, or what
        // the call yields is sent legitimately and the test is asserting a
        // race. The actor's inbox emptying is the observable proof: the
        // perception thread drains it only when it acts on a stop.
        while !bench.inbox.is_empty() {
            std::thread::yield_now();
        }
        release.send(()).unwrap();
        actor.join().unwrap();
        // The router's senders go with the actor, so the peer's channel is
        // disconnected rather than merely empty; what the test asserts is that
        // nothing came down it.
        assert_eq!(
            bench.peer.try_iter().count(),
            0,
            "nothing yielded after the stop is carried out"
        );

        let lines = bench.lines();
        let kinds = |kind: &str| -> Vec<&Value> {
            lines.iter().filter(|line| line["type"] == kind).collect()
        };
        assert_eq!(
            kinds("observation").len(),
            1,
            "only the message the handler was already called with was observed"
        );
        assert_eq!(
            kinds("undelivered").len(),
            3,
            "the three behind the stop reached nobody: {lines:?}"
        );
        assert_eq!(
            kinds("unsent").len(),
            1,
            "what the call yielded after the stop was carried nowhere"
        );
        assert!(kinds("action").is_empty(), "and no action was sent");
    }

    #[test]
    fn a_held_reminder_is_undelivered_when_the_actor_is_stopped() {
        /// An agent that sets one far-off reminder and then speaks, so that
        /// the test can tell when the reminder has been set.
        struct Waiting(Instant);

        impl Policy<TestPayload> for Waiting {
            fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
                [
                    Action::remind(self.0, step(3)),
                    Action::to(["peer"], step(0)),
                ]
            }

            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                []
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let (timer, _fire) = Timer::held();
        let actor = spawn_agent(
            unstarted,
            Waiting(Instant::now() + Duration::from_secs(3600)),
            router,
            bench.records.clone(),
            Clock::start(),
            timer,
            report,
        );
        actor.control().send(Control::Start).unwrap();
        // The reminder is yielded before the send, and actions are carried out
        // in the order they are yielded, so hearing the send means the
        // reminder is held. There is still the handoff to the perception
        // thread's timer between the two, which `stop` cannot overtake: it
        // drains the request channel the reminder is on before it drains the
        // reminders themselves.
        bench.heard();
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        let lines = bench.lines();
        // Undelivered if the perception thread had it, unsent if the handler
        // thread still did when the flag went up — the two threads race over
        // one handoff and both answers are honest. What must hold is that the
        // reminder is accounted for exactly once, and that it is the actor's
        // own message with the number it was set under.
        let lost: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "undelivered" || line["type"] == "unsent")
            .collect();
        assert_eq!(
            lost.len(),
            1,
            "the reminder is accounted for once: {lines:?}"
        );
        assert_eq!(lost[0]["seq"], json!(0));
        assert_eq!(
            lost[0]["message"]["sender"],
            json!("a"),
            "a reminder is the actor's own message: {}",
            lost[0]
        );
    }

    #[test]
    fn a_lazy_iterator_delivers_each_action_before_producing_the_next() {
        /// An agent whose iterator does slow work between yields: it waits
        /// for the test to say the last one was heard.
        struct Streaming {
            heard: Receiver<()>,
        }

        impl Policy<TestPayload> for Streaming {
            fn policy(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Action<TestPayload>> {
                let heard = self.heard.clone();
                // A lazy iterator: each item after the first is produced only
                // once the test has confirmed the previous one arrived. If the
                // runtime waited for the iterator to finish before sending,
                // this would deadlock.
                (0..3u64).map(move |n| {
                    if n > 0 {
                        heard.recv_timeout(PATIENCE).expect("the last was heard");
                    }
                    Action::to(["peer"], step(n))
                })
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>("a");
        let (confirm, heard) = unbounded();
        let actor = spawn_agent(
            unstarted,
            Streaming { heard },
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        bench
            .inbox
            .send(Message::new("peer", ["a"], 0, step(0)))
            .unwrap();
        for n in 0..3 {
            assert_eq!(
                bench.heard().payload,
                step(n),
                "the iterator's item {n} arrived before the next was produced"
            );
            confirm.send(()).unwrap();
        }
        actor.control().send(Control::Stop).unwrap();
        actor.join().unwrap();
        bench.lines();
    }

    /// An environment that starts the actors it is given, rewards one and
    /// then stops everybody, itself included.
    struct Referee {
        agents: Vec<ActorId>,
    }

    impl Step<i32, TestPayload> for Referee {
        fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
            let mut effects = vec![Effect::command(self.agents.clone(), Control::Start)];
            effects.push(Effect::reward("peer", 1));
            let mut everybody = self.agents.clone();
            everybody.push(ActorId::new(ENVIRONMENT));
            effects.push(Effect::command(everybody, Control::Stop));
            effects
        }

        fn step(
            &mut self,
            _observation: Observation<TestPayload>,
        ) -> impl IntoIterator<Item = Effect<i32, TestPayload>> {
            []
        }
    }

    #[test]
    fn an_environment_commands_rewards_and_stops_itself() {
        let (unstarted, router, bench, report) = wired::<TestPayload>(ENVIRONMENT);
        let actor = spawn_environment::<i32, _, _>(
            unstarted,
            Referee {
                agents: vec![ActorId::new("peer")],
            },
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(bench.told.recv_timeout(PATIENCE).unwrap(), Control::Start);
        assert_eq!(bench.told.recv_timeout(PATIENCE).unwrap(), Control::Stop);
        actor.join().unwrap();
        let lines = bench.lines();
        let rewards: Vec<&Value> = lines
            .iter()
            .filter(|line| line["type"] == "reward")
            .collect();
        assert_eq!(rewards.len(), 1);
        assert_eq!(rewards[0]["agent"], json!("peer"));
        assert_eq!(rewards[0]["value"], json!(1));
    }

    #[test]
    fn a_reward_that_will_not_serialize_fails_the_actor() {
        /// A reward type the log cannot hold: a map with non-string keys is
        /// not JSON.
        #[derive(Clone, Copy)]
        struct Unwritable;

        impl Serialize for Unwritable {
            fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
                Err(serde::ser::Error::custom("no"))
            }
        }

        struct Paying;

        impl Step<Unwritable, TestPayload> for Paying {
            fn start(
                &mut self,
                _clock: Clock,
            ) -> impl IntoIterator<Item = Effect<Unwritable, TestPayload>> {
                [Effect::reward("peer", Unwritable)]
            }

            fn step(
                &mut self,
                _observation: Observation<TestPayload>,
            ) -> impl IntoIterator<Item = Effect<Unwritable, TestPayload>> {
                []
            }
        }

        let (unstarted, router, bench, report) = wired::<TestPayload>(ENVIRONMENT);
        let actor = spawn_environment::<Unwritable, _, _>(
            unstarted,
            Paying,
            router,
            bench.records.clone(),
            Clock::start(),
            Timer::real(),
            report,
        );
        actor.control().send(Control::Start).unwrap();
        assert_eq!(
            actor.join().err(),
            Some(ActorError::Unserializable(ActorId::new("peer")))
        );
        bench.lines();
    }

    #[test]
    fn errors_explain_themselves() {
        assert_eq!(
            ActorError::WriterClosed.to_string(),
            "the log writer has gone away"
        );
        assert_eq!(
            ActorError::Refused(RouteError::Loopback(ActorId::new("a"))).to_string(),
            "the router refused a send: actor a addressed itself"
        );
        assert_eq!(
            ActorError::Unserializable(ActorId::new("a")).to_string(),
            "the reward for a could not be serialized"
        );
        assert_eq!(
            ActorError::TimerClosed.to_string(),
            "the reminder timer has gone away"
        );
    }
}
