//! What an application implements and what it returns: [`Observation`],
//! [`Action`], [`Effect`], [`Reminder`], and the two traits [`Policy`] and
//! [`Step`] (ADR-0016).
//!
//! This is the whole of the framework's surface. A user implements one
//! function, from an observation to actions, and never sees a thread, a
//! channel or a timer.
//!
//! # A handler is called once per observation
//!
//! The runtime never batches, and it does not tell a handler what is waiting
//! behind the current observation. An agent sees its observations one at a
//! time, as a person hears a conversation one utterance at a time, with no
//! view into its own queue.
//!
//! An observation is taken **by value**, and the runtime keeps no history on
//! a handler's behalf. A reactive policy, π(a | o), keeps nothing; a
//! history-dependent one, π(a | h), keeps its own *h*, including the actions
//! it returned and any window, summary or belief state it builds from them.
//!
//! An agent that wants to decide from several observations at once does it
//! with its own state. The usual shape is to wait for a lull: fold each
//! observation in, set a [`Reminder`] a short interval ahead, and decide when
//! one arrives with nothing newer folded since it was set. How an agent
//! batches, and whether it does at all, is its own business.
//!
//! # Actions are carried out as they are yielded
//!
//! Each returns `impl IntoIterator`, and the handler thread carries out each
//! item **as the iterator yields it**, not after the call returns. A handler
//! that returns a `Vec` sends everything at the end; one that returns a lazy
//! iterator can yield an action, do slow work inside `next()`, and yield
//! another. An LLM player streaming its model's response yields each piece of
//! speech as it arrives, and its listeners hear the utterance while the model
//! is still generating it.
//!
//! The iterator borrows the handler, so neither trait is `dyn`-compatible.
//! Nothing requires them to be: an actor's threads are generic over the
//! handler, and an [`Episode`](super::episode::Episode) erases the type into
//! a boxed closure when a handler is added.

use std::time::Instant;

use crate::clock::Clock;
use crate::message::{ActorId, Control, Message, Payload};

/// A message that has arrived, with the instant it arrived.
///
/// The arrival is the one time an actor has about a message, and the one it
/// needs: what latency an actor perceives is when its observations arrived
/// relative to each other and to itself (ADR-0017). When the sender's clock
/// read as it sent is a fact about that clock, and no message carries it.
///
/// It is stamped by the **perception thread**, the moment the message is
/// received, which is the whole point of that thread being separate: a
/// handler blocked in a model call does not delay the stamp of anything said
/// during the call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Observation<P: Payload> {
    /// When this actor received it.
    pub at: Instant,
    /// The message.
    pub message: Message<P>,
}

/// A payload to deliver back to this actor at a deadline: an address in time.
///
/// At the deadline the payload arrives as an **ordinary message from this
/// actor itself**, carrying the sequence number the reminder was given when
/// it was set, and is observed and logged like any other message. There is no
/// wake-up type: the payload says why the reminder was set.
///
/// - **Always self-directed.** Another actor cannot decide when you think; it
///   can only send you a message asking you to, and your handler may answer
///   with a reminder of its own. A reminder has no recipients.
/// - **It is the only way an actor sends itself a message.** The
///   [`Router`](super::router::Router) refuses a send whose recipients
///   include its sender, so there is no loopback anywhere else.
/// - **The deadline is required.** Since a reminder is the only way to send
///   oneself a message, one wanted at once is a reminder whose deadline is
///   now.
/// - **Reminders accumulate.** Each fires once. A handler that has changed
///   its mind ignores a stale reminder when it arrives; nothing is cancelled.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Reminder<P> {
    /// When the payload comes back, as an absolute instant on the process's
    /// monotonic clock — the same clock an [`Observation`]'s `at` is read
    /// from, so a handler may compare them directly.
    pub deadline: Instant,
    /// What comes back. Its meaning belongs to the game.
    pub payload: P,
}

impl<P> Reminder<P> {
    /// A reminder that brings `payload` back at `deadline`.
    pub const fn new(deadline: Instant, payload: P) -> Self {
        Self { deadline, payload }
    }
}

/// What an agent does: send a message, or set a reminder.
///
/// An action is **intent**: recipients and a payload, with no sender,
/// sequence number or time. The runtime supplies those when it turns the
/// action into a [`Message`] and logs it. The facts live in the log.
///
/// **Every message an actor sends is its own.** There is no way to name
/// another sender, so no record ever claims a sender other than the actor
/// that wrote it. An actor passing another's message on sends a message of
/// its own whose payload carries an [`Envelope`](crate::Envelope) of what it
/// received (ADR-0017).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action<P> {
    /// Say something to some other actors.
    Send {
        /// The actors to say it to.
        ///
        /// It **may be empty**: an action need not be directed at anyone, and
        /// one addressed to nobody is still logged as its sender's action and
        /// delivered to nobody.
        ///
        /// It may **not** include the sender. There is no loopback; a
        /// [`Remind`](Self::Remind) is how an actor reaches itself, and a
        /// send that names its sender fails that actor's thread.
        to: Vec<ActorId>,
        /// What to say.
        payload: P,
    },
    /// Bring a payload back to this actor at a deadline.
    Remind(Reminder<P>),
}

impl<P> Action<P> {
    /// A send to the given recipients.
    pub fn to<I, A>(to: I, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self::Send {
            to: to.into_iter().map(Into::into).collect(),
            payload,
        }
    }

    /// A reminder bringing `payload` back at `deadline`.
    pub const fn remind(deadline: Instant, payload: P) -> Self {
        Self::Remind(Reminder::new(deadline, payload))
    }
}

/// What an environment does: an action like any agent's, a control for some
/// of the actors, or a reward for one of them.
///
/// The two type parameters are the two things a game names: `W`, the type its
/// rewards are in, and `P`, what its messages carry. The reward type appears
/// here and nowhere else in the runtime, because this is the one place a
/// reward is produced: a reward is assigned by the environment and logged, it
/// is never sent, and no actor, message or router holds one (ADR-0016).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect<W, P> {
    /// Say something, or set a reminder, as any agent does.
    Act(Action<P>),
    /// Tell some actors to start or to stop.
    Command {
        /// The actors told. **The environment may name itself**: an episode
        /// ends when the environment stops every actor, itself included.
        to: Vec<ActorId>,
        /// What they are told.
        control: Control,
    },
    /// Record what one actor's behavior was worth.
    ///
    /// Nothing is sent and nobody is told: the reward is written to the log
    /// the instant it is yielded.
    Reward {
        /// The actor rewarded, whose records the reward belongs to.
        to: ActorId,
        /// What its behavior was worth, in the game's own units.
        reward: W,
    },
}

impl<W, P> Effect<W, P> {
    /// A send to the given recipients.
    pub fn to<I, A>(to: I, payload: P) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self::Act(Action::to(to, payload))
    }

    /// A control for the given actors.
    pub fn command<I, A>(to: I, control: Control) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self::Command {
            to: to.into_iter().map(Into::into).collect(),
            control,
        }
    }

    /// A reward of `reward` for `to`.
    pub fn reward(to: impl Into<ActorId>, reward: W) -> Self {
        Self::Reward {
            to: to.into(),
            reward,
        }
    }
}

/// An agent's behavior: a function from one observation to actions.
///
/// The handler thread calls [`policy`](Policy::policy) once per observation,
/// in arrival order, and carries out each action as the iterator yields it.
/// Agent state lives in the implementing type. A handler never sees a
/// [`Control`]: a control is out-of-domain, an instruction about the episode
/// rather than a move within it, and the runtime acts on it itself.
pub trait Policy<P: Payload> {
    /// The agent's opening actions, called once when it receives
    /// [`Control::Start`].
    ///
    /// This is the **one optional hook**. Handlers run only when an
    /// observation arrives, so something must speak first — the moderator
    /// opening night one, the first sender in a Collatz ring — and the
    /// framework cannot construct a game's payload to prompt it.
    ///
    /// `clock` is the **episode's**: the one origin the episode chose before
    /// any actor started, shared by every actor and the log writer
    /// (ADR-0017), so a handler that measures time from the start of the game
    /// is on the log's timeline. Actors are started by the environment's
    /// `Start`, so hooks run a little after the origin; offsets are measured
    /// from the shared origin and not from when a hook ran.
    fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<P>> {
        []
    }

    /// Folds one observation into the agent's state and says what to do.
    ///
    /// One observation, because this is a decision point and what an agent
    /// conditions on at one is an observation, not a pile of them (ADR-0008).
    ///
    /// Nothing interrupts it. An actor does not know it is being stopped, so
    /// a handler is never asked to give up early; what it yields after its
    /// actor has been stopped is logged as unsent and carried nowhere. A
    /// handler that blocks delays its own decisions, and with it the end of
    /// its episode, for as long as it blocks — but not its own perception,
    /// which is another thread.
    fn policy(&mut self, observation: Observation<P>) -> impl IntoIterator<Item = Action<P>>;
}

/// An environment's behavior: a function from one observation to effects.
///
/// It is [`Policy`] with a wider return type, and the runtime runs both
/// through the same two threads. An environment is the one actor that may
/// command and reward, and it is what **ends an episode**: it stops every
/// actor, itself included.
pub trait Step<W, P: Payload> {
    /// The environment's opening effects, called once when it receives
    /// [`Control::Start`].
    ///
    /// The episode starts only the environment, so this is where the agents
    /// are started, with a [`Command`](Effect::Command) of
    /// [`Control::Start`]. `clock` is the episode's; see
    /// [`Policy::start`].
    fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Effect<W, P>> {
        []
    }

    /// Folds one observation into the environment's state and says what to
    /// do.
    fn step(&mut self, observation: Observation<P>) -> impl IntoIterator<Item = Effect<W, P>>;
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::testing::TestPayload;

    /// An agent that answers each step with the next one, and is testable
    /// with no threads at all.
    struct Doubler;

    impl Policy<TestPayload> for Doubler {
        fn start(&mut self, _clock: Clock) -> impl IntoIterator<Item = Action<TestPayload>> {
            [Action::to(["b"], TestPayload::Step(1))]
        }

        fn policy(
            &mut self,
            observation: Observation<TestPayload>,
        ) -> impl IntoIterator<Item = Action<TestPayload>> {
            let n = observation.message.payload.step();
            [Action::to(["b"], TestPayload::Step(n * 2))]
        }
    }

    fn observing(n: u64) -> Observation<TestPayload> {
        Observation {
            at: Instant::now(),
            message: Message::new("b", ["a"], 0, TestPayload::Step(n)),
        }
    }

    #[test]
    fn a_policy_is_a_function_from_an_observation_to_actions() {
        let mut agent = Doubler;
        let sent: Vec<_> = agent.policy(observing(3)).into_iter().collect();
        assert_eq!(sent, [Action::to(["b"], TestPayload::Step(6))]);
    }

    #[test]
    fn a_start_hook_opens_with_actions_of_its_own() {
        let mut agent = Doubler;
        let opened: Vec<_> = agent.start(Clock::start()).into_iter().collect();
        assert_eq!(opened, [Action::to(["b"], TestPayload::Step(1))]);
    }

    #[test]
    fn an_action_may_be_addressed_to_nobody() {
        let announced: Action<TestPayload> =
            Action::to(Vec::<ActorId>::new(), TestPayload::Step(1));
        assert_eq!(
            announced,
            Action::Send {
                to: Vec::new(),
                payload: TestPayload::Step(1)
            }
        );
    }

    #[test]
    fn a_reminder_carries_a_deadline_and_a_payload_and_no_recipients() {
        let deadline = Instant::now() + Duration::from_secs(1);
        let set = Action::remind(deadline, TestPayload::Step(7));
        assert_eq!(
            set,
            Action::Remind(Reminder {
                deadline,
                payload: TestPayload::Step(7)
            })
        );
    }

    #[test]
    fn an_environment_commands_and_rewards_and_an_agent_cannot() {
        // The difference between the two roles is the return type, and this
        // is what it buys: these two effects have no `Action` counterpart,
        // so no agent can construct them.
        let started: Effect<i32, TestPayload> = Effect::command(["a", "b"], Control::Start);
        assert_eq!(
            started,
            Effect::Command {
                to: vec![ActorId::new("a"), ActorId::new("b")],
                control: Control::Start
            }
        );
        let paid: Effect<i32, TestPayload> = Effect::reward("a", 3);
        assert_eq!(
            paid,
            Effect::Reward {
                to: ActorId::new("a"),
                reward: 3
            }
        );
    }

    #[test]
    fn an_environment_may_command_itself() {
        // Which is how an episode ends: the environment stops every actor,
        // itself included.
        let ended: Effect<i32, TestPayload> = Effect::command(["a", "environment"], Control::Stop);
        let Effect::Command { to, .. } = ended else {
            panic!("a command is a command");
        };
        assert!(to.contains(&ActorId::new("environment")));
    }

    /// A payload type a game never names, to show that the runtime's bound
    /// is only what [`Payload`] asks for.
    #[allow(dead_code)]
    fn payloads_are_whatever_satisfies_the_bound<P: Payload>(observation: Observation<P>) -> P {
        observation.message.payload
    }
}
