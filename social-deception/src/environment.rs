//! The environment: the one agent in an episode that controls it.
//!
//! An [`Environment`] is an agent like any other — one thread, one queue,
//! and records of its own — with one power nobody else has. Where an ordinary [`Handler`] returns only
//! [`Action`]s, an environment returns [`Effect`]s, and an `Effect` is
//! either an action or a [`Control`] addressed to some of the agents. That
//! is how an episode starts and how it ends: the episode starts the
//! environment, the environment starts the agents, and the environment
//! stops them when the game it is running is over (ADR-0007).
//!
//! Each episode has exactly one, and a game implements the trait under its
//! own name: Werewolf's is [`Moderator`](crate::werewolf::Moderator).
//!
//! # Rewards are logged where they are decided
//!
//! An environment's third power is the [`Reward`](Effect::Reward). A reward
//! is a single number the environment assigns to one agent, and it is
//! **logged, never sent** (ADR-0007): an agent never needs to observe its
//! reward while acting, because the reward exists for training, which reads
//! the log. So the adapter writes the reward record the instant the handler
//! returns it, routes nothing, and counts nothing: a reward is not a
//! delivery and does not touch the episode's in-flight count.
//!
//! The record belongs to the agent rewarded, not to the environment that
//! wrote it, and carries no sequence number; see
//! [`RewardRecord`]. Which agents may be rewarded is
//! the episode's to police, and it does: a reward naming an agent not in
//! the roster, or the environment itself, is a bug in the environment, and
//! the episode stops and reports it the way it reports a control addressed
//! to a stranger.
//!
//! # An ordinary agent cannot send a control, and the types say so
//!
//! [`Handler::handle`] returns `Vec<Action<P>>`. An action carries a
//! payload and a set of recipients, and there is no constructor for a
//! control or a reward on one, so a player's strategy has nothing to reach
//! for. The check the [`Router`](crate::Router) makes — that a control's
//! sender is the environment — is a backstop for a hole in the runtime,
//! not a rule the game code has to remember.
//!
//! # The environment's cycle is an agent's cycle
//!
//! The environment runs in [`Agent`](crate::Agent)'s loop, wrapped in the
//! adapter this module provides, so everything said about a cycle is true
//! of it: it takes the controls at the head of its queue before the one
//! message behind them, it observes one thing per cycle, and its
//! `Effect::Act` actions are stamped, logged and sent exactly as an
//! agent's. The one thing the adapter adds is the seam the controls leave
//! by, and the two controls leave it differently. A `Start` goes out as
//! soon as the episode sees it, ahead of the messages of the cycle that
//! asked for it, so that an agent is started before anything is addressed
//! to it. A `Stop` is held until nothing is in flight, so a player told to
//! stop in the same cycle it is told something first observes the message
//! and then stops — which is what lets the moderator narrate an outcome
//! and stop everybody in one breath.

use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::time::Instant;

use crossbeam_channel::Sender;
use serde::Serialize;

use crate::agent::{Action, Handler, Observation};
use crate::log::{Record, RewardRecord};
use crate::message::{ActorId, Control, Payload};

/// What an environment's cycle produces: an action like any agent's, a
/// control for some of the agents, or a reward for one of them.
///
/// The two type parameters are the two things a game names: `W`, the
/// numeric type its rewards are in, and `P`, what its messages carry. The
/// reward type appears here and nowhere else in the runtime, because this
/// is the one place a reward is produced (ADR-0016).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Effect<W, P: Payload> {
    /// Say something, as any agent says anything.
    Act(Action<P>),
    /// Tell some agents to start or to stop.
    Control {
        /// The agents told. Never the environment itself: the episode is
        /// what starts and stops the environment.
        to: BTreeSet<ActorId>,
        /// What they are told.
        control: Control,
    },
    /// Record what one agent's behavior was worth.
    ///
    /// Nothing is sent and nobody is told. The reward is written to the
    /// log the instant it is returned; see the
    /// [module documentation](self).
    Reward {
        /// The agent rewarded, whose records the reward belongs to.
        /// Never the environment itself, which plays no game and so has
        /// nothing to be rewarded for.
        agent: ActorId,
        /// What its behavior was worth, in the game's own units.
        value: W,
    },
}

impl<W, P: Payload> Effect<W, P> {
    /// A control for the given agents.
    pub fn control<I, A>(to: I, control: Control) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<ActorId>,
    {
        Self::Control {
            to: to.into_iter().map(Into::into).collect(),
            control,
        }
    }

    /// A reward of `value` for `agent`.
    pub fn reward(agent: impl Into<ActorId>, value: W) -> Self {
        Self::Reward {
            agent: agent.into(),
            value,
        }
    }
}

/// The distinguished agent that controls an episode.
///
/// It is a [`Handler`] with a wider return type, and the episode runs it
/// through the same loop as everything else; see the
/// [module documentation](self).
pub trait Environment<W, P: Payload> {
    /// The environment's opening effects, called once when the episode
    /// starts it.
    ///
    /// This is where an episode's agents are started: an environment that
    /// returns no `Effect::Control { control: Start, .. }` here has an
    /// episode nobody may act in, which goes quiescent at once and is a
    /// [`Stalled`](crate::EpisodeError::Stalled) episode.
    ///
    /// Opening effects are decided from nothing, for the reason
    /// [`Handler::start`]'s are.
    fn start(&mut self, now: Instant) -> Vec<Effect<W, P>>;

    /// Folds one observation into the environment's state and says what to
    /// send, whom to control, and whom to reward.
    ///
    /// One observation, because a cycle handles exactly one (ADR-0008), and
    /// an environment's cycle is an agent's cycle. A cycle that popped no
    /// observation does not call this at all.
    ///
    /// Nothing interrupts it, for the reason nothing interrupts
    /// [`Handler::handle`]: an agent does not know it is being stopped
    /// (ADR-0009), and the environment is the one deciding when anybody is.
    fn handle(&mut self, observation: &Observation<P>) -> Vec<Effect<W, P>>;

    /// What the environment does when its deadline passes and nothing has
    /// arrived.
    ///
    /// Its own method for the reason [`Handler::timeout`] is: waking on a
    /// deadline is not observing anything. The default does nothing, which
    /// is what both of the environments in the tree want — neither is
    /// configured with a timeout at all.
    fn timeout(&mut self, _now: Instant) -> Vec<Effect<W, P>> {
        Vec::new()
    }

    /// The next instant this environment wants a cycle, or `None`.
    ///
    /// The same contract as [`Handler::deadline`], which the adapter
    /// forwards this to: an absolute instant on the agent's clock, replacing
    /// whatever was pending, with a past one firing at once. Neither
    /// environment in the tree sets one.
    fn deadline(&self) -> Option<Instant> {
        None
    }
}

/// A boxed environment is an environment, so that an [`Episode`] can hold
/// one without being generic over which.
///
/// [`Episode`]: crate::Episode
impl<W, P: Payload, E: Environment<W, P> + ?Sized> Environment<W, P> for Box<E> {
    fn start(&mut self, now: Instant) -> Vec<Effect<W, P>> {
        (**self).start(now)
    }

    fn handle(&mut self, observation: &Observation<P>) -> Vec<Effect<W, P>> {
        (**self).handle(observation)
    }

    fn timeout(&mut self, now: Instant) -> Vec<Effect<W, P>> {
        (**self).timeout(now)
    }

    fn deadline(&self) -> Option<Instant> {
        (**self).deadline()
    }
}

/// One control an environment asked for, on its way to the router.
///
/// `Debug`, `Clone` and equality are derived: neither field mentions the
/// domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commanded {
    /// The agents told.
    pub to: BTreeSet<ActorId>,
    /// What they are told.
    pub control: Control,
}

/// A reward the adapter refused to write, and why.
///
/// Only a refusal travels here. A reward the adapter accepts is written and
/// nothing is sent; this says the environment produced a reward that could
/// not become a record, which is a bug in the environment, and the episode
/// fails rather than finishing with a log that quietly lacks it.
///
/// There are two ways to fail and they are kept apart, because a reader
/// told the wrong one would look in the wrong place: an agent that could
/// not be rewarded is a name the environment got wrong, and a value that
/// would not serialize is a reward *type* that cannot be logged at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rewarded {
    /// The agent the environment named.
    pub agent: ActorId,
    /// What was wrong with the reward.
    pub refusal: Refusal,
}

/// Why a reward was refused; see [`Rewarded`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// The agent named is not one this episode can reward: not in the
    /// roster, or the environment itself.
    NotRewardable,
    /// The value would not serialize, so there is no record to write. The
    /// reward type is the game's, and a type the log cannot hold is a bug
    /// in the game rather than in any one reward.
    Unserializable,
}

/// An [`Environment`] wearing a [`Handler`]'s face, so that the agent loop
/// can run it without knowing what it is.
///
/// The adapter splits each cycle's effects three ways: the actions go back
/// to the loop, which stamps, records and dispatches them as it would any
/// agent's; the controls go on `commands`, a channel the episode drains;
/// and a reward is **written to the log here**, the instant the
/// handler returns it, if it names somebody this episode can reward.
///
/// # Why a reward is written here and not by the episode
///
/// Everything else an environment produces has somewhere else to be: an
/// action is routed, a control is delivered. A reward is neither. Its
/// whole life is the line it becomes, so the shortest honest path is to
/// write it where it is decided, stamped with the instant of the decision.
/// Handing it to the episode to write would put an unbounded queue and a
/// scheduling delay between the instant the record claims and the instant
/// it was made.
///
/// Whether the agent named is one this episode could reward is the
/// roster's question, and the roster is settled before any thread is
/// spawned, so the adapter is handed it and answers before it writes. A
/// reward for a stranger is therefore never written at all, rather than
/// written and then disowned: the episode is about to fail, and a
/// log holding a reward for an agent that has no records of its own would
/// be evidence of nothing. The name goes on `rewards` instead, and the
/// episode fails the episode where it drains `commands`.
///
/// # The ordering the two channels keep
///
/// A cycle's controls are queued on `commands` while the handler is
/// running, which is strictly before the loop sends that cycle's dispatch.
/// The episode never looks at `commands` except just after it has taken a
/// dispatch off the environment's, so a control this cycle asked for is
/// already there to be found, and the messages it accompanies have already
/// been routed. That is the whole of the ordering guarantee, and it is what
/// makes "narrate the outcome, then stop everybody" a thing one cycle can
/// say: each player observes the narration and stops afterwards, rather
/// than stopping with the narration still on its queue and never seeing it.
pub struct Adapter<W, P: Payload, E> {
    environment: E,
    /// Which reward type this adapter serializes. The adapter stores no
    /// reward — a reward is written the instant it is assigned — but `E`
    /// alone does not say which [`Environment`] impl is meant when a type
    /// implements more than one, so the parameter is carried here rather
    /// than inferred. The function pointer makes the marker own nothing
    /// and demand nothing of `W`.
    reward: PhantomData<fn() -> W>,
    /// Whom this episode may reward: every agent in the roster but the
    /// environment itself. Fixed before any thread is spawned and never
    /// added to, so the adapter can answer the question rather than ask.
    rewardable: BTreeSet<ActorId>,
    commands: Sender<Commanded>,
    rewards: Sender<Rewarded>,
    records: Sender<Record<P>>,
}

impl<W: Serialize + Copy + Send + 'static, P: Payload, E: Environment<W, P>> Adapter<W, P, E> {
    /// Wraps `environment`, sending the controls it asks for on `commands`,
    /// writing the rewards it assigns to `records`, and naming each rewarded
    /// agent on `rewards` for the episode to check.
    pub const fn new(
        environment: E,
        rewardable: BTreeSet<ActorId>,
        commands: Sender<Commanded>,
        rewards: Sender<Rewarded>,
        records: Sender<Record<P>>,
    ) -> Self {
        Self {
            environment,
            reward: PhantomData,
            rewardable,
            commands,
            rewards,
            records,
        }
    }

    /// Queues the controls among `effects`, writes the rewards, and returns
    /// the actions.
    ///
    /// A closed channel at either end means the episode has stopped
    /// listening, which happens only once it is shutting the environment
    /// down. The control is then nobody's to act on, and dropping it is the
    /// same silence as sending it to an agent that has already stopped. A
    /// reward whose record cannot be written is the writer having gone
    /// away, which the loop's own next record will report as
    /// [`Error::WriterClosed`](crate::Error::WriterClosed); there is
    /// nothing useful to add here, and failing the split would lose the
    /// cycle's actions as well.
    fn split(&self, effects: Vec<Effect<W, P>>) -> Vec<Action<P>> {
        let mut actions = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                Effect::Act(action) => actions.push(action),
                Effect::Control { to, control } => {
                    let _ = self.commands.send(Commanded { to, control });
                }
                Effect::Reward { agent, value } => {
                    // Checked before it is written, so that the log
                    // never carries a reward for somebody who has none:
                    // the roster is settled before any thread starts, so
                    // the answer is here to be had rather than somewhere
                    // to be asked.
                    if !self.rewardable.contains(&agent) {
                        let _ = self.rewards.send(Rewarded {
                            agent,
                            refusal: Refusal::NotRewardable,
                        });
                        continue;
                    }
                    // Serialized here, where the reward type is still
                    // known: the record carries JSON so that nothing
                    // downstream of this line is generic over a reward
                    // (ADR-0016). Reported and not written if it will not
                    // serialize, for the reason a reward for a stranger is:
                    // the episode is about to fail either way, and a
                    // log silently missing a reward the environment
                    // assigned would be evidence of nothing.
                    let Ok(value) = serde_json::to_value(value) else {
                        let _ = self.rewards.send(Rewarded {
                            agent,
                            refusal: Refusal::Unserializable,
                        });
                        continue;
                    };
                    // Stamped now, so that `t` is the instant the reward was
                    // decided rather than the instant anybody got around to
                    // it. The instant goes on unconverted; the writer is
                    // what measures it from the episode's origin.
                    let record = RewardRecord {
                        agent,
                        t: Instant::now(),
                        value,
                    };
                    let _ = self.records.send(record.into());
                }
            }
        }
        actions
    }
}

impl<W: Serialize + Copy + Send + 'static, P: Payload, E: Environment<W, P>> Handler<P>
    for Adapter<W, P, E>
{
    fn start(&mut self, now: Instant) -> Vec<Action<P>> {
        let effects = self.environment.start(now);
        self.split(effects)
    }

    fn handle(&mut self, observation: &Observation<P>) -> Vec<Action<P>> {
        let effects = self.environment.handle(observation);
        self.split(effects)
    }

    fn timeout(&mut self, now: Instant) -> Vec<Action<P>> {
        let effects = self.environment.timeout(now);
        self.split(effects)
    }

    fn deadline(&self) -> Option<Instant> {
        self.environment.deadline()
    }
}

impl<W, P: Payload, E> std::fmt::Debug for Adapter<W, P, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adapter").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {

    use crossbeam_channel::{Receiver, unbounded};

    use super::*;
    use crate::message::Message;
    use crate::testing::{TestPayload, id, ids};

    /// The one observation a cycle hands the adapter. What it says never
    /// matters here: `Opener` answers whatever it is told with the same
    /// three effects, and these tests are about where each effect goes.
    fn observation() -> Observation<TestPayload> {
        Observation {
            message: Message::new("a", ["environment"], 0, TestPayload::Step(1)),
            at: Instant::now(),
        }
    }

    /// An environment that starts two agents, says one thing, rewards one
    /// of them, and stops them.
    struct Opener;

    impl Environment<i32, TestPayload> for Opener {
        fn start(&mut self, _now: Instant) -> Vec<Effect<i32, TestPayload>> {
            vec![
                Effect::control(["a", "b"], Control::Start),
                Effect::Act(Action::to(["a"], TestPayload::Step(1))),
            ]
        }

        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Effect<i32, TestPayload>> {
            vec![
                Effect::reward("a", 1),
                Effect::control(["a", "b"], Control::Stop),
            ]
        }
    }

    /// The three channels an adapter writes to, and the adapter itself.
    struct Rig {
        adapter: Adapter<i32, TestPayload, Opener>,
        commanded: Receiver<Commanded>,
        rewarded: Receiver<Rewarded>,
        records: Receiver<Record<TestPayload>>,
    }

    fn rig() -> Rig {
        let (commands, commanded) = unbounded();
        let (paid, rewarded) = unbounded();
        let (recorder, records) = unbounded();
        Rig {
            adapter: Adapter::new(Opener, ids(["a", "b"]), commands, paid, recorder),
            commanded,
            rewarded,
            records,
        }
    }

    #[test]
    fn the_adapter_hands_the_loop_the_actions_and_the_episode_the_controls() {
        let mut rig = rig();
        assert_eq!(
            rig.adapter.start(at(0)),
            [Action::to(["a"], TestPayload::Step(1))],
            "the loop sees an action and nothing else"
        );
        assert_eq!(
            rig.commanded.try_recv(),
            Ok(Commanded {
                to: ids(["a", "b"]),
                control: Control::Start
            })
        );
        assert!(rig.adapter.handle(&observation()).is_empty());
        assert_eq!(
            rig.commanded.try_recv(),
            Ok(Commanded {
                to: ids(["a", "b"]),
                control: Control::Stop
            })
        );
    }

    #[test]
    fn a_reward_is_written_where_it_is_decided_and_routed_nowhere() {
        // The record is the whole of what a reward does. It goes to the
        // writer the instant the handler returns it, naming the agent
        // rewarded and not the environment that decided it, and the loop
        // is handed nothing to send on its account.
        let mut rig = rig();
        rig.adapter.start(at(0));
        assert!(rig.records.try_recv().is_err(), "the start rewards nobody");
        let actions = rig.adapter.handle(&observation());
        assert!(actions.is_empty(), "a reward is not an action: {actions:?}");
        let Record::Reward(record) = rig.records.try_recv().unwrap() else {
            panic!("a reward is written as a reward record");
        };
        assert_eq!(record.agent, id("a"));
        assert_eq!(record.value, 1);
        assert!(
            rig.records.try_recv().is_err(),
            "one effect writes one record"
        );
        // And the episode is told nothing: a reward the adapter accepts
        // needs no checking, because the checking is why it was written.
        assert!(
            rig.rewarded.try_recv().is_err(),
            "an accepted reward is not reported"
        );
    }

    /// An environment whose rewards are reals rather than integers, which
    /// is the reason the reward type is a parameter at all (ADR-0007).
    struct Scorer;

    impl Environment<f64, TestPayload> for Scorer {
        fn start(&mut self, _now: Instant) -> Vec<Effect<f64, TestPayload>> {
            vec![Effect::reward("a", 0.5)]
        }

        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Effect<f64, TestPayload>> {
            Vec::new()
        }
    }

    #[test]
    fn a_reward_is_serialized_where_it_is_assigned_whatever_its_type() {
        // The record carries JSON, so the reward type stops at this
        // adapter (ADR-0016) — and it stops without being flattened to an
        // integer on the way. A game scored in reals logs reals.
        let (commands, _commanded) = unbounded();
        let (paid, _rewarded) = unbounded();
        let (recorder, records) = unbounded::<Record<TestPayload>>();
        let mut adapter = Adapter::new(Scorer, ids(["a"]), commands, paid, recorder);
        assert!(adapter.start(at(0)).is_empty());
        let Record::Reward(record) = records.try_recv().unwrap() else {
            panic!("a reward is written as a reward record");
        };
        assert_eq!(record.value, serde_json::json!(0.5));
    }

    /// An environment that rewards somebody who is not in the roster.
    struct Stranger;

    impl Environment<i32, TestPayload> for Stranger {
        fn start(&mut self, _now: Instant) -> Vec<Effect<i32, TestPayload>> {
            vec![Effect::reward("nobody", 1)]
        }

        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Effect<i32, TestPayload>> {
            Vec::new()
        }
    }

    #[test]
    fn a_reward_for_somebody_outside_the_roster_is_reported_and_not_written() {
        // The episode is about to fail. A log carrying a reward for an
        // agent that has no records of its own would be evidence of
        // nothing, so the name is reported and the line is never written.
        let (commands, _commanded) = unbounded();
        let (paid, rewarded) = unbounded();
        let (recorder, records) = unbounded::<Record<TestPayload>>();
        let mut adapter = Adapter::new(Stranger, ids(["a", "b"]), commands, paid, recorder);
        assert!(adapter.start(at(0)).is_empty());
        assert_eq!(
            rewarded.try_recv(),
            Ok(Rewarded {
                agent: id("nobody"),
                refusal: Refusal::NotRewardable,
            })
        );
        assert!(
            records.try_recv().is_err(),
            "a reward for a stranger is not written"
        );
    }

    #[test]
    fn a_control_or_a_reward_nobody_is_listening_for_is_dropped_rather_than_failing_the_cycle() {
        let mut rig = rig();
        drop(rig.commanded);
        drop(rig.rewarded);
        drop(rig.records);
        assert_eq!(
            rig.adapter.start(at(0)),
            [Action::to(["a"], TestPayload::Step(1))]
        );
        assert!(rig.adapter.handle(&observation()).is_empty());
    }

    #[test]
    fn an_effect_says_what_it_is() {
        let act: Effect<i32, TestPayload> = Effect::Act(Action::to(["a"], TestPayload::Step(1)));
        let control: Effect<i32, TestPayload> = Effect::control(["a"], Control::Stop);
        let reward: Effect<i32, TestPayload> = Effect::reward("a", -1);
        assert_eq!(act, act.clone());
        assert_eq!(control, control.clone());
        assert_eq!(reward, reward.clone());
        assert_ne!(act, control);
        assert_ne!(control, reward);
        assert_ne!(reward, act);
        assert_ne!(reward, Effect::reward("a", 1));
        assert_ne!(reward, Effect::reward("b", -1));
        assert!(format!("{act:?}").starts_with("Act"));
        assert!(format!("{control:?}").starts_with("Control"));
        assert!(format!("{reward:?}").starts_with("Reward"));
        assert_eq!(
            control,
            Effect::Control {
                to: [id("a")].into(),
                control: Control::Stop
            }
        );
        assert_eq!(
            reward,
            Effect::Reward {
                agent: id("a"),
                value: -1
            }
        );
        assert!(format!("{:?}", rig().adapter).starts_with("Adapter"));
    }

    /// An environment that wants waking at an instant of its own, and
    /// remembers the `now` of each hook it is given.
    struct Punctual {
        deadline: Option<Instant>,
        readings: Vec<Instant>,
    }

    impl Environment<i32, TestPayload> for Punctual {
        fn start(&mut self, now: Instant) -> Vec<Effect<i32, TestPayload>> {
            self.readings.push(now);
            Vec::new()
        }

        fn handle(&mut self, _: &Observation<TestPayload>) -> Vec<Effect<i32, TestPayload>> {
            Vec::new()
        }

        fn timeout(&mut self, now: Instant) -> Vec<Effect<i32, TestPayload>> {
            self.readings.push(now);
            Vec::new()
        }

        fn deadline(&self) -> Option<Instant> {
            self.deadline
        }
    }

    /// The instants these tests name, offset from the one fixed base every
    /// test module in the crate shares; see [`testing::BASE`](crate::testing).
    use crate::testing::at_nanos as at;

    #[test]
    fn the_adapter_forwards_the_environments_deadline_and_the_time() {
        // An environment sets deadlines the way a handler does (ADR-0010);
        // the adapter is what carries them between the two traits.
        let wanted = at(3);
        let (commands, _commanded) = unbounded();
        let (paid, _rewarded) = unbounded();
        let (recorder, _records) = unbounded();
        let mut adapter = Adapter::new(
            Punctual {
                deadline: Some(wanted),
                readings: Vec::new(),
            },
            ids(["a"]),
            commands,
            paid,
            recorder,
        );
        assert_eq!(Handler::deadline(&adapter), Some(wanted));

        let (start, timed_out) = (at(1), at(2));
        assert!(adapter.start(start).is_empty());
        assert!(adapter.timeout(timed_out).is_empty());
        assert_eq!(adapter.environment.readings, [start, timed_out]);

        // And an environment with no opinion says so, the same as a handler.
        adapter.environment.deadline = None;
        assert_eq!(Handler::deadline(&adapter), None);
    }

    /// A boxed environment forwards every hook, including the two ADR-0010
    /// added, so that an episode holding one is not a special case.
    #[test]
    fn a_boxed_environment_forwards_the_new_hooks() {
        let wanted = at(4);
        let mut boxed: Box<dyn Environment<i32, TestPayload>> = Box::new(Punctual {
            deadline: Some(wanted),
            readings: Vec::new(),
        });
        let now = at(1);
        assert!(boxed.start(now).is_empty());
        assert!(boxed.timeout(now).is_empty());
        assert_eq!(boxed.deadline(), Some(wanted));
    }
}
