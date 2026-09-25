//! The environment: the one agent in an episode that controls it.
//!
//! An [`Environment`] is an agent like any other — one thread, two queues,
//! a [`Cancel`] per cycle, and a trajectory of its own — with one power
//! nobody else has. Where an ordinary [`Handler`] returns only
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
//! [`Handler::handle`] returns `Vec<Action<D>>`. There is no constructor
//! for a control or a reward on an action, and no variant of
//! [`Recipients`](crate::Recipients) that could carry one, so a player's
//! policy has nothing to reach for. The check the
//! [`Router`](crate::Router) makes — that a control's sender is the
//! environment — is a backstop for a hole in the runtime, not a rule the
//! game code has to remember.
//!
//! # The environment's cycle is an agent's cycle
//!
//! The environment runs in [`Agent`](crate::Agent)'s loop, wrapped in the
//! adapter this module provides, so everything ADR-0007 says about a cycle
//! is true of it: it pops controls before events, its `Effect::Act`
//! actions are stamped and logged exactly as an agent's, and a `Stop` that
//! arrives while it is deciding preempts the cycle and drops what it
//! produced. The one thing the adapter adds is the seam the controls leave
//! by, and it is built so they leave **after** the cycle's events: a
//! player told to stop in the same cycle it is told something first
//! observes the message and then stops, which is what lets the moderator
//! narrate an outcome and stop everybody in one breath.

use std::collections::BTreeSet;

use crossbeam_channel::Sender;

use crate::agent::{Action, Handler, Observation};
use crate::cancel::Cancel;
use crate::clock::Clock;
use crate::event::{AgentId, Control, Domain};
use crate::trajectory::{LogRecord, RewardRecord};

/// What an environment's cycle produces: an action like any agent's, a
/// control for some of the agents, or a reward for one of them.
///
/// `Debug`, `Clone` and equality are written out rather than derived, for
/// the reason [`Event`](crate::Event)'s are: a derive would ask them of
/// `D`, the marker type, when what has to have them is `D::Payload`.
pub enum Effect<D: Domain> {
    /// Say something, as any agent says anything.
    Act(Action<D>),
    /// Tell some agents to start or to stop.
    Control {
        /// The agents told. Never the environment itself: the episode is
        /// what starts and stops the environment.
        to: BTreeSet<AgentId>,
        /// What they are told.
        control: Control,
    },
    /// Record what one agent's behavior was worth.
    ///
    /// Nothing is sent and nobody is told. The reward is written to the
    /// trajectory the instant it is returned; see the
    /// [module documentation](self).
    Reward {
        /// The agent rewarded, whose trajectory the record belongs to.
        /// Never the environment itself, which plays no game and so has
        /// nothing to be rewarded for.
        agent: AgentId,
        /// What its behavior was worth, in the game's own units.
        value: D::Reward,
    },
}

impl<D: Domain> Effect<D> {
    /// A control for the given agents.
    pub fn control<I, A>(to: I, control: Control) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<AgentId>,
    {
        Self::Control {
            to: to.into_iter().map(Into::into).collect(),
            control,
        }
    }

    /// A reward of `value` for `agent`.
    pub fn reward(agent: impl Into<AgentId>, value: D::Reward) -> Self {
        Self::Reward {
            agent: agent.into(),
            value,
        }
    }
}

impl<D: Domain> std::fmt::Debug for Effect<D>
where
    D::Payload: std::fmt::Debug,
    D::Reward: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Act(action) => f.debug_tuple("Act").field(action).finish(),
            Self::Control { to, control } => f
                .debug_struct("Control")
                .field("to", to)
                .field("control", control)
                .finish(),
            Self::Reward { agent, value } => f
                .debug_struct("Reward")
                .field("agent", agent)
                .field("value", value)
                .finish(),
        }
    }
}

impl<D: Domain> Clone for Effect<D> {
    fn clone(&self) -> Self {
        match self {
            Self::Act(action) => Self::Act(action.clone()),
            Self::Control { to, control } => Self::Control {
                to: to.clone(),
                control: *control,
            },
            Self::Reward { agent, value } => Self::Reward {
                agent: agent.clone(),
                value: *value,
            },
        }
    }
}

impl<D: Domain> PartialEq for Effect<D>
where
    D::Payload: PartialEq,
    D::Reward: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Act(mine), Self::Act(theirs)) => mine == theirs,
            (
                Self::Control { to, control },
                Self::Control {
                    to: other_to,
                    control: other_control,
                },
            ) => to == other_to && control == other_control,
            (
                Self::Reward { agent, value },
                Self::Reward {
                    agent: other_agent,
                    value: other_value,
                },
            ) => agent == other_agent && value == other_value,
            _ => false,
        }
    }
}

impl<D: Domain> Eq for Effect<D>
where
    D::Payload: Eq,
    D::Reward: Eq,
{
}

/// The distinguished agent that controls an episode.
///
/// It is a [`Handler`] with a wider return type, and the episode runs it
/// through the same loop as everything else; see the
/// [module documentation](self).
pub trait Environment<D: Domain> {
    /// The environment's opening effects, called once when the episode
    /// starts it.
    ///
    /// This is where an episode's agents are started: an environment that
    /// returns no `Effect::Control { control: Start, .. }` here has an
    /// episode nobody may act in, which goes quiescent at once and is a
    /// [`Stalled`](crate::EpisodeError::Stalled) episode.
    ///
    /// It takes no [`Cancel`], for the reason [`Handler::start`] does not:
    /// opening effects are decided from nothing.
    fn start(&mut self) -> Vec<Effect<D>>;

    /// Folds one observation into the environment's state and says what to
    /// send, whom to control, and whom to reward.
    ///
    /// One observation, because a cycle handles exactly one (ADR-0008), and
    /// an environment's cycle is an agent's cycle. A cycle that popped no
    /// observation does not call this at all.
    ///
    /// `cancel` is this cycle's, and means what it means for any handler.
    fn handle(&mut self, observation: &Observation<D>, cancel: &Cancel) -> Vec<Effect<D>>;

    /// What the environment does when its deadline passes and nothing has
    /// arrived.
    ///
    /// Its own method for the reason [`Handler::timeout`] is: waking on a
    /// deadline is not observing anything. The default does nothing, which
    /// is what both of the environments in the tree want — neither is
    /// configured with a timeout at all.
    fn timeout(&mut self, _cancel: &Cancel) -> Vec<Effect<D>> {
        Vec::new()
    }
}

/// A boxed environment is an environment, so that an [`Episode`] can hold
/// one without being generic over which.
///
/// [`Episode`]: crate::Episode
impl<D: Domain, E: Environment<D> + ?Sized> Environment<D> for Box<E> {
    fn start(&mut self) -> Vec<Effect<D>> {
        (**self).start()
    }

    fn handle(&mut self, observation: &Observation<D>, cancel: &Cancel) -> Vec<Effect<D>> {
        (**self).handle(observation, cancel)
    }

    fn timeout(&mut self, cancel: &Cancel) -> Vec<Effect<D>> {
        (**self).timeout(cancel)
    }
}

/// One control an environment asked for, on its way to the router.
///
/// `Debug`, `Clone` and equality are derived: neither field mentions the
/// domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Commanded {
    /// The agents told.
    pub to: BTreeSet<AgentId>,
    /// What they are told.
    pub control: Control,
}

/// An agent an environment tried to reward and could not: one that is not
/// in the roster, or the environment itself.
///
/// Only a refusal travels here. A reward the adapter accepts is written and
/// nothing is sent; this says the environment named somebody it could not
/// be rewarding, which is a bug in the environment, and the episode turns
/// it into the same error the router would give for addressing a stranger.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rewarded {
    /// The agent rewarded.
    pub agent: AgentId,
}

/// An [`Environment`] wearing a [`Handler`]'s face, so that the agent loop
/// can run it without knowing what it is.
///
/// The adapter splits each cycle's effects three ways: the actions go back
/// to the loop, which stamps, records and dispatches them as it would any
/// agent's; the controls go on `commands`, a channel the episode drains;
/// and a reward is **written to the trajectory here**, the instant the
/// handler returns it, if it names somebody this episode can reward.
///
/// # What preemption does to each of the three
///
/// The split happens inside the handler call, which is before the loop
/// asks whether a `Stop` arrived while the handler was deciding. So the
/// three effects of a preempted cycle do not share a fate: the actions are
/// dropped, because the loop holds them and drops them, while the controls
/// are already on `commands` and the rewards are already written.
///
/// Nothing preempts a live environment cycle today — the episode stops the
/// environment only once every other agent has ended, and a `Stop` is held
/// until nothing is in flight — so the asymmetry is not reachable. It is
/// stated because it would not be obvious to whoever first makes it
/// reachable, and because the honest fix then is to split the effects
/// after the preemption question is asked rather than before.
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
/// trajectory holding a reward for an agent that has no trajectory would
/// be evidence of nothing. The name goes on `rewards` instead, and the
/// episode fails the episode where it drains `commands`.
///
/// # The ordering the two channels keep
///
/// A cycle's controls are queued on `commands` while the handler is
/// running, which is strictly before the loop sends that cycle's dispatch.
/// The episode never looks at `commands` except just after it has taken a
/// dispatch off the environment's, so a control this cycle asked for is
/// already there to be found, and the events it accompanies have already
/// been routed. That is the whole of the ordering guarantee, and it is what
/// makes "narrate the outcome, then stop everybody" a thing one cycle can
/// say: each player observes the narration and stops afterwards, rather
/// than stopping with the narration still on its queue and never seeing it.
pub struct Adapter<D: Domain, E> {
    environment: E,
    /// Whom this episode may reward: every agent in the roster but the
    /// environment itself. Fixed before any thread is spawned and never
    /// added to, so the adapter can answer the question rather than ask.
    rewardable: BTreeSet<AgentId>,
    commands: Sender<Commanded>,
    rewards: Sender<Rewarded>,
    records: Sender<LogRecord<D>>,
    clock: Clock,
}

impl<D: Domain, E: Environment<D>> Adapter<D, E> {
    /// Wraps `environment`, sending the controls it asks for on `commands`,
    /// writing the rewards it assigns to `records` stamped by `clock`, and
    /// naming each rewarded agent on `rewards` for the episode to check.
    pub const fn new(
        environment: E,
        rewardable: BTreeSet<AgentId>,
        commands: Sender<Commanded>,
        rewards: Sender<Rewarded>,
        records: Sender<LogRecord<D>>,
        clock: Clock,
    ) -> Self {
        Self {
            environment,
            rewardable,
            commands,
            rewards,
            records,
            clock,
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
    fn split(&self, effects: Vec<Effect<D>>) -> Vec<Action<D>> {
        let mut actions = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                Effect::Act(action) => actions.push(action),
                Effect::Control { to, control } => {
                    let _ = self.commands.send(Commanded { to, control });
                }
                Effect::Reward { agent, value } => {
                    // Checked before it is written, so that a trajectory
                    // never carries a reward for somebody who has none:
                    // the roster is settled before any thread starts, so
                    // the answer is here to be had rather than somewhere
                    // to be asked.
                    if !self.rewardable.contains(&agent) {
                        let _ = self.rewards.send(Rewarded { agent });
                        continue;
                    }
                    // Stamped now, so that `created` is the instant the
                    // reward was decided rather than the instant anybody
                    // got around to it.
                    let record = RewardRecord {
                        agent,
                        created: self.clock.now(),
                        value,
                    };
                    let _ = self.records.send(record.into());
                }
            }
        }
        actions
    }
}

impl<D: Domain, E: Environment<D>> Handler<D> for Adapter<D, E> {
    fn start(&mut self) -> Vec<Action<D>> {
        let effects = self.environment.start();
        self.split(effects)
    }

    fn handle(&mut self, observation: &Observation<D>, cancel: &Cancel) -> Vec<Action<D>> {
        let effects = self.environment.handle(observation, cancel);
        self.split(effects)
    }

    fn timeout(&mut self, cancel: &Cancel) -> Vec<Action<D>> {
        let effects = self.environment.timeout(cancel);
        self.split(effects)
    }
}

impl<D: Domain, E> std::fmt::Debug for Adapter<D, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adapter").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};

    use super::*;
    use crate::clock::Timestamp;
    use crate::event::Event;
    use crate::testing::{TestDomain, TestPayload, id, ids};

    /// The one observation a cycle hands the adapter. What it says never
    /// matters here: `Opener` answers whatever it is told with the same
    /// three effects, and these tests are about where each effect goes.
    fn observation() -> Observation<TestDomain> {
        Observation {
            event: Event::new(
                "a",
                ["environment"],
                Timestamp::default(),
                TestPayload::Step(1),
            ),
            received: Timestamp::default(),
        }
    }

    /// An environment that starts two agents, says one thing, rewards one
    /// of them, and stops them.
    struct Opener;

    impl Environment<TestDomain> for Opener {
        fn start(&mut self) -> Vec<Effect<TestDomain>> {
            vec![
                Effect::control(["a", "b"], Control::Start),
                Effect::Act(Action::to(["a"], TestPayload::Step(1))),
            ]
        }

        fn handle(&mut self, _: &Observation<TestDomain>, _: &Cancel) -> Vec<Effect<TestDomain>> {
            vec![
                Effect::reward("a", 1),
                Effect::control(["a", "b"], Control::Stop),
            ]
        }
    }

    /// The three channels an adapter writes to, and the adapter itself.
    struct Rig {
        adapter: Adapter<TestDomain, Opener>,
        commanded: Receiver<Commanded>,
        rewarded: Receiver<Rewarded>,
        records: Receiver<LogRecord<TestDomain>>,
    }

    fn rig() -> Rig {
        let (commands, commanded) = unbounded();
        let (paid, rewarded) = unbounded();
        let (recorder, records) = unbounded();
        Rig {
            adapter: Adapter::new(
                Opener,
                ids(["a", "b"]),
                commands,
                paid,
                recorder,
                Clock::start(),
            ),
            commanded,
            rewarded,
            records,
        }
    }

    #[test]
    fn the_adapter_hands_the_loop_the_actions_and_the_episode_the_controls() {
        let mut rig = rig();
        assert_eq!(
            rig.adapter.start(),
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
        assert!(
            rig.adapter
                .handle(&observation(), &Cancel::cancelled())
                .is_empty()
        );
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
        rig.adapter.start();
        assert!(rig.records.try_recv().is_err(), "the start rewards nobody");
        let actions = rig.adapter.handle(&observation(), &Cancel::cancelled());
        assert!(actions.is_empty(), "a reward is not an action: {actions:?}");
        let LogRecord::Reward(record) = rig.records.try_recv().unwrap() else {
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

    /// An environment that rewards somebody who is not in the roster.
    struct Stranger;

    impl Environment<TestDomain> for Stranger {
        fn start(&mut self) -> Vec<Effect<TestDomain>> {
            vec![Effect::reward("nobody", 1)]
        }

        fn handle(&mut self, _: &Observation<TestDomain>, _: &Cancel) -> Vec<Effect<TestDomain>> {
            Vec::new()
        }
    }

    #[test]
    fn a_reward_for_somebody_outside_the_roster_is_reported_and_not_written() {
        // The episode is about to fail. A trajectory carrying a reward for
        // an agent that has no trajectory would be evidence of nothing, so
        // the name is reported and the line is never written.
        let (commands, _commanded) = unbounded();
        let (paid, rewarded) = unbounded();
        let (recorder, records) = unbounded::<LogRecord<TestDomain>>();
        let mut adapter = Adapter::new(
            Stranger,
            ids(["a", "b"]),
            commands,
            paid,
            recorder,
            Clock::start(),
        );
        assert!(adapter.start().is_empty());
        assert_eq!(
            rewarded.try_recv(),
            Ok(Rewarded {
                agent: id("nobody")
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
            rig.adapter.start(),
            [Action::to(["a"], TestPayload::Step(1))]
        );
        assert!(
            rig.adapter
                .handle(&observation(), &Cancel::cancelled())
                .is_empty()
        );
    }

    #[test]
    fn an_effect_says_what_it_is() {
        let act: Effect<TestDomain> = Effect::Act(Action::to(["a"], TestPayload::Step(1)));
        let control: Effect<TestDomain> = Effect::control(["a"], Control::Stop);
        let reward: Effect<TestDomain> = Effect::reward("a", -1);
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
}
