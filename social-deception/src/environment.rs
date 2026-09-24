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
//! # An ordinary agent cannot send a control, and the types say so
//!
//! [`Handler::handle`] returns `Vec<Action<D>>`. There is no constructor
//! for a control on an action, and no variant of [`Recipients`](crate::Recipients) that could
//! carry one, so a player's policy has nothing to reach for. The check the
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
use crate::event::{AgentId, Control, Domain};

/// What an environment's cycle produces: an action like any agent's, or a
/// control for some of the agents.
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
    // `Reward { agent, value }` arrives in #44.
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
}

impl<D: Domain> std::fmt::Debug for Effect<D>
where
    D::Payload: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Act(action) => f.debug_tuple("Act").field(action).finish(),
            Self::Control { to, control } => f
                .debug_struct("Control")
                .field("to", to)
                .field("control", control)
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
        }
    }
}

impl<D: Domain> PartialEq for Effect<D>
where
    D::Payload: PartialEq,
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
            _ => false,
        }
    }
}

impl<D: Domain> Eq for Effect<D> where D::Payload: Eq {}

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

    /// Folds one cycle's observations into the environment's state and says
    /// what to send and whom to control.
    ///
    /// `cancel` is this cycle's, and means what it means for any handler.
    fn handle(&mut self, observations: &[Observation<D>], cancel: &Cancel) -> Vec<Effect<D>>;
}

/// A boxed environment is an environment, so that an [`Episode`] can hold
/// one without being generic over which.
///
/// [`Episode`]: crate::Episode
impl<D: Domain, E: Environment<D> + ?Sized> Environment<D> for Box<E> {
    fn start(&mut self) -> Vec<Effect<D>> {
        (**self).start()
    }

    fn handle(&mut self, observations: &[Observation<D>], cancel: &Cancel) -> Vec<Effect<D>> {
        (**self).handle(observations, cancel)
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

/// An [`Environment`] wearing a [`Handler`]'s face, so that the agent loop
/// can run it without knowing what it is.
///
/// The adapter splits each cycle's effects in two: the actions go back to
/// the loop, which stamps, records and dispatches them as it would any
/// agent's, and the controls go on `commands`, a channel the episode
/// drains.
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
    commands: Sender<Commanded>,
    marker: std::marker::PhantomData<fn() -> D>,
}

impl<D: Domain, E: Environment<D>> Adapter<D, E> {
    /// Wraps `environment`, sending the controls it asks for on `commands`.
    pub const fn new(environment: E, commands: Sender<Commanded>) -> Self {
        Self {
            environment,
            commands,
            marker: std::marker::PhantomData,
        }
    }

    /// Queues the controls among `effects` and returns the actions.
    ///
    /// A closed `commands` channel means the episode has stopped listening,
    /// which happens only once it is shutting the environment down. The
    /// control is then nobody's to act on, and dropping it is the same
    /// silence as sending it to an agent that has already stopped.
    fn split(&self, effects: Vec<Effect<D>>) -> Vec<Action<D>> {
        let mut actions = Vec::with_capacity(effects.len());
        for effect in effects {
            match effect {
                Effect::Act(action) => actions.push(action),
                Effect::Control { to, control } => {
                    let _ = self.commands.send(Commanded { to, control });
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

    fn handle(&mut self, observations: &[Observation<D>], cancel: &Cancel) -> Vec<Action<D>> {
        let effects = self.environment.handle(observations, cancel);
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
    use crossbeam_channel::unbounded;

    use super::*;
    use crate::testing::{TestDomain, TestPayload, id, ids};

    /// An environment that starts two agents, says one thing, and stops
    /// them.
    struct Opener;

    impl Environment<TestDomain> for Opener {
        fn start(&mut self) -> Vec<Effect<TestDomain>> {
            vec![
                Effect::control(["a", "b"], Control::Start),
                Effect::Act(Action::to(["a"], TestPayload::Step(1))),
            ]
        }

        fn handle(&mut self, _: &[Observation<TestDomain>], _: &Cancel) -> Vec<Effect<TestDomain>> {
            vec![Effect::control(["a", "b"], Control::Stop)]
        }
    }

    #[test]
    fn the_adapter_hands_the_loop_the_actions_and_the_episode_the_controls() {
        let (commands, commanded) = unbounded();
        let mut adapter = Adapter::new(Opener, commands);
        assert_eq!(
            adapter.start(),
            [Action::to(["a"], TestPayload::Step(1))],
            "the loop sees an action and nothing else"
        );
        assert_eq!(
            commanded.try_recv(),
            Ok(Commanded {
                to: ids(["a", "b"]),
                control: Control::Start
            })
        );
        assert!(adapter.handle(&[], &Cancel::cancelled()).is_empty());
        assert_eq!(
            commanded.try_recv(),
            Ok(Commanded {
                to: ids(["a", "b"]),
                control: Control::Stop
            })
        );
    }

    #[test]
    fn a_control_nobody_is_listening_for_is_dropped_rather_than_failing_the_cycle() {
        let (commands, commanded) = unbounded();
        let mut adapter = Adapter::new(Opener, commands);
        drop(commanded);
        assert_eq!(adapter.start(), [Action::to(["a"], TestPayload::Step(1))]);
    }

    #[test]
    fn an_effect_says_what_it_is() {
        let act: Effect<TestDomain> = Effect::Act(Action::to(["a"], TestPayload::Step(1)));
        let control: Effect<TestDomain> = Effect::control(["a"], Control::Stop);
        assert_eq!(act, act.clone());
        assert_eq!(control, control.clone());
        assert_ne!(act, control);
        assert!(format!("{act:?}").starts_with("Act"));
        assert!(format!("{control:?}").starts_with("Control"));
        assert_eq!(
            control,
            Effect::Control {
                to: [id("a")].into(),
                control: Control::Stop
            }
        );
        let (commands, _commanded) = unbounded::<Commanded>();
        let adapter = Adapter::<TestDomain, _>::new(Opener, commands);
        assert!(format!("{adapter:?}").starts_with("Adapter"));
    }
}
