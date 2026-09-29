//! The router: the fixed roster from actor id to that actor's two channels.
//!
//! An episode's topology is fixed when it starts. The roster is known, it does
//! not change, and actors do not discover each other. A send names its
//! recipients explicitly, and the router copies the message onto each
//! recipient's inbox. Application code addresses actor ids and never touches
//! transport.
//!
//! # Two channels, not one
//!
//! Each actor has an **inbox** for messages and a **control channel** for
//! [`Control`]s, and they are separate because a `Stop` now preempts
//! everything in the inbox (ADR-0016). The old runtime's one queue was right
//! only while a control never had to overtake anything; the perception thread
//! checks its control channel before every `select!` so that it does.
//!
//! # What the router refuses
//!
//! - A sender or recipient not in the roster.
//! - A sender among its own recipients. **There is no loopback**: an actor
//!   that wants to send itself a message sets a
//!   [`Reminder`](crate::Reminder), which is the one way a message reaches
//!   the actor that sent it.
//! - A control whose sender is not the episode's environment. The types
//!   already say so — a [`Policy`](crate::Policy) returns
//!   [`Action`](crate::Action)s and has no way to name a control — so this is
//!   a backstop against a hole in the runtime rather than against a game's
//!   code.
//!
//! **An empty recipient set is not an error.** An action need not be directed
//! at anyone, and one addressed to nobody is logged as its sender's action and
//! delivered to nobody.
//!
//! A refused send is a **bug in the sending actor**: it fails that actor's
//! handler thread and surfaces through
//! [`EpisodeError::Agents`](crate::EpisodeError::Agents).
//!
//! Channels are unbounded. With bounded channels one actor slow to drain its
//! inbox would apply back-pressure through the router to every other actor in
//! the episode. A slow actor is a normal condition here and must not be able
//! to stall the world.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use crossbeam_channel::Sender;

use crate::message::{ActorId, Control, Message, Payload};

/// Why a message or a control could not be routed.
///
/// Each is an invariant of the system: an actor that trips one has a bug, and
/// its thread fails rather than carrying on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteError {
    /// A sender or recipient that is not in the roster.
    UnknownActor(ActorId),
    /// A sender that addressed itself. A reminder is the one way an actor
    /// reaches itself; see the [module documentation](self).
    Loopback(ActorId),
    /// A recipient whose channel has been dropped, so the copy for it could
    /// not be delivered.
    Closed(ActorId),
    /// A control whose sender is not the episode's environment.
    NotTheEnvironment(ActorId),
}

impl fmt::Display for RouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownActor(id) => write!(f, "no actor {id} in the roster"),
            Self::Loopback(id) => write!(f, "actor {id} addressed itself"),
            Self::Closed(id) => write!(f, "the channels of actor {id} are closed"),
            Self::NotTheEnvironment(id) => {
                write!(f, "actor {id} is not the environment and cannot command")
            }
        }
    }
}

impl Error for RouteError {}

/// Where to reach one actor: its inbox and its control channel.
///
/// `Debug` is written out because a derive would print the channels, which say
/// nothing a reader wants and change from run to run.
#[derive(Clone)]
pub struct Seat<P: Payload> {
    /// Where messages said to the actor go.
    pub inbox: Sender<Message<P>>,
    /// Where controls for the actor go, and which its perception thread
    /// checks before everything else.
    pub control: Sender<Control>,
}

impl<P: Payload> fmt::Debug for Seat<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Seat").finish_non_exhaustive()
    }
}

/// The map from actor id to that actor's channels, shared by every handler
/// thread in the episode.
#[derive(Debug)]
pub struct Router<P: Payload> {
    seats: BTreeMap<ActorId, Seat<P>>,
    environment: ActorId,
}

impl<P: Payload> Router<P> {
    /// A router over `seats`, whose `environment` is the one actor allowed to
    /// [`command`](Router::command).
    #[must_use]
    pub const fn new(seats: BTreeMap<ActorId, Seat<P>>, environment: ActorId) -> Self {
        Self { seats, environment }
    }

    /// The ids in the roster, in order.
    pub fn ids(&self) -> impl Iterator<Item = &ActorId> {
        self.seats.keys()
    }

    /// The environment's id: the one actor that may command.
    #[must_use]
    pub const fn environment(&self) -> &ActorId {
        &self.environment
    }

    /// Copies `message` onto the inbox of each of its recipients.
    ///
    /// **An empty recipient set delivers nothing and is not an error.** Every
    /// recipient is resolved before anything is sent, so a message addressed
    /// to a stranger delivers to nobody rather than to whoever happened to be
    /// named before it.
    ///
    /// # Errors
    ///
    /// - [`RouteError::Loopback`] if the sender is among the recipients;
    /// - [`RouteError::UnknownActor`] if the sender or a recipient is not in
    ///   the roster;
    /// - [`RouteError::Closed`] if a recipient's inbox has been dropped.
    ///   Recipients before it in the list have already received the message.
    pub fn route(&self, message: &Message<P>) -> Result<(), RouteError> {
        let Message {
            sender, recipients, ..
        } = message;
        if recipients.contains(sender) {
            return Err(RouteError::Loopback(sender.clone()));
        }
        let mut resolved = Vec::with_capacity(recipients.len());
        for id in recipients {
            resolved.push((id, self.seat(id)?));
        }
        for (id, seat) in resolved {
            seat.inbox
                .send(message.clone())
                .map_err(|_| RouteError::Closed(id.clone()))?;
        }
        Ok(())
    }

    /// Delivers `control` to each of `to`, having checked that `from` is the
    /// environment.
    ///
    /// The environment **may name itself**: an episode ends when the
    /// environment stops every actor, itself included, so there is no
    /// loopback rule here. A control is not a message and nothing about it is
    /// observed; what would be circular about an actor talking to itself does
    /// not arise.
    ///
    /// # Errors
    ///
    /// [`RouteError::NotTheEnvironment`] if `from` is not the environment this
    /// router was built with, and whatever [`command_as_episode`] returns.
    ///
    /// [`command_as_episode`]: Router::command_as_episode
    pub fn command(
        &self,
        from: &ActorId,
        to: &[ActorId],
        control: Control,
    ) -> Result<(), RouteError> {
        if *from != self.environment {
            return Err(RouteError::NotTheEnvironment(from.clone()));
        }
        self.command_as_episode(to, control)
    }

    /// Delivers `control` to each of `to` on the episode's own behalf.
    ///
    /// This is how an [`Episode`](crate::Episode) starts its environment, and
    /// how it stops everybody when it runs out of time or an actor leaves
    /// unbidden. It asks no questions about who is sending, because the
    /// episode is not an actor and is not playing.
    ///
    /// # Errors
    ///
    /// [`RouteError::UnknownActor`] for a recipient not in the roster, and
    /// [`RouteError::Closed`] if a recipient's control channel has been
    /// dropped. Recipients before it in the list have already received it.
    pub fn command_as_episode(&self, to: &[ActorId], control: Control) -> Result<(), RouteError> {
        let mut resolved = Vec::with_capacity(to.len());
        for id in to {
            resolved.push((id, self.seat(id)?));
        }
        for (id, seat) in resolved {
            seat.control
                .send(control)
                .map_err(|_| RouteError::Closed(id.clone()))?;
        }
        Ok(())
    }

    /// Whether `who` is somebody the environment may reward.
    ///
    /// A reward travels nowhere, but whom it may name is the same question
    /// about the same roster, so the router is where it is asked.
    ///
    /// # Errors
    ///
    /// [`RouteError::NotTheEnvironment`] if `from` is not the environment, and
    /// [`RouteError::UnknownActor`] if `who` is not in the roster.
    pub fn rewardable(&self, from: &ActorId, who: &ActorId) -> Result<(), RouteError> {
        if *from != self.environment {
            return Err(RouteError::NotTheEnvironment(from.clone()));
        }
        self.seat(who).map(|_| ())
    }

    /// The seat of `id`.
    fn seat(&self, id: &ActorId) -> Result<&Seat<P>, RouteError> {
        self.seats
            .get(id)
            .ok_or_else(|| RouteError::UnknownActor(id.clone()))
    }
}

#[cfg(test)]
mod tests {
    use crossbeam_channel::{Receiver, unbounded};

    use super::*;
    use crate::testing::TestPayload;

    const ENVIRONMENT: &str = "environment";

    /// A receiving end of both channels of one actor.
    struct Ears<P: Payload> {
        inbox: Receiver<Message<P>>,
        controls: Receiver<Control>,
    }

    /// A router over `ids` with `environment` in charge, and everybody's ears.
    fn roster<const N: usize>(
        ids: [&str; N],
    ) -> (Router<TestPayload>, BTreeMap<ActorId, Ears<TestPayload>>) {
        let mut seats = BTreeMap::new();
        let mut ears = BTreeMap::new();
        for id in ids {
            let (inbox, heard) = unbounded();
            let (control, told) = unbounded();
            seats.insert(ActorId::new(id), Seat { inbox, control });
            ears.insert(
                ActorId::new(id),
                Ears {
                    inbox: heard,
                    controls: told,
                },
            );
        }
        (Router::new(seats, ActorId::new(ENVIRONMENT)), ears)
    }

    fn said<const N: usize>(from: &str, to: [&str; N]) -> Message<TestPayload> {
        Message::new(from, to, 0, TestPayload::Step(1))
    }

    #[test]
    fn a_message_reaches_every_recipient_and_nobody_else() {
        let (router, ears) = roster(["a", "b", "c", ENVIRONMENT]);
        let message = said("a", ["b", "c"]);
        router.route(&message).unwrap();
        assert_eq!(ears[&ActorId::new("b")].inbox.recv().unwrap(), message);
        assert_eq!(ears[&ActorId::new("c")].inbox.recv().unwrap(), message);
        assert!(ears[&ActorId::new(ENVIRONMENT)].inbox.try_recv().is_err());
    }

    #[test]
    fn an_empty_recipient_set_delivers_nothing_and_is_not_an_error() {
        let (router, ears) = roster(["a", "b", ENVIRONMENT]);
        router.route(&said("a", [])).unwrap();
        assert!(ears[&ActorId::new("b")].inbox.try_recv().is_err());
    }

    #[test]
    fn a_sender_among_its_own_recipients_is_refused() {
        let (router, _) = roster(["a", "b", ENVIRONMENT]);
        assert_eq!(
            router.route(&said("a", ["a", "b"])),
            Err(RouteError::Loopback(ActorId::new("a")))
        );
    }

    #[test]
    fn a_stranger_among_the_recipients_delivers_to_nobody() {
        let (router, ears) = roster(["a", "b", ENVIRONMENT]);
        assert_eq!(
            router.route(&said("a", ["b", "stranger"])),
            Err(RouteError::UnknownActor(ActorId::new("stranger")))
        );
        assert!(
            ears[&ActorId::new("b")].inbox.try_recv().is_err(),
            "nothing is delivered when a recipient cannot be resolved"
        );
    }

    #[test]
    fn a_closed_inbox_is_reported() {
        let (router, mut ears) = roster(["a", "b", ENVIRONMENT]);
        ears.remove(&ActorId::new("b"));
        assert_eq!(
            router.route(&said("a", ["b"])),
            Err(RouteError::Closed(ActorId::new("b")))
        );
    }

    #[test]
    fn only_the_environment_commands() {
        let (router, ears) = roster(["a", ENVIRONMENT]);
        let to = [ActorId::new("a")];
        assert_eq!(
            router.command(&ActorId::new("a"), &to, Control::Start),
            Err(RouteError::NotTheEnvironment(ActorId::new("a")))
        );
        router
            .command(&ActorId::new(ENVIRONMENT), &to, Control::Start)
            .unwrap();
        assert_eq!(
            ears[&ActorId::new("a")].controls.recv().unwrap(),
            Control::Start
        );
    }

    #[test]
    fn the_environment_may_command_itself() {
        // Which is how an episode ends.
        let (router, ears) = roster(["a", ENVIRONMENT]);
        let environment = ActorId::new(ENVIRONMENT);
        let everybody = [ActorId::new("a"), environment.clone()];
        router
            .command(&environment, &everybody, Control::Stop)
            .unwrap();
        assert_eq!(ears[&environment].controls.recv().unwrap(), Control::Stop);
    }

    #[test]
    fn the_episode_commands_without_being_an_actor() {
        let (router, ears) = roster(["a", ENVIRONMENT]);
        let environment = ActorId::new(ENVIRONMENT);
        router
            .command_as_episode(std::slice::from_ref(&environment), Control::Start)
            .unwrap();
        assert_eq!(ears[&environment].controls.recv().unwrap(), Control::Start);
    }

    #[test]
    fn a_control_for_a_stranger_is_refused() {
        let (router, _) = roster(["a", ENVIRONMENT]);
        assert_eq!(
            router.command_as_episode(&[ActorId::new("nobody")], Control::Stop),
            Err(RouteError::UnknownActor(ActorId::new("nobody")))
        );
    }

    #[test]
    fn only_the_environment_rewards_and_only_somebody_in_the_roster() {
        let (router, _) = roster(["a", ENVIRONMENT]);
        let environment = ActorId::new(ENVIRONMENT);
        router.rewardable(&environment, &ActorId::new("a")).unwrap();
        assert_eq!(
            router.rewardable(&environment, &ActorId::new("nobody")),
            Err(RouteError::UnknownActor(ActorId::new("nobody")))
        );
        assert_eq!(
            router.rewardable(&ActorId::new("a"), &ActorId::new("a")),
            Err(RouteError::NotTheEnvironment(ActorId::new("a")))
        );
    }

    #[test]
    fn the_roster_is_the_ids_it_was_built_with() {
        let (router, _) = roster(["b", "a", ENVIRONMENT]);
        assert_eq!(
            router.ids().cloned().collect::<Vec<_>>(),
            [
                ActorId::new("a"),
                ActorId::new("b"),
                ActorId::new(ENVIRONMENT)
            ]
        );
        assert_eq!(router.environment(), &ActorId::new(ENVIRONMENT));
    }

    #[test]
    fn errors_explain_themselves() {
        assert_eq!(
            RouteError::UnknownActor(ActorId::new("a")).to_string(),
            "no actor a in the roster"
        );
        assert_eq!(
            RouteError::Loopback(ActorId::new("a")).to_string(),
            "actor a addressed itself"
        );
        assert_eq!(
            RouteError::Closed(ActorId::new("a")).to_string(),
            "the channels of actor a are closed"
        );
        assert_eq!(
            RouteError::NotTheEnvironment(ActorId::new("a")).to_string(),
            "actor a is not the environment and cannot command"
        );
    }
}
