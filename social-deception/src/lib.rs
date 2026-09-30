//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process. Every participant in it is an
//! **actor**: two threads, one that perceives and one that decides
//! (ADR-0016). The **perception thread** does only fast work — it receives
//! from the actor's inbox, from its reminder timer and from its control
//! channel, stamps what arrives with `Instant::now()`, logs it and forwards
//! it. The **handler thread** takes those observations one at a time, calls
//! one application function per observation, and carries out each thing that
//! function yields as it is yielded.
//!
//! The point of the second thread is that **perceiving never waits on
//! deciding.** An agent's sense of time is the times at which its
//! observations arrive, so an agent that stopped perceiving during a
//! multi-second model call would observe everything said during the call as
//! arriving when the call returned, erasing the spacing of the conversation
//! exactly when the conversation was busiest.
//!
//! What sits at the center is the shape that makes this a reinforcement
//! learning system: **a function from one observation to actions**, called
//! once per observation, in arrival order.
//!
//! # The two roles
//!
//! An actor plays one of two roles, and they differ only in what a call may
//! return:
//!
//! | Role | Trait | Returns |
//! |---|---|---|
//! | agent | [`Policy`] | [`Action`]s: send a message, or set a reminder |
//! | environment | [`Step`] | [`Effect`]s: an action, a control, or a reward |
//!
//! An agent cannot command or reward because [`Action`] has no variant for
//! either; the compiler says so, and the [`Router`]'s check that only the
//! environment commands stays as a backstop.
//!
//! # The vocabulary
//!
//! The runtime layer takes its names from the actor model so that the
//! reinforcement learning names stay exact at the layer above: an agent
//! chooses actions, an environment applies rules and assigns rewards, and
//! neither is a special kind of the other (ADR-0007, ADR-0016).
//!
//! | Term | Is |
//! |---|---|
//! | [`Message`] | in-domain data on the wire: sender, recipients, a per-sender sequence number and a payload |
//! | [`Envelope`] | a received message's origin and payload, for an actor relaying it inside its own payload |
//! | [`Control`] | out-of-domain data on the wire: start and stop, on a channel of their own, never seen by a handler |
//! | [`Observation`] | a [`Message`] that arrived, with the instant it arrived |
//! | [`Action`] | what an agent does: send a message, or set a [`Reminder`] |
//! | [`Effect`] | what an environment does: an [`Action`], a [`Control`], or a reward |
//! | [`Reminder`] | an address in time: a payload and a deadline, delivered back to the actor that set it as an ordinary message |
//! | [`Actor`] | a running participant: its id, its control sender, and the join handles of its two threads |
//! | [`Episode`] | the owner of one run: the wiring, the threads, the log writer, a time limit, and joining |
//! | [`Clock`] | the episode's one origin, chosen before any actor starts and shared by every actor and the log writer |
//!
//! [`Observation`] and [`Action`] are relative to an actor; on the wire there
//! are only messages and controls. The same [`Message`] is the sent action of
//! its sender and an observation of each of its recipients, joined across
//! actors by its sender and its sequence number, which is what lets a
//! trajectory be built from the log (ADR-0017).
//!
//! The runtime is generic over one parameter, the [`Payload`] a game's
//! messages carry. A game's reward type is a second parameter, but it appears
//! only where rewards are assigned — on [`Step`] and [`Effect`] — because a
//! reward is logged and never sent, so no message, actor or router ever holds
//! one (ADR-0016).
//!
//! # The modules, from the bottom up
//!
//! - [`clock`]: the episode [`Clock`], the one origin the log's offsets are
//!   measured from;
//! - [`message`]: the [`Payload`] a game's messages carry, the [`Message`]
//!   and [`Control`] that travel on the wire, and the [`Envelope`] a game
//!   puts in its payload to relay one;
//! - [`log`]: the records an actor produces and the [`Writer`] that hands
//!   each one to every [`Sink`] it was given;
//! - [`contract`]: the types an application sees — [`Observation`],
//!   [`Action`], [`Effect`], [`Reminder`], [`Policy`] and [`Step`];
//! - [`timer`]: the reminders an actor is holding and when they fire;
//! - [`router`]: the fixed roster from actor id to that actor's inbox and
//!   control channel;
//! - [`thread`]: the two threads themselves, and the [`Context`] that is the
//!   per-actor plumbing no application code sees;
//! - [`episode`]: the [`Episode`] that wires a roster, starts the
//!   environment, and waits on a completion channel with a hard time limit.
//!
//! [`Context`]: thread::Context
//!
//! On top of the runtime sits one game, [`werewolf`]: the roles, phases and
//! the [`werewolf::Message`] payload that a runtime [`Message`] carries in a
//! game of Werewolf, the [`Knowledge`](werewolf::Knowledge) a player folds
//! its observations into, the [`Strategy`](werewolf::Strategy) that picks its
//! moves, the [`Role`](werewolf::Role) whose rules say which moves it may
//! pick from and the [`Player`](werewolf::Player) that plays one as an agent,
//! the [`Game`](werewolf::Game) whose rules decide what is said to whom, and
//! the [`Moderator`](werewolf::Moderator), Werewolf's environment, which runs
//! the game and starts and stops the players. [`werewolf::run`] plays one
//! episode of it from a [`Config`](werewolf::Config), and the `werewolf`
//! binary is the command line for that.

pub mod clock;
pub mod contract;
pub mod episode;
pub mod log;
pub mod message;
pub mod router;
#[cfg(test)]
mod testing;
pub mod thread;
pub mod timer;
pub mod werewolf;

// The runtime's public vocabulary, which ADR-0016 fixes at the crate root:
// `Actor`, `Episode`, `Clock`, `Policy`, `Step`, `Observation`, `Action`,
// `Effect`, `Reminder`, `Envelope`, `Message`, `Control`, `ActorId`, `Router`.
// These are the names an application is written against, so where they live is
// part of the contract rather than an accident of the module layout: a name
// that moves back into a submodule is a breaking change, and this block is
// where it would have to be made.
pub use clock::Clock;
pub use contract::{Action, Effect, Observation, Policy, Reminder, Step};
pub use episode::{Episode, EpisodeError, Failure};
pub use message::{ActorId, Control, Envelope, Message, Payload};
pub use router::{RouteError, Router};
pub use thread::{Actor, ActorError, Context, Ended, UnstartedActor};
pub use timer::{Reminders, Timer};
// [`log::Policy`] is deliberately not re-exported here. The crate root is a
// shared vocabulary, and the name `Policy` is the framework's to give to the
// one function a user implements: [`contract::Policy`], which is what
// `Policy` means here. A sink's policy is read in the company of the sink it
// belongs to, where `log::Policy::Required` says what it means and collides
// with nothing.
pub use log::{
    ActionRecord, ControlRecord, CycleRecord, Elapsed, EpisodeRecord, JsonLines, Key,
    ObservationRecord, Record, RewardRecord, Sink, Sinks, UndeliveredRecord, UnsentRecord, Writer,
};
