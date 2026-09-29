//! The actor runtime: every participant perceives on one thread and decides
//! on another (ADR-0016).
//!
//! An actor is two threads. The **perception thread** does only fast work:
//! it receives from the inbox, from its reminder timer and from its control
//! channel, stamps what arrives with `Instant::now()`, logs it and forwards
//! it. The **handler thread** takes those observations one at a time, calls
//! one application function per observation, and carries out each thing that
//! function yields as it is yielded.
//!
//! The point of the second thread is that **perceiving never waits on
//! deciding.** Under the old runtime an agent was one thread, so an agent
//! blocked in a multi-second model call popped nothing, and everything said
//! during the call was observed as arriving when the call returned. An
//! agent's sense of time is when its observations arrive, so the loop erased
//! the spacing of the conversation exactly when the conversation was busiest.
//!
//! What stays is the shape that makes this a reinforcement learning system:
//! **a function from one observation to actions**, called once per
//! observation, in arrival order.
//!
//! # The two roles
//!
//! [`Agent`] and [`Environment`] are roles an actor plays, and they differ
//! only in what a call may return:
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
//! [`Agent`]: Policy
//! [`Environment`]: Step
//!
//! # The modules
//!
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
//! [`Episode`]: episode::Episode
//! [`Router`]: router::Router

pub mod contract;
pub mod episode;
pub mod router;
pub mod thread;
pub mod timer;

pub use contract::{Action, Effect, Observation, Policy, Reminder, Step};
pub use episode::{Episode, EpisodeError, Failure};
pub use router::{RouteError, Router};
pub use thread::{Actor, ActorError, Context, Ended, UnstartedActor};
pub use timer::{Reminders, Timer};
