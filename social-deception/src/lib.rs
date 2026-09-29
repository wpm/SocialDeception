//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels.
//!
//! # The vocabulary
//!
//! The runtime speaks the vocabulary of reinforcement learning, which is
//! what the log it writes is read in (ADR-0007):
//!
//! | Term | Is |
//! |---|---|
//! | [`Message`] | in-domain data on the wire: sender, recipients, creation time and a payload |
//! | [`Control`] | out-of-domain data on the wire: start and stop |
//! | [`Delivery`] | either of the two, which is what an agent's one queue carries |
//! | [`Observation`] | a [`Message`] popped off an agent's queue |
//! | [`Action`] | what a handler returns for the loop to send |
//! | cycle | one turn of an agent's loop: pop one observation, hand it to the handler, send |
//! | [`Environment`] | the one agent per episode that starts and stops the others |
//!
//! [`Observation`] and [`Action`] are relative to an agent; on the wire
//! there are only messages and controls. The same [`Message`] is the sent
//! action of its sender and an observation of each of its recipients, which
//! is what lets the log be joined across agents.
//!
//! The runtime is generic over one parameter, the [`Payload`] a game's
//! messages carry. A game's reward type is a second parameter, but it
//! appears only where rewards are assigned — on [`Environment`] and
//! [`Effect`] — because a reward is logged and never sent, so no message,
//! agent or router ever holds one (ADR-0016).
//!
//! # The modules, from the bottom up
//!
//! - [`clock`]: the episode [`Clock`] everything is timestamped with, and
//!   the [`Created`] and [`Received`] traits that say what is known about
//!   a thing's time;
//! - [`message`]: the [`Payload`] a game's messages carry, the [`Message`]
//!   and [`Control`] that travel on the wire, and the [`Delivery`] that
//!   carries either of them to an agent;
//! - [`log`]: the records an agent's loop produces and the [`Writer`]
//!   that hands each one to every [`Sink`] it was given;
//! - [`timer`]: the [`TimerSource`] an agent's deadlines come from;
//! - [`agent`]: the [`Agent`] thread that pops its queue, folds the
//!   [`Observation`] it took through a [`Handler`], and records what it saw
//!   and sent;
//! - [`router`]: the [`Router`] from actor ids to their channels;
//! - [`environment`]: the [`Environment`], the one agent per episode whose
//!   cycle may produce a [`Control`] as well as an [`Action`];
//! - [`episode`]: the [`Episode`] that runs a roster and its environment
//!   from start to stop.
//!
//! On top of that runtime sits one game, [`werewolf`]: the roles, phases
//! and the [`werewolf::Message`] payload that a runtime [`Message`]
//! carries in a game of Werewolf, the [`Knowledge`](werewolf::Knowledge)
//! a player folds its observations into, the
//! [`Strategy`](werewolf::Strategy) that picks its moves, the
//! [`Role`](werewolf::Role) whose rules say which moves it may pick from and
//! the [`Player`](werewolf::Player) that plays one as an agent, the
//! [`Game`](werewolf::Game) whose rules decide what is
//! said to whom, and the [`Moderator`](werewolf::Moderator), Werewolf's
//! [`Environment`], which runs the game and starts and stops the players.
//! [`werewolf::run`] plays one episode of it from a
//! [`Config`](werewolf::Config), and the `werewolf` binary is the command
//! line for that.

pub mod agent;
pub mod clock;
pub mod environment;
pub mod episode;
pub mod log;
pub mod message;
pub mod router;
#[cfg(test)]
mod testing;
pub mod timer;
pub mod werewolf;

pub use agent::{Action, Agent, CycleDispatch, Error, Handler, Instruction, Observation, Wiring};
pub use clock::{Clock, Created, Received, Timestamp};
pub use environment::{Effect, Environment, Refusal};
pub use episode::{Episode, EpisodeError, Failure};
pub use message::{ActorId, Control, Delivery, Message, Payload};
pub use router::{Queues, RouteError, Router};
pub use timer::{ManualTimer, ManualTimerControl, TimerSource};
// [`log::Policy`] is deliberately not re-exported here. The crate root is a
// shared vocabulary, and the name `Policy` is the framework's to give to the
// one function a user implements. A sink's policy is read in the company of
// the sink it belongs to, where `log::Policy::Required` says what it means
// and collides with nothing.
pub use log::{
    ActionRecord, ControlRecord, CycleRecord, JsonLines, ObservationRecord, Record, RewardRecord,
    Seq, Sink, Woken, Writer,
};
