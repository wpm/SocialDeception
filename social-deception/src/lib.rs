//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels.
//!
//! # The vocabulary
//!
//! The runtime speaks the vocabulary of reinforcement learning, which is
//! what the trajectories it writes are read in (ADR-0007):
//!
//! | Term | Is |
//! |---|---|
//! | [`Event`] | in-domain data on the wire: sender, recipients, creation time and a payload |
//! | [`Control`] | out-of-domain data on the wire: start and stop |
//! | [`Observation`] | an [`Event`] popped off an agent's queue |
//! | [`Action`] | what a handler returns for the loop to send |
//! | [`Domain`] | the types one game contributes: its payload and its reward |
//! | cycle | one turn of an agent's loop: pop, hand to the handler, send |
//! | [`Cancel`] | the handler's view of whether its cycle has been preempted |
//!
//! [`Observation`] and [`Action`] are relative to an agent; on the wire
//! there are only events and controls. The same [`Event`] is the sent action
//! of its sender and an observation of each of its recipients, which is what
//! lets a trajectory be joined across agents.
//!
//! The runtime is generic over one parameter, a [`Domain`], rather than over
//! each of a game's types separately.
//!
//! # The modules, from the bottom up
//!
//! - [`clock`]: the episode [`Clock`] everything is timestamped with, and
//!   the [`Created`] and [`Timestamped`] traits that say what is known about
//!   a thing's time;
//! - [`event`]: the [`Domain`] a game names its types with, and the
//!   [`Event`] and [`Control`] that travel on the wire;
//! - [`trajectory`]: the records an agent's loop produces and the [`Writer`]
//!   that puts them on disk;
//! - [`timer`]: the [`TimerSource`] an agent's deadlines come from;
//! - [`cancel`]: the [`Cancel`] a handler is given each cycle and the
//!   [`ControlSender`] that trips it;
//! - [`agent`]: the [`Agent`] thread that pops its two queues, folds the
//!   [`Observation`]s through a [`Handler`], and records what it saw and
//!   sent;
//! - [`router`]: the [`Router`] from agent ids to their channels;
//! - [`episode`]: the [`Episode`] that runs a roster from start to stop.
//!
//! On top of that runtime sits one environment, [`werewolf`], whose
//! [`WerewolfDomain`](werewolf::WerewolfDomain) names its types: the roles,
//! phases and the [`Message`](werewolf::Message) payload that travels over
//! [`Event`] in a game of Werewolf, the [`Knowledge`](werewolf::Knowledge)
//! a player folds its observations into, the [`Policy`](werewolf::Policy)
//! that picks its moves, the roles ([`werewolf::roles`]) whose rules say
//! which moves it may pick from and the [`Seat`](werewolf::Seat) that plays
//! one as an agent, the [`Game`](werewolf::Game) whose rules decide what is
//! said to whom, and the [`Moderator`](werewolf::Moderator) that runs it as
//! an agent in the roster. [`werewolf::run`] plays one episode of it from a
//! [`Config`](werewolf::Config), and the `werewolf` binary is the command
//! line for that.

pub mod agent;
pub mod cancel;
pub mod clock;
pub mod episode;
pub mod event;
pub mod router;
#[cfg(test)]
mod testing;
pub mod timer;
pub mod trajectory;
pub mod werewolf;

pub use agent::{
    Action, Agent, CycleDispatch, Error, Handler, Instruction, Observation, Recipients, Wiring,
};
pub use cancel::{Arm, Cancel, ControlSender, Never, Signal, Trip};
pub use clock::{Clock, Created, Timestamp, Timestamped};
pub use episode::{Episode, EpisodeError, Failure};
pub use event::{AgentId, Control, Domain, Event, Payload};
pub use router::{Queues, RouteError, Router};
pub use timer::{ManualTimer, ManualTimerControl, TimerSource};
pub use trajectory::{
    ActionRecord, ControlRecord, CycleRecord, DroppedRecord, LogRecord, ObservationRecord, Seq,
    Woken, Writer,
};
