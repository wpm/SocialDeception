//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels.
//!
//! The crate's modules, from the bottom up:
//!
//! - [`clock`]: the episode [`Clock`] everything is timestamped with;
//! - [`event`]: the [`Event`] type that arrives on an agent's receiver;
//! - [`trajectory`]: the records an agent's loop produces and the [`Writer`]
//!   that puts them on disk;
//! - [`timer`]: the [`TimerSource`] an agent's deadlines come from;
//! - [`agent`]: the [`Agent`] thread that drains its inbox, folds the batch
//!   through a [`Handler`], and records what it saw and sent;
//! - [`router`]: the [`Router`] from agent ids to their channels;
//! - [`episode`]: the [`Episode`] that runs a roster from start to stop.

pub mod agent;
pub mod clock;
pub mod episode;
pub mod event;
pub mod router;
#[cfg(test)]
mod testing;
pub mod timer;
pub mod trajectory;

pub use agent::{Agent, CycleReport, Delivery, Error, Handler, Outgoing, Recipients, Wiring};
pub use clock::{Clock, Timestamp};
pub use episode::{Episode, EpisodeError, Failure};
pub use event::{AgentId, Control, Event, Payload};
pub use router::{RouteError, Router};
pub use timer::{ManualTimer, ManualTimerControl, TimerSource};
pub use trajectory::{CycleRecord, EventRecord, LogRecord, Seq, Writer};
