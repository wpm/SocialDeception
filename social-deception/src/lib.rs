//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels.
//!
//! The crate has five modules:
//!
//! - [`event`]: the [`Event`] type that arrives on an agent's receiver;
//! - [`agent`]: the [`Agent`] thread that drains its inbox, folds the batch
//!   through a [`Handler`], and records what it saw and sent;
//! - [`timer`]: the [`TimerSource`] an agent's deadlines come from;
//! - [`trajectory`]: the records an agent's loop produces and the [`Writer`]
//!   that puts them on disk;
//! - [`clock`]: the episode [`Clock`] the records are timestamped with.

pub mod agent;
pub mod clock;
pub mod event;
pub mod timer;
pub mod trajectory;

pub use agent::{Agent, CycleReport, Delivery, Error, Handler, Outgoing, Wiring};
pub use clock::{Clock, Timestamp};
pub use event::{AgentId, Control, Event, Payload};
pub use timer::{ManualTimer, ManualTimerControl, TimerSource};
pub use trajectory::{CycleRecord, EventRecord, LogRecord, Seq, Writer};
