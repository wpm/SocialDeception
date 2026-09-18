//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels.
//!
//! The crate has three modules:
//!
//! - [`event`]: the [`Event`] type that arrives on an agent's receiver;
//! - [`trajectory`]: the records an agent's loop produces and the [`Writer`]
//!   that puts them on disk;
//! - [`clock`]: the episode [`Clock`] the records are timestamped with.

pub mod clock;
pub mod event;
pub mod trajectory;

pub use clock::{Clock, Timestamp};
pub use event::{AgentId, Control, Event, Payload};
pub use trajectory::{CycleRecord, EventRecord, LogRecord, Seq, Writer};
