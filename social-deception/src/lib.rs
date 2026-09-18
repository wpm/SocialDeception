//! Social Deception: a framework for environments in which several LLM agents
//! interact in real time without turn-taking.
//!
//! An episode runs as a single process with one thread per agent, communicating
//! over in-process channels. See `docs/decision-history/` for the decisions
//! behind that shape.
//!
//! The crate is organised around three ideas from ADR-0001:
//!
//! - every agent has exactly one receiver, and everything that can happen to it
//!   arrives there as an [`Event`];
//! - the fold is the log: an agent's own event sequence is its trajectory, and
//!   the agent loop records it as it goes ([`trajectory`]);
//! - times come from one monotonic clock per episode ([`clock`]).

pub mod clock;
pub mod event;
pub mod trajectory;

pub use clock::{Clock, Timestamp};
pub use event::{AgentId, Control, Event, Payload};
pub use trajectory::{CycleRecord, EventRecord, LogRecord, Seq, Writer};
