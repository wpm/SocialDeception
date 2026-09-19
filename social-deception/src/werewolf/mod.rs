//! Werewolf: the types every part of the game is written against, the
//! setup that happens before an episode runs, and the rules of the game as
//! a pure state machine.
//!
//! The vocabulary is the [`Role`]s a player can be dealt, the [`Faction`]s
//! they play for, the [`Round`] and [`Phase`] that locate a moment in a
//! game, and [`Message`], the one payload type that travels over the
//! runtime's [`Event`](crate::Event) between the moderator and the players.
//! The only facts it states are properties of a request kind itself, such as
//! which phase it belongs to; every rule that depends on who is alive lives
//! with the moderator and the roles, not here.
//!
//! The setup is a [`Config`] read from a TOML file ([`config`]), the
//! [`Assignment`] of roles dealt from its seed ([`assignment`]), and
//! [`seed_for`], which derives every generator's seed from the master seed
//! ([`seed`]). The `werewolf` binary loads a configuration, deals the roles
//! and prints the result.
//!
//! The rules live in [`Game`] ([`game`]), which takes an [`Assignment`] and
//! plays the game as a fold over players' responses, producing
//! [`Directive`]s that say what to tell whom. The [`Moderator`]
//! ([`moderator`]) is the agent that runs a game: the thin
//! [`Handler`](crate::Handler) that folds the events on its inbox into the
//! game and sends the directives as messages.
//!
//! A trajectory written by a run reads back as a [`Transcript`]
//! ([`transcript`]): the logical game, with the timestamps and the
//! interleaving of agents' records projected out, so that two transcripts
//! are equal exactly when the same game was played. The `werewolf` binary's
//! `replay` renders one.
//!
//! # Three kinds of message
//!
//! | Message | Direction | Is |
//! |---|---|---|
//! | [`Narration`] | moderator → a chosen set of players | a true statement the recipients now observe |
//! | [`Request`] | moderator → one player | a decision point: the moment a policy is invoked |
//! | [`Response`] | player → moderator | the reply, echoing the request's id and carrying one [`Action`] |
//!
//! In the reinforcement-learning vocabulary of the design, `Event<Message>`
//! is the observation type and [`Action`] is the action type. The set of
//! moves the rules permit for one request is the *action space*, a
//! `Vec<Action>` computed by the rules; an action outside it is a policy bug.
//! A request and its response are correlated by [`RequestId`] on purpose:
//! they are RPC-shaped, and a response naming an id the moderator is not
//! waiting for is a bug rather than a judgement call.
//!
//! # Narration is addressed, not broadcast
//!
//! No message here names its recipients, because that is the moderator's
//! decision: a narration goes to one player, to the living, or to the pack,
//! and that choice of recipients is the whole hidden-information mechanism
//! (see ADR-0004). A player observes many narrations but acts only on a
//! request, which is what gives an agent in a turnless runtime its decision
//! points.
//!
//! Every narration is true. Player-to-player dialogue, which may be false,
//! would be a fourth kind of message and is not defined here.
//!
//! # What a player knows, and how it decides
//!
//! [`Knowledge`] is the state a player carries between passes: the fold of
//! every observation it has received, and what a policy conditions on. It
//! records only what the moderator said, so nothing in it can be false.
//!
//! A [`Policy`] is handed a [`View`] of that state, the request in front of
//! it and the action space, and returns one [`Action`]. [`RandomPolicy`] is
//! the uniform random baseline; a language-model policy is the same trait
//! ([`policy`]).
//!
//! The action space is the rules' to compute, and the rules are a role's:
//! [`Villager`], [`Werewolf`], [`Seer`] and [`Doctor`] ([`roles`]) each
//! carry their own [`Knowledge`] and say which actions a request permits
//! them, and nothing else. A [`Seat`] ([`player`]) pairs a role with the
//! policy that decides for it and is the agent the episode runs: it folds
//! every event into the role's state, answers each request with the
//! policy's choice from the role's action space, and addresses nobody but
//! the moderator.
//!
//! # What no message carries
//!
//! No message carries the episode's seed, and no type here has a field that
//! could. With the seed, the roster and the public algorithm, anyone could
//! recompute the deal and every agent's random stream. The seed is recorded
//! beside the trajectory, never in it.
//!
//! # Serialization
//!
//! Every type here is `Serialize` and `Deserialize`, so a trajectory's
//! payloads can be read back as typed data. Collections are `BTreeMap` and
//! `BTreeSet`, so serialization is in a canonical order and two runs of the
//! same seed produce byte-identical payloads.
//!
//! The payload uses serde's defaults: enums are externally tagged with the
//! variant name as written, and newtypes are transparent. The runtime's
//! envelope, [`Event`](crate::Event) and its records, is internally tagged
//! and snake-cased instead, because its field names are a contract with
//! whatever reads a trajectory back; the payload's shape belongs to the
//! environment alone.

pub mod assignment;
pub mod config;
pub mod game;
pub mod knowledge;
pub mod message;
pub mod moderator;
pub mod player;
pub mod policy;
pub mod role;
pub mod roles;
pub mod seed;
pub mod transcript;

pub use assignment::Assignment;
pub use config::{Config, ConfigError, RoleCounts};
pub use game::{Directive, Game};
pub use knowledge::{Death, Heard, Knowledge};
pub use message::{
    Action, Cause, Message, Narration, Outcome, Phase, Request, RequestId, RequestKind, Response,
    Round,
};
pub use moderator::Moderator;
pub use player::{Player, Seat};
pub use policy::{Policy, RandomPolicy, View};
pub use role::{Faction, Role};
pub use roles::{Doctor, Seer, Villager, Werewolf};
pub use seed::seed_for;
pub use transcript::{PhaseRecord, RoundRecord, Transcript, TranscriptError};
