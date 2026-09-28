//! Werewolf: the types every part of the game is written against, the
//! setup that happens before an episode runs, and the rules of the game as
//! a pure state machine.
//!
//! The vocabulary is the [`Role`]s a player can be dealt, the [`Faction`]s
//! they play for, the [`Round`] and [`Phase`] that locate a moment in a
//! game, and [`Message`], the one payload type that travels over the
//! runtime's [`Event`](crate::Event) between the moderator and the players.
//! The only facts it states are properties of a session kind itself, such
//! as which phase it belongs to; every rule that depends on who is alive
//! lives with the moderator and the roles, not here.
//!
//! The setup is a [`Config`] read from a TOML file ([`config`]), the
//! [`Assignment`] of roles dealt from its seed ([`assignment`]), and
//! [`seed_for`], which derives every generator's seed from the master seed
//! ([`seed`]).
//!
//! The rules live in [`Game`] ([`game`]), which takes an [`Assignment`] and
//! plays the game as a fold over players' responses, producing
//! [`Directive`]s that say what to tell whom. The [`Moderator`]
//! ([`moderator`]) is the agent that runs a game: the thin
//! [`Environment`](crate::Environment) that folds the observations it pops
//! into the game, sends the directives as messages, and — being the
//! episode's environment — starts the players when it begins and stops them
//! when the game is over.
//!
//! The seam with the runtime is [`setup`]: [`episode`] builds a populated
//! [`Episode`](crate::Episode) from a [`Config`], seating every player and
//! the moderator, and [`run`] runs one to its [`Outcome`] or a [`RunError`].
//! The `werewolf` binary's `play` is that, with the effective configuration
//! written beside the trajectory.
//!
//! A trajectory written by a run reads back as a [`Transcript`]
//! ([`transcript`]): the logical game, with the timestamps and the
//! interleaving of agents' records projected out, so that two transcripts
//! are equal exactly when the same game was played. The `werewolf` binary's
//! `replay` renders one.
//!
//! # Two kinds of message
//!
//! | Message | Direction | Is |
//! |---|---|---|
//! | [`Narration`] | moderator → a chosen set of players | a true statement the recipients now observe |
//! | [`Point`] | player → the moderator and whoever else may see it | a target, which the player may revise |
//!
//! There is no third kind, and in particular nothing that asks a player to
//! act. A player observes that a phase has begun and consults its own role
//! (ADR-0014); being told what its own role already says would inform it of
//! nothing.
//!
//! In the reinforcement-learning vocabulary of the design, `Event<Message>`
//! is the observation type and the move a player's action carries is an
//! [`AgentId`](crate::AgentId): the target inside the point, not the point
//! itself. The set of targets the rules permit in a session is the *action
//! space*, a `Vec<AgentId>` computed by the rules; a target outside it is a
//! policy bug. A point names the session it was made in, by round and
//! [`RequestKind`], which both sides derive from what they each know — so
//! there is nothing to correlate, and a point naming a round that has
//! passed is one whose session closed while it was in flight.
//!
//! A member may point as often as it likes while its session is open, and
//! its most recent point is its vote; pointing nowhere is how it abstains,
//! which is why there is no move meaning "nobody" (ADR-0011).
//!
//! # Nothing is broadcast
//!
//! No message here names its recipients, because that is the sender's
//! decision: a narration goes to one player, to the living, or to the pack,
//! and that choice of recipients is the whole hidden-information mechanism
//! (see ADR-0004). A player acts on observing that a phase has begun,
//! which is what gives an agent in a turnless runtime its decision points:
//! nobody asks it to, and it works out from its own role whether the
//! phase asks anything of it (ADR-0014).
//!
//! A point is addressed the same way, by the player making it: a `Devour`
//! to the living pack, a `Nominate` to every other living player, and the
//! seer's and the doctor's to the moderator alone. That is how a pack
//! agrees on a victim without speaking and how a village's vote forms in
//! the open (ADR-0011).
//!
//! Nothing is excepted. ADR-0004 broadcast the final [`Outcome`] to every
//! player, living and dead, because it was a dead player's terminal reward
//! signal; a reward is now logged rather than said (ADR-0007), so the
//! outcome is narrated to the living like everything else and a dead
//! player hears nothing after the announcement of its own death.
//!
//! Every narration is true. Player-to-player dialogue, which may be false,
//! would be a fourth kind of message and is not defined here.
//!
//! # What a player knows, and how it decides
//!
//! [`Knowledge`] is the state a player carries between cycles: the fold of
//! every observation it has received, and what a policy conditions on. It
//! records only what the moderator said, so nothing in it can be false.
//!
//! A [`Policy`] is handed a [`View`] of that state, the kind of session in
//! front of it and the action space, and returns a target or none.
//! [`RandomPolicy`] is the uniform random baseline; a language-model
//! policy is the same trait ([`policy`]).
//!
//! The action space is the rules' to compute, and the rules are a role's:
//! [`Villager`], [`Werewolf`], [`Seer`] and [`Doctor`] ([`roles`]) each
//! carry their own [`Knowledge`] and say which targets a session permits
//! them, and nothing else. It may be empty — the doctor may be left with
//! nobody it can protect — and a player with an empty one points nowhere.
//! A [`Seat`] ([`player`]) pairs a role with the policy that decides for
//! it and is the agent the episode runs: it folds every event into the
//! role's state, acts when it observes a phase begin, and addresses each
//! point as the rules allow.
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

use crate::event::Domain;

pub mod assignment;
pub mod config;
pub mod game;
pub mod knowledge;
pub mod live;
pub mod message;
pub mod moderator;
pub mod player;
pub mod policy;
pub mod role;
pub mod roles;
pub mod seed;
pub mod setup;
pub mod transcript;

/// Werewolf as a [`Domain`]: the types this game contributes to the
/// runtime.
///
/// Its events carry a [`Message`], and a player's reward is an integer,
/// because a game of Werewolf is won or lost and nothing finer is scored.
///
/// The name is not `Werewolf`, which is the role a player may be dealt. A
/// domain is the whole game; the role is one thing inside it, and the two
/// would be hard to tell apart in a signature if they shared a name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WerewolfDomain;

impl Domain for WerewolfDomain {
    type Payload = Message;
    type Reward = i32;
}

pub use assignment::Assignment;
pub use config::{Config, ConfigError, RoleCounts};
pub use game::{Directive, Game};
pub use knowledge::{Death, Knowledge, Phased};
pub use live::Text;
pub use message::{Cause, Message, Narration, Outcome, Phase, Point, RequestKind, Round};
pub use moderator::Moderator;
pub use player::{Player, Seat};
pub use policy::{Policy, RandomPolicy, View};
pub use role::{Faction, Role};
pub use roles::{Doctor, Seer, Villager, Werewolf};
pub use seed::seed_for;
pub use setup::{RunError, episode, run};
pub use transcript::{PhaseRecord, RoundRecord, Transcript, TranscriptError};
