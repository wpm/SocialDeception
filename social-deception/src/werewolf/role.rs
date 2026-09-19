//! The roles a player can hold, and the sides of the game they belong to.

use std::fmt;

use serde::{Deserialize, Serialize};

/// A player's role, dealt at the start of an episode and revealed at death.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Role {
    /// Has no power beyond the day vote.
    Villager,
    /// Devours at night, together with the rest of the pack.
    Werewolf,
    /// Learns one player's [`Faction`] each night.
    Seer,
    /// Protects one player each night, never itself and never the same
    /// player two nights running.
    Doctor,
}

/// Which side of the game a player is on: both what a seer learns about
/// someone and what a winning side is.
///
/// The two are one type because they are one partition of the players. A
/// role that wins alone, or one that investigates falsely, would split them;
/// neither exists in this version.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Faction {
    /// Everyone who is not a werewolf.
    Village,
    /// The pack.
    Werewolves,
}

impl fmt::Display for Role {
    /// The role's name as written, the same spelling it serializes as.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Villager => "Villager",
            Self::Werewolf => "Werewolf",
            Self::Seer => "Seer",
            Self::Doctor => "Doctor",
        })
    }
}

impl fmt::Display for Faction {
    /// The faction's name as written, the same spelling it serializes as.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Village => "Village",
            Self::Werewolves => "Werewolves",
        })
    }
}

impl Role {
    /// The side this role plays for. Only a werewolf is on the werewolves'.
    #[must_use]
    pub const fn faction(self) -> Faction {
        match self {
            Self::Werewolf => Faction::Werewolves,
            Self::Villager | Self::Seer | Self::Doctor => Faction::Village,
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::testing::json;

    #[test]
    fn only_a_werewolf_is_on_the_werewolves_side() {
        assert_eq!(Role::Werewolf.faction(), Faction::Werewolves);
        for role in [Role::Villager, Role::Seer, Role::Doctor] {
            assert_eq!(role.faction(), Faction::Village, "{role:?}");
        }
    }

    #[test]
    fn a_role_displays_as_its_name() {
        for role in [Role::Villager, Role::Werewolf, Role::Seer, Role::Doctor] {
            assert_eq!(json(&role), json!(role.to_string()), "{role:?}");
        }
    }

    #[test]
    fn a_faction_displays_as_its_name() {
        for faction in [Faction::Village, Faction::Werewolves] {
            assert_eq!(json(&faction), json!(faction.to_string()), "{faction:?}");
        }
    }

    #[test]
    fn roles_and_factions_serialize_as_their_names() {
        assert_eq!(json(&Role::Seer), json!("Seer"));
        assert_eq!(json(&Faction::Werewolves), json!("Werewolves"));
        let role: Role = serde_json::from_value(json!("Doctor")).unwrap();
        assert_eq!(role, Role::Doctor);
    }
}
