//! Which player holds which role.
//!
//! Roles are fixed before an episode runs: there is a Rust type per player
//! type, and a type is fixed at construction, so a role cannot be dealt by a
//! message. The [`Assignment`] is the one value the typed players and the
//! moderator are built from, and the moderator's opening narration tells
//! each player what it already is.

use std::collections::{BTreeMap, BTreeSet};

use crate::event::AgentId;
use crate::werewolf::role::Role;

/// A role for every player, and the pack that follows from it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    roles: BTreeMap<AgentId, Role>,
    pack: BTreeSet<AgentId>,
}

impl Assignment {
    /// An assignment giving each player the role paired with it.
    ///
    /// The pack is every player assigned [`Role::Werewolf`].
    ///
    /// # Panics
    ///
    /// If a player appears more than once.
    pub fn new<I, A>(roles: I) -> Self
    where
        I: IntoIterator<Item = (A, Role)>,
        A: Into<AgentId>,
    {
        let mut assigned = BTreeMap::new();
        for (who, role) in roles {
            let who = who.into();
            assert!(
                assigned.insert(who.clone(), role).is_none(),
                "player {who} is assigned more than one role"
            );
        }
        let pack = assigned
            .iter()
            .filter(|(_, role)| **role == Role::Werewolf)
            .map(|(who, _)| who.clone())
            .collect();
        Self {
            roles: assigned,
            pack,
        }
    }

    /// The role of a player, or `None` for an id that is not a player.
    #[must_use]
    pub fn role(&self, who: &AgentId) -> Option<Role> {
        self.roles.get(who).copied()
    }

    /// The werewolves.
    #[must_use]
    pub fn pack(&self) -> &BTreeSet<AgentId> {
        &self.pack
    }

    /// Every player with its role, in agent-id order.
    pub fn players(&self) -> impl Iterator<Item = (&AgentId, Role)> {
        self.roles.iter().map(|(who, role)| (who, *role))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(name: &str) -> AgentId {
        AgentId::new(name)
    }

    fn assignment() -> Assignment {
        Assignment::new([
            ("carol", Role::Werewolf),
            ("alice", Role::Seer),
            ("bob", Role::Villager),
            ("dave", Role::Werewolf),
        ])
    }

    #[test]
    fn each_player_has_its_role() {
        let assignment = assignment();
        assert_eq!(assignment.role(&id("alice")), Some(Role::Seer));
        assert_eq!(assignment.role(&id("bob")), Some(Role::Villager));
        assert_eq!(assignment.role(&id("carol")), Some(Role::Werewolf));
        assert_eq!(assignment.role(&id("nobody")), None);
    }

    #[test]
    fn the_pack_is_exactly_the_werewolves() {
        assert_eq!(*assignment().pack(), ["carol", "dave"].map(id).into());
    }

    #[test]
    fn players_come_in_agent_order() {
        let players: Vec<_> = assignment()
            .players()
            .map(|(who, role)| (who.clone(), role))
            .collect();
        assert_eq!(
            players,
            [
                (id("alice"), Role::Seer),
                (id("bob"), Role::Villager),
                (id("carol"), Role::Werewolf),
                (id("dave"), Role::Werewolf),
            ]
        );
    }

    #[test]
    #[should_panic(expected = "player alice is assigned more than one role")]
    fn a_player_cannot_hold_two_roles() {
        Assignment::new([("alice", Role::Seer), ("alice", Role::Villager)]);
    }
}
