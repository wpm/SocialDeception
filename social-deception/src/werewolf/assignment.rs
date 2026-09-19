//! Dealing roles to players, deterministically from the seed.
//!
//! Roles are dealt before the episode is built, not by a message during it:
//! there is a Rust type per player type, and a type is fixed at
//! construction (ADR-0004). The deal is a function of the configuration
//! alone, so the same seed and roster give the same [`Assignment`] on every
//! run, on every machine, and after any dependency bump. The golden test in
//! this module pins one deal so that a change to the shuffle or to the seed
//! derivation is caught rather than silently reproducing a different
//! experiment.

use std::collections::{BTreeMap, BTreeSet};
use std::iter::repeat_n;

use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use super::config::Config;
use super::role::{Faction, Role};
use super::seed::{ASSIGNMENT, seed_for};
use crate::event::AgentId;

/// Which player holds which role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    roles: BTreeMap<AgentId, Role>,
    pack: BTreeSet<AgentId>,
}

impl Assignment {
    /// Deals the roles the configuration asks for to its players.
    ///
    /// The role multiset is built in a fixed order, werewolves then seers
    /// then doctors then villagers, and shuffled with a
    /// [`ChaCha8Rng`] seeded from [`seed_for`]`(config.seed, `[`ASSIGNMENT`]`)`
    /// against the players *sorted*, so reordering the players in the file
    /// does not change the deal. Adding or removing a player does reshuffle
    /// the whole deal, which is expected: unlike a player's own choice
    /// stream, the deal is a property of the whole roster.
    ///
    /// # Panics
    ///
    /// If the configuration has more special roles than players, which
    /// [`Config::validate`] rejects.
    #[must_use]
    pub fn deal(config: &Config) -> Self {
        let players: BTreeSet<&AgentId> = config.players.iter().collect();
        let counts = config.roles;
        let villagers = players
            .len()
            .checked_sub(counts.special())
            .expect("a validated configuration has at least as many players as special roles");
        let mut roles: Vec<Role> = repeat_n(Role::Werewolf, counts.werewolves)
            .chain(repeat_n(Role::Seer, counts.seers))
            .chain(repeat_n(Role::Doctor, counts.doctors))
            .chain(repeat_n(Role::Villager, villagers))
            .collect();
        let mut rng = ChaCha8Rng::seed_from_u64(seed_for(config.seed, ASSIGNMENT));
        // `ChaCha8Rng` and `seed_from_u64` are value-stable across `rand`
        // releases; `shuffle` is not, so a `rand` bump can change the deal
        // for the same seed. The golden test below is what catches that. If
        // it ever fires for that reason, replace this call with a hand-rolled
        // Fisher-Yates over `rng.next_u64()`, which pins the permutation by
        // construction, and keep the golden value that the hand-rolled
        // version produces.
        roles.shuffle(&mut rng);
        let roles: BTreeMap<AgentId, Role> = players.into_iter().cloned().zip(roles).collect();
        let pack = roles
            .iter()
            .filter(|(_, role)| role.faction() == Faction::Werewolves)
            .map(|(who, _)| who.clone())
            .collect();
        Self { roles, pack }
    }

    /// An assignment giving each player the role paired with it, for a deal
    /// decided by something other than the seed, such as a test.
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
        let mut dealt = BTreeMap::new();
        for (who, role) in roles {
            let who = who.into();
            assert!(
                dealt.insert(who.clone(), role).is_none(),
                "player {who} is assigned more than one role"
            );
        }
        let pack = dealt
            .iter()
            .filter(|(_, role)| role.faction() == Faction::Werewolves)
            .map(|(who, _)| who.clone())
            .collect();
        Self { roles: dealt, pack }
    }

    /// The role dealt to `who`, or `None` if `who` is not a player.
    #[must_use]
    pub fn role(&self, who: &AgentId) -> Option<Role> {
        self.roles.get(who).copied()
    }

    /// The werewolves.
    #[must_use]
    pub const fn pack(&self) -> &BTreeSet<AgentId> {
        &self.pack
    }

    /// Every player with its role, in agent-id order.
    pub fn players(&self) -> impl Iterator<Item = (&AgentId, Role)> {
        self.roles.iter().map(|(who, role)| (who, *role))
    }

    /// How many players were dealt `role`.
    #[must_use]
    pub fn count(&self, role: Role) -> usize {
        self.roles.values().filter(|dealt| **dealt == role).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::werewolf::config::RoleCounts;

    fn game(seed: u64, players: &[&str], roles: RoleCounts) -> Config {
        let config = Config {
            seed,
            players: players.iter().copied().map(AgentId::new).collect(),
            roles,
            trajectory: None,
            max_rounds: 100,
            moderator: AgentId::new("moderator"),
        };
        config.validate().unwrap();
        config
    }

    const SEVEN: [&str; 7] = ["alice", "bob", "carol", "dave", "erin", "frank", "grace"];
    const FULL_HOUSE: RoleCounts = RoleCounts {
        werewolves: 2,
        seers: 1,
        doctors: 1,
    };

    fn roster(assignment: &Assignment) -> Vec<(&str, Role)> {
        assignment
            .players()
            .map(|(who, role)| (who.as_str(), role))
            .collect()
    }

    #[test]
    fn the_same_seed_gives_the_same_deal() {
        let config = game(20_260_918, &SEVEN, FULL_HOUSE);
        let first = Assignment::deal(&config);
        assert_eq!(Assignment::deal(&config), first, "repeated call");
        let rebuilt = game(20_260_918, &SEVEN, FULL_HOUSE);
        assert_eq!(Assignment::deal(&rebuilt), first, "rebuilt config");
    }

    #[test]
    fn a_different_seed_gives_a_different_deal() {
        let one = Assignment::deal(&game(1, &SEVEN, FULL_HOUSE));
        let two = Assignment::deal(&game(2, &SEVEN, FULL_HOUSE));
        assert_ne!(one, two);
    }

    #[test]
    fn reordering_the_players_does_not_change_the_deal() {
        let written = game(20_260_918, &SEVEN, FULL_HOUSE);
        let reversed: Vec<&str> = SEVEN.iter().rev().copied().collect();
        let reordered = game(20_260_918, &reversed, FULL_HOUSE);
        assert_eq!(Assignment::deal(&written), Assignment::deal(&reordered));
    }

    #[test]
    fn the_counts_match_the_config_and_the_pack_is_the_werewolves() {
        for seed in 0..50 {
            let assignment = Assignment::deal(&game(seed, &SEVEN, FULL_HOUSE));
            assert_eq!(assignment.count(Role::Werewolf), 2, "seed {seed}");
            assert_eq!(assignment.count(Role::Seer), 1, "seed {seed}");
            assert_eq!(assignment.count(Role::Doctor), 1, "seed {seed}");
            assert_eq!(assignment.count(Role::Villager), 3, "seed {seed}");
            assert_eq!(assignment.players().count(), 7, "seed {seed}");
            let werewolves: BTreeSet<AgentId> = assignment
                .players()
                .filter(|(_, role)| *role == Role::Werewolf)
                .map(|(who, _)| who.clone())
                .collect();
            assert_eq!(*assignment.pack(), werewolves, "seed {seed}");
        }
    }

    #[test]
    fn every_player_has_a_role_and_nobody_else_does() {
        let assignment = Assignment::deal(&game(3, &SEVEN, FULL_HOUSE));
        for who in SEVEN {
            assert!(assignment.role(&AgentId::new(who)).is_some(), "{who}");
        }
        assert_eq!(assignment.role(&AgentId::new("moderator")), None);
        assert_eq!(assignment.role(&AgentId::new("")), None);
    }

    #[test]
    fn a_game_of_only_special_roles_has_no_villagers() {
        let counts = RoleCounts {
            werewolves: 1,
            seers: 1,
            doctors: 1,
        };
        let assignment = Assignment::deal(&game(9, &["alice", "bob", "carol"], counts));
        assert_eq!(assignment.count(Role::Villager), 0);
        assert_eq!(assignment.players().count(), 3);
    }

    #[test]
    fn golden_deal() {
        // Change the shuffle, the generator or `seed_for` and this changes.
        let assignment = Assignment::deal(&game(20_260_918, &SEVEN, FULL_HOUSE));
        assert_eq!(
            roster(&assignment),
            [
                ("alice", Role::Werewolf),
                ("bob", Role::Doctor),
                ("carol", Role::Villager),
                ("dave", Role::Villager),
                ("erin", Role::Villager),
                ("frank", Role::Werewolf),
                ("grace", Role::Seer),
            ]
        );
    }
}
