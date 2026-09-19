//! Deriving one stream's seed from the episode's master seed.
//!
//! Everything a Werewolf episode draws at random comes from a
//! `ChaCha8Rng`, and every one of those generators is seeded from the one
//! master seed mixed with a label naming the stream: the role deal, each
//! player's own policy, the moderator's tie-break. Separate streams are what
//! keep the agents independent of each other and of thread scheduling: an
//! agent's actions depend on its own history alone, and adding a player does
//! not perturb anyone else's draws.
//!
//! The mix is written out here rather than borrowed from `std`'s
//! `DefaultHasher` or from `rand`, because neither guarantees the same output
//! across releases, and a seed that stops meaning the same thing after a
//! dependency bump is not a reproducible experiment. It is an FNV-1a pass
//! over the label's bytes, combined with the master seed and finalized with
//! splitmix64's mixing function.

/// Mixes a master seed with a label into a stable per-stream seed.
///
/// The labels used across the game are the agent's own id for a player's
/// policy, `"assignment"` for the role deal, and `"moderator:ties"` for the
/// moderator's tie-break generator. The function is case-sensitive in the
/// label, is not the identity in the master seed, and its output is pinned
/// by test: changing it changes every experiment ever run.
#[must_use]
pub fn seed_for(master: u64, label: &str) -> u64 {
    const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = FNV_OFFSET;
    for byte in label.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    splitmix64(master ^ hash)
}

/// The output function of the splitmix64 generator, used as a finalizer.
fn splitmix64(state: u64) -> u64 {
    let mut z = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn is_not_the_identity() {
        for master in [0, 1, 20_260_918, u64::MAX] {
            assert_ne!(seed_for(master, "alice"), master);
            assert_ne!(seed_for(master, ""), master);
        }
    }

    #[test]
    fn distinguishes_every_label_the_game_uses() {
        let labels = [
            "assignment",
            "moderator:ties",
            "moderator",
            "alice",
            "bob",
            "carol",
        ];
        let seeds: BTreeSet<u64> = labels.iter().map(|label| seed_for(7, label)).collect();
        assert_eq!(seeds.len(), labels.len());
    }

    #[test]
    fn is_case_sensitive_in_the_label() {
        assert_ne!(seed_for(7, "alice"), seed_for(7, "Alice"));
    }

    #[test]
    fn depends_on_the_master_seed() {
        assert_ne!(seed_for(1, "alice"), seed_for(2, "alice"));
    }

    #[test]
    fn is_pinned() {
        // Golden values. Change these only knowingly: every seeded
        // experiment depends on them.
        assert_eq!(seed_for(0, ""), 0xc381_7c01_6ba4_ff30);
        assert_eq!(seed_for(20_260_918, "assignment"), 0xdb73_681a_503b_48f0);
        assert_eq!(
            seed_for(20_260_918, "moderator:ties"),
            0x7ae3_2d17_5f17_aa83
        );
        assert_eq!(seed_for(20_260_918, "alice"), 0xbfae_7f2f_b73c_7088);
    }
}
