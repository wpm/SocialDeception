//! Deriving one stream's seed from the episode's master seed.
//!
//! Every source of randomness in an episode draws from a generator of its
//! own: the role deal, each player's policy, the moderator's tie-breaks. Each
//! generator is seeded with the master seed mixed with a label naming the
//! stream, so that one stream can be recomputed without the others and so
//! that a player's own choices do not depend on how many other players there
//! are.
//!
//! The mix is written out here rather than borrowed from `std`'s
//! `DefaultHasher` or from `rand`, both of which are allowed to change their
//! output between releases. A seed that stops meaning the same thing after a
//! dependency bump is not a reproducible experiment, so the golden values in
//! this module's tests are part of the contract.

/// Mixes a master seed with a label into a stable per-stream seed.
///
/// FNV-1a over the master seed's little-endian bytes followed by the label's
/// bytes, finished with the splitmix64 mixer so that nearby inputs give
/// unrelated outputs. Labels in use: `"assignment"` for the role deal, an
/// agent's own id for its policy, and `"moderator:ties"` for the moderator's
/// tie-breaks.
#[must_use]
pub fn seed_for(master: u64, label: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut hash = OFFSET;
    for byte in master.to_le_bytes().into_iter().chain(label.bytes()) {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(PRIME);
    }
    splitmix64(hash)
}

/// The splitmix64 output function: a bijection on `u64` that spreads every
/// input bit across the output.
fn splitmix64(z: u64) -> u64 {
    let z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    /// The labels the epic uses, plus a few that differ from them only
    /// slightly.
    const LABELS: [&str; 8] = [
        "assignment",
        "moderator:ties",
        "alice",
        "Alice",
        "alice ",
        "bob",
        "moderator",
        "",
    ];

    #[test]
    fn seed_for_is_not_the_identity() {
        for master in [0, 1, 42, 20_260_918, u64::MAX] {
            assert_ne!(seed_for(master, "assignment"), master);
            assert_ne!(seed_for(master, ""), master);
        }
    }

    #[test]
    fn labels_do_not_collide_under_one_master() {
        for master in [0, 20_260_918] {
            let seeds: BTreeSet<u64> = LABELS.iter().map(|label| seed_for(master, label)).collect();
            assert_eq!(seeds.len(), LABELS.len(), "master {master}");
        }
    }

    #[test]
    fn masters_do_not_collide_under_one_label() {
        let seeds: BTreeSet<u64> = (0..1000).map(|master| seed_for(master, "alice")).collect();
        assert_eq!(seeds.len(), 1000);
    }

    #[test]
    fn labels_are_case_sensitive() {
        assert_ne!(seed_for(7, "alice"), seed_for(7, "Alice"));
    }

    #[test]
    fn golden_seeds() {
        // Change the mix function and these change; that is the point.
        assert_eq!(seed_for(0, ""), 0x5ba3_14b8_cfda_3b6b);
        assert_eq!(seed_for(0, "assignment"), 0x6e44_e884_d36f_d824);
        assert_eq!(seed_for(20_260_918, "assignment"), 0x8ca5_bab1_1158_1de2);
        assert_eq!(seed_for(20_260_918, "alice"), 0xbfc4_09ee_0382_00f2);
        assert_eq!(
            seed_for(20_260_918, "moderator:ties"),
            0x5e82_906e_0c4c_387e
        );
        assert_eq!(seed_for(u64::MAX, "bob"), 0x89af_1b06_5ba4_d6dc);
    }
}
