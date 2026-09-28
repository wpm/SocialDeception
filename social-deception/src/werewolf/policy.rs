//! The decision boundary: a [`Policy`] is handed what an agent sees and
//! returns the target to point at, or none.
//!
//! A policy is a conditional distribution over the action space given the
//! state, in the vocabulary the [module](super) documentation states and
//! ADR-0005 explains. [`View`] is what it conditions on: the state, the
//! kind of session in front of it, and the action space. Nothing here
//! decides what
//! the rules permit; this module only picks from what they permit.
//!
//! # Every policy sees the same action space
//!
//! Trajectories from one policy are training data for another only if the
//! two share a support, so the action space handed to a policy is never
//! narrowed on its behalf, and `View` carries no strategy hints. Everything
//! a policy might want to reason from, the role, the living set, the pack,
//! the seer's findings and how every phase so far pointed, is already in
//! [`Knowledge`], and a policy computes whatever heuristic it wants from
//! that. The action space arrives in a canonical order, so an index into it
//! is a stable action label: the same on every run and in every episode
//! with the same living set, which is what a learner needs and what a
//! constrained decode over a model's output needs.
//!
//! # The baseline's heuristic, and why it is private
//!
//! [`RandomPolicy`] narrows the action space to its own candidates and
//! samples uniformly among them. A baseline that nominates its packmates and
//! re-investigates players it has already seen is a strawman, and a win rate
//! measured against a strawman means nothing, so the baseline plays
//! sensibly. But its list of sensible moves is a property of the policy, not
//! of the action space: it is private to `RandomPolicy` and is not exposed
//! on `View`, on the trait, or anywhere another policy could inherit it.
//! Handing a hand-written list of sensible moves to a model would make its
//! win rate partly the model's play and partly the list's, with no clean way
//! to separate the two.
//!
//! The heuristic is deliberately no cleverer than it is. A seer that used its
//! findings when nominating and a villager that read the phase's points would both
//! play better, and both are left out on purpose: the uniform baseline leaves
//! the seer's information unused, which is what makes its win rate a function
//! of the role counts alone.
//!
//! # The path to a language model
//!
//! A language-model policy is the same trait and sees the same action space,
//! and nothing else has to move. [`Policy::choose`] is infallible on
//! purpose: a model will time out, refuse, and emit output naming nobody in
//! the action space, and there is no correct thing for the *rules* to do
//! about that. The decision belongs to the policy that failed, so a model
//! policy owns its retries and holds a `RandomPolicy` for the case where no
//! action in the space can be extracted from what the model said.
//!
//! A policy may also block: an agent owns a thread (ADR-0001), so a policy
//! waiting on a model provider delays only its own agent — and its own
//! agent's stop, because nothing interrupts a running handler (ADR-0009).
//! A policy that blocks for thirty seconds delays its episode's shutdown by
//! thirty seconds, and nothing in the loop will shorten it.
//!
//! So a model-backed policy bounds its own call. It makes the call on its
//! own thread, **streams** the response, and gives up on a deadline it
//! sets, closing the connection so that generation stops rather than
//! running to completion unread. That wait is where its per-call deadline
//! and its [`RandomPolicy`] fallback live. The obligation is the policy's
//! because the knowledge is: only it knows what its call costs and when
//! waiting longer has stopped being worth it.
//!
//! # Determinism
//!
//! Each agent's policy draws from its own generator, seeded by [`seed_for`]
//! from the master seed and the agent's id, so its actions depend on its own
//! history alone and adding a player perturbs nobody else's; the [`seed`]
//! module says why that, and the choice of generator, make an experiment
//! reproducible. The golden test in this module pins the first moves of
//! one seeded policy, so that a change to the mixing or the sampling fails
//! a test rather than silently becoming a different experiment.
//!
//! [`seed`]: super::seed

use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use super::knowledge::Knowledge;
use super::message::RequestKind;
use super::role::Role;
use super::seed::{pick, seed_for};
use crate::event::AgentId;

/// What a policy sees when it decides: the agent's state, the kind of
/// session in front of it, and the action space.
///
/// It carries no strategy hints; see the [module documentation](self).
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    /// The agent's state: the fold of everything it has been told.
    pub knowledge: &'a Knowledge,
    /// Which of the phase's sessions this is: what the pointing is for.
    pub kind: RequestKind,
    /// Every target the rules permit, in canonical order: sorted agent
    /// order. May be empty, and a policy handed an empty one has nowhere to
    /// point.
    pub action_space: &'a [AgentId],
}

/// How an agent picks an action from the action space.
///
/// Implemented by the uniform random baseline, [`RandomPolicy`], and later
/// by a language-model policy, which sees exactly the same [`View`].
pub trait Policy {
    /// Picks a target, or `None` to point nowhere for now. A `Some` must be
    /// in `view.action_space`.
    ///
    /// `None` is how a member abstains (ADR-0011): nothing is sent, and a
    /// member that never points is simply one nobody saw point. It is
    /// also the only thing a policy can do with an empty action space.
    ///
    /// Infallible, and free to block; the [module documentation](self) says
    /// why, and what a policy that blocks owes its own deadline. Nothing
    /// here can be interrupted (ADR-0009), so a policy that blocks is the
    /// only thing that can bound how long it blocks for.
    ///
    /// [`View`] is `Copy` and is passed by value, so a policy that hands it
    /// on does not have to thread a reference through.
    fn choose(&mut self, view: View<'_>) -> Option<AgentId>;
}

/// The uniform random baseline: a policy that samples uniformly from its own
/// candidates, narrowed from the action space by the private heuristic the
/// [module documentation](self) describes.
#[derive(Debug, Clone)]
pub struct RandomPolicy {
    rng: ChaCha8Rng,
}

impl RandomPolicy {
    /// A policy for one agent, seeded from the master seed mixed with the
    /// agent's own id, so that its actions depend on its own history and
    /// nothing else.
    #[must_use]
    pub fn for_agent(master: u64, who: &AgentId) -> Self {
        Self::from_seed(seed_for(master, who.as_str()))
    }

    /// A policy from an explicit seed, for tests.
    #[must_use]
    pub fn from_seed(seed: u64) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }
}

impl Policy for RandomPolicy {
    /// Sampling from a list cannot block, so this needs no deadline of its
    /// own and the question the [module documentation](self) raises does
    /// not arise. That is also what keeps a deterministic episode
    /// deterministic: nothing about the draw depends on timing at all.
    fn choose(&mut self, view: View<'_>) -> Option<AgentId> {
        let candidates = candidates(&view);
        // Nowhere to point: the doctor with nobody left it may protect.
        // Drawing from an empty list is the one thing sampling cannot do,
        // and pointing nowhere is what the rules call for.
        if candidates.is_empty() {
            return None;
        }
        Some((*pick(&mut self.rng, &candidates)).clone())
    }
}

/// The baseline's candidates, in the action space's order: the targets
/// [`excluded`] leaves, or the whole action space when the heuristic has
/// excluded everything, because a policy handed nothing has no correct
/// behavior.
///
/// Empty only when the action space is, which is the doctor with nobody it
/// may protect. The heuristic never empties a non-empty space: when it
/// would, the space itself is the fallback, and the draw is the same one
/// the old `Abstain`-less path made.
fn candidates<'a>(view: &View<'a>) -> Vec<&'a AgentId> {
    let space = view.action_space;
    let targets: Vec<&AgentId> = space.iter().filter(|who| !excluded(view, who)).collect();
    if targets.is_empty() {
        return space.iter().collect();
    }
    targets
}

/// Whether the heuristic drops `who` as a target in this session: a
/// werewolf's living packmates for `Devour` and `Nominate`, and the targets
/// the seer has already investigated for `Investigate`.
fn excluded(view: &View<'_>, who: &AgentId) -> bool {
    let knowledge = view.knowledge;
    match (knowledge.role, view.kind) {
        (Role::Werewolf, RequestKind::Devour | RequestKind::Nominate) => {
            knowledge.pack.contains(who)
        }
        (Role::Seer, RequestKind::Investigate) => knowledge.investigations.contains_key(who),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Range;

    use super::*;
    use crate::testing::{id, knowing, seer_knowing, target, werewolf_knowing};
    use crate::werewolf::roles::base_action_space;

    const MASTER: u64 = 20_260_918;
    const OTHERS: [&str; 5] = ["alice", "bob", "carol", "dave", "erin"];
    /// The seeds the property tests run a fresh policy under.
    const SEEDS: Range<u64> = 0..200;

    fn choose(
        policy: &mut RandomPolicy,
        knowledge: &Knowledge,
        kind: RequestKind,
        space: &[AgentId],
    ) -> Option<AgentId> {
        policy.choose(View {
            knowledge,
            kind,
            action_space: space,
        })
    }

    /// The first action a fresh policy under each of [`SEEDS`] takes for
    /// `kind`, in the action space the rules would hand it.
    fn first_choices(knowledge: &Knowledge, kind: RequestKind) -> Vec<(u64, Option<AgentId>)> {
        let space = base_action_space(knowledge, kind);
        SEEDS
            .map(|seed| {
                let action = choose(&mut RandomPolicy::from_seed(seed), knowledge, kind, &space);
                (seed, action)
            })
            .collect()
    }

    /// The first `n` nominations a policy makes as a villager among
    /// [`OTHERS`], where no heuristic is in play.
    fn nominations(mut policy: RandomPolicy, n: usize) -> Vec<Option<AgentId>> {
        let knowledge = knowing(Role::Villager, OTHERS);
        let space = base_action_space(&knowledge, RequestKind::Nominate);
        (0..n)
            .map(|_| choose(&mut policy, &knowledge, RequestKind::Nominate, &space))
            .collect()
    }

    #[test]
    fn the_same_master_and_id_give_the_same_sequence() {
        let alice = id("alice");
        let first = nominations(RandomPolicy::for_agent(MASTER, &alice), 50);
        let second = nominations(RandomPolicy::for_agent(MASTER, &alice), 50);
        assert_eq!(first, second);
    }

    #[test]
    fn different_ids_under_one_master_give_different_sequences() {
        let alice = nominations(RandomPolicy::for_agent(MASTER, &id("alice")), 50);
        let bob = nominations(RandomPolicy::for_agent(MASTER, &id("bob")), 50);
        assert_ne!(alice, bob);
    }

    #[test]
    fn for_agent_is_from_seed_under_the_agents_id() {
        let labeled = nominations(RandomPolicy::for_agent(MASTER, &id("carol")), 20);
        let explicit = nominations(RandomPolicy::from_seed(seed_for(MASTER, "carol")), 20);
        assert_eq!(labeled, explicit);
    }

    #[test]
    fn a_returned_action_is_always_in_the_action_space() {
        let werewolf = werewolf_knowing(["alice", "bob", "carol"], ["alice"]);
        let seer = seer_knowing(["alice", "bob", "carol"], ["alice"]);
        let doctor = knowing(Role::Doctor, ["alice", "bob", "carol"]);
        let villager = knowing(Role::Villager, ["alice", "bob", "carol"]);
        let cases = [
            (&werewolf, RequestKind::Devour),
            (&werewolf, RequestKind::Nominate),
            (&seer, RequestKind::Investigate),
            (&seer, RequestKind::Nominate),
            (&doctor, RequestKind::Protect),
            (&doctor, RequestKind::Nominate),
            (&villager, RequestKind::Nominate),
        ];
        for (knowledge, kind) in cases {
            let space = base_action_space(knowledge, kind);
            for (seed, action) in first_choices(knowledge, kind) {
                let action = action.expect("a non-empty action space is pointed into");
                assert!(
                    space.contains(&action),
                    "seed {seed}, {kind:?}: {action:?} outside {space:?}"
                );
            }
        }
    }

    #[test]
    fn the_sequence_is_stable_under_interleaved_single_option_decisions() {
        let doctor = knowing(Role::Doctor, ["alice"]);
        let forced = base_action_space(&doctor, RequestKind::Protect);
        let villager = knowing(Role::Villager, OTHERS);
        let open = base_action_space(&villager, RequestKind::Nominate);

        let mut policy = RandomPolicy::from_seed(7);
        let mut interleaved = Vec::new();
        for _ in 0..20 {
            for _ in 0..3 {
                let action = choose(&mut policy, &doctor, RequestKind::Protect, &forced);
                assert_eq!(action, Some(target("alice")));
            }
            interleaved.push(choose(&mut policy, &villager, RequestKind::Nominate, &open));
        }
        assert_eq!(nominations(RandomPolicy::from_seed(7), 20), interleaved);
    }

    #[test]
    fn every_candidate_is_drawn_and_none_dominates() {
        // A guard against an off-by-one that could never return the last
        // element, not a statistical test.
        let draws = 1000;
        let mut counts: BTreeMap<Option<AgentId>, usize> = BTreeMap::new();
        for action in nominations(RandomPolicy::from_seed(MASTER), draws) {
            *counts.entry(action).or_default() += 1;
        }
        assert_eq!(counts.len(), OTHERS.len(), "{counts:?}");
        for (action, count) in counts {
            assert!(
                count * 2 < draws,
                "{action:?} was drawn {count} times of {draws}"
            );
        }
    }

    #[test]
    fn a_werewolf_never_picks_a_living_packmate() {
        let knowledge = werewolf_knowing(["alice", "bob", "carol", "dave"], ["bob", "dave"]);
        for kind in [RequestKind::Devour, RequestKind::Nominate] {
            for (seed, action) in first_choices(&knowledge, kind) {
                assert!(
                    action == Some(target("alice")) || action == Some(target("carol")),
                    "seed {seed}, {kind:?}: {action:?}"
                );
            }
        }
    }

    #[test]
    fn a_seer_never_re_investigates() {
        let knowledge = seer_knowing(["alice", "bob", "carol", "dave"], ["alice", "carol"]);
        for (seed, action) in first_choices(&knowledge, RequestKind::Investigate) {
            assert!(
                action == Some(target("bob")) || action == Some(target("dave")),
                "seed {seed}: {action:?}"
            );
        }
    }

    #[test]
    fn a_policy_points_somewhere_whenever_it_can() {
        // Pointing nowhere is for a member with nowhere to point. While the
        // action space holds anybody at all, the baseline names somebody.
        let doctor = knowing(Role::Doctor, ["alice", "bob"]);
        let seer = seer_knowing(["alice", "bob"], []);
        for (knowledge, kind) in [
            (&doctor, RequestKind::Protect),
            (&seer, RequestKind::Investigate),
        ] {
            assert!(!base_action_space(knowledge, kind).is_empty());
            for (seed, action) in first_choices(knowledge, kind) {
                assert!(action.is_some(), "seed {seed}, {kind:?}");
            }
        }
    }

    #[test]
    fn a_werewolf_whose_only_living_others_are_packmates_still_acts() {
        let knowledge = werewolf_knowing(["bob", "dave"], ["bob", "dave"]);
        for kind in [RequestKind::Devour, RequestKind::Nominate] {
            let space = base_action_space(&knowledge, kind);
            for (seed, action) in first_choices(&knowledge, kind) {
                let action = action.expect("a non-empty action space is pointed into");
                assert!(space.contains(&action), "seed {seed}, {kind:?}: {action:?}");
            }
        }
    }

    #[test]
    fn a_seer_that_has_investigated_everyone_living_looks_again() {
        // The heuristic would leave it nothing, and a policy handed a
        // non-empty space still points: the space itself is the fallback,
        // so it re-investigates rather than wasting its night.
        let knowledge = seer_knowing(["alice", "bob"], ["alice", "bob"]);
        let space = base_action_space(&knowledge, RequestKind::Investigate);
        for (seed, action) in first_choices(&knowledge, RequestKind::Investigate) {
            let action = action.expect("a non-empty action space is pointed into");
            assert!(space.contains(&action), "seed {seed}: {action:?}");
        }
    }

    #[test]
    fn an_empty_action_space_points_nowhere() {
        // The doctor with nobody left it may protect. Drawing from an empty
        // list is the one thing sampling cannot do, and pointing nowhere is
        // what the rules call for (ADR-0011).
        let knowledge = knowing(Role::Doctor, ["alice"]);
        for seed in SEEDS {
            let action = choose(
                &mut RandomPolicy::from_seed(seed),
                &knowledge,
                RequestKind::Protect,
                &[],
            );
            assert_eq!(action, None, "seed {seed}");
        }
    }

    #[test]
    fn golden_actions() {
        // Change the seed mixing, the generator or the sampling and this
        // changes; that is the point.
        let actions = nominations(RandomPolicy::for_agent(MASTER, &id("alice")), 8);
        assert_eq!(
            actions,
            [
                "carol", "bob", "alice", "alice", "erin", "erin", "dave", "alice"
            ]
            .map(|who| Some(target(who)))
        );
    }
}
