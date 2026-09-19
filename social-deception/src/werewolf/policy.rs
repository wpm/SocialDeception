//! The decision boundary: a [`Policy`] is handed what an agent sees and
//! returns one action.
//!
//! In the reinforcement-learning vocabulary of the design (ADR-0005), the
//! *state* is [`Knowledge`], the *action* type is [`Action`], and the
//! *action space* is the subset of actions the rules permit for one
//! decision, a value the rules compute anew at every request. A policy is a
//! conditional distribution over the action space given the state. [`View`]
//! is what it conditions on: the state, the request in front of it, and the
//! action space. Nothing here decides what the rules permit; this module
//! only picks from what they permit.
//!
//! # Every policy sees the same action space
//!
//! Trajectories from one policy are training data for another only if the
//! two share a support, so the action space handed to a policy is never
//! narrowed on its behalf, and `View` carries no strategy hints. Everything
//! a policy might want to reason from, the role, the living set, the pack,
//! the seer's findings and every tally the agent heard, is already in
//! `Knowledge`, and a policy computes whatever heuristic it wants from that.
//!
//! The action space arrives in a canonical order: targets in sorted agent
//! order, [`Action::Abstain`] last where it is permitted. An index into it
//! is therefore a stable action label, the same on every run and in every
//! episode with the same living set, which is what a learner needs and what
//! a constrained decode over a model's output needs.
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
//! findings when nominating and a villager that read the tallies would both
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
//! action in the space can be extracted from what the model said. A policy
//! may also block: an agent owns a thread (ADR-0001), so a policy waiting on
//! a model provider over ordinary blocking HTTP delays only its own agent,
//! and nothing here needs to accommodate a slow one.
//!
//! # Determinism
//!
//! `RandomPolicy` draws from a [`ChaCha8Rng`], whose algorithm is fixed
//! across `rand` releases; `StdRng`'s is explicitly allowed to change, and a
//! seed that stops meaning the same thing after a dependency bump is not a
//! reproducible experiment. Each agent's policy is seeded from the master
//! seed mixed with the agent's own id by [`seed_for`], never from a shared
//! generator: a shared one would make an agent's actions depend on the order
//! the threads happened to run, whereas a per-agent stream depends on that
//! agent's own history alone, and adding a player perturbs nobody else's.
//! The golden test in this module pins the first actions of one seeded
//! policy, so that a change to the mixing or the sampling fails a test
//! rather than silently becoming a different experiment.

use rand::{RngExt, SeedableRng};
use rand_chacha::ChaCha8Rng;

use super::knowledge::Knowledge;
use super::message::{Action, Request, RequestKind};
use super::role::Role;
use super::seed::seed_for;
use crate::event::AgentId;

/// What a policy sees when it decides: the agent's state, the request in
/// front of it, and the action space.
///
/// It carries no strategy hints; see the [module documentation](self).
#[derive(Debug, Clone, Copy)]
pub struct View<'a> {
    /// The agent's state: the fold of everything it has been told.
    pub knowledge: &'a Knowledge,
    /// The request being answered.
    pub request: &'a Request,
    /// Every action the rules permit, in canonical order: targets in sorted
    /// agent order, [`Action::Abstain`] last where it is permitted. Never
    /// empty.
    pub action_space: &'a [Action],
}

/// How an agent picks an action from the action space.
///
/// Implemented by the uniform random baseline, [`RandomPolicy`], and later
/// by a language-model policy, which sees exactly the same [`View`].
pub trait Policy {
    /// Picks an action. The result must be in `view.action_space`.
    ///
    /// Infallible by design: a policy that cannot decide, such as a model
    /// policy whose model would not answer, substitutes an action of its
    /// own choosing rather than reporting failure to the rules. It may block
    /// for as long as it needs; only its own agent waits.
    fn choose(&mut self, view: &View<'_>) -> Action;
}

/// The uniform random baseline: a policy that samples uniformly from its own
/// candidates, narrowed from the action space by the private heuristic the
/// [module documentation](self) describes.
///
/// Where narrowing leaves exactly one candidate, the policy returns it
/// without drawing, so a run of forced decisions leaves the generator where
/// it was and the actions that follow do not depend on how many there were.
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
    fn choose(&mut self, view: &View<'_>) -> Action {
        let candidates = candidates(view);
        let chosen = match candidates.as_slice() {
            [only] => only,
            _ => candidates[self.rng.random_range(0..candidates.len())],
        };
        chosen.clone()
    }
}

/// The baseline's candidates, in the action space's order.
///
/// The action space narrowed by the heuristic: a werewolf's living packmates
/// are dropped for `Devour` and `Nominate`, the seer's already-investigated
/// targets for `Investigate`, and `Abstain` whenever a target remains. If
/// narrowing would leave nothing, the candidates are the whole action space,
/// because a policy handed nothing has no correct behaviour.
fn candidates<'a>(view: &View<'a>) -> Vec<&'a Action> {
    let mut candidates: Vec<&Action> = view
        .action_space
        .iter()
        .filter(|action| match action {
            Action::Target(who) => !excluded(view, who),
            Action::Abstain => true,
        })
        .collect();
    if candidates.iter().any(|action| **action != Action::Abstain) {
        candidates.retain(|action| **action != Action::Abstain);
    }
    if candidates.is_empty() {
        candidates = view.action_space.iter().collect();
    }
    candidates
}

/// Whether the heuristic drops `who` as a target for this request.
fn excluded(view: &View<'_>, who: &AgentId) -> bool {
    let knowledge = view.knowledge;
    match (knowledge.role, view.request.kind) {
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

    use rand::Rng;

    use super::*;
    use crate::testing::{id, ids};
    use crate::werewolf::message::{RequestId, Round};
    use crate::werewolf::role::Faction;

    const MASTER: u64 = 20_260_918;
    const ME: &str = "me";
    const OTHERS: [&str; 5] = ["alice", "bob", "carol", "dave", "erin"];

    fn target(who: &str) -> Action {
        Action::Target(id(who))
    }

    /// An action space in canonical order: `targets` sorted, then `Abstain`
    /// where the kind permits it.
    fn action_space<const N: usize>(kind: RequestKind, targets: [&str; N]) -> Vec<Action> {
        let mut space: Vec<Action> = ids(targets).into_iter().map(Action::Target).collect();
        if kind.may_abstain() {
            space.push(Action::Abstain);
        }
        space
    }

    fn request(kind: RequestKind) -> Request {
        Request {
            id: RequestId(1),
            round: Round(1),
            kind,
        }
    }

    /// The knowledge of a player of `role` whose living others are `others`.
    fn knowing<const N: usize>(role: Role, others: [&str; N]) -> Knowledge {
        let mut knowledge = Knowledge::new(id(ME), role);
        knowledge.living = ids(others);
        knowledge.living.insert(id(ME));
        knowledge
    }

    fn werewolf<const L: usize, const P: usize>(others: [&str; L], pack: [&str; P]) -> Knowledge {
        let mut knowledge = knowing(Role::Werewolf, others);
        knowledge.pack = ids(pack);
        knowledge.pack.insert(id(ME));
        knowledge
    }

    fn seer<const L: usize, const I: usize>(others: [&str; L], seen: [&str; I]) -> Knowledge {
        let mut knowledge = knowing(Role::Seer, others);
        knowledge.investigations = ids(seen)
            .into_iter()
            .map(|who| (who, Faction::Village))
            .collect();
        knowledge
    }

    fn choose(
        policy: &mut RandomPolicy,
        knowledge: &Knowledge,
        kind: RequestKind,
        space: &[Action],
    ) -> Action {
        policy.choose(&View {
            knowledge,
            request: &request(kind),
            action_space: space,
        })
    }

    /// The first `n` nominations a fresh policy makes as a villager among
    /// [`OTHERS`], where no heuristic is in play.
    fn nominations(mut policy: RandomPolicy, n: usize) -> Vec<Action> {
        let knowledge = knowing(Role::Villager, OTHERS);
        let space = action_space(RequestKind::Nominate, OTHERS);
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
        let labelled = nominations(RandomPolicy::for_agent(MASTER, &id("carol")), 20);
        let explicit = nominations(RandomPolicy::from_seed(seed_for(MASTER, "carol")), 20);
        assert_eq!(labelled, explicit);
    }

    #[test]
    fn a_returned_action_is_always_in_the_action_space() {
        let werewolf = werewolf(["alice", "bob", "carol"], ["alice"]);
        let seer = seer(["alice", "bob", "carol"], ["alice"]);
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
        for seed in 0..200 {
            let mut policy = RandomPolicy::from_seed(seed);
            for (knowledge, kind) in cases {
                let space = action_space(kind, ["alice", "bob", "carol"]);
                let action = choose(&mut policy, knowledge, kind, &space);
                assert!(
                    space.contains(&action),
                    "seed {seed}: {action:?} outside {space:?}"
                );
            }
        }
    }

    #[test]
    fn a_single_candidate_decision_leaves_the_stream_where_it_was() {
        let doctor = knowing(Role::Doctor, ["alice"]);
        let forced = action_space(RequestKind::Protect, ["alice"]);
        let mut untouched = RandomPolicy::from_seed(7);
        let mut interrupted = RandomPolicy::from_seed(7);
        for _ in 0..25 {
            let action = choose(&mut interrupted, &doctor, RequestKind::Protect, &forced);
            assert_eq!(action, target("alice"));
        }
        assert_eq!(untouched.rng.next_u64(), interrupted.rng.next_u64());
    }

    #[test]
    fn the_sequence_is_stable_under_interleaved_single_option_decisions() {
        let doctor = knowing(Role::Doctor, ["alice"]);
        let forced = action_space(RequestKind::Protect, ["alice"]);
        let villager = knowing(Role::Villager, OTHERS);
        let open = action_space(RequestKind::Nominate, OTHERS);

        let mut policy = RandomPolicy::from_seed(7);
        let mut interleaved = Vec::new();
        for _ in 0..20 {
            for _ in 0..3 {
                choose(&mut policy, &doctor, RequestKind::Protect, &forced);
            }
            interleaved.push(choose(&mut policy, &villager, RequestKind::Nominate, &open));
        }
        assert_eq!(nominations(RandomPolicy::from_seed(7), 20), interleaved);
    }

    #[test]
    fn every_candidate_is_drawn_and_none_dominates() {
        // A guard against an off-by-one that could never return the last
        // element, not a statistical test.
        let draws = 2000;
        let mut counts: BTreeMap<Action, usize> = BTreeMap::new();
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
        let knowledge = werewolf(["alice", "bob", "carol", "dave"], ["bob", "dave"]);
        for kind in [RequestKind::Devour, RequestKind::Nominate] {
            let space = action_space(kind, ["alice", "bob", "carol", "dave"]);
            for seed in 0..200 {
                let action = choose(&mut RandomPolicy::from_seed(seed), &knowledge, kind, &space);
                assert!(
                    action == target("alice") || action == target("carol"),
                    "seed {seed}, {kind:?}: {action:?}"
                );
            }
        }
    }

    #[test]
    fn a_seer_never_re_investigates() {
        let knowledge = seer(["alice", "bob", "carol", "dave"], ["alice", "carol"]);
        let kind = RequestKind::Investigate;
        let space = action_space(kind, ["alice", "bob", "carol", "dave"]);
        for seed in 0..200 {
            let action = choose(&mut RandomPolicy::from_seed(seed), &knowledge, kind, &space);
            assert!(
                action == target("bob") || action == target("dave"),
                "seed {seed}: {action:?}"
            );
        }
    }

    #[test]
    fn abstain_is_never_chosen_while_a_target_is_available() {
        let doctor = knowing(Role::Doctor, ["alice", "bob"]);
        let seer = seer(["alice", "bob"], []);
        for (knowledge, kind) in [
            (&doctor, RequestKind::Protect),
            (&seer, RequestKind::Investigate),
        ] {
            let space = action_space(kind, ["alice", "bob"]);
            assert_eq!(space.last(), Some(&Action::Abstain));
            for seed in 0..200 {
                let action = choose(&mut RandomPolicy::from_seed(seed), knowledge, kind, &space);
                assert_ne!(action, Action::Abstain, "seed {seed}, {kind:?}");
            }
        }
    }

    #[test]
    fn a_werewolf_whose_only_living_others_are_packmates_still_acts() {
        let knowledge = werewolf(["bob", "dave"], ["bob", "dave"]);
        for kind in [RequestKind::Devour, RequestKind::Nominate] {
            let space = action_space(kind, ["bob", "dave"]);
            for seed in 0..50 {
                let action = choose(&mut RandomPolicy::from_seed(seed), &knowledge, kind, &space);
                assert!(space.contains(&action), "seed {seed}, {kind:?}: {action:?}");
            }
        }
    }

    #[test]
    fn a_seer_that_has_investigated_everyone_living_still_acts() {
        let knowledge = seer(["alice", "bob"], ["alice", "bob"]);
        let kind = RequestKind::Investigate;
        let space = action_space(kind, ["alice", "bob"]);
        for seed in 0..50 {
            let action = choose(&mut RandomPolicy::from_seed(seed), &knowledge, kind, &space);
            assert!(space.contains(&action), "seed {seed}: {action:?}");
        }
    }

    #[test]
    fn a_doctor_whose_action_space_is_abstain_abstains() {
        let knowledge = knowing(Role::Doctor, ["alice"]);
        let space = [Action::Abstain];
        for seed in 0..50 {
            let action = choose(
                &mut RandomPolicy::from_seed(seed),
                &knowledge,
                RequestKind::Protect,
                &space,
            );
            assert_eq!(action, Action::Abstain, "seed {seed}");
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
            .map(target)
        );
    }
}
