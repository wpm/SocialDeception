# ADR-0006: Knowledge holds what the agent did in secret

**Status:** Accepted
**Date:** 2026-09-19
**Deciders:** Bill McNeill
**Refines:** [ADR-0005](0005-policy-separates-decisions-from-rules.md)

## Context

[ADR-0005](0005-policy-separates-decisions-from-rules.md) makes `Knowledge`
the state a policy conditions on: a sufficient statistic of the agent's
observation history, holding only true statements, and folded from moderator
narration alone. That last clause was the mechanism by which the other two
were guaranteed. Every observation is a statement from the moderator, the
moderator does not lie, so nothing in the state can be false, and inference
is left to the policy.

The doctor's rule does not fit that mechanism. The doctor may not protect
the same player on two consecutive nights, so its `Protect` action space
depends on whom it protected last night. Nothing narrates that back to it: a
save is announced as "no one died", never as "you saved someone", and a
protection that saved nobody is announced as nothing at all. The rules need
a fact the moderator never states.

The first implementation kept that fact on the doctor's role type, in a
field beside its `Knowledge`, written by a callback after each choice.
Review found the cost. A policy is handed `Knowledge` and the action space;
with the protection kept off `Knowledge`, the doctor's `Protect` action
space was one target shorter for a reason the policy could not see in the
state it was given, and the sufficient-statistic claim of ADR-0005 was
false for one role.

## Decision

**`Knowledge` is what the agent knows to be true: what it was told, and
what it did.** Beside `observe`, which folds an observation, `Knowledge`
has `acted`, which folds the agent's own answer to a request. The seat calls
it after every accepted choice, so the state a policy sees next includes
what the policy itself did.

Nothing here can be false, so the line ADR-0005 draws between knowledge and
belief is unchanged. A player's own action is true for the same reason a
narration is: the player was there.

The fold is deliberately narrow. Almost every action a player takes comes
back to it by narration anyway: a nomination or a devour in the tally, an
investigation as its result. Recording those on the way out would put the
same fact in the state twice, once as taken and once as told. The doctor's
protection is the one action announced to nobody, and the only one the
rules need, so it is the only thing `acted` records: a `Protect` sets
`last_protected`, an abstention clears it, and every other kind of request
leaves the state unchanged.

## Alternatives considered

### Keep the protection on the role, off `Knowledge`

The original shape. Rejected because it makes the action space depend on
state the policy cannot see, which contradicts the sufficient-statistic
claim, and because it spends a trait method with a no-op default on exactly
one implementor.

### Record every own action

A general log of the agent's answers, from which the doctor derives its
last protection. Rejected as duplication: every own action but the
protection already reaches the state by narration, and a second copy of
each would have to be kept consistent with the first for no consumer. If a
language-model prompt later wants an own-action history, `acted` is the hook
and the change is local.

### Have the moderator narrate the protection back

A `Protected { target }` narration to the doctor alone, so that `Knowledge`
could stay narration-only. Rejected because it adds a message that carries
no information, since the doctor already knows what it did, purely to
preserve a mechanism, and because it puts a message on the wire that a
transcript reader would then have to know is redundant.

## Consequences

- **A policy sees why its action space is what it is.** The doctor's
  `Protect` space excludes `knowledge.last_protected`, and both are in the
  `View`.
- **A prompt can be rendered from the state alone.** "You protected X last
  night" comes from `Knowledge`, not from anything the policy remembered
  for itself.
- **The purity property of `Knowledge` widens by one input.** The state is
  a pure function of the observations and the own actions folded into it,
  in order. The tests that assert purity fold both.
- **The narrow fold is a commitment.** A future role whose rules depend on
  its own past actions, and whose actions are not narrated back, extends
  `acted`; a role whose actions are narrated back does not.
