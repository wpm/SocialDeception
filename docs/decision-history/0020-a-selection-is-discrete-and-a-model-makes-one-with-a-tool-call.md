# ADR-0020: A selection is discrete, and a model makes one with a tool call

**Status:** Accepted
**Date:** 2026-09-29
**Deciders:** Bill McNeill
**Amends:** [ADR-0005](0005-policy-separates-decisions-from-rules.md)
**Related:** [ADR-0019](0019-speech-streams-and-nobody-announces-typing.md)

## Context

A **selection** is Werewolf's move: during a session, a member selects a
target from the action space its role permits, may change its mind, and its
most recent selection is its vote
([ADR-0011](0011-werewolf-phases-are-timed-pointing-sessions.md), where it
was called pointing). Today every selection comes from a scripted or random
strategy.

The plans for language-model players bound selection to other things. At
night a model was to answer with a structured JSON object, `{"target": …}`;
by day with `{"say": …, "target": …}`, so that one answer carried both what
the player said and whom it selected.
[ADR-0019](0019-speech-streams-and-nobody-announces-typing.md) makes speech
stream piece by piece as the model generates it, which a JSON object does
not do without a parser for half-finished strings. More basically, the two
are different kinds of thing:

- **Speech** is produced incrementally, only by players that talk, and only
  a model talks.
- **A selection** is one decision, made at once. Any strategy can make one —
  random, scripted, or a model — and nothing about it is partial.

## Decision

**A selection is its own discrete message, never streamed and never part of
an utterance. Any strategy may make one. A model-backed strategy makes one
only by calling a `select` tool; there is no other way for a model's output
to become a selection.**

### The message

`Select` stays what it is: one message naming one target, sent to the
moderator (ADR-0018) and relayed by it in an envelope to whoever the rules
let see it. It carries no utterance number and has nothing to do with
speech. A strategy that selects nobody sends nothing, which is how a member
abstains.

### A model selects with a tool call

A model-backed strategy offers the model one tool:

```json
{
  "type": "function",
  "function": {
    "name": "select",
    "parameters": {
      "type": "object",
      "properties": { "target": { "enum": ["alice", "carol", "dave"] } },
      "required": ["target"],
      "additionalProperties": false
    },
    "strict": true
  }
}
```

The `enum` is the action space for this decision. Calling the tool selects;
not calling it selects nobody. A call whose target is not in the action
space — a model that ignores the schema — falls back to the strategy's
fallback, as a failed model call does under
[ADR-0005](0005-policy-separates-decisions-from-rules.md). The same tool is
used at night and by day; the structured `{target}` and `{say, target}`
answers are not built.

### Where selection and speech meet

By day one model call may do both: stream speech as its text content
([ADR-0019](0019-speech-streams-and-nobody-announces-typing.md)) and call
`select`. They remain separate outputs of the same response:

- speech pieces are sent as they stream, as `Say` messages of one utterance;
- the selection is sent as one `Select` message when the tool call is
  complete, which in the protocol is at the end of the response;
- the text content ends where the tool call begins, so that is also when
  the utterance's last piece is known and sent.

A response with only text is speech without a selection; a response with
only a tool call is a silent selection. At night a model does not speak:
any text content it produces is not sent.

## Consequences

**Speech and selection can change independently.** How speech is produced
and relayed (ADR-0019) and how selection is decided and validated (this
record) share nothing but, by day, a model call.

**Every model-backed player needs a model that supports tool calls.** The
OpenAI protocol's tool calling is widely implemented, including by the
local servers in view (vLLM, Ollama), but a model without it cannot select.

**The chat client handles tool calls in a stream.** Tool-call arguments
arrive in pieces like text does; the client assembles them and reports the
completed call at the end of the response, alongside the text pieces it
reports as they arrive.

## Alternatives considered

### Structured output for the whole answer

A JSON object with `target` (and, by day, `say`). Rejected: it binds a
discrete decision to incrementally produced speech, and makes selection look
like a field of a message rather than an act of its own.

### Select in a separate model call from speaking

Keeps each call single-purpose. Rejected as the default because it doubles
the calls when a player both talks and selects, and the protocol already
separates a tool call from text within one response. A strategy is free to
make separate calls if it wants to.
