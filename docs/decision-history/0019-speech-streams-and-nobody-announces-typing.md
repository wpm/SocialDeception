# ADR-0019: Speech streams as it is generated, and nobody announces typing

**Status:** Accepted
**Date:** 2026-09-29
**Deciders:** Bill McNeill
**Supersedes:** [ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md)
**Related:** [ADR-0020](0020-a-selection-is-discrete-and-a-model-makes-one-with-a-tool-call.md)
**Depends on:** [ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md),
[ADR-0017](0017-messages-carry-a-sequence-number-and-the-log-keeps-the-time.md),
[ADR-0018](0018-werewolf-speech-goes-through-the-moderator.md)

## Context

[ADR-0013](0013-speech-typing-and-a-scheduler-for-when-to-speak.md) planned
talk for Werewolf around three events, `TypingStarted`, `Say` and
`TypingEnded`, sent in two handler cycles, because under ADR-0009 a
handler's actions left only when it returned: a player announced it was
typing in one cycle and called its model in the next. It also made *when* a
player considers speaking a swappable `Scheduler` strategy object.

Three things have changed.

**Actions leave as they are produced.** Under
[ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md) a
policy returns a lazy iterator whose actions are carried out as they are
yielded. The two-cycle dance has no reason to exist.

**"Started talking" is not an action.** Nobody in a real conversation
performs a separate act of beginning to speak; they speak, and a listener
notices that they have started. A typing indicator is something a listener
*infers* from speech arriving — which is where a chat interface's animation
belongs too. Sending it as its own message puts an interpretation on the
wire and in the log, which [ADR-0017](0017-messages-carry-a-sequence-number-and-the-log-keeps-the-time.md)'s
"just the facts" rules out.

**Models stream.** A provider generates an answer piece by piece and can
send each piece as it is produced. A player that waits for the whole answer
before speaking hides from its listeners exactly the timing that makes
real-time talk different from turn-taking: that someone is mid-sentence,
how fast, and when they stop.

## Decision

**A player's speech is sent as it is generated: each piece the model streams
back becomes its own speech message, sent at once. Every piece of one
utterance carries the same utterance number and its own position in the
utterance, and the last piece is marked as the last. There are no typing events; a listener that wants to know
someone is talking infers it from the pieces arriving.**

### The speech message

```rust
pub struct UtteranceId(pub u64);   // per player, counting from 1

Say { utterance: UtteranceId, part: u32, text: String, last: bool }
```

- **One model response is one utterance.** Every `Say` produced from it
  carries the same `UtteranceId`, so that the pieces can be seen to belong
  together. The player counts its utterances; a sender and an utterance
  number name one utterance.
- **Pieces are the model's.** Whenever a model can stream, the player asks
  it to, and sends each piece of text in the size the model sent it. The
  framework and the game impose no chunking. A model that cannot stream
  produces an utterance of one piece.
- **The model decides the order; the sender numbers it.** Pieces are sent
  in the order the model produced them, and the sending actor numbers each
  one within its utterance, `part` counting from 0. A receiver puts an
  utterance's pieces in order by `part`, so the order survives any channel
  that delivers them out of order, whatever the latency between the
  speaker, the moderator and the listener. Today's in-process channels
  happen to preserve order; the meaning of an utterance does not depend on
  it.
- **The last piece says so.** The end of a stream is only known once it has
  ended, so the player holds each piece until the next one arrives, then
  sends it; when the stream ends it sends the held piece with
  `last: true`. The cost is one piece of latency.
- **An utterance that never gets its `last` was cut off**: the model's call
  timed out, the day closed and the moderator stopped relaying, or the
  player was stopped. That is information, and it is recorded, not
  papered over with a closing message.

### What a listener can infer

| Seen | Meaning |
|---|---|
| a piece of an utterance it has not heard before | the speaker started talking |
| more pieces | still talking, at this pace |
| a piece with `last` | finished, once every `part` before it has arrived |
| pieces, then nothing, and no `last` | cut off |
| a gap in `part` | a piece still in transit |

None of these is a message. Each is a fact about messages a listener
observed, at the times it observed them.

### Speech is only speech

A player's selection is not part of its speech. It is its own discrete
message, which any strategy may make, and which a model makes with a tool
call ([ADR-0020](0020-a-selection-is-discrete-and-a-model-makes-one-with-a-tool-call.md)).
The one place the two meet is a single model response by day, which may
both stream speech and call the selection tool; ADR-0020 says how.

### Relaying

Under [ADR-0018](0018-werewolf-speech-goes-through-the-moderator.md) speech
goes to the moderator alone, which relays it in an `Envelope` to every
living player except the speaker. The moderator relays **each piece as it
arrives**; it never gathers an utterance before relaying it. A piece that
arrives after the day has closed, or from a player who has died, is
relayed to nobody, which is how the utterance comes to be cut off for its
listeners.

### When a player considers speaking

What ADR-0013 made a separate, swappable `Scheduler` strategy object
becomes part of the player's own state, as
[ADR-0016](0016-actors-perceive-on-one-thread-and-decide-on-another.md)
has an agent do all batching: after something it should react to, a
player sets itself a `Reminder` a random delay ahead (drawn from its own
seeded generator), and considers speaking when one arrives. Several things
heard before the reminder fires are considered together, in one model call.
Keeping this swappable was a generalization with one implementation;
factor it out when a second one exists.

### Text only at the edges

Kept from ADR-0013: `Say` carries text, and nothing outside `Say` and the
renderer that turns a player's `Knowledge` into a prompt may assume it, so
that a later experiment can put something other than language on the wire.

## Consequences

**A conversation is visible as it happens.** Listeners, the live view and
the log all see speech at the pace it was generated, interleaved across
speakers, and a player can hear someone else start talking while it is
itself mid-utterance — though, as before, it cannot act on that until its
own call returns.

**Many more messages.** An utterance is as many messages as the model sent
pieces, each relayed to every living listener and logged at both ends. At
Werewolf's scale that is fine; it is also the log recording exactly what
happened.

**The prompt shows whole utterances.** A player's prompt history must only
ever be appended to, so that providers can cache it by prefix. An utterance
in progress would have to be rewritten as more of it arrives, so it is not
in the history: the history gains an utterance once its `last` piece
arrives, and an utterance still in progress, or cut off, is described in
the part of the prompt that changes on every call anyway.

**The chat client must stream.** The OpenAI-protocol client planned for
language-model players returns a stream of text pieces as they arrive, not
only a finished completion.

**Collisions remain.** Two players will sometimes talk over each other
within the window of one model call. That is realistic, and measuring it
still comes before building anything smarter.

## Alternatives considered

### Keep typing events

Rejected: starting to talk is not an act, and the log would record an
interpretation a listener can make for itself.

### No end marker

Send every piece at once and let listeners infer the end from silence.
Rejected because silence cannot tell a pause from an end, and the prompt
renderer needs to know when an utterance is whole.

### Send one message per utterance

Wait for the whole model response and send it as a single `Say`. Rejected:
it hides the timing that makes this real time rather than turn-taking.

### Stream a JSON object's `say` field

Keep the planned `{say, target}` structured answer and parse the `say`
string incrementally as it streams. Rejected: it binds speech and selection
into one structure, when they are separate things (ADR-0020), and makes
speech wait on a JSON parser.
