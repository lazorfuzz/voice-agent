# Prompt-Based End-of-Utterance: the `hold_for_more_input` Pattern

An implementation guide for giving a voice agent's LLM an **escape hatch to say nothing** —
to silently wait for the user to finish instead of being forced to answer every committed
turn. Written from a working implementation (LiveKit Agents 1.6.x + a local 4-bit Qwen via
an OpenAI-compatible server), including every failure mode we hit in production and how
each was fixed. The pattern generalizes to any voice stack with function calling.

---

## 1. The problem

Voice pipelines commit a "turn" using an end-of-utterance (EOU) detector: VAD silence +
(optionally) a semantic turn model, bounded by endpointing delays. No matter how well you
tune it, some genuinely-unfinished turns get committed:

- trail-offs: *"turn on the…"*
- filler endings: *"yeah, so um."*
- meta-announcements: *"I have a question."* (the real request is still coming)
- STT artifacts: a spurious `?` makes *"can you take a look at the?"* score as a complete
  question and blow through any semantic detector

Once the turn commits, a naive agent MUST respond — so it talks over the user mid-thought.
Tightening the endpointing delay makes latency better and clipping worse; loosening it does
the reverse. The detector alone cannot solve this: it sees audio/transcript shape, not
meaning-in-context.

**The insight: the LLM already sees the turn in full conversational context and can judge
"is this finished?" far better than the detector — during the same generation call it was
going to make anyway.** Give it a tool that means "I choose to say nothing and wait."

## 2. Architecture overview

Two layers, splitting the work by pause length:

```
  user audio ──► VAD ──► EOU detector ──► turn committed ──► LLM generation
                          │                                    │
     LAYER 1: raise the detector's "unlikely finished"         │  LAYER 2: the LLM can call
     threshold so low-confidence fragments wait the full       │  hold_for_more_input(seconds)
     max_endpointing_delay and MERGE (covers pauses            │  instead of replying
     shorter than max_endpointing_delay, zero LLM cost)        │  (covers pauses LONGER than
                                                               │  the endpointing window)
                                                               ▼
                                        tool fires: stay SILENT + start a flush timer
                                                               │
                              ┌────────────────────────────────┼──────────────────┐
                              ▼                                ▼                  ▼
                    user continues before             timer expires        agent speaks
                    timeout → new turn re-runs        (user really         (filler in same
                    the LLM with fuller history       stopped) → answer    generation) →
                    → cancel timer, re-decide         what we have         DO NOT cancel timer
```

Key properties:

- **Zero latency tax on normal turns.** The hold decision is made inside the same LLM call
  that produces the reply — the model just picks "hold" instead of words.
- **Turns accumulate naturally.** The committed-but-held user message is already in chat
  history; when the user continues, the next generation sees the concatenated fragments.
- **Bounded silence.** A flush timer guarantees an answer if the user really stopped, and a
  consecutive-holds cap guarantees the agent can never sit in dead air indefinitely.

## 3. Layer 1: detector threshold

If your turn detector exposes a confidence threshold, raise it so that fragments the model
itself scores as unlikely-finished take the slow path (full `max_endpointing_delay`) instead
of fast-committing. Continuations within that window merge **without any LLM involvement** —
this is the cheapest anti-clipping you can buy.

Ours (LiveKit `EnglishModel`): the calibrated default `unlikely_threshold=0.0289` was
hairline — *"Yeah. So."* scored P(finished)=0.054 and fast-committed anyway. We raised it:

```python
turn_detection=EnglishModel(unlikely_threshold=0.15),
min_endpointing_delay=0.2,
max_endpointing_delay=0.9,
```

Measure before choosing the number: log each commit's probability, look at where your real
fragments vs real commands score, and put the threshold between the clusters (our fragments
scored 0.03–0.07, real commands 0.27–0.73).

## 4. Layer 2: the hold tool

### 4.1 The tool itself

```python
_MAX_HOLDS = 3   # consecutive holds before we force an answer — never sit in dead air

class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions=PERSONA, chat_ctx=_seed_hold_example())  # see §6.2
        self._hold_task: asyncio.Task | None = None   # pending "answer on timeout" timer
        self._hold_count = 0                          # consecutive holds without speaking
        self._suppress_hold = False                   # True while flushing → don't re-hold

    @function_tool
    async def hold_for_more_input(self, ctx: RunContext, seconds: int) -> str | None:
        """Stay SILENT and keep listening because the user isn't finished talking yet.
        Call this INSTEAD of replying when the user's turn is clearly incomplete — they
        trailed off, paused mid-thought, or ended on a filler/connector. Never call it
        for a short-but-complete request.

        Args:
            seconds: how long to wait for them to continue, 1-10.
        """
        if self._suppress_hold:              # we're already flushing → answer now
            return "The user has stopped; answer now with what you have."
        if self._hold_count >= _MAX_HOLDS:   # cap the streak
            self._hold_count = 0
            return "You've waited long enough; answer now with what you have."

        secs = max(1, min(10, int(seconds)))

        # CRITICAL (§6.1): record this call + a synthetic output in chat history BEFORE
        # raising StopResponse, because StopResponse otherwise ERASES the call from
        # history and the model never sees a well-formed example of this tool.
        chat_ctx = self.chat_ctx.copy()
        chat_ctx.items.append(ctx.function_call.model_copy())
        chat_ctx.items.append(FunctionCallOutput(
            call_id=ctx.function_call.call_id, name=ctx.function_call.name,
            output=f"Holding silently for {secs}s to let the user finish.",
            is_error=False))
        await self.update_chat_ctx(chat_ctx)

        self._begin_hold(ctx.session, secs)
        raise StopResponse()                 # produce no spoken reply — the escape hatch
```

Notes:

- Returning a string from the guard branches (instead of raising) makes the model produce a
  normal answer — that's intentional: those branches mean "stop holding, talk."
- `seconds` is a **required** int. Optional params interact badly with some local-model tool
  parsers; keep this tool's schema as boring as possible.

### 4.2 The hold state machine

```python
    def _cancel_hold(self):
        if self._hold_task is not None and not self._hold_task.done():
            self._hold_task.cancel()
        self._hold_task = None

    def _reset_holds(self):
        """Called when the agent SPEAKS → the hold STREAK is over. Deliberately does NOT
        cancel a pending flush timer — see §6.3 (the filler-speech deadlock)."""
        self._hold_count = 0

    def _begin_hold(self, session, secs: int):
        self._hold_count += 1

        async def _flush():
            try:
                await asyncio.sleep(secs)
            except asyncio.CancelledError:
                return                        # superseded: the user continued
            # Timer elapsed with no further speech → answer what we have. Clear the task
            # ref FIRST (so nothing can cancel our own reply) and suppress re-holding on
            # this flush (so the model can't decide to wait yet again).
            self._hold_task = None
            self._hold_count = 0
            self._suppress_hold = True
            try:
                await session.generate_reply(
                    instructions="The user paused and didn't continue. Respond now to "
                                 "what they've said so far; do not wait for more.")
            finally:
                self._suppress_hold = False

        self._cancel_hold()                   # supersede any previous hold
        self._hold_task = asyncio.create_task(_flush())
        # keep a strong reference to the task or it can be garbage-collected mid-sleep

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # The user produced more input → supersede the pending timeout-answer. The fuller
        # transcript is already in history; generation will re-decide (answer, or hold
        # again if still incomplete).
        self._cancel_hold()
```

**The complete set of timer-cancel paths — audit yours against this list:**

| event | action on timer |
|---|---|
| user continues (`on_user_turn_completed`) | cancel — generation re-decides |
| a new hold begins (`_begin_hold`) | cancel + replace |
| the flush itself fires | clears its own ref before answering |
| **agent speaks** | **NOTHING** (see §6.3) |

### 4.3 The prompt

Put an explicit, example-driven turn-taking section in the system prompt. Ours:

> TURN-TAKING — READ CAREFULLY: The speech system often hands you a turn while the user is
> still mid-thought. Before answering, ALWAYS ask yourself: is this a finished request, or
> did they get cut off? If it is not clearly finished, do NOT reply — use the
> hold_for_more_input tool (a real tool call, NEVER say or write its name in your spoken
> reply) and stay silent so they can finish. HOLD on all of these: trail-offs
> ('turn on the...', 'maybe we should.'), filler/connector endings ('so', 'um', 'and',
> 'like', 'yeah so'), and meta-announcements that promise more ('I have a question',
> 'so I was thinking' — the real request is still coming, wait for it). ANSWER normally only
> when the request is complete and actionable, even if short ('mute the TV'). When in doubt,
> HOLD once rather than talk over them.

Two hard-won rules:

1. **Never write function-call syntax in the prompt.** Our first version said "call
   hold_for_more_input(seconds)" — the model *imitated the literal string as speech* and TTS
   read `hold_for_more_input(seconds=2)` aloud. Name the tool, never show a call.
2. Include the **meta-announcement** category explicitly — models reliably answer
   "I have a question" with "sure, go ahead!" unless told to hold, which costs a turn.

### 4.4 The exact production prompt for this tool

Copy-paste ready — the verbatim TURN-TAKING block from the production system prompt
(append it to your own persona/instructions), and the tool's docstring, which is what the
model sees as the function description:

**System-prompt block:**

```text
TURN-TAKING — READ CAREFULLY: The speech system often hands you a turn while the user is
still mid-thought. Before answering, ALWAYS ask yourself: is this a finished request, or
did they get cut off? If it is not clearly finished, do NOT reply — use the
hold_for_more_input tool (a real tool call, NEVER say or write its name in your spoken
reply) and stay silent so they can finish. HOLD on all of these: trail-offs ('turn on
the...', 'maybe we should.'), filler/connector endings ('so', 'um', 'and', 'like', 'yeah
so'), and meta-announcements that promise more ('I have a question', 'so I was thinking',
'okay here's the thing' — the real request is still coming, wait for it). ANSWER normally
only when the request is complete and actionable, even if short ('mute the TV', 'what's
the weather'). When in doubt, HOLD once rather than talk over them.
```

**Tool description (function docstring / schema description):**

```text
Stay SILENT and keep listening because the user isn't finished talking yet. Call this
INSTEAD of replying when the user's turn is clearly incomplete — they trailed off, paused
mid-thought, or ended on a filler/connector. Never call it for a short-but-complete
request.

Args:
    seconds: how long to wait for them to continue, 5-10 (longer if they sound like
             they need a beat to think, shorter if they'll likely finish fast).
```

(We originally ran 1–10; the model near-always picked 3s, which flush-answered before slow
"thinking out loud" pauses finished — raising the floor to 5s gave users more room. Clamp
whatever the model passes: `max(5, min(10, int(seconds)))`.)

## 5. Interaction with the LLM server (the session-killer)

If your flush uses your framework's `generate_reply(instructions=...)`, know how those
instructions are injected. **LiveKit appends them as a system message at the END of the chat
context.** Strict chat templates (Qwen among them) hard-reject any system message that isn't
first → our server returned a non-retryable 404 → two in a row and LiveKit **closed the
whole session**. Every flush had been silently failing before that.

Fix in `llm_node` (or wherever you can intercept the request), and mind the prompt cache:

```python
# DON'T merge trailing system messages into the leading one — that edits the FRONT of the
# token sequence and busts the server's prefix cache for the whole conversation.
# DO re-role them in place: they stay at the end (append-only tokens = cache-optimal).
for i, it in enumerate(chat_ctx.items):
    if i > 0 and getattr(it, "role", None) == "system":
        it.role = "user"
        it.content[0] = "[Note to assistant] " + it.content[0]
```

## 6. Local-model failure modes (and fixes)

These three bit us with a 4-bit local Qwen. Sampled cloud models may hide them; greedy local
decoding will not.

### 6.1 StopResponse erases the tool call from history → the model never learns the format

In LiveKit, a tool that raises `StopResponse` produces `fnc_call_out=None`, and calls with
no output are **dropped from chat history entirely**. Consequence: every other tool's calls
appear (correctly formatted) in the model's own transcript on every request — this tool's
never do. Under greedy decoding the model went knife-edge on call-vs-prose *for this tool
only*, and in some contexts deterministically emitted the call as plain text — which TTS
then spoke aloud.

Fix: **record the call yourself** (the `update_chat_ctx` block in §4.1) so real holds leave
a well-formed (call, output) pair in history like every other tool.

### 6.2 The first hold of a session has no example yet → seed one

History recording can't help the *first* hold. Seed the session's initial chat context with
one synthetic worked example (fragment → proper tool call → output → user continues →
normal answer). Measured effect: a context that leaked 2/2 under greedy decoding flipped to
clean tool calls. It sits right after the system message = static prefix = prompt-cache
friendly.

```python
def _seed_hold_example():
    seed = ChatContext.empty()
    seed.add_message(role="user", content=["Hey, so."])
    seed.items.append(FunctionCall(call_id="seed_hold_1",
        name="hold_for_more_input", arguments='{"seconds": 3}'))
    seed.items.append(FunctionCallOutput(call_id="seed_hold_1",
        name="hold_for_more_input",
        output="Held silently; the user continued.", is_error=False))
    seed.add_message(role="user", content=["Hey, so. What's good with you?"])
    seed.add_message(role="assistant", content=["All good, just vibing. What's up?"])
    return seed
```

Keep the seed's content free of fake device actions or facts — harmless small talk only —
so the model can't "remember" doing something it never did.

### 6.3 Filler speech + tool call in one generation → the deadlock

The model may emit text AND the hold call together: *"Hold on, let me get the full
picture"* + `hold_for_more_input(3)`. That's fine — **unless agent speech cancels the flush
timer**. Ours did, producing the worst failure mode: the agent *promises* an answer, the
timer that would deliver it is dead, the user politely waits, both sides wait forever.

Rule: **agent speech resets the streak counter only, never the timer** (§4.2 table). With
that rule, filler+hold becomes the best UX variant: the filler covers the wait, the flush
answers seconds later.

### 6.4 Long-context recency bias — the model stops holding

Reproduced live: with a short context the model holds correctly, but past ~15–25 messages it
starts *replying* to borderline fragments ("I have a question", "can you", "okay so") instead of
holding — while clear trail-offs still hold. Cause: every recent turn shows a "user speaks →
assistant answers" pattern, and that recency outweighs the single hold example stuck far back in
the prefix. The seed (§6.2) fixes the *first* hold but decays as "reply" turns accumulate.

Fix: **re-inject a compact turn-taking reminder as a user-role message right before the current
turn**, once the context is long enough — putting the rule where recency helps. Make it idempotent
(strip any prior reminder by a tag, then re-insert) so it never accumulates even if you're editing
the persistent context. Measured: restores holding on the degrading fragments at 25+ turns, and
does NOT cause false holds on complete commands. This composes with the seed, not replaces it.

### 6.5 Backstop: a leak guard in the LLM output stream

Even with §6.1/§6.2, keep a guard that watches the reply stream for the tool name leaking as
prose and converts it into a real hold instead of letting TTS speak it. Match on a
**normalized** prefix (lowercase, letters+digits only) — the model invents spellings
(`hold_for_more_input(seconds=2)`, `HoldForMoreInput(2)`, `Hold for more input (3)`), and
all must collapse to the same key (`holdformoreinput`). Buffer only while the reply's prefix
is still ambiguous (~20 chars) so normal replies stream with no added latency, and verify
near-misses pass through ("Hold on, let me…", "Hold for a moment").

## 7. Observability (build it in from day one)

Log, at minimum:

- every hold: `HOLD wait=3s #1`, `HOLD superseded (user continued)`,
  `HOLD timeout → answering` — and leak-guard conversions
- every turn commit with the detector's probability and threshold
- user VAD state transitions (`listening→speaking`, `→away`)
- STT drops with reasons (energy gate, hallucination filter)
- an input-audio heartbeat (frames + peak level per ~5s)

This is what turns "the bot went quiet" from an unfalsifiable vibe into a one-look
diagnosis: no VAD events = upstream/mic problem; VAD events + STT drops = your gates; a
hold with no flush = your state machine; probabilities pinned at the max delay = your
detector threshold. Every failure in this guide was found this way.

## 8. Tuning notes

- **Hold duration**: our model defaults to picking 3s. Fine for trail-offs; long
  "thinking out loud" pauses may want prompt guidance toward 5–8s.
- **Cap**: 3 consecutive holds ≈ up to ~10–30s of accumulated waiting worst-case; the cap
  message ("answer now with what you have") converts the last hold into a reply.
- **Flush suppression**: without `_suppress_hold`, the model can hold again *during the
  flush generation* and loop.
- **Don't force silent holds if you don't want them.** We initially forbade verbal
  acknowledgments; in practice an occasional "take your time" is decent UX. Decide
  deliberately.
- **Temperature matters for repro.** Our leaks never reproduced at temp 0.8 and reproduced
  2/2 at the production setting (greedy). When debugging tool-format issues, test at the
  temperature your stack actually runs.

## 9. Results (this stack)

- Complete commands still commit at ~0.2s past silence (detector fast path intact,
  EOU ≈ 500ms end-to-end including STT).
- Sub-threshold fragments merge at the detector with zero LLM cost.
- Fragments that leak past the detector get held: in validation the LLM held on 100% of
  incomplete fragments and answered 100% of complete ones, including short commands.
- Failure modes fixed along the way: spoken tool-call leaks (prompt syntax → imitation;
  then greedy knife-edge → history recording + seed), trailing-system-message session
  death (re-role sanitizer), filler-speech deadlock (timer-cancel discipline).
