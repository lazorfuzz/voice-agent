import os
import re
import time
import random
import asyncio
import logging
import contextlib
import numpy as np

_eot_log = logging.getLogger("kronik.eot")
_audio_log = logging.getLogger("kronik.audio")
from dotenv import load_dotenv
from livekit import agents
from livekit.agents import (
    AgentServer, AgentSession, Agent, JobProcess, RoomInputOptions,
    function_tool, RunContext,
)
from livekit.agents.llm import StopResponse
from livekit.plugins import openai, silero, dtln
from livekit.plugins.turn_detector.english import EnglishModel

# MUST run before the local tool imports below: several tool modules read their config from
# the environment at IMPORT time (e.g. wiz_tools._LIGHTS / _BROADCAST). run_all.sh does NOT
# export .env.local into the shell, so without this the vars are unset when those modules
# load and the config silently freezes empty (lights show "none configured", etc.).
# Absolute path (next to this file), NOT cwd-relative — the worker may be launched from any
# directory; loading the wrong .env.local would silently disable every tool.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))

from local_stt import LocalWhisperSTT
from local_tts import PocketTTS
import agent_tools
import tv_tools
import nest_tools
import roomba_tools
import wiz_tools
import ring_tools
import ring_events
import mage_ring_client
import tesla_tools
import sing_tools
import petlibro_tools
import doordash_tools
import light_fx
import personas
import commands_freq      # frequent-command pills: voice turns that ran tools feed them too
import session_db as db   # the poller reads finished sessions to announce them
import config             # which integrations are enabled -> which tools to expose

WHISPER_REPO = "mlx-community/whisper-large-v3-turbo"

_BG_TASKS: set = set()   # strong refs to fire-and-forget background tasks (e.g. TV playback)

# --- Lead-hiding ("speak while a slow tool runs") -------------------------------------------
# Moshi/MoshiRAG-style: hide slow tool-call latency behind a short spoken "lead" ("Let me check
# the car...") so the wait isn't dead air. Uses LiveKit's RunContext.with_filler — a background
# scheduler that only speaks after the session has been CONTINUOUSLY IDLE for `delay`s, so
# fast/local tools (lights, roomba) that return first never fire a lead; only genuinely slow
# cloud/wake tools do. Leads are add_to_chat_ctx=False (ephemeral UX) so they never pollute the
# tool-vs-narrate history hygiene (_strip_unbacked_narrations). Disable with LEAD_FILLER=0.
_LEAD_ENABLED = os.environ.get("LEAD_FILLER", "1") != "0"
_LEAD_DELAY = float(os.environ.get("LEAD_FILLER_DELAY", "0.4"))


def _norm_text(s: str) -> str:
    """Casefold + collapse to alphanumerics-and-spaces, for fuzzy text identity checks."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s.lower()).split())


_LEAD_LAST = {"text": "", "t": 0.0}


def _lead(ctx, text, *, delay=_LEAD_DELAY, interval=None, max_steps=None):
    """Context manager wrapping a slow tool body: speak `text` WHILE the body runs, but only
    if it's still going after `delay`s of idle. `text` may be a str, a list of phrases
    (indexed by fire count, last one sticky), or a callable step->str|None.

    `interval` re-fires while the tool is STILL running — capped at 2 fires total by default:
    without a cap the SAME phrase repeated every ~4-5s for the length of an asleep-car Tesla
    wake ("Let me check the car." x4, heard live). No-op context when LEAD_FILLER=0."""
    if not _LEAD_ENABLED:
        return contextlib.nullcontext()
    if interval is not None and max_steps is None:
        max_steps = 2                      # lead + one "still working", never a chant

    def _src(step):
        if callable(text):
            t = text(step)
        elif isinstance(text, (list, tuple)):
            t = text[min(step, len(text) - 1)]
        else:
            t = text
        # Chained tool calls in one turn each open their own _lead — without this the
        # user hears the SAME phrase back-to-back ("One sec, setting the thermostat." x2
        # for set_thermostat_mode + set_thermostat_temperature, heard live 2026-07-25).
        if t and t == _LEAD_LAST["text"] and time.time() - _LEAD_LAST["t"] < 12.0:
            return None
        if t:
            _LEAD_LAST["text"], _LEAD_LAST["t"] = t, time.time()
        return ctx.session.say(t, add_to_chat_ctx=False) if t else None

    return ctx.with_filler(_src, delay=delay, interval=interval, max_steps=max_steps)


def _voice_llm():
    """The VOICE pipeline's LLM endpoint. VOICE_LLM_MODEL / VOICE_LLM_URL (agent/.env.local)
    override the shared LLM_MODEL / OPENAI_BASE_URL so voice can run a different model (e.g.
    a dedicated gemma server) while chat/opencode stay on the prod 0GM at :8081."""
    model = os.environ.get("VOICE_LLM_MODEL") or os.environ["LLM_MODEL"]
    url = (os.environ.get("VOICE_LLM_URL") or os.environ["OPENAI_BASE_URL"]).rstrip("/")
    return model, url


class _FarFieldGain:
    """Chains adaptive makeup gain BEFORE the DTLN denoiser (drop-in FrameProcessor).

    Far-field speech arrives at peak ~0.002-0.05 — browser AGC's range is limited and
    its noise suppressor gates faint reverberant speech — so ambient requests from
    across the room never even trip VAD (seen live 2026-07-26: close speech peak ~1.0,
    across-the-room peak 0.002 -> zero transcripts). Tracks a decaying rolling peak and
    applies a smoothed gain toward TARGET (capped at MAX_GAIN); near-silence below
    NOISE_FLOOR is left alone so hiss doesn't become phantom VAD hits. Close speech
    drives the gain to ~1, so this is a no-op for the normal near-mic case."""

    TARGET = 0.30
    MAX_GAIN = 25.0
    NOISE_FLOOR = 0.0012      # rolling peak below this = treat as silence, don't boost

    def __init__(self, inner):
        self._inner = inner
        self._rolling = 0.0
        self._gain = 1.0

    @property
    def enabled(self):
        return self._inner.enabled

    @enabled.setter
    def enabled(self, value):
        self._inner.enabled = value

    def _process(self, frame):
        try:
            arr = np.frombuffer(frame.data, dtype=np.int16)
            if arr.size:
                peak = float(np.abs(arr).max()) / 32768.0
                # rolling peak: instant attack, ~1.5s half-life decay (10ms frames)
                self._rolling = max(peak, self._rolling * 0.9955)
                if self._rolling > self.NOISE_FLOOR:
                    want = min(self.MAX_GAIN, max(1.0, self.TARGET / self._rolling))
                else:
                    want = 1.0
                # fast attack (quiet room -> speech), slower release: a short quiet
                # utterance must reach full boost within ~0.3s, not 1s+
                rate = 0.25 if want > self._gain else 0.05
                self._gain += (want - self._gain) * rate
                if self._gain > 1.05:
                    boosted = np.clip(arr.astype(np.float32) * self._gain,
                                      -32767, 32767).astype(np.int16)
                    # rewrite the frame buffer in place (same length/layout)
                    np.frombuffer(frame.data, dtype=np.int16)[:] = boosted
        except Exception:
            pass                                   # gain is best-effort, never break audio
        return self._inner._process(frame)

    def _close(self):
        return self._inner._close()

    def _on_stream_info_updated(self, **kw):
        return self._inner._on_stream_info_updated(**kw)

    def _on_stream_info_cleared(self):
        return self._inner._on_stream_info_cleared()

    def _on_credentials_updated(self, **kw):
        return self._inner._on_credentials_updated(**kw)

    def _on_credentials_cleared(self):
        return self._inner._on_credentials_cleared()


def _pipelog(line):
    """Millisecond-stamped pipeline trace -> latency.log (decomposes the 'hidden' gap between
    LLM first token and audible audio: sentence buffering, synthesis start, first frame out)."""
    try:
        t = time.time()
        stamp = time.strftime("%H:%M:%S", time.localtime(t)) + f".{int(t*1000)%1000:03d}"
        with open(os.path.join(config.REPO_ROOT, "latency.log"), "a") as f:
            f.write(f"{stamp} {line}\n")
    except Exception:
        pass


def _soulx_detector():
    """Lazy import: only touched when TURN_DETECTOR=soulx (keeps default path unchanged)."""
    from soulx_turn import SoulXTurnDetector
    return SoulXTurnDetector()


def _gpu_warm_blocking():
    """One trivial 1-token LLM call to keep the mlx server's GPU context ramped.

    The LLM itself is ~165ms TTFT warm, but the *server process* idles between
    turns and its Metal context ramps down -> the next real call pays ~300ms. A
    fixed tiny prompt (stable, so it doesn't thrash the LRU prompt cache) fired
    during idle 'listening' periods keeps it hot. Best-effort; errors ignored."""
    import json as _json
    import urllib.request
    body = {
        "model": _voice_llm()[0], "max_tokens": 1, "temperature": 0.0,
        "messages": [{"role": "user", "content": "ok"}],
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        _voice_llm()[1] + "/chat/completions",
        data=_json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"})
    try:
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass


def setup(proc: JobProcess):
    """Runs once per (idle) job process, BEFORE a call is assigned. Preloads the
    heavy local models so a warm process can greet instantly instead of cold-loading
    Pocket-TTS/Whisper/VAD (~15-20s) at call time. Paired with num_idle_processes so the
    worker always has a ready process -> the agent reliably joins on first try."""
    proc.userdata["vad"] = silero.VAD.load(
        activation_threshold=0.55,     # a bit above default 0.5 -> ignores background noise, still responsive
        min_speech_duration=0.15,      # ignore <150ms blips (default 0.05) but react quickly to real speech
        min_silence_duration=0.25,     # was 0.35: with SoulX gating turn commits (not VAD finals),
                                       # a shorter close is safe — offline test 2026-07-21: segment
                                       # counts IDENTICAL at 0.35 vs 0.25 on fluent + hesitation
                                       # clips, so this is a free ~100ms off transcription_delay
        prefix_padding_duration=0.3,   # keep the leading audio so onsets aren't clipped
    )
    # Prewarm every distinct persona voice, ALL sharing one loaded pocket-tts model
    # (load_model isn't a singleton, so a fresh load per voice would duplicate the
    # ~440MB weights). entry() picks the voice for the selected persona. ~22ms TTFB.
    from pocket_tts import TTSModel
    _tts_model = TTSModel.load_model()
    proc.userdata["tts_model"] = _tts_model   # shared model, for building new voices on demand
    proc.userdata["tts_by_voice"] = {
        v: PocketTTS(voice=v, model=_tts_model) for v in personas.distinct_voices()
    }
    # STT engine (STT_ENGINE env): parakeet (default — Parakeet-TDT-0.6b-v3: lower WER, ~4x faster
    # STT than whisper, and a matched live A/B (2026-07-16) confirmed it's latency-neutral for LLM
    # TTFT — the earlier "slower" read was an unmatched conversation, not the STT). whisper =
    # large-v3-turbo w/ initial_prompt wake-word bias; qwen = Qwen3-ASR. STT hides under the ~620ms
    # EOU wait, so engine choice is about accuracy, not speed. Wake word fixed in post for parakeet.
    # Default engine is platform-aware: Apple Silicon runs the MLX Parakeet path; anything
    # else (Linux/Docker, CPU or CUDA) has no MLX, so default to the faster-whisper backend.
    # STT_ENGINE always overrides.
    _default_stt = "parakeet" if os.uname().sysname == "Darwin" else "faster-whisper"
    _stt_engine = os.environ.get("STT_ENGINE", _default_stt).lower()
    if _stt_engine in ("faster-whisper", "faster_whisper", "fasterwhisper", "fw"):
        from local_stt import FasterWhisperSTT
        proc.userdata["stt"] = FasterWhisperSTT()
    elif _stt_engine == "whisper":
        proc.userdata["stt"] = LocalWhisperSTT(model=WHISPER_REPO)
        try:
            import mlx_whisper
            mlx_whisper.transcribe(np.zeros(16000, dtype=np.float32), path_or_hf_repo=WHISPER_REPO)
        except Exception:
            pass
    elif _stt_engine == "qwen":
        from local_stt import LocalQwenASR
        proc.userdata["stt"] = LocalQwenASR()
    elif _stt_engine in ("parakeet-stream", "parakeet_stream", "stream"):
        # Phase 2 dual-decoder: streaming INTERIM/PREFLIGHT partials (-> preemptive generation)
        # + authoritative batch biased FINAL. Roll back with STT_ENGINE=parakeet.
        from local_stt import LocalStreamingParakeetSTT
        proc.userdata["stt"] = LocalStreamingParakeetSTT()
    else:
        from local_stt import LocalParakeetSTT
        proc.userdata["stt"] = LocalParakeetSTT()
    # Populate the WiZ MAC->IP cache so the light-pulse hot path never does a slow
    # in-loop discovery (keeps the per-utterance snapshot instant).
    try:
        wiz_tools.prime_discovery()
    except Exception:
        pass
    # Preload the speaker-verification encoder (ambient voice gate) so the first gated
    # utterance isn't cold.
    try:
        import speaker_id
        speaker_id.warm()
    except Exception:
        pass


# Persona instructions + greetings moved to personas.py (selectable Kronik/Consuela).


def _fmt_age(seconds: float) -> str:
    """Human/agent-friendly 'last active' age, so Kronik can reason about staleness."""
    s = int(max(0, seconds))
    if s < 60:
        return "just now"
    m = s // 60
    if m < 60:
        return f"{m}m ago"
    h, rm = divmod(m, 60)
    if h < 24:
        return f"{h}h {rm}m ago" if rm else f"{h}h ago"
    d = h // 24
    return f"{d}d ago"


# Max consecutive holds before we force an answer, so the agent can never sit in
# dead air if the LLM keeps deciding to wait.
_MAX_HOLDS = 3

# Ambient mode: after being addressed by name, stay engaged for this long so natural
# follow-ups ("...make it 20") work without repeating the wake word. Going quiet past
# this returns to passive listening (name required again).
_WAKE_WINDOW = 10.0

# Ambient sessions run for hours, so the conversation history (and thus the LLM's KV /
# GPU prompt cache) grows unbounded -> the mlx server eventually Metal-OOMs and crashes.
# Cap each request to the static prefix (system + seed) + the last N conversation items.
_AMBIENT_CTX_TAIL = 20

# The 4-bit local model occasionally writes the hold call as TEXT instead of emitting a
# real tool call — observed live: "hold_for_more_input(seconds=2)" and later
# "HoldForMoreInput(2)" (it invents spellings, so match on a NORMALIZED prefix: lowercase,
# letters+digits only). llm_node watches for replies that start like this and converts
# them into a real hold instead of speech.
# Silent-control tools the model must call (not narrate). llm_node catches them leaking
# as prose and enacts them instead of speaking. Names chosen so their normalized form
# never prefixes a normal reply. "hold" -> a real hold; "ignore" -> just stay silent.
_LEAK_TARGETS = ("holdformoreinput", "ignoreinput")
_LEAK_NORM_RE = re.compile(r"[^a-z0-9]+")

# In a long conversation the model develops a recency bias toward REPLYING (every recent
# turn shows "user speaks -> assistant answers"), which outweighs the silent-control
# examples stuck far back in the prefix -> it stops holding on borderline fragments ("I
# have a question", "can you", "okay so"), and in ambient mode stops ignoring overheard
# talk. Re-injecting this compact reminder just before the current turn (where recency
# helps) restores both, without causing false holds/ignores on complete addressed
# commands. Tagged so we can strip/reposition it idempotently.
_HOLD_REMINDER_TAG = "[turn-taking]"
_HOLD_REMINDER_BODY = (
    "If my latest message is not a complete, actionable request — it trails off, is a filler, "
    "or a lead-in like 'I have a question' / 'okay so' / 'can you' — call hold_for_more_input "
    "and stay silent instead of replying.")
_IGNORE_REMINDER_BODY = (
    "Also DEFAULT TO ignore_input: only reply if my latest message is CLEARLY a request or command "
    "for you. For chatter, reactions, or anything ambiguous, call ignore_input. When in doubt, ignore.")


def _turn_taking_reminder(ambient: bool) -> str:
    body = _HOLD_REMINDER_BODY + (" " + _IGNORE_REMINDER_BODY if ambient else "")
    return _HOLD_REMINDER_TAG + " " + body


def _is_hold_reminder(it) -> bool:
    return (getattr(it, "role", None) == "user" and getattr(it, "content", None)
            and isinstance(it.content[0], str) and it.content[0].startswith(_HOLD_REMINDER_TAG))


# Root-cause fix for the tool-vs-narrate cascade. On the 4-bit MoE, once the model has spoken an
# UNBACKED device narration ("Lamps are 10%" with no control_lights call) it IMITATES that pattern
# and keeps narrating instead of acting, so the device stops changing — a doom loop seeded by its
# own history (measured: 0% tool-call rate in a narration-polluted context, 100% once removed).
# The fix is structural: drop the model's own OLDER unbacked prose from the context it plans
# against, so a pile of prior narrations can't poison its act-vs-talk decision. An assistant
# message is "backed" iff a real tool call produced it (previous item is a FunctionCall/
# FunctionCallOutput) — those confirmations, tool history, all user turns, and the seed are kept.
# BUT we KEEP the last _KEEP_RECENT_REPLIES unbacked replies: measured that a tail of purely
# CONVERSATIONAL replies (even ones naming devices, e.g. "I can do lights, TV…") does NOT poison
# tool-calling (100%), only device-STATE narrations do (33%) — and real device actions produce
# BACKED confirmations, not unbacked narrations, so the recent unbacked tail is ~always chat, not
# poison. Keeping it restores multi-turn memory (else "what else besides those?" re-lists the same
# things, because the model can't see what it just said).
_KEEP_RECENT_REPLIES = 4


def _strip_unbacked_narrations(items, keep_prefix, keep_recent=_KEEP_RECENT_REPLIES):
    """Drop unbacked assistant prose past the seed prefix EXCEPT the last `keep_recent` such
    messages (kept for conversational continuity). Backed confirmations + non-assistant items
    are always kept. This breaks the long-run narration cascade without erasing recent memory."""
    unbacked = []
    for i in range(keep_prefix, len(items)):
        it = items[i]
        if getattr(it, "role", None) == "assistant":
            prev = items[i - 1] if i > 0 else None
            if type(prev).__name__ not in ("FunctionCall", "FunctionCallOutput"):
                unbacked.append(i)
    drop = set(unbacked if keep_recent <= 0 else unbacked[:-keep_recent])
    return [it for i, it in enumerate(items) if i not in drop]


def _norm_alnum(s: str) -> str:
    return _LEAK_NORM_RE.sub("", s.lower())


def _chunk_text(ch) -> str:
    """Text content of an llm_node chunk (ChatChunk | str | FlushSentinel)."""
    if isinstance(ch, str):
        return ch
    delta = getattr(ch, "delta", None)
    return getattr(delta, "content", None) or ""


def _chunk_has_tool_call(ch) -> bool:
    delta = getattr(ch, "delta", None)
    return bool(getattr(delta, "tool_calls", None))


def _seed_examples(ambient: bool = False):
    """Synthetic worked examples of the silent-control tools, seeded into the session's
    initial chat context. Why: the 4-bit Qwen is knife-edge on the call-vs-prose format
    decision for a StopResponse tool under greedy decoding (reproduced live: a context
    leaked 'HoldForMoreInput(seconds=2)' as speech 2/2 while near-identical contexts
    called cleanly). These tools have no organic examples in history — StopResponse erases
    them — so provide them explicitly; measured to flip the leaking context to clean tool
    calls. Static prefix right after the system message = prompt-cache friendly. The
    ignore_input example is only seeded in ambient mode (where that tool exists)."""
    from livekit.agents.llm import ChatContext, FunctionCall, FunctionCallOutput
    seed = ChatContext.empty()
    seed.add_message(role="user", content=["Hey, so."])
    seed.items.append(FunctionCall(
        call_id="seed_hold_1", name="hold_for_more_input", arguments='{"seconds": 5}'))
    seed.items.append(FunctionCallOutput(
        call_id="seed_hold_1", name="hold_for_more_input",
        output="Held silently; the user continued.", is_error=False))
    seed.add_message(role="user", content=["Hey, so. What's good with you?"])
    seed.add_message(role="assistant", content=["All good, just vibing. What's up?"])
    # Anchor "device command -> REAL tool call" (both modes). Without it the model's
    # tool-vs-narrate decision for action tools is knife-edge, and once it narrates one
    # light command ("Okay, I set the lamps to X" with no tool call) that turn sits in
    # history and self-reinforces narration on the next one -> the device stops changing.
    # End at the tool OUTPUT (no trailing spoken confirmation): a trailing "Done." gets
    # imitated as a bare narration ("Done." with no tool call). The hold/ignore examples
    # work for the same reason — they end at the output.
    seed.add_message(role="user", content=["Set the living room lamp to 40."])
    seed.items.append(FunctionCall(
        call_id="seed_act_1", name="control_lights",
        arguments='{"action": "brightness", "brightness": 40, "which": "living room"}'))
    seed.items.append(FunctionCallOutput(
        call_id="seed_act_1", name="control_lights",
        output="Set living room 1 to 40 percent.", is_error=False))
    # Anchor "sing request -> sing_song tool call with SELF-WRITTEN lyrics" for the same
    # knife-edge reason: without it the model sometimes SPEAKS the lyrics as its reply
    # ("Here's a quick one for you: ..." — heard live 2026-08-06) and no song plays.
    # Same rule as above: end at the tool OUTPUT, no trailing spoken confirmation.
    if sing_tools.available():
        seed.add_message(role="user", content=["Can you sing something for me?"])
        seed.items.append(FunctionCall(
            call_id="seed_sing_1", name="sing_song",
            arguments='{"lyrics": "Sunlight through the window, coffee in my hand, '
                      'lazy morning music only we can understand, nowhere we should '
                      'hurry, nothing left to do, just a little melody from me to you"}'))
        seed.items.append(FunctionCallOutput(
            call_id="seed_sing_1", name="sing_song",
            output="You are now writing the song and warming up your voice. Tell the user "
                   "you're writing a song for them — it starts playing automatically in "
                   "about half a minute. Do NOT speak the lyrics.",
            is_error=False))
    if ambient:
        # ONE overheard-chatter -> ignore_input example. Counterintuitively, MORE ignore
        # examples (had 3) over-bias the 4-bit model toward silence and it starts
        # NARRATING action commands instead of calling their tools (the device stops
        # responding). One ignore example + the every-turn ignore reminder keeps ignore
        # working while leaving action tool-calls intact. Order matters: after the action
        # example, not before.
        seed.add_message(role="user", content=["...yeah I'll text her back later, one sec."])
        seed.items.append(FunctionCall(call_id="seed_ign_1", name="ignore_input", arguments="{}"))
        seed.items.append(FunctionCallOutput(
            call_id="seed_ign_1", name="ignore_input",
            output="Ignored; not addressed to me.", is_error=False))
    return seed


class Assistant(Agent):
    def __init__(self, instructions: str = None, ambient: bool = False, wake_words=None,
                 greetings=None):
        super().__init__(
            instructions=instructions or personas.build_voice_instructions(
                personas.PERSONAS[personas.DEFAULT]),
            chat_ctx=_seed_examples(ambient))
        # --- prompt-based EOT "escape hatch" state (layered on top of EnglishModel) ---
        self._hold_task: asyncio.Task | None = None   # pending "answer on timeout" timer
        self._hold_count = 0                            # consecutive holds w/o speaking
        self._suppress_hold = False                     # true while flushing -> don't re-hold
        # --- ambient (wake-word) mode ---
        self._ambient = ambient                         # only respond when addressed by name
        self._wake_words = wake_words or []
        self._greetings = [_norm_text(g) for g in (greetings or [])]  # recital guard
        self._persona_voice = "jean"          # set at session setup (persona's TTS voice)
        self._tts_model = None                # shared pocket-tts model (for sing reference)
        self._room = None                     # rtc room, set at session setup
        self._ambient_active_until = 0.0                # follow-up window deadline (monotonic)
        self._mage_ring = mage_ring_client.MageRingWatcher()
        # seed size (system msg is added separately at request time) -> the static prefix
        # to always keep when truncating a long ambient session's history.
        self._seed_count = len(self.chat_ctx.items)

    def _cancel_hold(self):
        if self._hold_task is not None and not self._hold_task.done():
            self._hold_task.cancel()
        self._hold_task = None

    def _reset_holds(self):
        """Called when the agent SPEAKS -> the hold STREAK is over. Deliberately does
        NOT cancel a pending flush timer: the model sometimes emits filler text in the
        SAME generation as the hold tool call ("Hold on, let me get the full picture") —
        that speech must not kill the flush, or the agent promises an answer that never
        comes and the call deadlocks in silence (observed live 2026-07-10 02:13). Only
        on_user_turn_completed (the user actually continued) cancels the timer."""
        self._hold_count = 0

    def _begin_hold(self, session, secs: int):
        """Schedule the silent wait + flush timer. Shared by the hold tool and the
        llm_node leak interceptor (when the model writes the call as text)."""
        self._hold_count += 1
        _eot_log.info("EOT-HOLD  wait=%ss  hold#%d", secs, self._hold_count)

        async def _flush():
            try:
                await asyncio.sleep(secs)
            except asyncio.CancelledError:
                _eot_log.info("EOT-HOLD  superseded (user continued)")
                return
            _eot_log.info("EOT-HOLD  timeout after %ss -> answering", secs)
            # Timer elapsed with no further speech -> answer what we have. Clear the task
            # ref FIRST so the speak->_reset_holds hook can't cancel our own reply, and
            # set _suppress_hold so the model can't decide to hold yet again on this flush.
            self._hold_task = None
            self._hold_count = 0
            self._suppress_hold = True
            try:
                await session.generate_reply(
                    instructions="The user paused and didn't continue. Respond now to what "
                                 "they've said so far; do not wait for more.",
                )
            except Exception:
                pass
            finally:
                self._suppress_hold = False

        self._cancel_hold()
        self._hold_task = asyncio.create_task(_flush())
        _BG_TASKS.add(self._hold_task)
        self._hold_task.add_done_callback(_BG_TASKS.discard)

    async def on_user_turn_completed(self, turn_ctx, new_message):
        # The user produced more input, so supersede any pending "answer on timeout":
        # the fuller transcript is already in history and generation will re-decide
        # (respond now, or hold again if it's still incomplete).
        self._cancel_hold()
        # Ambient mode: only respond when addressed by name, or during the brief follow-up
        # window after a response. Otherwise stay silent — raising StopResponse here also
        # keeps the un-addressed turn OUT of chat history (framework drops it), so ambient
        # chatter never pollutes context.
        if self._ambient:
            text = getattr(new_message, "text_content", None) or ""
            now = time.monotonic()
            if personas.is_addressed(text, self._wake_words):
                self._ambient_active_until = now + _WAKE_WINDOW   # named -> open window, respond
            elif now < self._ambient_active_until:
                # in the follow-up window -> let the LLM decide (it may call ignore_input
                # if the message wasn't for it). Do NOT extend the window here, or overheard
                # chatter would keep it open forever; only a REAL response reopens it (via
                # the agent_state 'speaking' hook).
                pass
            else:
                _eot_log.info("AMBIENT ignore (not addressed): %r", text[:60])
                raise StopResponse()

    def _truncate_ctx(self, chat_ctx):
        """Keep the static prefix (leading system message(s) + the seed examples) plus
        only the last _AMBIENT_CTX_TAIL conversation items. Never start the kept tail on
        an orphaned FunctionCall/FunctionCallOutput — the API requires each call to be
        paired with its output, so drop leading tool items until a real message."""
        try:
            items = chat_ctx.items
            p = 0
            while p < len(items) and getattr(items[p], "role", None) == "system":
                p += 1                        # leading persona/system message(s)
            p += self._seed_count             # + the seed examples
            tail = items[p:]
            if len(tail) <= _AMBIENT_CTX_TAIL:
                return
            tail = tail[-_AMBIENT_CTX_TAIL:]
            while tail and type(tail[0]).__name__ in ("FunctionCall", "FunctionCallOutput"):
                tail.pop(0)
            chat_ctx.items[:] = items[:p] + tail
        except Exception:
            pass  # truncation must never take down the LLM path

    async def llm_node(self, chat_ctx, tools, model_settings):
        """Wrap the default LLM stream with context hygiene only: (a) re-role any trailing
        system message to a user note (Qwen3.6 rejects non-leading system messages), (b) inject
        the turn-taking reminder before a fresh user turn, (c) strip the model's own unbacked
        narrations. Then stream the reply straight through — no output buffering."""
        # (a) Qwen3.6's chat template hard-rejects any system message that isn't FIRST
        # ("System message must be at the beginning" -> unrecoverable 404 that KILLED a
        # live session 2026-07-10). LiveKit's generate_reply(instructions=...) appends
        # instructions as a trailing system message (agent_activity add_message). Fix by
        # RE-ROLING it to a marked user message IN PLACE — not by merging into the
        # leading system message, which would edit the front of the token sequence and
        # bust the server's prefix cache for the entire conversation history. Re-roling
        # keeps the history prefix byte-identical; only the appended note is new tokens.
        try:
            for i, it in enumerate(chat_ctx.items):
                if i > 0 and getattr(it, "role", None) == "system":
                    it.role = "user"
                    if it.content and isinstance(it.content[0], str):
                        it.content[0] = "[Note to assistant] " + it.content[0]
        except Exception:
            pass  # sanitizing must never take down the LLM path

        # ignore_input only makes sense in ambient mode; don't even offer it otherwise or
        # the model spuriously calls it (e.g. on "tell me a joke") and wastes a turn.
        if not self._ambient:
            tools = [t for t in tools if getattr(t, "__name__", None) != "ignore_input"]
        else:
            self._truncate_ctx(chat_ctx)      # bound long ambient history -> avoid GPU OOM

        # DoorDash has a large multi-step toolset. On an explicit DoorDash request, and on the
        # model turn immediately following a DoorDash tool result, omit unrelated home-control
        # schemas from this one completion. This is deliberately narrow: an ordinary fresh user
        # turn still gets every enabled tool, so changing the subject never traps the session in
        # an ordering-only mode.
        try:
            items = chat_ctx.items
            last = items[-1] if items else None
            last_text = getattr(last, "text_content", None) or ""
            follows_doordash_tool = (
                type(last).__name__ == "FunctionCallOutput"
                and str(getattr(last, "name", "")).startswith("doordash_")
            )
            explicit_doordash_request = (
                getattr(last, "role", None) == "user"
                and "doordash" in last_text.casefold()
            )
            if follows_doordash_tool or explicit_doordash_request:
                tools = [
                    tool for tool in tools
                    if str(getattr(getattr(tool, "info", None), "name", "")).startswith(
                        "doordash_"
                    )
                    or getattr(getattr(tool, "info", None), "name", "") == "hold_for_more_input"
                ]
        except Exception:
            pass

        # Counter the long-context recency bias that makes the model stop holding on
        # borderline fragments: strip any stale reminder, then (once the convo is long
        # enough) re-inject a fresh one right before the current turn. Idempotent so it
        # never accumulates even if chat_ctx is the persistent context.
        try:
            from livekit.agents.llm import ChatMessage
            items = chat_ctx.items
            items[:] = [it for it in items if not _is_hold_reminder(it)]
            # ONLY inject before a FRESH user turn (last item is a user message). The
            # reminder guides the respond/hold/ignore decision for that utterance. On a
            # tool-continuation turn (last item is a FunctionCallOutput/FunctionCall, where
            # the model should just voice the tool result) injecting it is wrong twice over:
            # it splits the call/output pair, AND the more instruction-following finetune
            # RECITES the instruction aloud instead of the result — observed live: "check
            # the thermostat" ran thermostat_status, then the agent SPOKE "Got it. I'll stay
            # quiet unless you have a clear request for me." instead of the reading.
            # Ambient: inject EVERY (fresh-user) turn — the ignore bias must be present even
            # early in a session (short context) or overheard chatter in the wake window gets
            # replied to, and each reply reopens the window -> a runaway cascade. Normal mode
            # only needs it once the context is long enough to induce the recency bias.
            last_is_user_turn = bool(items) and getattr(items[-1], "role", None) == "user"
            if last_is_user_turn and (self._ambient or len(items) > self._seed_count + 8):
                items.insert(len(items) - 1,
                             ChatMessage(role="user", content=[_turn_taking_reminder(self._ambient)]))
        except Exception:
            pass  # reminder is best-effort; never break the LLM path

        # History hygiene: strip the model's own UNBACKED narrations from the context before it
        # plans this turn (see _strip_unbacked_narrations). This is the real cascade fix — purely
        # structural, no guessing whether a turn "is a command". Mutating in place keeps the
        # pollution from ever accumulating in the persistent context. Best-effort.
        try:
            chat_ctx.items[:] = _strip_unbacked_narrations(chat_ctx.items, 1 + self._seed_count)
        except Exception:
            pass

        # Leak guard removed (2026-07-16): stream the reply straight through with no output
        # buffering. Trade-off: if the model ever writes hold_for_more_input/ignore_input as
        # prose instead of a real tool call, it will now be spoken aloud (and no auto-hold);
        # the model normally emits real tool calls, so this is rare.
        #
        # Greeting-recital guard (observed live 2026-07-25): on a rare degenerate
        # tool-continuation turn the finetune replies with its own seeded GREETING verbatim
        # ("turn on the AC" -> both thermostat tools ran -> spoke "Consuela here. I no
        # sleeping, I just waiting. What you want?"). Chunks are held only WHILE the text is
        # still a prefix of a known greeting — normal replies diverge within a token or two,
        # so this adds no latency — and on a full match that stream is dropped and the turn
        # regenerated once (sampling temperature usually escapes the degenerate mode).
        _first_ch = True
        for _attempt in (0, 1):
            held, text = [], ""
            deciding = bool(self._greetings) and _attempt == 0
            recited = False
            stream = Agent.default.llm_node(self, chat_ctx, tools, model_settings)
            async for ch in stream:
                if _first_ch:
                    _first_ch = False
                    _pipelog("PIPE llm_first_chunk")
                if not deciding:
                    yield ch
                    continue
                delta = getattr(ch, "delta", None)
                c = (getattr(delta, "content", None) or "") if delta else ""
                held.append(ch)
                text += c
                if delta is not None and getattr(delta, "tool_calls", None):
                    deciding = False               # tool call -> not a recital
                else:
                    nt = _norm_text(text)
                    live = [g for g in self._greetings
                            if g.startswith(nt) or nt.startswith(g)]
                    if live and any(nt.startswith(g) for g in live):
                        recited = True             # reproduced a full greeting
                        break
                    if live:
                        continue                   # still ambiguous: keep holding
                    deciding = False               # diverged from every greeting
                for h in held:
                    yield h
                held = []
            if recited:
                _eot_log.warning("greeting recital detected — regenerating the turn")
                try:
                    await stream.aclose()
                except Exception:
                    pass
                continue
            for h in held:                         # stream ended while still ambiguous
                yield h
            return

    def stt_node(self, audio, model_settings):
        """Tap the incoming mic stream with a ~5s heartbeat (frame count + peak level)
        to agent.log. Decisive forensics for 'the bot stopped hearing me': heartbeats
        stop = upstream/phone stopped delivering frames (e.g. iOS screen lock);
        heartbeats with peak~0 = phone sends silence; normal peaks but no transcripts =
        VAD/STT issue. Frames pass through untouched."""
        async def tapped():
            n, peak, last = 0, 0.0, time.monotonic()
            async for f in audio:
                try:
                    arr = np.frombuffer(f.data, dtype=np.int16)
                    if arr.size:
                        peak = max(peak, float(np.abs(arr).max()) / 32768.0)
                    n += 1
                    now = time.monotonic()
                    if now - last >= 5.0:
                        _audio_log.info("AUDIO-IN %d frames/%.1fs peak=%.3f", n, now - last, peak)
                        n, peak, last = 0, 0.0, now
                except Exception:
                    pass
                yield f
        return Agent.default.stt_node(self, tapped(), model_settings)

    async def tts_node(self, text, model_settings):
        """Tap the TTS frame stream to drive the WiZ lights like a waveform (light_fx):
        aggregate frames into ~100ms RMS buckets stamped with CUMULATIVE AUDIO TIME (TTS
        synthesizes faster than realtime, so wall-clock arrival would run ahead of what's
        audible) and feed them to the pulse driver. Frames pass through untouched."""
        fx = light_fx.driver
        fx.begin()
        cum = 0.0          # audio seconds emitted before the current bucket
        b_dur = 0.0        # current bucket duration
        b_rms = 0.0        # current bucket peak RMS
        b_pcm = []         # current bucket samples (for the spectral centroid -> color)

        async def _tapped_text():
            # PIPE trace: when the FIRST text reaches TTS = end of the sentence-buffer wait
            # (the default pipeline accumulates LLM tokens into a full first sentence before
            # a non-streaming TTS may start — invisible to the standard metrics).
            first = True
            async for chunk in text:
                if first:
                    first = False
                    _pipelog("PIPE tts_first_text")
                yield chunk

        _first_frame = True
        try:
            async for frame in Agent.default.tts_node(self, _tapped_text(), model_settings):
                if _first_frame:
                    _first_frame = False
                    _pipelog("PIPE tts_first_frame")
                try:
                    arr = np.frombuffer(frame.data, dtype=np.int16)
                    if arr.size:
                        f = arr.astype(np.float32) / 32768.0
                        b_rms = max(b_rms, float(np.sqrt(np.mean(f * f))))
                        b_pcm.append(f)
                    b_dur += frame.samples_per_channel / frame.sample_rate
                    if b_dur >= 0.1:
                        # spectral centroid (timbre -> color) + F0 (pitch -> hue sweep)
                        centroid = f0 = 0.0
                        if b_pcm:
                            centroid, f0 = light_fx.analyze(
                                np.concatenate(b_pcm), frame.sample_rate)
                        fx.feed(cum, b_rms, centroid, f0)
                        cum += b_dur
                        b_dur = b_rms = 0.0
                        b_pcm = []
                except Exception:
                    pass   # light show must never break speech
                yield frame
        finally:
            fx.end()

    @function_tool
    async def hold_for_more_input(self, ctx: RunContext, seconds: int) -> str | None:
        """Stay SILENT and keep listening because the user isn't finished talking yet.
        Call this INSTEAD of replying when the user's turn is clearly incomplete — they
        trailed off, paused mid-thought, or ended on a filler/connector. Never call it for
        a short-but-complete request.

        Args:
            seconds: how long to wait for them to continue, 5-10 (longer if they sound like
                     they need a beat to think, shorter if they'll likely finish fast).
        """
        # If we're already flushing a held turn, don't hold again -> answer now.
        if self._suppress_hold:
            return "The user has stopped; answer now with what you have."
        # Safety cap: never hold more than _MAX_HOLDS in a row without speaking.
        if self._hold_count >= _MAX_HOLDS:
            self._reset_holds()
            return "You've waited long enough; answer now with what you have."

        secs = max(5, min(10, int(seconds)))
        # Record this call + a synthetic output in the chat history BEFORE stopping the
        # response. StopResponse makes LiveKit produce fnc_call_out=None, and calls with
        # no output are DROPPED from history entirely (agent_activity: `if fnc_call_out
        # is not None`) — so the model never saw a well-formed example of THIS tool call
        # in its own transcript (every other tool returns a string and gets recorded).
        # That asymmetry taught it to imitate hold turns as plain text
        # ("HoldForMoreInput(2)" spoken aloud) while all other tools called cleanly.
        try:
            from livekit.agents.llm import FunctionCallOutput
            chat_ctx = self.chat_ctx.copy()
            chat_ctx.items.append(ctx.function_call.model_copy())
            chat_ctx.items.append(FunctionCallOutput(
                call_id=ctx.function_call.call_id, name=ctx.function_call.name,
                output=f"Holding silently for {secs}s to let the user finish.",
                is_error=False,
            ))
            await self.update_chat_ctx(chat_ctx)
        except Exception:
            _eot_log.exception("EOT-HOLD failed to record call in history")

        self._begin_hold(ctx.session, secs)
        # Produce no spoken reply for this turn — the escape hatch.
        raise StopResponse()

    @function_tool
    async def ignore_input(self, ctx: RunContext) -> str | None:
        """Ambient mode only: STAY SILENT because the latest message was NOT addressed to
        you (overheard background talk, someone speaking to another person, or a stray
        fragment). Call this instead of replying. Do NOT call it when the user is talking
        to you."""
        if not self._ambient:
            return "Not in ambient mode; just answer the user normally."
        # Record the call in history (like hold) so the model keeps seeing a well-formed
        # example — StopResponse otherwise erases it, which is what makes the 4-bit model
        # start narrating the tool name instead of calling it.
        try:
            from livekit.agents.llm import FunctionCallOutput
            chat_ctx = self.chat_ctx.copy()
            chat_ctx.items.append(ctx.function_call.model_copy())
            chat_ctx.items.append(FunctionCallOutput(
                call_id=ctx.function_call.call_id, name=ctx.function_call.name,
                output="Ignored; not addressed to me.", is_error=False))
            await self.update_chat_ctx(chat_ctx)
        except Exception:
            _eot_log.exception("AMBIENT ignore_input failed to record call")
        _eot_log.info("AMBIENT ignore_input (model judged not-for-me)")
        raise StopResponse()

    @function_tool
    async def run_agent(self, ctx: RunContext, task: str, name: str) -> str:
        """Dispatch a background local AI agent (opencode) to do a coding, research, or analysis
        task, or to figure out something you don't know the answer to. Runs asynchronously and is
        saved under `name` so it can be resumed later.

        Args:
            task: what the agent should do, in plain language.
            name: a short memorable name to save the session under (e.g. "bug hunt").
        """
        # offload the DB write + detached spawn to a thread: returns immediately,
        # never blocks the voice loop, and never stalls on a busy DB lock.
        return await asyncio.to_thread(agent_tools.dispatch, name, task)

    @function_tool
    async def list_sessions(self, ctx: RunContext) -> str:
        """List the user's recent saved agent sessions with their status and how long ago each was
        last active. Use when the user asks about recent sessions, which agents are running, or
        which are old/stale (so you can decide what to delete)."""
        rows = agent_tools.list_recent()   # fast WAL read; caller needs the data
        if not rows:
            return "No agent sessions yet."
        now = time.time()
        parts = [
            f"{r['name']} — {r['status']}, last active {_fmt_age(now - (r.get('updated_at') or now))}"
            for r in rows
        ]
        return "Recent sessions: " + "; ".join(parts)

    @function_tool
    async def get_session_status(self, ctx: RunContext, name: str) -> str:
        """Get the latest state/result of a saved agent session by name. Use when the user asks
        how a specific session is going or what it found.

        Args:
            name: the saved session name to look up.
        """
        s = agent_tools.get_state(name)   # fast WAL read; caller needs the data
        if not s:
            return f"No session called {name}."
        age = _fmt_age(time.time() - (s.get("updated_at") or time.time()))
        if s["status"] == "running":
            return f"{name} is still working on it (last active {age})."
        out = (s.get("last_output") or "").strip()
        return f"{name} is {s['status']}, last active {age}. Result: {out[-3000:]}"

    @function_tool
    async def message_session(self, ctx: RunContext, name: str, message: str) -> str:
        """Send a new instruction into an existing saved agent session to continue its work.
        Use when the user wants to follow up on or add to an existing session.

        Args:
            name: the saved session name to continue.
            message: the new instruction to send into that session.
        """
        return await asyncio.to_thread(agent_tools.send_message, name, message)

    @function_tool
    async def delete_session(self, ctx: RunContext, name: str) -> str:
        """Delete a saved agent session by name (removes it and its working files).
        Use when the user wants to delete, remove, forget, or clean up a session.

        Args:
            name: the saved session name to delete.
        """
        return await asyncio.to_thread(agent_tools.delete_session, name)

    @function_tool
    async def control_tv(self, ctx: RunContext, action: str, level: int | None = None) -> str:
        """Control the living-room TV's volume and playback. Use for: turn up/down the TV,
        set volume, mute/unmute, pause, play/resume, stop, or "what's on the TV".

        Args:
            action: one of "volume_up", "volume_down", "set_volume", "mute", "unmute",
                    "play", "pause", "stop", "status".
            level: volume 0-100 (only used with set_volume).
        """
        return await asyncio.to_thread(tv_tools.control, action, level)

    @function_tool
    async def play_on_tv(self, ctx: RunContext, title: str, service: str | None = None) -> str:
        """Play a show, movie, or video on the TV. If a streaming app is named (Netflix,
        Prime, Disney+, Hulu, Max...) it plays it THERE specifically; otherwise it plays the
        top match on whatever app has it. For music, songs, specific YouTube videos, or ANY
        content by a YouTuber/creator/channel (video essays, documentaries by channels like
        Fern or Veritasium, reaction videos, podcasts), pass service="youtube" (casts the
        video directly — creator content is NOT in the TV's movie/show catalog).

        IMPORTANT: infer the ACTUAL title the user means and pass that — don't just repeat the
        raw speech-to-text. STT often mangles show/movie names (especially unusual ones), so
        use your knowledge to correct it: e.g. heard "tempting island" -> pass "Temptation
        Island"; "squid gaME" -> "Squid Game"; "the bear" stays "The Bear". Pass your best
        corrected guess as `title`.

        Args:
            title: the real title you believe the user means (corrected from any STT errors).
            service: "youtube" for music/videos; a streaming app name to force that app; else empty.
        """
        # The TV automation (search, type, navigate) takes ~20-35s; run it in the
        # background so Kronik can confirm instantly instead of going silent. Keep a
        # reference so the task isn't garbage-collected mid-run.
        session = ctx.session

        def _done(task):
            _BG_TASKS.discard(task)
            try:
                res = task.result()
            except Exception:
                return
            # the instant confirmation was optimistic — if the background cast failed,
            # SAY so instead of leaving the user staring at a TV that never starts
            if res and res.startswith(("Couldn't", "I couldn't")):
                try:
                    session.say(f"Hm, that didn't work after all — {res}")
                except Exception:
                    pass

        t = asyncio.create_task(asyncio.to_thread(tv_tools.play, title, service))
        _BG_TASKS.add(t)
        t.add_done_callback(_done)
        where = f" on {service}" if service else ""
        return f"Cool, starting '{title}'{where} on the TV — it'll come up in a few seconds."

    @function_tool
    async def thermostat_status(self, ctx: RunContext) -> str:
        """Get the Nest thermostat's current state: room temperature, humidity, target
        setpoint(s), mode, and whether it's actively heating or cooling. Use for questions
        like "what's the thermostat at" or "is the heat on"."""
        async with _lead(ctx, "Checking the thermostat."):
            return await asyncio.to_thread(nest_tools.status)

    @function_tool
    async def set_thermostat_temperature(self, ctx: RunContext, temperature: int,
                                         target: str | None = None) -> str:
        """Set the Nest thermostat's target temperature in Fahrenheit.

        Args:
            temperature: the target temperature in degrees Fahrenheit.
            target: only when the thermostat is in auto mode — "heat" or "cool" to say which
                    setpoint to change. Leave empty otherwise.
        """
        async with _lead(ctx, "One sec, setting the thermostat."):
            return await asyncio.to_thread(nest_tools.set_temperature, temperature, target)

    @function_tool
    async def sing_song(self, ctx: RunContext, lyrics: str, mood: str | None = None) -> str:
        """Perform an ORIGINAL song with lyrics YOU wrote, sung in your own voice. Use when
        the user asks you to make up a song, sing about something/someone specific, or
        improvise. WRITE the lyrics yourself first (40-55 simple singable words, plain
        language, no markdown) and pass them — then the singing plays automatically after
        about a minute of preparation. Each song gets its own melody.

        Args:
            lyrics: the full lyric text to sing (40-55 words, plain prose, no line numbers).
            mood: optional melody feel (upbeat, ballad, folk, pop, silly, lullaby) OR a
                named tune the user asks for (e.g. "viva la vida"); omit to let the
                system pick.
        """
        session = ctx.session
        voice = self._persona_voice
        tts_model = self._tts_model

        def _synth_ref(text, out_path):
            import numpy as _np
            import soundfile as _sf
            state = tts_model.get_state_for_audio_prompt(voice)
            audio = _np.concatenate([_np.asarray(a).reshape(-1)
                                     for a in tts_model.generate_audio_stream(state, text)])
            _sf.write(out_path, audio, 24000)

        async def _perform():
            log = logging.getLogger("kronik.sing")
            try:
                out_dir, proc = await asyncio.to_thread(
                    sing_tools.start_streaming_render, voice, lyrics, _synth_ref, mood)
            except Exception:
                log.exception("sing render failed to start")
                try:
                    session.say("Ah, my songwriting fell apart — sorry, Mister.",
                                add_to_chat_ctx=False)
                except Exception:
                    pass
                return
            if proc is None:                       # full cache hit
                await self._play_wav(session, os.path.join(out_dir, "generated.wav"),
                                     f"original:{lyrics[:30]}")
                return
            # stream: play each segment as it lands while later ones render
            await self._play_segments(session, out_dir, proc, f"original:{lyrics[:30]}")

        t = asyncio.create_task(_perform())
        _BG_TASKS.add(t)
        t.add_done_callback(_BG_TASKS.discard)
        return ("You are now writing the song and warming up your voice. Tell the user "
                "you're writing a song for them (in your own style, e.g. 'give me a moment, "
                "I'm writing your song') — it starts playing automatically in about half a "
                "minute. Do NOT speak the lyrics.")

    async def _play_segments(self, session, out_dir, proc, label):
        """Stream segment_NN.wav files into the room as the renderer produces them."""
        log = logging.getLogger("kronik.sing")
        try:
            from livekit import rtc
            import soundfile as sf
            idx = 0
            source = track = pub = None
            fb = frame = None
            said = False
            while True:
                seg = os.path.join(out_dir, f"segment_{idx:02d}.wav")
                if os.path.isfile(seg):
                    data, sr = await asyncio.to_thread(sf.read, seg)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    # attenuate only clipping-hot segments; never boost — per-segment
                    # boosting would make loudness jump between segments of one song
                    peak = float(np.abs(data).max()) or 1.0
                    if peak > 0.85:
                        data = data * (0.85 / peak)
                    if source is None:
                        await session.say("Okay, here goes.", add_to_chat_ctx=False)
                        said = True
                        source = rtc.AudioSource(sr, 1)
                        track = rtc.LocalAudioTrack.create_audio_track("singing", source)
                        pub = await self._room.local_participant.publish_track(
                            track, rtc.TrackPublishOptions(
                                source=rtc.TrackSource.SOURCE_MICROPHONE))
                        n = sr // 100
                        frame = rtc.AudioFrame.create(sr, 1, n)
                        fb = np.frombuffer(frame.data, dtype=np.int16)
                        log.info("performing %s (streaming segments)", label)
                    n = sr // 100
                    pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
                    for i in range(0, len(pcm), n):
                        c = pcm[i:i + n]
                        if len(c) < n:
                            c = np.pad(c, (0, n - len(c)))
                        fb[:] = c
                        await source.capture_frame(frame)
                    idx += 1
                    continue
                if os.path.isfile(os.path.join(out_dir, "ALL_DONE")):
                    break                           # seg was checked above: nothing left
                if proc.poll() is not None and proc.returncode != 0:
                    raise RuntimeError("renderer died")
                await asyncio.sleep(0.5)
            if source is not None:
                await source.wait_for_playout()
                await self._room.local_participant.unpublish_track(pub.sid)
                log.info("performance done (%d segments)", idx)
            elif not said:
                raise RuntimeError("no segments produced")
        except Exception:
            log.exception("streaming sing failed")
            try:
                session.say("Ah, the song fell apart mid-verse — sorry.",
                            add_to_chat_ctx=False)
            except Exception:
                pass
            try:
                proc.kill()
            except Exception:
                pass

    async def _play_wav(self, session, path, label):
        """Publish a dedicated track and stream a wav into the room (normalized)."""
        try:
            from livekit import rtc
            import soundfile as sf
            data, sr = await asyncio.to_thread(sf.read, path)
            if data.ndim > 1:
                data = data.mean(axis=1)
            peak = float(np.abs(data).max()) or 1.0
            data = data * (0.85 / peak)
            await session.say("Okay, here goes.", add_to_chat_ctx=False)
            source = rtc.AudioSource(sr, 1)
            track = rtc.LocalAudioTrack.create_audio_track("singing", source)
            pub = await self._room.local_participant.publish_track(
                track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
            n = sr // 100
            frame = rtc.AudioFrame.create(sr, 1, n)
            fb = np.frombuffer(frame.data, dtype=np.int16)
            pcm = (np.clip(data, -1, 1) * 32767).astype(np.int16)
            logging.getLogger("kronik.sing").info(
                "performing %s (%.0fs @ %dHz)", label, len(pcm) / sr, sr)
            for i in range(0, len(pcm), n):
                c = pcm[i:i + n]
                if len(c) < n:
                    c = np.pad(c, (0, n - len(c)))
                fb[:] = c
                await source.capture_frame(frame)
            await source.wait_for_playout()
            await self._room.local_participant.unpublish_track(pub.sid)
            logging.getLogger("kronik.sing").info("performance done")
        except Exception:
            logging.getLogger("kronik.sing").exception("sing playback failed")

    @function_tool
    async def feed_cat(self, ctx: RunContext, portions: int | None = None) -> str:
        """Dispense food from the automatic pet feeder. Use for "feed the cat", "feed the pet",
        "give a snack". Each portion is one feeder unit (~1/12 cup).

        Args:
            portions: how many portions (leave empty for a normal feeding = 2 portions; only pass
                      a number if the user explicitly asks for a different amount).
        """
        async with _lead(ctx, f"One sec, feeding {config.pet_name()}."):
            return await petlibro_tools.feed(portions or 2)

    @function_tool
    async def cat_feeder_status(self, ctx: RunContext) -> str:
        """Check the PetLibro cat feeder: online, the HOPPER food store, how many times it's fed
        today, battery, and desiccant. NOTE: this feeder has no bowl scale, so it can't measure how
        much food is IN the bowl — today's feeding count is the best proxy for that. Use for "is the
        feeder full", "has the cat been fed today", "check the feeder"."""
        async with _lead(ctx, "Checking the feeder."):
            return await petlibro_tools.status()

    @function_tool
    async def doordash_find_items(self, ctx: RunContext, item_query: str, intent: str,
                                  store_query: str | None = None,
                                  limit: int | None = None) -> str:
        """Find a requested food or drink across nearby stores. Prefer this for item requests: it
        infers plausible restaurant types when store_query is empty, then scans their menus.
        Preserve the user's requested food; never replace it with an example or different item.

        Args:
            item_query: the user's requested food, with the likely spelling corrected if needed.
            intent: the user's original DoorDash request verbatim.
            store_query: optional restaurant, cuisine, or seller hint. Leave empty for automatic
                         inference unless the user named a store or supplied a useful constraint.
                         A named store is matched strictly and never replaced with another merchant.
            limit: nearby stores to scan, 1-10; default 5.
        """
        async with _lead(ctx, "Searching DoorDash menus."):
            return await asyncio.to_thread(
                doordash_tools.find_items, item_query, intent, store_query or "", limit or 5)

    @function_tool
    async def doordash_search(self, ctx: RunContext, store_query: str, intent: str,
                              limit: int | None = None) -> str:
        """Find nearby restaurants by specialty, cuisine, or store name using the saved default
        address. Preserve distinctive food terms that identify a restaurant concept.

        Args:
            store_query: restaurant/store search likely to surface the requested sellers.
            intent: the user's original DoorDash request verbatim.
            limit: number of stores, 1-10; default 5.
        """
        async with _lead(ctx, "Checking DoorDash."):
            return await asyncio.to_thread(doordash_tools.search, store_query, intent, limit or 5)

    @function_tool
    async def doordash_default_address(self, ctx: RunContext, intent: str) -> str:
        """Get the canonical saved DoorDash delivery address. Use whenever the user asks what their
        delivery or default address is. Never guess an address or ask the user for it unless this
        tool reports that no default address exists.

        Args:
            intent: the user's original DoorDash request verbatim.
        """
        async with _lead(ctx, "Checking your saved DoorDash address."):
            return await asyncio.to_thread(doordash_tools.default_address, intent)

    @function_tool
    async def doordash_menu(self, ctx: RunContext, store_id: str, query: str, intent: str) -> str:
        """Find items inside one restaurant's menu. Use the store_id from doordash_search.

        Args:
            store_id: DoorDash store ID from search results.
            query: desired menu item or category, such as "matcha" or "chicken sandwich".
            intent: the user's original DoorDash request verbatim.
        """
        async with _lead(ctx, "Checking the menu."):
            return await asyncio.to_thread(doordash_tools.menu, store_id, query, intent)

    @function_tool
    async def doordash_item_details(self, ctx: RunContext, store_id: str, menu_id: str,
                                    item_id: str, intent: str) -> str:
        """Internal compatibility tool; modifier resolution normally happens inside add-to-cart.

        Args:
            store_id: store ID from doordash_search.
            menu_id: menu ID from doordash_menu.
            item_id: item ID from doordash_menu.
            intent: the user's original DoorDash request verbatim.
        """
        async with _lead(ctx, "Checking the item options."):
            return await asyncio.to_thread(
                doordash_tools.item_details, store_id, menu_id, item_id, intent)

    @function_tool
    async def doordash_cart(self, ctx: RunContext, intent: str,
                            cart_uuid: str | None = None, store_id: str | None = None) -> str:
        """Show one active DoorDash cart by cart_uuid, or list active carts when cart_uuid is empty.
        Optionally filter the list by store_id.

        Args:
            intent: the user's original DoorDash request verbatim.
            cart_uuid: active cart UUID to inspect; empty to list carts.
            store_id: optional store filter when listing carts.
        """
        async with _lead(ctx, "Checking the DoorDash cart."):
            return await asyncio.to_thread(
                doordash_tools.cart, intent, cart_uuid or "", store_id or "")

    @function_tool
    async def doordash_add_to_cart(self, ctx: RunContext, store_id: str, menu_id: str,
                                   item_id: str, item_name: str, quantity: int, intent: str,
                                   cart_uuid: str | None = None,
                                   fulfillment: str | None = None,
                                   configuration_request: str | None = None) -> str:
        """Add one restaurant item to a DoorDash cart. When the user has stated natural-language
        modifier preferences, call this directly with configuration_request containing their words;
        the tool infers quantities and distinct configurations, then fetches and resolves modifier
        IDs internally. Do not say you will check modifiers without making this tool call. An
        existing same-store cart is reused only if it has never been submitted.

        Args:
            store_id: store ID from doordash_search.
            menu_id: menu ID from doordash_menu.
            item_id: item ID from doordash_menu.
            item_name: exact item name from doordash_menu.
            quantity: best known total number of units, 1-20. The tool corrects this from the
                      original request when configuration_request is supplied.
            intent: the user's original DoorDash request verbatim.
            cart_uuid: existing cart to extend; empty to create one.
            fulfillment: "delivery" or "pickup" when creating a cart; default delivery.
            configuration_request: the user's natural-language quantity and modifier request. Pass
                                   their wording rather than constructing JSON or inventing option
                                   IDs. Leave empty only when the item has no modifiers.
        """
        async with _lead(ctx, "Updating the DoorDash cart."):
            return await asyncio.to_thread(
                doordash_tools.add_to_cart, store_id, menu_id, item_id, item_name, quantity,
                intent, cart_uuid or "", fulfillment or "delivery", "",
                "", configuration_request or "")

    @function_tool
    async def doordash_remove_from_cart(self, ctx: RunContext, cart_uuid: str,
                                        cart_item_id: str, intent: str) -> str:
        """Remove one line from a DoorDash cart. Use the cart-line id from doordash_cart, not the
        menu item ID."""
        async with _lead(ctx, "Updating the DoorDash cart."):
            return await asyncio.to_thread(
                doordash_tools.remove_from_cart, cart_uuid, cart_item_id, intent)

    @function_tool
    async def doordash_delete_cart(self, ctx: RunContext, cart_uuid: str, intent: str) -> str:
        """Delete and abandon an active DoorDash cart after the user asks to replace or delete it."""
        async with _lead(ctx, "Deleting the DoorDash cart."):
            return await asyncio.to_thread(doordash_tools.delete_cart, cart_uuid, intent)

    @function_tool
    async def doordash_preview_order(self, ctx: RunContext, cart_uuid: str, intent: str) -> str:
        """Get the canonical DoorDash price/fee/ETA preview plus the masked default payment method.
        This must be called after the cart is ready and before stating any total, tip, payment, ETA,
        or delivery address. Use only its exact values; never guess checkout facts. State total,
        suggested tip, and payment, then ask only whether the address is correct."""
        async with _lead(ctx, "Getting the DoorDash total."):
            return await asyncio.to_thread(
                doordash_tools.preview_order, cart_uuid, intent, False)

    @function_tool
    async def doordash_submit_order(self, ctx: RunContext, cart_uuid: str, tip_cents: int,
                                    confirmed: bool, intent: str) -> str:
        """Place and charge a DoorDash order. Call only after doordash_preview_order and after the
        user answers yes when asked whether the disclosed delivery address is correct. This polls
        DoorDash internally. Report success only when the result has order_successful=true and
        status=successful; every other status means the order is not confirmed. Never retry the
        same cart after a submit attempt. If payment needs attention, tell the user to open DoorDash;
        never ask for payment credentials and never read, spell, or include a URL in a voice reply.

        Args:
            cart_uuid: cart UUID from the cart tools.
            tip_cents: suggested tip from doordash_preview_order in cents; 500 means $5.00. Use a
                       different amount only when the user requests it; use 0 for pickup.
            confirmed: true only after that one explicit yes.
            intent: the user's original DoorDash request verbatim.
        """
        async with _lead(ctx, "Placing the DoorDash order."):
            return await asyncio.to_thread(
                doordash_tools.submit_order, cart_uuid, tip_cents, confirmed, intent, False)

    @function_tool
    async def doordash_order_status(self, ctx: RunContext, order_uuid: str, intent: str) -> str:
        """Check a prior DoorDash order when the user later asks for an update. Top-level success is
        true only when the order status itself is successful. Do not call immediately after submit,
        because doordash_submit_order already polls internally."""
        async with _lead(ctx, "Checking the DoorDash order."):
            return await asyncio.to_thread(doordash_tools.order_status, order_uuid, intent)

    @function_tool
    async def doordash_roulette_prepare(self, ctx: RunContext, intent: str,
                                        max_total_cents: int | None = None) -> str:
        """Prepare a random surprise based on recent reorderable restaurant deliveries. It skips
        broken stores and existing carts, uses the suggested tip, and enforces an all-in spending
        cap. Read its one confirmation question without revealing the food, then wait.

        Args:
            intent: the user's original surprise-me request verbatim.
            max_total_cents: optional all-in cap including tip; default $80 (8000 cents).
        """
        async with _lead(ctx, "Spinning DoorDash roulette."):
            return await asyncio.to_thread(
                doordash_tools.roulette_prepare, intent, max_total_cents or 0)

    @function_tool
    async def doordash_roulette_submit(self, ctx: RunContext, confirmed: bool, intent: str) -> str:
        """Submit the prepared DoorDash roulette order after the user clearly says yes to its one
        delivery-address question. This charges the saved payment method."""
        async with _lead(ctx, "Placing the surprise order."):
            return await asyncio.to_thread(
                doordash_tools.roulette_submit, confirmed, intent, False)

    @function_tool
    async def doordash_roulette_cancel(self, ctx: RunContext, intent: str) -> str:
        """Cancel and delete the prepared roulette cart when the user declines the surprise."""
        return await asyncio.to_thread(doordash_tools.roulette_cancel, intent)

    @function_tool
    async def tesla_status(self, ctx: RunContext) -> str:
        """Check the car: battery %, range, charging state, whether it's
        locked, and climate. Use for "how much charge does the car have", "is the car locked",
        "check on the car". Waking the car can take several seconds."""
        async with _lead(ctx, [f"Let me check {config.car_name()}.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.status()

    @function_tool
    async def tesla_climate(self, ctx: RunContext, on: bool, temperature: int | None = None) -> str:
        """Start or stop the Tesla's climate / pre-conditioning (warm up or cool down the cabin
        before driving).

        Args:
            on: true to start climate, false to stop it.
            temperature: optional target cabin temperature in Fahrenheit.
        """
        async with _lead(ctx, [f"One sec, adjusting {config.car_name()}'s climate.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.climate(on, temperature)

    @function_tool
    async def tesla_lock(self, ctx: RunContext, locked: bool) -> str:
        """Lock or unlock the Tesla.

        Args:
            locked: true to lock, false to unlock.
        """
        async with _lead(ctx, [f"One sec, reaching {config.car_name()}.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.lock(locked)

    @function_tool
    async def tesla_charging(self, ctx: RunContext, action: str, limit: int | None = None) -> str:
        """Control the Tesla's charging.

        Args:
            action: "start", "stop", or "limit".
            limit: with action "limit", the charge limit as a percent (50-100).
        """
        async with _lead(ctx, [f"One sec, reaching {config.car_name()}.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.charge(action, limit)

    @function_tool
    async def tesla_find(self, ctx: RunContext, mode: str | None = None) -> str:
        """Honk the horn or flash the lights to locate the Tesla in a parking lot.

        Args:
            mode: "honk" or "flash" (default flash).
        """
        async with _lead(ctx, [f"One sec, reaching {config.car_name()}.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.find(mode or "flash")

    @function_tool
    async def tesla_locate(self, ctx: RunContext) -> str:
        """Get where the Tesla is currently parked (its location)."""
        async with _lead(ctx, [f"Let me locate {config.car_name()}.", "Still working — the car might be asleep, waking it up."], interval=4.0):
            return await tesla_tools.locate()

    @function_tool
    async def check_camera(self, ctx: RunContext, camera: str | None = None) -> str:
        """Look at a Ring security camera RIGHT NOW: captures a fresh snapshot and describes what
        is visible (people, pets, notable activity, the scene). Use for "what's in the bedroom",
        "check the camera", "is anyone in the bedroom", "what does the camera see". Takes a few
        seconds (the camera has to wake and capture).

        Args:
            camera: which camera by name (e.g. "bedroom"). Leave empty for the default camera.
        """
        _cam = f"the {camera} camera" if camera else "the camera"
        async with _lead(ctx, f"Let me pull up {_cam}."):
            desc, cam = await ring_tools.look(camera or "")
        if not desc:
            return f"I couldn't get an image from the {camera or 'camera'} just now."
        return f"The {cam} camera shows: {desc}"

    @function_tool
    async def camera_recent_activity(self, ctx: RunContext, camera: str | None = None,
                                     hours: int | None = None) -> str:
        """Check a Ring camera's recent motion/activity history — "any motion in the bedroom",
        "was there activity in the last hour". Reports what kind of events and how long ago.

        Args:
            camera: camera name (e.g. "bedroom"). Empty = default camera.
            hours: optionally only report events within the last N hours.
        """
        async with _lead(ctx, "One sec, checking recent activity."):
            evs, cam = await ring_tools.recent_events(camera or "", limit=10)
        if not evs:
            return f"No recent activity on the {cam or camera or 'camera'}."
        now = time.time()
        cutoff = now - hours * 3600 if hours else 0
        lines = []
        for e in evs:
            t = e.get("time")
            if t is None or (cutoff and t < cutoff):
                continue
            lines.append(f"{e.get('kind', 'event')} {_fmt_age(now - t)}")
        if not lines:
            span = f"the last {hours} hour(s)" if hours else "recently"
            return f"No activity on the {cam} camera in {span}."
        return f"{cam} camera activity: " + "; ".join(lines) + "."

    @function_tool
    async def watch_camera_live(self, ctx: RunContext, camera: str | None = None,
                                minutes: float | None = None) -> str:
        """Start live Ring video commentary. Call this whenever the user asks to watch, monitor,
        or narrate a camera. It runs in the background until its duration ends or
        stop_camera_watch is called.

        Args:
            camera: which camera by name (e.g. "bedroom"). Empty = bedroom.
            minutes: how long to watch, from 0.5 to 10 minutes. Default = 1 minute.
        """
        _, message = await self._mage_ring.start(
            ctx.session, camera or "bedroom", minutes if minutes is not None else 1.0,
        )
        return message

    @function_tool
    async def stop_camera_watch(self, ctx: RunContext) -> str:
        """Stop the active Mage-VL live camera commentary."""
        if await self._mage_ring.stop():
            return "Stopped the live camera commentary."
        return "No live camera commentary is running."

    @function_tool
    async def set_thermostat_mode(self, ctx: RunContext, mode: str) -> str:
        """Switch the Nest thermostat's mode.

        Args:
            mode: "heat", "cool", "auto" (heat/cool automatically), or "off".
        """
        async with _lead(ctx, "One sec, setting the thermostat."):
            return await asyncio.to_thread(nest_tools.set_mode, mode)

    @function_tool
    async def roomba_status(self, ctx: RunContext) -> str:
        """Get the Roomba vacuum's current status: what it's doing (cleaning, charging,
        docked, paused), battery level, and whether the bin is full."""
        return await asyncio.to_thread(roomba_tools.status)

    @function_tool
    async def control_roomba(self, ctx: RunContext, action: str) -> str:
        """Control the Roomba vacuum.

        Args:
            action: "clean" (start vacuuming), "stop", "pause", "resume", "dock" (send it
                    back to its charging base), or "find" (make it beep to locate it).
        """
        # connecting can take a few seconds on first use; run in the background and
        # confirm instantly so Kronik doesn't go quiet.
        t = asyncio.create_task(asyncio.to_thread(roomba_tools.control, action))
        _BG_TASKS.add(t)
        t.add_done_callback(_BG_TASKS.discard)
        a = (action or "").lower().strip().replace(" ", "_")
        return {
            "clean": "Bet — sending the vacuum off to clean.", "start": "Bet — sending the vacuum off to clean.",
            "stop": "Stopping the vacuum.", "pause": "Paused it.", "resume": "Back to cleaning.",
            "dock": "Sending it home to the dock.", "home": "Sending it home to the dock.",
            "return": "Sending it home to the dock.", "find": "Pinging it so you can find it.",
        }.get(a, f"On it — {action}.")

    @function_tool
    async def lights_status(self, ctx: RunContext, which: str | None = None) -> str:
        """Check the smart lights: whether each is on, its brightness, and its current
        color/white-tone or scene.

        Args:
            which: a light or group name ("bedside", "bedside 1", "living room");
                   leave empty for all lights.
        """
        return await asyncio.to_thread(wiz_tools.status, which)

    @function_tool
    async def control_lights(self, ctx: RunContext, action: str,
                             brightness: int | None = None,
                             color: str | None = None,
                             which: str | None = None) -> str:
        """Control the smart lights (all of them by default).

        Args:
            action: "on", "off", "brightness" (needs brightness), "color" (needs color),
                    "identify" (blink so the user can tell which light is which),
                    "pulse_on" or "pulse_off" (the lights pulsing along with my voice).
            brightness: 1-100 percent (with action "brightness", or alongside on/color).
            color: a color name ("red", "blue", "pink"...), a white tone ("warm",
                   "neutral", "cool", "daylight"), or a mood scene ("nightlight", "cozy",
                   "candlelight", "relax", "fireplace", "bedtime", "romance"...).
            which: a light or group name ("bedside", "bedside 2", "living room");
                   leave empty for all lights.
        """
        a = (action or "").lower().strip()
        if a in ("pulse_on", "pulse_off"):
            light_fx.driver.enabled = a == "pulse_on"
            if not light_fx.driver.enabled:
                light_fx.driver.abort()
            return ("Bet — the lights will vibe with my voice now."
                    if light_fx.driver.enabled else "Okay, lights will sit still while I talk.")
        # Stop any live voice-pulse FIRST so it can't push RGB over the change we're about to
        # make (a stray pulse color would otherwise get captured as the next baseline and the
        # lamp would stick on it). Let in-flight fire-and-forget pushes land, THEN apply.
        light_fx.driver.suspend()
        await asyncio.sleep(0.06)
        result = await asyncio.to_thread(wiz_tools.control, action, brightness, color, which)
        # Forget the old baseline so the confirmation speech re-snapshots the SETTLED new state
        # (not the pre-change value, and not a pulse transient).
        light_fx.driver.invalidate_baseline()
        return result


server = AgentServer(
    setup_fnc=setup,             # preload models per idle process
    num_idle_processes=2,        # keep warm processes ready -> reliable, instant join
    initialize_process_timeout=120,  # Kokoro+Whisper warmup is slow; don't kill it early
)


@server.rtc_session()  # no agent_name -> auto-dispatch to every room a participant joins
async def entry(ctx: agents.JobContext):
    # Persona is chosen in the UI and passed as a participant attribute on the token.
    # Resolve it before building the session so we pick the matching prewarmed voice.
    await ctx.connect()
    # moshi* rooms belong to the experimental full-duplex Moshi POC agent (~/moshi-poc);
    # this worker auto-dispatches to EVERY room, so bail out to avoid two agents answering.
    if (ctx.room.name or "").startswith(("moshi", "gemma", "voicechat")):
        ctx.shutdown(reason="experimental room — handled by its own agent")
        return
    # Cold-start guard: the mlx LLM server (:8081) ramps its Metal context DOWN between calls,
    # so the FIRST turn of a fresh session otherwise pays ~3.7s cold TTFT (tok/s ~4, measured).
    # Fire one warm ping the instant we connect — fire-and-forget so it never delays the join —
    # so it re-ramps during participant-wait + greeting and the first real inference lands warm.
    _cw = asyncio.create_task(asyncio.to_thread(_gpu_warm_blocking))
    _BG_TASKS.add(_cw)
    _cw.add_done_callback(_BG_TASKS.discard)
    participant = await ctx.wait_for_participant()
    attrs = participant.attributes or {}
    persona = personas.resolve(attrs.get("persona"))
    ambient = str(attrs.get("ambient", "")).lower() in ("1", "true", "on", "yes")
    tts_by_voice = ctx.proc.userdata["tts_by_voice"]
    tts = tts_by_voice.get(persona["voice"])
    if tts is None:
        # Persona created after the worker started (its voice wasn't prewarmed). Build it now —
        # shares the loaded model, so it's just the ~1-2s voice-state load, not a full reload —
        # and cache it so later sessions on this voice are instant.
        try:
            tts = PocketTTS(voice=persona["voice"], model=ctx.proc.userdata["tts_model"])
            tts_by_voice[persona["voice"]] = tts
        except Exception:
            _audio_log.exception("failed to build voice %r; using default", persona["voice"])
            tts = next(iter(tts_by_voice.values()))
    # the light-pulse hue range must match THIS persona's voice pitch, not the env default
    light_fx.driver.set_voice(persona["voice"])
    # bias Whisper's wake-word spelling toward THIS persona's name (Kronik/Consuela)
    stt = ctx.proc.userdata["stt"]
    if hasattr(stt, "set_persona"):
        stt.set_persona(persona["label"])
    # Ambient speaker gate: if any voices are enrolled, only they can wake/talk to the
    # agent (unenrolled voices are dropped at STT). Off in normal mode or if nobody's
    # enrolled. The STT instance is reused across jobs, so set/clear every session.
    if hasattr(stt, "set_speaker_gate"):
        gate = None
        if ambient:
            try:
                import speaker_id
                gate = speaker_id.load_signatures() or None
                if gate:
                    _audio_log.info("ambient speaker gate ON — enrolled: %s", list(gate.keys()))
                else:
                    _audio_log.info("ambient speaker gate OFF — no voices enrolled (responds to anyone)")
            except Exception:
                gate = None
        stt.set_speaker_gate(gate)

    # Named so the agent_state handler can gate its audio during thinking/speaking
    # (SoulX inference then is pure GPU contention against LLM prefill + TTS).
    _turn_detector = (_soulx_detector() if os.environ.get("TURN_DETECTOR") == "soulx"
                      else EnglishModel(unlikely_threshold=0.15))

    session = AgentSession(
        stt=ctx.proc.userdata["stt"],   # reuse the prewarmed models
        llm=openai.LLM(
            model=_voice_llm()[0],
            base_url=_voice_llm()[1],
            api_key=os.environ["OPENAI_API_KEY"],
            # Hybrid-thinking models (e.g. qwen3.5 on Ollama) MUST have reasoning disabled
            # for voice — thinking adds 10-30s before the first spoken token. Env-gated
            # because some OpenAI-compatible servers reject the field outright.
            reasoning_effort=os.environ.get("LLM_REASONING_EFFORT") or None,
            # 0.4, not 0.7: on the 4-bit MoE the tool-vs-narrate decision is knife-edge, and
            # 0.7 sampling flips it ~1/3 of the time so device commands get NARRATED instead of
            # executed (measured 66% tool-call rate at 0.7 vs 100% at <=0.4). 0.4 keeps some
            # conversational variety while staying below that reliability cliff.
            temperature=0.4,
            # disable Qwen3.6 reasoning for low voice latency (voice-only;
            # the shared LLM server keeps thinking on for other clients)
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            # Emit legacy (anyOf) tool schemas, NOT strict ones. Strict serializes an
            # optional param `x: T | None` as {"type": ["T","null"]} — a type LIST that
            # mlx_lm's qwen3_coder tool parser doesn't recognize, so it falls back to
            # ast.literal_eval() on the value and crashes on any bare string (e.g. a
            # service name like "Netflix"), silently dropping the whole tool call ->
            # the agent goes mute. The anyOf form parses cleanly. (Optional params must
            # still be typed `T | None` so pydantic accepts the model's explicit null.)
            _strict_tool_schema=False,
        ),
        tts=tts,                        # persona-selected voice (prewarmed)
        vad=ctx.proc.userdata["vad"],
        # Text-based on-device turn detector: correctly flags COMPLETE commands as done ->
        # they commit at min_endpointing_delay (~0.2s) instead of the fallback's blanket wait.
        # It still waits (up to max) when you genuinely trail off mid-sentence.
        # NOTE: tried the audio-based v1-mini (2026-07-09) — on THIS pipeline it scored every
        # turn-end P<0.36 (below its threshold), so every turn pinned at max_endpointing_delay
        # (~900ms) vs EnglishModel's ~520ms early commits. Rolled back. See voice-latency memory.
        # unlikely_threshold: the calibrated default (0.0289) is hairline — fragments like
        # "Yeah. So." (P=0.054) cleared it and committed on the FAST path, clipping the user
        # mid-thought. At 0.15, anything the model scores <15% likely-finished waits the full
        # max_endpointing_delay instead; real commands (P 0.27-0.73) still commit fast.
        # TURN_DETECTOR=soulx (POC): SoulX-Duplug streaming full-duplex turn model — decides
        # end-of-turn from audio+semantics (~250ms) and holds on genuinely incomplete
        # utterances. Needs the server: cd ~/soulx-poc && ./.venv/bin/uvicorn server:app
        # --host 127.0.0.1 --port 8765. Rollback: unset TURN_DETECTOR (EnglishModel default).
        turn_detection=_turn_detector,
        min_endpointing_delay=0.2,
        # was 1.5; lowered to shave the dominant EOU wait (~772ms median). The turn
        # detector still commits complete commands at min_endpointing_delay (~0.2s);
        # this only caps how long it waits when you trail off mid-sentence. Kept >= the
        # ~250ms STT floor so transcription still hides under the wait.
        max_endpointing_delay=0.9,
    )
    # ---- per-turn latency trace (EOU/STT/LLM/TTS) -> ~/voice-agent/latency.log ----
    from livekit.agents.metrics import EOUMetrics, STTMetrics, LLMMetrics, TTSMetrics

    def _latlog(line):
        try:
            with open(os.path.join(config.REPO_ROOT, "latency.log"), "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        except Exception:
            pass

    _latlog(f"===== SESSION START  stt={type(ctx.proc.userdata['stt']).__name__} =====")

    @session.on("metrics_collected")
    def _on_metrics(ev):
        m = ev.metrics
        ms = lambda k: (f"{getattr(m, k)*1000:.0f}ms" if isinstance(getattr(m, k, None), (int, float)) else "?")
        if isinstance(m, EOUMetrics):
            # eou_delay = silence-end -> turn-committed (endpointing wait). biggest user-felt lever.
            _latlog(f"EOU  eou_delay={ms('end_of_utterance_delay')}  transcription_delay={ms('transcription_delay')}")
        elif isinstance(m, STTMetrics):
            _latlog(f"STT  duration={ms('duration')}")
        elif isinstance(m, LLMMetrics):
            tps = getattr(m, "tokens_per_second", None)
            _latlog(f"LLM  ttft={ms('ttft')}  total={ms('duration')}"
                    + (f"  tok/s={tps:.0f}" if isinstance(tps, (int, float)) else ""))
        elif isinstance(m, TTSMetrics):
            _latlog(f"TTS  ttfb={ms('ttfb')}  total={ms('duration')}")

    # single Assistant instance so the prompt-based EOT hold state (timer/counter) is
    # shared across turns and reachable from the agent_state hook below.
    _caps = set(config.enabled_integrations())
    if sing_tools.available():
        _caps.add("sing")
    instructions = personas.build_voice_instructions(persona, _caps, ambient=ambient)
    assistant = Assistant(instructions=instructions,
                          ambient=ambient, wake_words=persona["wake_words"],
                          greetings=personas.all_greetings(persona))
    assistant._persona_voice = persona["voice"]
    assistant._tts_model = ctx.proc.userdata.get("tts_model")
    assistant._room = ctx.room

    async def _teardown_mage_ring():
        await assistant._mage_ring.stop()
    ctx.add_shutdown_callback(_teardown_mage_ring)

    # Gate tools by the enabled integrations. LiveKit auto-discovers EVERY @function_tool method,
    # so narrow them here to the always-on session/turn-taking tools + the enabled integrations'
    # tools (out of the box that's only the OpenCode session tools).
    _enabled_tools = config.enabled_tool_names()
    if sing_tools.available():
        _enabled_tools |= {"sing_song"}       # needs the local SoulX-Singer checkout
    await assistant.update_tools([t for t in assistant.tools if t.info.name in _enabled_tools])

    # ---- silent-bot forensics: VAD-level user state to latency.log. If the bot "stops
    # hearing", this distinguishes (a) no USER-speaking events = audio never tripped VAD
    # (phone/mic/upstream) from (b) events but STT-DROP lines in agent.log (gate too hot)
    # from (c) events and transcripts but no reply (pipeline bug). ----
    _user_speaking = {"v": False}

    # ---- frequent-command pills: a voice turn that runs tools counts, same as chat. ----
    # The final STT transcript is the command text; the first tool batch of the turn
    # records it (a turn can chain several batches — later ones would double-count).
    _pending_cmd = {"text": None}

    @session.on("user_input_transcribed")
    def _on_cmd_transcript(ev):
        if getattr(ev, "is_final", False) and (ev.transcript or "").strip():
            _pending_cmd["text"] = ev.transcript.strip()

    @session.on("function_tools_executed")
    def _on_cmd_tools(ev):
        text, _pending_cmd["text"] = _pending_cmd["text"], None
        names = [fc.name for fc in getattr(ev, "function_calls", [])]
        if text and names:
            # off the event loop — record() does file IO under a cross-process lock
            _t = asyncio.create_task(asyncio.to_thread(commands_freq.record, text, names))
            _BG_TASKS.add(_t)
            _t.add_done_callback(_BG_TASKS.discard)

    @session.on("user_state_changed")
    def _on_user_state(ev):
        was = _user_speaking["v"]
        _user_speaking["v"] = getattr(ev, "new_state", "") == "speaking"
        # Warm the GPU context ONCE at speech START: it heats during the utterance (commit is
        # >=1s away, so no collision with the real request) and is hot when the turn ends.
        # Skipping pings entirely while speaking let the context ramp DOWN mid-utterance
        # (measured: TTFT 650->850 regression); pinging periodically risked a commit collision.
        if _user_speaking["v"] and not was:
            _t = asyncio.create_task(asyncio.to_thread(_gpu_warm_blocking))
            _BG_TASKS.add(_t)
            _t.add_done_callback(_BG_TASKS.discard)
        _latlog(f"USER {getattr(ev, 'old_state', '?')}->{getattr(ev, 'new_state', '?')}")

    # ---- GPU keep-warm: ping the LLM server only while idle-listening so the
    # server's Metal context stays ramped (saves ~140ms of first-token idle-ramp
    # per turn) WITHOUT adding GPU contention during actual turn processing. ----
    _agent_state = {"v": "initializing"}

    @session.on("agent_state_changed")
    def _on_agent_state(ev):
        prev = _agent_state["v"]
        st = getattr(ev, "new_state", prev)
        _agent_state["v"] = st
        # Gate SoulX audio while the agent thinks/speaks (frees the Metal GPU for
        # LLM prefill + TTS); reopen the moment we're listening again. SOULX_GATE=0 disables.
        if hasattr(_turn_detector, "set_gated") and os.environ.get("SOULX_GATE", "1") != "0":
            _turn_detector.set_gated(st in ("thinking", "speaking"))
        # The agent actually spoke -> the hold streak is over; clear any pending
        # timeout-answer and reset the consecutive-hold cap.
        if st == "speaking":
            assistant._reset_holds()
        # Playback over (finished or interrupted) -> stop the light pulse and do the ONE
        # restore for this speaking period. tts_node's per-sentence begin/end drive the
        # pulses but never restore; only this terminal transition does, so no stray push
        # can land after the restore (that was leaving lamps stuck bright/colored).
        elif prev == "speaking":
            light_fx.driver.end_speaking()
            # ambient: reopen the follow-up window after each response so the user can
            # continue without repeating the wake word.
            if assistant._ambient:
                assistant._ambient_active_until = time.monotonic() + _WAKE_WINDOW

    async def _keep_warm():
        # Ping while listening INCLUDING during user speech. Measured A/B (2026-07-21):
        # skipping pings while the user talks let the Metal context ramp down mid-utterance
        # (TTFT 650->850 across the board); a single speech-start ping fixed short turns
        # (TTFT 475-620) but long 3s+ utterances cooled again (spikes 925-1197). Continuous
        # pinging gave consistent 589-709 — the rare in-flight collision at commit (~130ms)
        # is far cheaper than the ramp-down. Speech-start ping (above) keeps the fast lows.
        while True:
            await asyncio.sleep(1.5)
            if _agent_state["v"] == "listening":
                try:
                    await asyncio.to_thread(_gpu_warm_blocking)
                except Exception:
                    pass

    _kw = asyncio.create_task(_keep_warm())
    _BG_TASKS.add(_kw)
    _kw.add_done_callback(_BG_TASKS.discard)

    await session.start(
        room=ctx.room,
        agent=assistant,
        # self-hosted DTLN denoiser cleans incoming audio before VAD/STT (local ONNX, no cloud),
        # with adaptive far-field makeup gain in front of it (see _FarFieldGain)
        room_input_options=RoomInputOptions(
            noise_cancellation=_FarFieldGain(dtln.noise_suppression(strength=0.6)),
        ),
    )
    # Canned greeting via TTS (no LLM call): instant, and avoids the Qwen3.6 chat
    # template's system-only-message guard. Random pick keeps it fresh; the LLM
    # handles the actual conversation.
    # In ambient mode the agent stays silent until addressed by name — no greeting.
    # The watch drops the session when idle and reconnects on every wake (battery saving), so a
    # greeting on each connect would just be repetitive noise — skip it for the watch room.
    is_watch = (ctx.room.name or "") == "watch"
    if not ambient and not is_watch:
        await session.say(personas.pick_greeting(persona))

    # Real-time Ring motion/ding -> fresh snapshot + Gemma description -> announce, but ONLY while
    # this session is live (the handler is registered here and torn down on shutdown, so it's
    # silent when no one's connected). ring_events is a general hook system — register more handlers
    # (phone push, logging, automations) later without touching the agent. Each handler receives the
    # raw snapshot bytes AND the description.
    if config.is_enabled("ring") and os.environ.get("RING_TOKEN_JSON"):
        async def _announce_motion(event):
            # Mage is already consuming and describing the live stream. Suppress
            # the separate snapshot-based motion alert so the user does not hear
            # a delayed/duplicate "Heads up" and mistake it for Mage commentary.
            if assistant._mage_ring.running:
                return
            # A motion heads-up only needs the gist — are there people/animals? — so use a short,
            # cheap describe (~1s), NOT the full detailed one the `look` tool uses (~8s).
            desc = await event.describe(
                "Look at this security camera image and, in one or two short sentences, say "
                "whether any people or animals are present and what they're doing. If none are "
                "present, just say the area looks empty and quiet and briefly note the setting.",
                max_tokens=128,
            )
            if not desc:                    # no fresh snapshot (debounced/failed) -> don't spam
                return
            where = (f"motion at the {event.camera} camera" if event.kind == "motion"
                     else f"someone at the {event.camera} doorbell")
            try:
                # The gist is already brief and spoken-ready — announce it directly, no LLM summary.
                await session.say(f"Heads up — {where}. {desc}")
            except Exception:
                pass
        ring_events.register(_announce_motion)
        asyncio.create_task(ring_events.start())     # non-blocking (firebase register ~seconds)

        async def _teardown_ring_events():
            ring_events.unregister(_announce_motion)
            if ring_events.handler_count() == 0:
                await ring_events.stop()
        ctx.add_shutdown_callback(_teardown_ring_events)

    # ---- Session-completion notification: poll the DB for finished sessions that haven't
    # been announced yet (announced_at IS NULL) and whose completion happened AFTER this
    # call started.  Announce via session.say() once, then mark confirmed.  Mirrors the
    # ring_events pattern: registered here, torn down on shutdown. ----
    call_start = time.time()   # only announce completions that happened during this call

    async def _poll_completions():
        """Periodically check for sessions that finished while this call was live."""
        poll_interval = 5.0       # seconds between polls
        while True:
            await asyncio.sleep(poll_interval)
            try:
                rows = db.list_recent(limit=20)
                for r in rows:
                    if r["status"] not in ("done", "error"):
                        continue
                    if r.get("announced_at"):
                        continue
                    # Only announce completions that happened AFTER this call started
                    updated = r.get("updated_at") or 0
                    if updated < call_start:
                        continue
                    # Distill a SHORT spoken summary via the LLM from the END of the output (the
                    # agent's conclusion), NOT the opening "Let me start by…" narration. Using
                    # generate_reply (rather than say(raw_text)) means the model produces one clean
                    # sentence AND its reply — not the verbose transcript — is what lands in the chat
                    # context, so a follow-up ("what did it find?") has sensible memory.
                    out = (r.get("last_output") or "").strip()
                    tail = out[-3000:] if len(out) > 3000 else out   # focus on the result; keep prompt small
                    status_label = "finished" if r["status"] == "done" else "hit an error"
                    try:
                        await session.generate_reply(
                            instructions=(
                                f"A background task you dispatched earlier, named '{r['name']}', just "
                                f"{status_label}. In ONE short spoken sentence, tell the user it's done "
                                f"and the key result. Summarize from this output — do NOT read it "
                                f"verbatim:\n\n{tail}"
                            ),
                            tool_choice="none",
                        )
                    except Exception:
                        pass
                    # Mark as announced so we never re-announce
                    db.mark_announced(r["name"])
            except Exception:
                pass  # polling must never take down the voice loop

    _completion_task = asyncio.create_task(_poll_completions())
    _BG_TASKS.add(_completion_task)
    _completion_task.add_done_callback(_BG_TASKS.discard)

    async def _teardown_completions():
        _completion_task.cancel()
    ctx.add_shutdown_callback(_teardown_completions)


if __name__ == "__main__":
    agents.cli.run_app(server)
