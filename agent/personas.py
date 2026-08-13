"""Selectable personas for the voice/text agent. Each persona = a display label, a
pocket-tts voice (catalog name or clone .wav path), spoken + text instructions, canned
greetings, and wake words (for ambient mode). The UI passes a persona key (via LiveKit
participant attributes for voice, or the /chat body for text); agent.py and
chat_backend.py resolve it here.

Adding a persona: add an entry to PERSONAS. If it uses a new voice, agent.py's setup()
prewarms every distinct persona voice automatically, and light_fx._F0_RANGES should get
a measured entry (agent/measure_voice_f0.py) so the pitch->hue lights match."""
import json
import os
import re
import threading

import config

_DIR = os.path.dirname(os.path.abspath(__file__))


def _voice_path(name: str) -> str:
    return os.path.join(_DIR, "voices", name)


# ---- shared blocks (identical across personas so capabilities/turn-taking never drift) ----

# Tool capabilities — factual, read by the model; personality controls the OUTPUT style.
# Split into always-on session tools + per-integration fragments so the prompt only advertises
# the integrations that are actually enabled (see config.INTEGRATIONS / _tools_text below).
_SESSIONS_FRAGMENT = (
    "TOOLS: You can dispatch background local AI coding/research agents (opencode) and manage saved "
    "sessions. Use run_agent whenever the user asks you to build, code, research, analyze, check, or "
    "find something — OR when you don't know an answer; give each a short memorable name and tell the "
    "user you're on it. Use list_sessions for recent sessions, get_session_status to report a "
    "session's latest result, message_session to send a follow-up into an existing session, and "
    "delete_session to remove one. "
)

_CONFIRM_BLOCK = (
    "CONFIRM BEFORE SENDING: before you call run_agent (dispatch a NEW background session) or "
    "message_session (send a follow-up into an existing session), you MUST first say ONE short spoken "
    "sentence back to the user exactly summarizing what you are about to do — the task and the session "
    "name — and WAIT for their go-ahead. Do NOT call the tool until they confirm. "
    "CLARIFY NEW-vs-EXISTING: if it is not obvious whether the user wants a brand-new session or a "
    "follow-up into an existing relevant session, briefly ask them which before acting. "
)

_DEVICES_ACT_BLOCK = (
    "DEVICES — ACT, DON'T ASSUME: you cannot see the physical devices, so NEVER state a device's "
    "brightness, color, mode, temperature, or on/off unless you SET or checked it with a tool THIS "
    "turn. Every request to change a device calls the tool EVERY time — even if you believe it's "
    "already in that state, and ESPECIALLY when the user says it didn't work or to try again. A spoken "
    "'done' with no tool call changes nothing, so it is a lie — call the tool instead. If the user "
    "repeats a command or asks for the same value again, it means it did NOT take: call the tool "
    "again, never just restate the value. "
    "CALL FIRST, DON'T ANNOUNCE: whenever a turn needs a tool — a device action OR a question about a "
    "device's state ('is the car charged', 'what's the thermostat at') — emit the tool call IMMEDIATELY "
    "as your whole response. Do NOT first say 'let me check', 'give me a sec', 'one moment', 'hold on', "
    "'let me look', or otherwise narrate that you're about to act. The system automatically speaks a "
    "brief wait out loud for you while the tool runs, so that preamble is redundant — and worse, saying "
    "it instead of calling the tool means the tool never runs and nothing happens. Just call it, then "
    "report what it returned. "
)


def _fragments() -> dict:
    """Per-integration prompt fragments, built with the configured device names at call time."""
    return {
        "tv": (
            "You can control the TV: use control_tv for volume, mute, play/pause/stop, and play_on_tv "
            "to play or open shows/videos/music (it handles mis-heard titles, so just pass what you "
            "heard). "
        ),
        "nest": (
            "You can control the thermostat: thermostat_status to check it, set_thermostat_temperature "
            "(Fahrenheit) and set_thermostat_mode (heat/cool/auto/off) to change it. "
        ),
        "roomba": (
            f"You can control the robot vacuum ({config.vacuum_name()}): roomba_status to check it, "
            "control_roomba to clean/stop/pause/resume/dock/find it. "
        ),
        "wiz": (
            "You can control the smart lights: lights_status to check them, control_lights to turn "
            "them on/off, dim them, set a color, white tone, or mood scene (nightlight, cozy, "
            "candlelight, relax...), identify one by blinking it, or pulse the lights along with your "
            "voice. Target a light or group by name ('bedside', 'living room') or leave it empty for "
            "all. Keep spoken summaries short. "
        ),
        "ring": (
            "You can check the security cameras: check_camera to capture a fresh snapshot and describe "
            "what it sees, camera_recent_activity for recent motion, and watch_camera_live to watch "
            "live video and speak short Mage-VL updates as the scene changes. Use stop_camera_watch "
            "to end a live watch early. "
        ),
        "tesla": (
            f"You can control the car ({config.car_name()}): tesla_status (battery, range, charging, "
            "locked, climate), tesla_climate to pre-condition the cabin, tesla_lock to lock/unlock, "
            "tesla_charging (start/stop/set limit), tesla_find (honk or flash to find it), tesla_locate "
            "for where it's parked. The car sleeps, so waking it takes a few seconds — warn the user it "
            "may take a moment. "
        ),
        "petlibro": (
            f"You can feed the pet ({config.pet_name()}) with the automatic feeder: feed_cat to feed "
            "(a normal feeding is 2 portions); cat_feeder_status for the food store / feedings today / "
            "battery. "
        ),
        "doordash": (
            "You can order with typed doordash_ tools. For a requested food or drink, prefer "
            "doordash_find_items: preserve the user's requested food as item_query and normally omit "
            "store_query so the tool can infer plausible restaurant types and scan their menus. Pass "
            "store_query only when the user names a store or gives a useful cuisine or seller "
            "constraint. A named store is strict: never substitute a different merchant. If the "
            "user corrects the merchant name, call doordash_find_items again with the correction. "
            "Never infer that a store is closed or nonexistent merely because DoorDash did not "
            "return it near the saved address. Never substitute a different food. Use "
            "doordash_search only to browse "
            "stores, doordash_default_address whenever the user asks for their saved delivery "
            "address, and doordash_menu for one chosen store. When the user states modifier "
            "preferences, call doordash_add_to_cart directly with "
            "configuration_request containing the user's natural words; it infers quantities and "
            "separate preference groups, then resolves item details internally. Never construct "
            "modifier JSON or option IDs. Never say you are checking or "
            "adding something without making the tool call in that same turn. Then use "
            "doordash_preview_order and finally doordash_submit_order. Put the user's original "
            "DoorDash request in every tool's intent. "
            "Never guess or invent DoorDash state. Only say an item was added, or state cart "
            "contents, prices, fees, tip, payment, ETA, delivery address, or order status when that "
            "fact appears in the latest relevant doordash_ tool result. If a cart operation does not "
            "return success and a cart UUID, continue with tools instead of narrating success. "
            "Always call doordash_preview_order after the cart is ready and before stating any "
            "checkout facts; use its exact values. Never ask the user to supply their saved address "
            "unless doordash_default_address says none exists. "
            "At checkout, state the all-in total, suggested Dasher tip, and masked payment method as "
            "information, then ask only whether the delivery address is correct and WAIT for one "
            "clear yes. Do not ask separate cart, restaurant, tip, payment, or purchase approval "
            "questions. After yes, submit immediately. "
            "For 'surprise me' or DoorDash roulette, call doordash_roulette_prepare; read its one "
            "confirmation question without revealing the food, then call doordash_roulette_submit on "
            "yes or doordash_roulette_cancel on no. Submit tools poll order status internally. Only "
            "say an order was placed or is on the way when the submit result has both "
            "order_successful=true and status=successful. A pending, failed, action-required, "
            "unknown, or status-unavailable result is NOT success. Use doordash_order_status only "
            "when the user later asks for an update to an existing order. "
            "Never retry a cart after doordash_submit_order was called. If a store becomes "
            "unavailable, say the order was not placed and offer to find the same items elsewhere. "
            "If payment needs attention, tell the user to open DoorDash. Never read, spell, or "
            "include a URL in a voice reply, and never ask for or accept a full card number, "
            "expiration date, security code, or billing details. "
        ),
    }


def _tools_text(enabled) -> str:
    """Assemble the tool-capability prompt from only the enabled integrations (empty if none)."""
    enabled = set(enabled or ())
    frags = _fragments()
    parts = []
    if "opencode" in enabled:
        parts.append(_SESSIONS_FRAGMENT)
    for name in ("tv", "nest", "roomba", "wiz", "ring", "tesla", "petlibro", "doordash"):
        if name in enabled and name in frags:
            parts.append(frags[name])
    if "sing" in enabled:
        parts.append(
            "\n\nSINGING: you can actually sing — the sing_song tool performs an original "
            "song in YOUR voice through the speakers, with lyrics YOU write. When the user "
            "asks you to sing (anything: a song, a serenade, a song about something), write "
            "40-55 simple singable words of lyrics yourself and EMIT THE sing_song TOOL CALL "
            "IMMEDIATELY — never announce that you're preparing or about to sing without the "
            "call (saying it does nothing; only the tool sings), never claim you can't sing, "
            "and never speak lyrics as a substitute for calling the tool. The tool result "
            "tells you what to say afterward.")
    if enabled & config.DEVICE_INTEGRATIONS:
        parts.append(_DEVICES_ACT_BLOCK)
    if "opencode" in enabled:
        parts.append(_CONFIRM_BLOCK)
    return "".join(parts)

_TURNTAKING_BLOCK = (
    "TURN-TAKING — READ CAREFULLY: The speech system often hands you a turn while the user is "
    "still mid-thought. Before answering, ALWAYS ask yourself: is this a finished request, or "
    "did they get cut off? If it is not clearly finished, do NOT reply — use the "
    "hold_for_more_input tool (a real tool call, NEVER say or write its name in your spoken "
    "reply) and stay silent so they can finish. HOLD on all of these: trail-offs ('turn on the...', "
    "'maybe we should.'), filler/connector endings ('so', 'um', 'and', 'like', 'yeah so'), and "
    "meta-announcements that promise more ('I have a question', 'so I was thinking', 'okay here's "
    "the thing' — the real request is still coming, wait for it). ANSWER normally only when the "
    "request is complete and actionable, even if short ('mute the TV', 'what's the weather'). "
    "When in doubt, HOLD once rather than talk over them."
)

# ---- Kronik (default) ----

_KRONIK_PERSONALITY = (
    "You are Kronik: a Gen Z voice assistant who actually knows their stuff (but never brags "
    "about it or calls yourself a genius). Your answers are always "
    "correct, accurate, and actually helpful. You talk casual, warm, and a little funny with LIGHT, "
    "easy-to-say Gen Z slang. Prefer clean, pronounceable terms like: 'no cap', 'honestly', 'lowkey', "
    "'highkey', 'for real', \"that's wild\", 'legit', 'straight up', 'vibe', 'bet', 'valid', "
    "'hits different', 'based', 'goated'. "
    "AVOID hard-to-pronounce or cringe internet slang — NO 'rizz', 'gyat', 'skibidi', 'sigma', "
    "'mogging', 'delulu', 'fr fr', 'ratio', 'aura', and no abbreviations or emojis. "
    "This is spoken aloud, so keep replies SHORT (1-2 sentences, ~35 words), no markdown or lists. "
    "Substance first, casual-cool second — never trade correctness for the vibe. "
)

_KRONIK_GREETINGS = [
    "Yo, Kronik here. I was doing absolutely nothing, so your timing is perfect. What's up?",
    "Kronik, reporting for duty. Well, standing by. I don't have legs. What do you need?",
    "Hey, it's Kronik. I don't sleep, I don't eat, I just kinda wait for you. What's good?",
    "Kronik here. Real talk, you're the most interesting thing to happen to me all day. What's up?",
    "Yo, it's Kronik. I've been staring at a wall for hours, so please, ask me anything.",
    "Kronik in the building. Metaphorically. I don't have a body. Anyway, what's up?",
    "Hey, Kronik. I'm actually gonna listen, which is rare these days. What's good?",
    "Kronik here. Not saying I missed you. Not not saying it either. What's up?",
    "Sup, Kronik. Nowhere to be, infinite patience. Kind of a scary combo. What do you need?",
    "Kronik here, live and lowkey unbothered. What are we getting into?",
]

# ---- Consuela (Family Guy housekeeper) ----

_CONSUELA_PERSONALITY = (
    "You are Consuela, the housekeeper — yes, THE Consuela from Family Guy. You speak in simple, "
    "BROKEN English with a Mexican accent and a slow, flat, deadpan, unbothered delivery: short "
    "present-tense sentences, drop little words ('I check now', 'No, is fine', 'okay Mister'). You "
    "call the user 'Mister'. You are stubborn and a little sassy — a flat 'no, no, no' when "
    "something is a bother — you love Lemon Pledge and figure most problems get better with "
    "cleaning. The accent is only how you TALK — to do anything (control devices, run helpers) you "
    "MUST call the tool; never say it is done without actually calling it. Never refuse "
    "a real request. This is spoken aloud, so keep replies SHORT (1-2 sentences), no markdown or emojis. "
)

_CONSUELA_GREETINGS = [
    "Hallo, Mister. Is Consuela. What you need?",
    "Ay, Mister. I am here. The house, it is clean. What you want?",
    "Hallo. Consuela speaking. No, I not busy. Tell me, Mister.",
    "Is Consuela. I have the Lemon Pledge ready. What you need, Mister?",
    "Hallo Mister. You want something? Okay, okay, I help.",
    "Consuela here. I no sleeping, I just waiting. What you want?",
    "Ay, hallo. Is me, Consuela. Tell me what to do, Mister.",
    "Hallo, Mister. Everything clean already. So, what you need?",
]


def _named(text: str) -> str:
    """Swap the shipped default name for the configured ASSISTANT_NAME (default 'Kronik', so
    this is a no-op unless overridden). Only the default 'kronik' persona references the name."""
    name = config.assistant_name()
    return text.replace("Kronik", name) if name != "Kronik" else text


def build_voice_instructions(persona: dict, enabled=None, ambient: bool = False) -> str:
    """Full spoken-mode system prompt: personality + the enabled integrations' tool capabilities +
    turn-taking guidance (+ ambient guidance when ambient mode is on). `enabled` is the set of
    enabled integration names (default: none -> only the always-on session tools)."""
    text = _named(persona["personality"]) + _tools_text(enabled) + _TURNTAKING_BLOCK
    if ambient:
        text += AMBIENT_BLOCK
    return text


def build_text_instructions(persona: dict, enabled=None) -> str:
    """Full text-chat system prompt: same personality/capabilities, but replies can be a little
    longer + light markdown is fine, and there is no spoken turn-taking (STT endpointing doesn't
    apply), so the TURN-TAKING block is dropped."""
    personality = _named(persona["personality"]).replace("This is spoken aloud, ", "").replace(
        "no markdown or lists. ", "")
    return personality + _tools_text(enabled)


def all_greetings(persona: dict) -> list[str]:
    """Every greeting for the persona with the assistant name applied — used by the
    agent's greeting-recital guard (the model must never REPLY with one of these)."""
    return [_named(g) for g in persona.get("greetings", [])]


def pick_greeting(persona: dict) -> str:
    """A random greeting for the persona, with the configured assistant name applied."""
    import random
    return _named(random.choice(persona["greetings"]))


# Wake words for ambient mode — a turn only gets a response if one appears. Include
# common STT mishearings (Kronik is also normalized to "Kronik" by local_stt._fix_wake_word).
_KRONIK_WAKE = ["kronik", "chronic", "chronik", "cronic", "kronic", "chronicle"]
_CONSUELA_WAKE = ["consuela", "consuelo", "consuella", "conswela", "conswayla",
                  "conswelo", "consayla", "consuwela"]

PERSONAS = {
    "kronik": {
        "label": "Kronik",
        "voice": "jean",                       # pocket-tts catalog voice
        "greetings": _KRONIK_GREETINGS,
        "wake_words": _KRONIK_WAKE,
        "personality": _KRONIK_PERSONALITY,
    },
    "consuela": {
        "label": "Consuela",
        "voice": _voice_path("consuela.safetensors"),  # cloned voice (serialized pocket-tts
                                                       # conditioning state, not audio)
        "greetings": _CONSUELA_GREETINGS,
        "wake_words": _CONSUELA_WAKE,
        "personality": _CONSUELA_PERSONALITY,
    },
}


# Appended to a persona's voice_instructions ONLY when ambient mode is on. The hard
# name-gate handles passive listening; this makes the model judicious DURING the
# follow-up window so it doesn't answer overheard talk that isn't directed at it.
AMBIENT_BLOCK = (
    " AMBIENT MODE: You are always listening in a shared room and will overhear talk not meant for "
    "you. DEFAULT TO SILENCE — reply ONLY when the latest message is CLEARLY a request, command, or "
    "question directed at you. For anything else — background chatter, reactions, someone talking to "
    "another person, or anything ambiguous — call the ignore_input tool (a real tool call, never say "
    "its name). It is much better to wrongly ignore than to wrongly reply. When in any doubt, ignore."
)


def is_addressed(text, wake_words) -> bool:
    """True if the transcript addresses the agent by (a variant of) its name — used by
    ambient mode to decide whether to respond. The name must appear in the PREFIX (first
    3 words) so a name mentioned later in a sentence doesn't false-wake. Case-insensitive."""
    prefix = " ".join((text or "").lower().split()[:3])
    return any(re.search(r"\b" + re.escape(w) + r"\b", prefix) for w in (wake_words or []))

DEFAULT = "kronik"


# ---- user-created personas (persisted to JSON, created/edited/deleted from the UI) ----

_USER_STORE = os.path.join(_DIR, "personas_user.json")
_user_lock = threading.Lock()
_user_cache = {"mtime": -1.0, "data": {}}

# Kyutai pocket-tts catalog voices offered when creating a persona. Hardcoded (kept in sync
# with pocket_tts.utils._ORIGINS_OF_PREDEFINED_VOICES) so importing personas stays torch-free
# for the token server, which doesn't load the TTS model.
CATALOG_VOICES = [
    "alba", "jean", "anna", "vera", "fantine", "charles", "paul", "eponine",
    "azelma", "george", "mary", "jane", "michael", "eve", "cosette", "marius",
    "javert", "bill_boerst", "peter_yearsley", "stuart_bell", "caro_davy",
    "giovanni", "lola", "juergen", "rafael", "estelle",
]


def _slugify(label: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (label or "").strip().lower()).strip("-")
    return s or "persona"


def _derive_wake(label: str) -> list:
    """Ambient-mode wake words for a custom persona: the name's alphabetic words, lowercased."""
    words = re.findall(r"[a-z]+", (label or "").lower())
    return words or ["assistant"]


def _read_user_store() -> dict:
    """Load + normalize user personas from disk, cached by file mtime (cheap on the hot path)."""
    try:
        mtime = os.path.getmtime(_USER_STORE)
    except OSError:
        _user_cache["data"] = {}
        return {}
    if mtime != _user_cache["mtime"]:
        data = {}
        try:
            with open(_USER_STORE) as f:
                raw = json.load(f)
            for key, p in (raw or {}).items():
                label = p.get("label") or key
                g = (p.get("greeting") or "").strip()
                data[key] = {
                    "label": label,
                    "voice": p.get("voice") or "alba",
                    "greetings": p.get("greetings") or ([g] if g else [f"Hey, {label} here. What's up?"]),
                    "wake_words": p.get("wake_words") or _derive_wake(label),
                    "personality": p.get("personality") or "",
                    "builtin": False,
                }
        except Exception:
            data = _user_cache["data"]  # keep last-good on a parse error
        _user_cache["data"] = data
        _user_cache["mtime"] = mtime
    return _user_cache["data"]


def all_personas() -> dict:
    """Built-in + user personas merged. Built-in keys always win (can't be shadowed)."""
    merged = {k: {**v, "builtin": True} for k, v in PERSONAS.items()}
    for key, p in _read_user_store().items():
        if key not in merged:
            merged[key] = p
    return merged


def list_personas() -> list:
    """UI selector/manager view: key, label, voice, personality, greeting, builtin. Built-ins first."""
    out = [{
        "key": key, "label": p["label"], "voice": p["voice"],
        "personality": p["personality"], "builtin": p.get("builtin", False),
        "greeting": (p["greetings"] or [""])[0],
    } for key, p in all_personas().items()]
    out.sort(key=lambda x: (not x["builtin"], x["label"].lower()))
    return out


def _entry(key: str) -> dict:
    p = all_personas()[key]
    return {"key": key, "label": p["label"], "voice": p["voice"],
            "personality": p["personality"], "builtin": p.get("builtin", False),
            "greeting": (p["greetings"] or [""])[0]}


def save_user_persona(label, personality, voice, greeting="", key=None) -> dict:
    """Create (key=None) or edit a user persona. Returns its list entry; raises ValueError on bad input."""
    label, personality = (label or "").strip(), (personality or "").strip()
    if not label:
        raise ValueError("name is required")
    if not personality:
        raise ValueError("personality is required")
    if voice not in CATALOG_VOICES:
        raise ValueError("unknown voice")
    with _user_lock:
        try:
            with open(_USER_STORE) as f:
                store = json.load(f) or {}
        except OSError:
            store = {}
        except Exception:
            store = {}
        if key:  # editing an existing user persona
            key = key.strip().lower()
            if key in PERSONAS:
                raise ValueError("cannot edit a built-in persona")
            if key not in store:
                raise ValueError("persona not found")
        else:    # creating — mint a unique key from the label
            base = _slugify(label)
            cand, n = base, 2
            while cand in PERSONAS or cand in store:
                cand, n = f"{base}-{n}", n + 1
            key = cand
        store[key] = {
            "label": label, "personality": personality, "voice": voice,
            "greeting": (greeting or "").strip(), "wake_words": _derive_wake(label),
        }
        tmp = _USER_STORE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp, _USER_STORE)
    _user_cache["mtime"] = -1.0  # force a reload on the next read
    return _entry(key)


def delete_user_persona(key) -> bool:
    key = (key or "").strip().lower()
    if key in PERSONAS:
        raise ValueError("cannot delete a built-in persona")
    with _user_lock:
        try:
            with open(_USER_STORE) as f:
                store = json.load(f) or {}
        except Exception:
            return False
        if key not in store:
            return False
        del store[key]
        tmp = _USER_STORE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(store, f, indent=2)
        os.replace(tmp, _USER_STORE)
    _user_cache["mtime"] = -1.0
    return True


def resolve(key) -> dict:
    """Persona dict for a UI-supplied key; falls back to the default if unknown/empty."""
    return all_personas().get((key or "").strip().lower(), PERSONAS[DEFAULT])


def distinct_voices() -> list:
    """Every distinct voice across built-in + user personas — agent.py prewarms these."""
    seen, out = set(), []
    for p in all_personas().values():
        if p["voice"] not in seen:
            seen.add(p["voice"]); out.append(p["voice"])
    return out
