"""Central runtime configuration — all env-driven (agent/.env.local).

Out of the box ONLY the OpenCode background-session tools are active. Each home integration is
opt-in via an env flag (e.g. WIZ_ENABLED=true), which `python agent/onboard.py <name>` sets for
you after linking the integration. Generic identity + device display names also live here so the
project ships with no personal/home-specific constants baked into prompts or responses.

Everything is read LAZILY (via functions), NOT as import-time constants, because some callers
import this module before their .env.local is loaded — reading at call time avoids stale defaults.
"""
import os

# Repo root = the directory above agent/, derived from THIS file — never a hardcoded
# ~/voice-agent and never cwd. Lets the whole stack run from a clone in any directory
# (logs, sessions DB, device certs all land beside the code).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _flag(key: str) -> bool:
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _name(key: str, default: str) -> str:
    return (os.environ.get(key, "") or "").strip() or default


# ---- identity (overridable; ships defaulting to "Kronik") ----
def assistant_name() -> str:
    return _name("ASSISTANT_NAME", "Kronik")


def wake_word() -> str:
    return _name("WAKE_WORD", "kronik").lower()


# ---- generic device / pet display names (used in prompts + tool responses) ----
def vacuum_name() -> str:
    return _name("VACUUM_NAME", "the vacuum")


def car_name() -> str:
    return _name("CAR_NAME", "the car")


def pet_name() -> str:
    return _name("PET_NAME", "the cat")


# ---- integration registry: name -> enable flag + the tool names it exposes ----
# Most tool names also have chat_backend entries. The live Ring watcher is voice-only because it
# sends asynchronous session.say() updates to an active audio session.
INTEGRATIONS = {
    "opencode": {"flag": "OPENCODE_ENABLED", "tools": ["run_agent", "list_sessions", "get_session_status", "message_session", "delete_session"]},
    "tv":       {"flag": "TV_ENABLED",       "tools": ["control_tv", "play_on_tv"]},
    "nest":     {"flag": "NEST_ENABLED",     "tools": ["thermostat_status", "set_thermostat_temperature", "set_thermostat_mode"]},
    "roomba":   {"flag": "ROOMBA_ENABLED",   "tools": ["roomba_status", "control_roomba"]},
    "wiz":      {"flag": "WIZ_ENABLED",      "tools": ["lights_status", "control_lights"]},
    "ring":     {"flag": "RING_ENABLED",     "tools": ["check_camera", "camera_recent_activity", "watch_camera_live", "stop_camera_watch"]},
    "tesla":    {"flag": "TESLA_ENABLED",    "tools": ["tesla_status", "tesla_climate", "tesla_lock", "tesla_charging", "tesla_find", "tesla_locate"]},
    "petlibro": {"flag": "PETLIBRO_ENABLED", "tools": ["cat_feeder_status", "feed_cat"]},
    "doordash": {"flag": "DOORDASH_ENABLED", "tools": [
        "doordash_find_items", "doordash_search", "doordash_default_address", "doordash_menu",
        "doordash_cart", "doordash_add_to_cart", "doordash_remove_from_cart", "doordash_delete_cart",
        "doordash_preview_order", "doordash_submit_order", "doordash_order_status",
        "doordash_roulette_prepare", "doordash_roulette_submit", "doordash_roulette_cancel",
    ]},
}

# Physical-device integrations — these get the "you can't see the device, call the tool" guidance.
DEVICE_INTEGRATIONS = {"tv", "nest", "roomba", "wiz", "ring", "tesla", "petlibro"}

# Always exposed regardless of config: only the turn-taking control tools (agent-internal, not a
# user-facing feature). Every actual capability — including OpenCode sessions — is an opt-in integration.
ALWAYS_ON_TOOLS = {
    "hold_for_more_input", "ignore_input",
}


def is_enabled(name: str) -> bool:
    spec = INTEGRATIONS.get(name)
    return bool(spec) and _flag(spec["flag"])


def enabled_integrations() -> set:
    """The set of integration names currently switched on via their env flags."""
    return {name for name in INTEGRATIONS if is_enabled(name)}


def enabled_tool_names() -> set:
    """All tool names that should be exposed: always-on tools + every enabled integration's tools."""
    names = set(ALWAYS_ON_TOOLS)
    for name in enabled_integrations():
        names.update(INTEGRATIONS[name]["tools"])
    return names
