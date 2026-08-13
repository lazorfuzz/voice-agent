"""Backend for Kronik's WiZ light tools. WiZ bulbs speak a local JSON-over-UDP protocol
on port 38899 (no cloud, no auth): {"method":"setPilot","params":{...}}.

Lights are configured by NAME in .env.local:
    WIZ_LIGHTS=bedside 1=9877d5dd0bae, bedside 2=9877d5dcfa32, living room 1=aabbcc...
Names are free-form ("bedside 2", "living room lamp") and matched fuzzily, so a voice
command can target one light ("bedside 1"), a group by shared words ("bedside",
"living room"), or everything ("all"/empty). IPs are DHCP so we resolve MAC->IP by LAN
broadcast at first use and re-discover automatically if a light stops answering."""
import os
import json
import socket
import threading

_PORT = 38899
_BROADCAST = os.environ.get("WIZ_BROADCAST", "255.255.255.255")


def _parse_lights():
    """WIZ_LIGHTS='name=mac, name=mac' -> ordered [(name, mac)]."""
    out = []
    for entry in os.environ.get("WIZ_LIGHTS", "").split(","):
        if "=" in entry:
            name, mac = entry.split("=", 1)
            if name.strip() and mac.strip():
                out.append((name.strip().lower(), mac.strip().lower()))
    if not out:  # legacy fallback from the first integration round
        for i, mac in enumerate(m.strip().lower() for m in
                                os.environ.get("WIZ_LAMPS", "").split(",") if m.strip()):
            out.append((f"bedside {i + 1}", mac))
    return out


_LIGHTS = _parse_lights()

_lock = threading.Lock()
_ip_by_mac: dict[str, str] = {}


def _send(ip, method, params=None, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        s.sendto(json.dumps({"method": method, "params": params or {}}).encode(), (ip, _PORT))
        data, _ = s.recvfrom(4096)
        return json.loads(data.decode())
    finally:
        s.close()


def _discover():
    """Broadcast getPilot and map every responding WiZ MAC -> IP."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(0.4)
    msg = json.dumps({"method": "getPilot", "params": {}}).encode()
    # Re-read at call time (not just the import-time _BROADCAST): if this module is imported
    # before .env.local is loaded, the module-level default would otherwise be frozen.
    broadcast = os.environ.get("WIZ_BROADCAST", _BROADCAST)
    found = {}
    try:
        for _ in range(2):                      # UDP is lossy; two rounds is plenty on a LAN
            s.sendto(msg, (broadcast, _PORT))
            while True:
                try:
                    data, addr = s.recvfrom(4096)
                    mac = json.loads(data.decode()).get("result", {}).get("mac", "").lower()
                    if mac:
                        found[mac] = addr[0]
                except socket.timeout:
                    break
                except Exception:
                    pass
    finally:
        s.close()
    return found


def _ip_for(mac):
    with _lock:
        ip = _ip_by_mac.get(mac)
        if ip:
            return ip
        _ip_by_mac.update(_discover())
        return _ip_by_mac.get(mac)


def _call(mac, method, params=None):
    """Send to a light by MAC. UDP is lossy -> retry once on the known IP; if it's still
    silent, re-discover (DHCP moved it) and try again."""
    ip = _ip_for(mac)
    if ip:
        for _ in range(2):
            try:
                return _send(ip, method, params)
            except Exception:
                pass
    with _lock:
        _ip_by_mac.update(_discover())
        ip = _ip_by_mac.get(mac)
    if not ip:
        raise OSError(f"light {mac} not found on the network")
    return _send(ip, method, params)


def push_pilot(mac, params):
    """Fire-and-forget setPilot for the pulse engine (dimming and/or r,g,b): no reply
    wait, no retry — at ~10Hz a lost packet is corrected by the next one anyway."""
    ip = _ip_by_mac.get(mac) or _ip_for(mac)
    if not ip:
        return
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.sendto(json.dumps({"method": "setPilot", "params": params}).encode(), (ip, _PORT))
    except Exception:
        pass
    finally:
        s.close()


def get_pilot(mac):
    """Current bulb state dict (state/dimming/temp/sceneId/r,g,b...)."""
    return _call(mac, "getPilot")["result"]


def get_pilot_fast(mac, timeout=0.18):
    """Single-shot getPilot on the CACHED ip only — no retry, no re-discovery. For the
    pulse hot path: an unreachable light must not block (a dead lamp used to trigger a
    ~2s retry+rediscover and offset the whole light show). Returns None if it doesn't
    answer promptly; caller just skips it this turn."""
    ip = _ip_by_mac.get(mac)
    if not ip:
        return None
    try:
        return _send(ip, "getPilot", timeout=timeout)["result"]
    except Exception:
        return None


def prime_discovery():
    """Populate the MAC->IP cache once (blocking broadcast). Call from prewarm so the
    pulse hot path always has cached IPs and never needs a slow in-loop discovery."""
    with _lock:
        _ip_by_mac.update(_discover())


def restore_pilot(mac, pilot):
    """Put a bulb back to a previously captured getPilot result, exactly."""
    p = {"state": pilot.get("state", True)}
    if "dimming" in pilot:
        p["dimming"] = pilot["dimming"]
    if pilot.get("sceneId"):
        p["sceneId"] = pilot["sceneId"]
    elif "temp" in pilot:
        p["temp"] = pilot["temp"]
    elif "r" in pilot:
        p.update({k: pilot[k] for k in ("r", "g", "b") if k in pilot})
    return _call(mac, "setPilot", p)


_ALL_WORDS = ("", "all", "everything", "every light", "all lights", "lights", "the lights",
              "all the lights", "both", "both lamps", "lamps")


def resolve(which=None):
    """Fuzzy-match a spoken target to [(name, mac)]. '' / 'all' / 'lights' -> everything;
    otherwise substring/word match against configured names ('bedside' -> bedside 1+2)."""
    if not _LIGHTS:
        return []
    t = (which or "").lower().strip().rstrip(".!?")
    if t in _ALL_WORDS:
        return list(_LIGHTS)
    hits = [(n, m) for n, m in _LIGHTS if t in n or n in t]
    if hits:
        return hits
    # word-subset match after dropping filler words: "the bedside lamps" -> bedside 1+2,
    # "the living room lamp" -> living room 1, etc.
    _STOP = {"the", "a", "my", "lamp", "lamps", "light", "lights", "bulb", "bulbs"}
    tw = {w for w in t.split() if w not in _STOP}
    return [(n, m) for n, m in _LIGHTS if tw and tw.issubset(set(n.split()) | _STOP)]


def _label(lights):
    if len(lights) == len(_LIGHTS):
        return "all the lights" if len(_LIGHTS) > 2 else "both lamps"
    return " and ".join(n for n, _ in lights)


_COLORS = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
    "orange": (255, 120, 0), "yellow": (255, 200, 0), "purple": (150, 0, 255),
    "violet": (150, 0, 255), "pink": (255, 60, 120), "magenta": (255, 0, 255),
    "cyan": (0, 255, 255), "teal": (0, 200, 180), "white": (255, 255, 255),
}

# WiZ built-in scene ids (subset that makes sense for home lamps)
_SCENES = {
    "ocean": 1, "romance": 2, "sunset": 3, "party": 4, "fireplace": 5, "cozy": 6,
    "forest": 7, "wake up": 9, "wakeup": 9, "bedtime": 10, "warm white": 11,
    "daylight": 12, "cool white": 13, "night light": 14, "nightlight": 14,
    "focus": 15, "relax": 16, "tv time": 18, "christmas": 27, "halloween": 28,
    "candlelight": 29, "candle": 29, "golden white": 30,
}

_TEMPS = {"warm": 2700, "warm white": 2700, "soft": 3000, "neutral": 4000,
          "cool": 5500, "cool white": 5500, "daylight": 6500}


def status(which=None) -> str:
    lights = resolve(which)
    if not lights:
        known = ", ".join(n for n, _ in _LIGHTS) or "none configured"
        return f"I couldn't match '{which}'. The lights I know: {known}."
    parts = []
    for name, mac in lights:
        try:
            r = get_pilot(mac)
        except Exception:
            parts.append(f"{name} isn't responding")
            continue
        if not r.get("state"):
            parts.append(f"{name} is off")
        elif r.get("sceneId"):
            sname = next((k for k, v in _SCENES.items() if v == r["sceneId"]), f"scene {r['sceneId']}")
            parts.append(f"{name} is on at {r.get('dimming', '?')}% ({sname} scene)")
        elif "temp" in r:
            parts.append(f"{name} is on at {r.get('dimming', '?')}% ({r['temp']}K white)")
        elif "r" in r:
            parts.append(f"{name} is on at {r.get('dimming', '?')}% (color)")
        else:
            parts.append(f"{name} is on at {r.get('dimming', '?')}%")
    return "; ".join(parts) + "."


def control(action, brightness=None, color=None, which=None) -> str:
    lights = resolve(which)
    if not lights:
        known = ", ".join(n for n, _ in _LIGHTS) or "none configured"
        return f"I couldn't match '{which}'. The lights I know: {known}."
    a = (action or "").lower().strip()
    label = _label(lights)

    params, said = None, None
    if a in ("on", "turn_on"):
        params, said = {"state": True}, f"Turned {label} on."
    elif a in ("off", "turn_off"):
        params, said = {"state": False}, f"Turned {label} off."
    elif a in ("brightness", "dim", "set", "set_brightness"):
        if brightness is None:
            return "What brightness (1 to 100)?"
        b = max(1, min(100, int(brightness)))
        params, said = {"state": True, "dimming": b}, f"Set {label} to {b} percent."
    elif a in ("color", "scene", "mood"):
        c = (color or "").lower().strip()
        if c in _TEMPS:
            params, said = {"state": True, "temp": _TEMPS[c]}, f"Set {label} to {c} white."
        elif c in _SCENES:
            params, said = {"state": True, "sceneId": _SCENES[c]}, f"Put {label} on the {c} scene."
        elif c in _COLORS:
            r, g, b = _COLORS[c]
            params, said = {"state": True, "r": r, "g": g, "b": b}, f"Set {label} to {c}."
        else:
            return (f"I don't know '{color}'. Try a color (red, blue, pink...), a white "
                    "(warm, neutral, cool, daylight), or a scene (nightlight, cozy, "
                    "candlelight, relax, fireplace...).")
    elif a in ("identify", "blink", "find"):
        # pulse each targeted light so the user can tell which is which
        import time
        for name, mac in lights:
            try:
                before = get_pilot(mac)
                _call(mac, "setPilot", {"state": True, "dimming": 100})
                time.sleep(0.8)
                _call(mac, "setPilot", {"state": False})
                time.sleep(0.8)
                restore_pilot(mac, before)
            except Exception:
                return f"Couldn't reach {name} to blink it."
        return f"Blinked {label}."
    else:
        return f"I don't know the action '{action}'. Try on, off, brightness, color, or identify."

    if brightness is not None and "dimming" not in params:
        params["dimming"] = max(1, min(100, int(brightness)))

    failed = []
    for name, mac in lights:
        try:
            ok = _call(mac, "setPilot", params).get("result", {}).get("success")
            if not ok:
                failed.append(name)
        except Exception:
            failed.append(name)
    if not failed:
        return said
    if len(failed) == len(lights):
        return f"{label} didn't respond — are they powered at the switch?"
    return f"{said} But {', '.join(failed)} didn't respond."
