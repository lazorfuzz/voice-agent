"""Backend for Kronik's Roomba tools — LOCAL control via roombapy (MQTT over TLS on
:8883). No cloud at runtime. Credentials (ip/blid/local password) are in .env.local; the
local password was fetched once from the iRobot cloud (newer firmware disabled on-robot
retrieval). Connect PER CALL and disconnect immediately — the Roomba only allows ONE local
connection at a time, so a persistent one would lock out other callers (voice vs. text)."""
import os
import time
import threading

from roombapy import RoombaFactory

_IP = os.environ.get("ROOMBA_IP", "")
_BLID = os.environ.get("ROOMBA_BLID", "")
_PW = os.environ.get("ROOMBA_PASSWORD", "")
_DEBUG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "roomba_debug.log")

# The Roomba allows only ONE local MQTT connection at a time, and both the text-chat
# backend and every voice job-process would fight over it. So connect PER CALL and
# disconnect immediately — the single slot is always freed for the next caller.
_lock = threading.Lock()   # serialize connects within one process


def _log(msg):
    try:
        with open(_DEBUG, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    except Exception:
        pass


def _connect(tries=2):
    """Open a fresh local connection and wait for state. Caller MUST disconnect()."""
    last = None
    for attempt in range(tries):
        r = RoombaFactory.create_roomba(address=_IP, blid=_BLID, password=_PW)
        try:
            r.connect()
        except Exception as e:   # e.g. slot briefly held by a releasing connection
            last = e
            try:
                r.disconnect()
            except Exception:
                pass
            time.sleep(1.0)
            continue
        for _ in range(25):      # wait for the retained state message
            if getattr(r, "master_state", None):
                break
            time.sleep(0.2)
        return r
    raise last or RuntimeError("connect failed")


def _reported(r):
    st = r.master_state or {}
    s = st.get("state", {})
    return s.get("reported", s) if isinstance(s, dict) else {}


_PHASE = {
    "charge": "charging on the dock", "run": "cleaning", "stop": "stopped",
    "pause": "paused", "hmMidMsn": "heading back to empty its bin",
    "hmUsrDock": "returning to the dock", "hmPostMsn": "returning to the dock",
    "evac": "emptying its bin",
}


def status() -> str:
    with _lock:
        try:
            r = _connect()
        except Exception as e:
            _log(f"status ERROR {type(e).__name__}: {e}")
            return f"I couldn't reach the Roomba ({type(e).__name__}). Is it powered on?"
        try:
            rep = _reported(r)
        finally:
            try:
                r.disconnect()
            except Exception:
                pass
    name = rep.get("name") or "The Roomba"
    bat = rep.get("batPct")
    phase = (rep.get("cleanMissionStatus", {}) or {}).get("phase")
    ph = _PHASE.get(phase, phase or "idle")
    binfull = (rep.get("bin", {}) or {}).get("full")
    extra = " Heads up, its bin is full." if binfull else ""
    bat_s = f", battery at {bat}%" if bat is not None else ""
    return f"{name} is {ph}{bat_s}.{extra}"


# roombapy/dorita980 command names
_ALIASES = {
    "clean": "start", "start": "start", "go": "start", "vacuum": "start",
    "stop": "stop", "halt": "stop",
    "pause": "pause", "resume": "resume", "continue": "resume",
    "dock": "dock", "home": "dock", "return": "dock", "go_home": "dock",
    "empty": "evac", "evac": "evac", "find": "find", "locate": "find",
}
_CONFIRM = {
    "start": "On it — the vacuum's off to clean.", "stop": "Stopped the vacuum.",
    "pause": "Paused it.", "resume": "Back to cleaning.",
    "dock": "Sending it home to the dock.", "evac": "Emptying its bin.",
    "find": "Pinging it so you can find it.",
}


def control(action: str) -> str:
    a = _ALIASES.get((action or "").lower().strip().replace(" ", "_"))
    if not a:
        return f"I don't know how to '{action}' the Roomba."
    _log(f"control({action!r} -> {a})")
    with _lock:
        try:
            r = _connect()
        except Exception as e:
            _log(f"control connect ERROR {type(e).__name__}: {e}")
            return f"I couldn't reach the Roomba ({type(e).__name__})."
        try:
            if a == "dock":
                # A bare `dock` is IGNORED while a mission is running, and `stop`+`dock`
                # strands it mid-floor (error 34). The app's real "send home" flow is
                # pause -> dock, which reliably kicks it into hmUsrDock.
                r.send_command("pause")
                time.sleep(2.5)
                r.send_command("dock")
                time.sleep(1.5)
            else:
                r.send_command(a)
                time.sleep(1.5)   # let paho actually publish before we disconnect
            return _CONFIRM.get(a, f"Roomba: {a}.")
        except Exception as e:
            _log(f"control ERROR {type(e).__name__}: {e}")
            return f"That Roomba command failed ({type(e).__name__})."
        finally:
            try:
                r.disconnect()
            except Exception:
                pass
