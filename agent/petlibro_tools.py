"""PetLibro cat feeder — cloud control (the feeder exposes NO local API; all ports
filtered). Uses the reverse-engineered PetLibro app API (api.us.petlibro.com).

Auth: PetLibro logs in with email + MD5(password) -> a token. We store the TOKEN plus the
MD5 HASH (not the plaintext password, which is discarded) so the session can silently
re-login when the token expires — otherwise "feed the cat" would break whenever it lapses.
Set up once with petlibro_auth.py.
"""
import asyncio
import hashlib
import json
import os
import uuid

import aiohttp

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
BASE = "https://api.us.petlibro.com"
_APP_SN = "c35772530d1041699c87fe62348507a8"

_lock = asyncio.Lock()
_session = None
_token = None
_device = None   # cached (id, deviceSn, name)


def md5(pw: str) -> str:
    return hashlib.md5(pw.encode("utf-8")).hexdigest()


def _upsert_env(key, value):
    lines = open(_ENV).read().splitlines() if os.path.exists(_ENV) else []
    if any(l.startswith(key + "=") for l in lines):
        lines = [f"{key}={value}" if l.startswith(key + "=") else l for l in lines]
    else:
        lines.append(f"{key}={value}")
    open(_ENV, "w").write("\n".join(lines) + "\n")
    os.chmod(_ENV, 0o600)


def _tz():
    return os.environ.get("PETLIBRO_TZ", "America/Los_Angeles")


def _headers(token=None):
    h = {"source": "ANDROID", "language": "EN", "timezone": _tz(),
         "version": "1.3.45", "Content-Type": "application/json"}
    if token:
        h["token"] = token
    return h


async def _post(path, body, token=None):
    global _session
    if _session is None:
        _session = aiohttp.ClientSession()
    async with _session.post(BASE + path, json=body, headers=_headers(token)) as r:
        return await r.json()


async def login() -> str:
    """Log in with the stored email + password hash, cache + persist the token."""
    global _token
    email = os.environ["PETLIBRO_EMAIL"]
    pmd5 = os.environ["PETLIBRO_PWD_MD5"]
    body = {"appId": 1, "appSn": _APP_SN, "country": os.environ.get("PETLIBRO_COUNTRY", "US"),
            "email": email, "password": pmd5, "phoneBrand": "", "phoneSystemVersion": "",
            "timezone": _tz(), "thirdId": None, "type": None}
    d = await _post("/member/auth/login", body)
    if d.get("code") != 0 or not d.get("data"):
        raise RuntimeError(f"PetLibro login failed: {d.get('msg', d)}")
    _token = d["data"]["token"]
    _upsert_env("PETLIBRO_TOKEN", _token)
    return _token


async def _call(path, body):
    """POST an authenticated endpoint; transparently re-login once if the token lapsed."""
    global _token
    if _token is None:
        _token = os.environ.get("PETLIBRO_TOKEN") or await login()
    d = await _post(path, body, token=_token)
    # token-expiry / unauthorized codes -> re-login and retry once
    if d.get("code") in (1009, 1005, 401) or (d.get("msg") or "").upper() in ("TOKEN_ERROR", "NOT_LOGIN"):
        await login()
        d = await _post(path, body, token=_token)
    return d


async def devices() -> list:
    d = await _call("/device/device/list", {})
    return d.get("data", []) or []


async def _first():
    """The first feeder's live record from the device list (it already carries all status —
    online/surplusGrain/battery/weight — so no separate realInfo call). PetLibro identifies
    the device by deviceSn; there is no numeric id."""
    devs = await devices()
    return devs[0] if devs else None


def _resilient(fn):
    async def wrap(*a, **k):
        if not os.environ.get("PETLIBRO_EMAIL") or not os.environ.get("PETLIBRO_PWD_MD5"):
            return "The cat feeder isn't linked yet — run agent/petlibro_auth.py to log in."
        try:
            return await fn(*a, **k)
        except Exception as e:
            return f"Couldn't reach the cat feeder just now ({type(e).__name__}). Try again shortly."
    wrap.__name__ = fn.__name__
    return wrap


@_resilient
async def status() -> str:
    d = await _first()
    if not d:
        return "No PetLibro feeder found on the account."
    name = d.get("name", "the feeder")
    if not d.get("online"):
        return f"{name} is offline right now."
    parts = []
    # HOPPER — the food STORE. warehouseSurplusGrain (GOOD/LOW/…) is richer than the surplusGrain
    # bool. NOTE: the PLAF203 has NO bowl weight sensor (weight/weightPercent are vestigial always-0
    # fields inherited from feeders that do), so we DON'T report a bowl fill — it would be wrong.
    wh = d.get("warehouseSurplusGrain")
    if wh:
        parts.append("hopper full" if str(wh).upper() == "GOOD" else f"hopper {str(wh).lower()}")
    elif d.get("surplusGrain") is not None:
        parts.append("hopper has food" if d.get("surplusGrain") else "hopper empty — refill it")
    # today's feedings = the real "has food been dispensed into the bowl" signal (no bowl scale).
    try:
        gs = (await _call("/device/data/grainStatus", {"id": d["deviceSn"]})).get("data") or {}
        times, qty = gs.get("todayFeedingTimes"), gs.get("todayFeedingQuantity")
        if times is not None:
            parts.append(f"fed {times}× today" + (f" ({qty} portions)" if qty else ""))
    except Exception:
        pass
    # battery (feeder is usually on AC; battery is backup)
    bs, eq = d.get("batteryState"), d.get("electricQuantity")
    if bs:
        parts.append(f"battery {bs}" + (f" ({eq}%)" if eq else ""))
    if d.get("desiccantState") not in (None, ""):
        parts.append(f"desiccant {str(d['desiccantState']).lower()}")
    return f"{name}: " + (", ".join(parts) if parts else "online") + "."


@_resilient
async def feed(portions: int = 1) -> str:
    d = await _first()
    if not d:
        return "No PetLibro feeder found on the account."
    sn, name = d["deviceSn"], d.get("name", "the feeder")
    cap = int(d.get("maxFeedingCup") or 12)
    n = max(1, min(int(portions or 1), cap))
    d = await _call("/device/device/manualFeeding",
                    {"deviceSn": sn, "grainNum": n, "requestId": uuid.uuid4().hex})
    if d.get("code") != 0:
        return f"Couldn't feed right now ({d.get('msg', 'error')})."
    return f"Fed the cat — dispensed {n} portion{'s' if n != 1 else ''} from {name}."


async def probe() -> dict:
    """Connectivity check for setup: logs in, lists devices, dumps one realInfo (raw keys)."""
    await login()
    devs = await devices()
    return {"devices": [{"name": d.get("name"), "sn": d.get("deviceSn"),
                         "product": d.get("productName")} for d in devs]}
