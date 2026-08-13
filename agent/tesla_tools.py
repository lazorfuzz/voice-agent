"""Tesla vehicle control via the Fleet API (cloud + signed Vehicle Command Protocol).

The car has no local API — everything is Tesla's cloud. Commands on 2021+
cars are cryptographically signed with our private key (tesla_keys/private-key.pem); the
matching public key is registered to the car as a virtual key. Auth is the OAuth token from
tesla_auth.py (TESLA_TOKEN_JSON in .env.local); Tesla ROTATES the refresh token on every
refresh, so we persist it back after each call or auth breaks.

Teslas sleep within minutes, so every read/command wakes the car first (can take 5-20s).
"""
import asyncio
import json
import os
import time

import aiohttp
from tesla_fleet_api import TeslaFleetOAuth
from tesla_fleet_api.exceptions import VehicleOffline

_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV = os.path.join(_DIR, ".env.local")
_KEY = os.path.join(_DIR, "..", "tesla_keys", "private-key.pem")
_BASE = os.environ.get("TESLA_REGION_BASE", "https://fleet-api.prd.na.vn.cloud.tesla.com")

_lock = asyncio.Lock()
_session = None
_api = None
_vin = None
_saved_token = None


def _upsert_env(key, value):
    lines = open(_ENV).read().splitlines() if os.path.exists(_ENV) else []
    if any(l.startswith(key + "=") for l in lines):
        lines = [f"{key}={value}" if l.startswith(key + "=") else l for l in lines]
    else:
        lines.append(f"{key}={value}")
    open(_ENV, "w").write("\n".join(lines) + "\n")
    os.chmod(_ENV, 0o600)


def _persist(api):
    """Save the (possibly refreshed+rotated) token back to .env.local."""
    global _saved_token
    tok = json.dumps({"access_token": api._access_token, "refresh_token": api.refresh_token,
                      "expires": api.expires})
    if tok != _saved_token:
        _upsert_env("TESLA_TOKEN_JSON", tok)
        _saved_token = tok


async def _client():
    """Cached authenticated API + VIN, private key loaded (needed for signed commands)."""
    global _session, _api, _vin, _saved_token
    async with _lock:
        if _api is None:
            raw = os.environ.get("TESLA_TOKEN_JSON")
            if not raw:
                raise RuntimeError("Tesla not linked — run: .venv/bin/python agent/tesla_auth.py")
            _saved_token = raw
            tok = json.loads(raw)
            _session = aiohttp.ClientSession()
            _api = TeslaFleetOAuth(
                _session, region="na", client_id=os.environ["TESLA_CLIENT_ID"],
                client_secret=os.environ["TESLA_CLIENT_SECRET"],
                redirect_uri=os.environ["TESLA_REDIRECT_URI"],
                access_token=tok["access_token"], refresh_token=tok["refresh_token"],
                expires=tok["expires"])
            await _api.get_private_key(_KEY)          # MUST precede any specificSigned()
            _vin = (await _api.products())["response"][0]["vin"]
        return _api, _vin


async def _state(api, vin):
    at = await api.access_token()
    async with _session.get(f"{_BASE}/api/1/vehicles/{vin}",
                            headers={"Authorization": f"Bearer {at}"}) as r:
        return (await r.json()).get("response", {}).get("state")


async def _wake(api, vin, timeout=25):
    """Wake the car and wait until 'online'. Uses a manual POST because the library's wake_up
    omits the Content-Type header and Tesla 406s it."""
    at = await api.access_token()
    h = {"Authorization": f"Bearer {at}", "Content-Type": "application/json"}
    if await _state(api, vin) == "online":
        return True
    async with _session.post(f"{_BASE}/api/1/vehicles/{vin}/wake_up", headers=h, json={}):
        pass
    t0 = time.time()
    while time.time() - t0 < timeout:
        await asyncio.sleep(2.5)
        if await _state(api, vin) == "online":
            await asyncio.sleep(1.5)   # settle: the data/command endpoints lag the state flip
            return True
    return False


async def _veh():
    api, vin = await _client()
    awake = await _wake(api, vin)
    return api, vin, awake


async def _data(api, vin, endpoints=None, tries=4):
    """vehicle_data with retries — right after a wake the state says 'online' before the data
    endpoint is actually ready, so it briefly raises VehicleOffline."""
    veh = api.vehicles.specificSigned(vin)
    for i in range(tries):
        try:
            return (await (veh.vehicle_data(endpoints=endpoints) if endpoints
                           else veh.vehicle_data()))["response"]
        except VehicleOffline:
            if i == tries - 1:
                raise
            await asyncio.sleep(2.5)


def _f2c(f):
    return round((f - 32) * 5 / 9, 1)


def _resilient(fn):
    """Tools must never throw at the agent. A Tesla in a dead zone / deep sleep reports online
    then 408s (VehicleOffline); Tesla's cloud can also hiccup. Return a spoken-friendly message."""
    async def wrap(*a, **k):
        try:
            return await fn(*a, **k)
        except VehicleOffline:
            return ("the car isn't reachable right now — it's asleep or has weak signal "
                    "(a garage or dead zone). Try again in a minute, or wake it in the Tesla app.")
        except Exception as e:
            return f"Couldn't reach the car just now ({type(e).__name__}). Try again shortly."
    wrap.__name__ = fn.__name__
    return wrap


@_resilient
async def status() -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "the car isn't responding right now — it may be asleep or out of signal."
    d = await _data(api, vin)
    _persist(api)
    cs, cl, vs = d.get("charge_state", {}), d.get("climate_state", {}), d.get("vehicle_state", {})
    parts = [f"{cs.get('battery_level')}% battery (~{cs.get('battery_range', 0):.0f} miles)"]
    chg = cs.get("charging_state")
    if chg == "Charging":
        parts.append(f"charging to {cs.get('charge_limit_soc')}%")
    elif chg == "Disconnected":
        parts.append("unplugged")
    elif chg:
        parts.append(chg.lower())
    parts.append("locked" if vs.get("locked") else "UNLOCKED")
    if cl.get("is_climate_on"):
        parts.append(f"climate on, cabin {cl.get('inside_temp')}°C")
    return "the car: " + ", ".join(parts) + "."


@_resilient
async def climate(on: bool, temp_f: int | None = None) -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "Couldn't reach the car to change the climate."
    veh = api.vehicles.specificSigned(vin)
    if on:
        await veh.auto_conditioning_start()
        msg = "Climate on — pre-conditioning the cabin"
        if temp_f is not None:
            c = _f2c(temp_f)
            await veh.set_temps(driver_temp=c, passenger_temp=c)
            msg += f" to {temp_f}°F"
    else:
        await veh.auto_conditioning_stop()
        msg = "Climate off"
    _persist(api)
    return msg + "."


@_resilient
async def lock(locked: bool) -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "Couldn't reach the car to lock it."
    veh = api.vehicles.specificSigned(vin)
    await (veh.door_lock() if locked else veh.door_unlock())
    _persist(api)
    return "Locked the car." if locked else "Unlocked the car."


@_resilient
async def charge(action: str, limit: int | None = None) -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "Couldn't reach the car to change charging."
    veh = api.vehicles.specificSigned(vin)
    a = (action or "").lower()
    if a == "start":
        await veh.charge_start(); msg = "Started charging."
    elif a == "stop":
        await veh.charge_stop(); msg = "Stopped charging."
    elif a in ("limit", "set_limit") and limit is not None:
        await veh.set_charge_limit(percent=int(limit)); msg = f"Set the charge limit to {int(limit)}%."
    else:
        return "For charging, say start, stop, or set the limit to a percent."
    _persist(api)
    return msg


@_resilient
async def find(mode: str = "flash") -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "Couldn't reach the car."
    veh = api.vehicles.specificSigned(vin)
    if (mode or "").lower() == "honk":
        await veh.honk_horn(); msg = "Honked the horn."
    else:
        await veh.flash_lights(); msg = "Flashed the lights."
    _persist(api)
    return msg


@_resilient
async def locate() -> str:
    api, vin, awake = await _veh()
    if not awake:
        return "Couldn't reach the car to locate it."
    d = await _data(api, vin, endpoints="location_data")
    _persist(api)
    ds = d.get("drive_state", {})
    lat, lon = ds.get("latitude"), ds.get("longitude")
    if lat is None:
        return "Location isn't available (sharing may be off)."
    # best-effort reverse geocode (free, no key); fall back to a maps link
    try:
        async with _session.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lon, "format": "json"},
            headers={"User-Agent": "kronik-voice-agent/1.0"}) as r:
            name = (await r.json()).get("display_name", "")
        if name:
            return f"the car is near {', '.join(name.split(', ')[:3])}."
    except Exception:
        pass
    return f"the car is at {lat:.4f}, {lon:.4f} (maps.google.com/?q={lat},{lon})."
