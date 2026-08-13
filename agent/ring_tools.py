"""Ring camera integration (cloud API — Ring exposes NO local access, every LAN port is
filtered). Auth is an OAuth token minted once by ring_auth.py and stored in .env.local as
RING_TOKEN_JSON; the token_updater rewrites it here whenever the library refreshes it.

Exposes async helpers the agent wraps as tools: snapshot (near-live still), recent motion/
ding events, and the URL of the latest recorded clip. Camera names are fuzzy-matched so
"bedroom" resolves to whatever the device is actually called in the Ring app.
"""
import asyncio
import json
import os
import re
import time

from ring_doorbell import Auth, Ring

_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local")
_UA = "kronik-voice-agent/1.0"
_SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_snaps")
os.makedirs(_SNAP_DIR, exist_ok=True)

_ring = None
_auth = None
_lock = asyncio.Lock()
_last_update = 0.0
_UPDATE_TTL = 30.0   # re-poll device list/state at most this often


def _upsert_env(key: str, value: str):
    lines, found = [], False
    if os.path.exists(_ENV):
        with open(_ENV) as f:
            lines = f.read().splitlines()
    for i, ln in enumerate(lines):
        if ln.startswith(key + "="):
            lines[i] = f"{key}={value}"; found = True; break
    if not found:
        lines.append(f"{key}={value}")
    with open(_ENV, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(_ENV, 0o600)


def _token_updater(token: dict):
    # library refreshed the OAuth token -> persist the new one (never the password)
    try:
        _upsert_env("RING_TOKEN_JSON", json.dumps(token))
    except Exception:
        pass


async def _session() -> Ring:
    """Cached, authenticated Ring session; refreshes device data on a short TTL."""
    global _ring, _auth, _last_update
    async with _lock:
        if _ring is None:
            raw = os.environ.get("RING_TOKEN_JSON")
            if not raw:
                raise RuntimeError("Ring not linked — run: .venv/bin/python agent/ring_auth.py")
            _auth = Auth(_UA, json.loads(raw), _token_updater)
            _ring = Ring(_auth)
            await _ring.async_create_session()
            await _ring.async_update_data()
            _last_update = time.time()
        elif time.time() - _last_update > _UPDATE_TTL:
            await _ring.async_update_data()
            _last_update = time.time()
        return _ring


async def cameras() -> list:
    r = await _session()
    return list(r.video_devices())


async def camera_names() -> list:
    return [c.name for c in await cameras()]


async def resolve(name: str):
    """Fuzzy-match a camera by name ('bedroom' -> 'Bedroom Cam'). Falls back to the only
    camera if there's just one; returns None if nothing matches."""
    cams = await cameras()
    if not cams:
        return None
    if not name:
        return cams[0] if len(cams) == 1 else None
    q = name.lower().strip()
    for c in cams:
        if q == c.name.lower():
            return c
    for c in cams:
        if q in c.name.lower() or c.name.lower() in q:
            return c
    tokens = set(q.split())
    for c in cams:
        if tokens & set(c.name.lower().split()):
            return c
    return cams[0] if len(cams) == 1 else None


_SNAP_TS = "/clients_api/snapshots/timestamps"
_SNAP_IMG = "/clients_api/snapshots/image/{0}"


async def _fetch_snapshot(cam, tries: int = 6, delay: float = 1.5) -> bytes | None:
    """Robust snapshot: ask for a fresh capture, wait for it to register, then download the
    JPEG. Tolerates the empty-`timestamps` case the stock library crashes on (battery cams /
    cams with 'Snapshot Capture' off), and falls back to the last available image so we always
    return SOMETHING rather than nothing."""
    r = await _session()
    dev_id = cam.id
    payload = {"doorbot_ids": [dev_id]}
    try:
        await r.async_query(_SNAP_TS, method="POST", json=payload)
    except Exception:
        pass
    request_time = time.time()
    for _ in range(tries):
        await asyncio.sleep(delay)
        try:
            resp = await r.async_query(_SNAP_TS, method="POST", json=payload)
            stamps = resp.json().get("timestamps") or []
            fresh = stamps and stamps[0].get("timestamp", 0) / 1000 > request_time
        except Exception:
            fresh = False
        if fresh:
            break
    # download whatever image is available (fresh if the loop caught it, else the last one)
    try:
        img = await r.async_query(_SNAP_IMG.format(dev_id))
        data = img.content
        return data if data else None
    except Exception:
        return None


async def snapshot(name: str = "") -> tuple:
    """(filepath, camera_name) of a JPEG still (fresh if the cam produced one), or (None, cam)."""
    cam = await resolve(name)
    if cam is None:
        return (None, None)
    data = await _fetch_snapshot(cam)
    if not data:
        return (None, cam.name)
    path = os.path.join(_SNAP_DIR, f"{cam.id}.jpg")
    with open(path, "wb") as f:
        f.write(data)
    return (path, cam.name)


async def recent_events(name: str = "", limit: int = 5) -> tuple:
    """(events, camera_name). Each event: {kind, time (epoch), answered, recording_id}."""
    cam = await resolve(name)
    if cam is None:
        return ([], None)
    hist = await cam.async_history(limit=limit)
    out = []
    for h in hist:
        created = h.get("created_at")
        ts = created.timestamp() if hasattr(created, "timestamp") else None
        out.append({
            "kind": h.get("kind"),
            "time": ts,
            "answered": h.get("answered"),
            "recording_id": h.get("id"),
        })
    return (out, cam.name)


async def last_recording_url(name: str = "") -> tuple:
    """(url, camera_name) of the most recent recorded clip, or (None, cam). Recorded clips
    require a Ring Protect subscription — without one this returns (None, cam) instead of
    raising, so callers can fall back to a live snapshot."""
    cam = await resolve(name)
    if cam is None:
        return (None, None)
    try:
        rid = await cam.async_get_last_recording_id()
        if rid is None:
            return (None, cam.name)
        return (await cam.async_recording_url(rid), cam.name)
    except Exception:
        return (None, cam.name)   # e.g. "no active subscription"


# Vision describer = Gemma-4-E4B, served ON-DEMAND: vlm_ondemand spawns it on first use and it
# self-exits after idle (see gemma_vlm_server.py), so ~4GB stays out of RAM except when actually
# describing. Gemma is non-thinking, so no thinking control is needed.
_VLM_MODEL = os.environ.get("VLM_MODEL", "mlx-community/gemma-4-e4b-it-4bit")


async def describe_image(path: str, question: str = "", max_tokens: int = 1024) -> str | None:
    """Send a JPEG to the on-demand Gemma-4 vision service and return a text description.
    max_tokens caps the length: ~1024 for a full detailed description (the `look` tool),
    a small value (e.g. 128) for a quick gist (the motion-announcement path). The first call
    after idle spawns + loads Gemma (~5-15s); subsequent calls are fast."""
    import base64
    import aiohttp
    import vlm_ondemand
    url = await vlm_ondemand.ensure()   # bring Gemma up on demand
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    q = question or (
        "Describe this security camera image in as much detail as you can, so an assistant has "
        "full context. Cover: any people (count, what they're doing, appearance), pets, and "
        "activity or motion; the room and layout; furniture and its arrangement; objects on "
        "surfaces and the floor; anything on the walls; colors and textures; lighting and any "
        "time-of-day cues; the state of the bed/seating; and anything unusual or out of place. "
        "If the scene is empty and quiet, say so plainly.")
    body = {
        "model": _VLM_MODEL, "max_tokens": max_tokens, "temperature": 0.2,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": q}]}],
    }
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url, json=body, timeout=aiohttp.ClientTimeout(total=90)) as r:
                data = await r.json()
                return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None


async def look(name: str = "", question: str = "") -> tuple:
    """Capture a fresh snapshot AND describe it. (description, camera_name). This is the
    'what's happening in the bedroom?' path: snapshot -> Gemma-4 vision -> text for the LLM."""
    path, cam = await snapshot(name)
    if not path:
        return (None, cam)
    desc = await describe_image(path, question)
    return (desc, cam)


async def probe() -> dict:
    """One-shot connectivity check for setup: lists cameras + battery/kind."""
    cams = await cameras()
    return {
        "linked": True,
        "cameras": [{"name": c.name, "kind": c.kind, "battery": c.battery_life} for c in cams],
    }
