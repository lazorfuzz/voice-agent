"""Real-time Ring event system — modular + extensible.

Flow:  Ring push (motion/ding, via firebase)  ->  CAPTURE (fresh snapshot only)  ->  DISPATCH
       to every registered handler.  The Gemma VISION DESCRIPTION is NOT done here — it's the
       slow/heavy part, so it's deferred to the handler via `await event.describe()` (lazy +
       cached). That keeps dispatch fast and lets handlers that don't need a description (logging,
       raw-image push, automations) skip it entirely.

A handler is any `async def handler(event: MotionEvent)`. Register with `register()`; write new
ones for future use cases without touching this file. Each handler gets the RAW snapshot (bytes +
path) and the camera/kind/time, and can call `await event.describe()` if it wants the vision text.

Design notes:
  * Snapshot capture WAKES a battery cam (~5-13s), so it only runs when ≥1 handler is registered
    (silent + no camera drain otherwise) and is debounced per-camera (_DEBOUNCE) against bursts.
  * The listener holds a firebase push connection; credentials are persisted to .env.local so it
    doesn't re-register every start.
"""
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field

import ring_tools
from ring_doorbell.listen import RingEventListener

_log = logging.getLogger("ring_events")

_DEBOUNCE = 25.0            # seconds between snapshots per camera (battery + spam guard)
_CAPTURE_KINDS = ("motion", "ding")   # event kinds worth a snapshot
_MAX_EVENT_AGE = 30.0        # FCM replays queued pushes when the listener reconnects


@dataclass
class MotionEvent:
    camera: str                     # e.g. "Bedroom"
    kind: str                       # "motion" | "ding" | ...
    timestamp: float                # epoch seconds
    snapshot_path: str | None       # freshly captured JPEG on disk (None if capture failed/debounced)
    snapshot_bytes: bytes | None    # the raw JPEG (same image)
    _desc: str | None = field(default=None, repr=False)
    _desc_done: bool = field(default=False, repr=False)

    async def describe(self, question: str = "", max_tokens: int = 1024) -> str | None:
        """0GM vision description of the snapshot — LAZY + cached. The event pipeline does not
        run this (kept off the hot path); a handler calls it only if it wants the text. Returns
        None if there's no snapshot. Multiple handlers share one describe call via the cache."""
        if not self._desc_done:
            self._desc_done = True
            if self.snapshot_path:
                self._desc = await ring_tools.describe_image(self.snapshot_path, question, max_tokens=max_tokens)
        return self._desc


# ---- handler registry (the "hook system") ----
_HANDLERS: list = []


def register(handler) -> None:
    """Add an async `handler(event: MotionEvent)`. Idempotent."""
    if handler not in _HANDLERS:
        _HANDLERS.append(handler)


def unregister(handler) -> None:
    if handler in _HANDLERS:
        _HANDLERS.remove(handler)


def handler_count() -> int:
    return len(_HANDLERS)


# ---- capture + dispatch (fast: snapshot only, NO description) ----
_last_snap: dict = {}


async def _capture(device_name: str, kind: str) -> MotionEvent:
    now = time.time()
    path = bytes_ = None
    if now - _last_snap.get(device_name, 0) > _DEBOUNCE:
        _last_snap[device_name] = now
        try:
            path, _cam = await ring_tools.snapshot(device_name)
            if path:
                with open(path, "rb") as f:
                    bytes_ = f.read()
        except Exception:
            _log.exception("ring_events: snapshot failed for %s", device_name)
    return MotionEvent(camera=device_name, kind=kind, timestamp=now,
                       snapshot_path=path, snapshot_bytes=bytes_)


async def _dispatch(device_name: str, kind: str) -> None:
    event = await _capture(device_name, kind)   # fast — no Gemma; handlers describe() on demand
    for h in list(_HANDLERS):
        try:
            await h(event)
        except Exception:
            _log.exception("ring_events: handler %r failed", getattr(h, "__name__", h))


# ---- listener (firebase push) ----
_listener = None
_loop = None
_lock = asyncio.Lock()


def _on_ring_event(ev) -> None:
    """Called by the firebase listener (possibly off-loop). Filter, then schedule async work.
    Skips entirely when no handler is registered → silent + no camera wake."""
    if not _HANDLERS or getattr(ev, "kind", None) not in _CAPTURE_KINDS:
        return
    # The FCM client can deliver its backlog immediately after reconnecting. Those
    # messages retain the Ring event's original epoch timestamp in ``now``; without
    # this guard an event from an hour ago wakes the camera and is announced as if it
    # just happened. Drop it before snapshot capture (and before battery drain).
    event_time = getattr(ev, "now", None)
    if isinstance(event_time, (int, float)):
        age = time.time() - event_time
        if age > _MAX_EVENT_AGE:
            _log.info(
                "ring_events: dropped stale %s from %s (age=%.1fs)",
                getattr(ev, "kind", "event"), getattr(ev, "device_name", "?"), age,
            )
            return
    if _loop is not None:
        asyncio.run_coroutine_threadsafe(_dispatch(ev.device_name, ev.kind), _loop)


async def start() -> bool:
    """Start the push listener (idempotent). Reuses the authed Ring session from ring_tools and
    persists the firebase registration to .env.local (RING_LISTEN_CREDENTIALS)."""
    global _listener, _loop
    async with _lock:
        if _listener is not None:
            return True
        ring = await ring_tools._session()
        raw = os.environ.get("RING_LISTEN_CREDENTIALS")
        creds = json.loads(raw) if raw else None

        def _save_creds(c):
            try:
                ring_tools._upsert_env("RING_LISTEN_CREDENTIALS", json.dumps(c))
                os.environ["RING_LISTEN_CREDENTIALS"] = json.dumps(c)
            except Exception:
                _log.exception("ring_events: failed to persist listener credentials")

        listener = RingEventListener(ring, creds, _save_creds)
        listener.add_notification_callback(_on_ring_event)
        _loop = asyncio.get_running_loop()
        ok = await listener.start(timeout=15)
        if ok:
            _listener = listener
            _log.info("ring_events: push listener started")
        else:
            _log.warning("ring_events: push listener failed to start")
        return bool(ok)


async def stop() -> None:
    global _listener
    async with _lock:
        if _listener is not None:
            try:
                await _listener.stop()
            except Exception:
                pass
            _listener = None
