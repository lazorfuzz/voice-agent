"""SoulX-Duplug as a LiveKit streaming turn detector (POC, flag-gated: TURN_DETECTOR=soulx).

SoulX-Duplug (Soul-AILab, 0.6B, Apache 2.0) is a TRAINED full-duplex turn-taking model:
it streams 16k audio in 160ms chunks and emits dialogue states — 'nonidle' (speech),
'speak' (utterance semantically COMPLETE -> commit the turn now), 'idle'. Unlike the
text-based EnglishModel EOU (which waits ~620ms of silence then scores the transcript),
SoulX decides from audio+semantics with a ~250ms design latency and holds on genuinely
incomplete utterances ("Can you set the thermostat to ...").

Server: the ported repo at ~/soulx-poc (MPS; see memory soulx-duplug.md) run via
  cd ~/soulx-poc && ./.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765
This adapter implements livekit's _StreamingTurnDetector protocol: push_audio() feeds a
websocket; predict() resolves a TurnDetectionEvent from SoulX's states:
  'speak' seen   -> end_of_turn_probability 1.0  (commit)
  'nonidle'      -> 0.0 (user still talking; framework re-asks later)
  timeout/idle   -> stay pending; the framework's max_endpointing_delay is the fallback.
"""
import asyncio
import base64
import json
import logging
import os
import time
import uuid

import numpy as np

from livekit import rtc
from livekit.agents.voice.turn import TurnDetectionEvent

log = logging.getLogger("kronik.soulx")

SOULX_URL = os.environ.get("SOULX_URL", "ws://127.0.0.1:8765/turn")
CHUNK = 2560                      # 160ms @ 16k — SoulX's native chunk
_SPEAK_FRESH_S = 2.0              # a 'speak' this recent counts for the next predict()


class SoulXTurnDetector:
    """_StreamingTurnDetector: one instance per session; stream() opens the transport."""

    def __init__(self):
        # While the agent is thinking/speaking, SoulX's per-chunk inference is pure GPU
        # contention against LLM prefill + TTS (its states aren't consulted then — barge-in
        # is VAD-driven). agent.py's agent_state handler closes this gate for those windows.
        self._gate = {"closed": False}

    def set_gated(self, closed: bool):
        self._gate["closed"] = bool(closed)

    @property
    def model(self) -> str:
        return "soulx-duplug-0.6b"

    @property
    def provider(self) -> str:
        return "local"

    def stream(self, *, conn_options=None):
        return SoulXStream(self._gate)


class SoulXStream:
    def __init__(self, gate=None):
        self._gate = gate or {"closed": False}
        self._session_id = f"kronik-{uuid.uuid4().hex[:8]}"
        self._buf = np.zeros(0, dtype=np.float32)
        self._loop = asyncio.get_event_loop()
        self._send_q: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._pred_fut: asyncio.Future | None = None
        self._last_state = "idle"
        self._last_speak_t = 0.0
        self._speak_consumed = True
        self._last_audio_t = time.time()
        self._closed = False
        self._task = asyncio.create_task(self._run())

    # ---- protocol properties ----
    @property
    def model(self) -> str:
        return "soulx-duplug-0.6b"

    @property
    def provider(self) -> str:
        return "local"

    @property
    def is_fallback(self) -> bool:
        return False

    @property
    def prediction_timeout(self) -> float:
        return 1.5        # SoulX decides ~240-500ms after true end; leave margin

    async def unlikely_threshold(self, language) -> float | None:
        return 0.5

    async def backchannel_threshold(self, language) -> float | None:
        return None       # SoulX server API doesn't expose backchannel probability

    async def supports_language(self, language) -> bool:
        return True       # bilingual en/zh; Kronik runs en

    # ---- transport ----
    async def _run(self):
        import aiohttp
        try:
            async with aiohttp.ClientSession() as http:
                async with http.ws_connect(SOULX_URL, heartbeat=20) as ws:
                    log.info("soulx connected (%s)", self._session_id)

                    async def sender():
                        while True:
                            payload = await self._send_q.get()
                            if payload is None:
                                return
                            await ws.send_str(payload)

                    send_task = asyncio.create_task(sender())
                    try:
                        async for msg in ws:
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                continue
                            data = json.loads(msg.data)
                            if data.get("type") != "turn_state":
                                continue
                            st = (data.get("state") or {}).get("state", "idle")
                            self._on_state(st)
                    finally:
                        send_task.cancel()
        except Exception as e:
            if not self._closed:
                log.warning("soulx transport ended: %s", e)

    def _on_state(self, st: str):
        if st == "blank":
            return
        if st != self._last_state:
            log.info("SOULX state=%s", st)
        self._last_state = st
        if st == "speak":
            self._last_speak_t = time.time()
            self._speak_consumed = False
        # resolve a pending predict() as soon as we have a decisive signal
        fut = self._pred_fut
        if fut is not None and not fut.done():
            if st == "speak":
                self._speak_consumed = True
                fut.set_result(self._event(1.0))
                self._pred_fut = None
            elif st == "nonidle":
                fut.set_result(self._event(0.0))
                self._pred_fut = None

    def _event(self, prob: float) -> TurnDetectionEvent:
        return TurnDetectionEvent(
            type="eot_prediction",
            end_of_turn_probability=prob,
            last_speaking_time=self._last_audio_t,
            detection_delay=None,
            backchannel_probability=None,
        )

    # ---- protocol methods ----
    def push_audio(self, frame: rtc.AudioFrame) -> None:
        if self._closed:
            return
        if self._gate["closed"]:
            self._buf = np.zeros(0, dtype=np.float32)   # drop; no stale partial chunks later
            return
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)
        if frame.sample_rate != 16000:
            n = int(len(data) * 16000 / frame.sample_rate)
            data = np.interp(np.linspace(0, len(data), n, endpoint=False),
                             np.arange(len(data)), data).astype(np.float32)
        self._buf = np.concatenate([self._buf, data])
        self._last_audio_t = time.time()
        while len(self._buf) >= CHUNK:
            chunk, self._buf = self._buf[:CHUNK], self._buf[CHUNK:]
            payload = json.dumps({
                "type": "audio", "session_id": self._session_id,
                "audio": base64.b64encode(chunk.tobytes()).decode(),
            })
            try:
                self._send_q.put_nowait(payload)
            except asyncio.QueueFull:
                pass      # transport behind; drop rather than lag the pipeline

    def predict(self) -> asyncio.Future:
        fut: asyncio.Future = self._loop.create_future()
        # a fresh, unconsumed 'speak' answers immediately (SoulX often decides BEFORE
        # the framework's VAD asks)
        if not self._speak_consumed and time.time() - self._last_speak_t < _SPEAK_FRESH_S:
            self._speak_consumed = True
            fut.set_result(self._event(1.0))
            return fut
        if self._pred_fut is not None and not self._pred_fut.done():
            self._pred_fut.cancel()
        self._pred_fut = fut
        return fut

    def cancel_inference(self, *, timed_out: bool = False) -> None:
        if self._pred_fut is not None and not self._pred_fut.done():
            self._pred_fut.cancel()
        self._pred_fut = None

    def flush(self, reason: str | None = None) -> None:
        self._speak_consumed = True         # discard any stale decision

    def end_input(self) -> None:
        self.flush()

    async def aclose(self) -> None:
        self._closed = True
        try:
            self._send_q.put_nowait(None)
        except Exception:
            pass
        self._task.cancel()
        try:
            await self._task
        except (asyncio.CancelledError, Exception):
            pass
