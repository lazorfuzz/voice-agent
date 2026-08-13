"""Live-view WebRTC broker for Ring cameras.

Ring live view is cloud WebRTC. The `ring-doorbell` library handles the Ring-side signalling
(ticket -> its own WS to Ring -> answer + ICE via a callback). This service is the thin bridge
between a BROWSER WebSocket and that library, so the browser's RTCPeerConnection ends up talking
directly to Ring's media servers and shows the live feed.

Browser protocol over the WS (`/live/ws?camera=bedroom`):
  browser -> {type:"offer", sdp}                     (start)
  server  -> {type:"answer", sdp}                    (Ring's answer)
  server  -> {type:"ice", candidate, sdpMLineIndex}  (Ring's trickled ICE)
  browser -> {type:"ice", candidate, sdpMLineIndex}  (browser's trickled ICE)
  browser -> {type:"stop"}                           (teardown)

GET `/live/ice` returns the ICE servers to configure the browser peer connection with.
Runs on :8791 (its own async process; token_server is sync http.server and can't do WS).
"""
import asyncio
import logging
import os
import uuid

from aiohttp import web, WSMsgType

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
# surface Ring's WebRTC signalling (answer/ICE/errors) so a browser test is debuggable
logging.getLogger("ring_doorbell.webrtcstream").setLevel(logging.DEBUG)
_log = logging.getLogger("ring_live")

try:
    from dotenv import load_dotenv
    load_dotenv(".env.local")
except ImportError:
    pass

import ring_tools
from ring_doorbell.webrtcstream import RingWebRtcMessage

_KEEPALIVE_S = 30


async def _ice(request: web.Request) -> web.Response:
    try:
        ring = await ring_tools._session()
        servers = ring.get_ice_servers()
    except Exception:
        servers = ["stun:stun.l.google.com:19302"]
    return web.json_response(
        {"iceServers": [{"urls": servers}]},
        headers={"Access-Control-Allow-Origin": "*"})


async def _ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(heartbeat=25)
    await ws.prepare(request)

    cam = await ring_tools.resolve(request.query.get("camera", ""))
    if cam is None:
        await ws.send_json({"type": "error", "message": "camera not found"})
        await ws.close()
        return ws

    loop = asyncio.get_running_loop()
    outbox: asyncio.Queue = asyncio.Queue()
    session_id = uuid.uuid4().hex
    started = False

    def on_message(msg: RingWebRtcMessage) -> None:
        # called from the library's reader task (same loop): forward to the browser
        if msg.answer is not None:
            outbox.put_nowait({"type": "answer", "sdp": msg.answer})
        if msg.candidate is not None:
            outbox.put_nowait({"type": "ice", "candidate": msg.candidate,
                               "sdpMLineIndex": msg.sdp_m_line_index or 0})

    async def pump_outbox():
        while not ws.closed:
            item = await outbox.get()
            if item is None:
                return
            await ws.send_json(item)

    async def keepalive():
        while not ws.closed:
            await asyncio.sleep(_KEEPALIVE_S)
            try:
                await cam.keep_alive_webrtc_stream(session_id)
            except Exception:
                return

    pump_task = asyncio.create_task(pump_outbox())
    ka_task = None
    try:
        async for msg in ws:
            if msg.type != WSMsgType.TEXT:
                continue
            data = msg.json()
            t = data.get("type")
            if t == "offer" and not started:
                started = True
                _log.info("live view start: camera=%s session=%s", cam.name, session_id)
                # generate() blocks until Ring answers, so run it as a task; the answer +
                # ICE arrive via on_message and get pumped to the browser meanwhile.
                asyncio.create_task(
                    cam.generate_async_webrtc_stream(data["sdp"], session_id, on_message))
                ka_task = asyncio.create_task(keepalive())
            elif t == "ice" and data.get("candidate"):
                try:
                    await cam.on_webrtc_candidate(
                        session_id, data["candidate"], int(data.get("sdpMLineIndex") or 0))
                except Exception:
                    pass
            elif t == "stop":
                break
    finally:
        _log.info("live view closed: session=%s", session_id)
        outbox.put_nowait(None)
        pump_task.cancel()
        if ka_task:
            ka_task.cancel()
        try:
            await cam.close_webrtc_stream(session_id)
        except Exception:
            pass
    return ws


def main():
    app = web.Application()
    app.add_routes([web.get("/live/ice", _ice), web.get("/live/ws", _ws)])
    web.run_app(app, host="127.0.0.1", port=8791)


if __name__ == "__main__":
    main()
