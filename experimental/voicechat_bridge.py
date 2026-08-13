"""Nemotron VoiceChat bridge — LiveKit room agent for NVIDIA's end-to-end S2S model.

Joins room "voicechat" and bridges it to a NemotronLabs VoiceChat container (remote GPU
box, reached through an SSH tunnel) over its OpenAI-Realtime-compatible WebSocket. The
model IS the whole voice pipeline — ASR, turn-taking, barge-in, LLM, TTS in one graph —
so this bridge only moves audio and executes function calls locally:

  mic frames (24k PCM16) --80ms chunks--> input_audio_buffer.append
  response.output_audio.delta --24k PCM16--> room track
  response.function_call_arguments.done -> run local tool -> function_call_output
  transcripts (both sides) -> lk.transcription streams (kronik.role tagged)

Run (voice-agent venv):  python experimental/voicechat_bridge.py
Env: VOICECHAT_WS (default ws://127.0.0.1:9443/v1/realtime — the SSH tunnel).
"""
import asyncio
import base64
import json
import logging
import os
import sys
import time
import uuid

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("voicechat-bridge")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "agent"))

ROOM = os.environ.get("VOICECHAT_ROOM", "voicechat")
WS_URL = os.environ.get("VOICECHAT_WS", "ws://127.0.0.1:9443/v1/realtime")
SR = 24000
CHUNK = int(SR * 0.08)               # 80ms — the API's recommended chunk

# VOICECHAT_PROMPT= (empty) sends no instructions -> the model's default persona.
_DEFAULT_PROMPT = ("You are Kronik, a friendly, concise smart-home voice assistant. Keep replies "
                   "short and conversational. You control smart lights and a thermostat via tools; "
                   "for device requests call the matching tool and confirm briefly. If the user is "
                   "silent, stay silent and wait — never re-introduce yourself or repeat your "
                   "capabilities unprompted.")
SYSTEM = os.environ.get("VOICECHAT_PROMPT", _DEFAULT_PROMPT) or None

TOOLS = [
    {"type": "function", "name": "control_lights",
     "description": "Control the smart lights: on, off, brightness, or color.",
     "parameters": {"type": "object", "properties": {
         "action": {"type": "string"}, "brightness": {"type": "integer"},
         "color": {"type": "string"}, "which": {"type": "string"}},
         "required": ["action"]}},
    {"type": "function", "name": "thermostat_status",
     "description": "Get the thermostat's current temperature, humidity, and mode.",
     "parameters": {"type": "object", "properties": {}}},
    {"type": "function", "name": "set_thermostat_temperature",
     "description": "Set the thermostat target temperature in Fahrenheit.",
     "parameters": {"type": "object", "properties": {"temperature": {"type": "integer"}},
                    "required": ["temperature"]}},
]


def run_tool(name, args):
    """Execute a tool locally (bridge runs in the agent venv with agent/.env.local)."""
    try:
        if name == "control_lights":
            import wiz_tools
            return str(wiz_tools.control(args.get("action"), args.get("brightness"),
                                         args.get("color"), args.get("which")))
        if name == "thermostat_status":
            import nest_tools
            return str(nest_tools.status())
        if name == "set_thermostat_temperature":
            import nest_tools
            return str(nest_tools.set_temperature(int(args["temperature"])))
    except Exception as e:
        return f"Tool failed: {e}"
    return f"Unknown tool {name}."


def load_env():
    for line in open(os.path.join(_ROOT, "agent", ".env.local")):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def _ev(t, **kw):
    return json.dumps({"type": t, "event_id": str(uuid.uuid4()), **kw})


async def main():
    load_env()
    import websockets
    from livekit import rtc, api

    token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity("voicechat-bridge").with_name("Nemotron VoiceChat")
             .with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt())
    room = rtc.Room()
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("voicechat-voice", source)
    n = SR // 100

    async def transcribe(text, as_user=False):
        try:
            w = await room.local_participant.stream_text(
                topic="lk.transcription",
                attributes={"kronik.role": "user" if as_user else "agent"})
            await w.write(text)
            await w.aclose()
        except Exception:
            pass

    sess = {"task": None}

    async def session(rtc_track, ident):
        """One model session per caller: fresh WS on join, torn down on leave/drop.
        The previous design held one eternal WS — the server closes idle sockets
        (~15-20 min) and the whole bridge died with it (seen live 2026-08-04)."""
        log.info("session start for %s", ident)
        ws = await websockets.connect(WS_URL, max_size=16 * 1024 * 1024)
        pump_task = None
        # last_solicit = last moment the user "asked for" a response (voiced audio or
        # a tool result we submitted). Responses created outside this window are the
        # model's unsolicited silence-monologue -> suppressed from playback.
        st = {"last_solicit": time.time(), "suppress": False, "last_delta_ts": 0.0}
        try:
            hello = json.loads(await ws.recv())
            log.info("server: %s", hello.get("type"))
            await ws.send(_ev("session.update", session={
                "audio": {"input": {"format": {"type": "audio/pcm", "rate": SR}},
                          "output": {"format": {"type": "audio/pcm", "rate": SR}}},
                **({"instructions": SYSTEM} if SYSTEM else {}),
                "tools": TOOLS}))

            async def pump():
                # Continuous uplink, no gating: input-side muting proved unable to stop
                # the model's unprompted silence-monologue (server generates without
                # input) and a mid-response mute once froze a real answer. The bridge
                # instead filters DOWNLINK: only responses solicited by recent user
                # speech (or a tool result) get played — see response.created handler.
                stream = rtc.AudioStream(rtc_track)
                buf = np.zeros(0, dtype=np.int16)
                async for ev in stream:
                    fr = ev.frame
                    d = np.frombuffer(fr.data, dtype=np.int16)
                    if fr.num_channels > 1:
                        d = d.reshape(-1, fr.num_channels).mean(axis=1).astype(np.int16)
                    if fr.sample_rate != SR:
                        m = int(len(d) * SR / fr.sample_rate)
                        d = np.interp(np.linspace(0, len(d), m, endpoint=False),
                                      np.arange(len(d)), d.astype(np.float32)).astype(np.int16)
                    buf = np.concatenate([buf, d])
                    while len(buf) >= CHUNK:
                        chunk, buf = buf[:CHUNK], buf[CHUNK:]
                        rms = float(np.sqrt(np.mean((chunk.astype(np.float32) / 32768) ** 2)))
                        if rms > 0.004:
                            st["last_solicit"] = time.time()
                        await ws.send(_ev("input_audio_buffer.append",
                                          audio=base64.b64encode(chunk.tobytes()).decode()))
            pump_task = asyncio.create_task(pump())
            frame = rtc.AudioFrame.create(SR, 1, n)
            fb = np.frombuffer(frame.data, dtype=np.int16)
            rem = np.zeros(0, dtype=np.int16)

            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                t = msg.get("type", "")
                if t == "response.output_audio.delta":
                    # The server emits response.created ONCE per session (greeting only) —
                    # later responses are bare delta bursts. Infer a response boundary
                    # from a >=1s delta gap, and gate THAT burst on solicitation:
                    # solicited answers start within ~2.5s of user speech / tool result;
                    # the silence-monologue starts at 4s+ with nothing from the user.
                    now = time.time()
                    if now - st["last_delta_ts"] > 1.0:
                        st["suppress"] = now - st["last_solicit"] > 4.0
                        if st["suppress"]:
                            log.info("suppressing unsolicited burst (solicit_age=%.1fs)",
                                     now - st["last_solicit"])
                    st["last_delta_ts"] = now
                    if st["suppress"]:
                        continue
                    pcm = np.frombuffer(base64.b64decode(msg.get("delta", "")), dtype=np.int16)
                    pcm = np.concatenate([rem, pcm])
                    usable = (len(pcm) // n) * n
                    for i in range(0, usable, n):
                        fb[:] = pcm[i:i + n]
                        await source.capture_frame(frame)
                    rem = pcm[usable:]
                elif t == "input_audio_buffer.speech_started":
                    # cut only ALREADY-BUFFERED playback; never drop future deltas.
                    # The model is full duplex: speech_started fires for CONTINUED user
                    # speech after response.created, and a client-side drop flag then
                    # discarded entire legitimate answers (heard live 2026-08-04: model
                    # answered per transcripts, user heard nothing). Interruption
                    # handling is the SERVER's job — it stops sending deltas itself.
                    rem = np.zeros(0, dtype=np.int16)
                    try:
                        source.clear_queue()
                    except Exception:
                        pass
                elif t == "response.function_call_arguments.done":
                    name = msg.get("name", "")
                    call_id = msg.get("call_id", "")
                    try:
                        args = json.loads(msg.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    log.info("TOOL %s %s", name, args)
                    result = await asyncio.to_thread(run_tool, name, args)
                    log.info("TOOL result: %s", str(result)[:120])
                    st["last_solicit"] = time.time()   # the follow-up answer is solicited
                    await ws.send(_ev("conversation.item.create",
                                      item={"type": "function_call_output",
                                            "call_id": call_id, "output": str(result)}))
                elif t == "conversation.item.input_audio_transcription.completed":
                    txt = (msg.get("transcript") or "").strip()
                    if txt:
                        log.info("user: %s", txt)
                        await transcribe(txt, as_user=True)
                elif t == "response.output_audio_transcript.done":
                    txt = (msg.get("transcript") or "").strip()
                    if txt:
                        if st["suppress"]:
                            log.info("agent (suppressed): %s", txt)
                        else:
                            log.info("agent: %s", txt)
                            await transcribe(txt)
                elif t == "error":
                    log.warning("server error: %s", msg)
                elif t == "session.end":
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.warning("session error: %r", e)
        finally:
            if pump_task:
                pump_task.cancel()
            try:
                await ws.close()
            except Exception:
                pass
            try:
                source.clear_queue()
            except Exception:
                pass
            log.info("session closed for %s", ident)

    @room.on("track_subscribed")
    def on_track(rtc_track, pub, part):
        ident = part.identity or ""
        if rtc_track.kind != rtc.TrackKind.KIND_AUDIO or ident.startswith(("agent-", "voicechat-")):
            return
        if sess["task"] and not sess["task"].done():
            sess["task"].cancel()
        log.info("mic bound to %s", ident)
        sess["task"] = asyncio.create_task(session(rtc_track, ident))

    @room.on("participant_disconnected")
    def on_gone(part):
        if (part.identity or "") and sess["task"] and not sess["task"].done():
            log.info("%s left — closing session", part.identity)
            sess["task"].cancel()

    await room.connect(os.environ["LIVEKIT_URL"], token)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    log.info("connected to room %r; bridging to %s", ROOM, WS_URL)
    await asyncio.Event().wait()          # serve forever; sessions come and go


asyncio.run(main())
