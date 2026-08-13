"""Gemma Ears rtc agent — joins LiveKit room "gemma"; user speech is segmented by energy
gate and sent as raw AUDIO to the ears server (:8084, gemma-4-12b hears it directly — no
STT); the reply text is spoken with pocket-tts. Experimental UI mode ("🧪 Gemma Ears").

Run (voice-agent venv):  python gemma_ears_agent.py
"""
import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib.request

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("gemma-ears")

ROOM = os.environ.get("EARS_ROOM", "gemma")
EARS_URL = "http://127.0.0.1:8084"
SR = 24000                    # room audio + pocket-tts rate
ENERGY_GATE = 0.006
BARGE_GATE = 0.015            # while the agent speaks: echo leakage is quiet, real
                              # interruptions are not — quiet chunks count for nothing
UTT_END_SIL = 0.6
UTT_MIN_SPEECH = 0.35
GREETING = "Hey! Kronik here — gemma ears mode. What can I do for you?"
VOICE = os.environ.get("POCKET_VOICE", "jean")


def load_env():
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     "agent", ".env.local")
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v)


def ears(path, payload):
    req = urllib.request.Request(EARS_URL + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=90))


def ears_stream(payload, on_text):
    """POST /utterance and stream reply text chunks to on_text as they generate.
    Returns the trailing meta dict (after the \x1e sentinel)."""
    req = urllib.request.Request(EARS_URL + "/utterance", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    tail = ""
    meta = {}
    in_meta = False
    while True:
        chunk = resp.read1(4096) if hasattr(resp, "read1") else resp.read(4096)
        if not chunk:
            break
        text = chunk.decode(errors="ignore")
        if in_meta:
            tail += text
            continue
        if "\x1e" in text:
            spoken, _, rest = text.partition("\x1e")
            if spoken:
                on_text(spoken)
            tail = rest
            in_meta = True
        else:
            on_text(text)
    try:
        meta = json.loads(tail.strip() or "{}")
    except Exception:
        meta = {}
    return meta


async def main():
    load_env()
    from livekit import rtc, api
    from pocket_tts import TTSModel

    log.info("loading pocket-tts...")
    tts_model = TTSModel.load_model()
    tts_state = tts_model.get_state_for_audio_prompt(VOICE)
    log.info("tts ready; checking ears server...")
    urllib.request.urlopen(EARS_URL + "/health", timeout=5)

    token = (api.AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
             .with_identity("gemma-ears").with_name("Gemma Ears")
             .with_grants(api.VideoGrants(room_join=True, room=ROOM)).to_jwt())
    room = rtc.Room()
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("gemma-voice", source)

    utt_q: asyncio.Queue = asyncio.Queue()
    speaking = {"n": 0}            # >0 while TTS frames are being captured
    interrupt = asyncio.Event()    # set on barge-in: stop playback + drop queued sentences

    async def pump(rtc_track):
        stream = rtc.AudioStream(rtc_track)
        buf = np.zeros(0, dtype=np.float32)
        utt = []
        speech_s = sil_s = 0.0
        async for ev in stream:
            fr = ev.frame
            d = np.frombuffer(fr.data, dtype=np.int16).astype(np.float32) / 32768.0
            if fr.num_channels > 1:
                d = d.reshape(-1, fr.num_channels).mean(axis=1)
            if fr.sample_rate != 16000:
                n = int(len(d) * 16000 / fr.sample_rate)
                d = np.interp(np.linspace(0, len(d), n, endpoint=False),
                              np.arange(len(d)), d).astype(np.float32)
            buf = np.concatenate([buf, d])
            while len(buf) >= 1600:                      # 100ms @16k
                chunk, buf = buf[:1600], buf[1600:]
                rms = float(np.sqrt(np.mean(chunk ** 2)))
                utt.append((chunk, rms))
                gate = BARGE_GATE if speaking["n"] else ENERGY_GATE
                if rms >= gate:
                    if speaking["n"] and speech_s == 0.0:
                        log.info("speech onset while agent speaking")
                    speech_s += 0.1; sil_s = 0.0
                    if speaking["n"] and speech_s >= 0.3 and not interrupt.is_set():
                        log.info("barge-in detected")
                        interrupt.set()
                else:
                    sil_s += 0.1
                if speech_s >= UTT_MIN_SPEECH and sil_s >= UTT_END_SIL:
                    # trim to 0.3s before first speech and 0.2s after last speech —
                    # leading silence is pure prefill cost (25 audio tok/s)
                    first = next((i for i, (_, r) in enumerate(utt) if r >= ENERGY_GATE), 0)
                    u = np.concatenate([c for c, _ in
                                        utt[max(0, first - 3):len(utt) - int((sil_s - 0.4) * 10)]])
                    if len(u) > 12 * 16000:
                        # noise blips can keep the buffer alive across long pauses; a
                        # giant clip (~650 audio tokens) destabilizes the model
                        log.info("capping %.1fs utterance to last 12s", len(u) / 16000)
                        u = u[-12 * 16000:]
                    log.info("dispatch utt: %.2fs (raw %.2fs) speech=%.1fs",
                             len(u) / 16000, len(utt) * 0.1, speech_s)
                    utt_q.put_nowait(u)
                    utt = []; speech_s = sil_s = 0.0
                elif sil_s > 3.0 and not speech_s:
                    utt = []; sil_s = 0.0

    active = {"pump": None}

    async def transcribe(text, *, as_user=False, writer=None, close=True):
        """Publish a chat bubble via the lk.transcription text-stream topic (what the
        UI renders). sender_identity=user makes it a right-side user bubble."""
        try:
            w = writer or await room.local_participant.stream_text(
                topic="lk.transcription",
                attributes={"kronik.role": "user" if as_user else "agent"})
            if text:
                await w.write(text)
            if close:
                await w.aclose()
                return None
            return w
        except Exception as e:
            log.warning("transcribe failed: %s", e)
            return None

    @room.on("track_subscribed")
    def on_track(rtc_track, pub, part):
        if rtc_track.kind == rtc.TrackKind.KIND_AUDIO:
            log.info("subscribed to %s", part.identity)
            active["user_id"] = part.identity
            if active["pump"] and not active["pump"].done():
                active["pump"].cancel()   # one mic at a time — stale pumps double-reply
            try:
                ears("/reset", {})
            except Exception:
                pass
            active["pump"] = asyncio.create_task(pump(rtc_track))
            asyncio.create_task(say(GREETING))
            asyncio.create_task(transcribe(GREETING))

    @room.on("track_unsubscribed")
    def on_untrack(rtc_track, pub, part):
        if active["pump"] and not active["pump"].done():
            active["pump"].cancel()
            active["pump"] = None

    async def say(text):
        # stream TTS chunks straight into the track — playback starts on the first
        # synthesized chunk instead of after the whole sentence
        t_s = time.time()
        chunk_q: asyncio.Queue = asyncio.Queue()
        loop2 = asyncio.get_running_loop()

        def synth():
            for a in tts_model.generate_audio_stream(tts_state, text):
                loop2.call_soon_threadsafe(chunk_q.put_nowait, np.asarray(a).reshape(-1))
            loop2.call_soon_threadsafe(chunk_q.put_nowait, None)

        synth_task = asyncio.create_task(asyncio.to_thread(synth))
        n = SR // 100
        frame = rtc.AudioFrame.create(SR, 1, n)
        fb = np.frombuffer(frame.data, dtype=np.int16)
        rem = np.zeros(0, dtype=np.int16)
        first = True
        speaking["n"] += 1
        try:
            barged = False
            while not barged:
                a = await chunk_q.get()
                if a is None:
                    break
                if first:
                    log.info("say: first chunk %.0fms for %r", (time.time() - t_s) * 1000, text[:40])
                    first = False
                pcm = np.concatenate([rem, (np.clip(a.astype(np.float32), -1, 1) * 32767).astype(np.int16)])
                usable = (len(pcm) // n) * n
                for i in range(0, usable, n):
                    if interrupt.is_set():
                        barged = True
                        source.clear_queue()   # drop the ~1s of already-buffered audio
                        break
                    fb[:] = pcm[i:i + n]
                    await source.capture_frame(frame)
                rem = pcm[usable:]
            if len(rem) and not barged:
                fb[:] = np.pad(rem, (0, n - len(rem)))
                await source.capture_frame(frame)
        except Exception as e:
            log.warning("playback interrupted: %s", e)
        finally:
            speaking["n"] -= 1
            await synth_task

    await room.connect(os.environ["LIVEKIT_URL"], token)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    log.info("connected to room %r", ROOM)

    loop = asyncio.get_running_loop()
    while True:
        pcm = await utt_q.get()
        interrupt.clear()
        t0 = time.time()
        say_q: asyncio.Queue = asyncio.Queue()
        user_wr = await transcribe("", as_user=True, close=False)
        agent_wr = {"w": None}

        async def speaker():
            # speak sentences as they arrive; synthesis of sentence N overlaps playback of N-1
            while True:
                s = await say_q.get()
                if s is None:
                    return
                if interrupt.is_set():
                    continue               # barged: swallow the rest of this reply
                if agent_wr["w"] is None:
                    agent_wr["w"] = await transcribe(None, close=False)
                if agent_wr["w"]:
                    await transcribe(s + " ", writer=agent_wr["w"], close=False)
                await say(s)

        sp_task = asyncio.create_task(speaker())
        acc = {"buf": "", "first": None, "s1": None}

        def on_text(chunk):
            if interrupt.is_set():
                return
            if acc["first"] is None:
                acc["first"] = time.time() - t0
            acc["buf"] += chunk
            # flush complete sentences to the speaker
            while True:
                # before anything is spoken, a comma-clause is enough to start TTS early;
                # after that, flush only on sentence enders (keeps prosody natural)
                pat = r"[.!?,]\s" if acc.get("s1") is None else r"[.!?]\s"
                m2 = re.search(pat, acc["buf"])
                if not m2:
                    break
                sentence = acc["buf"][:m2.end()].strip()
                acc["buf"] = acc["buf"][m2.end():]
                if sentence:
                    if acc.get("s1") is None:
                        acc["s1"] = time.time() - t0
                        log.info("first sentence flushed at %.0fms: %r",
                                 acc["s1"] * 1000, sentence[:40])
                    loop.call_soon_threadsafe(say_q.put_nowait, sentence)

        try:
            meta = await asyncio.to_thread(
                ears_stream,
                {"audio_b64": base64.b64encode(pcm.astype(np.float32).tobytes()).decode()},
                on_text)
        except Exception as e:
            log.warning("ears error: %s", e)
            meta = {}
        if acc["buf"].strip():
            say_q.put_nowait(acc["buf"].strip())
        say_q.put_nowait(None)
        await sp_task
        await transcribe(meta.get("heard") or "…", as_user=True, writer=user_wr)
        if agent_wr["w"]:
            await transcribe(None, writer=agent_wr["w"])
        log.info("heard=%r tool=%s first_text=%.0fms total=%.0fms",
                 meta.get("heard"), meta.get("tool"),
                 (acc["first"] or 0) * 1000, (time.time() - t0) * 1000)


asyncio.run(main())
