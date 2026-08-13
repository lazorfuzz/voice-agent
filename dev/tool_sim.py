import asyncio, json, urllib.request, numpy as np
from livekit import rtc

QUESTION = "Run an agent to name the three primary colors, and call it colors demo."

def to_np(a):
    if hasattr(a, "detach"): a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float32)

async def main():
    print("[sim] generating question audio with kokoro...", flush=True)
    from kokoro import KPipeline
    kp = KPipeline(lang_code="a")
    q = np.concatenate([to_np(a) for _, _, a in kp(QUESTION, voice="af_heart")])  # 24kHz
    SR = 24000
    q16 = (np.clip(q, -1, 1) * 32767).astype(np.int16)

    r = json.load(urllib.request.urlopen("http://127.0.0.1:8790/token?room=voice&identity=simbrowser"))
    room = rtc.Room()
    reply_stream = [None]
    @room.on("track_subscribed")
    def _t(track, pub, part):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            reply_stream[0] = rtc.AudioStream(track)
    await room.connect("ws://127.0.0.1:7880", r["token"]); print("[sim] connected", flush=True)

    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    print("[sim] mic published; waiting out the greeting (9s)...", flush=True)
    await asyncio.sleep(9)

    spf = SR // 100  # 10ms frames
    print(f"[sim] speaking question ({len(q16)/SR:.1f}s): {QUESTION!r}", flush=True)
    for i in range(0, len(q16), spf):
        c = q16[i:i+spf]
        if len(c) < spf: c = np.pad(c, (0, spf - len(c)))
        await source.capture_frame(rtc.AudioFrame(c.tobytes(), SR, 1, spf))
        await asyncio.sleep(0.01)
    # trailing silence so VAD endpoints the utterance
    sil = np.zeros(spf, dtype=np.int16)
    for _ in range(120):
        await source.capture_frame(rtc.AudioFrame(sil.tobytes(), SR, 1, spf))
        await asyncio.sleep(0.01)
    print("[sim] question sent; capturing agent reply...", flush=True)

    reply = []; rate = [16000]
    async def reader():
        async for ev in reply_stream[0]:
            rate[0] = ev.frame.sample_rate
            reply.append(np.frombuffer(ev.frame.data, dtype=np.int16).copy())
    try: await asyncio.wait_for(reader(), timeout=22)
    except asyncio.TimeoutError: pass

    if reply:
        rep = np.concatenate(reply).astype(np.float32) / 32768.0
        sr = rate[0]
        n = int(len(rep) * 16000 / sr)
        rep16 = np.interp(np.linspace(0, len(rep), n, endpoint=False), np.arange(len(rep)), rep).astype(np.float32)
        peak = float(np.abs(rep).max())
        print(f"[sim] reply audio: {len(rep)/sr:.1f}s @ {sr}Hz, peak={peak:.2f}", flush=True)
        import mlx_whisper
        txt = mlx_whisper.transcribe(rep16, path_or_hf_repo="mlx-community/whisper-large-v3-turbo")["text"].strip()
        print("[sim] AGENT REPLY (transcribed):", repr(txt), flush=True)
    else:
        print("[sim] NO reply audio captured", flush=True)
    await room.disconnect(); print("[sim] done", flush=True)

asyncio.run(main())
