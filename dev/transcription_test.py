import asyncio, json, urllib.request, numpy as np
from livekit import rtc

QUESTION = "What is the capital of France? Answer in one short sentence."
def to_np(a):
    if hasattr(a, "detach"): a = a.detach().cpu().numpy()
    return np.asarray(a, dtype=np.float32)

async def main():
    from kokoro import KPipeline
    kp = KPipeline(lang_code="a")
    q = np.concatenate([to_np(a) for _, _, a in kp(QUESTION, voice="af_heart")])
    SR = 24000; q16 = (np.clip(q, -1, 1) * 32767).astype(np.int16)

    me = "user-simtest"
    r = json.load(urllib.request.urlopen(f"http://127.0.0.1:8790/token?room=voice&identity={me}"))
    room = rtc.Room()
    got = {"user": [], "agent": []}

    def on_text(reader, participant_identity):
        who = "user" if participant_identity == me else "agent"
        final = reader.info.attributes.get("lk.transcription_final")
        async def read():
            txt = await reader.read_all()
            got[who].append(txt)
            print(f"[transcription] from={who}({participant_identity}) final={final} text={txt!r}", flush=True)
        asyncio.ensure_future(read())

    room.register_text_stream_handler("lk.transcription", on_text)
    await room.connect(r["url"], r["token"]); print("[test] connected", flush=True)

    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    await asyncio.sleep(9)  # let greeting play

    spf = SR // 100
    print("[test] speaking question...", flush=True)
    for i in range(0, len(q16), spf):
        c = q16[i:i+spf]
        if len(c) < spf: c = np.pad(c, (0, spf - len(c)))
        await source.capture_frame(rtc.AudioFrame(c.tobytes(), SR, 1, spf)); await asyncio.sleep(0.01)
    sil = np.zeros(spf, dtype=np.int16)
    for _ in range(120): await source.capture_frame(rtc.AudioFrame(sil.tobytes(), SR, 1, spf)); await asyncio.sleep(0.01)

    await asyncio.sleep(12)
    print(f"[test] SUMMARY user_transcripts={len(got['user'])} agent_transcripts={len(got['agent'])}", flush=True)
    await room.disconnect(); print("[test] done", flush=True)

asyncio.run(main())
