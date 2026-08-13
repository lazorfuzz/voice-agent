import asyncio, json, urllib.request, numpy as np
from livekit import rtc
async def main():
    r = json.load(urllib.request.urlopen("http://127.0.0.1:8790/token?room=voice&identity=simbrowser"))
    room = rtc.Room(); stream=[None]
    @room.on("track_subscribed")
    def _t(track,pub,part):
        if track.kind==rtc.TrackKind.KIND_AUDIO:
            print("[sim] agent audio track subscribed", flush=True); stream[0]=rtc.AudioStream(track)
    await room.connect(r['url'], r['token']); print("[sim] connected", flush=True)
    for _ in range(120):
        if stream[0]: break
        await asyncio.sleep(0.5)
    frames=0; peak=0.0
    async def reader():
        nonlocal frames, peak
        async for ev in stream[0]:
            a=np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32)
            frames+=1; peak=max(peak, float(np.abs(a).max()))
    try: await asyncio.wait_for(reader(), timeout=25)
    except asyncio.TimeoutError: pass
    print(f"[sim] agent audio frames={frames} peak_amp={peak:.0f} (>1000 = real speech)", flush=True)
    await room.disconnect(); print("[sim] done", flush=True)
asyncio.run(main())
