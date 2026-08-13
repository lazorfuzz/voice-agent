import asyncio, json, urllib.request, time, sys
from livekit import rtc
URL="ws://127.0.0.1:7880"
async def one(i, gap):
    r = json.load(urllib.request.urlopen(f"http://127.0.0.1:8790/token?room=voice&identity=cyc{i}"))
    room = rtc.Room(); done=asyncio.Event(); t0=time.time()
    @room.on("track_subscribed")
    def _t(track,pub,part):
        if track.kind==rtc.TrackKind.KIND_AUDIO: done.set()
    await room.connect(URL, r["token"])
    ok=True
    try: await asyncio.wait_for(done.wait(), timeout=30)
    except asyncio.TimeoutError: ok=False
    dt=time.time()-t0
    print(f"cycle {i}: {'AGENT JOINED' if ok else 'NO AGENT (30s timeout)'} in {dt:.1f}s", flush=True)
    await room.disconnect()
    await asyncio.sleep(gap)
async def main():
    for i in range(5):
        await one(i, gap=2.0)   # 2s between disconnect and next connect
asyncio.run(main())
