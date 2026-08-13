import asyncio, json, urllib.request, time
from livekit import rtc
async def main():
    r = json.load(urllib.request.urlopen("http://127.0.0.1:8790/token?room=voice&identity=probe"))
    room = rtc.Room(); t0=[None]; done=asyncio.Event()
    @room.on("participant_connected")
    def _p(p): print(f"  +{time.time()-t0[0]:.1f}s agent participant joined: {p.identity}", flush=True)
    @room.on("track_subscribed")
    def _t(track,pub,part):
        if track.kind==rtc.TrackKind.KIND_AUDIO:
            print(f"  +{time.time()-t0[0]:.1f}s agent AUDIO track subscribed", flush=True); done.set()
    t0[0]=time.time()
    await room.connect("ws://127.0.0.1:7880", r["token"])
    print(f"  +{time.time()-t0[0]:.1f}s connected to room", flush=True)
    try: await asyncio.wait_for(done.wait(), timeout=40)
    except asyncio.TimeoutError: print("  TIMEOUT: agent never published audio in 40s", flush=True)
    await room.disconnect()
asyncio.run(main())
