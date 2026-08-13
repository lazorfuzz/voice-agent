"""A/B latency sim: joins the live room as a caller, speaks 6 pre-generated utterances
(identical audio across runs), waits for each agent reply to finish, then reports the
turn metrics parsed from latency.log for this session. Run:  .venv/bin/python sim_ab.py <tag>
"""
import asyncio
import json
import sys
import time
import urllib.request
import wave

import numpy as np
from livekit import rtc

TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
UTTS = [f"/tmp/ab_utts/u{i}.wav" for i in range(1, 7)]
SR = 24000
SPF = SR // 100          # 10ms frames


def load(p):
    with wave.open(p, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


async def main():
    t_start = time.time()
    r = json.load(urllib.request.urlopen(
        "http://127.0.0.1:8790/token?room=voice&identity=sim-ab&persona=kronik&ambient=off"))
    room = rtc.Room()
    lvl = {"rms": 0.0, "t": 0.0}

    @room.on("track_subscribed")
    def _t(track, pub, part):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            async def pump():
                stream = rtc.AudioStream(track)
                async for ev in stream:
                    d = np.frombuffer(ev.frame.data, dtype=np.int16).astype(np.float32) / 32768.0
                    rms = float(np.sqrt(np.mean(d ** 2)))
                    if rms > 0.01:
                        lvl["rms"] = rms
                        lvl["t"] = time.time()
            asyncio.create_task(pump())

    await room.connect("ws://127.0.0.1:7880", r["token"])   # local, skips the caddy TLS cert
    source = rtc.AudioSource(SR, 1)
    track = rtc.LocalAudioTrack.create_audio_track("mic", source)
    await room.local_participant.publish_track(
        track, rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE))
    print(f"[{TAG}] connected; waiting out greeting...", flush=True)

    async def play(pcm):
        for i in range(0, len(pcm), SPF):
            c = pcm[i:i + SPF]
            if len(c) < SPF:
                c = np.pad(c, (0, SPF - len(c)))
            await source.capture_frame(rtc.AudioFrame(c.tobytes(), SR, 1, SPF))

    async def wait_reply(max_s=25):
        """Wait until agent audio has played and then been silent for 1.6s."""
        t0 = time.time()
        heard = False
        while time.time() - t0 < max_s:
            if lvl["t"] > t0 and not heard:
                heard = True
            if heard and time.time() - lvl["t"] > 1.6:
                return True
            await asyncio.sleep(0.1)
        return heard

    await wait_reply(12)             # greeting
    m2e = []                          # TRUE mouth-to-ear: end of our audio -> first reply frame
    for n, p in enumerate(UTTS, 1):
        pcm = load(p)
        print(f"[{TAG}] turn {n}: speaking {len(pcm)/SR:.1f}s", flush=True)
        await play(pcm)
        await source.wait_for_playout()   # frames are QUEUED up to ~1s ahead; wait for real playout
        t_done = time.time()          # our last frame actually transmitted = "finished talking"
        # wait for FIRST reply audio
        first = None
        while time.time() - t_done < 25:
            if lvl["t"] > t_done:
                first = lvl["t"]
                break
            await asyncio.sleep(0.005)
        if first:
            m2e.append(round((first - t_done) * 1000))
            print(f"[{TAG}] turn {n}: MOUTH-TO-EAR {m2e[-1]}ms", flush=True)
        ok = await wait_reply()
        print(f"[{TAG}] turn {n}: reply {'done' if ok else 'TIMEOUT'}", flush=True)
        await asyncio.sleep(0.8)
    print(f"[{TAG}] M2E ms: {m2e}  median={sorted(m2e)[len(m2e)//2] if m2e else -1}", flush=True)

    await room.disconnect()

    # parse this session's turns from latency.log
    await asyncio.sleep(1)
    lines = open("latency.log").read().splitlines()
    eou, ttft = [], []
    t0h = time.strftime("%H:%M:%S", time.localtime(t_start))
    started = False
    for ln in lines:
        if not started and ln[:8] >= t0h and "SESSION START" in ln:
            started = True
        if not started:
            continue
        if "EOU" in ln and "eou_delay=" in ln:
            eou.append(int(ln.split("eou_delay=")[1].split("ms")[0]))
        if "LLM" in ln and "ttft=" in ln:
            ttft.append(int(ln.split("ttft=")[1].split("ms")[0]))
    med = lambda v: sorted(v)[len(v) // 2] if v else -1
    print(f"\n[{TAG}] RESULTS  turns={len(eou)}")
    print(f"[{TAG}] EOU  ms: {eou}  median={med(eou)}")
    print(f"[{TAG}] TTFT ms: {ttft}  median={med(ttft)}")


asyncio.run(main())
