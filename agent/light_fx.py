"""TTS-reactive light pulsing: Kronik's voice drives the WiZ lights like a waveform.

agent.py's tts_node tap feeds ~100ms RMS envelope buckets, timestamped in CUMULATIVE
AUDIO TIME (not arrival time — TTS synthesizes faster than realtime, so arrival time
would race ahead of what's audible). The driver replays those samples against the wall
clock, offset by WIZ_PULSE_LAG_MS to roughly match the audio output buffer, sending
fire-and-forget setPilot dimming updates (~10Hz/lamp — the same rate WiZ's own music
sync uses) to every configured light that was ON when the speech started. Each light
pulses AROUND ITS OWN BASELINE (a 9% night lamp ripples 5-35%, a bright lamp swings
wide), and its exact prior state is restored when the speech ends — or immediately if
the user interrupts (agent.py aborts on leaving the "speaking" state).

Three audio dimensions drive the show: LOUDNESS (RMS) -> brightness (loud = bright,
sentence gaps = dip); PITCH (F0 via FFT autocorrelation) -> hue rotation while voiced
(low pitch = warm orange-red, rising intonation sweeps toward blue-violet); and TIMBRE
(spectral centroid) -> the unvoiced fallback (sibilants flash ice-blue on the
warm->cool gradient). WIZ_PULSE_COLOR selects: "pitch" (default), "centroid" (the older
timbre-only gradient), or "off" (brightness-only in the light's own color).
Toggle everything by voice ("stop pulsing the lights") or WIZ_PULSE=off.
"""
import os
import math
import asyncio
import logging
import colorsys

import numpy as np

import wiz_tools as W

_log = logging.getLogger("kronik.lightfx")

_LAG = max(0.0, float(os.environ.get("WIZ_PULSE_LAG_MS", "300")) / 1000.0)
_GAMMA = 0.6          # perceptual: lift quiet syllables so speech reads as motion
_BUCKET_LATE = 0.15   # drop samples more than this late instead of backlogging
_MODE = os.environ.get("WIZ_PULSE_COLOR", "pitch").lower()
if _MODE in ("on", "true", "1"):
    _MODE = "pitch"
elif _MODE in ("0", "false", "no"):
    _MODE = "off"                 # valid modes: pitch | centroid | off

# TIMBRE dimension: SPECTRAL CENTROID -> color. Vowels/low pitch sit low (warm amber);
# sibilants/crisp consonants spike high (ice blue). Piecewise-lerped gradient,
# positions in normalized log-frequency [0..1]:
_GRADIENT = [
    (0.00, (255, 60, 0)),     # deep amber — low, round sounds
    (0.45, (255, 170, 60)),   # warm gold — typical vowels
    (0.75, (200, 200, 160)),  # pale neutral — mixed speech
    (1.00, (70, 130, 255)),   # ice blue — sibilant flash
]
_C_LO, _C_HI = 300.0, 4000.0  # centroid Hz range mapped (log scale) onto the gradient


def _centroid_rgb(c_hz: float):
    if c_hz <= 0:
        return _GRADIENT[1][1]
    x = (math.log(max(c_hz, _C_LO)) - math.log(_C_LO)) / (math.log(_C_HI) - math.log(_C_LO))
    x = min(1.0, max(0.0, x))
    for (x0, c0), (x1, c1) in zip(_GRADIENT, _GRADIENT[1:]):
        if x <= x1:
            f = (x - x0) / (x1 - x0)
            return tuple(int(a + f * (b - a)) for a, b in zip(c0, c1))
    return _GRADIENT[-1][1]


# PITCH dimension: F0 -> hue rotation. The voice's speaking range mapped on a log scale
# onto the color wheel: low pitch = warm orange-red (hue ~18deg), rising intonation
# sweeps through green/cyan to blue-violet (~280deg). The range MUST match the active
# voice or the lights hug one color region.
#
# Measured speaking-F0 p10-p90 per voice (Hz), 2026-07-11, via measure_voice_f0.py (with
# octave-error correction). p10/p90 not p5/p95: the range must HUG the real distribution
# so pitch spreads across the whole hue wheel (a too-wide range = "mostly one color").
# Add a line here when adding a voice. Keyed by Path(POCKET_VOICE).stem so catalog names
# AND clone file paths (voices/tina.wav) both resolve.
_F0_RANGES = {
    "jean": (93.0, 118.0),      # median 108Hz
    "alba": (103.0, 174.0),     # median 134Hz
    "cosette": (171.0, 245.0),  # median 203Hz
    "tina": (101.0, 227.0),     # clone (full 16s prompt), median 190Hz
    "tina2": (161.0, 216.0),    # clone (24s natural-speech prompt), median 182Hz
    "consuela": (194.0, 322.0), # clone (Family Guy, 19s stitched+isolated), median 238Hz
}
_DEFAULT_F0 = (85.0, 250.0)     # unknown voice: wide + safe (compressed hue sweep);
                                # run measure_voice_f0.py and add it to _F0_RANGES.

from pathlib import Path as _Path

_HUE_LO, _HUE_HI = 0.05, 0.78

# Manual override (WIZ_PULSE_F0_RANGE=lo:hi) wins over the per-voice map when set.
_ENV_F0 = None
try:
    _lo, _hi = os.environ.get("WIZ_PULSE_F0_RANGE", "").split(":")
    _ENV_F0 = (float(_lo), float(_hi))
except ValueError:
    pass


def resolve_f0_range(voice):
    """(lo, hi) for a pocket-tts voice name or clone path. The ACTIVE voice is chosen
    per-session by the persona system, so this must be resolved at call time (not from a
    static env var) — otherwise every persona pulses with the default voice's range and
    high-pitched voices clamp to violet."""
    if _ENV_F0:
        return _ENV_F0
    key = _Path(voice or "").stem.lower()
    if key not in _F0_RANGES:
        _log.warning("lightfx: no measured F0 range for voice %r — using wide default %s; "
                     "run measure_voice_f0.py and add it to _F0_RANGES", key, _DEFAULT_F0)
    return _F0_RANGES.get(key, _DEFAULT_F0)


def _f0_rgb(f0_hz: float, lo: float, hi: float):
    x = (math.log(min(max(f0_hz, lo), hi)) - math.log(lo)) / (math.log(hi) - math.log(lo))
    h = _HUE_LO + x * (_HUE_HI - _HUE_LO)
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def analyze(pcm: "np.ndarray", sample_rate: int):
    """One ~100ms float32 bucket -> (spectral_centroid_hz, f0_hz). Either is 0.0 when
    not measurable (silence -> both; unvoiced sounds like sibilants -> f0 only)."""
    n = pcm.size
    if n < 256 or float(np.sqrt(np.mean(pcm * pcm))) < 0.008:
        return 0.0, 0.0
    # centroid over the speech band
    mag = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    band = (freqs > 100) & (freqs < 8000)
    e = float(np.sum(mag[band]))
    centroid = float(np.sum(freqs[band] * mag[band]) / e) if e > 1e-6 else 0.0
    # F0 by FFT autocorrelation over the 60-400Hz lag range; a weak periodicity peak
    # (<0.35 of r0) means unvoiced -> no pitch
    x = pcm - float(pcm.mean())
    nfft = 1 << (2 * n - 1).bit_length()
    X = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(X * np.conj(X))[:n]
    r0 = float(ac[0])
    lo, hi = int(sample_rate / 400), min(int(sample_rate / 60), n - 1)
    if r0 <= 1e-9 or hi <= lo:
        return centroid, 0.0
    k = int(np.argmax(ac[lo:hi])) + lo
    if float(ac[k]) / r0 < 0.35:
        return centroid, 0.0
    # Octave-down correction: strong harmonics can make the 2x-period lag win, so the
    # detector reports HALF the true pitch (e.g. 167Hz for a real 334Hz voice). If a
    # comparably strong peak sits at half the lag (= double the frequency), prefer it.
    half = k // 2
    if half >= lo and ac[half] > 0.8 * ac[k]:
        k = half
    return centroid, sample_rate / k


class _Driver:
    def __init__(self):
        self.enabled = os.environ.get("WIZ_PULSE", "on").lower() not in ("off", "0", "false", "no")
        self._q: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._t0_anchor = 0.0
        self._f0_lo, self._f0_hi = resolve_f0_range(os.environ.get("POCKET_VOICE", "alba"))
        self._base_cache = {}          # last-known-good per-lamp baseline (survives UDP jitter)
        # The TRUE baseline for the CURRENT speaking period, captured ONCE (from lamps at
        # rest) and reused by every sentence-pulse within that period. Never re-snapshot
        # mid-speech: a second pulse would otherwise capture the FIRST pulse's transient
        # (bright/color) as its baseline and restore the lamps to that garbage — the
        # "brightness reverts / lamp goes colored" bug. Cleared when the agent stops
        # speaking (end_speaking, from the agent_state hook), so the next reply re-reads.
        self._session_base = None

    def set_voice(self, voice):
        """Set the F0->hue range for the active persona's voice. Call from the job
        entrypoint once the persona is resolved, BEFORE speaking."""
        self._f0_lo, self._f0_hi = resolve_f0_range(voice)

    def suspend(self):
        """Stop pulsing RIGHT NOW without restoring — used just before a deliberate light
        change so the running pulse can't push RGB over the new color (which the next
        re-snapshot would then capture as the baseline and stick the lamp on). The change
        itself becomes the new baseline via invalidate_baseline()."""
        self._stop_task()

    def invalidate_baseline(self):
        """A light-control command changed the lamps: drop BOTH the jitter cache and the
        current speaking-period baseline so nothing restores them to a pre-change value, and
        cancel any pending end-of-speech restore (which captured the OLD state and would
        otherwise fire ~now and undo the change)."""
        self._base_cache.clear()
        self._session_base = None
        for t in list(_RESTORE_TASKS):
            t.cancel()

    # ---- called from agent.py's tts_node tap (same event loop) ----

    def _stop_task(self):
        self._q = None
        if self._task is not None and not self._task.done():
            self._task.cancel()            # _run's finally restores to _session_base
        self._task = None

    def begin(self):
        """One sentence of TTS is starting: start the replay task. Reuses the speaking
        period's baseline (captured on the first sentence), so it never re-snapshots a
        pulse transient."""
        if not self.enabled or not W._LIGHTS:
            return
        self._stop_task()                  # stop the prior sentence's task (keeps baseline)
        # Anchor the playback clock to AUDIO START (now), not to snapshot completion.
        self._t0_anchor = asyncio.get_running_loop().time()
        self._q = asyncio.Queue()
        self._task = asyncio.create_task(self._run(self._q))

    def feed(self, audio_ts: float, rms: float, centroid: float = 0.0, f0: float = 0.0):
        """One ~100ms envelope bucket at cumulative audio time `audio_ts` seconds.
        `centroid` = spectral centroid Hz, `f0` = pitch Hz (0 = unvoiced/unknown)."""
        if self._q is not None:
            self._q.put_nowait((audio_ts, rms, centroid, f0))

    def end(self):
        """TTS generation finished for this sentence: drain the queue then restore."""
        if self._q is not None:
            self._q.put_nowait(None)
            self._q = None

    def end_speaking(self):
        """Agent left the 'speaking' state (finished or interrupted): stop pulsing, then
        do the ONE definitive restore to the true baseline (after pulses have stopped, so
        no stray push lands after it), and forget the baseline so the next reply
        re-snapshots fresh."""
        self._stop_task()
        base = self._session_base
        self._session_base = None
        if base:
            t = asyncio.create_task(self._restore_after(base))
            _RESTORE_TASKS.add(t)
            t.add_done_callback(_RESTORE_TASKS.discard)

    async def _restore_after(self, base):
        await asyncio.sleep(0.06)          # let any in-flight fire-and-forget push settle
        try:
            await asyncio.to_thread(self._restore, base)
        except Exception:
            pass

    # kept for the pulse_off toggle
    abort = end_speaking

    # ---- replay task ----

    async def _run(self, q: asyncio.Queue):
        # Capture the speaking period's baseline ONCE (from lamps at rest), then reuse it
        # for every sentence-pulse. Snapshot is fast (cached-IP getPilot, no retry). Only
        # lights already ON take part; unreachable lights are simply absent.
        if self._session_base is None:
            self._session_base = await asyncio.to_thread(self._snapshot)
        base = self._session_base
        if not base:
            return
        loop = asyncio.get_running_loop()
        # t0 fixed to audio start (set in begin), so the snapshot's duration never
        # offsets the show; buckets whose moment already passed are dropped below.
        t0 = self._t0_anchor + _LAG
        peak = 0.05                        # adaptive loudness reference
        c_ema = 0.0                        # smoothed centroid (raw jumps -> hue strobe)
        f_ema = 0.0                        # smoothed pitch, same reason
        try:
            while True:
                item = await q.get()
                if item is None:
                    break
                ts, rms, centroid, f0 = item
                delay = (t0 + ts) - loop.time()
                if delay > 0:
                    await asyncio.sleep(delay)
                elif -delay > _BUCKET_LATE:
                    continue               # too stale to be worth showing
                peak = max(peak * 0.995, rms, 0.05)
                norm = min(1.0, rms / peak) ** _GAMMA
                params = {}
                if _MODE == "pitch" and f0 > 0:
                    # voiced: pitch drives the hue sweep (range = active persona's voice)
                    f_ema = f0 if f_ema == 0 else 0.5 * f_ema + 0.5 * f0
                    r, g, bl = _f0_rgb(f_ema, self._f0_lo, self._f0_hi)
                    params.update({"r": r, "g": g, "b": bl})
                elif _MODE != "off" and centroid > 0:
                    # centroid mode, or the unvoiced fallback in pitch mode
                    # (sibilants keep their ice-blue flash)
                    c_ema = centroid if c_ema == 0 else 0.55 * c_ema + 0.45 * centroid
                    r, g, bl = _centroid_rgb(c_ema)
                    params.update({"r": r, "g": g, "b": bl})
                for mac, b in base.items():
                    W.push_pilot(mac, {**params,
                                       "dimming": int(b["lo"] + norm * (b["hi"] - b["lo"]))})
        except asyncio.CancelledError:
            pass
        # NOTE: no restore here. Restoring per-sentence races with the next sentence's
        # pulses (and can leave a lamp stuck mid-transient in color mode). Restore happens
        # exactly once, in end_speaking(), after all pulsing for the period has stopped.

    def _snapshot(self):
        # Fast, CONCURRENT getPilot (cached IP, no retry/rediscover) so one dead lamp
        # can't stall the show — an unknown IP returns instantly, a live lamp answers in
        # ~0.25s. A lamp that MISSES its window (UDP jitter, ~13% of the time) would
        # otherwise drop out that turn (the "only one lamp tracks" bug) — so fall back to
        # its last-known-good baseline; only a definitive "off" removes it.
        from concurrent.futures import ThreadPoolExecutor
        lights = W._LIGHTS
        with ThreadPoolExecutor(max_workers=max(1, len(lights))) as ex:
            pilots = list(ex.map(lambda nm: (nm[1], W.get_pilot_fast(nm[1], 0.25)), lights))
        base = {}
        for mac, p in pilots:
            if p is None:
                if mac in self._base_cache:   # momentary miss -> reuse last-good
                    base[mac] = self._base_cache[mac]
                continue
            if not p.get("state"):
                self._base_cache.pop(mac, None)   # genuinely off -> forget it
                continue
            d = int(p.get("dimming", 50))
            b = {
                "pilot": p,
                # pulse band around the light's own baseline: dim lamps ripple gently,
                # bright lamps swing. WiZ min dim is ~10 on many firmwares; floor there.
                "lo": max(10, int(d * 0.5)),
                "hi": min(100, max(d + 25, int(d * 1.8))),
            }
            base[mac] = self._base_cache[mac] = b
        return base

    def _restore(self, base):
        for mac, b in base.items():
            try:
                W.restore_pilot(mac, b["pilot"])
            except Exception:
                _log.warning("lightfx: failed to restore %s", mac)


_RESTORE_TASKS = set()   # strong refs so end-of-speech restores aren't GC'd mid-flight

driver = _Driver()
