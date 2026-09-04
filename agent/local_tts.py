import asyncio
import logging
import os
import re

from livekit.agents import tts, utils
import numpy as np

# pocket-tts logs a DEBUG line per 80ms audio chunk — far too chatty for our log.
logging.getLogger("pocket_tts").setLevel(logging.WARNING)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1

# ---------------------------------------------------------------------------
# Spoken-text normalization. pocket-tts (110M) does no text normalization: "."
# and "," inside numbers read as pauses, so "1.5M" came out "one, five, m" and
# "$1.85" as "one, eighty five". Expand money / magnitude suffixes / decimals /
# percents into speakable words BEFORE synthesis. Applied in synthesize(), which
# receives whole sentences (the framework assembles them first), so patterns
# never split across streaming chunk boundaries.
# ---------------------------------------------------------------------------
_MAG = {"k": "thousand", "K": "thousand", "M": "million", "B": "billion", "T": "trillion"}


def _spoken_decimal(m):
    whole, frac = m.group(1), m.group(2)
    return f"{whole} point {' '.join(frac)}"      # 3.14 -> "3 point 1 4"


def _dollars(n: str) -> str:
    return f"{n} dollar" if n == "1" else f"{n} dollars"


def normalize_spoken(text: str) -> str:
    t = text
    # 1,200 -> 1200 (thousands separators pause mid-number otherwise)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)
    # $1.5M / $2B -> "1.5 million dollars" (decimal expanded below)
    t = re.sub(r"\$(\d+(?:\.\d+)?)\s?([kKMBT])\b",
               lambda m: f"{m.group(1)} {_MAG[m.group(2)]} dollars", t)
    # 1.5M / 300k -> "1.5 million" / "300 thousand"
    t = re.sub(r"\b(\d+(?:\.\d+)?)\s?([kKMBT])\b(?!\w)",
               lambda m: f"{m.group(1)} {_MAG[m.group(2)]}", t)
    # $1.85 -> "1 dollar and 85 cents"; $0.50 -> "50 cents"
    t = re.sub(r"\$(\d+)\.(\d{2})\b",
               lambda m: (f"{int(m.group(2))} cents" if m.group(1) == "0"
                          else f"{_dollars(m.group(1))} and {int(m.group(2))} cents"), t)
    # $5 -> "5 dollars"
    t = re.sub(r"\$(\d+)\b", lambda m: _dollars(m.group(1)), t)
    # bare decimals: 1.5 -> "1 point 5" (skip versions like 1.17.14)
    t = re.sub(r"(?<![.\d])\b(\d+)\.(\d+)\b(?!\.\d)", _spoken_decimal, t)
    # 12 percent, not "twelve <pause>"
    t = re.sub(r"(?<=\d)\s?%", " percent", t)
    return t


# ---------------------------------------------------------------------------
# Pocket TTS (Kyutai) — current engine. ~22ms TTFB vs Kokoro's ~180ms because
# it truly streams: first audio chunk comes out in a couple of frames while the
# rest generates. 100M params, CPU, ~6-14x realtime on this Mac. Same 24kHz /
# torch-tensor output as Kokoro, so the conversion path is identical.
# ---------------------------------------------------------------------------
class PocketTTS(tts.TTS):
    def __init__(self, voice: str = None, model=None, **kwargs):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        from pocket_tts import TTSModel

        # `model` lets multiple voices (personas) SHARE one loaded model — load_model()
        # is NOT a singleton, so a fresh call per voice would double the ~440MB weights.
        self._model = model or TTSModel.load_model()
        voice = voice or os.environ.get("POCKET_VOICE", "alba")
        # POCKET_VOICE is either a catalog voice name ("alba", "jean", ...) or a path to
        # a local audio file -> VOICE CLONE (e.g. voices/tina.wav; needs the gated
        # kyutai/pocket-tts weights: accept terms on HF + `hf auth login`). A str is a
        # name/URL to pocket-tts, a Path is a local file — hence the isfile branch. If
        # the clone fails for any reason (gate, missing file), fall back to a catalog
        # voice instead of taking the whole agent down at prewarm.
        from pathlib import Path
        try:
            if os.path.isfile(voice):
                self._voice_state = self._model.get_state_for_audio_prompt(
                    Path(voice), truncate=True)
            else:
                self._voice_state = self._model.get_state_for_audio_prompt(voice)
        except Exception:
            logging.getLogger("kronik.tts").exception(
                "voice %r failed to load; falling back to 'jean'", voice)
            self._voice_state = self._model.get_state_for_audio_prompt("jean")
        # warm up the compute graph so the first real turn isn't the cold one.
        for _ in self._model.generate_audio_stream(self._voice_state, "hey."):
            break

    def synthesize(self, text, *, conn_options=None) -> "PocketStream":
        return PocketStream(tts=self, input_text=normalize_spoken(text), conn_options=conn_options)


class PocketStream(tts.ChunkedStream):
    async def _run(self, output_emitter):
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        model = self._tts._model
        gen = model.generate_audio_stream(self._tts._voice_state, self.input_text)

        # Pull each chunk off the event loop (blocking CPU work) so the first
        # ~22ms chunk can flush to the wire while the rest is still generating.
        def _next():
            try:
                return next(gen)
            except StopIteration:
                return None

        while True:
            audio = await asyncio.to_thread(_next)
            if audio is None:
                break
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            pcm_bytes = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
            output_emitter.push(pcm_bytes)

        output_emitter.flush()


# ---------------------------------------------------------------------------
# Kokoro TTS — previous engine, kept for easy rollback (swap the class used in
# agent.py's prewarm back to KokoroTTS). Non-streaming: synthesizes the whole
# first sentence before yielding, so TTFB ~= full first-sentence synth (~180ms).
# ---------------------------------------------------------------------------
class KokoroTTS(tts.TTS):
    def __init__(self, **kwargs):
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
        )
        from kokoro import KPipeline

        self._pipeline = KPipeline(lang_code="a")
        self._voice = "af_heart"

    def synthesize(self, text, *, conn_options=None) -> "ChunkedStream":
        return ChunkedStream(tts=self, input_text=normalize_spoken(text), conn_options=conn_options)


class ChunkedStream(tts.ChunkedStream):
    async def _run(self, output_emitter):
        output_emitter.initialize(
            request_id=utils.shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=NUM_CHANNELS,
            mime_type="audio/pcm",
        )

        for (_, _, audio) in self._tts._pipeline(
            self.input_text, voice=self._tts._voice
        ):
            # kokoro yields torch Tensors -> convert to numpy float32
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            pcm_bytes = (
                (np.clip(audio, -1, 1) * 32767).astype(np.int16)
            ).tobytes()
            output_emitter.push(pcm_bytes)

        output_emitter.flush()
