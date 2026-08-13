import asyncio
import logging
import os

from livekit.agents import tts, utils
import numpy as np

# pocket-tts logs a DEBUG line per 80ms audio chunk — far too chatty for our log.
logging.getLogger("pocket_tts").setLevel(logging.WARNING)

SAMPLE_RATE = 24000
NUM_CHANNELS = 1


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
        return PocketStream(tts=self, input_text=text, conn_options=conn_options)


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
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)


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
