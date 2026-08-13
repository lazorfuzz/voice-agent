"""Lazy TTS voice previews for the persona-create UI.

Synthesizes a short fixed sample of a pocket-tts catalog voice and returns it as a WAV.
pocket-tts (torch, ~440MB) loads on the FIRST preview only, so the token server stays light
until someone actually previews a voice. The model + per-voice states are cached, and calls
serialize on a lock (pocket-tts generation isn't reentrant)."""
import struct
import threading

_SAMPLE_RATE = 24000
_SAMPLE_TEXT = "Hey there — this is how I sound. Pretty good, right?"

_TTL_SECONDS = 300   # unload the model 5 min after it LOADS — fixed TTL, not idle-based

_lock = threading.Lock()
_model = None
_states = {}          # voice name -> cached voice state
_unload_timer = None


def _unload():
    """Drop the model + cached voice states so torch frees ~440MB. Acquires the lock, so it
    waits for any in-flight synth and never unloads mid-generation. The next preview reloads."""
    global _model, _states, _unload_timer
    import gc
    with _lock:
        _model = None
        _states = {}
        _unload_timer = None
    gc.collect()


def _ensure_model():
    global _model, _unload_timer
    if _model is None:
        from pocket_tts import TTSModel
        _model = TTSModel.load_model()
        # schedule the unload once, when the model loads — a flat 5-min TTL regardless of use
        _unload_timer = threading.Timer(_TTL_SECONDS, _unload)
        _unload_timer.daemon = True
        _unload_timer.start()
    return _model


def _state_for(voice: str):
    if voice not in _states:
        _states[voice] = _ensure_model().get_state_for_audio_prompt(voice)
    return _states[voice]


def _wav(pcm: bytes, rate: int = _SAMPLE_RATE, channels: int = 1, bits: int = 16) -> bytes:
    byte_rate = rate * channels * bits // 8
    block_align = channels * bits // 8
    return (
        b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, rate, byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", len(pcm)) + pcm
    )


def synth_wav(voice: str, text: str = None) -> bytes:
    """Return a mono 24kHz 16-bit WAV of `text` (default sample) spoken in `voice`.
    First call loads the model (~15-20s); subsequent calls are fast (cached)."""
    import numpy as np
    text = text or _SAMPLE_TEXT
    with _lock:
        model = _ensure_model()
        state = _state_for(voice)
        chunks = []
        for audio in model.generate_audio_stream(state, text):
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32)
            chunks.append((np.clip(audio, -1, 1) * 32767).astype(np.int16))
    pcm = np.concatenate(chunks).tobytes() if chunks else b""
    return _wav(pcm)
