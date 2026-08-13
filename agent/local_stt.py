import asyncio
import logging
import os
import re
import types

from livekit.agents import stt, utils
from livekit import rtc
import numpy as np
import mlx_whisper

import speaker_id  # cheap import (encoder is lazy); source of truth for the gate threshold
import config      # assistant name (for the Whisper wake-word bias prompt)

# Observability for "the bot stopped hearing me": every segment VAD hands us that we
# then DROP (energy gate / hallucination filter) is logged with the reason + RMS, so a
# silent-bot incident shows whether audio reached STT at all and why it was rejected.
_stt_log = logging.getLogger("kronik.stt")

# RMS below this (on [-1,1] audio) is treated as non-speech -> skip Whisper entirely.
# Prevents transcribing background noise/silence into hallucinated words.
_ENERGY_GATE = 0.006

# Whisper commonly hallucinates these on near-silence/noise; drop them.
_HALLUCINATIONS = {
    "", ".", "you", "thank you", "thanks for watching", "thank you.",
    "thanks for watching!", "please subscribe", "bye", "bye.",
}


def _prep_audio(buffer):
    """Combine WebRTC frames -> mono float32 @16kHz, or None if the segment is silent/noise."""
    frame = rtc.combine_audio_frames(buffer)
    data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0
    if frame.num_channels > 1:
        data = data.reshape(-1, frame.num_channels).mean(axis=1)
    if frame.sample_rate != 16000:
        n = int(len(data) * 16000 / frame.sample_rate)
        data = np.interp(
            np.linspace(0, len(data), n, endpoint=False),
            np.arange(len(data)), data,
        ).astype(np.float32)
    rms = float(np.sqrt(np.mean(data ** 2))) if len(data) else 0.0
    if len(data) == 0 or rms < _ENERGY_GATE:
        _stt_log.info("STT-DROP energy gate: rms=%.4f (gate=%.4f) dur=%.1fs",
                      rms, _ENERGY_GATE, len(data) / 16000)
        return None
    return data


# The wake word "Kronik" is made-up; ASR that lacks a biasing hint hears the real
# word "Chronic" (and near-spellings). Rewrite it back. Word-boundary only so we
# don't maul legitimate mid-word text.
_WAKE_FIX = re.compile(r"\b(chronic|chronick|kronic|chronik|cronic)\b", re.IGNORECASE)


def _fix_wake_word(text: str) -> str:
    return _WAKE_FIX.sub("Kronik", text)


def _looks_hallucinated(text: str) -> bool:
    t = text.strip().lower().strip(" .!?")
    if not t:
        return True
    if t in _HALLUCINATIONS:
        return True
    # heavy word repetition (e.g. "Fineless Fineless Fineless ...") -> hallucination
    words = t.split()
    if len(words) >= 6 and len(set(words)) <= 2:
        return True
    return False


class LocalWhisperSTT(stt.STT):
    def __init__(self, *, model: str = "mlx-community/whisper-large-v3-turbo",
                 persona_name: str = None):
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=False, interim_results=False
            )
        )
        self._model = model
        # Drives the Whisper initial_prompt so the wake word is spelled right. Set per
        # session from the selected persona (agent.py entry) — one job per process, so a
        # plain instance attribute is safe.
        self.persona_name = persona_name or config.assistant_name()
        # Ambient speaker gate: when set (dict {name: [emb]}), any utterance whose voice
        # doesn't match an enrolled speaker is dropped (empty transcript) BEFORE Whisper,
        # so only enrolled people can wake or talk to the agent. None = no gate.
        self._spk_sigs = None
        self._spk_threshold = speaker_id.DEFAULT_THRESHOLD

    def set_persona(self, name: str):
        self.persona_name = name or config.assistant_name()

    def set_speaker_gate(self, signatures, threshold: float = None):
        self._spk_sigs = signatures or None
        self._spk_threshold = (speaker_id.DEFAULT_THRESHOLD
                               if threshold is None else threshold)

    def _empty(self, language):
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", language=language or "en")],
        )

    async def _recognize_impl(
        self, buffer, *, language=None, conn_options=None
    ):
        frame = rtc.combine_audio_frames(buffer)
        data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32) / 32768.0

        # downmix to mono if needed
        if frame.num_channels > 1:
            data = data.reshape(-1, frame.num_channels).mean(axis=1)

        # whisper expects 16 kHz; WebRTC audio is usually 48 kHz -> resample
        if frame.sample_rate != 16000:
            n = int(len(data) * 16000 / frame.sample_rate)
            data = np.interp(
                np.linspace(0, len(data), n, endpoint=False),
                np.arange(len(data)), data,
            ).astype(np.float32)

        # energy gate: if the whole segment is quiet, it's noise/silence -> no transcript
        if len(data) == 0 or float(np.sqrt(np.mean(data ** 2))) < _ENERGY_GATE:
            return self._empty(language)

        # speaker gate (ambient): drop utterances from non-enrolled voices BEFORE Whisper
        # (saves the transcription too). Runs off the loop; embed is ~tens of ms.
        if self._spk_sigs:
            try:
                import speaker_id
                # verify_and_adapt: on a high-confidence match it folds this utterance into
                # the speaker's profile (mutates self._spk_sigs in place) so the gate adapts
                # to the room/mic/distance over the session -> robust from one enrollment.
                name, sim = await asyncio.to_thread(
                    speaker_id.verify_and_adapt, data, self._spk_sigs, self._spk_threshold)
                if name is None:
                    _stt_log.info("STT-DROP speaker not enrolled (sim=%.2f)", sim)
                    return self._empty(language)
                _stt_log.info("STT-PASS speaker=%s (sim=%.2f)", name, sim)
            except Exception:
                pass  # verification failure must not block transcription

        result = mlx_whisper.transcribe(
            data,
            path_or_hf_repo=self._model,
            language="en",                      # English-only assistant; avoid lang drift
            temperature=0.0,                    # deterministic -> fewer hallucinations
            # Bias the spelling of the wake word to the ACTIVE persona's name (e.g. the
            # made-up "Kronik" defaults to "Chronic" without this hint; "Consuela" biases
            # that spelling). Updated per session in agent.py entry().
            initial_prompt=f"The user is talking to a voice assistant named {self.persona_name}.",
            condition_on_previous_text=True,    # keep context for better accuracy (guarded by energy gate + filter)
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            hallucination_silence_threshold=2.0,
        )

        text = result["text"].strip()
        # Only the Kronik persona needs the chronic->Kronik rescue; Consuela is a real
        # name Whisper spells fine (and biasing already helps).
        if self.persona_name.lower() == "kronik":
            text = _fix_wake_word(text)
        if _looks_hallucinated(text):
            return self._empty(language)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "en")],
        )


# --- Parakeet contextual biasing (NeMo GPU-PB-style phrase boosting) -------------------
# parakeet-mlx exposes no hotword API, but Parakeet-TDT's greedy decode computes per-frame
# token logits before argmax. NeMo's supported biasing for TDT is "phrase boosting": a trie
# of hotword subword-sequences that adds a fixed logit bonus to the next expected piece
# during decode. Flips 'Chronic'->'Kronik' and 'Pipe'->'Pepe' at boost~6; >9 over-fires. A
# flat bonus alone hallucinates hotwords on random NON-speech sounds (the entry subword gets
# tipped over blank), so entry tokens are gated on unbiased probability — see
# _PARAKEET_BOOST_MIN_PROB and the contender gate in decode(). NOTE: the model's
# <|startofcontext|> token is an inherited-tokenizer artifact — priming the TDT prediction
# net with it DESTROYS decoding (verified), so biasing must be decode-side, not prompt-side.
# Tune/disable with PARAKEET_BOOST (0 => plain unbiased generate()).
_PARAKEET_BOOST = float(os.environ.get("PARAKEET_BOOST", "6.0"))

# Contender gate for the boost (see _ParakeetBiaser.decode). A flat bonus on a hotword's
# FIRST subword is enough to tip near-silence/noise into a phantom hotword ("Kronik" on a
# random sound). So only boost an ENTRY token when the UNBIASED model already assigns it at
# least this probability — i.e. tip a genuine near-tie, never conjure a hotword from nothing.
# Continuations of an already-started hotword are exempt so real corrections still finish.
# 0 disables the gate (old always-boost behavior). Raise it if hallucinations persist.
_PARAKEET_BOOST_MIN_PROB = float(os.environ.get("PARAKEET_BOOST_MIN_PROB", "0.03"))


def _parakeet_hotwords() -> list:
    """Names to bias toward: assistant/wake word + every persona label + the configured
    device/pet names. Custom personas are picked up automatically via their labels."""
    names = set()
    names.add(config.assistant_name())
    try:
        import personas
        for p in personas.all_personas().values():
            lbl = (p.get("label") or "").strip()
            if lbl:
                names.add(lbl)
    except Exception:
        pass
    for fn in (config.vacuum_name, config.car_name, config.pet_name):
        try:
            v = (fn() or "").strip()
        except Exception:
            v = ""
        if v and not v.lower().startswith("the "):   # skip generic defaults ("the vacuum")
            names.add(v)
    return sorted(n for n in names if n)


def _subword_encoder(vocab):
    """Greedy longest-match SentencePiece encoder over the model vocab (parakeet-mlx ships
    only `decode`). Run once at init to turn hotwords into subword-id sequences."""
    piece2id = {p: i for i, p in enumerate(vocab)}
    normal = sorted((p for p in vocab if p and not p.startswith("<")), key=len, reverse=True)
    unk = piece2id.get("<unk>", 0)

    def encode(text: str) -> list:
        out = []
        for word in text.split():
            s = "▁" + word          # ▁ word-boundary marker
            i = 0
            while i < len(s):
                for p in normal:
                    if s.startswith(p, i):
                        out.append(piece2id[p]); i += len(p); break
                else:
                    out.append(unk); i += 1
        return out
    return encode


class _ParakeetBiaser:
    """Boosted greedy TDT decode. Builds a trie of hotword subword-sequences; at each frame
    adds `boost` to the logits of tokens that would START or CONTINUE a hotword. Mirrors
    parakeet-mlx's own decode_greedy loop (same speed), plus the bias. A contender gate
    (min_prob) withholds the boost from ENTRY tokens the unbiased model finds implausible, so
    noise can't be tipped into a phantom hotword."""

    def __init__(self, model, phrases, boost: float):
        self._m = model
        self._boost = boost
        self._min_prob = _PARAKEET_BOOST_MIN_PROB
        enc = _subword_encoder(model.vocabulary)
        unk = model.vocabulary.index("<unk>") if "<unk>" in model.vocabulary else 0
        self._root = {}
        self.hotwords = []
        for ph in phrases:
            ids = enc(ph)
            if unk in ids:                 # can't reliably bias -> skip (regex covers wake word)
                continue
            node = self._root
            for t in ids:
                node = node.setdefault(t, {})
            self.hotwords.append(ph)

    def decode(self, mel) -> str:
        import mlx.core as mx
        from parakeet_mlx import tokenizer as _tk
        m = self._m
        V = len(m.vocabulary)                        # V == blank id
        if len(mel.shape) == 2:
            mel = mx.expand_dims(mel, 0)
        feats, lens = m.encoder(mel); mx.eval(feats, lens)
        feature = feats[0:1]; length = int(lens[0])
        boost = self._boost; root = self._root
        active = [root]                              # root stays active: a hotword may start anytime
        last = None; hs = None; toks = []
        step = 0; guard = 0
        while step < length and guard < length * 8:
            guard += 1
            dout, hs2 = m.decoder(mx.array([[last]]) if last is not None else None, hs)
            dout = dout.astype(feature.dtype)
            joint = m.joint(feature[:, step:step + 1], dout)
            logits = joint[0, 0, :, :V + 1]
            if boost > 0:
                # Contender gate: NEVER let the boost conjure a hotword out of noise. On
                # non-speech the model wants blank/low-confidence tokens; a flat bonus on the
                # hotword's first subword tips it over -> phantom hotword on random sounds. So
                # only boost a hotword-ENTRY token (from the trie root) if the UNBIASED model
                # already makes it a real contender (prob >= min_prob). Continuations of an
                # already-started hotword (deeper nodes) are always boosted so genuine
                # corrections finish. Keeps 'Chronic'->'Kronik' (where 'Kron' IS a close
                # acoustic contender) while killing the noise hallucinations.
                probs = None
                if self._min_prob > 0:
                    lg = np.array(logits, dtype=np.float32).reshape(-1)   # unbiased (V+1,)
                    lg -= lg.max()
                    np.exp(lg, out=lg); lg /= lg.sum()
                    probs = lg
                bias = np.zeros(V + 1, dtype=np.float32)
                for node in active:
                    is_root = node is root
                    for t in node:
                        if is_root and probs is not None and probs[t] < self._min_prob:
                            continue
                        bias[t] += boost
                logits = logits + mx.array(bias)
            pred = int(mx.argmax(logits))
            decision = int(mx.argmax(joint[0, 0, :, V + 1:]))
            if pred != V:
                toks.append(pred); last = pred; hs = hs2
                nxt = [root]
                for node in active + [root]:
                    if pred in node:
                        nxt.append(node[pred])
                active = nxt
            step += m.durations[decision]
        return _tk.decode(toks, m.vocabulary).strip()


class LocalParakeetSTT(stt.STT):
    """NVIDIA Parakeet-TDT-0.6b-v3 via parakeet-mlx (Apple Silicon MLX). Much lighter/faster
    than whisper-large-v3-turbo (sub-100ms) with lower WER, and — the point here — far less
    Metal-GPU contention with the LLM during a live turn. Made-up names (wake word etc.) are
    biased at decode time via _ParakeetBiaser (NeMo-style phrase boosting), with the
    'Chronic'->'Kronik' regex kept as a cheap safety net. Roll back via STT_ENGINE=whisper
    (agent.py setup()); disable boosting with PARAKEET_BOOST=0."""

    def __init__(self, *, model: str = "mlx-community/parakeet-tdt-0.6b-v3",
                 persona_name: str = None):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        from parakeet_mlx import from_pretrained
        self._model = from_pretrained(model)
        self._biaser = None
        if _PARAKEET_BOOST > 0:
            try:
                hw = _parakeet_hotwords()
                self._biaser = _ParakeetBiaser(self._model, hw, _PARAKEET_BOOST)
                _stt_log.info("parakeet phrase-boost=%.1f hotwords=%s",
                              _PARAKEET_BOOST, hw)
            except Exception:
                _stt_log.exception("parakeet biaser init failed; unbiased decode")
        self.persona_name = persona_name or config.assistant_name()
        self._spk_sigs = None
        self._spk_threshold = speaker_id.DEFAULT_THRESHOLD
        try:                                    # warm the compute graph
            self._transcribe(np.zeros(16000, dtype=np.float32))
        except Exception:
            pass

    def set_persona(self, name: str):
        self.persona_name = name or config.assistant_name()

    def set_speaker_gate(self, signatures, threshold: float = None):
        self._spk_sigs = signatures or None
        self._spk_threshold = (speaker_id.DEFAULT_THRESHOLD
                               if threshold is None else threshold)

    def _empty(self, language):
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", language=language or "en")],
        )

    def _transcribe(self, data) -> str:
        import mlx.core as mx
        import parakeet_mlx.audio as _pa
        mel = _pa.get_logmel(mx.array(data), self._model.preprocessor_config)
        if self._biaser is not None:
            return self._biaser.decode(mel)
        return self._model.generate(mel)[0].text

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        data = _prep_audio(buffer)
        if data is None:
            return self._empty(language)
        # speaker gate (ambient): drop non-enrolled voices BEFORE transcription
        if self._spk_sigs:
            try:
                name, sim = await asyncio.to_thread(
                    speaker_id.verify_and_adapt, data, self._spk_sigs, self._spk_threshold)
                if name is None:
                    _stt_log.info("STT-DROP speaker not enrolled (sim=%.2f)", sim)
                    return self._empty(language)
                _stt_log.info("STT-PASS speaker=%s (sim=%.2f)", name, sim)
            except Exception:
                pass
        # MLX GPU streams are thread-local: run inference DIRECTLY on the event-loop thread
        # (where the model was warmed), like LocalWhisperSTT. A to_thread worker has no gpu
        # stream -> "no Stream(gpu, 0) in current thread". ~190ms blocking, same as Whisper.
        text = (self._transcribe(data) or "").strip()
        if self.persona_name.lower() == "kronik":
            text = _fix_wake_word(text)
        if _looks_hallucinated(text):
            _stt_log.info("STT-DROP hallucination filter: %r", text[:80])
            return self._empty(language)
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "en")],
        )


# --- Streaming dual-decoder Parakeet (Phase 2: decouple "start reasoning" from "user finished") ---
# Emits rolling INTERIM/PREFLIGHT partials from a WINDOWED streaming decode DURING speech (PREFLIGHT
# drives LiveKit preemptive generation = the LLM starts before end-of-turn; INTERIM is smooth display
# + a future mouth loop). The AUTHORITATIVE FINAL is the full-utterance BATCH biased decode
# (_ParakeetBiaser — current accuracy, hotword fix intact). Streaming decode uses windowed local
# attention and is materially LESS accurate (validated: it mangles "Kronik is the front door locked?"
# -> "Consuela lock."), so it is NEVER the final. A streaming STT bypasses LiveKit's VAD StreamAdapter
# and receives a continuous frame stream + a silence-flush at commit, so this stream does its OWN
# trailing-silence segmentation. Enable with STT_ENGINE=parakeet-stream; roll back to STT_ENGINE=parakeet.
_STREAM_CTX = (256, 256)
_STREAM_DEPTH = 1
_STREAM_FEED = int(os.environ.get("STT_STREAM_FEED", "6400"))              # samples per streaming decode (~0.4s)
_STREAM_PREFLIGHT_SIL = float(os.environ.get("STT_STREAM_PREFLIGHT_SIL", "0.15"))  # micro-pause -> PREFLIGHT (preemptive)
_STREAM_FINAL_SIL = float(os.environ.get("STT_STREAM_FINAL_SIL", "0.5"))   # end-of-turn silence -> batch FINAL
_STREAM_MIN_SPEECH = float(os.environ.get("STT_STREAM_MIN_SPEECH", "0.25"))
# Run the windowed streaming decode for partials. OFF => no INTERIM/PREFLIGHT (no preemptive gen), just
# a batch final at end-of-turn = pure overhead; the knob exists to isolate the streaming decode's
# event-loop cost (each decode runs INLINE on the loop, MLX being thread-local).
_STREAM_INTERIM = os.environ.get("STT_STREAM_INTERIM", "1") != "0"


def _biased_decode_greedy_factory(root, boost, min_prob):
    """A copy of ParakeetTDT.decode_greedy + the contender-gated hotword boost, to bind onto the
    streaming model instance so StreamingParakeet's partials are biased too (it calls
    model.decode -> model.decode_greedy). Trie active-state resets per call (per finalized/draft
    region) — mirrors _ParakeetBiaser but threads decoder hidden state for the rolling window."""
    def decode_greedy(model, features, lengths=None, last_token=None, hidden_state=None, *, config):
        import mlx.core as mx
        from parakeet_mlx.alignment import AlignedToken
        from parakeet_mlx import tokenizer as _tk
        B, S, *_ = features.shape
        V = len(model.vocabulary)
        if hidden_state is None:
            hidden_state = [None] * B
        if lengths is None:
            lengths = mx.array([S] * B)
        if last_token is None:
            last_token = [None] * B
        results = []
        for b in range(B):
            hyp = []
            feature = features[b:b + 1]
            length = int(lengths[b])
            active = [root]
            step = 0
            new_symbols = 0
            while step < length:
                dout, (hid, cell) = model.decoder(
                    mx.array([[last_token[b]]]) if last_token[b] is not None else None,
                    hidden_state[b])
                dout = dout.astype(feature.dtype)
                dhid = (hid.astype(feature.dtype), cell.astype(feature.dtype))
                joint_out = model.joint(feature[:, step:step + 1], dout)
                tl = joint_out[0, 0, :, :V + 1]
                if boost > 0:
                    lg = np.array(tl, dtype=np.float32).reshape(-1)
                    probs = None
                    if min_prob > 0:
                        m = lg - lg.max(); np.exp(m, out=m); m /= m.sum(); probs = m
                    bias = np.zeros(V + 1, dtype=np.float32)
                    for node in active:
                        is_root = node is root
                        for t in node:
                            if is_root and probs is not None and probs[t] < min_prob:
                                continue
                            bias[t] += boost
                    tl = tl + mx.array(bias)
                pred = int(mx.argmax(tl))
                tp = mx.softmax(tl, axis=-1)
                conf = float(1.0 - (-mx.sum(tp * mx.log(tp + 1e-10), axis=-1)
                                    / mx.log(mx.array(V + 1, dtype=tp.dtype))))
                decision = int(mx.argmax(joint_out[0, 0, :, V + 1:]))
                if pred != V:
                    hyp.append(AlignedToken(int(pred), start=step * model.time_ratio,
                                            duration=model.durations[decision] * model.time_ratio,
                                            confidence=conf,
                                            text=_tk.decode([pred], model.vocabulary)))
                    last_token[b] = pred
                    hidden_state[b] = dhid
                    nxt = [root]
                    for node in active + [root]:
                        if pred in node:
                            nxt.append(node[pred])
                    active = nxt
                step += model.durations[int(decision)]
                new_symbols += 1
                if model.durations[int(decision)] != 0:
                    new_symbols = 0
                elif model.max_symbols is not None and model.max_symbols <= new_symbols:
                    step += 1
                    new_symbols = 0
            results.append(hyp)
        return results, hidden_state
    return decode_greedy


class LocalStreamingParakeetSTT(stt.STT):
    """Streaming dual-decoder Parakeet-TDT. See module comment above. Drop-in for LocalParakeetSTT
    (same set_persona/set_speaker_gate); STT_ENGINE=parakeet-stream selects it in agent.py."""

    def __init__(self, *, model: str = "mlx-community/parakeet-tdt-0.6b-v3", persona_name=None):
        super().__init__(capabilities=stt.STTCapabilities(streaming=True, interim_results=True))
        from parakeet_mlx import from_pretrained
        self._model = from_pretrained(model)
        self._biaser = None
        if _PARAKEET_BOOST > 0:
            try:
                hw = _parakeet_hotwords()
                self._biaser = _ParakeetBiaser(self._model, hw, _PARAKEET_BOOST)
                if _STREAM_INTERIM:
                    self._model.decode_greedy = types.MethodType(
                        _biased_decode_greedy_factory(self._biaser._root, self._biaser._boost,
                                                      self._biaser._min_prob), self._model)
                _stt_log.info("parakeet STREAMING dual-decoder; boost=%.1f interim=%s hotwords=%s",
                              _PARAKEET_BOOST, _STREAM_INTERIM, hw)
            except Exception:
                _stt_log.exception("parakeet streaming biaser init failed; unbiased")
        self.persona_name = persona_name or config.assistant_name()
        self._spk_sigs = None
        self._spk_threshold = speaker_id.DEFAULT_THRESHOLD
        try:                                    # warm batch + streaming compute graphs
            self._batch_final(np.zeros(16000, dtype=np.float32))
            if _STREAM_INTERIM:
                import mlx.core as mx
                with self._model.transcribe_stream(context_size=_STREAM_CTX, depth=_STREAM_DEPTH) as _s:
                    _s.add_audio(mx.array(np.zeros(_STREAM_FEED, dtype=np.float32)))
        except Exception:
            pass

    def set_persona(self, name):
        self.persona_name = name or config.assistant_name()

    def set_speaker_gate(self, signatures, threshold=None):
        self._spk_sigs = signatures or None
        self._spk_threshold = (speaker_id.DEFAULT_THRESHOLD if threshold is None else threshold)

    def _batch_final(self, data) -> str:
        import mlx.core as mx
        import parakeet_mlx.audio as _pa
        mel = _pa.get_logmel(mx.array(data), self._model.preprocessor_config)
        if self._biaser is not None:
            return self._biaser.decode(mel)
        return self._model.generate(mel)[0].text

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        # batch fallback (recognize() called directly, e.g. outside the streaming path)
        data = _prep_audio(buffer)
        text = "" if data is None else (self._batch_final(data) or "").strip()
        if self.persona_name.lower() == "kronik":
            text = _fix_wake_word(text)
        if _looks_hallucinated(text):
            text = ""
        return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                               alternatives=[stt.SpeechData(text=text, language=language or "en")])

    def stream(self, *, language=None, conn_options=None):
        if conn_options is None:
            from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
            conn_options = DEFAULT_API_CONNECT_OPTIONS
        return _ParakeetRecognizeStream(self, conn_options)


class _ParakeetRecognizeStream(stt.RecognizeStream):
    def __init__(self, parent, conn_options):
        super().__init__(stt=parent, conn_options=conn_options, sample_rate=16000)
        self._p = parent

    def _emit(self, etype, text):
        self._event_ch.send_nowait(stt.SpeechEvent(
            type=etype, alternatives=[stt.SpeechData(text=text, language="en")]))

    def _final_text(self, audio) -> str:
        """Authoritative batch biased decode + persona wake-fix + hallucination filter ('' if dropped)."""
        text = (self._p._batch_final(audio) or "").strip()
        if self._p.persona_name.lower() == "kronik":
            text = _fix_wake_word(text)
        if _looks_hallucinated(text):
            return ""
        return text

    async def _run(self):
        import mlx.core as mx
        p = self._p
        s = None
        if _STREAM_INTERIM:
            s = p._model.transcribe_stream(context_size=_STREAM_CTX, depth=_STREAM_DEPTH)
            s.__enter__()
        seg = []            # full speech segment -> batch final
        pending = []        # frames buffered for the streaming decoder
        had_speech = False
        sil = 0.0
        previewed = False
        last_interim = ""

        def reset_stream():
            nonlocal s
            if s is not None:
                try:
                    s.__exit__(None, None, None)
                except Exception:
                    pass
                s = p._model.transcribe_stream(context_size=_STREAM_CTX, depth=_STREAM_DEPTH)
                s.__enter__()

        try:
            async for item in self._input_ch:
                flush = isinstance(item, self._FlushSentinel)
                if not flush:
                    x = np.frombuffer(item.data, np.int16).astype(np.float32) / 32768.0
                    if item.num_channels > 1:
                        x = x.reshape(-1, item.num_channels).mean(axis=1)
                    dur = len(x) / 16000
                    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
                    seg.append(x)
                    if rms >= _ENERGY_GATE:
                        had_speech = True; sil = 0.0; previewed = False
                    else:
                        sil += dur
                    if _STREAM_INTERIM and had_speech:
                        pending.append(x)
                        if sum(len(pp) for pp in pending) >= _STREAM_FEED:
                            s.add_audio(mx.array(np.concatenate(pending))); pending.clear()
                            txt = s.result.text
                            if txt and txt != last_interim:
                                last_interim = txt
                                self._emit(stt.SpeechEventType.INTERIM_TRANSCRIPT, txt)

                seg_len = sum(len(c) for c in seg)
                # micro-pause -> PREFLIGHT (streaming text, cheap) drives preemptive generation
                if (had_speech and not previewed and sil >= _STREAM_PREFLIGHT_SIL
                        and seg_len >= _STREAM_MIN_SPEECH * 16000 and last_interim):
                    previewed = True
                    self._emit(stt.SpeechEventType.PREFLIGHT_TRANSCRIPT, last_interim)
                # end-of-turn -> authoritative batch FINAL (also on flush for short utterances)
                if had_speech and (flush or sil >= _STREAM_FINAL_SIL):
                    text = self._final_text(np.concatenate(seg))
                    if text and p._spk_sigs:          # ambient speaker gate
                        try:
                            name, _ = await asyncio.to_thread(
                                speaker_id.verify_and_adapt, np.concatenate(seg),
                                p._spk_sigs, p._spk_threshold)
                            if name is None:
                                _stt_log.info("STT-DROP speaker not enrolled")
                                text = ""
                        except Exception:
                            pass
                    self._emit(stt.SpeechEventType.FINAL_TRANSCRIPT, text)
                    seg = []; pending.clear(); had_speech = False; sil = 0.0
                    previewed = False; last_interim = ""
                    reset_stream()
                elif flush:
                    # flush without un-finalized speech (buffered silence/noise, or already finalized
                    # this segment). Emit NOTHING — a second (empty) final would re-fire the
                    # framework's EOU detection and risk a double reply. Just drop the buffer.
                    seg = []; pending.clear(); had_speech = False; sil = 0.0
                    previewed = False; last_interim = ""
                    reset_stream()
        finally:
            if s is not None:
                try:
                    s.__exit__(None, None, None)
                except Exception:
                    pass


class LocalQwenASR(stt.STT):
    """Qwen3-ASR-1.7B via the pure-MLX port. ~2.4x faster than whisper-large-v3-turbo
    (~170ms vs ~407ms per utterance) at comparable accuracy. Non-streaming, like the
    Whisper wrapper. The port has no wake-word biasing, so we fix "Chronic"->"Kronik"
    in post. To roll back, point agent.py's setup() at LocalWhisperSTT instead."""

    def __init__(self, *, model: str = "mlx-community/Qwen3-ASR-1.7B-bf16"):
        super().__init__(
            capabilities=stt.STTCapabilities(streaming=False, interim_results=False)
        )
        from qwen3_asr_mlx import Qwen3ASR

        self._asr = Qwen3ASR.from_pretrained(model)
        try:
            self._asr.warm_up()
        except Exception:
            pass

    def _empty(self, language):
        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text="", language=language or "en")],
        )

    async def _recognize_impl(self, buffer, *, language=None, conn_options=None):
        data = _prep_audio(buffer)
        if data is None:
            return self._empty(language)

        # qwen decode blocks (~170ms); keep it off the event loop. Force English:
        # auto-detect occasionally hears mumble as Chinese/Hindi/etc. and transcribes
        # the whole turn in that language (observed live 2026-07-10).
        result = await asyncio.to_thread(
            self._asr.transcribe, data, language=(language or "en"))
        text = _fix_wake_word((result.text or "").strip())
        if _looks_hallucinated(text):
            _stt_log.info("STT-DROP hallucination filter: %r", text[:80])
            return self._empty(language)

        return stt.SpeechEvent(
            type=stt.SpeechEventType.FINAL_TRANSCRIPT,
            alternatives=[stt.SpeechData(text=text, language=language or "en")],
        )
