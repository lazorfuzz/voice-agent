"""Singing via SoulX-Singer voice conversion (local, Apple Silicon MPS).

The agent's persona "sings" by converting a prepared song vocal into the persona's
cloned timbre — zero-shot, from a short reference clip of the voice (a spoken clip
works; verified live with Consuela's 19s conditioning clip).

Requires a SoulX-Singer checkout with downloaded checkpoints (SING_DIR, default
~/soulx-singer — see the soulx-singer memory/README for setup). Everything runs in
THAT repo's own venv as subprocesses: it needs a different torch than the agent, and
the ~50s/song generation must never share the agent process (GPU contention).

Song library: voice-agent/songs/<name>/ containing vocal.(wav|mp3) + f0.npy — an
ISOLATED singing vocal plus its RMVPE F0 track. Renders are cached per (voice, song)
in .sing_cache/, so repeat performances are instant.
"""
import hashlib
import os
import subprocess

_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_DIR)
SING_DIR = os.path.expanduser(os.environ.get("SING_DIR", "~/soulx-singer"))
SONGS_DIR = os.path.join(_ROOT, "songs")
CACHE_DIR = os.path.join(_ROOT, ".sing_cache")

_PY = os.path.join(SING_DIR, ".venv", "bin", "python")
_SVC_CKPT = os.path.join(SING_DIR, "pretrained_models", "SoulX-Singer", "model-svc.pt")
_RMVPE = os.path.join(SING_DIR, "pretrained_models", "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt")

# Spoken reference synthesized for catalog voices — long enough for a stable timbre read.
REF_TEXT = ("Let me tell you about my day. I woke up early, made some coffee, and watched "
            "the sunrise from the window. The sky turned orange and pink over the rooftops, "
            "and for a little while everything felt calm and quiet and completely still.")


def available() -> bool:
    """Singing works iff the SoulX checkout + checkpoints + at least one song exist."""
    has_melody = (os.path.isdir(MELODIES_DIR) and any(
        f.endswith(".json") for f in os.listdir(MELODIES_DIR)))
    return (os.path.isfile(_PY) and os.path.isfile(_SVS_CKPT) and os.path.isfile(_RMVPE)
            and has_melody)


def list_songs() -> list:
    if not os.path.isdir(SONGS_DIR):
        return []
    out = []
    for name in sorted(os.listdir(SONGS_DIR)):
        d = os.path.join(SONGS_DIR, name)
        if _vocal_path(d) and os.path.isfile(os.path.join(d, "f0.npy")):
            out.append(name)
    return out


def pick_song(query) -> str | None:
    """Fuzzy-pick a song by name; None/empty -> first song."""
    songs = list_songs()
    if not songs:
        return None
    if not query:
        return songs[0]
    q = str(query).lower().strip()
    for s in songs:
        if q == s.lower():
            return s
    for s in songs:
        if q in s.lower() or s.lower() in q:
            return s
    import difflib
    m = difflib.get_close_matches(q, songs, n=1, cutoff=0.4)
    return m[0] if m else songs[0]


def _vocal_path(song_dir):
    for ext in ("wav", "mp3", "flac"):
        p = os.path.join(song_dir, f"vocal.{ext}")
        if os.path.isfile(p):
            return p
    return None


def _run(cmd, timeout):
    r = subprocess.run(cmd, cwd=SING_DIR, capture_output=True, text=True, timeout=timeout,
                       env={**os.environ, "PYTHONPATH": SING_DIR})
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-400:])
    return r


def _extract_f0(wav_path, f0_path):
    code = ("import sys; from preprocess.tools.f0_extraction import F0Extractor; "
            f"F0Extractor(model_path={_RMVPE!r}, device='cpu', verbose=False)"
            ".process(sys.argv[1], f0_path=sys.argv[2])")
    _run([_PY, "-c", code, wav_path, f0_path], timeout=300)


def reference_for_voice(voice: str, synth_fn=None) -> tuple:
    """Resolve/prepare the (ref_wav, ref_f0) pair for a persona voice.

    `voice` is a pocket-tts catalog name or a file path (wav = use directly;
    .safetensors voice state = use a sibling .wav if one exists, else synthesize).
    `synth_fn(text, out_wav_path)` renders REF_TEXT in the persona's voice when no
    audio reference exists on disk (catalog voices). Results cached in .sing_cache/refs.
    """
    refs = os.path.join(CACHE_DIR, "refs")
    os.makedirs(refs, exist_ok=True)
    key = hashlib.sha1(voice.encode()).hexdigest()[:12]

    wav = None
    if os.path.isfile(voice) and not voice.endswith(".safetensors"):
        wav = voice
    elif voice.endswith(".safetensors") and os.path.isfile(voice[:-len(".safetensors")] + ".wav"):
        wav = voice[:-len(".safetensors")] + ".wav"
    else:
        wav = os.path.join(refs, f"{key}.wav")
        if not os.path.isfile(wav):
            if synth_fn is None:
                raise RuntimeError(f"no audio reference for voice {voice!r} and no synth_fn")
            synth_fn(REF_TEXT, wav)

    f0 = os.path.join(refs, f"{key}_f0.npy")
    if not os.path.isfile(f0) or os.path.getmtime(f0) < os.path.getmtime(wav):
        _extract_f0(wav, f0)
    return wav, f0


def is_cached(voice: str, song: str) -> bool:
    key = hashlib.sha1(f"{voice}|{song}".encode()).hexdigest()[:16]
    return os.path.isfile(os.path.join(CACHE_DIR, key, "generated.wav"))


_SVS_CKPT = os.path.join(SING_DIR, "pretrained_models", "SoulX-Singer", "model.pt")
_PHONESET = os.path.join(SING_DIR, "soulxsinger", "utils", "phoneme", "phone_set.json")
_SCORE_BUILDER = os.path.join(_DIR, "sing_score.py")


def annotated_reference(voice: str, synth_fn=None) -> tuple:
    """(spoken_ref_wav, prompt_metadata_json) for a persona voice. The SVS prompt is the
    NATURAL spoken clip with auto-built score annotation (parakeet timings + RMVPE
    notes) — prompting with a vocoder-rendered singing bootstrap made every song
    inherit the vocoder's sterile texture and a wrong register (similarity 0.773 vs
    0.905 with the natural prompt). Annotation cached next to the reference."""
    ref_wav, _ = reference_for_voice(voice, synth_fn)
    meta = os.path.join(CACHE_DIR, "refs",
                        hashlib.sha1(voice.encode()).hexdigest()[:12] + "_meta.json")
    if not os.path.isfile(meta) or os.path.getmtime(meta) < os.path.getmtime(ref_wav):
        r = subprocess.run([os.path.join(_ROOT, ".venv", "bin", "python"),
                            os.path.join(_DIR, "sing_annotate.py"), ref_wav, meta],
                           capture_output=True, text=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    return ref_wav, meta


MELODIES_DIR = os.path.join(_ROOT, "melodies")


def pick_melody(lyrics: str, mood: str = None) -> str:
    """Melody spec for a song: a curated template file or a generated tune. Seeded by
    the lyrics so re-requesting the same song reuses the same melody (cache-stable),
    while different songs get different melodies. `mood` biases the choice."""
    seed = int(hashlib.sha1(lyrics.encode()).hexdigest()[:8], 16)
    import random
    rng = random.Random(seed)
    files = sorted(os.path.join(MELODIES_DIR, f) for f in os.listdir(MELODIES_DIR)
                   if f.endswith(".json")) if os.path.isdir(MELODIES_DIR) else []
    # melodies/tunes/ = named song melodies (gitignored transcriptions): part of the
    # random pool AND matchable by name
    tunes_dir = os.path.join(MELODIES_DIR, "tunes")
    tunes = sorted(os.path.join(tunes_dir, f) for f in os.listdir(tunes_dir)
                   if f.endswith(".json")) if os.path.isdir(tunes_dir) else []
    if mood:
        m = mood.strip().lower().replace(" ", "-")
        named = {os.path.splitext(os.path.basename(f))[0]: f for f in files + tunes}
        aliases = {"upbeat": "bright-hook", "happy": "bright-hook", "pop": "pop-anthem",
                   "ballad": "gentle-ballad", "sad": "gentle-ballad", "slow": "gentle-ballad",
                   "folk": "folk-tune", "gentle": "gentle-ballad"}
        if m in named:
            return named[m]
        if m in aliases and aliases[m] in named:
            return named[aliases[m]]
        if m in ("silly", "lullaby"):
            return f"gen:{m}:{seed}"
    # default: draw from templates + two generated flavors
    pool = files + tunes + [f"gen:upbeat:{seed}", f"gen:ballad:{seed}"]
    return rng.choice(pool) if pool else os.path.join(SONGS_DIR, "demo-song", "score.json")


def start_streaming_render(voice: str, lyrics: str, synth_fn=None, mood: str = None):
    """Begin rendering ORIGINAL lyrics; returns (out_dir, proc_or_None).

    proc is None on a full cache hit (generated.wav already present). Otherwise the
    streaming renderer (sing_render_stream.py, score control + auto_shift baked in)
    is left running: segment_NN.wav files appear in out_dir as each ~8-13s piece
    renders (~0.7x realtime), so playback can start after segment 0 while the rest
    generate. ALL_DONE marks success; generated.wav is the merged cache file."""
    template = pick_melody(lyrics, mood)
    key = hashlib.sha1(f"svs3|{voice}|{lyrics}|{template}".encode()).hexdigest()[:16]
    out_dir = os.path.join(CACHE_DIR, key)
    if os.path.isfile(os.path.join(out_dir, "generated.wav")):
        return out_dir, None

    prompt_wav, prompt_meta = annotated_reference(voice, synth_fn)
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):                      # clear partials from failed runs
        if f.startswith("segment_") or f == "ALL_DONE":
            os.remove(os.path.join(out_dir, f))
    score = os.path.join(out_dir, "score.json")
    r = subprocess.run([_PY, _SCORE_BUILDER, template, score], input=lyrics,
                       cwd=SING_DIR, capture_output=True, text=True, timeout=300,
                       env={**os.environ, "PYTHONPATH": SING_DIR})
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip()[-300:])
    proc = subprocess.Popen(
        [_PY, os.path.join(_DIR, "sing_render_stream.py"),
         "--prompt_wav_path", prompt_wav,
         "--prompt_metadata_path", prompt_meta,
         "--target_metadata_path", score,
         "--save_dir", out_dir,
         "--device", "mps"],
        cwd=SING_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        env={**os.environ, "PYTHONPATH": SING_DIR, "SING_DIR": SING_DIR})
    return out_dir, proc


def render_original(voice: str, lyrics: str, synth_fn=None, mood: str = None) -> str:
    """Blocking render (kept for cache warming / non-streaming callers)."""
    out_dir, proc = start_streaming_render(voice, lyrics, synth_fn, mood)
    out = os.path.join(out_dir, "generated.wav")
    if proc is not None:
        log = proc.communicate(timeout=1200)[0]
        if proc.returncode != 0 or not os.path.isfile(out):
            raise RuntimeError((log or "").strip()[-300:])
    return out


def render(voice: str, song: str, synth_fn=None) -> str:
    """Render `song` sung in `voice`'s timbre; returns the wav path (cached)."""
    song_dir = os.path.join(SONGS_DIR, song)
    vocal = _vocal_path(song_dir)
    if not vocal:
        raise RuntimeError(f"unknown song {song!r}")
    key = hashlib.sha1(f"{voice}|{song}".encode()).hexdigest()[:16]
    out_dir = os.path.join(CACHE_DIR, key)
    out = os.path.join(out_dir, "generated.wav")
    if os.path.isfile(out):
        return out

    ref_wav, ref_f0 = reference_for_voice(voice, synth_fn)
    _run([_PY, "-m", "cli.inference_svc",
          "--device", "mps",
          "--model_path", _SVC_CKPT,
          "--config", os.path.join(SING_DIR, "soulxsinger", "config", "soulxsinger.yaml"),
          "--prompt_wav_path", ref_wav,
          "--target_wav_path", vocal,
          "--prompt_f0_path", ref_f0,
          "--target_f0_path", os.path.join(song_dir, "f0.npy"),
          "--save_dir", out_dir,
          "--auto_shift", "--pitch_shift", "0"], timeout=1200)
    if not os.path.isfile(out):
        raise RuntimeError("SVC produced no output")
    return out
