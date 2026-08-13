"""Speaker enrollment + verification for ambient mode.

ECAPA-TDNN speaker embeddings (SpeechBrain `spkrec-ecapa-voxceleb`, 192-d). Trained on
VoxCeleb (in-the-wild, noisy, varied channels), it is FAR more robust to far-field / clipped
/ reverberant ambient audio than the old resemblyzer d-vector — measured cross-speaker
cosine ~0.24 vs same-speaker ~0.5-0.9 (a wide ~0.25+ margin), where resemblyzer overlapped
legit voice (0.59-0.66) with a stranger (0.51-0.57) and could not separate them.

Enrolled speakers are stored as per-sample .npy embeddings under voices/enrolled/. In ambient
mode the STT embeds each utterance and DROPS it (empty transcript) unless it matches an
enrolled speaker, so only enrolled people can wake or talk to the agent.

ONLINE ADAPTATION: when an utterance matches with comfortable margin, its embedding is folded
into that speaker's profile (a rolling buffer). The gate thus self-improves and adapts to the
actual room / mic / distance from a single quick enrollment — no need to record many samples
up front. Enrollment samples are permanent; adapted samples are capped and roll over, so a
profile can't grow unbounded or be permanently poisoned by a borderline accept.
"""
import os
import glob
import threading

import numpy as np

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices", "enrolled")
_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "ecapa")
os.makedirs(_DIR, exist_ok=True)

# ECAPA needs ~1s to give a stable embedding; below this, skip (treat as unknown/dropped).
MIN_SAMPLES_16K = 16000  # 1.0s @ 16kHz

# ECAPA cosine: same-speaker 0.5-0.9 in MATCHED conditions, but a clean close-mic enrollment
# vs noisy far-field ambient audio drops legit same-speaker to ~0.33-0.44 (measured: additive
# noise/reverb, not gain — ECAPA is gain-invariant). Cross-speaker maxes ~0.24. 0.30 fits that
# real-world gap. Single source of truth (local_stt reads this, not a hardcode).
DEFAULT_THRESHOLD = 0.30

# Online adaptation: fold an utterance into a profile when it clears the threshold by this
# margin. Kept SMALL so it can BOOTSTRAP out of a cross-condition cold start — a clean
# enrollment yields low ambient sims, and only by folding in the user's better ambient
# utterances does the profile gain noisy-condition coverage (verify = max over samples) and
# future sims rise. Still a positive margin so we never adapt at the very edge. Rolling cap.
_ADAPT_MARGIN = 0.05
_MAX_ADAPT = 20
# With >1 enrolled speaker, only adapt when the winner beats the runner-up speaker by at least
# this cosine gap — guarantees a borderline/ambiguous utterance never trains the wrong profile
# (keeps two enrolled voices from drifting together). Enrollments here separate by ~0.12 cross
# vs ~0.35+ same, so this never blocks a legit self-improve.
_ADAPT_SEPARATION = 0.10

_lock = threading.Lock()
_model = None


def _enc():
    global _model
    if _model is None:
        import warnings
        warnings.filterwarnings("ignore")
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        from speechbrain.inference.speaker import EncoderClassifier
        _model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=_MODEL_DIR, run_opts={"device": "cpu"})
    return _model


def warm():
    """Preload the model (call from prewarm so the first real utterance isn't cold)."""
    try:
        embed(np.zeros(16000, dtype=np.float32))
    except Exception:
        pass


def embed(wav16k):
    """L2-normalized 192-d ECAPA speaker embedding for 16k mono float32 audio."""
    import torch
    w = np.asarray(wav16k, dtype=np.float32)
    with torch.no_grad():
        e = _enc().encode_batch(torch.from_numpy(w).unsqueeze(0)).squeeze().cpu().numpy()
    return e / (np.linalg.norm(e) + 1e-9)


def _safe(name: str) -> str:
    n = "".join(c for c in (name or "").lower() if c.isalnum() or c in "-_ ").strip()
    return n.replace(" ", "_") or "me"


def enroll(name: str, wav16k) -> str:
    """Add PERMANENT enrollment embedding(s) for `name`. From one recording we store the full
    clip PLUS two overlapping sub-windows (when long enough) as separate exemplars — verify
    takes MAX over samples, so this gives more to match against at zero extra user effort."""
    name = _safe(name)
    w = np.asarray(wav16k, dtype=np.float32)
    clips = [w]
    if len(w) >= 4 * MIN_SAMPLES_16K:      # ~>=4s: add two overlapping halves for variation
        h = len(w) // 2
        clips += [w[: h + MIN_SAMPLES_16K // 2], w[h - MIN_SAMPLES_16K // 2:]]
    with _lock:
        # index past any existing enrollment samples (adapted files carry an `adapt_` infix,
        # so they never collide with these integer-indexed enrollment names)
        idx = len(glob.glob(os.path.join(_DIR, f"{name}__[0-9]*.npy")))
        for c in clips:
            np.save(os.path.join(_DIR, f"{name}__{idx}.npy"), embed(c))
            idx += 1
    return name


def load_signatures() -> dict:
    """{name: [emb, ...]} for every enrolled speaker (enrollment + adapted), empty if none.
    Skips any stored embedding whose dimensionality doesn't match the current model (e.g.
    stale resemblyzer 256-d files from before the ECAPA switch)."""
    sigs = {}
    for p in glob.glob(os.path.join(_DIR, "*.npy")):
        name = os.path.basename(p).rsplit("__", 1)[0]
        try:
            v = np.load(p)
            if v.ndim == 1 and v.shape[0] == 192:
                sigs.setdefault(name, []).append(v)
        except Exception:
            pass
    return sigs


def enrolled_names() -> list:
    return sorted(load_signatures().keys())


def delete(name: str) -> int:
    name = _safe(name)
    n = 0
    for p in glob.glob(os.path.join(_DIR, f"{name}__*.npy")):  # enrollment + adapted
        try:
            os.remove(p); n += 1
        except Exception:
            pass
    return n


def _cos(a, b) -> float:
    return float(a @ (b / (np.linalg.norm(b) + 1e-9)))


def _rank(emb, sigs):
    """[(name, best_sim), ...] across enrolled speakers, sorted high→low. best_sim = MAX
    cosine over that speaker's samples (enrollment + adapted)."""
    scored = [(name, max(_cos(emb, v) for v in embs)) for name, embs in sigs.items()]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def match(emb, sigs, threshold: float = DEFAULT_THRESHOLD):
    """(name, sim) of the best-matching enrolled speaker for a precomputed embedding, or
    (None, sim) if below threshold / no speakers."""
    r = _rank(emb, sigs) if sigs else []
    if not r:
        return (None, 0.0)
    name, sim = r[0]
    return (name, sim) if sim >= threshold else (None, sim)


def _adapt(name, emb, sigs):
    """Fold a high-confidence embedding into `name`'s profile: persist a rolling adapted
    sample (evict oldest beyond _MAX_ADAPT) and refresh the in-memory `sigs` dict IN PLACE
    so the live gate uses it immediately."""
    with _lock:
        existing = glob.glob(os.path.join(_DIR, f"{name}__adapt_*.npy"))
        nxt = 0
        for f in existing:
            try:
                nxt = max(nxt, 1 + int(os.path.basename(f).rsplit("_", 1)[1].split(".")[0]))
            except Exception:
                pass
        np.save(os.path.join(_DIR, f"{name}__adapt_{nxt}.npy"), emb)
        rolled = sorted(glob.glob(os.path.join(_DIR, f"{name}__adapt_*.npy")),
                        key=os.path.getmtime)
        for f in rolled[:-_MAX_ADAPT]:
            try:
                os.remove(f)
            except Exception:
                pass
    # keep the same dict object (the gate holds a reference to it) but sync it to disk
    fresh = load_signatures()
    sigs.clear()
    sigs.update(fresh)


def verify(wav16k, signatures=None, threshold: float = DEFAULT_THRESHOLD):
    """(name, sim) of the best-matching enrolled speaker, or (None, sim) if below threshold.
    Returns (None, 0.0) when there are no enrolled speakers (caller decides: no gate)."""
    sigs = signatures if signatures is not None else load_signatures()
    if not sigs or len(np.asarray(wav16k)) < MIN_SAMPLES_16K:
        return (None, 0.0)
    return match(embed(wav16k), sigs, threshold)


def verify_and_adapt(wav16k, signatures, threshold: float = DEFAULT_THRESHOLD):
    """Like verify(), but on a CONFIDENT, UNAMBIGUOUS match it folds the utterance's embedding
    into the matched profile (online adaptation) so the gate adapts to the room. `signatures`
    is mutated in place. Returns (name, sim)."""
    sigs = signatures
    if not sigs or len(np.asarray(wav16k)) < MIN_SAMPLES_16K:
        return (None, 0.0)
    e = embed(wav16k)
    r = _rank(e, sigs)
    name, sim = r[0]
    if sim < threshold:
        return (None, sim)
    # Self-improve ONLY when the match is (a) comfortably above the accept bar AND (b) beats
    # the runner-up ENROLLED speaker by a clear gap — so an utterance that sits between two
    # enrolled voices is never folded into either profile. This is what stops the "wrong one"
    # from adapting: with >1 enrolled speaker we require the separation; poisoning one profile
    # with another person's voice can't happen at a near-tie.
    runner = r[1][1] if len(r) > 1 else -1.0
    if sim >= threshold + _ADAPT_MARGIN and (sim - runner) >= _ADAPT_SEPARATION:
        try:
            _adapt(name, e, sigs)
        except Exception:
            pass
    return name, sim
