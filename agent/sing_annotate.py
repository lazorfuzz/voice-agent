"""Annotate a SPOKEN reference clip as SoulX-Singer prompt metadata (runs in the AGENT
venv: parakeet word timings; shells to the SoulX venv for RMVPE f0 + g2p phonemes).

Usage: python sing_annotate.py <ref.wav> <out_meta.json>

Why: prompting SVS with the persona's natural spoken clip instead of a vocoder-rendered
singing bootstrap fixed timbre similarity 0.773->0.905, brightness to ground truth, and
register to the voice's true range (the "through a filter" bug, 2026-08-06)."""
import json
import os
import re
import subprocess
import sys

import numpy as np

SING_DIR = os.path.expanduser(os.environ.get("SING_DIR", "~/soulx-singer"))
_SOULX_PY = os.path.join(SING_DIR, ".venv", "bin", "python")
_RMVPE = os.path.join(SING_DIR, "pretrained_models", "SoulX-Singer-Preprocess", "rmvpe", "rmvpe.pt")
HOP = 0.02


def annotate(wav, out_path):
    from parakeet_mlx import from_pretrained
    m = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
    r = m.transcribe(wav)

    words = []
    for s in r.sentences:
        cur, start, end = "", None, None
        for t in s.tokens:
            if t.text.startswith(" ") and cur:
                words.append((cur, start, end)); cur, start = "", None
            piece = re.sub(r"[^A-Za-z']", "", t.text)
            if piece:
                if start is None:
                    start = t.start
                cur += piece; end = t.end
        if cur:
            words.append((cur, start, end))
    words = [(w, s, e) for w, s, e in words if w and s is not None and e > s]
    if not words:
        raise RuntimeError("no words recognized in reference clip")

    f0_path = out_path + ".f0.npy"
    subprocess.run([_SOULX_PY, "-c",
        "import sys; sys.path.insert(0, %r); " % SING_DIR +
        "from preprocess.tools.f0_extraction import F0Extractor; "
        "F0Extractor(model_path=%r, device='cpu', verbose=False)" % _RMVPE +
        ".process(sys.argv[1], f0_path=sys.argv[2])", wav, f0_path],
        cwd=SING_DIR, check=True, capture_output=True, timeout=600)
    f0 = np.load(f0_path)

    g2 = subprocess.run([_SOULX_PY, "-c",
        "import sys, json\n"
        "from g2p_en import G2p\n"
        "g2p = G2p()\n"
        "out = []\n"
        "for w in json.loads(sys.stdin.read()):\n"
        "    ph = [p for p in g2p(w) if p.strip() and p[0].isalpha()]\n"
        "    out.append('en_' + '-'.join(ph) if ph else 'en_AH0')\n"
        "print(json.dumps(out))"],
        input=json.dumps([w for w, _, _ in words]), capture_output=True, text=True,
        cwd=SING_DIR, timeout=600)
    phons = json.loads(g2.stdout.strip().splitlines()[-1])

    toks, durs, phs, pitches, types = [], [], [], [], []
    t_cursor = 0.0
    for (w, s, e), ph in zip(words, phons):
        if s > t_cursor + 0.03:
            toks.append("<SP>"); durs.append(f"{s - t_cursor:.2f}"); phs.append("<SP>")
            pitches.append("0"); types.append("1")
        seg = f0[int(s / HOP):max(int(s / HOP) + 1, int(e / HOP))]
        voiced = seg[seg > 0]
        midi = int(round(69 + 12 * np.log2(np.median(voiced) / 440.0))) if len(voiced) else 60
        toks.append(w.lower()); durs.append(f"{e - s:.2f}"); phs.append(ph)
        pitches.append(str(midi)); types.append("2")
        t_cursor = e
    total = len(f0) * HOP
    if total > t_cursor + 0.03:
        toks.append("<SP>"); durs.append(f"{total - t_cursor:.2f}"); phs.append("<SP>")
        pitches.append("0"); types.append("1")

    meta = [{"index": "spoken_ref", "language": "English", "time": [0, int(total * 1000)],
             "duration": " ".join(durs), "text": " ".join(toks), "phoneme": " ".join(phs),
             "note_pitch": " ".join(pitches), "note_type": " ".join(types)}]
    json.dump(meta, open(out_path, "w"))
    os.remove(f0_path)
    print(f"annotated {len(toks)} tokens over {total:.1f}s")


if __name__ == "__main__":
    annotate(sys.argv[1], sys.argv[2])
