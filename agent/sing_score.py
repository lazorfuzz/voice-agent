"""Build a SoulX-Singer score (target metadata JSON) from fresh English lyrics fitted to
a melody template. RUNS IN THE SOULX-SINGER VENV (needs g2p_en) — invoked by sing_tools
as a subprocess:  python sing_score.py <template.json> <out.json> <<< "lyric words..."

The template is an existing score whose per-slot melody (duration / note_pitch /
note_type + <SP> rests) is reused verbatim; only the words (and their phonemes) are
replaced. One template slot = one word. Fewer words than slots -> the melody is cut at
the last filled phrase; more words -> extra words are dropped. Pitch/duration slots are
language-neutral, so a Mandarin-sourced melody sings English lyrics fine.
"""
import json
import re
import sys


def generate_melody(mood, seed):
    """Procedural loopable melody -> (toks, durs, pitches, types) slot arrays.
    Simple but musical: AABA phrase structure, scale-degree random walk with a lift
    in the B phrase and a cadence home, mood-dependent scale/tempo."""
    import random
    rng = random.Random(seed)
    moods = {
        "upbeat":  {"scale": [0, 2, 4, 5, 7, 9, 11], "beat": 0.32, "root": rng.randint(62, 66)},
        "ballad":  {"scale": [0, 2, 3, 5, 7, 8, 10], "beat": 0.55, "root": rng.randint(59, 63)},
        "silly":   {"scale": [0, 2, 4, 7, 9],        "beat": 0.28, "root": rng.randint(63, 67)},
        "lullaby": {"scale": [0, 2, 4, 5, 7, 9],     "beat": 0.60, "root": rng.randint(60, 64)},
    }
    m = moods.get(mood, moods["upbeat"])
    scale, beat, root = m["scale"], m["beat"], m["root"]

    def phrase(start_deg, lift=0):
        deg = start_deg + lift
        slots = []
        n = rng.choice([7, 8, 9])
        for k in range(n):
            deg += rng.choice([-2, -1, -1, 0, 1, 1, 2])
            deg = max(-2, min(len(scale) + 3, deg))
            pitch = root + scale[deg % len(scale)] + 12 * (deg // len(scale))
            while pitch > 71:                     # stay inside the model's singable
                pitch -= 12                        # range (training data ~50-71 MIDI)
            while pitch < 52:
                pitch += 12
            dur = beat * rng.choice([1, 1, 1, 2, 1.5])
            if k == n - 1:
                dur = beat * 2.5                      # phrase-final long note
            slots.append((round(dur, 2), pitch))
        return slots

    A = phrase(2)
    B = phrase(4, lift=1)
    A2 = A[:-1] + [(round(beat * 3, 2), root + scale[0])]   # cadence to tonic
    toks, durs, pitches, types = [], [], [], []
    for ph in (A, A, B, A2):
        for d, p in ph:
            toks.append("w"); durs.append(str(min(d, 1.0))); pitches.append(str(p)); types.append("2")
        toks.append("<SP>"); durs.append("0.40"); pitches.append("0"); types.append("1")
    return toks, durs, pitches, types


def build(template_path, lyrics, out_path):
    from g2p_en import G2p
    g2p = G2p()

    if template_path.startswith("gen:"):
        _, mood, seed = (template_path.split(":") + ["0"])[:3]
        toks, durs, pitches, types = generate_melody(mood, int(seed))
    else:
        seg = json.load(open(template_path))[0]
        durs = seg["duration"].split()
        toks = seg["text"].split()
        pitches = seg["note_pitch"].split()
        types = seg["note_type"].split()

    words = [w for w in re.findall(r"[A-Za-z']+", lyrics)]
    if not words:
        raise SystemExit("no usable lyric words")

    # The template's slots are timed for MONOSYLLABLES (Mandarin: one char = one
    # syllable). An English word gets as many consecutive slots as it has syllables —
    # their durations merge into one note — so "housekeeper" sings across three slots'
    # time instead of being crammed into one (the live "sounds horrible" bug).
    out_t, out_d, out_p, out_pi, out_ty = [], [], [], [], []
    wi = 0
    i = 0
    passes = 0
    while True:
        if i >= len(toks):
            # template exhausted but lyrics remain: LOOP the melody as a new verse
            # (same tune, next words — how actual songs handle length)
            if wi >= len(words) or passes >= 6:
                break
            passes += 1
            i = 0
            out_t.append("<SP>"); out_d.append("0.60"); out_p.append("<SP>")
            out_pi.append("0"); out_ty.append("1")
        tok = toks[i]
        if tok == "<SP>":
            out_t.append("<SP>"); out_d.append(durs[i]); out_p.append("<SP>")
            out_pi.append("0"); out_ty.append("1")
            i += 1
            continue
        if wi >= len(words):
            break                                # lyrics exhausted: end the song
        w = words[wi].lower(); wi += 1
        phones = [p for p in g2p(w) if p not in (" ", "")]
        phones = [p for p in phones if re.fullmatch(r"[A-Z]+[0-2]?", p)]
        if not phones:
            phones = ["AH0"]
        syls = max(1, sum(1 for p in phones if p[-1].isdigit()))
        # consume up to `syls` consecutive word slots (stop at rests), summing durations
        dur = 0.0
        used = 0
        j = i
        while j < len(toks) and used < syls and toks[j] != "<SP>":
            dur += float(durs[j]); used += 1; j += 1
        # cap sustained notes: >1s vowels sit outside the training distribution and
        # destabilize generation into screech (measured at the 1.46s "i", 2026-08-06)
        dur = min(dur, 1.0)
        out_t.append(w)
        out_d.append(f"{dur:.2f}")
        out_p.append("en_" + "-".join(phones))
        out_pi.append(pitches[i])
        out_ty.append(types[i])
        i = j

    if out_t and out_t[-1] != "<SP>":            # close with a rest so the vocal releases
        out_t.append("<SP>"); out_d.append("0.30"); out_p.append("<SP>")
        out_pi.append("0"); out_ty.append("1")

    # Split into SHORT segments at rests (~<=10s each): the model's training segments
    # are ~7s, and one 30-50s generation drifts into gibberish mid-song (heard live).
    # Each segment renders independently into its absolute time window.
    segs = []
    cur = {"t": [], "d": [], "p": [], "pi": [], "ty": []}
    cur_dur = 0.0
    seg_start = 0.0

    def flush():
        nonlocal cur, cur_dur, seg_start
        if not any(t != "<SP>" for t in cur["t"]):
            cur = {"t": [], "d": [], "p": [], "pi": [], "ty": []}
            cur_dur = 0.0
            return
        segs.append({
            "index": f"generated_{len(segs)}",
            "language": "English",
            "time": [int(seg_start * 1000), int((seg_start + cur_dur) * 1000)],
            "duration": " ".join(cur["d"]),
            "text": " ".join(cur["t"]),
            "phoneme": " ".join(cur["p"]),
            "note_pitch": " ".join(cur["pi"]),
            "note_type": " ".join(cur["ty"]),
        })
        seg_start += cur_dur
        cur = {"t": [], "d": [], "p": [], "pi": [], "ty": []}
        cur_dur = 0.0

    for t, d, p, pi, ty in zip(out_t, out_d, out_p, out_pi, out_ty):
        cur["t"].append(t); cur["d"].append(d); cur["p"].append(p)
        cur["pi"].append(pi); cur["ty"].append(ty)
        cur_dur += float(d)
        if t == "<SP>" and cur_dur >= 8.0:
            flush()
    flush()
    json.dump(segs, open(out_path, "w"), ensure_ascii=False)
    total = sum(s["time"][1] - s["time"][0] for s in segs) / 1000
    print(f"score: {len(out_t)} slots, {wi}/{len(words)} words, "
          f"{len(segs)} segments, {total:.1f}s")


if __name__ == "__main__":
    build(sys.argv[1], sys.stdin.read(), sys.argv[2])
