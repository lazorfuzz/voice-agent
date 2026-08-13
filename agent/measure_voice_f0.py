"""Measure a pocket-tts voice's speaking-F0 range for the light_fx pitch->hue mapping.

Usage:
    .venv/bin/python measure_voice_f0.py <voice> [<voice> ...]

<voice> is a catalog name ("jean", "alba", ...) or a path to a clone audio file
("voices/tina.wav"). Prints a ready-to-paste line for light_fx._F0_RANGES. Uses the
same analyze() the live driver runs, on two intonation-varied sentences (statement +
question), p5-p95 over voiced 100ms buckets.
"""
import sys
from pathlib import Path

import numpy as np
import torch
from pocket_tts import TTSModel

import light_fx

TEXTS = [
    "Hey, it's Kronik. Honestly, your timing is perfect. What do you want to know?",
    "Is that really what you meant? Okay, let me check the thermostat and the lights real quick.",
]


def measure(model, voice: str):
    src = Path(voice)
    state = (model.get_state_for_audio_prompt(src, truncate=True) if src.is_file()
             else model.get_state_for_audio_prompt(voice))
    f0s = []
    for t in TEXTS:
        audio = torch.cat(list(model.generate_audio_stream(state, t))).numpy().astype(np.float32)
        for i in range(0, len(audio) - 2400, 2400):
            _, f0 = light_fx.analyze(audio[i:i + 2400], 24000)
            if f0 > 0:
                f0s.append(f0)
    f0s = np.array(f0s)
    key = Path(voice).stem.lower()
    # p10/p90 (not p5/p95): the range must HUG the real speaking distribution so pitch
    # variation spreads across the whole hue wheel. Too-wide a range compresses the real
    # pitches into one slice -> the lights sit on ~one color (the "mostly purple" bug).
    print(f'    "{key}": ({np.percentile(f0s, 10):.0f}.0, {np.percentile(f0s, 90):.0f}.0),'
          f'   # median {np.median(f0s):.0f}Hz, n={len(f0s)}')


if __name__ == "__main__":
    voices = sys.argv[1:]
    if not voices:
        print(__doc__)
        sys.exit(1)
    m = TTSModel.load_model()
    print("paste into light_fx._F0_RANGES:")
    for v in voices:
        measure(m, v)
