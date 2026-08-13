# Experimental

Research modes that ship as-is, off by default; nothing in the main stack depends on them.

- **Gemma Ears** (this file, below) — audio-native voice pipeline; the LLM hears raw audio, no STT.
- **Mage-Ring** ([`MAGE_RING.md`](MAGE_RING.md)) — live Ring-camera commentary via a local
  vision model, spoken through the agent. Surfaces as the `watch_camera_live` tool when the
  `ring` integration is enabled; the ~11 GB Mage-VL subprocess is spawned lazily on first use.

---

## Gemma Ears (audio-native voice mode)

An alternate voice pipeline where a unified multimodal LLM
(`mlx-community/gemma-4-12b-it-qat-4bit`, ~8 GB) **hears your raw audio directly** —
no STT step. One model transcribes, reasons, and answers in a single generation,
streaming its reply into TTS. Supports a small set of home tools, barge-in, and
KV-prefix caching for fast turns (~1.5s mouth-to-ear).

This is a research mode: quality and robustness are below the main pipeline. It ships
as-is, off by default, and nothing in the main stack depends on it.

## Run it

Needs its own venv with [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) (the main venv's
deps conflict):

```bash
python3 -m venv ~/mlx-vlm-svc/.venv
~/mlx-vlm-svc/.venv/bin/pip install mlx-vlm

# 1. the ears server (loads gemma-4-12b, serves :8084)
~/mlx-vlm-svc/.venv/bin/python experimental/gemma_ears_server.py

# 2. the room agent (joins LiveKit room "gemma"; uses the main .venv)
.venv/bin/python experimental/gemma_ears_agent.py
```

Then pick **🧪 Gemma Ears (audio-native)** from the persona dropdown in the web UI.

## Files

| File | What |
|---|---|
| `gemma_ears_server.py` | gemma-4-12b in-process; streams spoken reply text over chunked HTTP; executes tools; text-only history + KV prefix cache |
| `gemma_ears_agent.py` | LiveKit room agent: energy-gate utterance segmentation, streaming TTS, barge-in, deferred transcripts |
| `gemma_audiotest.py` / `gemma_earstest.py` / `gemma_tooltest.py` | the benchmark/validation suite that proved the mode viable |

Hard-won implementation notes (stale-audio KV pitfalls, markup holdback, echo gates)
are documented as comments in the two main files.
