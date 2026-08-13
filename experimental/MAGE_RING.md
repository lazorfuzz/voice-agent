# Mage-VL + Ring live-commentary experiment

## What this proves

`mage_ring_live.py` creates its own WebRTC peer connection to Ring, receives and
decodes the live H.264 track on the Mac, samples two frames per second into short
windows, and passes the rolling windows to Microsoft Mage-VL. It captures the
next window while Mage analyzes the current one, so inference does not make the
video drift behind live time. The script emits one JSON line per gate
decision/commentary result.

It stays out of the production voice-agent process because Mage's BF16 checkpoint
is roughly 10.8 GB and because `aiortc` currently selects PyAV 17 while the voice
environment uses PyAV 18.

## Setup on this M4 Mac

The local probe uses the unofficial `mamba-ssm-macos` package because Microsoft's
released proactive gate imports the CUDA-only `mamba_ssm` package. Keep that shim
experimental until its logits are compared with the official CUDA implementation.

```bash
cd ~/voice-agent
uv venv .mage-venv
uv pip install --python .mage-venv/bin/python \
  -r experimental/mage_ring_requirements.txt
```

The model is public. If an old `~/.cache/huggingface/token` is invalid, the probe
sets `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so anonymous model access still works.

## Test sequence

First verify server-side Ring media without loading Mage:

```bash
.mage-venv/bin/python experimental/mage_ring_live.py \
  --camera bedroom --capture-only --segment-seconds 4 --max-segments 1
```

Then force one video description, bypassing the proactive gate:

```bash
.mage-venv/bin/python experimental/mage_ring_live.py \
  --camera bedroom --skip-gate --segment-seconds 4 --max-segments 2
```

Finally exercise running event-gated commentary:

```bash
.mage-venv/bin/python experimental/mage_ring_live.py \
  --camera bedroom --segment-seconds 8 --sample-fps 2 \
  --max-pixels 150000 --gate-threshold 0.5 --max-segments 4
```

Output is JSONL. A response looks like:

```json
{"type":"commentary","camera":"Bedroom","gate_probability":0.73,"decision":"response","inference_seconds":3.9,"text":"Someone walks into the room and sets a bag on the bed."}
```

## Important limitation

This first live probe uses decoded frame sampling. Mage's gate was trained on its
codec-native representation, so the 0.5 threshold is only a starting point here.
The production-quality follow-up is to retain or re-mux Ring's H.264 segments and
run Mage's traditional codec processor. That requires `codec-video-prep`, FFmpeg,
and either an incremental gate API from Microsoft or a bounded rolling-window
adaptation of `streammind_gate_forward_segments`.

## Voice-agent integration

When Ring is enabled, the voice agent exposes `watch_camera_live` and
`stop_camera_watch`. For example: “Watch the bedroom camera for one minute and
tell me what happens.” The watcher starts this probe in `.mage-venv`, speaks the
first description, suppresses near-duplicate descriptions, waits rather than
talking over active speech, and still announces when LiveKit marks a silent
listener as `away`. The requested duration begins after Ring media is ready, and
the probe reconnects automatically if frame delivery stalls. The subprocess is
stopped automatically when the requested duration or voice session ends.

The bridge deliberately uses `--skip-gate`. On the tested quiet scene, the
frame-mode proactive probability moved from 0.4742 to 0.7280 on separate runs,
so it is not calibrated well enough to decide when the agent should speak. The
current bridge uses eight-second windows. Mage receives its previous update as
context and returns `SILENT` when the scene has not meaningfully changed; the
bridge also suppresses near-duplicate wording. Wiring the existing Ring
motion/ding events in as a compute gate is the safer next optimization. Evaluate
Mage's native codec gate on Linux/NVIDIA before making it authoritative.

On the M4 Max test machine, Mage inference took roughly two seconds after the
model was cached. Initial camera wake and WebRTC setup took about nine seconds.

Microsoft labels the Mage family research-only and not intended for product or
service deployment. Keep this camera path a controlled local experiment, expect
occasional incorrect descriptions, and do not use its output for safety-critical
monitoring.
