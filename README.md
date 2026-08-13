# Local Voice Agent

A real-time voice assistant for macOS on Apple Silicon. You talk to it in a browser. It
hears you, it thinks, and it replies in about one second. Every model runs on-device. No
audio leaves your Mac unless you turn on remote access.

- **Latency about 1 second.** Speech end to first audio is about one second. All stages stream.
- **Barge-in.** Talk over the assistant. It stops and it listens.
- **Personas.** Use a built-in persona, or make your own in the web UI (name, personality, voice).
- **Device tools, all opt-in.** Control WiZ lights, a Nest thermostat, a Tesla, Ring cameras, a Google TV, a robot vacuum, and a pet feeder.
- **Coding and research agents.** Start and monitor [opencode](https://opencode.ai) sessions by voice.
- **Ambient mode.** The assistant listens always. Only enrolled voices can talk to it.
- **Remote access (optional).** Get a password-gated public URL through ngrok and a TURN relay.
- **Text chat.** Type to the same brain and tools.

## Architecture

```
 Browser (web UI, LiveKit JS)
    │ HTTPS: page / token / signaling
    ├── This Mac : http://127.0.0.1:8000
    ├── LAN      : https://<LAN-IP>:8443     (Caddy, self-signed)
    └── Anywhere : https://your.ngrok.app    (optional; password-gated)
                      │
     Caddy ── / → frontend   ── /token → token server (:8790)   ── /rtc → LiveKit
                      │
     LiveKit server (native binary)  ←→  coturn TURN (optional, for remote voice)
                      │
     Python agent worker (LiveKit Agents)
       ├─ VAD  : silero + DTLN denoiser
       ├─ STT  : Parakeet-TDT 0.6B (MLX) with hotword biasing   [whisper/qwen3-asr selectable]
       ├─ LLM  : any OpenAI-compatible endpoint (tool calling required)
       ├─ TTS  : pocket-tts (~20ms to first audio; stock voices or clone from a wav)
       └─ Tools: per-integration, only what you onboard
```

## Quickstart

**You need:** an Apple Silicon Mac, macOS 15 or later, [Homebrew](https://brew.sh), and
about 10 GB of disk (the speech models and the local LLM).

```bash
git clone <this-repo> voice-agent && cd voice-agent
bash setup.sh     # installs everything; offers a local LLM (one Y/n question)
bash run_all.sh   # starts the stack
```

Open **http://127.0.0.1:8000**. Select a microphone. Tap **Connect**. Talk.

`setup.sh` installs the system dependencies (LiveKit, Caddy, Node, uv). It creates the
Python environment, builds the web UI, writes config from templates, and prefetches the
speech models. If you agree, it also installs [Ollama](https://ollama.com) with
`qwen3.5:9b-mlx`, so you do not configure an LLM yourself. It never overwrites existing config.

`run_all.sh` is idempotent. Run it again at any time. It starts only the parts that are
down. It self-heals when your Mac's DHCP address changes.

To reach the assistant from another LAN device, open `https://<LAN-IP>:8443`. Trust the
certificate at `/ca.crt` first. `run_all.sh` prints the address.

If a component fails, run the health check:

```bash
bash scripts/doctor.sh    # checks each component and prints a fix hint
```

<details>
<summary><b>Use your own LLM instead of Ollama</b></summary>

The agent works with any OpenAI-compatible endpoint that supports tool calling — `mlx_lm
server`, LM Studio, or a remote API. Point `agent/.env.local` at it:

```
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
OPENAI_API_KEY=anything-nonempty
LLM_MODEL=your-model-id
```

To serve a raw MLX model (for example, your own finetune), you need no extra software. The
venv already includes `mlx_lm`:

```bash
bash scripts/serve_mlx.sh mlx-community/Qwen3-8B-4bit   # or a local model path
```

The reference deployment uses this same pattern. It runs
[**0G-AI/0GM-1.0-35B-A3B-0427**](https://huggingface.co/0G-AI/0GM-1.0-35B-A3B-0427), an
apache-2.0 MoE finetune of Qwen3.6-35B-A3B (~3B active params) with strong tool calling.
The agent prompts and the latency tuning target that model. The 4-bit conversion is about
18 GB on disk. Convert once, then serve:

```bash
.venv/bin/python -m mlx_lm convert --hf-path 0G-AI/0GM-1.0-35B-A3B-0427 -q --mlx-path ~/models/0gm-4bit
bash scripts/serve_mlx.sh ~/models/0gm-4bit
```

Tool calling works with Qwen-family models. To override the model for voice only, set
`VOICE_LLM_MODEL` and `VOICE_LLM_URL`. If you expose the server past localhost, put a
bearer-auth proxy in front of it.
</details>

## Configuration

All settings are in **`agent/.env.local`**. Copy it from
[`agent/.env.local.example`](agent/.env.local.example), which documents every key. Main keys:

| Key | Purpose |
|---|---|
| `OPENAI_BASE_URL` / `LLM_MODEL` | your LLM endpoint and model |
| `STT_ENGINE` | `parakeet` (default) / `whisper` / `qwen3-asr` |
| `POCKET_VOICE` | a TTS voice name, or a path to a wav to clone |
| `ASSISTANT_NAME` / `WAKE_WORD` | rename the assistant |
| `<NAME>_ENABLED` | one flag per integration; all off by default |

Manage personas from the web UI. Tap the persona dropdown, then *New persona*.

## Integrations

At the start the assistant has **no tools**. It only converses. Enable one capability at a
time with the guided CLI. Each command stores its secrets in `agent/.env.local` and sets
its own `_ENABLED` flag:

```bash
.venv/bin/python agent/onboard.py <opencode|wiz|nest|tesla|ring|tv|petlibro|roomba>
```

See **[ONBOARDING.md](ONBOARDING.md)** for the needs of each integration (accounts, LAN
discovery, OAuth flows) and how to disable one. Restart the agent worker afterward:

```bash
pkill -f "agent.py dev"; bash run_all.sh
```

## Remote access (optional)

The default setup serves your Mac and your LAN. To allow access from anywhere:

1. **ngrok.** Run `brew install ngrok`. Add your authtoken. Set `NGROK_DOMAIN` in
   `agent/.env.local` (blank gives a random URL). `run_all.sh` starts the tunnel.
2. **Password gate.** Basic auth protects the public page. Set `WEB_AUTH_SECRET` and
   `WEB_AUTH_PWHASH` (see the `Caddyfile.example` comments).
3. **TURN relay.** WebRTC media from cellular networks needs a public relay. Port-forward
   coturn (UDP/TCP 3478 and UDP 49160-49180). For more reliability behind home NAT, run
   coturn on a cheap public-IP VPS and set the `TURN_*` values
   (see `coturn/turnserver.conf.example`).

## Manage the stack

```bash
tail -f agent.log                    # the agent worker (includes latency lines)
tail -f latency.log                  # per-turn pipeline timings
pkill -f "agent.py dev"; bash run_all.sh    # restart the agent only
pkill -f livekit-server              # stop the media server (container stop livekit if containerized)
```

## Advanced and experimental

- **SoulX-Duplug turn detector** (`TURN_DETECTOR=soulx`). A trained full-duplex end-of-turn
  model. It cuts response latency against plain VAD endpointing and removes mid-sentence
  clipping. It needs a server on `:8765`. If a `~/soulx-poc` checkout exists, `run_all.sh`
  starts the server. If not, leave `TURN_DETECTOR` blank.
- **Gemma Ears** (`experimental/`). An audio-native mode. A multimodal LLM (gemma-4-12b
  through mlx-vlm) hears your raw audio directly, with no STT stage. It appears as an
  "Experimental" persona in the UI when its server and agent run. See
  [`experimental/README.md`](experimental/README.md).
- **Live camera commentary** (`experimental/`, mage-ring). The assistant narrates your Ring
  camera feed in real time with a local vision model. The `watch_camera_live` tool appears
  only when the `ring` integration is enabled. See
  [`experimental/MAGE_RING.md`](experimental/MAGE_RING.md).
- **Latency tuning notes.** `docs/` and the comments in `agent/agent.py` describe the
  measured pipeline (EOU wait, GPU keep-warm pings, streaming TTS).

## Troubleshooting

Run `bash scripts/doctor.sh` first. It checks every component and prints a fix for each
problem. Common cases:

| Symptom | Fix |
|---|---|
| "no LLM" warning | run `bash setup.sh` (installs Ollama), or check `OPENAI_BASE_URL` |
| microphone connects, no replies | run `tail -f agent.log` — usually the LLM endpoint or model id |
| LAN page does not load | trust the certificate: open `https://<LAN-IP>:8443/ca.crt` |
| remote voice connects but stays silent | TURN is not reachable — see Remote access |
| first reply is slow after boot | models warm on the first call; the setup.sh prefetch avoids most delay |
| assistant echoes itself | use headphones, or rely on the built-in echo gate (on by default) |

## License

MIT — see [LICENSE](LICENSE).
