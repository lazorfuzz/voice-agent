# Local Voice Agent

A real-time, self-hosted voice assistant. You talk to it in a browser. It hears you, it
thinks, and it replies in about one second. Every model runs on your own hardware — no audio
leaves the machine unless you turn on remote access.

Runs two ways: **natively on Apple Silicon** (a one-line setup script) and **on Linux via
Docker** (CPU on any box, or ~1s latency on an NVIDIA GPU). The only platform-specific piece
is speech-to-text — everything else is identical.

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
    ├── This host : http://127.0.0.1:8000   (macOS)  ·  http://localhost:8080 (Docker)
    ├── LAN       : https://<LAN-IP>:8443    (Caddy, self-signed)
    └── Anywhere  : https://your.ngrok.app   (optional; password-gated)
                      │
     Caddy ── / → frontend   ── /token → token server (:8790)   ── /rtc → LiveKit
                      │
     LiveKit server (native binary or container)  ←→  coturn TURN (optional, remote voice)
                      │
     Python agent worker (LiveKit Agents)
       ├─ VAD  : silero + DTLN denoiser
       ├─ STT  : Apple Silicon → Parakeet-TDT 0.6B (MLX), hotword biasing [whisper/qwen3-asr selectable]
       │         Linux → faster-whisper (CTranslate2), CPU or NVIDIA CUDA
       ├─ LLM  : any OpenAI-compatible endpoint (tool calling required)
       ├─ TTS  : pocket-tts (~20ms to first audio; stock voices or clone from a wav)
       └─ Tools: per-integration, only what you onboard
```

The diagram shows the native macOS topology. The Docker stack is equivalent, with Caddy and
the services running as containers behind `http://localhost:8080` (see the Linux quickstart).

## Quickstart

Two supported platforms. **macOS (Apple Silicon)** runs natively; **Linux** runs in Docker,
on CPU or an NVIDIA GPU. Both open the same web UI and share the same configuration.

### macOS (Apple Silicon)

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
down, and self-heals when your DHCP address changes.

To reach the assistant from another LAN device, open `https://<LAN-IP>:8443`. Trust the
certificate at `/ca.crt` first. `run_all.sh` prints the address.

If a component fails, run the health check:

```bash
bash scripts/doctor.sh    # checks each component and prints a fix hint
```

### Linux (Docker — CPU or NVIDIA CUDA)

The native `setup.sh` path is Apple-only because its STT runs on MLX. On Linux, a Docker
stack swaps that one piece — speech-to-text — for
[faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) and runs everything
else unchanged. **You need:** Docker + the Compose plugin, and an OpenAI-compatible LLM
endpoint (the LLM is not bundled — see *Bring your own LLM* below).

**CPU — runs on any Linux box:**

```bash
git clone <this-repo> voice-agent && cd voice-agent
cp docker/.env.example .env      # then set OPENAI_BASE_URL to your LLM
docker compose up --build        # open http://localhost:8080
```

**NVIDIA GPU (CUDA — ~1s latency):** needs an NVIDIA GPU and the
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

Open **http://localhost:8080**, pick a mic, tap Connect, talk. (Mic access works on
`localhost`; for LAN/remote the browser requires HTTPS — see the Docker guide.) The CPU image
defaults to the `base.en` STT model; the CUDA image defaults to `large-v3` at float16. Full
guide (HTTPS/LAN, model overrides, integrations, troubleshooting):
**[docker/README.md](docker/README.md)**.

> Docker on a **Mac** can't accelerate the models (no Metal passthrough into containers) — on
> a Mac, use the native `setup.sh` path above.

### Bring your own LLM

The agent works with any OpenAI-compatible endpoint that supports tool calling — Ollama,
[vLLM](https://github.com/vllm-project/vllm), LM Studio, `mlx_lm server`, or a remote API.
Point config at it: `agent/.env.local` for the native path, or `.env` for Docker.

```
OPENAI_BASE_URL=http://127.0.0.1:11434/v1   # your endpoint (Docker: use host.docker.internal)
OPENAI_API_KEY=anything-nonempty
LLM_MODEL=your-model-id                      # a tool-calling model your endpoint serves
```

<details>
<summary><b>Serving a raw MLX model (macOS)</b></summary>

On Apple Silicon you can serve a raw MLX model — for example your own finetune — with no
extra software; the venv already includes `mlx_lm`:

```bash
bash scripts/serve_mlx.sh mlx-community/Qwen3-8B-4bit   # or a local model path
```

The reference deployment uses this pattern. It runs
[**0G-AI/0GM-1.0-35B-A3B-0427**](https://huggingface.co/0G-AI/0GM-1.0-35B-A3B-0427), an
apache-2.0 MoE finetune of Qwen3.6-35B-A3B (~3B active params) with strong tool calling.
The agent prompts and latency tuning target that model. The 4-bit conversion is about 18 GB
on disk. Convert once, then serve:

```bash
.venv/bin/python -m mlx_lm convert --hf-path 0G-AI/0GM-1.0-35B-A3B-0427 -q --mlx-path ~/models/0gm-4bit
bash scripts/serve_mlx.sh ~/models/0gm-4bit
```

To override the model for voice only, set `VOICE_LLM_MODEL` and `VOICE_LLM_URL`. If you expose
the server past localhost, put a bearer-auth proxy in front of it.
</details>

## Configuration

Settings live in **`agent/.env.local`** for the native macOS path (copy it from
[`agent/.env.local.example`](agent/.env.local.example)), or in **`.env`** for the Docker path
(copy it from [`docker/.env.example`](docker/.env.example)). Both document every key. Main keys:

| Key | Purpose |
|---|---|
| `OPENAI_BASE_URL` / `LLM_MODEL` | your LLM endpoint and model |
| `STT_ENGINE` | macOS: `parakeet` (default) / `whisper` / `qwen3-asr`; Linux: `faster-whisper` |
| `FASTER_WHISPER_MODEL` / `_DEVICE` / `_COMPUTE` | Linux STT model + CPU/CUDA selection |
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
discovery, OAuth flows) and how to disable one. On Docker, add the same vars to `.env`. Restart
the agent worker afterward:

```bash
pkill -f "agent.py dev"; bash run_all.sh    # native
docker compose restart agent                # Docker
```

## Remote access (optional)

The default setup serves the host machine and its LAN. To allow access from anywhere (this
section describes the native path; for Docker, front the `web` service with TLS as covered in
[docker/README.md](docker/README.md)):

1. **ngrok.** Run `brew install ngrok`. Add your authtoken. Set `NGROK_DOMAIN` in
   `agent/.env.local` (blank gives a random URL). `run_all.sh` starts the tunnel.
2. **Password gate.** Basic auth protects the public page. Set `WEB_AUTH_SECRET` and
   `WEB_AUTH_PWHASH` (see the `Caddyfile.example` comments).
3. **TURN relay.** WebRTC media from cellular networks needs a public relay. Port-forward
   coturn (UDP/TCP 3478 and UDP 49160-49180). For more reliability behind home NAT, run
   coturn on a cheap public-IP VPS and set the `TURN_*` values
   (see `coturn/turnserver.conf.example`).

## Manage the stack

Native (macOS):

```bash
tail -f agent.log                    # the agent worker (includes latency lines)
tail -f latency.log                  # per-turn pipeline timings
pkill -f "agent.py dev"; bash run_all.sh    # restart the agent only
pkill -f livekit-server              # stop the media server
```

Docker (Linux):

```bash
docker compose logs -f agent         # the agent worker
docker compose restart agent         # restart the agent only
docker compose down                  # stop the whole stack
```

## Advanced and experimental

These features are developed on the native macOS path; some (Gemma Ears) depend on MLX and are
Apple-Silicon-only.

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

On the native path, run `bash scripts/doctor.sh` first — it checks every component and prints a
fix for each. On Docker, check `docker compose logs -f agent`. Common cases:

| Symptom | Fix |
|---|---|
| "no LLM" warning | native: `bash setup.sh` (installs Ollama); Docker: set `OPENAI_BASE_URL` in `.env` (use `host.docker.internal`, not `127.0.0.1`) |
| chat says "connection refused" (Docker) | the host LLM only listens on loopback — for ollama set `OLLAMA_HOST=0.0.0.0` (see docker/README.md) |
| microphone connects, no replies | check the agent log — usually the LLM endpoint or model id |
| LAN page does not load | native: trust the cert at `https://<LAN-IP>:8443/ca.crt`; Docker: mic needs HTTPS off `localhost` (see docker/README.md) |
| remote voice connects but stays silent | TURN is not reachable — see Remote access |
| no audio / ICE fails (Docker) | keep the `50000-50019/udp` media range published and open in the firewall |
| first reply is slow after boot | models warm on the first call; the native `setup.sh` prefetch avoids most delay |
| assistant echoes itself | use headphones, or rely on the built-in echo gate (on by default) |

## License

MIT — see [LICENSE](LICENSE).
