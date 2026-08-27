# Running on Linux with Docker

The native `setup.sh` path is macOS-only because its STT models run on Apple's MLX. This
Docker stack is the **Linux** path: it swaps the one Apple-locked component — speech-to-text —
for [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CPU or NVIDIA CUDA) and runs
everything else unchanged (TTS, VAD, turn detector, the LiveKit server, and the web UI).

The **LLM is not included** — point it at any OpenAI-compatible endpoint with tool calling
(vLLM, ollama, a cloud API). That matches the native setup and keeps these images light.

> Note: this is for **Linux hosts**. Docker on a *Mac* can't accelerate the models (no Metal
> passthrough into containers) — on a Mac, use the native `setup.sh` instead.

## Quickstart (CPU — runs on any Linux box)

```bash
cp docker/.env.example .env         # then edit OPENAI_BASE_URL -> your LLM
docker compose up --build           # first build pulls torch + models; later starts are fast
```

Open **http://localhost:8080**, pick a mic, tap Connect, talk. (Mic access works on
`localhost` over plain HTTP; for LAN/remote you need HTTPS — see below.)

## GPU (NVIDIA CUDA — ~1s latency)

Needs an NVIDIA GPU + the [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up --build
```

The CUDA image defaults to the `large-v3` model at `float16`; the CPU image defaults to
`base.en` at `int8`. Override with `FASTER_WHISPER_MODEL` / `_DEVICE` / `_COMPUTE` in `.env`.

## Pointing at your LLM

Set `OPENAI_BASE_URL` in `.env`. From inside a container, your **host** machine is
`host.docker.internal` (the compose file maps it for Linux). So an LLM on the host at `:11434`:

```
OPENAI_BASE_URL=http://host.docker.internal:11434/v1
LLM_MODEL=qwen3:8b
```

**Binding gotcha:** `host.docker.internal` traffic arrives via the docker bridge, NOT
loopback — a host LLM listening only on `127.0.0.1` gives "connection refused". Ollama's
default binding is loopback, so set `OLLAMA_HOST=0.0.0.0` (systemd:
`systemctl edit ollama` → `Environment="OLLAMA_HOST=0.0.0.0"`), or bind it to the docker
bridge only: `OLLAMA_HOST=172.17.0.1`. If the host has a public IP, prefer the bridge
binding or a firewall rule so the LLM isn't exposed to the internet.

A cloud or another-host endpoint just uses its real URL.

## What runs where

| Service | Image | Port | Role |
|---|---|---|---|
| `web`     | Caddy + built SPA        | `8080` | UI + reverse-proxy for `/token`, `/chat`, `/rtc` |
| `livekit` | `livekit/livekit-server` | `7880`, `7881`, `50000-50019/udp` | WebRTC media |
| `token`   | `kronik-agent`           | (internal) | mints JWTs, serves text chat |
| `agent`   | `kronik-agent`           | (internal) | VAD + faster-whisper STT + pocket-tts + tools |

STT/TTS weights download on first use into the `models` volume, so they persist across restarts.

## HTTPS (needed for LAN / remote)

Browsers only grant mic access on `localhost` or over HTTPS. To reach the app from another
device, put TLS in front of the `web` service — either:

- **ngrok:** `ngrok http 8080`, then set `LIVEKIT_PUBLIC_URL` in `.env` to the `wss://…` URL it
  gives you, and restart; or
- **your own reverse proxy / a `tls internal` block** in `docker/Caddyfile`, with your host or
  IP named, then set `LIVEKIT_PUBLIC_URL` to that origin.

Remote voice through home NAT also needs a TURN relay — see the coturn notes in the top-level
README's *Remote access* section.

## Integrations

The device integrations are opt-in and configured exactly as in the native setup
(`agent/.env.local.example` + `onboard.py`), with their env vars added to `.env`. Caveat:
LAN-discovery integrations (WiZ lights, Roomba, Google TV) need the `agent` container to see
your LAN — run that service with `network_mode: host` (a compose override), since the default
bridge network can't reach LAN multicast/broadcast.

## Common issues

| Symptom | Fix |
|---|---|
| connects but no reply | the LLM isn't reachable from the container — check `OPENAI_BASE_URL` (use `host.docker.internal`, not `127.0.0.1`) |
| no audio / ICE fails | the UDP media range must be reachable: keep `50000-50019/udp` published and matching `docker/livekit.yaml`; for cloud hosts open that range in the firewall |
| mic blocked on LAN | serve over HTTPS (see above) — plain HTTP only grants mic on `localhost` |
| CUDA image won't start | verify `nvidia-smi` works in a test container (`docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu24.04 nvidia-smi`) |
