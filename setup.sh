#!/bin/bash
# One-shot bootstrap for a fresh clone. Idempotent — safe to re-run.
# After this finishes:  bash run_all.sh   then open http://127.0.0.1:8000
set -e
P="$(cd "$(dirname "$0")" && pwd)"
cd "$P"

fail() { echo "ERROR: $1" >&2; exit 1; }
note() { echo "==> $1"; }

# ---------- 0. sanity ----------
[ "$(uname -s)" = "Darwin" ] || fail "This stack targets macOS (Apple Silicon) — the STT/TTS models run on MLX."
[ "$(uname -m)" = "arm64" ]  || fail "Apple Silicon required (MLX models)."
command -v brew >/dev/null   || fail "Homebrew is required: https://brew.sh"

# ---------- 1. system deps ----------
note "checking system deps (brew)..."
for pkg in caddy node; do
  if ! command -v "$pkg" >/dev/null 2>&1; then
    note "  installing $pkg..."
    brew install "$pkg"
  fi
done
command -v livekit-server >/dev/null 2>&1 || { note "  installing livekit..."; brew install livekit; }
command -v ngrok >/dev/null 2>&1 || echo "  (optional) ngrok not found — only needed for access from outside your LAN: brew install ngrok"

# uv makes the Python setup fast + reproducible
if ! command -v uv >/dev/null 2>&1; then
  note "installing uv (Python package manager)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ---------- 2. python env ----------
note "creating .venv (python 3.12) + installing requirements..."
[ -d .venv ] || uv venv --python 3.12
uv pip install --python .venv/bin/python -r requirements.txt

# ---------- 3. frontend ----------
note "building the web UI..."
( cd frontend && npm install --silent && npm run build --silent )

# ---------- 4. config from templates (never overwrites yours) ----------
note "writing config files from templates (existing files are kept)..."
[ -f agent/.env.local ]          || cp agent/.env.local.example agent/.env.local
[ -f livekit/livekit.yaml ]      || cp livekit/livekit.yaml.example livekit/livekit.yaml
[ -f Caddyfile ]                 || cp Caddyfile.example Caddyfile
[ -f coturn/turnserver.conf ]    || cp coturn/turnserver.conf.example coturn/turnserver.conf

# fill in the LAN IP where templates carry a placeholder
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)
sed -i '' -e "s/<your-LAN-IP>/$LAN_IP/g" livekit/livekit.yaml Caddyfile coturn/turnserver.conf 2>/dev/null || true
{ grep -q "LIVEKIT_PUBLIC_URL=wss://your-host" agent/.env.local \
  && sed -i '' "s#LIVEKIT_PUBLIC_URL=wss://your-host:8443#LIVEKIT_PUBLIC_URL=wss://$LAN_IP:8443#" agent/.env.local; } || true

# ---------- 5. the LLM (the brain) ----------
OLLAMA_MODEL="qwen3.5:9b-mlx"  # the MLX build — same engine family as the STT/TTS models.
                               # Verified vs the GGUF build: 0.1-0.2s chat TTFT (vs 0.6s),
                               # 0.7-1.0s tool turns (vs 1.2s), correct tool calls. ~8.9 GB.
                               # Needs LLM_REASONING_EFFORT=none (thinking disabled).
LLM_URL=$(grep -E "^OPENAI_BASE_URL=" agent/.env.local | head -1 | cut -d= -f2- | awk '{print $1}')
LLM_URL="${LLM_URL:-http://127.0.0.1:11434/v1}"
echo
if curl -s -o /dev/null --max-time 2 "${LLM_URL%/}/models"; then
  note "LLM endpoint reachable at $LLM_URL"
else
  echo "The agent needs an LLM. I can set one up automatically with Ollama"
  echo "(fully local, ~9 GB download for $OLLAMA_MODEL) — or you can point"
  echo "agent/.env.local at any OpenAI-compatible server with tool calling."
  reply="y"
  if [ -t 0 ]; then read -r -p "Install the local LLM now? [Y/n] " reply; reply=${reply:-y}; fi
  if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
    command -v ollama >/dev/null 2>&1 || { note "installing ollama..."; brew install ollama; }
    pgrep -x ollama >/dev/null 2>&1 || { nohup ollama serve >/dev/null 2>&1 & sleep 3; }
    note "downloading $OLLAMA_MODEL (one-time, ~9 GB)..."
    ollama pull "$OLLAMA_MODEL"
    # point the agent at it (only overwrite untouched template values)
    sed -i ''       -e "s#^OPENAI_BASE_URL=http://127.0.0.1:8080/v1.*#OPENAI_BASE_URL=http://127.0.0.1:11434/v1#"       -e "s#^OPENAI_API_KEY=sk-\.\.\..*#OPENAI_API_KEY=ollama#"       -e "s#^LLM_MODEL=your-model-id.*#LLM_MODEL=$OLLAMA_MODEL#" agent/.env.local
    grep -q "^LLM_REASONING_EFFORT=" agent/.env.local \
      || echo "LLM_REASONING_EFFORT=none" >> agent/.env.local
    note "LLM ready (Ollama, $OLLAMA_MODEL)"
  else
    echo "OK — set OPENAI_BASE_URL / OPENAI_API_KEY / LLM_MODEL in agent/.env.local before running."
  fi
fi

# ---------- 6. prefetch the speech models (~2 GB; avoids a slow first call) ----------
note "prefetching STT/TTS models (skips if cached)..."
.venv/bin/python - <<'PY' || echo "  (model prefetch failed — they will download on first run instead)"
from parakeet_mlx import from_pretrained
from pocket_tts import TTSModel
from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")
TTSModel.load_model()
print("  models cached.")
PY

echo
note "setup complete. Next:"
echo "  bash run_all.sh        # start everything"
echo "  open http://127.0.0.1:8000   (LAN devices: https://$LAN_IP:8443)"
echo
echo "Health check anytime:  bash scripts/doctor.sh"
echo "Optional integrations (lights, thermostat, car, cameras, ...): see ONBOARDING.md"
