#!/bin/bash
# Health check for the whole stack, with a fix hint for anything red.
P="$(cd "$(dirname "$0")/.." && pwd)"
_ev() { grep -E "^$1=" "$P/agent/.env.local" 2>/dev/null | head -1 | cut -d= -f2- | awk '{print $1}'; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$1"; }
bad()  { printf "  \033[31m✗\033[0m %-42s fix: %s\n" "$1" "$2"; FAIL=1; }

echo "voice-agent doctor"
[ -x "$P/.venv/bin/python" ] \
  && ok "python venv" \
  || bad "python venv missing" "bash setup.sh"
[ -f "$P/agent/.env.local" ] \
  && ok "agent/.env.local" \
  || bad "agent/.env.local missing" "bash setup.sh"
[ -d "$P/frontend/dist" ] \
  && ok "web UI built" \
  || bad "web UI not built" "cd frontend && npm install && npm run build"

LLM_URL=$(_ev OPENAI_BASE_URL); LLM_URL="${LLM_URL:-http://127.0.0.1:11434/v1}"
if curl -s -o /dev/null --max-time 3 "${LLM_URL%/}/models"; then
  ok "LLM endpoint ($LLM_URL)"
else
  bad "LLM endpoint unreachable ($LLM_URL)" "bash setup.sh (installs Ollama) or fix OPENAI_BASE_URL"
fi

curl -s --max-time 2 -o /dev/null http://127.0.0.1:7880 \
  && ok "LiveKit server (:7880)" \
  || bad "LiveKit server down" "bash run_all.sh (needs: brew install livekit)"
curl -s --max-time 2 -o /dev/null http://127.0.0.1:8790 \
  && ok "token server (:8790)" \
  || bad "token server down" "bash run_all.sh; then tail token_server.log"
curl -s --max-time 2 -o /dev/null http://127.0.0.1:8000 \
  && ok "frontend (:8000)" \
  || bad "frontend down" "bash run_all.sh"
pgrep -f "agent.py dev" >/dev/null \
  && ok "agent worker" \
  || bad "agent worker not running" "bash run_all.sh; then tail agent.log"

# optional pieces — only report, never fail
pgrep -f "caddy run" >/dev/null && ok "caddy (LAN https) [optional]" || echo "  - caddy not running (LAN/phone access) — optional"
pgrep -f "ngrok http" >/dev/null && ok "ngrok tunnel [optional]"     || echo "  - ngrok not running (remote access) — optional"

echo
if [ -n "$FAIL" ]; then echo "problems found — apply the fixes above, then re-run."; exit 1
else echo "all good — open http://127.0.0.1:8000"; fi
