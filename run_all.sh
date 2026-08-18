#!/bin/bash
# Start the full local voice-agent stack (LAN/iPhone-accessible over HTTPS). Idempotent.
set -e
export PATH="/opt/homebrew/bin:$PATH"
P="$(cd "$(dirname "$0")" && pwd)"; PY="$P/.venv/bin/python"; D="$P/scripts/daemon.py"

# Remote-access config kept OUT of this committed script — set in agent/.env.local (or env vars).
#   NGROK_DOMAIN = your reserved ngrok domain (blank -> ngrok assigns a random URL)
#   TURN_VPS     = your public TURN relay host (shown in the status line)
_ev() { grep -E "^$1=" "$P/agent/.env.local" 2>/dev/null | head -1 | cut -d= -f2-; }
NGROK_DOMAIN="${NGROK_DOMAIN:-$(_ev NGROK_DOMAIN)}"
TURN_VPS="${TURN_VPS:-$(_ev TURN_VPS)}"

# Auto-detect the Mac's LAN IP (DHCP changes it). Only LiveKit's node_ip (WebRTC media
# candidates) and the TURN LAN host truly need it — Caddy binds :8443 on all interfaces,
# and the frontend derives the LiveKit URL from the browser's own host. On a CHANGE we
# rewrite those configs and force-restart the affected services so LAN voice self-heals.
LAN_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo 127.0.0.1)
LAST_IP=$(cat "$P/.lan_ip" 2>/dev/null || echo "")
if [ "$LAN_IP" != "$LAST_IP" ]; then
  echo "[ip] LAN IP ${LAST_IP:-unknown} -> $LAN_IP; updating configs + restarting media services"
  [ -f "$P/livekit/livekit.yaml" ] && sed -i '' "s/node_ip: .*/node_ip: $LAN_IP/" "$P/livekit/livekit.yaml"
  # Caddy site address (its cert SAN must include the current IP — browsers send no SNI
  # for a bare IP, so it must be named, not a :8443 wildcard).
  [ -f "$P/Caddyfile" ] && sed -i '' -E "s/^[0-9.]+:8443 \{/$LAN_IP:8443 {/" "$P/Caddyfile"
  for f in "$P/.env.local" "$P/agent/.env.local"; do
    [ -f "$f" ] && sed -i '' \
      -e "s#LIVEKIT_PUBLIC_URL=.*#LIVEKIT_PUBLIC_URL=wss://$LAN_IP:8443#" \
      -e "s/TURN_LAN_HOST=.*/TURN_LAN_HOST=$LAN_IP/" "$f"
  done
  container rm -f livekit >/dev/null 2>&1 || true   # recreate with the new node_ip below
  pkill -f "token_server.py" 2>/dev/null || true    # reload LIVEKIT_PUBLIC_URL/TURN env
  pkill -f "caddy run" 2>/dev/null || true          # reload the site cert for the new IP
  sleep 2                                            # let caddy fully exit before step 5 restarts it
  echo "$LAN_IP" > "$P/.lan_ip"
fi

# LiveKit binds EVERY address in bind_addresses (and advertises every rtc.ips.includes) or
# it refuses to start — so a tunnel/VPN IP that isn't up yet after a reboot (e.g. WireGuard
# for remote access) would wedge the WHOLE media server, even for localhost. Generate a
# runtime config that keeps only currently-present addresses (loopback always kept) and
# start LiveKit from THAT. If the effective set changes between runs (the tunnel came back),
# restart LiveKit so remote self-heals — same idea as the LAN-IP heal above.
LK_SRC="$P/livekit/livekit.yaml"; LK_RUN="$P/livekit/.livekit.runtime.yaml"; LK_CHANGED=0
if [ -f "$LK_SRC" ]; then
  "$PY" "$P/scripts/livekit_runtime_config.py" "$LK_SRC" "$LK_RUN.new" 2>/dev/null || cp "$LK_SRC" "$LK_RUN.new"
  cmp -s "$LK_RUN.new" "$LK_RUN" 2>/dev/null || LK_CHANGED=1
  mv "$LK_RUN.new" "$LK_RUN"
else
  LK_RUN="$LK_SRC"   # no local config (shouldn't happen after setup.sh) — fall back
fi

echo "[1/7] LiveKit server..."
if [ "$LK_CHANGED" = "1" ] && pgrep -f "livekit-server" >/dev/null; then
  echo "  bind addresses changed (interface up/down) — restarting LiveKit to apply"
  pkill -f "livekit-server" 2>/dev/null; container rm -f livekit >/dev/null 2>&1 || true; sleep 2
fi
if curl -s --max-time 2 -o /dev/null http://127.0.0.1:7880; then
  echo "  already up"
elif command -v livekit-server >/dev/null 2>&1; then
  # native binary (brew install livekit) — the simple path, no container runtime needed
  "$PY" "$D" "$P/livekit.log" "$(command -v livekit-server)" --config "$LK_RUN"
  sleep 2
  # Verify it actually bound — a bad config or an in-use port exits it silently (daemon.py
  # does not supervise). Surface it instead of limping on with a dead media server.
  curl -s --max-time 2 -o /dev/null http://127.0.0.1:7880 \
    || echo "  ERROR: LiveKit did not come up — see livekit.log (tail -5 livekit.log)" >&2
elif command -v container >/dev/null 2>&1; then
  container system start --enable-kernel-install >/dev/null 2>&1 || true
  if ! container ls 2>/dev/null | grep -q '\blivekit\b'; then
    container rm -f livekit >/dev/null 2>&1 || true
    container run -d --name livekit \
      --mount type=bind,source="$P/livekit",target=/config \
      -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
      livekit/livekit-server:latest --config "/config/$(basename "$LK_RUN")" >/dev/null
    sleep 4
  fi
else
  echo "  ERROR: LiveKit not running and no way to start it. Fix: brew install livekit" >&2
  exit 1
fi

echo "[2/7] coturn TURN server (remote WebRTC relay — optional)..."
if [ -n "$(_ev TURN_USERNAME)" ] && command -v container >/dev/null 2>&1; then
  if ! container ls 2>/dev/null | grep -q '\bcoturn\b'; then
    container rm -f coturn >/dev/null 2>&1 || true
    container run -d --name coturn --mount type=bind,source="$P/coturn",target=/conf \
      -p 3478:3478/udp -p 3478:3478/tcp -p 49160-49180:49160-49180/udp \
      coturn/coturn:latest -c /conf/turnserver.conf >/dev/null
    sleep 3
  fi
else
  echo "  skipped (only needed for voice from outside your LAN; see README - Remote access)"
fi

echo "[3/7] local LLM service..."
if [ -f "$HOME/.claude/skills/local_agents/scripts/ensure_up.sh" ]; then
  bash "$HOME/.claude/skills/local_agents/scripts/ensure_up.sh" || true
else
  LLM_URL=$(_ev OPENAI_BASE_URL); LLM_URL="${LLM_URL:-http://127.0.0.1:11434/v1}"
  # Ollama endpoints are managed here (started if down); anything else is bring-your-own
  # (mlx_lm.server, LM Studio, a remote API, ...) and only health-checked.
  case "$LLM_URL" in *11434*)
    if command -v ollama >/dev/null 2>&1 && ! curl -s -o /dev/null --max-time 2 "${LLM_URL%/}/models"; then
      "$PY" "$D" "$P/ollama.log" "$(command -v ollama)" serve
      sleep 2
    fi
  esac
  curl -s -o /dev/null --max-time 3 "${LLM_URL%/}/models" \
    && echo "  LLM endpoint reachable: $LLM_URL" \
    || echo "  WARNING: no LLM at $LLM_URL - run 'bash setup.sh' or set OPENAI_BASE_URL (see README)"
fi

# Idempotent: only (re)start a service if it isn't already running. This makes the
# whole script safe to run repeatedly as a launchd watchdog without disrupting live calls.
echo "[4/7] token server + agent worker + frontend..."
# SoulX-Duplug turn-taking server (used when TURN_DETECTOR=soulx in agent/.env.local).
# Trained full-duplex end-of-turn model; the agent's soulx_turn.py adapter connects to :8765.
SOULX="$HOME/soulx-poc"
[ -d "$SOULX" ] && { pgrep -f "uvicorn server:app" >/dev/null || ( cd "$SOULX" && "$PY" "$D" "$SOULX/server.log" "$SOULX/.venv/bin/uvicorn" server:app --host 127.0.0.1 --port 8765 ); }
pgrep -f "token_server.py"  >/dev/null || ( cd "$P/agent"    && "$PY" "$D" "$P/token_server.log"    "$PY" "$P/agent/token_server.py" )
# Worker health: a mere pgrep isn't enough — the worker process can stay ALIVE while its
# LiveKit connection is dead (e.g. LiveKit restarted underneath it), stuck retrying, so it
# accepts no calls. If the log shows a "failed to connect" newer than the last successful
# "registered worker", the link is wedged: kill it so the spawn below re-registers.
if pgrep -f "agent.py dev" >/dev/null && [ -f "$P/agent.log" ]; then
  last_reg=$(grep -n "registered worker" "$P/agent.log" | tail -1 | cut -d: -f1)
  last_fail=$(grep -n "failed to connect to livekit" "$P/agent.log" | tail -1 | cut -d: -f1)
  if [ -n "$last_fail" ] && { [ -z "$last_reg" ] || [ "$last_fail" -gt "$last_reg" ]; }; then
    echo "[4/7] worker alive but disconnected from LiveKit (retry loop) — restarting"
    pkill -f "agent.py dev"; sleep 2
  fi
fi
pgrep -f "agent.py dev"     >/dev/null || ( cd "$P/agent"    && "$PY" "$D" "$P/agent.log"           "$PY" "$P/agent/agent.py" dev )
pgrep -f "ring_live.py"     >/dev/null || ( cd "$P/agent"    && "$PY" "$D" "$P/ring_live.log"       "$PY" "$P/agent/ring_live.py" )
[ -d "$P/frontend/dist" ] || ( cd "$P/frontend" && npm install --silent && npm run build --silent )
pgrep -f "http.server 8000" >/dev/null || ( cd "$P/frontend/dist" && "$PY" "$D" "$P/frontend_server.log" "$PY" -m http.server 8000 )

echo "[5/7] Caddy reverse proxy (HTTPS :8443 for LAN devices)..."
if command -v caddy >/dev/null 2>&1 && [ -f "$P/Caddyfile" ]; then
  pgrep -f "caddy run" >/dev/null || "$PY" "$D" "$P/caddy.log" "$(command -v caddy)" run --config "$P/Caddyfile"
else
  echo "  skipped (needed for LAN/phone access; brew install caddy)"
fi

echo "[6/7] ngrok public tunnel (optional)..."
if command -v ngrok >/dev/null 2>&1 && [ -n "$NGROK_DOMAIN" ]; then
  NGROK_URL="--url https://$NGROK_DOMAIN"
  pgrep -f "ngrok http" >/dev/null || "$PY" "$D" "$P/ngrok.log" "$(command -v ngrok)" http 9000 $NGROK_URL --log stdout
  sleep 3
else
  echo "  skipped (set NGROK_DOMAIN in agent/.env.local for access from anywhere)"
fi

echo "[7/7] status:"
pgrep -f "caddy run" >/dev/null && curl -sk -o /dev/null -w "  LAN   https://$LAN_IP:8443/       -> HTTP %{http_code}\n" "https://$LAN_IP:8443/"
[ -n "$NGROK_DOMAIN" ] && curl -s -o /dev/null -w "  ngrok https://$NGROK_DOMAIN/  -> HTTP %{http_code}\n" -H "ngrok-skip-browser-warning: true" "https://$NGROK_DOMAIN/"
pgrep -f "agent.py dev" >/dev/null && echo "  agent worker : registered" || echo "  agent worker : DOWN (tail agent.log)"
echo
echo "ready:"
echo "  This Mac : http://127.0.0.1:8000"
pgrep -f "caddy run" >/dev/null && echo "  LAN      : https://$LAN_IP:8443   (trust cert: https://$LAN_IP:8443/ca.crt)"
[ -n "$NGROK_DOMAIN" ] && echo "  Anywhere : https://$NGROK_DOMAIN   (trusted cert; password-gated; remote audio relays via VPS TURN ${TURN_VPS:-<unset>})"
true
