import os
import json
import hmac
import hashlib
import time
import ipaddress
from datetime import timedelta
from http.cookies import SimpleCookie
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

try:
    from dotenv import load_dotenv

    # absolute path (next to this file), not cwd-relative — see agent.py note
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env.local"))
except ImportError:
    pass

from livekit import api
import chat_backend
import personas

LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://127.0.0.1:7880")
# URL handed to browsers (via the HTTPS/Caddy proxy); falls back to LIVEKIT_URL
LIVEKIT_PUBLIC_URL = os.environ.get("LIVEKIT_PUBLIC_URL", LIVEKIT_URL)

# coturn TURN/STUN servers advertised to the browser for NAT traversal.
# Both public (remote, needs router port-forward) and LAN hosts are offered;
# ICE picks whichever is reachable.
TURN_USER = os.environ.get("TURN_USERNAME", "kronik")
TURN_PASS = os.environ.get("TURN_PASS", "")
TURN_PUB = os.environ.get("TURN_PUBLIC_HOST", "")   # your relay's public IP / host
TURN_LAN = os.environ.get("TURN_LAN_HOST", "")      # your relay's LAN IP / host

# The watch fetches /personas + /token over the internet (forwarded through the VPS), so
# those two endpoints are gated for NON-LOCAL callers by a shared secret baked into the
# firmware. Loopback + LAN (browser via Caddy, local agent) stay key-free. Without a key
# set, remote callers are denied outright — never an open token minter on the public IP.
WATCH_TOKEN_KEY = os.environ.get("WATCH_TOKEN_KEY", "")
# Long TTL: the watch is an always-on device; a short-lived token could expire before it
# next reconnects. (Expiry only matters at JOIN — an active session isn't dropped.)
TOKEN_TTL = timedelta(days=7)
# Callers on these networks are trusted without a key (loopback + the home LAN). Anything
# else — e.g. the watch arriving via the public VPS tunnel — must present ?key=WATCH_TOKEN_KEY.
# NOTE: we deliberately do NOT trust all of 10.0.0.0/8, because the WireGuard tunnel the watch
# uses for remote access lives at 10.99.0.x — swallowing it would let remote callers skip the
# key. The two common consumer ranges (192.168/16, 172.16/12) don't overlap the tunnel and are
# safe to trust. Override with TOKEN_LOCAL_CIDRS (comma-separated) if your LAN is a different
# 10.x subnet — set it to just your LAN's /24, not all of 10/8.
_DEFAULT_LOCAL_CIDRS = "127.0.0.0/8,192.168.0.0/16,172.16.0.0/12,10.0.0.0/24"
_LOCAL_NETS = [
    ipaddress.ip_network(c.strip())
    for c in os.environ.get("TOKEN_LOCAL_CIDRS", _DEFAULT_LOCAL_CIDRS).split(",")
    if c.strip()
]


def _ice_servers():
    stun, turn = [], []
    for h in (TURN_PUB, TURN_LAN):
        if not h:
            continue
        stun.append(f"stun:{h}:3478")
        turn += [f"turn:{h}:3478?transport=udp", f"turn:{h}:3478?transport=tcp"]
    servers = []
    if stun:
        servers.append({"urls": stun})
    if turn and TURN_PASS:
        servers.append({"urls": turn, "username": TURN_USER, "credential": TURN_PASS})
    return servers


# ---- cookie session auth (gates the PUBLIC ngrok URL via Caddy forward_auth) ----
# WEB_AUTH_PWHASH format: "pbkdf2$<iters>$<salt_hex>$<hash_hex>".  WEB_AUTH_SECRET signs
# the session cookie (HMAC).  Both live in agent/.env.local; the plaintext password is
# never stored. LAN/localhost are NOT gated (Caddy only forward_auths the :9000 block).
WEB_AUTH_SECRET = os.environ.get("WEB_AUTH_SECRET", "")
WEB_AUTH_PWHASH = os.environ.get("WEB_AUTH_PWHASH", "")
SESSION_TTL = 30 * 24 * 3600   # 30 days


def _verify_password(pw: str) -> bool:
    try:
        _scheme, iters, salt, want = WEB_AUTH_PWHASH.split("$")
        got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters)).hex()
        return hmac.compare_digest(got, want)
    except Exception:
        return False


def _make_session() -> str:
    exp = str(int(time.time()) + SESSION_TTL)
    sig = hmac.new(WEB_AUTH_SECRET.encode(), exp.encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def _session_valid(cookie_header: str) -> bool:
    if not WEB_AUTH_SECRET:
        return False
    try:
        tok = SimpleCookie(cookie_header or "")["kronik_session"].value
        exp, sig = tok.split(".")
        want = hmac.new(WEB_AUTH_SECRET.encode(), exp.encode(), hashlib.sha256).hexdigest()
        return hmac.compare_digest(sig, want) and int(exp) > time.time()
    except Exception:
        return False


class TokenHandler(BaseHTTPRequestHandler):
    def _send_response(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        # Explicit Content-Length: strict HTTP clients (the ESP32's esp_http_client) can't
        # frame a body delimited only by connection-close and error with INCOMPLETE_DATA.
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(payload)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ---- cookie auth endpoints ----
    def _auth_check(self):
        """Called by Caddy forward_auth for every gated request. 204 if the session
        cookie is valid; otherwise redirect browser *navigations* to the login page and
        return 401 to API/XHR/WebSocket requests (so the app sees an auth failure)."""
        if _session_valid(self.headers.get("Cookie", "")):
            self.send_response(204)
            self.end_headers()
            return
        is_nav = (self.headers.get("Sec-Fetch-Mode") == "navigate"
                  or "text/html" in self.headers.get("Accept", ""))
        if is_nav:
            self.send_response(302)
            self.send_header("Location", "/login")
        else:
            self.send_response(401)
        self.end_headers()

    def _auth_login(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode()
        except Exception:
            raw = ""
        if "application/json" in self.headers.get("Content-Type", ""):
            try:
                pw = json.loads(raw or "{}").get("password", "")
            except Exception:
                pw = ""
        else:
            pw = parse_qs(raw).get("password", [""])[0]
        if _verify_password(pw):
            self.send_response(303)
            self.send_header("Location", "/")
            self.send_header("Set-Cookie",
                             f"kronik_session={_make_session()}; Path=/; Max-Age={SESSION_TTL}; "
                             "HttpOnly; Secure; SameSite=Lax")
        else:
            self.send_response(303)
            self.send_header("Location", "/login?e=1")
        self.end_headers()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/auth/login":
            return self._auth_login()
        if path == "/auth/logout":
            self.send_response(303)
            self.send_header("Location", "/login")
            self.send_header("Set-Cookie",
                             "kronik_session=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax")
            self.end_headers()
            return
        if path == "/enroll":
            return self._handle_enroll()
        if path == "/personas":   # create or edit a user persona
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
            except Exception:
                return self._send_response(400, {"error": "bad json"})
            try:
                entry = personas.save_user_persona(
                    data.get("label", ""), data.get("personality", ""),
                    data.get("voice", ""), data.get("greeting", ""),
                    key=(data.get("key") or None))
                return self._send_response(200, {"persona": entry})
            except ValueError as e:
                return self._send_response(400, {"error": str(e)})
        # Text chat over plain HTTP/SSE (no WebRTC/TURN). Streams the reply back.
        if path != "/chat":
            self._send_response(404, {"error": "Not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
        except Exception:
            self._send_response(400, {"error": "bad json"})
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        try:
            for ev in chat_backend.run_chat(data.get("messages", []), data.get("persona")):
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass   # client navigated away mid-stream
        except Exception as e:
            try:
                self.wfile.write(
                    f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

    # ---- speaker enrollment for ambient mode ----
    def _handle_enroll(self):
        """Accept an audio recording + name, compute a voice embedding, store it. The
        browser sends webm/opus (or wav); ffmpeg decodes to 16k mono."""
        import subprocess
        import tempfile
        import speaker_id
        try:
            name = parse_qs(urlparse(self.path).query).get("name", ["me"])[0]
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length)
            if not raw:
                return self._send_response(400, {"error": "no audio"})
            with tempfile.NamedTemporaryFile(suffix=".bin") as fin, \
                    tempfile.NamedTemporaryFile(suffix=".wav") as fout:
                fin.write(raw); fin.flush()
                subprocess.run(
                    ["/opt/homebrew/bin/ffmpeg", "-y", "-i", fin.name,
                     "-ar", "16000", "-ac", "1", "-f", "wav", fout.name],
                    capture_output=True, timeout=30)
                import soundfile as sf
                import numpy as np
                wav, _ = sf.read(fout.name, dtype="float32")
                if wav.ndim > 1:
                    wav = wav.mean(axis=1)
            if len(wav) < speaker_id.MIN_SAMPLES_16K:
                return self._send_response(400, {"error": "recording too short (need ~2s)"})
            saved = speaker_id.enroll(name, np.asarray(wav, dtype=np.float32))
            self._send_response(200, {"enrolled": saved, "speakers": speaker_id.enrolled_names()})
        except Exception as e:
            self._send_response(500, {"error": f"{type(e).__name__}: {e}"})

    def do_DELETE(self):
        parsed = urlparse(self.path)
        if parsed.path == "/personas":
            key = parse_qs(parsed.query).get("key", [""])[0]
            try:
                ok = personas.delete_user_persona(key)
                return self._send_response(200 if ok else 404, {"deleted": ok})
            except ValueError as e:
                return self._send_response(400, {"error": str(e)})
        if parsed.path != "/enroll":
            return self._send_response(404, {"error": "Not found"})
        import speaker_id
        name = parse_qs(parsed.query).get("name", [""])[0]
        speaker_id.delete(name)
        self._send_response(200, {"speakers": speaker_id.enrolled_names()})

    def _remote_ok(self, params):
        """True if this request may mint/read tokens+personas. Loopback/LAN are always
        trusted; any other source (e.g. the watch arriving via the VPS tunnel, source
        10.99.0.x) must present ?key=WATCH_TOKEN_KEY."""
        try:
            ip = ipaddress.ip_address(self.client_address[0])
            if any(ip in net for net in _LOCAL_NETS):
                return True
        except ValueError:
            pass
        return bool(WATCH_TOKEN_KEY) and params.get("key", [""])[0] == WATCH_TOKEN_KEY

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/auth/check":
            return self._auth_check()

        if parsed.path == "/enrolled":
            import speaker_id
            return self._send_response(200, {"speakers": speaker_id.enrolled_names()})

        if parsed.path == "/personas":
            q = parse_qs(parsed.query)
            if not self._remote_ok(q):
                return self._send_response(401, {"error": "unauthorized"})
            full = personas.list_personas()
            # keys=1: minimal {key,label} list for constrained clients (the watch) — the full
            # objects (personality/greeting text) blow past the ESP32's small response buffer.
            if q.get("keys", ["0"])[0] in ("1", "true"):
                full = [{"key": p["key"], "label": p["label"]} for p in full]
            return self._send_response(200, {"personas": full})

        if parsed.path == "/voices":
            return self._send_response(200, {"voices": personas.CATALOG_VOICES})

        if parsed.path == "/voice_preview":
            voice = parse_qs(parsed.query).get("voice", [""])[0]
            if voice not in personas.CATALOG_VOICES:
                return self._send_response(400, {"error": "unknown voice"})
            try:
                import voice_preview   # lazy: loads pocket-tts on first preview only
                wav = voice_preview.synth_wav(voice)
            except Exception as e:
                return self._send_response(500, {"error": f"{type(e).__name__}: {e}"})
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.send_header("Cache-Control", "public, max-age=86400")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            try:
                self.wfile.write(wav)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        if parsed.path != "/token":
            self._send_response(404, {"error": "Not found"})
            return

        params = parse_qs(parsed.query)
        if not self._remote_ok(params):
            return self._send_response(401, {"error": "unauthorized"})
        room = params.get("room", [None])[0]
        identity = params.get("identity", [None])[0]
        # UI persona selection -> travels as a participant attribute the agent reads at
        # join time to pick the voice + instructions. Defaults handled agent-side.
        persona = params.get("persona", ["kronik"])[0]
        # ambient mode: agent always listens but only responds when addressed by name.
        ambient = params.get("ambient", ["off"])[0]

        if not room or not identity:
            self._send_response(400, {"error": "room and identity are required"})
            return

        token = (
            api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
            .with_identity(identity)
            .with_name(identity)
            .with_attributes({"persona": persona, "ambient": ambient})
            .with_grants(api.VideoGrants(room_join=True, room=room))
            .with_ttl(TOKEN_TTL)
            .to_jwt()
        )

        self._send_response(200, {
            "token": token,
            "url": LIVEKIT_PUBLIC_URL,
            "iceServers": _ice_servers(),
        })

    def log_message(self, format, *args):
        print(f"[token_server] {args[0]}")


if __name__ == "__main__":
    server = ThreadingHTTPServer(("0.0.0.0", 8790), TokenHandler)
    print(f"Token + chat server listening on 0.0.0.0:8790")
    print(f"  LIVEKIT_URL={LIVEKIT_URL}")
    try:
        import speaker_id
        names = speaker_id.enrolled_names()
        print(f"  enrolled voices (persisted): {names or 'none'}")
    except Exception:
        pass
    server.serve_forever()
