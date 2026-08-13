"""Gemma Ears server — audio-native voice understanding for the experimental UI mode.

Runs gemma-4-12b (unified multimodal) in-process via mlx_vlm; per utterance it HEARS the
user's audio directly (no STT hop), transcribes + replies in one generation, executes tool
calls (lights / tesla climate+status / thermostat) via the voice-agent venv, and keeps
chat history as TEXT so the KV prefix cache stays valid and hot (~150ms TTFT warm; the
audio-placeholder cache pitfall means history must never contain raw audio).

Run (mlx-vlm-svc venv):  python gemma_ears_server.py     -> http://127.0.0.1:8084
API: POST /utterance {"audio_b64": <float32 16k mono>} -> {"heard","reply","tool"}
     POST /reset -> fresh conversation
"""
import base64
import json
import re
import subprocess
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

MODEL = "mlx-community/gemma-4-12b-it-qat-4bit"
PORT = 8084

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_vlm.generate.dispatch import PromptCacheState
from mlx_vlm.tool_parsers.gemma4 import parse_tool_call, tool_call_start

NAMES = ("Names you may hear (always spell them EXACTLY like this): Kronik (you, the "
         "assistant). Add your own household names here — pets, cars, people — so unusual "
         "or made-up names transcribe correctly.")
SYSTEM = (
    "You are Kronik, a friendly local smart-home voice assistant. " + NAMES + " You control "
    "the smart lights, the Nest thermostat, and the Tesla. Listen to the user's spoken "
    "request. Reply FIRST with your full spoken answer — usually one or two short "
    "sentences (spoken aloud, numbers as words), longer when the user asks for content "
    "like a joke or a story: deliver the whole thing, never just announce it. THEN on "
    "the next line write '[heard: <exact transcription of ONLY "
    "the words in the current audio clip — never text from earlier turns>]'; if the clip "
    "has no intelligible speech, write '[heard: -]'. If the user asks for a device action "
    "or status, the reply must be "
    "one very short acknowledgment (like 'Let me check.') and after the [heard:] line you "
    "call the matching tool.")

TOOLS = [
    {"type": "function", "function": {
        "name": "control_lights", "description": "Control the smart lights.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string", "description": "on, off, brightness, or color"},
            "brightness": {"type": "integer", "description": "1-100"},
            "color": {"type": "string", "description": "color name or scene"},
            "which": {"type": "string", "description": "light/group name, omit for all"}},
            "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "thermostat_status", "description": "Get the thermostat state.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "tesla_status", "description": "Check the car: battery, range, locked.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "tesla_climate", "description": "Start/stop the car's climate preheat.",
        "parameters": {"type": "object", "properties": {"on": {"type": "boolean"}},
                       "required": ["on"]}}},
]

import os
_VA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VA_PY = f"{_VA}/.venv/bin/python"
_TOOL_CODE = {
    "control_lights": ("import sys, json; sys.path.insert(0, 'agent');"
                       "from dotenv import load_dotenv; load_dotenv('agent/.env.local');"
                       "import wiz_tools; a=json.loads(sys.argv[1]);"
                       "print(wiz_tools.control(a.get('action'), a.get('brightness'), a.get('color'), a.get('which')))"),
    "thermostat_status": ("import sys; sys.path.insert(0, 'agent');"
                          "from dotenv import load_dotenv; load_dotenv('agent/.env.local');"
                          "import nest_tools; print(nest_tools.status())"),
    "tesla_status": ("import sys, asyncio; sys.path.insert(0, 'agent');"
                     "from dotenv import load_dotenv; load_dotenv('agent/.env.local');"
                     "import tesla_tools; print(asyncio.run(tesla_tools.status()))"),
    "tesla_climate": ("import sys, json, asyncio; sys.path.insert(0, 'agent');"
                      "from dotenv import load_dotenv; load_dotenv('agent/.env.local');"
                      "import tesla_tools; a=json.loads(sys.argv[1]);"
                      "print(asyncio.run(tesla_tools.climate(a.get('on', True), a.get('temperature'))))"),
}


def run_tool(name, args):
    code = _TOOL_CODE.get(name)
    if not code:
        return f"Unknown tool {name}."
    try:
        out = subprocess.run([_VA_PY, "-c", code, json.dumps(args or {})], cwd=_VA,
                             capture_output=True, text=True, timeout=40)
        lines = (out.stdout or "").strip().splitlines()
        return lines[-1] if lines else (out.stderr or "no output").strip()[-200:]
    except Exception as e:
        return f"Tool failed: {e}"


print("loading gemma-4-12b (unified)...", flush=True)
t0 = time.time()
model, processor = load(MODEL)
config = load_config(MODEL)
print(f"loaded in {time.time()-t0:.0f}s", flush=True)

state = {"history": [], "cache": PromptCacheState()}


def generate_stream(messages, audio=None, max_tokens=220, tools=None):
    kw = {"tools": tools} if tools else {}
    prompt = apply_chat_template(processor, config, messages,
                                 num_audios=1 if audio else 0, **kw)
    gkw = {"max_tokens": max_tokens, "temperature": 0.3,
           "prompt_cache_state": state["cache"]}
    for ch in stream_generate(model, processor, prompt,
                              audio=[audio] if audio else None, **gkw):
        yield ch.text


def generate(messages, audio=None, max_tokens=140, tools=None):
    return "".join(generate_stream(messages, audio, max_tokens, tools)).strip()


def stream_utterance(pcm16k, write):
    """Stream the spoken reply text via write(chunk); returns meta dict at the end.
    Reply-first format means tokens stream to TTS immediately; the [heard:] transcript
    arrives last and is parsed for history, never spoken."""
    path = "/tmp/gemma_ears_utt.wav"
    with wave.open(path, "w") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(pcm16k, -1, 1) * 32767).astype(np.int16).tobytes())

    messages = ([{"role": "system", "content": SYSTEM}] + state["history"] +
                [{"role": "user", "content": "(spoken audio)"}])
    t0 = time.time()

    def speakable(b):
        """The spoken projection of raw model output: completed [heard:]/thought blocks
        vanish (text AFTER them keeps flowing — the model sometimes drops [heard:] in
        the middle of a reply), tool markup and unclosed markers hold back the tail."""
        i = b.find(tool_call_start)
        if i != -1:
            b = b[:i]
        b = re.sub(r"<\|channel>.*?<channel\|>", "", b, flags=re.S)
        b = re.sub(r"\[heard:[^\]]*\]", " ", b)
        for mk in ("[", "<"):
            j = b.find(mk)
            if j != -1:
                b = b[:j]
        return b

    def gen_pass():
        buf = ""
        sent = ""          # what we streamed to the client (spoken part)
        tool_seen = False
        for ch in generate_stream(messages, audio=path, tools=TOOLS):
            buf += ch
            if not tool_seen and tool_call_start in buf:
                tool_seen = True
                if not sent.strip():
                    write("Let me check. ")   # spoken lead hides tool exec + follow-up gen
            safe = speakable(buf)
            if safe.startswith(sent) and len(safe) > len(sent):
                write(safe[len(sent):])
                sent = safe
            if tool_seen and "<tool_call|>" in buf[buf.find(tool_call_start):]:
                break                             # tool call complete; skip trailing tokens
        return buf, sent, tool_seen

    def parse_heard(buf):
        m = re.search(r"\[heard:\s*(.*?)\]", buf)
        if not m:
            return ""
        # scrub placeholder echoes the model sometimes copies from history
        h = re.sub(r"\((?:spoken audio|unintelligible)\)", "", m.group(1)).strip()
        return "" if h == "-" else h

    buf, sent, tool_seen = gen_pass()
    print(f"RAW {buf[:400]!r}", flush=True)
    heard = parse_heard(buf)
    if not (sent.strip() or heard or tool_seen):
        # markup-only output (e.g. thought-channel rambling) — a poisoned or stale
        # cache is the usual culprit; reset and retry once before giving up
        print(f"EMPTY GEN ({buf[:80]!r}) — cache reset + retry", flush=True)
        state["cache"] = PromptCacheState()
        buf, sent, tool_seen = gen_pass()
        heard = parse_heard(buf)
        if not (sent.strip() or heard or tool_seen):
            write("Sorry, I lost you for a second — could you say that again? ")
    tool_used = None

    if tool_seen:
        parsed = parse_tool_call(buf, TOOLS)
        calls = parsed if isinstance(parsed, list) else [parsed]
        results = []
        for c in calls[:2]:
            fn = c.get("function", c) if isinstance(c, dict) else {}
            name = fn.get("name"); args = fn.get("arguments") or {}
            if isinstance(args, str):
                try: args = json.loads(args)
                except Exception: args = {}
            tool_used = name
            print(f"TOOL {name} {args}", flush=True)
            results.append(run_tool(name, args))
        messages2 = messages + [
            {"role": "assistant", "content": buf.replace("\n", " ")[:300]},
            {"role": "user", "content": f"[tool result] {' | '.join(results)} — tell me in "
                                        "one short spoken sentence, then '[heard: -]' on the last line."}]
        buf2 = ""
        sent2 = ""
        for ch in generate_stream(messages2, max_tokens=70):
            buf2 += ch
            safe = speakable(buf2)
            if safe.startswith(sent2) and len(safe) > len(sent2):
                write(safe[len(sent2):])
                sent2 = safe
        spoken = (sent.strip() + " " + sent2.strip()).strip()
    else:
        spoken = sent.strip()

    dt = time.time() - t0
    if heard or tool_used:
        # unintelligible turns stay OUT of history: a run of "sorry, didn't catch that"
        # pairs pattern-locks the model into apologizing at clearly-heard speech
        state["history"].append({"role": "user", "content": heard or "(unintelligible)"})
        state["history"].append({"role": "assistant", "content": (spoken or buf)[:300]})
        state["history"][:] = state["history"][-12:]
    else:
        # skipped history makes the next prompt byte-identical to this one, and the
        # prefix cache would then reuse THIS turn's audio KV (placeholder token IDs
        # all match) — the stale-audio pitfall. Drop the cache instead.
        state["cache"] = PromptCacheState()
    print(f"UTT {dt*1000:.0f}ms heard={heard!r} tool={tool_used} reply={spoken[:80]!r}", flush=True)
    return {"heard": heard, "tool": tool_used, "ms": int(dt * 1000)}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"status": "ok", "turns": len(state["history"]) // 2})
        else:
            self._send({"error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        data = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/reset":
            state["history"] = []
            # keep state["cache"]: the system+tools prefix stays valid (prefix-matched),
            # so a fresh session skips the ~1.2s re-prefill
            self._send({"ok": True})
        elif self.path == "/utterance":
            pcm = np.frombuffer(base64.b64decode(data["audio_b64"]), dtype=np.float32)
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                def write(chunk):
                    b = chunk.encode()
                    self.wfile.write(f"{len(b):x}\r\n".encode() + b + b"\r\n")
                    self.wfile.flush()
                meta = stream_utterance(pcm, write)
                tailb = ("\x1e" + json.dumps(meta)).encode()
                self.wfile.write(f"{len(tailb):x}\r\n".encode() + tailb + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
            except Exception as e:
                import traceback; traceback.print_exc()
        else:
            self._send({"error": "not found"}, 404)


# warm: prefill the system+tools prefix TEXT-ONLY. A silent-audio warmup leaves
# silent KV under the audio placeholder tokens, and the first real turn prefix-matches
# straight through them — masking the first ~0.5s of the user's actual speech.
generate([{"role": "system", "content": SYSTEM},
          {"role": "user", "content": "warmup"}], tools=TOOLS, max_tokens=8)
print("system prefix warmed; serving", flush=True)

ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
