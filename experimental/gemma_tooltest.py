"""Go/no-go for gemma-4-12b as the voice LLM: does mlx_lm server emit proper OpenAI-style
tool_calls for it? Tests the agent's real schema shape + a device command + chitchat + TTFT."""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8082/v1/chat/completions"
MODEL = "mlx-community/gemma-4-12b-it-qat-4bit"

TOOLS = [{
    "type": "function",
    "function": {
        "name": "control_lights",
        "description": "Control the smart lights (all of them by default).",
        "parameters": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "on, off, brightness, or color"},
                "brightness": {"type": "integer", "description": "1-100 percent"},
                "color": {"type": "string", "description": "a color name"},
                "which": {"type": "string", "description": "a light or group name"},
            },
            "required": ["action"],
        },
    },
}, {
    "type": "function",
    "function": {
        "name": "thermostat_status",
        "description": "Get the thermostat's current state.",
        "parameters": {"type": "object", "properties": {}},
    },
}]

SYS = ("You are Kronik, a smart home voice assistant. Keep replies short — they are spoken "
       "aloud. When the user asks for a device action or status, call the tool.")


def ask(text, tools=True):
    body = {"model": MODEL, "max_tokens": 120, "temperature": 0.4,
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": text}]}
    if tools:
        body["tools"] = TOOLS
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    out = json.load(urllib.request.urlopen(req, timeout=60))
    dt = (time.time() - t0) * 1000
    ch = out["choices"][0]
    tc = ch["message"].get("tool_calls")
    return dt, ch.get("finish_reason"), tc, (ch["message"].get("content") or "")[:120]


for q, expect_tool in [
    ("Turn on the bedroom lights.", True),
    ("Set the lights to 50 percent.", True),
    ("What's the thermostat at?", True),
    ("How are you doing today?", False),
]:
    try:
        dt, fin, tc, content = ask(q)
        ok = (tc is not None) == expect_tool
        print(f"{'PASS' if ok else 'FAIL'} {dt:6.0f}ms fin={fin!s:12s} "
              f"tool={json.dumps(tc[0]['function']) if tc else None} text={content!r}"[:200])
    except Exception as e:
        print("ERROR", q, e)
