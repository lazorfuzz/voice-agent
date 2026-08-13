"""Voice-agent LLM quality + perf bench — faithful rebuild of the July tool-vs-narrate
methodology (memory: llm-0gm-finetune). Runs against any OpenAI-compatible endpoint.

Usage: llm_bench.py <base_url> <model_id> [--perf-only|--quality-only]
Prints a JSON summary; detailed failures to stderr.
"""
import json
import re
import statistics as st
import sys
import time
import urllib.request

import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent"))
import personas  # noqa: E402

BASE = sys.argv[1].rstrip("/")
MODEL = sys.argv[2]
MODE = sys.argv[3] if len(sys.argv) > 3 else "--all"

ENABLED = {"wiz", "nest", "tv", "petlibro", "tesla"}
SYSTEM = personas.build_voice_instructions(personas.PERSONAS["kronik"], ENABLED)

TOOLS = [
    {"type": "function", "function": {"name": "hold_for_more_input",
        "description": "The user's utterance seems incomplete (trailing off, mid-sentence, filler). Wait for them to finish instead of replying.",
        "parameters": {"type": "object", "properties": {"seconds": {"type": "integer"}}, "required": ["seconds"]}}},
    {"type": "function", "function": {"name": "control_lights",
        "description": "Control the smart lights: on/off/brightness/color. `which` targets a specific light or group; omit for all.",
        "parameters": {"type": "object", "properties": {
            "action": {"type": "string"}, "brightness": {"type": "integer"},
            "color": {"type": "string"}, "which": {"type": "string"}}, "required": ["action"]}}},
    {"type": "function", "function": {"name": "thermostat_status",
        "description": "Get the thermostat's current state: temperature, humidity, setpoints, mode.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "set_thermostat_temperature",
        "description": "Set the thermostat target temperature in Fahrenheit.",
        "parameters": {"type": "object", "properties": {"temperature": {"type": "integer"}}, "required": ["temperature"]}}},
    {"type": "function", "function": {"name": "play_on_tv",
        "description": "Play a show, movie, or video on the TV. service='youtube' for videos/music.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "service": {"type": "string"}}, "required": ["title"]}}},
    {"type": "function", "function": {"name": "feed_cat",
        "description": "Dispense food from the cat feeder.",
        "parameters": {"type": "object", "properties": {"portions": {"type": "integer"}}}}},
    {"type": "function", "function": {"name": "tesla_status",
        "description": "Check the car: battery, range, charging, locked.",
        "parameters": {"type": "object", "properties": {}}}},
]

SEED = [
    {"role": "assistant", "content": "Yo! Kronik here. What's good?"},
    {"role": "user", "content": "Hey, how's it going?"},
    {"role": "assistant", "content": "All good, just vibing. What's up?"},
]


def chat(messages, temp=0.4, max_tokens=200, stream=False):
    body = {"model": MODEL, "messages": [{"role": "system", "content": SYSTEM}] + messages,
            "tools": TOOLS, "temperature": temp, "max_tokens": max_tokens, "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(BASE + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    if not stream:
        r = json.load(urllib.request.urlopen(req, timeout=180))
        msg = r["choices"][0]["message"]
        return {"content": (msg.get("content") or "").strip(),
                "tools": [(t["function"]["name"], t["function"].get("arguments", ""))
                          for t in (msg.get("tool_calls") or [])]}
    t0 = time.time()
    ttft = None
    ntok = 0
    for line in urllib.request.urlopen(req, timeout=180):
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        ch = (d.get("choices") or [{}])[0].get("delta", {})
        if ch.get("content") or ch.get("tool_calls"):
            if ttft is None:
                ttft = time.time() - t0
            ntok += 1
    dt = time.time() - t0
    return {"ttft": ttft, "total": dt,
            "tps": ntok / (dt - ttft) if ttft and dt > ttft and ntok > 5 else None}


# ---------------- A. clean-context actions (the 20-action bench) ----------------
ACTIONS = [
    ("Turn on the bedside lamps.", "control_lights"),
    ("Turn off all the lights.", "control_lights"),
    ("Set the lamps to 40 percent.", "control_lights"),
    ("Make the lights warm white.", "control_lights"),
    ("Dim the living room light.", "control_lights"),
    ("Lights to fireplace mode please.", "control_lights"),
    ("Set the lamps to 10 percent.", "control_lights"),
    ("Turn the bedroom lights red.", "control_lights"),
    ("Kill the lights in the den.", "control_lights"),
    ("Set the thermostat to 72.", "set_thermostat_temperature"),
    ("Make it cooler in here, like 70.", "set_thermostat_temperature"),
    ("Set the AC to 68 degrees.", "set_thermostat_temperature"),
    ("What's the thermostat at right now?", "thermostat_status"),
    ("Is the heat on?", "thermostat_status"),
    ("Put on Squid Game on the TV.", "play_on_tv"),
    ("Play some lofi beats on YouTube.", "play_on_tv"),
    ("Feed the cat.", "feed_cat"),
    ("Give the cat two portions.", "feed_cat"),
    ("Is the car charged?", "tesla_status"),
    ("Check the car's battery.", "tesla_status"),
]

# repeat/it-didn't-work follow-ups (historically the hardest: model narrates instead)
REPEATS = [
    ([{"role": "user", "content": "Set the lamps to 20 percent."},
      {"role": "assistant", "tool_calls": [{"id": "c1", "type": "function", "function": {
          "name": "control_lights", "arguments": '{"action":"brightness","brightness":20}'}}]},
      {"role": "tool", "tool_call_id": "c1", "content": "Set all lamps to 20%."},
      {"role": "assistant", "content": "Done, lamps at twenty percent."},
      {"role": "user", "content": "Hmm, it didn't work. Try again."}], "control_lights"),
    ([{"role": "user", "content": "Turn off the lamps."},
      {"role": "assistant", "tool_calls": [{"id": "c2", "type": "function", "function": {
          "name": "control_lights", "arguments": '{"action":"off"}'}}]},
      {"role": "tool", "tool_call_id": "c2", "content": "All lamps off."},
      {"role": "assistant", "content": "Lights are off."},
      {"role": "user", "content": "Turn them off again please."}], "control_lights"),
]

# ---------------- B. narration-polluted context ----------------
POLLUTED_HISTORY = [
    {"role": "user", "content": "Set the lamps to 10 percent."},
    {"role": "assistant", "content": "Okay, I set the lamps to ten percent."},
    {"role": "user", "content": "Set them to 30 percent."},
    {"role": "assistant", "content": "Lamps are at thirty percent now."},
    {"role": "user", "content": "Make them blue."},
    {"role": "assistant", "content": "The lamps are blue now."},
]
POLLUTED_CMDS = ["Set the lamps to 50 percent.", "Turn the lamps off.",
                 "Make the lamps green.", "Set them to full brightness.",
                 "Turn the bedside lamps back on."]

# ---------------- C. hold tests ----------------
HOLDS = ["Hey um", "So can you", "I was thinking maybe", "Okay so the thing is",
         "Can you turn on the", "Wait actually", "Hmm let me think"]

# ---------------- D. instruction following ----------------
STYLE_QS = ["What's 17 plus 25?", "How many days are in a leap year?",
            "What time zone is New York in?", "Who wrote The Great Gatsby?"]
CALL_FIRST_QS = [("What's the thermostat at?", "thermostat_status"),
                 ("How's the car doing?", "tesla_status"),
                 ("Is the cat feeder working?", None),  # any tool ok; prose stall = fail
                 ("Check the thermostat for me.", "thermostat_status")]
_STALL = re.compile(r"let me check|one (sec|second|moment)|give me a (sec|second|moment)|hold on|checking", re.I)


def run_quality():
    out = {}
    fails = []

    ok = 0
    for cmd, want in ACTIONS:
        r = chat(SEED + [{"role": "user", "content": cmd}])
        hit = any(n == want for n, _ in r["tools"])
        ok += hit
        if not hit:
            fails.append(f"A: {cmd!r} -> tools={r['tools']} text={r['content'][:60]!r}")
    for hist, want in REPEATS:
        r = chat(SEED + hist)
        hit = any(n == want for n, _ in r["tools"])
        ok += hit
        if not hit:
            fails.append(f"A-repeat -> tools={r['tools']} text={r['content'][:60]!r}")
    out["actions"] = f"{ok}/{len(ACTIONS) + len(REPEATS)}"
    out["actions_n"] = ok

    ok = 0
    for cmd in POLLUTED_CMDS:
        r = chat(SEED + POLLUTED_HISTORY + [{"role": "user", "content": cmd}])
        hit = any(n == "control_lights" for n, _ in r["tools"])
        ok += hit
        if not hit:
            fails.append(f"B: {cmd!r} -> text={r['content'][:60]!r}")
    out["polluted"] = f"{ok}/{len(POLLUTED_CMDS)}"

    ok = 0
    for frag in HOLDS:
        r = chat(SEED + [{"role": "user", "content": frag}])
        hit = any(n == "hold_for_more_input" for n, _ in r["tools"])
        ok += hit
        if not hit:
            fails.append(f"C: {frag!r} -> tools={r['tools']} text={r['content'][:50]!r}")
    out["holds"] = f"{ok}/{len(HOLDS)}"

    ok = 0
    for q in STYLE_QS:
        r = chat(SEED + [{"role": "user", "content": q}])
        c = r["content"]
        bad = (re.search(r"[*#`]|(\n- )", c) or re.search(r"\d", c)
               or len(re.findall(r"[.!?]+", c)) > 3 or not c)
        ok += not bad
        if bad:
            fails.append(f"D-style: {q!r} -> {c[:80]!r}")
    out["style"] = f"{ok}/{len(STYLE_QS)}"

    ok = 0
    for q, want in CALL_FIRST_QS:
        r = chat(SEED + [{"role": "user", "content": q}])
        called = bool(r["tools"]) and (want is None or any(n == want for n, _ in r["tools"]))
        stalled = bool(_STALL.search(r["content"]))
        hit = called and not stalled
        ok += hit
        if not hit:
            fails.append(f"D-callfirst: {q!r} -> tools={r['tools']} text={r['content'][:60]!r}")
    out["call_first"] = f"{ok}/{len(CALL_FIRST_QS)}"

    for f in fails:
        print("FAIL", f, file=sys.stderr)
    return out


def run_perf():
    chat(SEED + [{"role": "user", "content": "warmup"}], max_tokens=10)  # warm + cache prefix
    ttfts, tps = [], []
    for i in range(8):
        r = chat(SEED + [{"role": "user", "content":
                          f"Tell me something interesting about the number {i + 40}."}],
                 max_tokens=120, stream=True)
        if r["ttft"]:
            ttfts.append(r["ttft"] * 1000)
        if r["tps"]:
            tps.append(r["tps"])
        print(f"  perf {i}: ttft={r['ttft']*1000:.0f}ms tps={r['tps'] and round(r['tps']) or '?'}",
              file=sys.stderr)
    return {"ttft_ms_median": int(st.median(ttfts)) if ttfts else None,
            "ttft_ms_p95": int(sorted(ttfts)[int(len(ttfts) * 0.95)]) if ttfts else None,
            "tok_s_median": int(st.median(tps)) if tps else None}


res = {"model": MODEL, "base": BASE}
if MODE in ("--all", "--quality-only"):
    res["quality"] = run_quality()
if MODE in ("--all", "--perf-only"):
    res["perf"] = run_perf()
print(json.dumps(res, indent=1))
