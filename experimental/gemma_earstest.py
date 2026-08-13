"""Three go/no-go tests for 'Gemma ears' (audio-native voice pipeline):
1. NAME BIASING: do made-up/unusual household names transcribe right with a
   names-list in the system prompt?
2. TOOLS FROM AUDIO: does a spoken device command produce a parseable tool call?
3. REAL-CONTEXT PREFILL + KV CACHE: with the real 5.4KB Kronik prompt, what's cold vs
   PromptCacheState-cached TTFT?
"""
import json
import time

MODEL = "mlx-community/gemma-4-12b-it-qat-4bit"
CLIPS = {"kronik1": "/tmp/gemma_clips/kronik1.wav", "pepe": "/tmp/gemma_clips/pepe.wav",
         "nyx": "/tmp/gemma_clips/nyx.wav"}

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from mlx_vlm.generate.dispatch import PromptCacheState

model, processor = load(MODEL)
config = load_config(MODEL)
print("model loaded")

NAMES = ("Names you may hear (spell them EXACTLY like this): Kronik (the assistant). "
         "Add your own household names here — pets, cars, people.")

def run(sys_prompt, clip, tools=None, cache=None, max_tokens=90):
    kw = {}
    if tools is not None:
        kw["tools"] = tools
    prompt = apply_chat_template(processor, config, sys_prompt, num_audios=1, **kw)
    t0 = time.time()
    first = None
    text = ""
    gkw = {"max_tokens": max_tokens, "temperature": 0.3}
    if cache is not None:
        gkw["prompt_cache_state"] = cache
    for ch in stream_generate(model, processor, prompt, audio=[clip], **gkw):
        if first is None:
            first = (time.time() - t0) * 1000
        text += ch.text
    return first, (time.time() - t0) * 1000, text.strip()

# warm
run("Listen and transcribe as [heard: ...].", CLIPS["kronik1"], max_tokens=30)

print("\n=== TEST 1: name biasing ===")
SYS1 = (f"You are Kronik, a smart home voice assistant. {NAMES} First transcribe the user "
        "exactly as [heard: ...], then reply with one short sentence.")
for name in ["nyx", "pepe", "kronik1"]:
    ttft, total, text = run(SYS1, CLIPS[name])
    print(f"[{name}] ttft={ttft:.0f}ms  {text[:150]!r}")

print("\n=== TEST 2: tools from audio ===")
TOOLS = [{"type": "function", "function": {
    "name": "control_lights",
    "description": "Control the smart lights.",
    "parameters": {"type": "object", "properties": {
        "action": {"type": "string", "description": "on, off, brightness, or color"},
        "which": {"type": "string", "description": "which light group"}},
        "required": ["action"]}}},
    {"type": "function", "function": {
        "name": "tesla_climate", "description": "Start or stop the car's climate/preheat.",
        "parameters": {"type": "object", "properties": {
            "on": {"type": "boolean"}}, "required": ["on"]}}}]
SYS2 = (f"You are Kronik, a smart home voice assistant. {NAMES} When the spoken request "
        "asks for a device action, call the matching tool.")
for name in ["kronik1", "pepe"]:
    ttft, total, text = run(SYS2, CLIPS[name], tools=TOOLS)
    print(f"[{name}] ttft={ttft:.0f}ms  {text[:180]!r}")

print("\n=== TEST 3: real 5.4KB Kronik prompt, cold vs cached ===")
BIG = open("/tmp/kronik_prompt.txt").read() + "\n" + NAMES + \
      "\nFirst transcribe the user exactly as [heard: ...], then reply with one short sentence."
cache = PromptCacheState()
for i, name in enumerate(["kronik1", "nyx", "pepe"]):
    ttft, total, text = run(BIG, CLIPS[name], cache=cache)
    print(f"[turn{i+1}:{name}] ttft={ttft:.0f}ms total={total:.0f}ms  {text[:120]!r}")
