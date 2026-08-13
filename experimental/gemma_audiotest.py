"""Direct audio->text benchmark for gemma-4-12b (unified multimodal, via mlx_vlm python API).
Feeds real spoken-request wavs; measures wall-clock to FIRST TOKEN (encoder + prefill) and
full reply. Also tests a transcribe+answer combined prompt (solves the chat-history problem)
and a tool-style request to gauge instruction fidelity from raw audio."""
import sys
import time

MODEL = "mlx-community/gemma-4-12b-it-qat-4bit"
HW = "/tmp/gemma_clips"
CLIPS = {
    "kronik1": f"{HW}/kronik1.wav",     # "Hey Kronik, turn on the bedroom lights."
    "car": f"{HW}/car.wav",             # "Ask the car to warm up before I leave."
    "cat": f"{HW}/cat.wav",             # "Did you feed the cat this morning?"
}

from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

t0 = time.time()
model, processor = load(MODEL)
config = load_config(MODEL)
print(f"loaded in {time.time()-t0:.0f}s")

SYS_ANSWER = ("You are Kronik, a smart home voice assistant. Listen to the user's spoken "
              "request and reply with ONE short spoken-style sentence.")
SYS_BOTH = ("You are Kronik, a smart home voice assistant. First transcribe the user's spoken "
            "request exactly as [heard: ...], then reply with one short sentence.")

def run(tag, sys_prompt, clip):
    prompt = apply_chat_template(processor, config, f"{sys_prompt}", num_audios=1)
    t0 = time.time()
    first = None
    text = ""
    for chunk in stream_generate(model, processor, prompt, audio=[clip],
                                 max_tokens=80, temperature=0.4):
        if first is None:
            first = time.time() - t0
        text += chunk.text
    total = time.time() - t0
    print(f"[{tag}] ttft={first*1000:.0f}ms total={total*1000:.0f}ms")
    print(f"[{tag}] reply: {text.strip()[:180]!r}")

# warm the graph once
run("warmup", SYS_ANSWER, CLIPS["kronik1"])
print("--- measured runs ---")
for name, clip in CLIPS.items():
    run(f"answer:{name}", SYS_ANSWER, clip)
run("both:kronik1", SYS_BOTH, CLIPS["kronik1"])
run("both:nyx", SYS_BOTH, CLIPS["nyx"])
