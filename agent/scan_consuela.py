import numpy as np, soundfile as sf, json
from qwen3_asr_mlx import Qwen3ASR

a, sr = sf.read("voices/sources/consuela_familyguy.wav", dtype="float32")
asr = Qwen3ASR.from_pretrained("mlx-community/Qwen3-ASR-1.7B-bf16")

WIN, HOP = 4.0, 4.0
fw = int(0.03*sr)  # 30ms frames for envelope
rows = []
for i in range(0, len(a)-int(WIN*sr), int(HOP*sr)):
    seg = a[i:i+int(WIN*sr)]
    env = np.array([np.sqrt(np.mean(seg[j:j+fw]**2)) for j in range(0, len(seg)-fw, fw)])
    if env.size == 0: continue
    noise_floor = float(np.percentile(env, 10))   # high = music/noise fills the pauses
    speech_frac = float((env > max(0.01, np.percentile(env,90)*0.15)).mean())
    peak = float(np.abs(seg).max())
    # 16k for asr
    n16 = int(len(seg)*16000/sr)
    s16 = np.interp(np.linspace(0,len(seg),n16,endpoint=False), np.arange(len(seg)), seg).astype(np.float32)
    txt = asr.transcribe(s16, language="en").text.strip()
    rows.append({"t": round(i/sr,1), "nf": round(noise_floor,4), "sf": round(speech_frac,2),
                 "peak": round(peak,2), "txt": txt})
json.dump(rows, open("%s/scan.json"%__import__("os").path.dirname(__file__), "w"))

# clean = low noise floor + real speech present
clean = [r for r in rows if r["nf"] < 0.03 and r["sf"] > 0.25 and len(r["txt"]) > 3]
print(f"{len(rows)} windows, {len(clean)} clean-speech windows\n")
print("=== cleanest 12 (low noise floor) ===")
for r in sorted(clean, key=lambda r: r["nf"])[:12]:
    print(f"  t={r['t']:>5}s nf={r['nf']:.3f} sf={r['sf']:.2f} peak={r['peak']:.2f} | {r['txt'][:70]}")

# contiguous clean runs (>=2 adjacent clean 4s windows)
print("\n=== contiguous clean runs (>=8s) ===")
cleanset = {r["t"] for r in clean}
i = 0; rows_by_t = {r["t"]: r for r in rows}
ts = sorted(cleanset)
runs = []
j = 0
while j < len(ts):
    k = j
    while k+1 < len(ts) and round(ts[k+1]-ts[k],1) == HOP: k += 1
    if k > j:
        runs.append((ts[j], ts[k]+WIN))
    j = k+1
for st, en in sorted(runs, key=lambda x: x[1]-x[0], reverse=True):
    txt = " ".join(rows_by_t[t]["txt"] for t in ts if st <= t < en)
    print(f"  {st:.0f}-{en:.0f}s ({en-st:.0f}s): {txt[:120]}")
