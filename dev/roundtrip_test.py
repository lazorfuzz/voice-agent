import numpy as np, soundfile as sf, sys
print("=== Kokoro TTS ===", flush=True)
from kokoro import KPipeline
p = KPipeline(lang_code='a')
chunks=[a for _,_,a in p("Hello, this is a local text to speech test.", voice='af_heart')]
audio=np.concatenate(chunks); sf.write('/tmp/kokoro_test.wav', audio, 24000)
print(f"kokoro OK: {len(audio)} samples @24k = {len(audio)/24000:.1f}s", flush=True)
print("=== mlx-whisper STT (transcribe kokoro output) ===", flush=True)
import mlx_whisper
r = mlx_whisper.transcribe('/tmp/kokoro_test.wav', path_or_hf_repo='mlx-community/whisper-large-v3-turbo')
print("TRANSCRIPT:", repr(r['text'].strip()), flush=True)
print("ROUNDTRIP_OK", flush=True)
