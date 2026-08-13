# local_agents stress test — findings (voice-agent build)

Experiment: plan a full local LiveKit voice-agent stack (myself), then dispatch 10 graded
implementation tasks to `local_agents` (headless opencode → local Qwen3.6-35B-A3B), and grade
the output against the real APIs (livekit-agents 1.6.4 installed for validation).

## Operational findings
- **GPU OOM is the real concurrency limit, not throughput.** 10 agents launched at once →
  `[METAL] Insufficient Memory` → mlx server **crashed**. Cause: every `opencode run` request carries a
  large system prompt + tool schemas (~10–20k tokens); parallel prefill of many big contexts exhausts GPU memory.
  The earlier "6–8 concurrent" number came from tiny-prompt benchmarks and does **not** apply to opencode.
  → Fix: **waves of 3–4**. Re-ran all 10 in 4 waves, zero crashes.
- **launchd KeepAlive works.** After the OOM crash the server auto-restarted; no manual intervention.
- **Throughput under real load is strong:** 3–4 heavy agents finish in 30–65s; the 10-way (pre-crash) attempt
  showed the model batching to the proxy's 32 in-flight cap.

## Code-quality findings (graded vs livekit-agents 1.6.4)
| File | Verdict |
|---|---|
| `.env.local`, `requirements.txt`, `livekit.yaml` | ✅ correct |
| `run_livekit.sh` | ✅ ~correct (docker-equiv; apple/container `-p`/single-file `-v` semantics differ — niche) |
| `token_server.py` | ✅ correct & idiomatic (followed provided `livekit.api` chain exactly) |
| `agent.py` | ✅ correct — imports & builds `AgentServer`; every API exists in 1.6.4 |
| `local_stt.py` | ✅ correct — constructs; `_recognize_impl` signature matches base |
| `local_tts.py` | ⚠️ 1 real bug — int16 scale/cast order → silent audio (fixed) |
| `index.html` | ✅ correct |
| `app.js` | ⚠️ 2 bugs — wrong UMD global `LiveKitClient`→`LivekitClient`; missing `setMicrophoneEnabled(true)` (fixed) |

**Every livekit API attribute the model used actually exists in 1.6.4** (AgentServer, rtc_session,
stt.STT/_recognize_impl, tts.ChunkedStream/_run, AudioEmitter.initialize kwargs, ChunkedStream.input_text).
No hallucinated APIs — because the prompts included accurate API snippets.

## The pattern that worked
Frontier (me): architecture + exact API specs. Local (35B): faithful drafting. Frontier again: review + fix.
Failure modes were NOT hallucinated APIs but: (1) subtle numeric bug, (2) silently dropping a listed requirement
(mic enable), (3) "correcting" an exact string I gave (`Livekit`→`LiveKit`). All caught in review.

## What is NOT done (needs sudo / heavy installs / live audio)
- `apple/container` not installed (needs `brew install container` or `.pkg` — sudo).
- `mlx-whisper`, `kokoro` not installed (heavy).
- Never run end-to-end with real audio; correctness verified by import/attribute/instantiation checks only.

---

# END-TO-END RESULT (it runs!)

Got the whole stack running locally and verified a full spoken round-trip.

**Verified working:**
- `apple/container` 1.1.0 running `livekit/livekit-server` — container gets a host-routable IP
  (192.168.64.x); pointing everything at that IP avoids all Docker-style port/NAT gymnastics for WebRTC.
- Token server mints valid JWTs; browser/agent both derive the LiveKit URL from `.env.local`.
- Agent worker auto-dispatches on participant join (removed `agent_name`).
- **Kokoro TTS**, **mlx-whisper STT**, **local Qwen LLM** all run locally.
- Headless-sim test: sim SPOKE "What is the capital of France?" -> agent replied (transcribed):
  "...The capital of France is Paris..." — full STT->LLM->TTS->audio loop, all local.
- Frontend served at http://127.0.0.1:8000 (real browser: Connect -> talk).

**Integration bugs found only at runtime (all fixed by me):**
1. mlx_lm rejects system-only LLM calls — the Qwen3.6 **chat template** raises
   `No user query found in messages` / `System message must be at the beginning`. The agent's
   `generate_reply()` greeting is system-only -> 404 -> silent audio. Fix: patched the template's
   user-query guard + switched the greeting to a fixed `session.say()` (no LLM).
2. **Kokoro yields torch Tensors, not numpy** -> `'Tensor' has no attribute astype`. Fix: `.detach().cpu().numpy()`.
3. STT assumed 16kHz; WebRTC audio is 48kHz -> would garble. Fix: added resample + mono downmix in `local_stt.py`.
4. LiveKit secret must be >=32 chars; `OPENAI_API_KEY` must be the real proxy key (else 401).

**Caveats:** container IP is dynamic (re-run `sync_ip.sh` after restart; a stable DNS name needs
`container system dns` + sudo). Qwen3.6 is a reasoning model -> a few seconds of "thinking" latency per
reply (could disable thinking for snappier voice). Whisper hallucinates on long trailing silence.

**Run it:** `bash ~/voice-agent/run_all.sh` then open http://127.0.0.1:8000
