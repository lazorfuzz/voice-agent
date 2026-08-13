#!/bin/bash
# Fan out 10 voice-agent implementation tasks to local_agents (local Qwen via opencode).
S="$HOME/.claude/skills/local_agents/scripts"
P="$HOME/voice-agent"; R="$P/results"
launch() { local id="$1" prompt="$2"; mkdir -p "$R/$id"; bash "$S/run_agent.sh" "$R/$id.out" "$prompt" "$R/$id"; }

read -r -d '' T_env <<'EOF'
Create a file named .env.local for a local LiveKit voice agent. Include exactly these variables with comments:
LIVEKIT_URL=ws://127.0.0.1:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
OPENAI_API_KEY=local-key-not-used
OPENAI_BASE_URL=http://127.0.0.1:8080/v1
LLM_MODEL=mlx-community/Qwen3.6-35B-A3B-4bit
Create only .env.local. Do not create other files.
EOF

read -r -d '' T_reqs <<'EOF'
Create a requirements.txt for a Python LiveKit voice agent with these exact packages (one per line), plus a one-line comment header:
livekit-agents
livekit-plugins-openai
livekit-plugins-silero
python-dotenv
livekit-api
mlx-whisper
kokoro
numpy
Create only requirements.txt.
EOF

read -r -d '' T_readme <<'EOF'
Write a README.md for a LOCAL voice-agent project on macOS (Apple Silicon). Architecture: a browser page (LiveKit JS) connects over WebRTC to a local LiveKit server (run via Apple's `container` CLI), a Python LiveKit agent handles the call using local STT (mlx-whisper), local TTS (kokoro), silero VAD, and a local OpenAI-compatible LLM at http://127.0.0.1:8080/v1. A small Python token server mints browser JWTs. Include sections: Overview, Prerequisites, Setup (python venv + pip install -r requirements.txt), Running (1. start livekit server, 2. start token server, 3. start agent with `python agent.py dev`, 4. open frontend), and Ports (7880 livekit, 8790 token server, 8080 llm). Create only README.md.
EOF

read -r -d '' T_lkyaml <<'EOF'
Create a LiveKit server config file livekit.yaml for LOCAL DEVELOPMENT (single node, no Redis). Requirements:
- port: 7880 (main ws/http)
- rtc: tcp_port 7881, port_range_start 50000, port_range_end 50100, use_external_ip false
- keys: a single api key named devkey with value secret
- logging: level info
Create only livekit.yaml.
EOF

read -r -d '' T_runlk <<'EOF'
Write a bash script run_livekit.sh that runs the LiveKit server using Apple's `container` CLI. NOTE: apple/container uses docker-like syntax: `container run [options] image [args]`, flags include `-d/--detach`, `--name`, `-v host:container` for volume mounts, `-p` for port publishing, `--rm`. Requirements:
- run image livekit/livekit-server:latest in the FOREGROUND
- name the container "livekit"
- mount ./livekit.yaml (absolute path via $(pwd)) to /livekit.yaml
- pass container args: --config /livekit.yaml
- publish TCP ports 7880 and 7881, and UDP port range 50000-50100
- remove the container on exit (--rm)
Add `set -e` and a comment. Create only run_livekit.sh.
EOF

read -r -d '' T_token <<'EOF'
Write token_server.py using ONLY the Python standard library http.server plus the `livekit.api` package.
The livekit.api API: `from livekit import api`, then
  token = api.AccessToken(api_key, api_secret).with_identity(identity).with_name(identity)\
      .with_grants(api.VideoGrants(room_join=True, room=room)).to_jwt()
Requirements:
- Serve GET /token?room=<room>&identity=<identity> -> JSON {"token": <jwt>, "url": <LIVEKIT_URL>}
- Read LIVEKIT_API_KEY, LIVEKIT_API_SECRET, LIVEKIT_URL from environment (defaults: devkey, secret, ws://127.0.0.1:7880)
- Send CORS header Access-Control-Allow-Origin: * on every response, and handle OPTIONS preflight
- Listen on 0.0.0.0:8790
- Load .env.local first if python-dotenv is available (best-effort)
Create only token_server.py.
EOF

read -r -d '' T_frontend <<'EOF'
Create a minimal single-page voice-agent web UI as TWO files: index.html and app.js.
Use the LiveKit JS client SDK from CDN: <script src="https://cdn.jsdelivr.net/npm/livekit-client/dist/livekit-client.umd.min.js"></script> (global is `LivekitClient`).
Behavior:
- A "Connect" button and a status line.
- On click: fetch('http://127.0.0.1:8790/token?room=voice&identity=user-'+Math.random().toString(36).slice(2)) -> {token,url}
- const room = new LivekitClient.Room(); await room.connect(url, token); await room.localParticipant.setMicrophoneEnabled(true);
- On room.on(LivekitClient.RoomEvent.TrackSubscribed, (track,pub,participant)=>{ if(track.kind==='audio'){ const el=track.attach(); document.body.appendChild(el);} })
- Update the status line for connecting/connected/error.
Create only index.html and app.js.
EOF

read -r -d '' T_agent <<'EOF'
Write a Python LiveKit voice agent file agent.py using the livekit-agents framework. Follow THIS current API pattern exactly:

from dotenv import load_dotenv
from livekit import agents
from livekit.agents import AgentServer, AgentSession, Agent
from livekit.plugins import openai, silero

load_dotenv(".env.local")

class Assistant(Agent):
    def __init__(self):
        super().__init__(instructions="You are a friendly local voice assistant. Keep replies short and conversational.")

server = AgentServer()

@server.rtc_session(agent_name="voice")
async def entry(ctx: agents.JobContext):
    session = AgentSession(
        stt=<LOCAL_STT>,
        llm=openai.LLM(model=os.environ["LLM_MODEL"], base_url=os.environ["OPENAI_BASE_URL"], api_key=os.environ["OPENAI_API_KEY"]),
        tts=<LOCAL_TTS>,
        vad=silero.VAD.load(),
    )
    await session.start(room=ctx.room, agent=Assistant())
    await session.generate_reply(instructions="Greet the user warmly and ask how you can help.")

if __name__ == "__main__":
    agents.cli.run_app(server)

Replace <LOCAL_STT> with LocalWhisperSTT() imported from local_stt, and <LOCAL_TTS> with KokoroTTS() imported from local_tts. Add `import os`. Keep everything else as shown. Create only agent.py.
EOF

read -r -d '' T_stt <<'EOF'
Write local_stt.py: a custom LiveKit Agents STT plugin. Requirements:
- from livekit.agents import stt, utils; from livekit import rtc; import numpy as np; import mlx_whisper
- class LocalWhisperSTT(stt.STT):
  - __init__ sets super().__init__(capabilities=stt.STTCapabilities(streaming=False, interim_results=False)); store model repo "mlx-community/whisper-large-v3-turbo"
  - implement `async def _recognize_impl(self, buffer, *, language=None, conn_options=None):`
    - combine frames: `frame = rtc.combine_audio_frames(buffer)` then get float32 mono 16k: data = np.frombuffer(frame.data, dtype=np.int16).astype(np.float32)/32768.0 (resample to 16000 if frame.sample_rate != 16000 — you may assume 16000)
    - result = mlx_whisper.transcribe(data, path_or_hf_repo=self._model)
    - return stt.SpeechEvent(type=stt.SpeechEventType.FINAL_TRANSCRIPT, alternatives=[stt.SpeechData(text=result["text"].strip(), language=language or "en")])
Create only local_stt.py.
EOF

read -r -d '' T_tts <<'EOF'
Write local_tts.py: a custom LiveKit Agents TTS plugin wrapping Kokoro. Requirements:
- from livekit.agents import tts, utils; from livekit import rtc; import numpy as np; from kokoro import KPipeline
- SAMPLE_RATE=24000, NUM_CHANNELS=1
- class KokoroTTS(tts.TTS):
  - __init__: super().__init__(capabilities=tts.TTSCapabilities(streaming=False), sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS); self._pipeline = KPipeline(lang_code="a"); self._voice="af_heart"
  - def synthesize(self, text, *, conn_options=None) -> "ChunkedStream": return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)
- class ChunkedStream(tts.ChunkedStream):
  - implement `async def _run(self, output_emitter):` that: output_emitter.initialize(request_id=utils.shortuuid(), sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS, mime_type="audio/pcm"); then for (_, _, audio) in self._tts._pipeline(self.input_text, voice=self._tts._voice): convert float32 numpy audio [-1,1] to int16 bytes (np.clip then *32767 astype int16 .tobytes()) and call output_emitter.push(pcm_bytes); finally output_emitter.flush()
Create only local_tts.py.
EOF

# launch only the task ids passed as args (default: all) — bash 3.2 safe
WANT="${*:-env reqs readme lkyaml runlk token frontend agent stt tts}"
want() { case " $WANT " in *" $1 "*) return 0;; *) return 1;; esac; }
want env      && launch env      "$T_env"
want reqs     && launch reqs     "$T_reqs"
want readme   && launch readme   "$T_readme"
want lkyaml   && launch lkyaml   "$T_lkyaml"
want runlk    && launch runlk    "$T_runlk"
want token    && launch token    "$T_token"
want frontend && launch frontend "$T_frontend"
want agent    && launch agent    "$T_agent"
want stt      && launch stt      "$T_stt"
want tts      && launch tts      "$T_tts"
echo "=== launched at $(date +%T): $WANT ==="
