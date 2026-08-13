"""On-demand launcher for the Gemma-4-E4B vision server (:8083).

ring_tools.describe_image calls ensure() before each description. First use spawns the
self-terminating gemma server (gemma_vlm_server.py) and waits for it to load (~5-15s);
when it's already warm this is instant. Each call touches a last-use file so the server
defers its idle self-exit. Keeps vision GPU/RAM cost off the voice stack except when needed.
"""
import asyncio
import os
import subprocess
import time
import urllib.request

_DIR = os.path.dirname(os.path.abspath(__file__))
# Python interpreter for the VLM server's venv (mlx-vlm deps). Override with VLM_PYTHON;
# defaults to a sibling mlx-vlm-svc checkout's venv next to this repo.
_VLM_PY = os.environ.get("VLM_PYTHON") or os.path.expanduser("~/mlx-vlm-svc/.venv/bin/python")
_PORT = int(os.environ.get("VLM_PORT", "8083"))
_URL = f"http://127.0.0.1:{_PORT}/v1/chat/completions"
_LASTUSE = os.path.join(os.path.dirname(_DIR), ".gemma_lastuse")  # repo root (agent/..), not ~/voice-agent
_lock = asyncio.Lock()


def _touch():
    try:
        with open(_LASTUSE, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


def _up() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{_PORT}/v1/models", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


async def ensure() -> str:
    """Ensure the on-demand Gemma vision server is up; return its chat-completions URL."""
    _touch()
    if _up():
        return _URL
    async with _lock:
        if _up():
            return _URL
        logf = open(os.path.join(os.path.dirname(_DIR), "gemma_vlm.log"), "ab", buffering=0)
        subprocess.Popen(
            [_VLM_PY, os.path.join(_DIR, "gemma_vlm_server.py")],
            stdout=logf, stderr=logf, start_new_session=True)
        for _ in range(120):
            await asyncio.sleep(1)
            if _up():
                _touch()
                return _URL
    return _URL  # best-effort; a failed POST just makes describe_image return None
