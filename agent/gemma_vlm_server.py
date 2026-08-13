#!/usr/bin/env python
"""On-demand Gemma-4-E4B vision server for the Ring describer.

Runs mlx_vlm.server (gemma-4-e4b) on :8083, but SELF-EXITS after GEMMA_IDLE_SECS of no
describe activity — the describer (ring_tools via vlm_ondemand) touches ~/voice-agent/.gemma_lastuse
on each use. Vision is rare, so this keeps ~4GB out of RAM except when actually describing.
Spawned by vlm_ondemand.ensure(); not launchd-managed, so a clean self-exit stays down until
the next describe re-spawns it.
"""
import os
import sys
import threading
import time

_LASTUSE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".gemma_lastuse")  # repo root, not ~/voice-agent
_IDLE = int(os.environ.get("GEMMA_IDLE_SECS", "300"))   # 5 min


def _reaper():
    started = time.time()
    try:
        with open(_LASTUSE, "w") as f:   # seed so a stale/missing file doesn't kill us at boot
            f.write(str(started))
    except Exception:
        pass
    while True:
        time.sleep(30)
        try:
            with open(_LASTUSE) as f:
                last = float(f.read().strip())
        except Exception:
            last = started
        if time.time() - max(last, started) > _IDLE:
            os._exit(0)   # abrupt is fine: only fires when idle, so never mid-request


threading.Thread(target=_reaper, daemon=True).start()

sys.argv = ["mlx_vlm.server", "--model", "mlx-community/gemma-4-e4b-it-4bit",
            "--host", "127.0.0.1", "--port", "8083"]
from mlx_vlm.server import main  # noqa: E402

if __name__ == "__main__":
    main()
