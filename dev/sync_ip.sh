#!/bin/bash
# NOTE: LIVEKIT_URL is pinned to ws://127.0.0.1:7880 (stable published port).
# The container IP is dynamic and must NOT be written into LIVEKIT_URL — doing so
# breaks the agent worker after any container restart. This script is now a no-op.
echo "sync_ip: LIVEKIT_URL stays ws://127.0.0.1:7880 (container IP is not used)"
