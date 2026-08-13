#!/bin/bash
# Serve any MLX model as the agent's LLM — the same pattern as a production
# mlx_lm deployment (vanilla `mlx_lm server`; the agent's OpenAI-compatible
# client + tool calling work as-is with Qwen-family models).
#
# Usage:
#   bash scripts/serve_mlx.sh                                # default public model
#   bash scripts/serve_mlx.sh mlx-community/Qwen3-8B-4bit    # any HF MLX model
#   bash scripts/serve_mlx.sh ~/models/my-finetune-4bit      # or a local path
#
# Then in agent/.env.local:
#   OPENAI_BASE_URL=http://127.0.0.1:8080/v1
#   OPENAI_API_KEY=anything-nonempty
#   LLM_MODEL=<the model id/path you served>
#   (and remove LLM_REASONING_EFFORT unless your server accepts it)
set -e
P="$(cd "$(dirname "$0")/.." && pwd)"
MODEL="${1:-mlx-community/Qwen3-8B-4bit}"
PORT="${2:-8080}"
echo "serving $MODEL on http://127.0.0.1:$PORT/v1 (Ctrl-C to stop)"
exec "$P/.venv/bin/python" -m mlx_lm server --model "$MODEL" --host 127.0.0.1 --port "$PORT"
