#!/usr/bin/env bash
# Download the Pokemon dataset and train the CVQ tokenizer on the server.
# Run from the repository root:  bash scripts/run_server.sh [config.yaml]
set -euo pipefail

CONFIG="${1:-configs/cvq_pokemon_cuda.yaml}"
PY="./.venv/bin/python"
[ -x "$PY" ] || PY="python3"

# Load secrets (e.g. WANDB_API_KEY) from a gitignored .env if present.
if [ -f .env ]; then
  set -a; . ./.env; set +a
  echo "==> loaded .env ($([ -n "${WANDB_API_KEY:-}" ] && echo 'WANDB_API_KEY set' || echo 'no WANDB_API_KEY'))"
fi

# 1) Build the dataset (idempotent + cached; safe to re-run).
if [ ! -f data/manifest.jsonl ]; then
  echo "==> downloading Pokemon dataset (all variants, official artwork)"
  "$PY" -m cvq.data.download_pokemon --size 256 --workers 16
else
  echo "==> dataset already present ($(wc -l < data/manifest.jsonl) images)"
fi

# 2) Train. Logs to runs/ (tensorboard), samples to samples/, checkpoints to checkpoints/.
echo "==> training with $CONFIG"
"$PY" -m cvq.train --config "$CONFIG"

echo "==> training finished. Evaluate with:"
echo "    ./.venv/bin/python -m cvq.reconstruct --ckpt checkpoints/latest.pt --n 8"
