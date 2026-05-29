#!/usr/bin/env bash
# Download the Pokemon dataset and train the CVQ tokenizer on the server.
# Run from the repository root:  bash scripts/run_server.sh
set -euo pipefail

CONFIG="${1:-configs/cvq_pokemon_cuda.yaml}"

# 1) Build the dataset (idempotent + cached; safe to re-run).
if [ ! -f data/manifest.jsonl ]; then
  echo "==> downloading Pokemon dataset (all variants, official artwork)"
  python -m cvq.data.download_pokemon --size 256 --workers 16
else
  echo "==> dataset already present ($(wc -l < data/manifest.jsonl) images)"
fi

# 2) Train. Logs to runs/ (tensorboard), samples to samples/, checkpoints to checkpoints/.
echo "==> training with $CONFIG"
python -m cvq.train --config "$CONFIG"

echo "==> training finished. Evaluate with:"
echo "    python -m cvq.reconstruct --ckpt checkpoints/latest.pt --n 8"
