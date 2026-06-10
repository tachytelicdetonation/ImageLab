#!/bin/bash
# Cold-start orchestrator for the NEW box (ssh7.vast.ai:21163, A100-40GB).
# Stage 1 only: sync code + .env + data. Venv build + launch are separate stages.
set -uo pipefail
LOCAL=/Users/tanmaydeshmukh/Projects/ImageLab
PORT=21163
HOST=root@ssh7.vast.ai
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $PORT $HOST"
RSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p $PORT"
cd "$LOCAL" || { echo "FATAL cd"; exit 9; }

echo "===[1] mkdir on box==="
$SSH 'mkdir -p /root/ImageLab/data' 2>&1

echo "===[2] RSYNC CODE==="
rsync -az \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'checkpoints_*' --exclude 'samples_*' --exclude 'wandb' \
  --exclude '.venv' --exclude '/data' --exclude 'server_samples' \
  --exclude '*.png' \
  -e "$RSH" ./ "$HOST":/root/ImageLab/ 2>&1 | tail -3
echo "rsync_code_exit=${PIPESTATUS[0]}"

echo "===[3] SHIP .env (secret; not printed)==="
rsync -az -e "$RSH" "$LOCAL/.env" "$HOST":/root/ImageLab/.env 2>&1 | tail -1
echo "env_ship_exit=${PIPESTATUS[0]}"
$SSH 'test -f /root/ImageLab/.env && echo "env on box: keys=$(grep -oE "^[A-Z_]+=" /root/ImageLab/.env | tr "\n" " ")"' 2>&1

echo "===[4] SHIP DATA (images_256 + manifest)==="
rsync -az -e "$RSH" "$LOCAL/data/images_256" "$HOST":/root/ImageLab/data/ 2>&1 | tail -2
rsync -az -e "$RSH" "$LOCAL/data/manifest.jsonl" "$HOST":/root/ImageLab/data/manifest.jsonl 2>&1 | tail -1
echo "data_ship_exit=${PIPESTATUS[0]}"
$SSH 'echo "box images_256=$(ls /root/ImageLab/data/images_256 2>/dev/null | wc -l | tr -d " ")"' 2>&1

echo "===STAGE1_DONE==="
