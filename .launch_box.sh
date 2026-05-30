#!/bin/bash
# Full orchestration: sync code -> ensure data -> kill stale -> launch run -> verify.
# Writes a compact status to /tmp/il_status.txt. Safe to re-run (idempotent-ish).
exec > /tmp/il_status.txt 2>&1
SSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 19093 root@ssh2.vast.ai"
RSH="ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 -p 19093"
LOCAL=/Users/tanmaydeshmukh/Projects/ImageLab
cd "$LOCAL" || { echo "FATAL cd"; exit 9; }

echo "===[1] GPU+TORCH==="
$SSH 'nvidia-smi --query-gpu=name,memory.used --format=csv,noheader' 2>&1

echo "===[2] RSYNC CODE==="
rsync -az \
  --exclude '.git' --exclude '__pycache__' --exclude '*.pyc' \
  --exclude 'checkpoints_*' --exclude 'samples_*' --exclude 'wandb' \
  --exclude '.venv' --exclude 'data' --exclude '.launch_box.sh' \
  -e "$RSH" ./ root@ssh2.vast.ai:/root/ImageLab/ 2>&1 | tail -3
echo "rsync_code_exit=${PIPESTATUS[0]}"
$SSH 'echo "box torch:"; cd /root/ImageLab && python -c "import torch;print(torch.__version__,torch.cuda.is_available())" 2>&1 | tail -1' 2>&1

echo "===[3] DATA==="
N64=$($SSH 'ls /root/ImageLab/data/images_64 2>/dev/null | wc -l' 2>&1 | tr -d ' ')
echo "box images_64(before)=$N64"
LN64=$(ls "$LOCAL/data/images_64" 2>/dev/null | wc -l | tr -d ' ')
LN256=$(ls "$LOCAL/data/images_256" 2>/dev/null | wc -l | tr -d ' ')
echo "local images_64=$LN64 images_256=$LN256"
if [ "${N64:-0}" -lt 1000 ]; then
  $SSH 'mkdir -p /root/ImageLab/data' 2>&1
  if [ "${LN64:-0}" -ge 1000 ]; then
    echo "rsync local images_64 -> box"
    rsync -az -e "$RSH" "$LOCAL/data/images_64" root@ssh2.vast.ai:/root/ImageLab/data/ 2>&1 | tail -2
  elif [ "${LN256:-0}" -ge 1000 ]; then
    echo "rsync local images_256 -> box, resize on box"
    rsync -az -e "$RSH" "$LOCAL/data/images_256" root@ssh2.vast.ai:/root/ImageLab/data/ 2>&1 | tail -2
    $SSH 'cd /root/ImageLab && python -c "from pathlib import Path;from PIL import Image;s=Path(\"data/images_256\");d=Path(\"data/images_64\");d.mkdir(parents=True,exist_ok=True);[Image.open(p).convert(\"RGB\").resize((64,64),Image.BICUBIC).save(d/p.name) for p in s.glob(\"*\")];print(\"resized\",len(list(d.glob(\"*\"))))" 2>&1 | tail -1'
  else
    echo "FATAL: no local data to ship"
  fi
fi
N64=$($SSH 'ls /root/ImageLab/data/images_64 2>/dev/null | wc -l' 2>&1 | tr -d ' ')
echo "box images_64(after)=$N64"

echo "===[4] KILL STALE==="
$SSH 'pkill -9 -f train_e2e 2>/dev/null; sleep 2; nvidia-smi --query-gpu=memory.used --format=csv,noheader' 2>&1

if [ "${N64:-0}" -lt 1000 ]; then echo "ABORT: data missing, not launching"; echo "===STATUS=DATA_FAIL==="; exit 1; fi

echo "===[5] LAUNCH==="
$SSH 'cd /root/ImageLab && rm -f run_64fast.log && nohup python -m cvq.train_e2e --config configs/car_e2e_pokemon_64_fast.yaml --max-steps 1700 > run_64fast.log 2>&1 & echo launched_pid=$!' 2>&1
echo "warmup 80s..."; sleep 80
echo "===[6] LOG TAIL==="
$SSH 'tail -45 /root/ImageLab/run_64fast.log; echo "---ALIVE---"; pgrep -af train_e2e | head -1; echo "---GPU---"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader' 2>&1
echo "===STATUS=LAUNCH_DONE==="
