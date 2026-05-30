#!/bin/bash
# Box-side launcher for the faithful+fast E2E run. Uses the venv python explicitly.
cd /root/ImageLab || exit 9
pkill -9 -f cvq.train_e2e 2>/dev/null
sleep 1
set -a; . ./.env; set +a            # load WANDB_API_KEY (never printed)
export PYTHONUNBUFFERED=1
rm -f run_64fast.log
PY=/root/ImageLab/.venv/bin/python
echo "launcher_python=$PY" >> run_64fast.log
"$PY" -m cvq.train_e2e --config configs/car_e2e_pokemon_64_fast.yaml --max-steps 1700 >> run_64fast.log 2>&1 &
echo "TRAIN_PID=$!"
